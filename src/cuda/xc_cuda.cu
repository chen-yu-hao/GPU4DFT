#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/xc_kernels.hpp"
#include "gpu4dft/xc.hpp"

namespace gpu4dft::detail {
namespace {

inline void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template <typename T>
class DevBuf {
 public:
  DevBuf() = default;
  explicit DevBuf(std::size_t count) { resize(count); }
  DevBuf(const T* host_src, std::size_t count) {
    resize(count);
    upload(host_src, count);
  }
  explicit DevBuf(const std::vector<T>& host_vec)
      : DevBuf(host_vec.data(), host_vec.size()) {}

  ~DevBuf() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  void resize(std::size_t count) {
    if (count <= size_) return;
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = count;
    if (size_ > 0) {
      check_cuda(cudaMalloc(&ptr_, size_ * sizeof(T)), "xc cudaMalloc");
    }
  }

  void upload(const T* src, std::size_t count, cudaStream_t stream = nullptr) {
    if (count == 0) return;
    resize(count);
    check_cuda(cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream),
               "xc cudaMemcpyAsync H2D");
  }

  void download(T* dst, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count == 0) return;
    check_cuda(cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
               "xc cudaMemcpyAsync D2H");
  }

  T* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

template <typename T>
class PinnedBuf {
 public:
  PinnedBuf() = default;
  explicit PinnedBuf(std::size_t count) { resize(count); }
  ~PinnedBuf() {
    if (ptr_ != nullptr) cudaFreeHost(ptr_);
  }
  PinnedBuf(const PinnedBuf&) = delete;
  PinnedBuf& operator=(const PinnedBuf&) = delete;
  PinnedBuf(PinnedBuf&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  PinnedBuf& operator=(PinnedBuf&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFreeHost(ptr_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  void resize(std::size_t count) {
    if (count <= size_) return;
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    size_ = count;
    if (size_ > 0) {
      check_cuda(cudaHostAlloc(&ptr_, size_ * sizeof(T), cudaHostAllocDefault),
                 "xc cudaHostAlloc");
    }
  }

  T* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct DevFlatBasis {
  int nshell = 0;
  int nao = 0;
  DevBuf<int> shell_l;
  DevBuf<int> shell_ao_offset;
  DevBuf<int> shell_prim_offset;
  DevBuf<int> shell_prim_count;
  DevBuf<double> shell_center;
  DevBuf<double> prim_exp;
  DevBuf<double> prim_coeff;

  explicit DevFlatBasis(const FlatBasisData& h)
      : nshell(h.nshell),
        nao(h.nao),
        shell_l(h.shell_l),
        shell_ao_offset(h.shell_ao_offset),
        shell_prim_offset(h.shell_prim_offset),
        shell_prim_count(h.shell_prim_count),
        shell_center(h.shell_center),
        prim_exp(h.prim_exp),
        prim_coeff(h.prim_coeff) {}

  FlatBasisView view() const {
    return FlatBasisView{
        nshell,
        nao,
        shell_l.data(),
        shell_ao_offset.data(),
        shell_prim_offset.data(),
        shell_prim_count.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
    };
  }
};

// -----------------------------------------------------------------------------
// 1. GPU Becke / Stratmann-Scuseria-Frisch (SSF) Atomic Partitioning Kernel
// -----------------------------------------------------------------------------
__device__ inline double dev_step_becke(double nu) {
  if (nu <= -1.0) return 1.0;
  if (nu >= 1.0) return 0.0;
  double f = nu;
  #pragma unroll
  for (int k = 0; k < 3; ++k) {
    f = 0.5 * f * (3.0 - f * f);
  }
  return 0.5 * (1.0 - f);
}

__device__ inline void dev_step_becke_val_deriv(double nu, double& s_val, double& s_deriv) {
  if (nu <= -1.0) {
    s_val = 1.0;
    s_deriv = 0.0;
    return;
  }
  if (nu >= 1.0) {
    s_val = 0.0;
    s_deriv = 0.0;
    return;
  }
  const double f0 = nu;
  const double f1 = 0.5 * f0 * (3.0 - f0 * f0);
  const double f2 = 0.5 * f1 * (3.0 - f1 * f1);
  const double f3 = 0.5 * f2 * (3.0 - f2 * f2);
  s_val = 0.5 * (1.0 - f3);
  s_deriv = -(27.0 / 16.0) * (1.0 - f0 * f0) * (1.0 - f1 * f1) * (1.0 - f2 * f2);
}

__device__ inline double dev_step_ssf(double nu) {
  constexpr double kA = 0.64;
  if (nu <= -kA) return 1.0;
  if (nu >= kA) return 0.0;
  const double z = nu / kA;
  const double z2 = z * z;
  const double g = (1.0 / 16.0) * z * (35.0 + z2 * (-35.0 + z2 * (21.0 - 5.0 * z2)));
  return 0.5 * (1.0 - g);
}

__device__ inline void dev_step_ssf_val_deriv(double nu, double& s_val, double& s_deriv) {
  constexpr double kA = 0.64;
  if (nu <= -kA) {
    s_val = 1.0;
    s_deriv = 0.0;
    return;
  }
  if (nu >= kA) {
    s_val = 0.0;
    s_deriv = 0.0;
    return;
  }
  const double z = nu / kA;
  const double z2 = z * z;
  const double g = (1.0 / 16.0) * z * (35.0 + z2 * (-35.0 + z2 * (21.0 - 5.0 * z2)));
  s_val = 0.5 * (1.0 - g);
  const double one_m_z2 = 1.0 - z2;
  s_deriv = -(35.0 / (32.0 * kA)) * one_m_z2 * one_m_z2 * one_m_z2;
}

__global__ void partition_weights_kernel(const double* __restrict__ raw_coords,
                                         const int* __restrict__ parent_atom,
                                         std::size_t n_raw,
                                         const double* __restrict__ atom_coords,
                                         const double* __restrict__ inv_rij,
                                         const double* __restrict__ a_ij,
                                         int natom,
                                         int use_ssf,
                                         double* __restrict__ weights) {
  constexpr int kMaxSharedAtoms = 256;
  __shared__ double s_atom_xyz[3 * kMaxSharedAtoms];
  if (natom <= kMaxSharedAtoms) {
    for (int i = static_cast<int>(threadIdx.x); i < 3 * natom;
         i += static_cast<int>(blockDim.x)) {
      s_atom_xyz[i] = atom_coords[i];
    }
    __syncthreads();
  }

  const std::size_t ig = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ig >= n_raw) return;

  const double gx = raw_coords[3 * ig + 0];
  const double gy = raw_coords[3 * ig + 1];
  const double gz = raw_coords[3 * ig + 2];
  const int ia = parent_atom[ig];
  const double* ac = (natom <= kMaxSharedAtoms) ? s_atom_xyz : atom_coords;

  double sum_p = 0.0;
  double p_parent = 0.0;

  for (int b = 0; b < natom; ++b) {
    const double dxb = gx - ac[3 * b + 0];
    const double dyb = gy - ac[3 * b + 1];
    const double dzb = gz - ac[3 * b + 2];
    const double rb = ::sqrt(dxb * dxb + dyb * dyb + dzb * dzb);

    double pb = 1.0;
    for (int c = 0; c < natom; ++c) {
      if (b == c) continue;
      const double dxc = gx - ac[3 * c + 0];
      const double dyc = gy - ac[3 * c + 1];
      const double dzc = gz - ac[3 * c + 2];
      const double rc = ::sqrt(dxc * dxc + dyc * dyc + dzc * dzc);

      const int pair_idx = b * natom + c;
      const double mu = (rb - rc) * inv_rij[pair_idx];
      const double nu = mu + a_ij[pair_idx] * (1.0 - mu * mu);
      pb *= (use_ssf ? dev_step_ssf(nu) : dev_step_becke(nu));
      if (pb == 0.0) break;
    }
    sum_p += pb;
    if (b == ia) p_parent = pb;
  }

  weights[ig] = (sum_p > 0.0) ? (weights[ig] * (p_parent / sum_p)) : 0.0;
}

__global__ void partition_weight_grad_kernel(const double* __restrict__ coords,
                                             const int* __restrict__ parent_atom,
                                             const double* __restrict__ weights,
                                             const double* __restrict__ e_xc_point,
                                             std::size_t npts,
                                             const double* __restrict__ atom_coords,
                                             const double* __restrict__ inv_rij,
                                             const double* __restrict__ a_ij,
                                             int natom,
                                             int use_ssf,
                                             double* __restrict__ d_grad) {
  constexpr int kMaxSharedAtoms = 256;
  __shared__ double s_atom_xyz[3 * kMaxSharedAtoms];
  __shared__ double s_grad[3 * kMaxSharedAtoms];
  const bool use_shared = (natom <= kMaxSharedAtoms);
  if (use_shared) {
    for (int i = static_cast<int>(threadIdx.x); i < 3 * natom;
         i += static_cast<int>(blockDim.x)) {
      s_atom_xyz[i] = atom_coords[i];
      s_grad[i] = 0.0;
    }
    __syncthreads();
  }

  const std::size_t ig = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const double* ac = use_shared ? s_atom_xyz : atom_coords;
  double* grad_target = use_shared ? s_grad : d_grad;

  if (ig < npts && natom > 1) {
    const double e_g = e_xc_point[ig];
    const double wg = weights[ig];
    const int B = parent_atom[ig];
    if (e_g != 0.0 && wg != 0.0 && B >= 0 && B < natom) {
      const double gx = coords[3 * ig + 0];
      const double gy = coords[3 * ig + 1];
      const double gz = coords[3 * ig + 2];

      double sum_p = 0.0;
      double p_B = 0.0;
      for (int b = 0; b < natom; ++b) {
        const double dxb = gx - ac[3 * b + 0];
        const double dyb = gy - ac[3 * b + 1];
        const double dzb = gz - ac[3 * b + 2];
        const double rb = ::sqrt(dxb * dxb + dyb * dyb + dzb * dzb);

        double pb = 1.0;
        for (int c = 0; c < natom; ++c) {
          if (b == c) continue;
          const double dxc = gx - ac[3 * c + 0];
          const double dyc = gy - ac[3 * c + 1];
          const double dzc = gz - ac[3 * c + 2];
          const double rc = ::sqrt(dxc * dxc + dyc * dyc + dzc * dzc);

          const int pair_idx = b * natom + c;
          const double mu = (rb - rc) * inv_rij[pair_idx];
          const double nu = mu + a_ij[pair_idx] * (1.0 - mu * mu);
          pb *= (use_ssf ? dev_step_ssf(nu) : dev_step_becke(nu));
          if (pb == 0.0) break;
        }
        sum_p += pb;
        if (b == B) p_B = pb;
      }

      if (sum_p > 0.0 && p_B > 0.0) {
        const double W_B = p_B / sum_p;
        const double pref_base = (wg / p_B) * e_g;

        double gB_x = 0.0, gB_y = 0.0, gB_z = 0.0;
        for (int b = 0; b < natom; ++b) {
          const double dxb = gx - ac[3 * b + 0];
          const double dyb = gy - ac[3 * b + 1];
          const double dzb = gz - ac[3 * b + 2];
          const double rb = ::sqrt(dxb * dxb + dyb * dyb + dzb * dzb);

          double pb = 1.0;
          for (int c = 0; c < natom; ++c) {
            if (b == c) continue;
            const double dxc = gx - ac[3 * c + 0];
            const double dyc = gy - ac[3 * c + 1];
            const double dzc = gz - ac[3 * c + 2];
            const double rc = ::sqrt(dxc * dxc + dyc * dyc + dzc * dzc);
            const int pair_idx = b * natom + c;
            const double mu = (rb - rc) * inv_rij[pair_idx];
            const double nu = mu + a_ij[pair_idx] * (1.0 - mu * mu);
            pb *= (use_ssf ? dev_step_ssf(nu) : dev_step_becke(nu));
            if (pb == 0.0) break;
          }
          if (pb <= 0.0) continue;

          const double factor_b = pref_base * pb * ((b == B ? 1.0 : 0.0) - W_B);
          if (factor_b == 0.0) continue;

          const double inv_rb = (rb > 1e-14) ? (1.0 / rb) : 0.0;
          const double ub_x = dxb * inv_rb;
          const double ub_y = dyb * inv_rb;
          const double ub_z = dzb * inv_rb;

          double gb_x = 0.0, gb_y = 0.0, gb_z = 0.0;
          for (int c = 0; c < natom; ++c) {
            if (b == c) continue;
            const int pair_idx = b * natom + c;
            const double inv_rbc = inv_rij[pair_idx];
            if (inv_rbc <= 0.0) continue;

            const double dxc = gx - ac[3 * c + 0];
            const double dyc = gy - ac[3 * c + 1];
            const double dzc = gz - ac[3 * c + 2];
            const double rc = ::sqrt(dxc * dxc + dyc * dyc + dzc * dzc);
            const double inv_rc = (rc > 1e-14) ? (1.0 / rc) : 0.0;
            const double uc_x = dxc * inv_rc;
            const double uc_y = dyc * inv_rc;
            const double uc_z = dzc * inv_rc;

            const double mu = (rb - rc) * inv_rbc;
            const double a_bc = a_ij[pair_idx];
            const double nu = mu + a_bc * (1.0 - mu * mu);
            double s_val = 0.0, s_prime = 0.0;
            if (use_ssf) dev_step_ssf_val_deriv(nu, s_val, s_prime);
            else dev_step_becke_val_deriv(nu, s_val, s_prime);
            if (s_val <= 1e-30 || s_prime == 0.0) continue;

            const double kappa_bc =
                factor_b * (s_prime / s_val) * (1.0 - 2.0 * a_bc * mu) * inv_rbc;
            const double ubc_x = (ac[3 * b + 0] - ac[3 * c + 0]) * inv_rbc;
            const double ubc_y = (ac[3 * b + 1] - ac[3 * c + 1]) * inv_rbc;
            const double ubc_z = (ac[3 * b + 2] - ac[3 * c + 2]) * inv_rbc;

            gb_x -= kappa_bc * (ub_x + mu * ubc_x);
            gb_y -= kappa_bc * (ub_y + mu * ubc_y);
            gb_z -= kappa_bc * (ub_z + mu * ubc_z);

            atomicAdd(&grad_target[3 * c + 0], kappa_bc * (uc_x + mu * ubc_x));
            atomicAdd(&grad_target[3 * c + 1], kappa_bc * (uc_y + mu * ubc_y));
            atomicAdd(&grad_target[3 * c + 2], kappa_bc * (uc_z + mu * ubc_z));

            gB_x += kappa_bc * (ub_x - uc_x);
            gB_y += kappa_bc * (ub_y - uc_y);
            gB_z += kappa_bc * (ub_z - uc_z);
          }
          atomicAdd(&grad_target[3 * b + 0], gb_x);
          atomicAdd(&grad_target[3 * b + 1], gb_y);
          atomicAdd(&grad_target[3 * b + 2], gb_z);
        }
        atomicAdd(&grad_target[3 * B + 0], gB_x);
        atomicAdd(&grad_target[3 * B + 1], gB_y);
        atomicAdd(&grad_target[3 * B + 2], gB_z);
      }
    }
  }

  if (use_shared) {
    __syncthreads();
    for (int i = static_cast<int>(threadIdx.x); i < 3 * natom;
         i += static_cast<int>(blockDim.x)) {
      const double v = s_grad[i];
      if (v != 0.0) atomicAdd(&d_grad[i], v);
    }
  }
}

// -----------------------------------------------------------------------------
// 2. Point-wise XC & Full-Grid AO Kernels
// -----------------------------------------------------------------------------
__global__ void eval_xc_points_kernel(int func_id,
                                      std::size_t n,
                                      const double* __restrict__ rho,
                                      const double* __restrict__ sigma,
                                      const double* __restrict__ tau,
                                      int deriv,
                                      double rho_cutoff,
                                      double* __restrict__ zk,
                                      double* __restrict__ vrho,
                                      double* __restrict__ vsigma,
                                      double* __restrict__ vtau,
                                      double* __restrict__ v2rho2,
                                      double* __restrict__ v2rhosigma,
                                      double* __restrict__ v2sigma2,
                                      double* __restrict__ v2rhotau,
                                      double* __restrict__ v2sigmatau,
                                      double* __restrict__ v2tau2) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  xc_gen::XcPointOut pt;
  xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(func_id),
                        rho[i], sigma[i], tau[i], deriv, pt, rho_cutoff);
  zk[i] = pt.zk;
  if (deriv >= 1) {
    vrho[i] = pt.vrho;
    vsigma[i] = pt.vsigma;
    vtau[i] = pt.vtau;
  }
  if (deriv >= 2) {
    v2rho2[i] = pt.v2rho2;
    v2rhosigma[i] = pt.v2rhosigma;
    v2sigma2[i] = pt.v2sigma2;
    v2rhotau[i] = pt.v2rhotau;
    v2sigmatau[i] = pt.v2sigmatau;
    v2tau2[i] = pt.v2tau2;
  }
}

__global__ void eval_ao_full_grid_kernel(FlatBasisView bview,
                                         const double* __restrict__ coords_xyz,
                                         std::size_t npts,
                                         int deriv,
                                         double* __restrict__ out_ao) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int is = static_cast<int>(blockIdx.y);
  if (g >= npts || is >= bview.nshell) return;

  const std::size_t nao = static_cast<std::size_t>(bview.nao);
  const std::size_t comp_stride = npts * nao;
  const std::size_t ao0 = static_cast<std::size_t>(bview.shell_ao_offset[is]);
  eval_shell_at_point(bview, is,
                      coords_xyz[3 * g + 0], coords_xyz[3 * g + 1], coords_xyz[3 * g + 2],
                      deriv, out_ao + g * nao + ao0, comp_stride, 1);
}

// -----------------------------------------------------------------------------
// 3. Super-Batched High-Occupancy Coalesced V_xc, f_xc & Nuclear Gradient Kernels
// -----------------------------------------------------------------------------
__global__ void eval_active_ao_batch_kernel(FlatBasisView bview,
                                            const double* __restrict__ coords,
                                            std::size_t point_offset,
                                            std::size_t ng,
                                            std::size_t na,
                                            const int* __restrict__ act_shells,
                                            const int* __restrict__ act_shell_local_ao,
                                            int n_act_shells,
                                            int deriv,
                                            double* __restrict__ phi) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int s_idx = static_cast<int>(blockIdx.y);
  if (g >= ng || s_idx >= n_act_shells) return;

  const int is = act_shells[s_idx];
  const std::size_t local_ao0 = static_cast<std::size_t>(act_shell_local_ao[s_idx]);
  const std::size_t gp = point_offset + g;
  const std::size_t comp_stride = na * ng;

  eval_shell_at_point(bview, is,
                      coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                      deriv, phi + local_ao0 * ng + g, comp_stride, ng);
}

__global__ void gather_sym_dm_kernel(const double* __restrict__ d_dm,
                                     std::size_t nao,
                                     const int* __restrict__ act_aos,
                                     std::size_t na,
                                     double* __restrict__ d_act) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * na) return;
  const std::size_t u = idx / na;
  const std::size_t v = idx % na;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  const std::size_t gv = static_cast<std::size_t>(act_aos[v]);
  d_act[idx] = 0.5 * (d_dm[gu * nao + gv] + d_dm[gv * nao + gu]);
}

__global__ void scatter_sym_vxc_kernel(const double* __restrict__ v_half,
                                       const int* __restrict__ act_aos,
                                       std::size_t na,
                                       std::size_t nao,
                                       double* __restrict__ d_vxc) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * na) return;
  const std::size_t u = idx / na;
  const std::size_t v = idx % na;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  const std::size_t gv = static_cast<std::size_t>(act_aos[v]);
  d_vxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
}

__global__ void reduce_stream_vxc_kernel(std::size_t nao2,
                                         int n_streams,
                                         const double* __restrict__ p0,
                                         const double* __restrict__ p1,
                                         const double* __restrict__ p2,
                                         const double* __restrict__ p3,
                                         double* __restrict__ out) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nao2) return;
  double sum = p0[i];
  if (n_streams >= 2) sum += p1[i];
  if (n_streams >= 3) sum += p2[i];
  if (n_streams >= 4) sum += p3[i];
  out[i] = sum;
}

__global__ void reduce_stream_scalars_kernel(int n_streams,
                                             const double* __restrict__ s0,
                                             const double* __restrict__ s1,
                                             const double* __restrict__ s2,
                                             const double* __restrict__ s3,
                                             double* __restrict__ out) {
  const int i = static_cast<int>(threadIdx.x);
  if (i >= 2) return;
  double sum = s0[i];
  if (n_streams >= 2) sum += s1[i];
  if (n_streams >= 3) sum += s2[i];
  if (n_streams >= 4) sum += s3[i];
  out[i] = sum;
}

__global__ void cast_f64_to_f32_kernel(const double* __restrict__ src,
                                       float* __restrict__ dst,
                                       std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<float>(src[i]);
}

__global__ void cast_f32_to_f64_kernel(const float* __restrict__ src,
                                       double* __restrict__ dst,
                                       std::size_t n,
                                       int accum) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    if (accum) dst[i] += static_cast<double>(src[i]);
    else dst[i] = static_cast<double>(src[i]);
  }
}

// Kernel A: High-occupancy 2D (64 points x 4 AO-reduction warps) density + symbolic XC evaluator.
// Coalesced reads of Phi & T, shared-memory reduction across `ty`, 1 thread per point evaluates
// `xc_kernels.hpp`, writes compact `wv[0..4 * ng + g]`, and issues 1 `atomicAdd` per threadblock!
__global__ void eval_rho_and_vxc_potentials_kernel(int func_id,
                                                   int family,
                                                   double rho_cutoff,
                                                   std::size_t na,
                                                   std::size_t ng,
                                                   const double* __restrict__ weights,
                                                   std::size_t point_offset,
                                                   const double* __restrict__ phi,
                                                   const double* __restrict__ t_mat,
                                                   double* __restrict__ wv_out,
                                                   double* __restrict__ scalars_out) {
  constexpr int kTileG = 64;
  constexpr int kRedY = 4;
  __shared__ double s_rho[kRedY][kTileG];
  __shared__ double s_gx[kRedY][kTileG];
  __shared__ double s_gy[kRedY][kTileG];
  __shared__ double s_gz[kRedY][kTileG];
  __shared__ double s_tau[kRedY][kTileG];
  __shared__ double s_warp_exc[2];
  __shared__ double s_warp_nelec[2];

  const int tx = static_cast<int>(threadIdx.x);  // [0, 64)
  const int ty = static_cast<int>(threadIdx.y);  // [0, 4)
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * kTileG + static_cast<std::size_t>(tx);

  double rho = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, tau = 0.0;
  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* __restrict__ phi0 = phi;
    const double* __restrict__ phix = (family >= 1) ? (phi + 1 * comp_stride) : nullptr;
    const double* __restrict__ phiy = (family >= 1) ? (phi + 2 * comp_stride) : nullptr;
    const double* __restrict__ phiz = (family >= 1) ? (phi + 3 * comp_stride) : nullptr;
    const double* __restrict__ t0 = t_mat;

    for (std::size_t u = static_cast<std::size_t>(ty); u < na; u += kRedY) {
      const std::size_t off = u * ng + g;
      const double t0_ug = t0[off];
      rho += phi0[off] * t0_ug;
      if (family >= 1) {
        const double px = phix[off];
        const double py = phiy[off];
        const double pz = phiz[off];
        gx += 2.0 * px * t0_ug;
        gy += 2.0 * py * t0_ug;
        gz += 2.0 * pz * t0_ug;
        if (family == 2) {
          tau += 0.5 * (px * t_mat[1 * comp_stride + off] +
                        py * t_mat[2 * comp_stride + off] +
                        pz * t_mat[3 * comp_stride + off]);
        }
      }
    }
  }

  s_rho[ty][tx] = rho;
  if (family >= 1) {
    s_gx[ty][tx] = gx;
    s_gy[ty][tx] = gy;
    s_gz[ty][tx] = gz;
    if (family == 2) {
      s_tau[ty][tx] = tau;
    }
  }
  __syncthreads();

  if (ty == 0) {
    double pt_exc = 0.0;
    double pt_nelec = 0.0;
    if (g < ng) {
      const double r = (s_rho[0][tx] + s_rho[1][tx]) + (s_rho[2][tx] + s_rho[3][tx]);
      double ggx = 0.0, ggy = 0.0, ggz = 0.0, tt = 0.0, sigma = 0.0;
      if (family >= 1) {
        ggx = (s_gx[0][tx] + s_gx[1][tx]) + (s_gx[2][tx] + s_gx[3][tx]);
        ggy = (s_gy[0][tx] + s_gy[1][tx]) + (s_gy[2][tx] + s_gy[3][tx]);
        ggz = (s_gz[0][tx] + s_gz[1][tx]) + (s_gz[2][tx] + s_gz[3][tx]);
        sigma = ggx * ggx + ggy * ggy + ggz * ggz;
        if (family == 2) {
          tt = (s_tau[0][tx] + s_tau[1][tx]) + (s_tau[2][tx] + s_tau[3][tx]);
        }
      }

      const double wg = weights[point_offset + g];
      if (r > 0.0) pt_nelec = wg * r;

      xc_gen::XcPointOut pt;
      xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(func_id),
                            r, sigma, tt, 1, pt, rho_cutoff);
      pt_exc = wg * r * pt.zk;

      wv_out[0 * ng + g] = 0.5 * wg * pt.vrho;
      if (family >= 1) {
        const double two_w_vs = 2.0 * wg * pt.vsigma;
        wv_out[1 * ng + g] = two_w_vs * ggx;
        wv_out[2 * ng + g] = two_w_vs * ggy;
        wv_out[3 * ng + g] = two_w_vs * ggz;
        if (family == 2) {
          wv_out[4 * ng + g] = 0.25 * wg * pt.vtau;
        }
      }
    }

    for (int mask = 16; mask > 0; mask >>= 1) {
      pt_exc += __shfl_down_sync(0xffffffffu, pt_exc, mask);
      pt_nelec += __shfl_down_sync(0xffffffffu, pt_nelec, mask);
    }
    const int warp_id = tx >> 5;
    if ((tx & 31) == 0) {
      s_warp_exc[warp_id] = pt_exc;
      s_warp_nelec[warp_id] = pt_nelec;
    }
  }
  __syncthreads();

  if (tx == 0 && ty == 0) {
    atomicAdd(&scalars_out[0], s_warp_exc[0] + s_warp_exc[1]);
    atomicAdd(&scalars_out[1], s_warp_nelec[0] + s_warp_nelec[1]);
  }
}

// Flat density reduction kernels for Libxc evaluation (no per-batch sync!)
__global__ void eval_density_flat_rks_kernel(int family,
                                             std::size_t na,
                                             std::size_t ng,
                                             std::size_t npoints_total,
                                             std::size_t point_offset,
                                             const double* __restrict__ phi,
                                             const double* __restrict__ t_mat,
                                             double* __restrict__ dens_in) {
  constexpr int kTileG = 64;
  constexpr int kRedY = 4;
  __shared__ double s_rho[kRedY][kTileG];
  __shared__ double s_gx[kRedY][kTileG];
  __shared__ double s_gy[kRedY][kTileG];
  __shared__ double s_gz[kRedY][kTileG];
  __shared__ double s_tau[kRedY][kTileG];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * kTileG + static_cast<std::size_t>(tx);

  double rho = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, tau = 0.0;
  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* __restrict__ phi0 = phi;
    const double* __restrict__ phix = (family >= 1) ? (phi + 1 * comp_stride) : nullptr;
    const double* __restrict__ phiy = (family >= 1) ? (phi + 2 * comp_stride) : nullptr;
    const double* __restrict__ phiz = (family >= 1) ? (phi + 3 * comp_stride) : nullptr;
    const double* __restrict__ t0 = t_mat;

    for (std::size_t u = static_cast<std::size_t>(ty); u < na; u += kRedY) {
      const std::size_t off = u * ng + g;
      const double t0_ug = t0[off];
      rho += phi0[off] * t0_ug;
      if (family >= 1) {
        const double px = phix[off];
        const double py = phiy[off];
        const double pz = phiz[off];
        gx += 2.0 * px * t0_ug;
        gy += 2.0 * py * t0_ug;
        gz += 2.0 * pz * t0_ug;
        if (family == 2) {
          tau += 0.5 * (px * t_mat[1 * comp_stride + off] +
                        py * t_mat[2 * comp_stride + off] +
                        pz * t_mat[3 * comp_stride + off]);
        }
      }
    }
  }

  s_rho[ty][tx] = rho;
  if (family >= 1) {
    s_gx[ty][tx] = gx;
    s_gy[ty][tx] = gy;
    s_gz[ty][tx] = gz;
    if (family == 2) {
      s_tau[ty][tx] = tau;
    }
  }
  __syncthreads();

  if (ty == 0 && g < ng) {
    const std::size_t gp = point_offset + g;
    dens_in[0 * npoints_total + gp] =
        (s_rho[0][tx] + s_rho[1][tx]) + (s_rho[2][tx] + s_rho[3][tx]);
    if (family >= 1) {
      dens_in[1 * npoints_total + gp] =
          (s_gx[0][tx] + s_gx[1][tx]) + (s_gx[2][tx] + s_gx[3][tx]);
      dens_in[2 * npoints_total + gp] =
          (s_gy[0][tx] + s_gy[1][tx]) + (s_gy[2][tx] + s_gy[3][tx]);
      dens_in[3 * npoints_total + gp] =
          (s_gz[0][tx] + s_gz[1][tx]) + (s_gz[2][tx] + s_gz[3][tx]);
      if (family == 2) {
        dens_in[4 * npoints_total + gp] =
            (s_tau[0][tx] + s_tau[1][tx]) + (s_tau[2][tx] + s_tau[3][tx]);
      }
    }
  }
}

__global__ void eval_density_flat_uks_kernel(int family,
                                             std::size_t na,
                                             std::size_t ng,
                                             std::size_t npoints_total,
                                             std::size_t point_offset,
                                             const double* __restrict__ phi,
                                             const double* __restrict__ t_mat_a,
                                             const double* __restrict__ t_mat_b,
                                             double* __restrict__ dens_in) {
  constexpr int kTileG = 64;
  constexpr int kRedY = 4;
  __shared__ double s_ra[kRedY][kTileG], s_rb[kRedY][kTileG];
  __shared__ double s_gxa[kRedY][kTileG], s_gya[kRedY][kTileG], s_gza[kRedY][kTileG];
  __shared__ double s_gxb[kRedY][kTileG], s_gyb[kRedY][kTileG], s_gzb[kRedY][kTileG];
  __shared__ double s_ta[kRedY][kTileG], s_tb[kRedY][kTileG];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * kTileG + static_cast<std::size_t>(tx);

  double ra = 0.0, rb = 0.0;
  double gxa = 0.0, gya = 0.0, gza = 0.0, gxb = 0.0, gyb = 0.0, gzb = 0.0;
  double ta = 0.0, tb = 0.0;

  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* __restrict__ phi0 = phi;
    const double* __restrict__ phix = (family >= 1) ? (phi + 1 * comp_stride) : nullptr;
    const double* __restrict__ phiy = (family >= 1) ? (phi + 2 * comp_stride) : nullptr;
    const double* __restrict__ phiz = (family >= 1) ? (phi + 3 * comp_stride) : nullptr;

    for (std::size_t u = static_cast<std::size_t>(ty); u < na; u += kRedY) {
      const std::size_t off = u * ng + g;
      const double p0 = phi0[off];
      const double t0a = t_mat_a[off];
      const double t0b = t_mat_b[off];
      ra += p0 * t0a;
      rb += p0 * t0b;
      if (family >= 1) {
        const double px = phix[off];
        const double py = phiy[off];
        const double pz = phiz[off];
        gxa += 2.0 * px * t0a;
        gya += 2.0 * py * t0a;
        gza += 2.0 * pz * t0a;
        gxb += 2.0 * px * t0b;
        gyb += 2.0 * py * t0b;
        gzb += 2.0 * pz * t0b;
        if (family == 2) {
          ta += 0.5 * (px * t_mat_a[1 * comp_stride + off] +
                       py * t_mat_a[2 * comp_stride + off] +
                       pz * t_mat_a[3 * comp_stride + off]);
          tb += 0.5 * (px * t_mat_b[1 * comp_stride + off] +
                       py * t_mat_b[2 * comp_stride + off] +
                       pz * t_mat_b[3 * comp_stride + off]);
        }
      }
    }
  }

  s_ra[ty][tx] = ra;
  s_rb[ty][tx] = rb;
  if (family >= 1) {
    s_gxa[ty][tx] = gxa;
    s_gya[ty][tx] = gya;
    s_gza[ty][tx] = gza;
    s_gxb[ty][tx] = gxb;
    s_gyb[ty][tx] = gyb;
    s_gzb[ty][tx] = gzb;
    if (family == 2) {
      s_ta[ty][tx] = ta;
      s_tb[ty][tx] = tb;
    }
  }
  __syncthreads();

  if (ty == 0 && g < ng) {
    const std::size_t gp = point_offset + g;
    dens_in[0 * npoints_total + gp] = (s_ra[0][tx] + s_ra[1][tx]) + (s_ra[2][tx] + s_ra[3][tx]);
    dens_in[1 * npoints_total + gp] = (s_rb[0][tx] + s_rb[1][tx]) + (s_rb[2][tx] + s_rb[3][tx]);
    if (family >= 1) {
      dens_in[2 * npoints_total + gp] =
          (s_gxa[0][tx] + s_gxa[1][tx]) + (s_gxa[2][tx] + s_gxa[3][tx]);
      dens_in[3 * npoints_total + gp] =
          (s_gya[0][tx] + s_gya[1][tx]) + (s_gya[2][tx] + s_gya[3][tx]);
      dens_in[4 * npoints_total + gp] =
          (s_gza[0][tx] + s_gza[1][tx]) + (s_gza[2][tx] + s_gza[3][tx]);
      dens_in[5 * npoints_total + gp] =
          (s_gxb[0][tx] + s_gxb[1][tx]) + (s_gxb[2][tx] + s_gxb[3][tx]);
      dens_in[6 * npoints_total + gp] =
          (s_gyb[0][tx] + s_gyb[1][tx]) + (s_gyb[2][tx] + s_gyb[3][tx]);
      dens_in[7 * npoints_total + gp] =
          (s_gzb[0][tx] + s_gzb[1][tx]) + (s_gzb[2][tx] + s_gzb[3][tx]);
      if (family == 2) {
        dens_in[8 * npoints_total + gp] =
            (s_ta[0][tx] + s_ta[1][tx]) + (s_ta[2][tx] + s_ta[3][tx]);
        dens_in[9 * npoints_total + gp] =
            (s_tb[0][tx] + s_tb[1][tx]) + (s_tb[2][tx] + s_tb[3][tx]);
      }
    }
  }
}

// Kernel B: 2D (point_block, ao) coalesced half-weighted W[u * ng + g] builder (100% SM saturation!).
__global__ void build_vxc_weight_matrix_kernel(int family,
                                               std::size_t na,
                                               std::size_t ng,
                                               std::size_t wv_stride,
                                               const double* __restrict__ phi,
                                               const double* __restrict__ wv,
                                               double* __restrict__ w_mat) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t u = static_cast<std::size_t>(blockIdx.y);
  if (g >= ng || u >= na) return;

  const std::size_t comp_stride = na * ng;
  const std::size_t off = u * ng + g;
  double w_ug = wv[0 * wv_stride + g] * phi[off];
  if (family >= 1) {
    w_ug += wv[1 * wv_stride + g] * phi[1 * comp_stride + off] +
            wv[2 * wv_stride + g] * phi[2 * comp_stride + off] +
            wv[3 * wv_stride + g] * phi[3 * comp_stride + off];
  }
  w_mat[off] = w_ug;
}

// 2D coalesced MGGA tau weight builder: w_mat[u * ng + g] = wv4[g] * phib[u * ng + g].
__global__ void scale_mgga_tau_2d_kernel(std::size_t na,
                                         std::size_t ng,
                                         const double* __restrict__ phib,
                                         const double* __restrict__ wv4,
                                         double* __restrict__ w_mat) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t u = static_cast<std::size_t>(blockIdx.y);
  if (g >= ng || u >= na) return;
  const std::size_t off = u * ng + g;
  w_mat[off] = wv4[g] * phib[off];
}

// Ground-state 2nd-derivative XC Hessian precomputation kernel for TDDFT (`set_reference_density_device`).
// Layout of `fxc_cache` [n_fields * npoints]:
//   0: v2rho2
//   1..3: g0x, g0y, g0z  | 4: vsigma | 5: v2rhosigma | 6: v2sigma2   (when family >= 1)
//   7: v2rhotau          | 8: v2sigmatau              | 9: v2tau2    (when family == 2)
__global__ void cache_fxc_reference_kernel(int func_id,
                                           int family,
                                           double rho_cutoff,
                                           std::size_t na,
                                           std::size_t ng,
                                           std::size_t npoints_total,
                                           std::size_t point_offset,
                                           const double* __restrict__ phi,
                                           const double* __restrict__ t_mat,
                                           double* __restrict__ fxc_cache) {
  constexpr int kTileG = 64;
  constexpr int kRedY = 4;
  __shared__ double s_rho[kRedY][kTileG];
  __shared__ double s_gx[kRedY][kTileG];
  __shared__ double s_gy[kRedY][kTileG];
  __shared__ double s_gz[kRedY][kTileG];
  __shared__ double s_tau[kRedY][kTileG];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * kTileG + static_cast<std::size_t>(tx);

  double rho = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, tau = 0.0;
  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* __restrict__ phi0 = phi;
    const double* __restrict__ phix = (family >= 1) ? (phi + 1 * comp_stride) : nullptr;
    const double* __restrict__ phiy = (family >= 1) ? (phi + 2 * comp_stride) : nullptr;
    const double* __restrict__ phiz = (family >= 1) ? (phi + 3 * comp_stride) : nullptr;
    const double* __restrict__ t0 = t_mat;

    for (std::size_t u = static_cast<std::size_t>(ty); u < na; u += kRedY) {
      const std::size_t off = u * ng + g;
      const double t0_ug = t0[off];
      rho += phi0[off] * t0_ug;
      if (family >= 1) {
        const double px = phix[off];
        const double py = phiy[off];
        const double pz = phiz[off];
        gx += 2.0 * px * t0_ug;
        gy += 2.0 * py * t0_ug;
        gz += 2.0 * pz * t0_ug;
        if (family == 2) {
          tau += 0.5 * (px * t_mat[1 * comp_stride + off] +
                        py * t_mat[2 * comp_stride + off] +
                        pz * t_mat[3 * comp_stride + off]);
        }
      }
    }
  }

  s_rho[ty][tx] = rho;
  if (family >= 1) {
    s_gx[ty][tx] = gx;
    s_gy[ty][tx] = gy;
    s_gz[ty][tx] = gz;
    if (family == 2) {
      s_tau[ty][tx] = tau;
    }
  }
  __syncthreads();

  if (ty == 0 && g < ng) {
    const std::size_t gp = point_offset + g;
    const double r = (s_rho[0][tx] + s_rho[1][tx]) + (s_rho[2][tx] + s_rho[3][tx]);
    double ggx = 0.0, ggy = 0.0, ggz = 0.0, tt = 0.0, sigma = 0.0;
    if (family >= 1) {
      ggx = (s_gx[0][tx] + s_gx[1][tx]) + (s_gx[2][tx] + s_gx[3][tx]);
      ggy = (s_gy[0][tx] + s_gy[1][tx]) + (s_gy[2][tx] + s_gy[3][tx]);
      ggz = (s_gz[0][tx] + s_gz[1][tx]) + (s_gz[2][tx] + s_gz[3][tx]);
      sigma = ggx * ggx + ggy * ggy + ggz * ggz;
      if (family == 2) {
        tt = (s_tau[0][tx] + s_tau[1][tx]) + (s_tau[2][tx] + s_tau[3][tx]);
      }
    }

    xc_gen::XcPointOut pt;
    xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(func_id),
                          r, sigma, tt, 2, pt, rho_cutoff);

    fxc_cache[0 * npoints_total + gp] = pt.v2rho2;
    if (family >= 1) {
      fxc_cache[1 * npoints_total + gp] = ggx;
      fxc_cache[2 * npoints_total + gp] = ggy;
      fxc_cache[3 * npoints_total + gp] = ggz;
      fxc_cache[4 * npoints_total + gp] = pt.vsigma;
      fxc_cache[5 * npoints_total + gp] = pt.v2rhosigma;
      fxc_cache[6 * npoints_total + gp] = pt.v2sigma2;
      if (family == 2) {
        fxc_cache[7 * npoints_total + gp] = pt.v2rhotau;
        fxc_cache[8 * npoints_total + gp] = pt.v2sigmatau;
        fxc_cache[9 * npoints_total + gp] = pt.v2tau2;
      }
    }
  }
}

// Linear-response f_xc[rho_0] perturbation potential kernel (`eval_fxc_device`):
// Zero transcendental calls — pure linear algebra against cached `fxc_cache`!
__global__ void eval_fxc_potentials_kernel(int family,
                                           std::size_t na,
                                           std::size_t ng,
                                           std::size_t npoints_total,
                                           const double* __restrict__ weights,
                                           std::size_t point_offset,
                                           const double* __restrict__ phi,
                                           const double* __restrict__ t_mat,
                                           const double* __restrict__ fxc_cache,
                                           double* __restrict__ wv_out) {
  constexpr int kTileG = 64;
  constexpr int kRedY = 4;
  __shared__ double s_drho[kRedY][kTileG];
  __shared__ double s_dgx[kRedY][kTileG];
  __shared__ double s_dgy[kRedY][kTileG];
  __shared__ double s_dgz[kRedY][kTileG];
  __shared__ double s_dtau[kRedY][kTileG];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * kTileG + static_cast<std::size_t>(tx);

  double drho = 0.0, dgx = 0.0, dgy = 0.0, dgz = 0.0, dtau = 0.0;
  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* __restrict__ phi0 = phi;
    const double* __restrict__ phix = (family >= 1) ? (phi + 1 * comp_stride) : nullptr;
    const double* __restrict__ phiy = (family >= 1) ? (phi + 2 * comp_stride) : nullptr;
    const double* __restrict__ phiz = (family >= 1) ? (phi + 3 * comp_stride) : nullptr;
    const double* __restrict__ t0 = t_mat;

    for (std::size_t u = static_cast<std::size_t>(ty); u < na; u += kRedY) {
      const std::size_t off = u * ng + g;
      const double t0_ug = t0[off];
      drho += phi0[off] * t0_ug;
      if (family >= 1) {
        const double px = phix[off];
        const double py = phiy[off];
        const double pz = phiz[off];
        dgx += 2.0 * px * t0_ug;
        dgy += 2.0 * py * t0_ug;
        dgz += 2.0 * pz * t0_ug;
        if (family == 2) {
          dtau += 0.5 * (px * t_mat[1 * comp_stride + off] +
                         py * t_mat[2 * comp_stride + off] +
                         pz * t_mat[3 * comp_stride + off]);
        }
      }
    }
  }

  s_drho[ty][tx] = drho;
  if (family >= 1) {
    s_dgx[ty][tx] = dgx;
    s_dgy[ty][tx] = dgy;
    s_dgz[ty][tx] = dgz;
    if (family == 2) {
      s_dtau[ty][tx] = dtau;
    }
  }
  __syncthreads();

  if (ty == 0 && g < ng) {
    const std::size_t gp = point_offset + g;
    const double wg = weights[gp];
    const double dr = (s_drho[0][tx] + s_drho[1][tx]) + (s_drho[2][tx] + s_drho[3][tx]);

    double d_vrho = fxc_cache[0 * npoints_total + gp] * dr;
    if (family >= 1) {
      const double dgx_v = (s_dgx[0][tx] + s_dgx[1][tx]) + (s_dgx[2][tx] + s_dgx[3][tx]);
      const double dgy_v = (s_dgy[0][tx] + s_dgy[1][tx]) + (s_dgy[2][tx] + s_dgy[3][tx]);
      const double dgz_v = (s_dgz[0][tx] + s_dgz[1][tx]) + (s_dgz[2][tx] + s_dgz[3][tx]);

      const double gx0 = fxc_cache[1 * npoints_total + gp];
      const double gy0 = fxc_cache[2 * npoints_total + gp];
      const double gz0 = fxc_cache[3 * npoints_total + gp];
      const double vs0 = fxc_cache[4 * npoints_total + gp];
      const double v2rs = fxc_cache[5 * npoints_total + gp];
      const double v2s2 = fxc_cache[6 * npoints_total + gp];

      const double dsig = 2.0 * (gx0 * dgx_v + gy0 * dgy_v + gz0 * dgz_v);
      d_vrho += v2rs * dsig;
      double d_vsigma = v2rs * dr + v2s2 * dsig;

      if (family == 2) {
        const double dt = (s_dtau[0][tx] + s_dtau[1][tx]) + (s_dtau[2][tx] + s_dtau[3][tx]);
        const double v2rt = fxc_cache[7 * npoints_total + gp];
        const double v2st = fxc_cache[8 * npoints_total + gp];
        const double v2t2 = fxc_cache[9 * npoints_total + gp];
        d_vrho += v2rt * dt;
        d_vsigma += v2st * dt;
        const double d_vtau = v2rt * dr + v2st * dsig + v2t2 * dt;
        wv_out[4 * ng + g] = 0.25 * wg * d_vtau;
      }

      wv_out[1 * ng + g] = 2.0 * wg * (d_vsigma * gx0 + vs0 * dgx_v);
      wv_out[2 * ng + g] = 2.0 * wg * (d_vsigma * gy0 + vs0 * dgy_v);
      wv_out[3 * ng + g] = 2.0 * wg * (d_vsigma * gz0 + vs0 * dgz_v);
    }
    wv_out[0 * ng + g] = 0.5 * wg * d_vrho;
  }
}

// Coalesced 2D (64 x 4) nuclear-gradient XC potential evaluator.
__global__ void grad_xc_point_weights_kernel(int func_id,
                                             int family,
                                             double rho_cutoff,
                                             std::size_t na,
                                             std::size_t ng,
                                             const double* __restrict__ weights,
                                             std::size_t point_offset,
                                             const double* __restrict__ phi,
                                             const double* __restrict__ f_mat,
                                             double* __restrict__ wv_out,
                                             double* __restrict__ e_xc_point_out) {
  constexpr int kTileG = 64;
  constexpr int kRedY = 4;
  __shared__ double s_rho[kRedY][kTileG];
  __shared__ double s_gx[kRedY][kTileG];
  __shared__ double s_gy[kRedY][kTileG];
  __shared__ double s_gz[kRedY][kTileG];
  __shared__ double s_tau[kRedY][kTileG];

  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const std::size_t g =
      static_cast<std::size_t>(blockIdx.x) * kTileG + static_cast<std::size_t>(tx);

  double rho = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, tau = 0.0;
  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* __restrict__ phi0 = phi;
    const double* __restrict__ phix = phi + 1 * comp_stride;
    const double* __restrict__ phiy = phi + 2 * comp_stride;
    const double* __restrict__ phiz = phi + 3 * comp_stride;
    const double* __restrict__ f0 = f_mat;

    for (std::size_t u = static_cast<std::size_t>(ty); u < na; u += kRedY) {
      const std::size_t off = u * ng + g;
      const double f0_ug = f0[off];
      rho += phi0[off] * f0_ug;
      if (family >= 1) {
        const double px = phix[off];
        const double py = phiy[off];
        const double pz = phiz[off];
        gx += 2.0 * px * f0_ug;
        gy += 2.0 * py * f0_ug;
        gz += 2.0 * pz * f0_ug;
        if (family == 2) {
          tau += 0.5 * (px * f_mat[1 * comp_stride + off] +
                        py * f_mat[2 * comp_stride + off] +
                        pz * f_mat[3 * comp_stride + off]);
        }
      }
    }
  }

  s_rho[ty][tx] = rho;
  if (family >= 1) {
    s_gx[ty][tx] = gx;
    s_gy[ty][tx] = gy;
    s_gz[ty][tx] = gz;
    if (family == 2) {
      s_tau[ty][tx] = tau;
    }
  }
  __syncthreads();

  if (ty == 0 && g < ng) {
    const double r = (s_rho[0][tx] + s_rho[1][tx]) + (s_rho[2][tx] + s_rho[3][tx]);
    double ggx = 0.0, ggy = 0.0, ggz = 0.0, tt = 0.0, sigma = 0.0;
    if (family >= 1) {
      ggx = (s_gx[0][tx] + s_gx[1][tx]) + (s_gx[2][tx] + s_gx[3][tx]);
      ggy = (s_gy[0][tx] + s_gy[1][tx]) + (s_gy[2][tx] + s_gy[3][tx]);
      ggz = (s_gz[0][tx] + s_gz[1][tx]) + (s_gz[2][tx] + s_gz[3][tx]);
      sigma = ggx * ggx + ggy * ggy + ggz * ggz;
      if (family == 2) {
        tt = (s_tau[0][tx] + s_tau[1][tx]) + (s_tau[2][tx] + s_tau[3][tx]);
      }
    }

    const double wg = weights[point_offset + g];
    xc_gen::XcPointOut pt;
    xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(func_id),
                          r, sigma, tt, 1, pt, rho_cutoff);

    if (e_xc_point_out != nullptr) {
      e_xc_point_out[point_offset + g] = r * pt.zk;
    }
    wv_out[0 * ng + g] = wg * pt.vrho;
    wv_out[1 * ng + g] = (family >= 1) ? (2.0 * wg * pt.vsigma * ggx) : 0.0;
    wv_out[2 * ng + g] = (family >= 1) ? (2.0 * wg * pt.vsigma * ggy) : 0.0;
    wv_out[3 * ng + g] = (family >= 1) ? (2.0 * wg * pt.vsigma * ggz) : 0.0;
    wv_out[4 * ng + g] = (family == 2) ? (wg * pt.vtau) : 0.0;
  }
}

__global__ void grad_xc_accum_atom_kernel(int family,
                                          std::size_t na,
                                          std::size_t ng,
                                          std::size_t wv_stride,
                                          const int* __restrict__ act_aos,
                                          const int* __restrict__ ao_to_atom,
                                          const double* __restrict__ phi,
                                          const double* __restrict__ f_mat,
                                          const double* __restrict__ wv,
                                          double* __restrict__ d_grad) {
  const std::size_t u = static_cast<std::size_t>(blockIdx.y);
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (u >= na) return;
  const int ia = ao_to_atom[act_aos[u]];
  if (ia < 0) return;

  double gx_acc = 0.0, gy_acc = 0.0, gz_acc = 0.0;
  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const std::size_t off = u * ng + g;
    const double f0_ug = f_mat[off];
    const double wv0 = wv[0 * wv_stride + g];
    double w0_ug = wv0 * f0_ug;

    if (family >= 1) {
      const double wvx = wv[1 * wv_stride + g];
      const double wvy = wv[2 * wv_stride + g];
      const double wvz = wv[3 * wv_stride + g];
      const double wvt = (family == 2) ? wv[4 * wv_stride + g] : 0.0;

      const double fx_ug = f_mat[1 * comp_stride + off];
      const double fy_ug = f_mat[2 * comp_stride + off];
      const double fz_ug = f_mat[3 * comp_stride + off];
      w0_ug += wvx * fx_ug + wvy * fy_ug + wvz * fz_ug;

      const double wbx = 2.0 * wvx * f0_ug + ((family == 2) ? wvt * fx_ug : 0.0);
      const double wby = 2.0 * wvy * f0_ug + ((family == 2) ? wvt * fy_ug : 0.0);
      const double wbz = 2.0 * wvz * f0_ug + ((family == 2) ? wvt * fz_ug : 0.0);

      const double pxx = phi[4 * comp_stride + off];
      const double pxy = phi[5 * comp_stride + off];
      const double pxz = phi[6 * comp_stride + off];
      const double pyy = phi[7 * comp_stride + off];
      const double pyz = phi[8 * comp_stride + off];
      const double pzz = phi[9 * comp_stride + off];

      gx_acc -= (pxx * wbx + pxy * wby + pxz * wbz);
      gy_acc -= (pxy * wbx + pyy * wby + pyz * wbz);
      gz_acc -= (pxz * wbx + pyz * wby + pzz * wbz);
    }

    gx_acc -= 2.0 * phi[1 * comp_stride + off] * w0_ug;
    gy_acc -= 2.0 * phi[2 * comp_stride + off] * w0_ug;
    gz_acc -= 2.0 * phi[3 * comp_stride + off] * w0_ug;
  }

  for (int mask = 16; mask > 0; mask >>= 1) {
    gx_acc += __shfl_down_sync(0xffffffffu, gx_acc, mask);
    gy_acc += __shfl_down_sync(0xffffffffu, gy_acc, mask);
    gz_acc += __shfl_down_sync(0xffffffffu, gz_acc, mask);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&d_grad[3 * ia + 0], gx_acc);
    atomicAdd(&d_grad[3 * ia + 1], gy_acc);
    atomicAdd(&d_grad[3 * ia + 2], gz_acc);
  }
}

// Moving-grid Term 2 for `grid_response`: 1 thread per point `g` (constant `parent_atom[gp]`),
// loops over active AOs `u`, issues 1 atomicAdd per point onto `d_grad[ib]`.
__global__ void grad_xc_move_grid_kernel(int family,
                                         std::size_t na,
                                         std::size_t ng,
                                         std::size_t wv_stride,
                                         std::size_t point_offset,
                                         const int* __restrict__ parent_atom,
                                         const double* __restrict__ phi,
                                         const double* __restrict__ f_mat,
                                         const double* __restrict__ wv,
                                         double* __restrict__ d_grad) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= ng) return;
  const int ib = parent_atom[point_offset + g];
  if (ib < 0) return;

  const double wv0 = wv[0 * wv_stride + g];
  const double wvx = (family >= 1) ? wv[1 * wv_stride + g] : 0.0;
  const double wvy = (family >= 1) ? wv[2 * wv_stride + g] : 0.0;
  const double wvz = (family >= 1) ? wv[3 * wv_stride + g] : 0.0;
  const double wvt = (family == 2) ? wv[4 * wv_stride + g] : 0.0;

  const std::size_t comp_stride = na * ng;
  double gx_acc = 0.0, gy_acc = 0.0, gz_acc = 0.0;
  for (std::size_t u = 0; u < na; ++u) {
    const std::size_t off = u * ng + g;
    const double f0_ug = f_mat[off];
    double w0_ug = wv0 * f0_ug;

    if (family >= 1) {
      const double fx_ug = f_mat[1 * comp_stride + off];
      const double fy_ug = f_mat[2 * comp_stride + off];
      const double fz_ug = f_mat[3 * comp_stride + off];
      w0_ug += wvx * fx_ug + wvy * fy_ug + wvz * fz_ug;

      const double wbx = 2.0 * wvx * f0_ug + ((family == 2) ? wvt * fx_ug : 0.0);
      const double wby = 2.0 * wvy * f0_ug + ((family == 2) ? wvt * fy_ug : 0.0);
      const double wbz = 2.0 * wvz * f0_ug + ((family == 2) ? wvt * fz_ug : 0.0);

      const double pxx = phi[4 * comp_stride + off];
      const double pxy = phi[5 * comp_stride + off];
      const double pxz = phi[6 * comp_stride + off];
      const double pyy = phi[7 * comp_stride + off];
      const double pyz = phi[8 * comp_stride + off];
      const double pzz = phi[9 * comp_stride + off];

      gx_acc += (pxx * wbx + pxy * wby + pxz * wbz);
      gy_acc += (pxy * wbx + pyy * wby + pyz * wbz);
      gz_acc += (pxz * wbx + pyz * wby + pzz * wbz);
    }

    gx_acc += 2.0 * phi[1 * comp_stride + off] * w0_ug;
    gy_acc += 2.0 * phi[2 * comp_stride + off] * w0_ug;
    gz_acc += 2.0 * phi[3 * comp_stride + off] * w0_ug;
  }

  atomicAdd(&d_grad[3 * ib + 0], gx_acc);
  atomicAdd(&d_grad[3 * ib + 1], gy_acc);
  atomicAdd(&d_grad[3 * ib + 2], gz_acc);
}

struct BatchGpuMeta {
  std::size_t point_offset = 0;
  std::size_t ng = 0;
  std::size_t na = 0;
  int n_act_shells = 0;
  std::size_t shell_list_offset = 0;
  std::size_t ao_list_offset = 0;
  std::size_t phi_cache_offset = 0;
};

struct XcStreamWorker {
  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  cublasHandle_t cublas = nullptr;

  DevBuf<double> phi_scratch;
  DevBuf<double> d_act;
  DevBuf<double> d_act_beta;
  DevBuf<double> t_mat;
  DevBuf<double> t_mat_beta;
  DevBuf<double> w_mat;
  DevBuf<double> v_half;
  DevBuf<double> wv;
  DevBuf<double> scalars;
  DevBuf<double> vxc_part;
  DevBuf<double> vxc_part_beta;

  DevBuf<float> f32_a;
  DevBuf<float> f32_b;
  DevBuf<float> f32_c;

  XcStreamWorker() = default;
  ~XcStreamWorker() {
    if (cublas != nullptr) cublasDestroy(cublas);
    if (done_event != nullptr) cudaEventDestroy(done_event);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
  XcStreamWorker(const XcStreamWorker&) = delete;
  XcStreamWorker& operator=(const XcStreamWorker&) = delete;

  void init_handles() {
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "xc stream create");
    check_cuda(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), "xc event create");
    check_cublas(cublasCreate(&cublas), "cublasCreate (XcStreamWorker)");
    check_cublas(cublasSetStream(cublas, stream), "cublasSetStream (XcStreamWorker)");
  }

  // Row-major C[M x N] = A[M x K] * B[K x N]
  void gemm_nn(int M, int N, int K, const double* dA, const double* dB, double* dC,
               bool accum, Precision prec) {
    if (prec == Precision::FP32) {
      const std::size_t szA = static_cast<std::size_t>(M) * K;
      const std::size_t szB = static_cast<std::size_t>(K) * N;
      const std::size_t szC = static_cast<std::size_t>(M) * N;
      cast_f64_to_f32_kernel<<<static_cast<unsigned>((szA + 255) / 256), 256, 0, stream>>>(
          dA, f32_a.data(), szA);
      cast_f64_to_f32_kernel<<<static_cast<unsigned>((szB + 255) / 256), 256, 0, stream>>>(
          dB, f32_b.data(), szB);
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               f32_b.data(), N,
                               f32_a.data(), K,
                               &beta, f32_c.data(), N),
                   "cublasSgemm NN");
      cast_f32_to_f64_kernel<<<static_cast<unsigned>((szC + 255) / 256), 256, 0, stream>>>(
          f32_c.data(), dC, szC, accum ? 1 : 0);
      return;
    }

    const double alpha = 1.0;
    const double beta = accum ? 1.0 : 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             dB, N,
                             dA, K,
                             &beta, dC, N),
                 "cublasDgemm NN");
  }

  // Strided-batched row-major C_b[M x N] = A[M x K] * B_b[K x N] for b in [0, batch_count)
  void gemm_nn_strided_b(int M, int N, int K, const double* dA, const double* dB,
                         long long strideB, double* dC, long long strideC,
                         int batch_count, Precision prec) {
    if (prec == Precision::FP32) {
      for (int b = 0; b < batch_count; ++b) {
        gemm_nn(M, N, K, dA, dB + b * strideB, dC + b * strideC, false, prec);
      }
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemmStridedBatched(
                     cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K, &alpha,
                     dB, N, strideB,
                     dA, K, 0LL,
                     &beta, dC, N, strideC, batch_count),
                 "cublasDgemmStridedBatched NN");
  }

  // Row-major C[M x N] = A[M x K] * B[N x K]^T
  void gemm_nt(int M, int N, int K, const double* dA, const double* dB, double* dC,
               bool accum, Precision prec) {
    if (prec == Precision::FP32) {
      const std::size_t szA = static_cast<std::size_t>(M) * K;
      const std::size_t szB = static_cast<std::size_t>(N) * K;
      const std::size_t szC = static_cast<std::size_t>(M) * N;
      cast_f64_to_f32_kernel<<<static_cast<unsigned>((szA + 255) / 256), 256, 0, stream>>>(
          dA, f32_a.data(), szA);
      cast_f64_to_f32_kernel<<<static_cast<unsigned>((szB + 255) / 256), 256, 0, stream>>>(
          dB, f32_b.data(), szB);
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               f32_b.data(), K,
                               f32_a.data(), K,
                               &beta, f32_c.data(), N),
                   "cublasSgemm NT");
      cast_f32_to_f64_kernel<<<static_cast<unsigned>((szC + 255) / 256), 256, 0, stream>>>(
          f32_c.data(), dC, szC, accum ? 1 : 0);
      return;
    }

    const double alpha = 1.0;
    const double beta = accum ? 1.0 : 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             dB, K,
                             dA, K,
                             &beta, dC, N),
                 "cublasDgemm NT");
  }
};

struct CudaXcEngineImpl {
  static constexpr int kMaxStreams = 4;

  DevFlatBasis basis;
  int func_id = 0;
  int family = 0;
  bool is_hf = false;
  const void* libxc_state = nullptr;
  XcOptions opts;
  std::size_t npoints = 0;

  DevBuf<double> coords;
  DevBuf<double> weights;
  std::vector<double> h_weights;
  bool has_parent_atom = false;
  DevBuf<int> parent_atom;
  std::vector<BatchGpuMeta> batches;
  DevBuf<int> act_shells;
  DevBuf<int> act_shell_local_ao;
  DevBuf<int> act_aos;

  // Persistent GPU active-AO Phi cache across all super-batches (when within VRAM budget)
  bool ao_cached = false;
  DevBuf<double> phi_cache;

  // Persistent multi-stream workers (zero cudaMalloc during SCF / TDDFT iterations!)
  int num_workers = 0;
  std::array<XcStreamWorker, kMaxStreams> workers;
  cudaEvent_t fork_event = nullptr;
  DevBuf<double> total_scalars;

  // Pre-allocated Libxc staging buffers (zero cudaMalloc in eval_vxc_device / eval_vxc_uks_device!)
  DevBuf<double> d_dens_in;
  DevBuf<double> d_wv_all;
  PinnedBuf<double> h_dens_in;
  PinnedBuf<double> h_wv_all;

  // Host-API staging buffers
  std::size_t max_na = 0;
  std::size_t max_ng = 0;
  DevBuf<double> dm_host_buf;
  DevBuf<double> vxc_host_buf;
  DevBuf<double> dm_beta_host_buf;
  DevBuf<double> vxc_beta_host_buf;

  // Lazy nuclear-gradient scratch buffer (deriv=1 or 2 -> up to 10 components)
  DevBuf<double> grad_phi_scratch;

  // Lazy ground-state f_xc[rho_0] cache for TDDFT linear response
  bool fxc_ready = false;
  DevBuf<double> fxc_cache;

  std::size_t rks_dens_planes() const noexcept {
    return (family == 0) ? 1 : ((family == 1) ? 4 : 5);
  }

  void ensure_uks_buffers() {
    const std::size_t nao = static_cast<std::size_t>(basis.nao);
    const std::size_t nao2 = nao * nao;
    const std::size_t uks_planes = 2 * rks_dens_planes();
    if (npoints > 0) {
      d_dens_in.resize(uks_planes * npoints);
      d_wv_all.resize(uks_planes * npoints);
      h_dens_in.resize(uks_planes * npoints);
      h_wv_all.resize(uks_planes * npoints);
    }
    for (int s = 0; s < num_workers; ++s) {
      if (max_na > 0 && max_ng > 0) {
        workers[s].d_act_beta.resize(max_na * max_na);
        const std::size_t n_tcomp = (family == 2) ? 4 : 1;
        workers[s].t_mat_beta.resize(n_tcomp * max_na * max_ng);
        if (num_workers >= 2 && nao2 > 0) {
          workers[s].vxc_part_beta.resize(nao2);
        }
      }
    }
    if (nao2 > 0) {
      dm_beta_host_buf.resize(nao2);
      vxc_beta_host_buf.resize(nao2);
    }
  }

  CudaXcEngineImpl(const FlatBasisData& flat_b,
                   const MolecularGrid& grid,
                   int fid,
                   int fam,
                   bool hf_flag,
                   const void* libxc_ptr,
                   const XcOptions& options)
      : basis(flat_b),
        func_id(fid),
        family(fam),
        is_hf(hf_flag),
        libxc_state(libxc_ptr),
        opts(options),
        npoints(grid.npoints()),
        coords(grid.coords()),
        weights(grid.weights()),
        h_weights(grid.weights()),
        has_parent_atom(grid.has_grid_response_data()) {
    if (has_parent_atom) {
      parent_atom.upload(grid.parent_atoms().data(), grid.parent_atoms().size());
    }
    check_cuda(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming),
               "xc fork_event create");

    // Coalesce spatial batches into ~2048-pt GPU super-batches even if caller built 256-pt leaves!
    const std::vector<SuperBatch> super_batches =
        build_super_batches(grid, flat_b, 2048, 1.40);

    const int deriv_vxc = (family == 0) ? 0 : 1;
    const std::size_t ncomp_vxc = (deriv_vxc == 0) ? 1 : 4;
    std::size_t total_phi_elems = 0;

    std::vector<int> h_act_shells;
    std::vector<int> h_act_shell_local_ao;
    std::vector<int> h_act_aos;

    batches.reserve(super_batches.size());
    for (const SuperBatch& sb : super_batches) {
      if (sb.npoints == 0 || sb.active_aos.empty()) continue;
      BatchGpuMeta m;
      m.point_offset = sb.point_offset;
      m.ng = sb.npoints;
      m.na = sb.active_aos.size();
      m.n_act_shells = static_cast<int>(sb.active_shells.size());
      m.shell_list_offset = h_act_shells.size();
      m.ao_list_offset = h_act_aos.size();
      m.phi_cache_offset = total_phi_elems;
      total_phi_elems += ncomp_vxc * m.na * m.ng;

      int local_cursor = 0;
      for (int is : sb.active_shells) {
        h_act_shells.push_back(is);
        h_act_shell_local_ao.push_back(local_cursor);
        local_cursor += 2 * flat_b.shell_l[static_cast<std::size_t>(is)] + 1;
      }
      for (int ao : sb.active_aos) {
        h_act_aos.push_back(ao);
      }

      max_na = std::max(max_na, m.na);
      max_ng = std::max(max_ng, m.ng);
      batches.push_back(m);
    }

    act_shells.upload(h_act_shells.data(), h_act_shells.size());
    act_shell_local_ao.upload(h_act_shell_local_ao.data(), h_act_shell_local_ao.size());
    act_aos.upload(h_act_aos.data(), h_act_aos.size());

    const std::size_t nao = static_cast<std::size_t>(flat_b.nao);
    const std::size_t nao2 = nao * nao;
    num_workers = static_cast<int>(std::min<std::size_t>(
        kMaxStreams, std::max<std::size_t>(1, batches.size())));

    // Check VRAM budget for persistent GPU active-AO Phi cache
    if (opts.cache_ao_on_device && total_phi_elems > 0 && max_na > 0 && max_ng > 0) {
      std::size_t free_bytes = 0, total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        free_bytes = opts.max_ao_cache_mb * (1024ULL * 1024ULL) * 2ULL;
      }
      const std::size_t user_cap = opts.max_ao_cache_mb * (1024ULL * 1024ULL);
      const std::size_t safe_free =
          (free_bytes > (512ULL << 20)) ? ((free_bytes * 3ULL) / 4ULL) : 0ULL;
      const std::size_t budget_bytes = std::min(user_cap, safe_free);
      if (total_phi_elems * sizeof(double) <= budget_bytes) {
        phi_cache.resize(total_phi_elems);
        ao_cached = true;
      }
    }

    const std::size_t n_tcomp = (family == 2) ? 4 : 1;
    for (int s = 0; s < num_workers; ++s) {
      workers[s].init_handles();
      if (max_na > 0 && max_ng > 0) {
        if (!ao_cached) {
          workers[s].phi_scratch.resize(ncomp_vxc * max_na * max_ng);
        }
        workers[s].d_act.resize(max_na * max_na);
        workers[s].t_mat.resize(n_tcomp * max_na * max_ng);
        workers[s].w_mat.resize(max_na * max_ng);
        workers[s].v_half.resize(max_na * max_na);
        workers[s].wv.resize(5 * max_ng);
        workers[s].scalars.resize(2);
        if (num_workers >= 2 && nao2 > 0) {
          workers[s].vxc_part.resize(nao2);
        }
        if (opts.precision == Precision::FP32) {
          const std::size_t max_dim = std::max(max_na * max_na, max_na * max_ng);
          workers[s].f32_a.resize(max_dim);
          workers[s].f32_b.resize(max_dim);
          workers[s].f32_c.resize(max_dim);
        }
      }
    }
    total_scalars.resize(2);

    if (nao2 > 0) {
      dm_host_buf.resize(nao2);
      vxc_host_buf.resize(nao2);
    }

    if ((opts.use_libxc || func_id == 0) && npoints > 0) {
      const std::size_t planes = opts.spin_polarized ? (2 * rks_dens_planes()) : rks_dens_planes();
      d_dens_in.resize(planes * npoints);
      d_wv_all.resize(planes * npoints);
      h_dens_in.resize(planes * npoints);
      h_wv_all.resize(planes * npoints);
    }
    if (opts.spin_polarized) {
      ensure_uks_buffers();
    }

    // Ensure all default-stream H2D uploads (basis, coords, weights, batches)
    // complete before non-blocking worker/SCF streams read them!
    check_cuda(cudaStreamSynchronize(nullptr), "xc constructor H2D sync");

    // Precompute compacted active-AO Phi ONCE across worker streams when cached!
    if (ao_cached) {
      const FlatBasisView bview = basis.view();
      for (std::size_t ib = 0; ib < batches.size(); ++ib) {
        const BatchGpuMeta& b = batches[ib];
        cudaStream_t ws_stream = workers[ib % static_cast<std::size_t>(num_workers)].stream;
        const dim3 ao_grid(static_cast<unsigned>((b.ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws_stream>>>(
            bview, coords.data(), b.point_offset, b.ng, b.na,
            act_shells.data() + b.shell_list_offset,
            act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, deriv_vxc, phi_cache.data() + b.phi_cache_offset);
      }
      for (int s = 0; s < num_workers; ++s) {
        check_cuda(cudaStreamSynchronize(workers[s].stream), "xc ao_cache init sync");
      }
    }
  }

  ~CudaXcEngineImpl() {
    if (fork_event != nullptr) cudaEventDestroy(fork_event);
  }
};

double bragg_slater_radius_cuda(int z) {
  static constexpr double kRadiiAng[] = {
      0.75,
      0.35, 0.30,
      1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 0.45,
      1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 0.95,
      2.20, 1.80, 1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35,
      1.35, 1.35, 1.30, 1.25, 1.15, 1.15, 1.15, 1.10,
      2.35, 2.00, 1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40,
      1.60, 1.55, 1.55, 1.45, 1.45, 1.40, 1.40, 1.35
  };
  constexpr double kAngToBohr = 1.8897261246257702;
  if (z >= 0 && z < static_cast<int>(sizeof(kRadiiAng) / sizeof(kRadiiAng[0]))) {
    return kRadiiAng[z] * kAngToBohr;
  }
  return 1.50 * kAngToBohr;
}

void launch_partition_weight_grad_cuda(const CudaXcEngineImpl* st,
                                       const MolecularGrid& grid,
                                       std::span<const XcAtom> atoms,
                                       const double* d_e_xc_point,
                                       double* d_grad,
                                       cudaStream_t stream) {
  const int natom = static_cast<int>(atoms.size());
  if (natom <= 1 || st->npoints == 0 || !st->has_parent_atom) return;

  std::vector<double> atom_coords(static_cast<std::size_t>(natom) * 3);
  std::vector<double> bragg_radii(static_cast<std::size_t>(natom));
  for (int i = 0; i < natom; ++i) {
    atom_coords[3 * i + 0] = atoms[static_cast<std::size_t>(i)].center[0];
    atom_coords[3 * i + 1] = atoms[static_cast<std::size_t>(i)].center[1];
    atom_coords[3 * i + 2] = atoms[static_cast<std::size_t>(i)].center[2];
    bragg_radii[static_cast<std::size_t>(i)] =
        bragg_slater_radius_cuda(atoms[static_cast<std::size_t>(i)].atomic_number);
  }

  std::vector<double> inv_rij(static_cast<std::size_t>(natom * natom), 0.0);
  std::vector<double> a_ij(static_cast<std::size_t>(natom * natom), 0.0);
  const bool radii_adjust = grid.becke_radii_adjust();
  for (int i = 0; i < natom; ++i) {
    for (int j = 0; j < natom; ++j) {
      if (i == j) continue;
      const double dx = atom_coords[3 * i + 0] - atom_coords[3 * j + 0];
      const double dy = atom_coords[3 * i + 1] - atom_coords[3 * j + 1];
      const double dz = atom_coords[3 * i + 2] - atom_coords[3 * j + 2];
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      inv_rij[i * natom + j] = (d > 1e-12) ? (1.0 / d) : 0.0;
      if (radii_adjust) {
        const double chi = bragg_radii[i] / bragg_radii[j];
        const double u = (chi - 1.0) / (chi + 1.0);
        const double a = u / (u * u - 1.0);
        a_ij[i * natom + j] = std::clamp(a, -0.5, 0.5);
      }
    }
  }

  DevBuf<double> d_atoms;
  d_atoms.upload(atom_coords.data(), atom_coords.size(), stream);
  DevBuf<double> d_inv_rij;
  d_inv_rij.upload(inv_rij.data(), inv_rij.size(), stream);
  DevBuf<double> d_aij;
  d_aij.upload(a_ij.data(), a_ij.size(), stream);

  const unsigned blocks = static_cast<unsigned>((st->npoints + 127) / 128);
  partition_weight_grad_kernel<<<blocks, 128, 0, stream>>>(
      st->coords.data(), st->parent_atom.data(), st->weights.data(), d_e_xc_point,
      st->npoints, d_atoms.data(), d_inv_rij.data(), d_aij.data(),
      natom, (grid.partition_scheme() == PartitionScheme::Stratmann) ? 1 : 0, d_grad);
  check_cuda(cudaGetLastError(), "partition_weight_grad_kernel launch");
  check_cuda(cudaStreamSynchronize(stream), "partition_weight_grad_kernel sync");
}

}  // namespace

int xc_cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

void compute_partition_weights_cuda(const double* raw_coords,
                                    const int* parent_atom,
                                    std::size_t n_raw,
                                    const double* atom_coords,
                                    const double* bragg_radii,
                                    int natom,
                                    PartitionScheme scheme,
                                    bool radii_adjust,
                                    double* weights_inout) {
  if (n_raw == 0 || natom <= 1) return;

  std::vector<double> inv_rij(static_cast<std::size_t>(natom * natom), 0.0);
  std::vector<double> a_ij(static_cast<std::size_t>(natom * natom), 0.0);
  for (int i = 0; i < natom; ++i) {
    for (int j = 0; j < natom; ++j) {
      if (i == j) continue;
      const double dx = atom_coords[3 * i + 0] - atom_coords[3 * j + 0];
      const double dy = atom_coords[3 * i + 1] - atom_coords[3 * j + 1];
      const double dz = atom_coords[3 * i + 2] - atom_coords[3 * j + 2];
      const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
      inv_rij[i * natom + j] = (d > 1e-12) ? (1.0 / d) : 0.0;
      if (radii_adjust) {
        const double chi = bragg_radii[i] / bragg_radii[j];
        const double u = (chi - 1.0) / (chi + 1.0);
        const double a = u / (u * u - 1.0);
        a_ij[i * natom + j] = std::clamp(a, -0.5, 0.5);
      }
    }
  }

  DevBuf<double> d_coords(raw_coords, n_raw * 3);
  DevBuf<int> d_parent(parent_atom, n_raw);
  DevBuf<double> d_atoms(atom_coords, static_cast<std::size_t>(natom) * 3);
  DevBuf<double> d_inv_rij(inv_rij);
  DevBuf<double> d_aij(a_ij);
  DevBuf<double> d_weights(weights_inout, n_raw);

  const unsigned blocks = static_cast<unsigned>((n_raw + 255) / 256);
  partition_weights_kernel<<<blocks, 256>>>(
      d_coords.data(), d_parent.data(), n_raw,
      d_atoms.data(), d_inv_rij.data(), d_aij.data(),
      natom, (scheme == PartitionScheme::Stratmann) ? 1 : 0,
      d_weights.data());
  check_cuda(cudaGetLastError(), "partition_weights_kernel launch");
  d_weights.download(weights_inout, n_raw);
  check_cuda(cudaDeviceSynchronize(), "partition_weights_kernel sync");
}

void eval_xc_points_cuda(int func_id,
                         std::size_t n,
                         const double* rho,
                         const double* sigma,
                         const double* tau,
                         int deriv,
                         double rho_cutoff,
                         XcPointwiseResult& out) {
  DevBuf<double> d_rho(rho, n);
  DevBuf<double> d_sigma(sigma, n);
  DevBuf<double> d_tau(tau, n);
  DevBuf<double> d_zk(n);
  DevBuf<double> d_vr(deriv >= 1 ? n : 0);
  DevBuf<double> d_vs(deriv >= 1 ? n : 0);
  DevBuf<double> d_vt(deriv >= 1 ? n : 0);
  DevBuf<double> d_v2r2(deriv >= 2 ? n : 0);
  DevBuf<double> d_v2rs(deriv >= 2 ? n : 0);
  DevBuf<double> d_v2s2(deriv >= 2 ? n : 0);
  DevBuf<double> d_v2rt(deriv >= 2 ? n : 0);
  DevBuf<double> d_v2st(deriv >= 2 ? n : 0);
  DevBuf<double> d_v2t2(deriv >= 2 ? n : 0);

  const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
  eval_xc_points_kernel<<<blocks, 256>>>(
      func_id, n, d_rho.data(), d_sigma.data(), d_tau.data(), deriv, rho_cutoff,
      d_zk.data(), d_vr.data(), d_vs.data(), d_vt.data(),
      d_v2r2.data(), d_v2rs.data(), d_v2s2.data(),
      d_v2rt.data(), d_v2st.data(), d_v2t2.data());
  check_cuda(cudaGetLastError(), "eval_xc_points_kernel launch");

  d_zk.download(out.zk.data(), n);
  if (deriv >= 1) {
    d_vr.download(out.vrho.data(), n);
    d_vs.download(out.vsigma.data(), n);
    d_vt.download(out.vtau.data(), n);
  }
  if (deriv >= 2) {
    d_v2r2.download(out.v2rho2.data(), n);
    d_v2rs.download(out.v2rhosigma.data(), n);
    d_v2s2.download(out.v2sigma2.data(), n);
    d_v2rt.download(out.v2rhotau.data(), n);
    d_v2st.download(out.v2sigmatau.data(), n);
    d_v2t2.download(out.v2tau2.data(), n);
  }
  check_cuda(cudaDeviceSynchronize(), "eval_xc_points_cuda sync");
}

void eval_ao_grid_cuda(const FlatBasisData& flat_basis,
                       const double* coords_xyz,
                       std::size_t npts,
                       int deriv,
                       double* out_ao) {
  const std::size_t nao = static_cast<std::size_t>(flat_basis.nao);
  const std::size_t ncomp = (deriv == 0) ? 1 : ((deriv == 1) ? 4 : 10);
  const std::size_t total = ncomp * npts * nao;

  DevFlatBasis d_basis(flat_basis);
  DevBuf<double> d_coords(coords_xyz, npts * 3);
  DevBuf<double> d_out(total);
  check_cuda(cudaMemset(d_out.data(), 0, total * sizeof(double)), "eval_ao_grid memset");

  const dim3 threads(128, 1, 1);
  const dim3 grid_dim(static_cast<unsigned>((npts + 127) / 128),
                      static_cast<unsigned>(flat_basis.nshell), 1);
  eval_ao_full_grid_kernel<<<grid_dim, threads>>>(
      d_basis.view(), d_coords.data(), npts, deriv, d_out.data());
  check_cuda(cudaGetLastError(), "eval_ao_full_grid_kernel launch");
  d_out.download(out_ao, total);
  check_cuda(cudaDeviceSynchronize(), "eval_ao_grid_cuda sync");
}

void* xc_cuda_engine_create(const FlatBasisData& flat_basis,
                            const MolecularGrid& grid,
                            int func_id,
                            int family,
                            bool is_hf,
                            const void* libxc_state,
                            const XcOptions& opts) {
  return new CudaXcEngineImpl(flat_basis, grid, func_id, family, is_hf, libxc_state, opts);
}

void xc_cuda_engine_destroy(void* impl) noexcept {
  delete static_cast<CudaXcEngineImpl*>(impl);
}

void xc_cuda_set_grid_response(void* impl, bool enabled) noexcept {
  auto* st = static_cast<CudaXcEngineImpl*>(impl);
  if (st != nullptr) {
    st->opts.grid_response = enabled;
  }
}

double xc_cuda_eval_vxc_device(void* raw_impl,
                               const double* d_dm,
                               double* d_vxc,
                               double* nelec_out,
                               void* cuda_stream) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  cudaStream_t caller_stream = static_cast<cudaStream_t>(cuda_stream);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const std::size_t nao2 = nao * nao;

  if (st->batches.empty() || nao == 0) {
    if (nao2 > 0) {
      check_cuda(cudaMemsetAsync(d_vxc, 0, nao2 * sizeof(double), caller_stream),
                 "xc zero d_vxc empty");
    }
    if (nelec_out) *nelec_out = 0.0;
    return 0.0;
  }

  const int n_streams = st->num_workers;
  const int deriv = (st->family == 0) ? 0 : 1;
  const FlatBasisView bview = st->basis.view();
  const bool use_libxc_path = st->opts.use_libxc || (st->func_id == 0);

  if (use_libxc_path) {
    // Pass 1: Evaluate rho, grad_rho, tau across all batches into flat d_dens_in
    check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc libxc fork1");
    for (int s = 0; s < n_streams; ++s) {
      XcStreamWorker& ws = st->workers[s];
      check_cuda(cudaStreamWaitEvent(ws.stream, st->fork_event, 0), "xc libxc wait1");
      double* target_vxc = (n_streams == 1) ? d_vxc : ws.vxc_part.data();
      check_cuda(cudaMemsetAsync(target_vxc, 0, nao2 * sizeof(double), ws.stream),
                 "xc zero target_vxc");
    }

    for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
      const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
      XcStreamWorker& ws = st->workers[s];
      const BatchGpuMeta& b = st->batches[ib];
      const std::size_t na = b.na;
      const std::size_t ng = b.ng;
      const std::size_t comp_stride = na * ng;
      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const double* d_phi = nullptr;
      if (st->ao_cached) {
        d_phi = st->phi_cache.data() + b.phi_cache_offset;
      } else {
        const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
            bview, st->coords.data(), b.point_offset, ng, na,
            st->act_shells.data() + b.shell_list_offset,
            st->act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, deriv, ws.phi_scratch.data());
        d_phi = ws.phi_scratch.data();
      }

      gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
          d_dm, nao, d_batch_aos, na, ws.d_act.data());

      if (st->family == 2) {
        ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), d_phi,
                             static_cast<long long>(comp_stride),
                             ws.t_mat.data(), static_cast<long long>(comp_stride),
                             4, st->opts.precision);
      } else {
        ws.gemm_nn(i_na, i_ng, i_na, ws.d_act.data(), d_phi, ws.t_mat.data(),
                   false, st->opts.precision);
      }

      const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
      const dim3 pot_threads(64, 4, 1);
      eval_density_flat_rks_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
          st->family, na, ng, st->npoints, b.point_offset,
          d_phi, ws.t_mat.data(), st->d_dens_in.data());
    }

    for (int s = 0; s < n_streams; ++s) {
      check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
                 "xc libxc join1 record");
      check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
                 "xc libxc join1 wait");
    }

    const std::size_t planes = st->rks_dens_planes();
    const std::size_t n_flat = planes * st->npoints;
    check_cuda(cudaMemcpyAsync(st->h_dens_in.data(), st->d_dens_in.data(),
                               n_flat * sizeof(double), cudaMemcpyDeviceToHost, caller_stream),
               "xc libxc D2H dens");
    check_cuda(cudaStreamSynchronize(caller_stream), "xc libxc D2H sync");

    double exc = 0.0, nelec = 0.0;
    eval_libxc_rks_grid(st->libxc_state, st->family, st->opts.rho_cutoff,
                        st->npoints, st->h_weights.data(), st->h_dens_in.data(),
                        /*for_grad=*/false, st->h_wv_all.data(), nullptr, &exc, &nelec);

    check_cuda(cudaMemcpyAsync(st->d_wv_all.data(), st->h_wv_all.data(),
                               n_flat * sizeof(double), cudaMemcpyHostToDevice, caller_stream),
               "xc libxc H2D wv");

    // Pass 2: Build W and scatter V_xc across worker streams
    check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc libxc fork2");
    for (int s = 0; s < n_streams; ++s) {
      check_cuda(cudaStreamWaitEvent(st->workers[s].stream, st->fork_event, 0), "xc libxc wait2");
    }

    for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
      const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
      XcStreamWorker& ws = st->workers[s];
      const BatchGpuMeta& b = st->batches[ib];
      const std::size_t na = b.na;
      const std::size_t ng = b.ng;
      const std::size_t comp_stride = na * ng;
      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const double* d_phi = nullptr;
      if (st->ao_cached) {
        d_phi = st->phi_cache.data() + b.phi_cache_offset;
      } else {
        const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
            bview, st->coords.data(), b.point_offset, ng, na,
            st->act_shells.data() + b.shell_list_offset,
            st->act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, deriv, ws.phi_scratch.data());
        d_phi = ws.phi_scratch.data();
      }

      const double* d_wv_batch = st->d_wv_all.data() + b.point_offset;
      const dim3 w_grid(static_cast<unsigned>((ng + 127) / 128),
                        static_cast<unsigned>(na), 1);
      build_vxc_weight_matrix_kernel<<<w_grid, 128, 0, ws.stream>>>(
          st->family, na, ng, st->npoints, d_phi, d_wv_batch, ws.w_mat.data());

      ws.gemm_nt(i_na, i_na, i_ng, d_phi, ws.w_mat.data(), ws.v_half.data(),
                 false, st->opts.precision);

      if (st->family == 2) {
        const double* d_wv4 = d_wv_batch + 4 * st->npoints;
        for (std::size_t dir = 1; dir <= 3; ++dir) {
          const double* phib = d_phi + dir * comp_stride;
          scale_mgga_tau_2d_kernel<<<w_grid, 128, 0, ws.stream>>>(
              na, ng, phib, d_wv4, ws.w_mat.data());
          ws.gemm_nt(i_na, i_na, i_ng, phib, ws.w_mat.data(), ws.v_half.data(),
                     true, st->opts.precision);
        }
      }

      double* target_vxc = (n_streams == 1) ? d_vxc : ws.vxc_part.data();
      scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
          ws.v_half.data(), d_batch_aos, na, nao, target_vxc);
    }

    for (int s = 0; s < n_streams; ++s) {
      check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
                 "xc libxc join2 record");
      check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
                 "xc libxc join2 wait");
    }

    if (n_streams >= 2) {
      reduce_stream_vxc_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
          nao2, n_streams,
          st->workers[0].vxc_part.data(),
          st->workers[1].vxc_part.data(),
          (n_streams >= 3) ? st->workers[2].vxc_part.data() : nullptr,
          (n_streams >= 4) ? st->workers[3].vxc_part.data() : nullptr,
          d_vxc);
    }
    check_cuda(cudaStreamSynchronize(caller_stream), "xc_cuda_eval_vxc_device libxc sync");
    if (nelec_out) *nelec_out = nelec;
    return exc;
  }

  // Zero-D2H symbolic xc_codegen GPU fast path (when opts.use_libxc == false)
  // Fork caller_stream onto concurrent worker streams
  check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc fork record");
  for (int s = 0; s < n_streams; ++s) {
    XcStreamWorker& ws = st->workers[s];
    check_cuda(cudaStreamWaitEvent(ws.stream, st->fork_event, 0), "xc fork wait");
    check_cuda(cudaMemsetAsync(ws.scalars.data(), 0, 2 * sizeof(double), ws.stream),
               "xc zero ws.scalars");
    double* target_vxc = (n_streams == 1) ? d_vxc : ws.vxc_part.data();
    check_cuda(cudaMemsetAsync(target_vxc, 0, nao2 * sizeof(double), ws.stream),
               "xc zero target_vxc");
  }

  for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
    const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
    XcStreamWorker& ws = st->workers[s];
    const BatchGpuMeta& b = st->batches[ib];
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const double* d_phi = nullptr;
    if (st->ao_cached) {
      d_phi = st->phi_cache.data() + b.phi_cache_offset;
    } else {
      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, ws.phi_scratch.data());
      d_phi = ws.phi_scratch.data();
    }

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        d_dm, nao, d_batch_aos, na, ws.d_act.data());

    if (st->family == 2) {
      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), d_phi,
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           4, st->opts.precision);
    } else {
      ws.gemm_nn(i_na, i_ng, i_na, ws.d_act.data(), d_phi, ws.t_mat.data(),
                 false, st->opts.precision);
    }

    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    eval_rho_and_vxc_potentials_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->func_id, st->family, st->opts.rho_cutoff,
        na, ng, st->weights.data(), b.point_offset,
        d_phi, ws.t_mat.data(), ws.wv.data(), ws.scalars.data());

    const dim3 w_grid(static_cast<unsigned>((ng + 127) / 128),
                      static_cast<unsigned>(na), 1);
    build_vxc_weight_matrix_kernel<<<w_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, ng, d_phi, ws.wv.data(), ws.w_mat.data());

    ws.gemm_nt(i_na, i_na, i_ng, d_phi, ws.w_mat.data(), ws.v_half.data(),
               false, st->opts.precision);

    if (st->family == 2) {
      const double* d_wv4 = ws.wv.data() + 4 * ng;
      for (std::size_t dir = 1; dir <= 3; ++dir) {
        const double* phib = d_phi + dir * comp_stride;
        scale_mgga_tau_2d_kernel<<<w_grid, 128, 0, ws.stream>>>(
            na, ng, phib, d_wv4, ws.w_mat.data());
        ws.gemm_nt(i_na, i_na, i_ng, phib, ws.w_mat.data(), ws.v_half.data(),
                   true, st->opts.precision);
      }
    }

    double* target_vxc = (n_streams == 1) ? d_vxc : ws.vxc_part.data();
    scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        ws.v_half.data(), d_batch_aos, na, nao, target_vxc);
  }

  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
               "xc join record");
    check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
               "xc join wait");
  }

  if (n_streams >= 2) {
    reduce_stream_vxc_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        nao2, n_streams,
        st->workers[0].vxc_part.data(),
        st->workers[1].vxc_part.data(),
        (n_streams >= 3) ? st->workers[2].vxc_part.data() : nullptr,
        (n_streams >= 4) ? st->workers[3].vxc_part.data() : nullptr,
        d_vxc);
  }

  double h_scalars[2] = {0.0, 0.0};
  if (n_streams == 1) {
    st->workers[0].scalars.download(h_scalars, 2, caller_stream);
  } else {
    reduce_stream_scalars_kernel<<<1, 32, 0, caller_stream>>>(
        n_streams,
        st->workers[0].scalars.data(),
        st->workers[1].scalars.data(),
        (n_streams >= 3) ? st->workers[2].scalars.data() : nullptr,
        (n_streams >= 4) ? st->workers[3].scalars.data() : nullptr,
        st->total_scalars.data());
    st->total_scalars.download(h_scalars, 2, caller_stream);
  }
  check_cuda(cudaStreamSynchronize(caller_stream), "xc_cuda_eval_vxc_device sync");
  if (nelec_out) *nelec_out = h_scalars[1];
  return h_scalars[0];
}

double xc_cuda_eval_vxc_host(void* raw_impl,
                             const double* h_dm,
                             double* h_vxc,
                             double* nelec_out) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao2 = static_cast<std::size_t>(st->basis.nao) * st->basis.nao;
  st->dm_host_buf.upload(h_dm, nao2);
  const double exc = xc_cuda_eval_vxc_device(
      raw_impl, st->dm_host_buf.data(), st->vxc_host_buf.data(), nelec_out, nullptr);
  st->vxc_host_buf.download(h_vxc, nao2);
  check_cuda(cudaDeviceSynchronize(), "xc_cuda_eval_vxc_host download sync");
  return exc;
}

double xc_cuda_eval_vxc_uks_device(void* raw_impl,
                                   const double* d_dm_alpha,
                                   const double* d_dm_beta,
                                   double* d_vxc_alpha,
                                   double* d_vxc_beta,
                                   double* nelec_alpha_out,
                                   double* nelec_beta_out,
                                   void* cuda_stream) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  cudaStream_t caller_stream = static_cast<cudaStream_t>(cuda_stream);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const std::size_t nao2 = nao * nao;

  if (st->batches.empty() || nao == 0) {
    if (nao2 > 0) {
      check_cuda(cudaMemsetAsync(d_vxc_alpha, 0, nao2 * sizeof(double), caller_stream),
                 "xc uks zero d_vxc_alpha empty");
      check_cuda(cudaMemsetAsync(d_vxc_beta, 0, nao2 * sizeof(double), caller_stream),
                 "xc uks zero d_vxc_beta empty");
    }
    if (nelec_alpha_out) *nelec_alpha_out = 0.0;
    if (nelec_beta_out) *nelec_beta_out = 0.0;
    return 0.0;
  }

  st->ensure_uks_buffers();

  const int n_streams = st->num_workers;
  const int deriv = (st->family == 0) ? 0 : 1;
  const FlatBasisView bview = st->basis.view();
  const std::size_t n_wv_comp = st->rks_dens_planes();
  const std::size_t n_flat = 2 * n_wv_comp * st->npoints;

  // Pass 1: Evaluate alpha and beta densities into flat d_dens_in
  check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc uks fork1");
  for (int s = 0; s < n_streams; ++s) {
    XcStreamWorker& ws = st->workers[s];
    check_cuda(cudaStreamWaitEvent(ws.stream, st->fork_event, 0), "xc uks wait1");
    double* target_va = (n_streams == 1) ? d_vxc_alpha : ws.vxc_part.data();
    double* target_vb = (n_streams == 1) ? d_vxc_beta : ws.vxc_part_beta.data();
    check_cuda(cudaMemsetAsync(target_va, 0, nao2 * sizeof(double), ws.stream),
               "xc uks zero target_va");
    check_cuda(cudaMemsetAsync(target_vb, 0, nao2 * sizeof(double), ws.stream),
               "xc uks zero target_vb");
  }

  for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
    const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
    XcStreamWorker& ws = st->workers[s];
    const BatchGpuMeta& b = st->batches[ib];
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const double* d_phi = nullptr;
    if (st->ao_cached) {
      d_phi = st->phi_cache.data() + b.phi_cache_offset;
    } else {
      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, ws.phi_scratch.data());
      d_phi = ws.phi_scratch.data();
    }

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        d_dm_alpha, nao, d_batch_aos, na, ws.d_act.data());
    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        d_dm_beta, nao, d_batch_aos, na, ws.d_act_beta.data());

    if (st->family == 2) {
      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), d_phi,
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           4, st->opts.precision);
      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act_beta.data(), d_phi,
                           static_cast<long long>(comp_stride),
                           ws.t_mat_beta.data(), static_cast<long long>(comp_stride),
                           4, st->opts.precision);
    } else {
      ws.gemm_nn(i_na, i_ng, i_na, ws.d_act.data(), d_phi, ws.t_mat.data(),
                 false, st->opts.precision);
      ws.gemm_nn(i_na, i_ng, i_na, ws.d_act_beta.data(), d_phi, ws.t_mat_beta.data(),
                 false, st->opts.precision);
    }

    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    eval_density_flat_uks_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, b.point_offset,
        d_phi, ws.t_mat.data(), ws.t_mat_beta.data(), st->d_dens_in.data());
  }

  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
               "xc uks join1 record");
    check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
               "xc uks join1 wait");
  }

  check_cuda(cudaMemcpyAsync(st->h_dens_in.data(), st->d_dens_in.data(),
                             n_flat * sizeof(double), cudaMemcpyDeviceToHost, caller_stream),
             "xc uks D2H dens");
  check_cuda(cudaStreamSynchronize(caller_stream), "xc uks D2H sync");

  double exc = 0.0, nelec_a = 0.0, nelec_b = 0.0;
  eval_libxc_uks_grid(st->libxc_state, st->family, st->opts.rho_cutoff,
                      st->npoints, st->h_weights.data(), st->h_dens_in.data(),
                      /*for_grad=*/false, st->h_wv_all.data(), nullptr,
                      &exc, &nelec_a, &nelec_b);

  check_cuda(cudaMemcpyAsync(st->d_wv_all.data(), st->h_wv_all.data(),
                             n_flat * sizeof(double), cudaMemcpyHostToDevice, caller_stream),
             "xc uks H2D wv");

  // Pass 2: Build W and scatter V_xc_alpha & V_xc_beta across worker streams
  check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc uks fork2");
  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaStreamWaitEvent(st->workers[s].stream, st->fork_event, 0), "xc uks wait2");
  }

  for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
    const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
    XcStreamWorker& ws = st->workers[s];
    const BatchGpuMeta& b = st->batches[ib];
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const double* d_phi = nullptr;
    if (st->ao_cached) {
      d_phi = st->phi_cache.data() + b.phi_cache_offset;
    } else {
      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, ws.phi_scratch.data());
      d_phi = ws.phi_scratch.data();
    }

    const dim3 w_grid(static_cast<unsigned>((ng + 127) / 128),
                      static_cast<unsigned>(na), 1);

    // Alpha V_xc
    const double* d_wv_a = st->d_wv_all.data() + b.point_offset;
    build_vxc_weight_matrix_kernel<<<w_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, d_phi, d_wv_a, ws.w_mat.data());
    ws.gemm_nt(i_na, i_na, i_ng, d_phi, ws.w_mat.data(), ws.v_half.data(),
               false, st->opts.precision);
    if (st->family == 2) {
      const double* d_wv4_a = d_wv_a + 4 * st->npoints;
      for (std::size_t dir = 1; dir <= 3; ++dir) {
        const double* phib = d_phi + dir * comp_stride;
        scale_mgga_tau_2d_kernel<<<w_grid, 128, 0, ws.stream>>>(
            na, ng, phib, d_wv4_a, ws.w_mat.data());
        ws.gemm_nt(i_na, i_na, i_ng, phib, ws.w_mat.data(), ws.v_half.data(),
                   true, st->opts.precision);
      }
    }
    double* target_va = (n_streams == 1) ? d_vxc_alpha : ws.vxc_part.data();
    scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        ws.v_half.data(), d_batch_aos, na, nao, target_va);

    // Beta V_xc
    const double* d_wv_b = st->d_wv_all.data() + n_wv_comp * st->npoints + b.point_offset;
    build_vxc_weight_matrix_kernel<<<w_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, d_phi, d_wv_b, ws.w_mat.data());
    ws.gemm_nt(i_na, i_na, i_ng, d_phi, ws.w_mat.data(), ws.v_half.data(),
               false, st->opts.precision);
    if (st->family == 2) {
      const double* d_wv4_b = d_wv_b + 4 * st->npoints;
      for (std::size_t dir = 1; dir <= 3; ++dir) {
        const double* phib = d_phi + dir * comp_stride;
        scale_mgga_tau_2d_kernel<<<w_grid, 128, 0, ws.stream>>>(
            na, ng, phib, d_wv4_b, ws.w_mat.data());
        ws.gemm_nt(i_na, i_na, i_ng, phib, ws.w_mat.data(), ws.v_half.data(),
                   true, st->opts.precision);
      }
    }
    double* target_vb = (n_streams == 1) ? d_vxc_beta : ws.vxc_part_beta.data();
    scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        ws.v_half.data(), d_batch_aos, na, nao, target_vb);
  }

  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
               "xc uks join2 record");
    check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
               "xc uks join2 wait");
  }

  if (n_streams >= 2) {
    reduce_stream_vxc_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        nao2, n_streams,
        st->workers[0].vxc_part.data(),
        st->workers[1].vxc_part.data(),
        (n_streams >= 3) ? st->workers[2].vxc_part.data() : nullptr,
        (n_streams >= 4) ? st->workers[3].vxc_part.data() : nullptr,
        d_vxc_alpha);
    reduce_stream_vxc_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        nao2, n_streams,
        st->workers[0].vxc_part_beta.data(),
        st->workers[1].vxc_part_beta.data(),
        (n_streams >= 3) ? st->workers[2].vxc_part_beta.data() : nullptr,
        (n_streams >= 4) ? st->workers[3].vxc_part_beta.data() : nullptr,
        d_vxc_beta);
  }

  check_cuda(cudaStreamSynchronize(caller_stream), "xc_cuda_eval_vxc_uks_device sync");
  if (nelec_alpha_out) *nelec_alpha_out = nelec_a;
  if (nelec_beta_out) *nelec_beta_out = nelec_b;
  return exc;
}

double xc_cuda_eval_vxc_uks_host(void* raw_impl,
                                 const double* h_dm_alpha,
                                 const double* h_dm_beta,
                                 double* h_vxc_alpha,
                                 double* h_vxc_beta,
                                 double* nelec_alpha_out,
                                 double* nelec_beta_out) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  st->ensure_uks_buffers();
  const std::size_t nao2 = static_cast<std::size_t>(st->basis.nao) * st->basis.nao;
  st->dm_host_buf.upload(h_dm_alpha, nao2);
  st->dm_beta_host_buf.upload(h_dm_beta, nao2);
  const double exc = xc_cuda_eval_vxc_uks_device(
      raw_impl, st->dm_host_buf.data(), st->dm_beta_host_buf.data(),
      st->vxc_host_buf.data(), st->vxc_beta_host_buf.data(),
      nelec_alpha_out, nelec_beta_out, nullptr);
  st->vxc_host_buf.download(h_vxc_alpha, nao2);
  st->vxc_beta_host_buf.download(h_vxc_beta, nao2);
  check_cuda(cudaDeviceSynchronize(), "xc_cuda_eval_vxc_uks_host download sync");
  return exc;
}

void xc_cuda_set_reference_density_device(void* raw_impl,
                                          const double* d_dm,
                                          void* cuda_stream) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  cudaStream_t caller_stream = static_cast<cudaStream_t>(cuda_stream);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);

  if (st->npoints == 0 || nao == 0 || st->batches.empty()) {
    st->fxc_ready = true;
    return;
  }

  const std::size_t n_fields = (st->family == 0) ? 1 : ((st->family == 1) ? 7 : 10);
  st->fxc_cache.resize(n_fields * st->npoints);
  check_cuda(cudaMemsetAsync(st->fxc_cache.data(), 0,
                             n_fields * st->npoints * sizeof(double), caller_stream),
             "xc zero fxc_cache");

  const int n_streams = st->num_workers;
  const int deriv = (st->family == 0) ? 0 : 1;
  const FlatBasisView bview = st->basis.view();

  check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc fxc_ref fork");
  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaStreamWaitEvent(st->workers[s].stream, st->fork_event, 0),
               "xc fxc_ref wait");
  }

  for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
    const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
    XcStreamWorker& ws = st->workers[s];
    const BatchGpuMeta& b = st->batches[ib];
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const double* d_phi = nullptr;
    if (st->ao_cached) {
      d_phi = st->phi_cache.data() + b.phi_cache_offset;
    } else {
      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, ws.phi_scratch.data());
      d_phi = ws.phi_scratch.data();
    }

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        d_dm, nao, d_batch_aos, na, ws.d_act.data());

    // Always use FP64 for reference ground-state density & Hessian cache
    if (st->family == 2) {
      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), d_phi,
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           4, Precision::FP64);
    } else {
      ws.gemm_nn(i_na, i_ng, i_na, ws.d_act.data(), d_phi, ws.t_mat.data(),
                 false, Precision::FP64);
    }

    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    cache_fxc_reference_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->func_id, st->family, st->opts.rho_cutoff,
        na, ng, st->npoints, b.point_offset,
        d_phi, ws.t_mat.data(), st->fxc_cache.data());
  }

  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
               "xc fxc_ref join record");
    check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
               "xc fxc_ref join wait");
  }
  check_cuda(cudaStreamSynchronize(caller_stream), "xc_cuda_set_reference_density_device sync");
  st->fxc_ready = true;
}

void xc_cuda_set_reference_density_host(void* raw_impl, const double* h_dm) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao2 = static_cast<std::size_t>(st->basis.nao) * st->basis.nao;
  if (nao2 > 0) {
    st->dm_host_buf.upload(h_dm, nao2);
  }
  xc_cuda_set_reference_density_device(raw_impl, st->dm_host_buf.data(), nullptr);
}

bool xc_cuda_has_reference_density(const void* raw_impl) noexcept {
  const auto* st = static_cast<const CudaXcEngineImpl*>(raw_impl);
  return st != nullptr && st->fxc_ready;
}

void xc_cuda_eval_fxc_device(void* raw_impl,
                             const double* d_delta_dm,
                             double* d_fxc_out,
                             void* cuda_stream) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  if (!st->fxc_ready) {
    throw std::logic_error(
        "xc_cuda_eval_fxc_device: set_reference_density_device must be called first");
  }

  cudaStream_t caller_stream = static_cast<cudaStream_t>(cuda_stream);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const std::size_t nao2 = nao * nao;
  if (nao == 0) return;

  if (st->batches.empty() || st->npoints == 0) {
    check_cuda(cudaMemsetAsync(d_fxc_out, 0, nao2 * sizeof(double), caller_stream),
               "xc zero d_fxc_out empty");
    check_cuda(cudaStreamSynchronize(caller_stream), "xc_cuda_eval_fxc_device empty sync");
    return;
  }

  const int n_streams = st->num_workers;
  const int deriv = (st->family == 0) ? 0 : 1;
  const FlatBasisView bview = st->basis.view();

  check_cuda(cudaEventRecord(st->fork_event, caller_stream), "xc fxc fork");
  for (int s = 0; s < n_streams; ++s) {
    XcStreamWorker& ws = st->workers[s];
    check_cuda(cudaStreamWaitEvent(ws.stream, st->fork_event, 0), "xc fxc wait");
    double* target_fxc = (n_streams == 1) ? d_fxc_out : ws.vxc_part.data();
    check_cuda(cudaMemsetAsync(target_fxc, 0, nao2 * sizeof(double), ws.stream),
               "xc zero target_fxc");
  }

  for (std::size_t ib = 0; ib < st->batches.size(); ++ib) {
    const int s = static_cast<int>(ib % static_cast<std::size_t>(n_streams));
    XcStreamWorker& ws = st->workers[s];
    const BatchGpuMeta& b = st->batches[ib];
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const double* d_phi = nullptr;
    if (st->ao_cached) {
      d_phi = st->phi_cache.data() + b.phi_cache_offset;
    } else {
      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, ws.phi_scratch.data());
      d_phi = ws.phi_scratch.data();
    }

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        d_delta_dm, nao, d_batch_aos, na, ws.d_act.data());

    if (st->family == 2) {
      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), d_phi,
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           4, st->opts.precision);
    } else {
      ws.gemm_nn(i_na, i_ng, i_na, ws.d_act.data(), d_phi, ws.t_mat.data(),
                 false, st->opts.precision);
    }

    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    eval_fxc_potentials_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, st->weights.data(), b.point_offset,
        d_phi, ws.t_mat.data(), st->fxc_cache.data(), ws.wv.data());

    const dim3 w_grid(static_cast<unsigned>((ng + 127) / 128),
                      static_cast<unsigned>(na), 1);
    build_vxc_weight_matrix_kernel<<<w_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, ng, d_phi, ws.wv.data(), ws.w_mat.data());

    ws.gemm_nt(i_na, i_na, i_ng, d_phi, ws.w_mat.data(), ws.v_half.data(),
               false, st->opts.precision);

    if (st->family == 2) {
      const double* d_wv4 = ws.wv.data() + 4 * ng;
      for (std::size_t dir = 1; dir <= 3; ++dir) {
        const double* phib = d_phi + dir * comp_stride;
        scale_mgga_tau_2d_kernel<<<w_grid, 128, 0, ws.stream>>>(
            na, ng, phib, d_wv4, ws.w_mat.data());
        ws.gemm_nt(i_na, i_na, i_ng, phib, ws.w_mat.data(), ws.v_half.data(),
                   true, st->opts.precision);
      }
    }

    double* target_fxc = (n_streams == 1) ? d_fxc_out : ws.vxc_part.data();
    scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        ws.v_half.data(), d_batch_aos, na, nao, target_fxc);
  }

  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
               "xc fxc join record");
    check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
               "xc fxc join wait");
  }

  if (n_streams >= 2) {
    reduce_stream_vxc_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        nao2, n_streams,
        st->workers[0].vxc_part.data(),
        st->workers[1].vxc_part.data(),
        (n_streams >= 3) ? st->workers[2].vxc_part.data() : nullptr,
        (n_streams >= 4) ? st->workers[3].vxc_part.data() : nullptr,
        d_fxc_out);
  }
  check_cuda(cudaStreamSynchronize(caller_stream), "xc_cuda_eval_fxc_device sync");
}

void xc_cuda_eval_fxc_host(void* raw_impl,
                           const double* h_delta_dm,
                           double* h_fxc_out) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao2 = static_cast<std::size_t>(st->basis.nao) * st->basis.nao;
  st->dm_host_buf.upload(h_delta_dm, nao2);
  xc_cuda_eval_fxc_device(raw_impl, st->dm_host_buf.data(), st->vxc_host_buf.data(), nullptr);
  st->vxc_host_buf.download(h_fxc_out, nao2);
  check_cuda(cudaDeviceSynchronize(), "xc_cuda_eval_fxc_host download sync");
}

std::vector<double> xc_cuda_nuclear_grad(void* raw_impl,
                                         const MolecularGrid& grid,
                                         std::span<const XcAtom> atoms,
                                         const int* ao_to_atom,
                                         const double* h_dm) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const int natom = static_cast<int>(atoms.size());
  std::vector<double> h_grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (st->batches.empty() || nao == 0 || natom == 0 || st->is_hf) return h_grad;

  XcStreamWorker& ws = st->workers[0];
  const int deriv = (st->family == 0) ? 1 : 2;
  const std::size_t ncomp_grad = (deriv == 1) ? 4 : 10;
  const std::size_t n_fcomp = (st->family >= 1) ? 4 : 1;
  const bool do_grid_response = st->opts.grid_response && st->has_parent_atom;
  const bool use_libxc_path = st->opts.use_libxc || (st->func_id == 0);

  st->grad_phi_scratch.resize(ncomp_grad * st->max_na * st->max_ng);
  ws.t_mat.resize(n_fcomp * st->max_na * st->max_ng);

  st->dm_host_buf.upload(h_dm, nao * nao, ws.stream);
  DevBuf<int> d_ao_to_atom;
  d_ao_to_atom.upload(ao_to_atom, nao, ws.stream);
  DevBuf<double> d_grad(static_cast<std::size_t>(natom) * 3);
  check_cuda(cudaMemsetAsync(d_grad.data(), 0,
                             static_cast<std::size_t>(natom) * 3 * sizeof(double), ws.stream),
             "xc grad memset");

  DevBuf<double> d_e_xc_point(do_grid_response ? st->npoints : 0);
  PinnedBuf<double> h_e_xc_pt(do_grid_response ? st->npoints : 0);
  const FlatBasisView bview = st->basis.view();

  if (use_libxc_path) {
    const std::size_t planes = st->rks_dens_planes();
    const std::size_t n_flat = planes * st->npoints;
    st->d_dens_in.resize(n_flat);
    st->d_wv_all.resize(n_flat);
    st->h_dens_in.resize(n_flat);
    st->h_wv_all.resize(n_flat);

    // Pass 1: Evaluate rho, grad_rho, tau into flat d_dens_in
    for (const BatchGpuMeta& b : st->batches) {
      const std::size_t na = b.na;
      const std::size_t ng = b.ng;
      const std::size_t comp_stride = na * ng;
      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, st->grad_phi_scratch.data());

      gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
          st->dm_host_buf.data(), nao, d_batch_aos, na, ws.d_act.data());

      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), st->grad_phi_scratch.data(),
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           static_cast<int>(n_fcomp), Precision::FP64);

      const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
      const dim3 pot_threads(64, 4, 1);
      eval_density_flat_rks_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
          st->family, na, ng, st->npoints, b.point_offset,
          st->grad_phi_scratch.data(), ws.t_mat.data(), st->d_dens_in.data());
    }

    check_cuda(cudaMemcpyAsync(st->h_dens_in.data(), st->d_dens_in.data(),
                               n_flat * sizeof(double), cudaMemcpyDeviceToHost, ws.stream),
               "xc grad libxc D2H");
    check_cuda(cudaStreamSynchronize(ws.stream), "xc grad libxc D2H sync");

    double dummy_exc = 0.0, dummy_nelec = 0.0;
    eval_libxc_rks_grid(st->libxc_state, st->family, st->opts.rho_cutoff,
                        st->npoints, st->h_weights.data(), st->h_dens_in.data(),
                        /*for_grad=*/true, st->h_wv_all.data(),
                        do_grid_response ? h_e_xc_pt.data() : nullptr,
                        &dummy_exc, &dummy_nelec);

    check_cuda(cudaMemcpyAsync(st->d_wv_all.data(), st->h_wv_all.data(),
                               n_flat * sizeof(double), cudaMemcpyHostToDevice, ws.stream),
               "xc grad libxc H2D");
    if (do_grid_response) {
      d_e_xc_point.upload(h_e_xc_pt.data(), st->npoints, ws.stream);
    }

    // Pass 2: Accumulate Term 1 (AO deriv) and Term 2 (moving grid)
    for (const BatchGpuMeta& b : st->batches) {
      const std::size_t na = b.na;
      const std::size_t ng = b.ng;
      const std::size_t comp_stride = na * ng;
      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, st->grad_phi_scratch.data());

      gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
          st->dm_host_buf.data(), nao, d_batch_aos, na, ws.d_act.data());

      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), st->grad_phi_scratch.data(),
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           static_cast<int>(n_fcomp), Precision::FP64);

      const double* d_wv_batch = st->d_wv_all.data() + b.point_offset;
      const dim3 red_grid(static_cast<unsigned>((ng + 127) / 128),
                          static_cast<unsigned>(na), 1);
      grad_xc_accum_atom_kernel<<<red_grid, 128, 0, ws.stream>>>(
          st->family, na, ng, st->npoints, d_batch_aos, d_ao_to_atom.data(),
          st->grad_phi_scratch.data(), ws.t_mat.data(), d_wv_batch, d_grad.data());

      if (do_grid_response) {
        const unsigned move_blocks = static_cast<unsigned>((ng + 127) / 128);
        grad_xc_move_grid_kernel<<<move_blocks, 128, 0, ws.stream>>>(
            st->family, na, ng, st->npoints, b.point_offset, st->parent_atom.data(),
            st->grad_phi_scratch.data(), ws.t_mat.data(), d_wv_batch, d_grad.data());
      }
    }
  } else {
    for (const BatchGpuMeta& b : st->batches) {
      const std::size_t na = b.na;
      const std::size_t ng = b.ng;
      const std::size_t comp_stride = na * ng;
      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                         static_cast<unsigned>(b.n_act_shells), 1);
      eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
          bview, st->coords.data(), b.point_offset, ng, na,
          st->act_shells.data() + b.shell_list_offset,
          st->act_shell_local_ao.data() + b.shell_list_offset,
          b.n_act_shells, deriv, st->grad_phi_scratch.data());

      gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
          st->dm_host_buf.data(), nao, d_batch_aos, na, ws.d_act.data());

      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), st->grad_phi_scratch.data(),
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           static_cast<int>(n_fcomp), Precision::FP64);

      const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
      const dim3 pot_threads(64, 4, 1);
      grad_xc_point_weights_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
          st->func_id, st->family, st->opts.rho_cutoff,
          na, ng, st->weights.data(), b.point_offset,
          st->grad_phi_scratch.data(), ws.t_mat.data(), ws.wv.data(),
          do_grid_response ? d_e_xc_point.data() : nullptr);

      const dim3 red_grid(static_cast<unsigned>((ng + 127) / 128),
                          static_cast<unsigned>(na), 1);
      grad_xc_accum_atom_kernel<<<red_grid, 128, 0, ws.stream>>>(
          st->family, na, ng, ng, d_batch_aos, d_ao_to_atom.data(),
          st->grad_phi_scratch.data(), ws.t_mat.data(), ws.wv.data(), d_grad.data());

      if (do_grid_response) {
        const unsigned move_blocks = static_cast<unsigned>((ng + 127) / 128);
        grad_xc_move_grid_kernel<<<move_blocks, 128, 0, ws.stream>>>(
            st->family, na, ng, ng, b.point_offset, st->parent_atom.data(),
            st->grad_phi_scratch.data(), ws.t_mat.data(), ws.wv.data(), d_grad.data());
      }
    }
  }

  if (do_grid_response) {
    launch_partition_weight_grad_cuda(st, grid, atoms, d_e_xc_point.data(), d_grad.data(),
                                      ws.stream);
  }

  d_grad.download(h_grad.data(), h_grad.size(), ws.stream);
  check_cuda(cudaStreamSynchronize(ws.stream), "xc_cuda_nuclear_grad sync");
  return h_grad;
}

std::vector<double> xc_cuda_nuclear_grad_uks(void* raw_impl,
                                             const MolecularGrid& grid,
                                             std::span<const XcAtom> atoms,
                                             const int* ao_to_atom,
                                             const double* h_dm_alpha,
                                             const double* h_dm_beta) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const int natom = static_cast<int>(atoms.size());
  std::vector<double> h_grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (st->batches.empty() || nao == 0 || natom == 0 || st->is_hf) return h_grad;

  st->ensure_uks_buffers();

  XcStreamWorker& ws = st->workers[0];
  const int deriv = (st->family == 0) ? 1 : 2;
  const std::size_t ncomp_grad = (deriv == 1) ? 4 : 10;
  const std::size_t n_fcomp = (st->family >= 1) ? 4 : 1;
  const bool do_grid_response = st->opts.grid_response && st->has_parent_atom;
  const std::size_t n_wv_comp = st->rks_dens_planes();
  const std::size_t n_flat = 2 * n_wv_comp * st->npoints;

  st->grad_phi_scratch.resize(ncomp_grad * st->max_na * st->max_ng);
  ws.t_mat.resize(n_fcomp * st->max_na * st->max_ng);
  ws.t_mat_beta.resize(n_fcomp * st->max_na * st->max_ng);

  st->dm_host_buf.upload(h_dm_alpha, nao * nao, ws.stream);
  st->dm_beta_host_buf.upload(h_dm_beta, nao * nao, ws.stream);
  DevBuf<int> d_ao_to_atom;
  d_ao_to_atom.upload(ao_to_atom, nao, ws.stream);
  DevBuf<double> d_grad(static_cast<std::size_t>(natom) * 3);
  check_cuda(cudaMemsetAsync(d_grad.data(), 0,
                             static_cast<std::size_t>(natom) * 3 * sizeof(double), ws.stream),
             "xc uks grad memset");

  DevBuf<double> d_e_xc_point(do_grid_response ? st->npoints : 0);
  PinnedBuf<double> h_e_xc_pt(do_grid_response ? st->npoints : 0);
  const FlatBasisView bview = st->basis.view();

  // Pass 1: Evaluate alpha and beta densities into flat d_dens_in
  for (const BatchGpuMeta& b : st->batches) {
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                       static_cast<unsigned>(b.n_act_shells), 1);
    eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
        bview, st->coords.data(), b.point_offset, ng, na,
        st->act_shells.data() + b.shell_list_offset,
        st->act_shell_local_ao.data() + b.shell_list_offset,
        b.n_act_shells, deriv, st->grad_phi_scratch.data());

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        st->dm_host_buf.data(), nao, d_batch_aos, na, ws.d_act.data());
    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        st->dm_beta_host_buf.data(), nao, d_batch_aos, na, ws.d_act_beta.data());

    ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), st->grad_phi_scratch.data(),
                         static_cast<long long>(comp_stride),
                         ws.t_mat.data(), static_cast<long long>(comp_stride),
                         static_cast<int>(n_fcomp), Precision::FP64);
    ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act_beta.data(), st->grad_phi_scratch.data(),
                         static_cast<long long>(comp_stride),
                         ws.t_mat_beta.data(), static_cast<long long>(comp_stride),
                         static_cast<int>(n_fcomp), Precision::FP64);

    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    eval_density_flat_uks_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, b.point_offset,
        st->grad_phi_scratch.data(), ws.t_mat.data(), ws.t_mat_beta.data(), st->d_dens_in.data());
  }

  check_cuda(cudaMemcpyAsync(st->h_dens_in.data(), st->d_dens_in.data(),
                             n_flat * sizeof(double), cudaMemcpyDeviceToHost, ws.stream),
             "xc uks grad D2H");
  check_cuda(cudaStreamSynchronize(ws.stream), "xc uks grad D2H sync");

  double dummy_exc = 0.0, dummy_na = 0.0, dummy_nb = 0.0;
  eval_libxc_uks_grid(st->libxc_state, st->family, st->opts.rho_cutoff,
                      st->npoints, st->h_weights.data(), st->h_dens_in.data(),
                      /*for_grad=*/true, st->h_wv_all.data(),
                      do_grid_response ? h_e_xc_pt.data() : nullptr,
                      &dummy_exc, &dummy_na, &dummy_nb);

  check_cuda(cudaMemcpyAsync(st->d_wv_all.data(), st->h_wv_all.data(),
                             n_flat * sizeof(double), cudaMemcpyHostToDevice, ws.stream),
             "xc uks grad H2D");
  if (do_grid_response) {
    d_e_xc_point.upload(h_e_xc_pt.data(), st->npoints, ws.stream);
  }

  // Pass 2: Accumulate alpha & beta Term 1 + Term 2
  for (const BatchGpuMeta& b : st->batches) {
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                       static_cast<unsigned>(b.n_act_shells), 1);
    eval_active_ao_batch_kernel<<<ao_grid, 128, 0, ws.stream>>>(
        bview, st->coords.data(), b.point_offset, ng, na,
        st->act_shells.data() + b.shell_list_offset,
        st->act_shell_local_ao.data() + b.shell_list_offset,
        b.n_act_shells, deriv, st->grad_phi_scratch.data());

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        st->dm_host_buf.data(), nao, d_batch_aos, na, ws.d_act.data());
    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        st->dm_beta_host_buf.data(), nao, d_batch_aos, na, ws.d_act_beta.data());

    ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), st->grad_phi_scratch.data(),
                         static_cast<long long>(comp_stride),
                         ws.t_mat.data(), static_cast<long long>(comp_stride),
                         static_cast<int>(n_fcomp), Precision::FP64);
    ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act_beta.data(), st->grad_phi_scratch.data(),
                         static_cast<long long>(comp_stride),
                         ws.t_mat_beta.data(), static_cast<long long>(comp_stride),
                         static_cast<int>(n_fcomp), Precision::FP64);

    const double* d_wv_a = st->d_wv_all.data() + b.point_offset;
    const double* d_wv_b = st->d_wv_all.data() + n_wv_comp * st->npoints + b.point_offset;
    const dim3 red_grid(static_cast<unsigned>((ng + 127) / 128),
                        static_cast<unsigned>(na), 1);

    grad_xc_accum_atom_kernel<<<red_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, d_batch_aos, d_ao_to_atom.data(),
        st->grad_phi_scratch.data(), ws.t_mat.data(), d_wv_a, d_grad.data());
    grad_xc_accum_atom_kernel<<<red_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, st->npoints, d_batch_aos, d_ao_to_atom.data(),
        st->grad_phi_scratch.data(), ws.t_mat_beta.data(), d_wv_b, d_grad.data());

    if (do_grid_response) {
      const unsigned move_blocks = static_cast<unsigned>((ng + 127) / 128);
      grad_xc_move_grid_kernel<<<move_blocks, 128, 0, ws.stream>>>(
          st->family, na, ng, st->npoints, b.point_offset, st->parent_atom.data(),
          st->grad_phi_scratch.data(), ws.t_mat.data(), d_wv_a, d_grad.data());
      grad_xc_move_grid_kernel<<<move_blocks, 128, 0, ws.stream>>>(
          st->family, na, ng, st->npoints, b.point_offset, st->parent_atom.data(),
          st->grad_phi_scratch.data(), ws.t_mat_beta.data(), d_wv_b, d_grad.data());
    }
  }

  if (do_grid_response) {
    launch_partition_weight_grad_cuda(st, grid, atoms, d_e_xc_point.data(), d_grad.data(),
                                      ws.stream);
  }

  d_grad.download(h_grad.data(), h_grad.size(), ws.stream);
  check_cuda(cudaStreamSynchronize(ws.stream), "xc_cuda_nuclear_grad_uks sync");
  return h_grad;
}

}  // namespace gpu4dft::detail
