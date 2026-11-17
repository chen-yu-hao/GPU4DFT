#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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

__device__ inline double dev_step_ssf(double nu) {
  constexpr double kA = 0.64;
  if (nu <= -kA) return 1.0;
  if (nu >= kA) return 0.0;
  const double z = nu / kA;
  const double z2 = z * z;
  const double g = (1.0 / 16.0) * z * (35.0 + z2 * (-35.0 + z2 * (21.0 - 5.0 * z2)));
  return 0.5 * (1.0 - g);
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
  constexpr int kMaxSharedAtoms = 64;
  __shared__ double s_atom_xyz[3 * kMaxSharedAtoms];
  if (natom <= kMaxSharedAtoms) {
    for (int i = threadIdx.x; i < 3 * natom; i += blockDim.x) {
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
// 3. Spatially Batched Active-AO + cuBLAS GEMM V_xc & Nuclear Gradient Kernels
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
  // Each (u, v) in a single spatial batch maps to a UNIQUE (gu, gv) because act_aos is strictly sorted!
  d_vxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
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

__global__ void density_xc_and_weight_kernel(int func_id,
                                             int family,
                                             double rho_cutoff,
                                             std::size_t na,
                                             std::size_t ng,
                                             const double* __restrict__ weights,
                                             std::size_t point_offset,
                                             const double* __restrict__ phi,
                                             const double* __restrict__ t_mat,
                                             double* __restrict__ w_mat,
                                             double* __restrict__ vtau_w,
                                             double* __restrict__ scalars_out) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  double pt_exc = 0.0;
  double pt_nelec = 0.0;

  if (g < ng) {
    const std::size_t comp_stride = na * ng;
    const double* phi0 = phi;
    const double* phix = (family >= 1) ? (phi + 1 * comp_stride) : nullptr;
    const double* phiy = (family >= 1) ? (phi + 2 * comp_stride) : nullptr;
    const double* phiz = (family >= 1) ? (phi + 3 * comp_stride) : nullptr;
    const double* t0 = t_mat;

    double rho = 0.0;
    double gx = 0.0, gy = 0.0, gz = 0.0;
    double tau = 0.0;

    for (std::size_t u = 0; u < na; ++u) {
      const std::size_t off = u * ng + g;
      const double t0_ug = t0[off];
      rho += phi0[off] * t0_ug;
      if (family >= 1) {
        gx += 2.0 * phix[off] * t0_ug;
        gy += 2.0 * phiy[off] * t0_ug;
        gz += 2.0 * phiz[off] * t0_ug;
      }
      if (family == 2) {
        tau += 0.5 * (phix[off] * t_mat[1 * comp_stride + off] +
                      phiy[off] * t_mat[2 * comp_stride + off] +
                      phiz[off] * t_mat[3 * comp_stride + off]);
      }
    }

    const double wg = weights[point_offset + g];
    if (rho > 0.0) pt_nelec = wg * rho;

    const double sigma = (family >= 1) ? (gx * gx + gy * gy + gz * gz) : 0.0;
    xc_gen::XcPointOut pt;
    xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(func_id),
                          rho, sigma, tau, 1, pt, rho_cutoff);
    pt_exc = wg * rho * pt.zk;

    const double wv0 = 0.5 * wg * pt.vrho;
    const double wvx = (family >= 1) ? (2.0 * wg * pt.vsigma * gx) : 0.0;
    const double wvy = (family >= 1) ? (2.0 * wg * pt.vsigma * gy) : 0.0;
    const double wvz = (family >= 1) ? (2.0 * wg * pt.vsigma * gz) : 0.0;
    if (family == 2) {
      vtau_w[g] = 0.5 * wg * pt.vtau;
    }

    for (std::size_t u = 0; u < na; ++u) {
      const std::size_t off = u * ng + g;
      double w_ug = wv0 * phi0[off];
      if (family >= 1) {
        w_ug += wvx * phix[off] + wvy * phiy[off] + wvz * phiz[off];
      }
      w_mat[off] = w_ug;
    }
  }

  // Warp-level reduction before atomicAdd to minimize global atomic contention
  for (int mask = 16; mask > 0; mask >>= 1) {
    pt_exc += __shfl_down_sync(0xffffffffu, pt_exc, mask);
    pt_nelec += __shfl_down_sync(0xffffffffu, pt_nelec, mask);
  }
  if ((threadIdx.x & 31) == 0) {
    atomicAdd(&scalars_out[0], pt_exc);
    atomicAdd(&scalars_out[1], pt_nelec);
  }
}

__global__ void scale_mgga_tau_weight_kernel(std::size_t na,
                                             std::size_t ng,
                                             const double* __restrict__ phib,
                                             const double* __restrict__ vtau_w,
                                             double* __restrict__ w_mat) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * ng) return;
  const std::size_t g = idx % ng;
  w_mat[idx] = 0.5 * vtau_w[g] * phib[idx];
}

__global__ void grad_xc_point_weights_kernel(int func_id,
                                             int family,
                                             double rho_cutoff,
                                             std::size_t na,
                                             std::size_t ng,
                                             const double* __restrict__ weights,
                                             std::size_t point_offset,
                                             const double* __restrict__ phi,
                                             const double* __restrict__ f_mat,
                                             double* __restrict__ wv_out) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (g >= ng) return;

  const std::size_t comp_stride = na * ng;
  const double* phi0 = phi;
  const double* phix = phi + 1 * comp_stride;
  const double* phiy = phi + 2 * comp_stride;
  const double* phiz = phi + 3 * comp_stride;
  const double* f0 = f_mat;

  double rho = 0.0, gx = 0.0, gy = 0.0, gz = 0.0, tau = 0.0;
  for (std::size_t u = 0; u < na; ++u) {
    const std::size_t off = u * ng + g;
    const double f0_ug = f0[off];
    rho += phi0[off] * f0_ug;
    if (family >= 1) {
      gx += 2.0 * phix[off] * f0_ug;
      gy += 2.0 * phiy[off] * f0_ug;
      gz += 2.0 * phiz[off] * f0_ug;
    }
    if (family == 2) {
      tau += 0.5 * (phix[off] * f_mat[1 * comp_stride + off] +
                    phiy[off] * f_mat[2 * comp_stride + off] +
                    phiz[off] * f_mat[3 * comp_stride + off]);
    }
  }

  const double wg = weights[point_offset + g];
  const double sigma = (family >= 1) ? (gx * gx + gy * gy + gz * gz) : 0.0;
  xc_gen::XcPointOut pt;
  xc_gen::eval_xc_point(static_cast<xc_gen::XcFunctionalId>(func_id),
                        rho, sigma, tau, 1, pt, rho_cutoff);

  wv_out[0 * ng + g] = wg * pt.vrho;
  wv_out[1 * ng + g] = (family >= 1) ? (2.0 * wg * pt.vsigma * gx) : 0.0;
  wv_out[2 * ng + g] = (family >= 1) ? (2.0 * wg * pt.vsigma * gy) : 0.0;
  wv_out[3 * ng + g] = (family >= 1) ? (2.0 * wg * pt.vsigma * gz) : 0.0;
  wv_out[4 * ng + g] = (family == 2) ? (wg * pt.vtau) : 0.0;
}

__global__ void grad_xc_accum_atom_kernel(int family,
                                          std::size_t na,
                                          std::size_t ng,
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
    const double wv0 = wv[0 * ng + g];
    double w0_ug = wv0 * f0_ug;

    if (family >= 1) {
      const double wvx = wv[1 * ng + g];
      const double wvy = wv[2 * ng + g];
      const double wvz = wv[3 * ng + g];
      const double wvt = wv[4 * ng + g];

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

struct BatchGpuMeta {
  std::size_t point_offset = 0;
  std::size_t ng = 0;
  std::size_t na = 0;
  int n_act_shells = 0;
  std::size_t shell_list_offset = 0;
  std::size_t ao_list_offset = 0;
};

struct CudaXcEngineImpl {
  DevFlatBasis basis;
  int func_id = 0;
  int family = 0;
  XcOptions opts;
  std::size_t npoints = 0;

  DevBuf<double> coords;
  DevBuf<double> weights;
  std::vector<BatchGpuMeta> batches;
  DevBuf<int> act_shells;
  DevBuf<int> act_shell_local_ao;
  DevBuf<int> act_aos;

  // Persistent GPU scratch workspaces (zero cudaMalloc during SCF iterations!)
  std::size_t max_na = 0;
  std::size_t max_ng = 0;
  DevBuf<double> phi;
  DevBuf<double> d_act;
  DevBuf<double> t_mat;
  DevBuf<double> w_mat;
  DevBuf<double> v_half;
  DevBuf<double> vtau_w;
  DevBuf<double> wv_grad;
  DevBuf<double> scalars;
  DevBuf<double> dm_host_buf;
  DevBuf<double> vxc_host_buf;

  // FP32 Tensor Core GEMM scratch buffers
  DevBuf<float> f32_a;
  DevBuf<float> f32_b;
  DevBuf<float> f32_c;

  cublasHandle_t cublas = nullptr;

  CudaXcEngineImpl(const FlatBasisData& flat_b,
                   const MolecularGrid& grid,
                   int fid,
                   int fam,
                   const XcOptions& options)
      : basis(flat_b),
        func_id(fid),
        family(fam),
        opts(options),
        npoints(grid.npoints()),
        coords(grid.coords()),
        weights(grid.weights()) {
    check_cublas(cublasCreate(&cublas), "cublasCreate (XcEngine)");

    std::vector<int> h_act_shells;
    std::vector<int> h_act_shell_local_ao;
    std::vector<int> h_act_aos;

    batches.reserve(grid.batches().size());
    for (const SpatialBatch& sb : grid.batches()) {
      if (sb.npoints == 0 || sb.active_aos.empty()) continue;
      BatchGpuMeta m;
      m.point_offset = sb.point_offset;
      m.ng = sb.npoints;
      m.na = sb.active_aos.size();
      m.n_act_shells = static_cast<int>(sb.active_shells.size());
      m.shell_list_offset = h_act_shells.size();
      m.ao_list_offset = h_act_aos.size();

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

    if (max_na > 0 && max_ng > 0) {
      // Allocate 10 components up front so both V_xc and nuclear_grad run allocation-free
      phi.resize(10 * max_na * max_ng);
      d_act.resize(max_na * max_na);
      t_mat.resize(4 * max_na * max_ng);
      w_mat.resize(max_na * max_ng);
      v_half.resize(max_na * max_na);
      vtau_w.resize(max_ng);
      wv_grad.resize(5 * max_ng);
      scalars.resize(2);

      if (opts.precision == Precision::FP32) {
        const std::size_t max_dim = std::max(max_na * max_na, max_na * max_ng);
        f32_a.resize(max_dim);
        f32_b.resize(max_dim);
        f32_c.resize(max_dim);
      }
    }

    const std::size_t nao2 = static_cast<std::size_t>(flat_b.nao) * flat_b.nao;
    if (nao2 > 0) {
      dm_host_buf.resize(nao2);
      vxc_host_buf.resize(nao2);
    }
  }

  ~CudaXcEngineImpl() {
    if (cublas != nullptr) cublasDestroy(cublas);
  }

  // Row-major C[M x N] = A[M x K] * B[K x N]
  void gemm_nn(int M, int N, int K, const double* dA, const double* dB, double* dC,
               bool accum, cudaStream_t stream) {
    if (opts.precision == Precision::FP32) {
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

  // Row-major C[M x N] = A[M x K] * B[N x K]^T
  void gemm_nt(int M, int N, int K, const double* dA, const double* dB, double* dC,
               bool accum, cudaStream_t stream) {
    if (opts.precision == Precision::FP32) {
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
                            const XcOptions& opts) {
  return new CudaXcEngineImpl(flat_basis, grid, func_id, family, opts);
}

void xc_cuda_engine_destroy(void* impl) noexcept {
  delete static_cast<CudaXcEngineImpl*>(impl);
}

double xc_cuda_eval_vxc_device(void* raw_impl,
                               const double* d_dm,
                               double* d_vxc,
                               double* nelec_out,
                               void* cuda_stream) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  check_cublas(cublasSetStream(st->cublas, stream), "cublasSetStream");

  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  check_cuda(cudaMemsetAsync(d_vxc, 0, nao * nao * sizeof(double), stream),
             "xc zero d_vxc");
  if (st->batches.empty() || nao == 0) {
    if (nelec_out) *nelec_out = 0.0;
    return 0.0;
  }

  check_cuda(cudaMemsetAsync(st->scalars.data(), 0, 2 * sizeof(double), stream),
             "xc zero scalars");

  const int deriv = (st->family == 0) ? 0 : 1;
  const FlatBasisView bview = st->basis.view();

  for (const BatchGpuMeta& b : st->batches) {
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    // 1. Evaluate active shells on GPU for this spatial block
    const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                       static_cast<unsigned>(b.n_act_shells), 1);
    eval_active_ao_batch_kernel<<<ao_grid, 128, 0, stream>>>(
        bview, st->coords.data(), b.point_offset, ng, na,
        st->act_shells.data() + b.shell_list_offset,
        st->act_shell_local_ao.data() + b.shell_list_offset,
        b.n_act_shells, deriv, st->phi.data());

    // 2. Gather symmetric active density block D_act [na x na]
    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, stream>>>(
        d_dm, nao, d_batch_aos, na, st->d_act.data());

    // 3. cuBLAS GEMM: T0 = D_act * Phi0 (plus Tx, Ty, Tz for MGGA)
    st->gemm_nn(i_na, i_ng, i_na, st->d_act.data(), st->phi.data(), st->t_mat.data(),
                false, stream);
    if (st->family == 2) {
      for (std::size_t dir = 1; dir <= 3; ++dir) {
        st->gemm_nn(i_na, i_ng, i_na, st->d_act.data(),
                    st->phi.data() + dir * comp_stride,
                    st->t_mat.data() + dir * comp_stride, false, stream);
      }
    }

    // 4. Density (rho, sigma, tau) + Generated Symbolic XC Kernel + Half-Weighted W [na x ng]
    density_xc_and_weight_kernel<<<static_cast<unsigned>((ng + 127) / 128), 128, 0, stream>>>(
        st->func_id, st->family, st->opts.rho_cutoff,
        na, ng, st->weights.data(), b.point_offset,
        st->phi.data(), st->t_mat.data(), st->w_mat.data(),
        st->vtau_w.data(), st->scalars.data());

    // 5. cuBLAS Symmetric GEMM: V_half = Phi0 * W^T (+ MGGA tau GEMMs)
    st->gemm_nt(i_na, i_na, i_ng, st->phi.data(), st->w_mat.data(), st->v_half.data(),
                false, stream);

    if (st->family == 2) {
      for (std::size_t dir = 1; dir <= 3; ++dir) {
        const double* phib = st->phi.data() + dir * comp_stride;
        scale_mgga_tau_weight_kernel<<<static_cast<unsigned>((comp_stride + 255) / 256), 256, 0, stream>>>(
            na, ng, phib, st->vtau_w.data(), st->w_mat.data());
        st->gemm_nt(i_na, i_na, i_ng, phib, st->w_mat.data(), st->v_half.data(),
                    true, stream);
      }
    }

    // 6. Symmetrize & scatter V_act = V_half + V_half^T into [nao x nao] d_vxc
    scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, stream>>>(
        st->v_half.data(), d_batch_aos, na, nao, d_vxc);
  }

  double h_scalars[2] = {0.0, 0.0};
  st->scalars.download(h_scalars, 2, stream);
  check_cuda(cudaStreamSynchronize(stream), "xc_cuda_eval_vxc_device sync");
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

std::vector<double> xc_cuda_nuclear_grad(void* raw_impl,
                                         const int* ao_to_atom,
                                         int natom,
                                         const double* h_dm) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  std::vector<double> h_grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (st->batches.empty() || nao == 0 || natom == 0) return h_grad;

  check_cublas(cublasSetStream(st->cublas, nullptr), "cublasSetStream grad");
  st->dm_host_buf.upload(h_dm, nao * nao);
  DevBuf<int> d_ao_to_atom(ao_to_atom, nao);
  DevBuf<double> d_grad(static_cast<std::size_t>(natom) * 3);
  check_cuda(cudaMemset(d_grad.data(), 0, static_cast<std::size_t>(natom) * 3 * sizeof(double)),
             "xc grad memset");

  const int deriv = (st->family == 0) ? 1 : 2;
  const std::size_t n_fcomp = (st->family >= 1) ? 4 : 1;
  const FlatBasisView bview = st->basis.view();
  const double one = 1.0, zero = 0.0;

  for (const BatchGpuMeta& b : st->batches) {
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const std::size_t comp_stride = na * ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                       static_cast<unsigned>(b.n_act_shells), 1);
    eval_active_ao_batch_kernel<<<ao_grid, 128>>>(
        bview, st->coords.data(), b.point_offset, ng, na,
        st->act_shells.data() + b.shell_list_offset,
        st->act_shell_local_ao.data() + b.shell_list_offset,
        b.n_act_shells, deriv, st->phi.data());

    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256>>>(
        st->dm_host_buf.data(), nao, d_batch_aos, na, st->d_act.data());

    // Always use FP64 cublasDgemm for analytical nuclear gradients to hit < 1e-12 accuracy
    for (std::size_t dir = 0; dir < n_fcomp; ++dir) {
      check_cublas(cublasDgemm(st->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               i_ng, i_na, i_na, &one,
                               st->phi.data() + dir * comp_stride, i_ng,
                               st->d_act.data(), i_na,
                               &zero, st->t_mat.data() + dir * comp_stride, i_ng),
                   "cublasDgemm grad F");
    }

    grad_xc_point_weights_kernel<<<static_cast<unsigned>((ng + 127) / 128), 128>>>(
        st->func_id, st->family, st->opts.rho_cutoff,
        na, ng, st->weights.data(), b.point_offset,
        st->phi.data(), st->t_mat.data(), st->wv_grad.data());

    const dim3 red_grid(static_cast<unsigned>((ng + 127) / 128),
                        static_cast<unsigned>(na), 1);
    grad_xc_accum_atom_kernel<<<red_grid, 128>>>(
        st->family, na, ng, d_batch_aos, d_ao_to_atom.data(),
        st->phi.data(), st->t_mat.data(), st->wv_grad.data(), d_grad.data());
  }

  d_grad.download(h_grad.data(), h_grad.size());
  check_cuda(cudaDeviceSynchronize(), "xc_cuda_nuclear_grad sync");
  return h_grad;
}

}  // namespace gpu4dft::detail
