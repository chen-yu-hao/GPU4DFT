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

// Kernel B: 2D (point_block, ao) coalesced half-weighted W[u * ng + g] builder (100% SM saturation!).
__global__ void build_vxc_weight_matrix_kernel(int family,
                                               std::size_t na,
                                               std::size_t ng,
                                               const double* __restrict__ phi,
                                               const double* __restrict__ wv,
                                               double* __restrict__ w_mat) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t u = static_cast<std::size_t>(blockIdx.y);
  if (g >= ng || u >= na) return;

  const std::size_t comp_stride = na * ng;
  const std::size_t off = u * ng + g;
  double w_ug = wv[0 * ng + g] * phi[off];
  if (family >= 1) {
    w_ug += wv[1 * ng + g] * phi[1 * comp_stride + off] +
            wv[2 * ng + g] * phi[2 * comp_stride + off] +
            wv[3 * ng + g] * phi[3 * comp_stride + off];
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
                                             double* __restrict__ wv_out) {
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
  std::size_t phi_cache_offset = 0;
};

struct XcStreamWorker {
  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  cublasHandle_t cublas = nullptr;

  DevBuf<double> phi_scratch;
  DevBuf<double> d_act;
  DevBuf<double> t_mat;
  DevBuf<double> w_mat;
  DevBuf<double> v_half;
  DevBuf<double> wv;
  DevBuf<double> scalars;
  DevBuf<double> vxc_part;

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
  XcOptions opts;
  std::size_t npoints = 0;

  DevBuf<double> coords;
  DevBuf<double> weights;
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

  // Host-API staging buffers
  std::size_t max_na = 0;
  std::size_t max_ng = 0;
  DevBuf<double> dm_host_buf;
  DevBuf<double> vxc_host_buf;

  // Lazy nuclear-gradient scratch buffer (deriv=1 or 2 -> up to 10 components)
  DevBuf<double> grad_phi_scratch;

  // Lazy ground-state f_xc[rho_0] cache for TDDFT linear response
  bool fxc_ready = false;
  DevBuf<double> fxc_cache;

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

    // 1. Retrieve persistent GPU cached active-batch Phi (or evaluate on-the-fly if over cap)
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

    // 2. Gather symmetric active density block D_act [na x na]
    gather_sym_dm_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        d_dm, nao, d_batch_aos, na, ws.d_act.data());

    // 3. cuBLAS GEMM: T0 = D_act * Phi0 (or single StridedBatched GEMM for MGGA T0..Tz!)
    if (st->family == 2) {
      ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), d_phi,
                           static_cast<long long>(comp_stride),
                           ws.t_mat.data(), static_cast<long long>(comp_stride),
                           4, st->opts.precision);
    } else {
      ws.gemm_nn(i_na, i_ng, i_na, ws.d_act.data(), d_phi, ws.t_mat.data(),
                 false, st->opts.precision);
    }

    // 4. Kernel A: 2D (64 x 4) high-occupancy density + symbolic XC potentials
    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    eval_rho_and_vxc_potentials_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->func_id, st->family, st->opts.rho_cutoff,
        na, ng, st->weights.data(), b.point_offset,
        d_phi, ws.t_mat.data(), ws.wv.data(), ws.scalars.data());

    // 5. Kernel B: 2D (point_block, ao) coalesced W[u * ng + g] build + cuBLAS GEMM V_half = Phi0 * W^T
    const dim3 w_grid(static_cast<unsigned>((ng + 127) / 128),
                      static_cast<unsigned>(na), 1);
    build_vxc_weight_matrix_kernel<<<w_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, d_phi, ws.wv.data(), ws.w_mat.data());

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

    // 6. Scatter symmetrized V_act = V_half + V_half^T into stream's V_xc buffer
    double* target_vxc = (n_streams == 1) ? d_vxc : ws.vxc_part.data();
    scatter_sym_vxc_kernel<<<static_cast<unsigned>((na * na + 255) / 256), 256, 0, ws.stream>>>(
        ws.v_half.data(), d_batch_aos, na, nao, target_vxc);
  }

  // Join worker streams back onto caller_stream
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
        st->family, na, ng, d_phi, ws.wv.data(), ws.w_mat.data());

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
                                         const int* ao_to_atom,
                                         int natom,
                                         const double* h_dm) {
  auto* st = static_cast<CudaXcEngineImpl*>(raw_impl);
  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  std::vector<double> h_grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (st->batches.empty() || nao == 0 || natom == 0) return h_grad;

  XcStreamWorker& ws = st->workers[0];
  const int deriv = (st->family == 0) ? 1 : 2;
  const std::size_t ncomp_grad = (deriv == 1) ? 4 : 10;
  const std::size_t n_fcomp = (st->family >= 1) ? 4 : 1;

  st->grad_phi_scratch.resize(ncomp_grad * st->max_na * st->max_ng);
  ws.t_mat.resize(n_fcomp * st->max_na * st->max_ng);

  st->dm_host_buf.upload(h_dm, nao * nao, ws.stream);
  DevBuf<int> d_ao_to_atom;
  d_ao_to_atom.upload(ao_to_atom, nao, ws.stream);
  DevBuf<double> d_grad(static_cast<std::size_t>(natom) * 3);
  check_cuda(cudaMemsetAsync(d_grad.data(), 0,
                             static_cast<std::size_t>(natom) * 3 * sizeof(double), ws.stream),
             "xc grad memset");

  const FlatBasisView bview = st->basis.view();

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

    // Always use FP64 cublasDgemmStridedBatched for analytical nuclear gradients (< 1e-12 accuracy)
    ws.gemm_nn_strided_b(i_na, i_ng, i_na, ws.d_act.data(), st->grad_phi_scratch.data(),
                         static_cast<long long>(comp_stride),
                         ws.t_mat.data(), static_cast<long long>(comp_stride),
                         static_cast<int>(n_fcomp), Precision::FP64);

    const dim3 pot_blocks(static_cast<unsigned>((ng + 63) / 64), 1, 1);
    const dim3 pot_threads(64, 4, 1);
    grad_xc_point_weights_kernel<<<pot_blocks, pot_threads, 0, ws.stream>>>(
        st->func_id, st->family, st->opts.rho_cutoff,
        na, ng, st->weights.data(), b.point_offset,
        st->grad_phi_scratch.data(), ws.t_mat.data(), ws.wv.data());

    const dim3 red_grid(static_cast<unsigned>((ng + 127) / 128),
                        static_cast<unsigned>(na), 1);
    grad_xc_accum_atom_kernel<<<red_grid, 128, 0, ws.stream>>>(
        st->family, na, ng, d_batch_aos, d_ao_to_atom.data(),
        st->grad_phi_scratch.data(), ws.t_mat.data(), ws.wv.data(), d_grad.data());
  }

  d_grad.download(h_grad.data(), h_grad.size(), ws.stream);
  check_cuda(cudaStreamSynchronize(ws.stream), "xc_cuda_nuclear_grad sync");
  return h_grad;
}

}  // namespace gpu4dft::detail
