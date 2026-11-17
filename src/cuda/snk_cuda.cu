#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/snk.hpp"

namespace gpu4dft::snk_detail {
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

inline void check_cusolver(cusolverStatus_t status, const char* what) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template <typename T>
class DevBuf {
 public:
  DevBuf() = default;
  explicit DevBuf(std::size_t count) { resize(count); }
  DevBuf(const T* src, std::size_t count) {
    resize(count);
    upload(src, count);
  }
  explicit DevBuf(const std::vector<T>& vec) : DevBuf(vec.data(), vec.size()) {}

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
      check_cuda(cudaMalloc(&ptr_, size_ * sizeof(T)), "snk cudaMalloc");
    }
  }

  void upload(const T* src, std::size_t count, cudaStream_t stream = nullptr) {
    if (count == 0) return;
    resize(count);
    check_cuda(cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream),
               "snk cudaMemcpyAsync H2D");
  }

  void download(T* dst, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count == 0) return;
    check_cuda(cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
               "snk cudaMemcpyAsync D2H");
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

  explicit DevFlatBasis(const detail::FlatBasisData& h)
      : nshell(h.nshell),
        nao(h.nao),
        shell_l(h.shell_l),
        shell_ao_offset(h.shell_ao_offset),
        shell_prim_offset(h.shell_prim_offset),
        shell_prim_count(h.shell_prim_count),
        shell_center(h.shell_center),
        prim_exp(h.prim_exp),
        prim_coeff(h.prim_coeff) {}

  detail::FlatBasisView view() const {
    return detail::FlatBasisView{
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

__global__ void esp_single_point_kernel(detail::FlatBasisView bview,
                                        const SnkShellPair* __restrict__ pairs,
                                        std::size_t npairs,
                                        double gx, double gy, double gz,
                                        double pair_cutoff,
                                        double* __restrict__ out_v) {
  const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ip >= npairs) return;

  const int ia = pairs[ip].ish_a;
  const int ib = pairs[ip].ish_b;
  const int nma = 2 * bview.shell_l[ia] + 1;
  const int nmb = 2 * bview.shell_l[ib] + 1;
  const std::size_t ao_a = static_cast<std::size_t>(bview.shell_ao_offset[ia]);
  const std::size_t ao_b = static_cast<std::size_t>(bview.shell_ao_offset[ib]);
  const std::size_t nao = static_cast<std::size_t>(bview.nao);

  double block[81];
  eval_shell_pair_esp(bview, ia, ib, gx, gy, gz, pair_cutoff, block);

  for (int ma = 0; ma < nma; ++ma) {
    for (int mb = 0; mb < nmb; ++mb) {
      const double val = block[ma * nmb + mb];
      const std::size_t u = ao_a + static_cast<std::size_t>(ma);
      const std::size_t v = ao_b + static_cast<std::size_t>(mb);
      out_v[u * nao + v] = val;
      out_v[v * nao + u] = val;
    }
  }
}

__global__ void snk_eval_active_ao_kernel(detail::FlatBasisView bview,
                                          const double* __restrict__ coords,
                                          std::size_t point_offset,
                                          std::size_t ng,
                                          std::size_t na,
                                          const int* __restrict__ act_shells,
                                          const int* __restrict__ act_shell_local_ao,
                                          int n_act_shells,
                                          double* __restrict__ phi) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int s_idx = static_cast<int>(blockIdx.y);
  if (g >= ng || s_idx >= n_act_shells) return;

  const int is = act_shells[s_idx];
  const std::size_t local_ao0 = static_cast<std::size_t>(act_shell_local_ao[s_idx]);
  const std::size_t gp = point_offset + g;

  detail::eval_shell_at_point(bview, is,
                              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                              0, phi + local_ao0 * ng + g, na * ng, ng);
}

__global__ void snk_weight_phi_2d_kernel(std::size_t na,
                                         std::size_t ng,
                                         const double* __restrict__ weights,
                                         std::size_t point_offset,
                                         const double* __restrict__ phi,
                                         double* __restrict__ phi_w) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t u = static_cast<std::size_t>(blockIdx.y);
  if (g >= ng || u >= na) return;
  const std::size_t idx = u * ng + g;
  phi_w[idx] = weights[point_offset + g] * phi[idx];
}

__global__ void snk_scatter_submatrix_kernel(const double* __restrict__ sg_act,
                                             const int* __restrict__ act_aos,
                                             std::size_t na,
                                             std::size_t nao,
                                             double* __restrict__ sg_full) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * na) return;
  const std::size_t u = idx / na;
  const std::size_t v = idx % na;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  const std::size_t gv = static_cast<std::size_t>(act_aos[v]);
  sg_full[gu * nao + gv] += sg_act[idx];
}

__global__ void snk_symmetrize_dm_kernel(const double* __restrict__ dm,
                                         std::size_t nao,
                                         double* __restrict__ dm_sym) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nao * nao) return;
  const std::size_t i = idx / nao;
  const std::size_t j = idx % nao;
  dm_sym[idx] = 0.5 * (dm[i * nao + j] + dm[j * nao + i]);
}

__global__ void snk_gather_dm_cols_kernel(const double* __restrict__ dm_sym,
                                          std::size_t nao,
                                          const int* __restrict__ act_aos,
                                          std::size_t na,
                                          double* __restrict__ d_cols) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nao * na) return;
  const std::size_t v = idx / na;
  const std::size_t u = idx % na;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  d_cols[idx] = dm_sym[v * nao + gu];
}

// Computes max_{m in shell is, g in 128-pt tile} |F[(ao0 + m) * chunk_ng + g]| for fast
// block-uniform Cauchy-Schwarz screening inside `snk_esp_contract_ab_kernel`.
__global__ void snk_compute_f_shell_max_kernel(detail::FlatBasisView bview,
                                               std::size_t chunk_ng,
                                               std::size_t n_tiles,
                                               const double* __restrict__ f_mat,
                                               double* __restrict__ f_shell_max) {
  constexpr int kTile = 128;
  __shared__ double s_warp[4];

  const std::size_t tile = static_cast<std::size_t>(blockIdx.x);
  const int is = static_cast<int>(blockIdx.y);
  if (tile >= n_tiles || is >= bview.nshell) return;

  const std::size_t g = tile * kTile + threadIdx.x;
  const int nm = 2 * bview.shell_l[is] + 1;
  const std::size_t ao0 = static_cast<std::size_t>(bview.shell_ao_offset[is]);

  double vmax = 0.0;
  if (g < chunk_ng) {
    for (int m = 0; m < nm; ++m) {
      vmax = ::fmax(vmax, ::fabs(f_mat[(ao0 + static_cast<std::size_t>(m)) * chunk_ng + g]));
    }
  }

  for (int mask = 16; mask > 0; mask >>= 1) {
    vmax = ::fmax(vmax, __shfl_down_sync(0xffffffffu, vmax, mask));
  }
  if ((threadIdx.x & 31) == 0) {
    s_warp[threadIdx.x >> 5] = vmax;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const double bmax = ::fmax(::fmax(s_warp[0], s_warp[1]), ::fmax(s_warp[2], s_warp[3]));
    f_shell_max[static_cast<std::size_t>(is) * n_tiles + tile] = bmax;
  }
}

// Zero-stack-spill, fast-compiling 1e ESP contraction kernel for all L in {0, 1, 2, 3, 4}:
//   - Block-uniform Cauchy-Schwarz + far-field support screening
//   - Block-uniform concentric multipole fast path (1 cooperative shared-mem load + 1 rsqrt(R^2)!)
//   - Cooperative shared-memory precomputed Spherical-to-Hermite H^{(k)} (zero md_hermite_1d on GPU!)
//   - Eliminates `v_blk[]` array completely by contracting `s_H * R` directly with `f_a, f_b`!
//   - Only `t = 0..KTET-1` is unrolled -> R[KTET] stays 100% in registers & nvcc compiles in ~3s!
template <int L>
__global__ void snk_esp_contract_l_kernel(const SnkShellPair* __restrict__ pairs,
                                          std::size_t npairs,
                                          const SnkPrimPair* __restrict__ prims,
                                          const double* __restrict__ h_coeffs,
                                          const double* __restrict__ coords,
                                          std::size_t chunk_pt_offset,
                                          std::size_t chunk_ng,
                                          std::size_t n_tiles,
                                          const double* __restrict__ tile_boxes,
                                          const double* __restrict__ f_shell_max,
                                          const double* __restrict__ f_mat,
                                          double* __restrict__ g_mat) {
  constexpr int kTile = 128;
  constexpr int KTET = tet_count(L);
  // Max (nma * nmb) for canonical la >= lb with la + lb == L:
  //   L=0: 1x1=1, L=1: 3x1=3, L=2: 3x3=9, L=3: 5x3=15, L=4: 5x5=25
  constexpr int kMaxUV = (L == 0) ? 1 : ((L == 1) ? 3 : ((L == 2) ? 9 : ((L == 3) ? 15 : 25)));

  __shared__ double s_H[kMaxUV * KTET];

  const std::size_t tile = static_cast<std::size_t>(blockIdx.x);
  const std::size_t ip = static_cast<std::size_t>(blockIdx.y);
  if (tile >= n_tiles || ip >= npairs) return;

  const SnkShellPair pr = pairs[ip];
  const double* __restrict__ tbox = tile_boxes + 6 * tile;

  double d2_box = 0.0;
  #pragma unroll
  for (int d = 0; d < 3; ++d) {
    double dd = 0.0;
    const double c = pr.center[d];
    if (c < tbox[d]) dd = tbox[d] - c;
    else if (c > tbox[3 + d]) dd = c - tbox[3 + d];
    d2_box += dd * dd;
  }
  const double dist_box = ::sqrt(d2_box);
  const double v_ub = (dist_box > pr.extent_radius + 1.0)
                          ? (pr.q_far_bound / (dist_box - pr.extent_radius))
                          : pr.v_max_near;
  const double f_ub = ::fmax(f_shell_max[static_cast<std::size_t>(pr.ish_a) * n_tiles + tile],
                             f_shell_max[static_cast<std::size_t>(pr.ish_b) * n_tiles + tile]);
  if (v_ub * f_ub < 1e-14) return;

  const std::size_t g = tile * kTile + threadIdx.x;
  const bool active = (g < chunk_ng);
  double gx = 0.0, gy = 0.0, gz = 0.0;
  if (active) {
    const std::size_t gp = chunk_pt_offset + g;
    gx = coords[3 * gp + 0];
    gy = coords[3 * gp + 1];
    gz = coords[3 * gp + 2];
  }

  const int nma = pr.nma;
  const int nmb = pr.nmb;
  const int h_len = nma * nmb * KTET;
  const std::size_t ao_a = static_cast<std::size_t>(pr.ao_a);
  const std::size_t ao_b = static_cast<std::size_t>(pr.ao_b);

  if constexpr (L == 0) {
    double v00 = 0.0;
    if (pr.asymp_h_offset >= 0 && pr.p_min * d2_box > boys_asymp_threshold(0)) {
      const double q_mono = h_coeffs[pr.asymp_h_offset];
      if (active) {
        const double xpc = pr.center[0] - gx;
        const double ypc = pr.center[1] - gy;
        const double zpc = pr.center[2] - gz;
        v00 = q_mono * ::rsqrt(xpc * xpc + ypc * ypc + zpc * zpc);
      }
    } else {
      for (int kp = 0; kp < pr.prim_count; ++kp) {
        const SnkPrimPair pp = prims[pr.prim_offset + kp];
        const double h0 = h_coeffs[pp.h_offset];
        if (active) {
          const double xpc = pp.px - gx;
          const double ypc = pp.py - gy;
          const double zpc = pp.pz - gz;
          double f0 = 0.0;
          boys_vec_hd(0, pp.p * (xpc * xpc + ypc * ypc + zpc * zpc), &f0);
          v00 += h0 * f0;
        }
      }
    }
    if (active) {
      const double fa = f_mat[ao_a * chunk_ng + g];
      const double fb = f_mat[ao_b * chunk_ng + g];
      atomicAdd(&g_mat[ao_a * chunk_ng + g], v00 * fb);
      if (pr.ish_a != pr.ish_b) {
        atomicAdd(&g_mat[ao_b * chunk_ng + g], v00 * fa);
      }
    }
    return;
  }

  // L in {1, 2, 3, 4}: accumulate in local acc_a / acc_b across ALL primitives first,
  // issuing only `nma + nmb` global atomicAdds ONCE at the end of the kernel!
  constexpr int kMaxNMA = (L == 1) ? 3 : ((L == 2) ? 5 : ((L == 3) ? 7 : 9));
  constexpr int kMaxNMB = (L <= 1) ? 1 : ((L <= 3) ? 3 : 5);
  double fa_loc[kMaxNMA];
  double fb_loc[kMaxNMB];
  double acc_a[kMaxNMA];
  double acc_b[kMaxNMB];
  if (active) {
    for (int ma = 0; ma < nma; ++ma) {
      fa_loc[ma] = f_mat[(ao_a + static_cast<std::size_t>(ma)) * chunk_ng + g];
      acc_a[ma] = 0.0;
    }
    for (int mb = 0; mb < nmb; ++mb) {
      fb_loc[mb] = f_mat[(ao_b + static_cast<std::size_t>(mb)) * chunk_ng + g];
      acc_b[mb] = 0.0;
    }
  }

  if (pr.asymp_h_offset >= 0 && pr.p_min * d2_box > boys_asymp_threshold(L)) {
    for (int i = static_cast<int>(threadIdx.x); i < h_len; i += kTile) {
      s_H[i] = h_coeffs[pr.asymp_h_offset + i];
    }
    __syncthreads();

    if (active) {
      const double xpc = pr.center[0] - gx;
      const double ypc = pr.center[1] - gy;
      const double zpc = pr.center[2] - gz;
      const double inv_r = ::rsqrt(xpc * xpc + ypc * ypc + zpc * zpc);
      const double inv_r2 = inv_r * inv_r;

      double d[5];
      d[0] = inv_r;
      d[1] = -d[0] * inv_r2;
      if constexpr (L >= 2) d[2] = -3.0 * d[1] * inv_r2;
      if constexpr (L >= 3) d[3] = -5.0 * d[2] * inv_r2;
      if constexpr (L >= 4) d[4] = -7.0 * d[3] * inv_r2;

      double R[KTET];
      eval_hermite_coulomb_reg<L>(d, xpc, ypc, zpc, R);

      #pragma unroll 1
      for (int ma = 0; ma < nma; ++ma) {
        const double fa_val = fa_loc[ma];
        double row_acc = 0.0;
        #pragma unroll 1
        for (int mb = 0; mb < nmb; ++mb) {
          const double* row = s_H + (ma * nmb + mb) * KTET;
          double a_uv = 0.0;
          #pragma unroll
          for (int t = 0; t < KTET; ++t) a_uv += row[t] * R[t];
          row_acc += a_uv * fb_loc[mb];
          acc_b[mb] += a_uv * fa_val;
        }
        acc_a[ma] = row_acc;
      }
    }
  } else {
    for (int kp = 0; kp < pr.prim_count; ++kp) {
      const SnkPrimPair pp = prims[pr.prim_offset + kp];
      for (int i = static_cast<int>(threadIdx.x); i < h_len; i += kTile) {
        s_H[i] = h_coeffs[pp.h_offset + i];
      }
      __syncthreads();

      if (active) {
        const double xpc = pp.px - gx;
        const double ypc = pp.py - gy;
        const double zpc = pp.pz - gz;
        double F[5];
        boys_vec_hd(L, pp.p * (xpc * xpc + ypc * ypc + zpc * zpc), F);

        const double m2p = -2.0 * pp.p;
        double d[5];
        d[0] = F[0];
        d[1] = m2p * F[1];
        if constexpr (L >= 2) d[2] = (m2p * m2p) * F[2];
        if constexpr (L >= 3) d[3] = (m2p * m2p * m2p) * F[3];
        if constexpr (L >= 4) d[4] = (m2p * m2p) * (m2p * m2p) * F[4];

        double R[KTET];
        eval_hermite_coulomb_reg<L>(d, xpc, ypc, zpc, R);

        #pragma unroll 1
        for (int ma = 0; ma < nma; ++ma) {
          const double fa_val = fa_loc[ma];
          double row_acc = 0.0;
          #pragma unroll 1
          for (int mb = 0; mb < nmb; ++mb) {
            const double* row = s_H + (ma * nmb + mb) * KTET;
            double a_uv = 0.0;
            #pragma unroll
            for (int t = 0; t < KTET; ++t) a_uv += row[t] * R[t];
            row_acc += a_uv * fb_loc[mb];
            acc_b[mb] += a_uv * fa_val;
          }
          acc_a[ma] += row_acc;
        }
      }
      __syncthreads();
    }
  }

  if (active) {
    for (int ma = 0; ma < nma; ++ma) {
      atomicAdd(&g_mat[(ao_a + static_cast<std::size_t>(ma)) * chunk_ng + g], acc_a[ma]);
    }
    if (pr.ish_a != pr.ish_b) {
      for (int mb = 0; mb < nmb; ++mb) {
        atomicAdd(&g_mat[(ao_b + static_cast<std::size_t>(mb)) * chunk_ng + g], acc_b[mb]);
      }
    }
  }
}

// High-angular-momentum fallback kernel for 5 <= L <= 8 (f-d up to g-g shells):
// Keeps SphCartTerm and Ex/Ey/Ez in shared memory per block to minimize per-thread stack!
__global__ void snk_esp_contract_high_l_kernel(detail::FlatBasisView bview,
                                               const SnkShellPair* __restrict__ pairs,
                                               std::size_t npairs,
                                               const SnkPrimPair* __restrict__ prims,
                                               const double* __restrict__ coords,
                                               std::size_t chunk_pt_offset,
                                               std::size_t chunk_ng,
                                               std::size_t n_tiles,
                                               const double* __restrict__ tile_boxes,
                                               const double* __restrict__ f_shell_max,
                                               const double* __restrict__ f_mat,
                                               double* __restrict__ g_mat) {
  constexpr int kTile = 128;
  __shared__ SphCartTerm s_ta[28];
  __shared__ SphCartTerm s_tb[28];
  __shared__ int s_nta;
  __shared__ int s_ntb;
  __shared__ double s_Ex[5][5][9];
  __shared__ double s_Ey[5][5][9];
  __shared__ double s_Ez[5][5][9];

  const std::size_t tile = static_cast<std::size_t>(blockIdx.x);
  const std::size_t ip = static_cast<std::size_t>(blockIdx.y);
  if (tile >= n_tiles || ip >= npairs) return;

  const SnkShellPair pr = pairs[ip];
  const double* __restrict__ tbox = tile_boxes + 6 * tile;

  double d2_box = 0.0;
  for (int d = 0; d < 3; ++d) {
    double dd = 0.0;
    const double c = pr.center[d];
    if (c < tbox[d]) dd = tbox[d] - c;
    else if (c > tbox[3 + d]) dd = c - tbox[3 + d];
    d2_box += dd * dd;
  }
  const double dist_box = ::sqrt(d2_box);
  const double v_ub = (dist_box > pr.extent_radius + 1.0)
                          ? (pr.q_far_bound / (dist_box - pr.extent_radius))
                          : pr.v_max_near;
  const double f_ub = ::fmax(f_shell_max[static_cast<std::size_t>(pr.ish_a) * n_tiles + tile],
                             f_shell_max[static_cast<std::size_t>(pr.ish_b) * n_tiles + tile]);
  if (v_ub * f_ub < 1e-14) return;

  if (threadIdx.x == 0) {
    s_nta = sph_cart_terms(pr.la, s_ta);
    s_ntb = sph_cart_terms(pr.lb, s_tb);
  }
  __syncthreads();

  const std::size_t g = tile * kTile + threadIdx.x;
  const bool active = (g < chunk_ng);
  double gx = 0.0, gy = 0.0, gz = 0.0;
  if (active) {
    const std::size_t gp = chunk_pt_offset + g;
    gx = coords[3 * gp + 0];
    gy = coords[3 * gp + 1];
    gz = coords[3 * gp + 2];
  }

  const int nma = pr.nma;
  const int nmb = pr.nmb;
  const int L = pr.la + pr.lb;
  const double ax = bview.shell_center[3 * pr.ish_a + 0];
  const double ay = bview.shell_center[3 * pr.ish_a + 1];
  const double az = bview.shell_center[3 * pr.ish_a + 2];
  const double bx = bview.shell_center[3 * pr.ish_b + 0];
  const double by = bview.shell_center[3 * pr.ish_b + 1];
  const double bz = bview.shell_center[3 * pr.ish_b + 2];

  double v_block[81];
  for (int i = 0; i < nma * nmb; ++i) v_block[i] = 0.0;
  double R_tet[165];

  for (int kp = 0; kp < pr.prim_count; ++kp) {
    const SnkPrimPair pp = prims[pr.prim_offset + kp];
    if (threadIdx.x == 0) {
      const double inv_2p = 0.5 / pp.p;
      md_hermite_1d(pr.la, pr.lb, pp.px - ax, pp.px - bx, inv_2p, s_Ex);
      md_hermite_1d(pr.la, pr.lb, pp.py - ay, pp.py - by, inv_2p, s_Ey);
      md_hermite_1d(pr.la, pr.lb, pp.pz - az, pp.pz - bz, inv_2p, s_Ez);
    }
    __syncthreads();

    if (active) {
      md_hermite_coulomb(L, pp.p, pp.px - gx, pp.py - gy, pp.pz - gz, R_tet);
      #pragma unroll 1
      for (int i_a = 0; i_a < s_nta; ++i_a) {
        const SphCartTerm& sa = s_ta[i_a];
        #pragma unroll 1
        for (int i_b = 0; i_b < s_ntb; ++i_b) {
          const SphCartTerm& sb = s_tb[i_b];
          const int mx = sa.ix + sb.ix;
          const int my = sa.iy + sb.iy;
          const int mz = sa.iz + sb.iz;
          double sum_tuv = 0.0;
          for (int t = 0; t <= mx; ++t) {
            const double ex = s_Ex[sa.ix][sb.ix][t];
            if (ex == 0.0) continue;
            for (int u = 0; u <= my; ++u) {
              const double exy = ex * s_Ey[sa.iy][sb.iy][u];
              if (exy == 0.0) continue;
              for (int v = 0; v <= mz; ++v) {
                sum_tuv += exy * s_Ez[sa.iz][sb.iz][v] * R_tet[tet_index(t, u, v)];
              }
            }
          }
          v_block[sa.m * nmb + sb.m] += pp.pre * (sa.coeff * sb.coeff) * sum_tuv;
        }
      }
    }
    __syncthreads();
  }

  if (active) {
    const std::size_t ao_a = static_cast<std::size_t>(pr.ao_a);
    const std::size_t ao_b = static_cast<std::size_t>(pr.ao_b);
    double acc_b[9] = {0.0};
    for (int ma = 0; ma < nma; ++ma) {
      const double fa_g = f_mat[(ao_a + static_cast<std::size_t>(ma)) * chunk_ng + g];
      double acc_a = 0.0;
      for (int mb = 0; mb < nmb; ++mb) {
        const double a_uv = v_block[ma * nmb + mb];
        acc_a += a_uv * f_mat[(ao_b + static_cast<std::size_t>(mb)) * chunk_ng + g];
        acc_b[mb] += a_uv * fa_g;
      }
      atomicAdd(&g_mat[(ao_a + static_cast<std::size_t>(ma)) * chunk_ng + g], acc_a);
    }
    if (pr.ish_a != pr.ish_b) {
      for (int mb = 0; mb < nmb; ++mb) {
        atomicAdd(&g_mat[(ao_b + static_cast<std::size_t>(mb)) * chunk_ng + g], acc_b[mb]);
      }
    }
  }
}

__global__ void snk_scatter_rows_kernel(const double* __restrict__ k_batch,
                                        const int* __restrict__ act_aos,
                                        std::size_t na,
                                        std::size_t nao,
                                        double* __restrict__ k_raw) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * nao) return;
  const std::size_t u = idx / nao;
  const std::size_t v = idx % nao;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  k_raw[gu * nao + v] += k_batch[idx];
}

__global__ void snk_reduce_stream_k_kernel(std::size_t nao2,
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

__global__ void snk_build_pinv_kernel(const double* __restrict__ evec,
                                      const double* __restrict__ eval,
                                      int nao,
                                      double lindep_tol,
                                      double* __restrict__ sg_pinv,
                                      double* __restrict__ p_null) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  if (idx >= nao * nao) return;
  const int i = idx / nao;
  const int j = idx % nao;

  double pinv_ij = 0.0;
  double null_ij = 0.0;
  for (int k = 0; k < nao; ++k) {
    const double lam = eval[k];
    const double vik = evec[k * nao + i];
    const double vjk = evec[k * nao + j];
    if (lam > lindep_tol) {
      pinv_ij += (vik * vjk) / lam;
    } else {
      null_ij += vik * vjk;
    }
  }
  sg_pinv[idx] = pinv_ij;
  p_null[idx] = null_ij;
}

__global__ void snk_add_inplace_kernel(const double* __restrict__ src,
                                       double* __restrict__ dst,
                                       std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void snk_cast_f64_to_f32_2d_kernel(const double* __restrict__ src,
                                              std::size_t src_ld,
                                              float* __restrict__ dst,
                                              std::size_t dst_ld,
                                              std::size_t rows,
                                              std::size_t cols) {
  const std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t r = static_cast<std::size_t>(blockIdx.y);
  if (c < cols && r < rows) {
    dst[r * dst_ld + c] = static_cast<float>(src[r * src_ld + c]);
  }
}

__global__ void snk_cast_f32_to_f64_2d_kernel(const float* __restrict__ src,
                                              std::size_t src_ld,
                                              double* __restrict__ dst,
                                              std::size_t dst_ld,
                                              std::size_t rows,
                                              std::size_t cols) {
  const std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t r = static_cast<std::size_t>(blockIdx.y);
  if (c < cols && r < rows) {
    dst[r * dst_ld + c] = static_cast<double>(src[r * src_ld + c]);
  }
}

struct SnkBatchGpuMeta {
  std::size_t point_offset = 0;
  std::size_t ng = 0;
  std::size_t na = 0;
  int n_act_shells = 0;
  std::size_t shell_list_offset = 0;
  std::size_t ao_list_offset = 0;
  std::size_t phi_cache_offset = 0;
};

struct SnkEspChunkMeta {
  std::size_t batch_begin = 0;
  std::size_t batch_end = 0;
  std::size_t point_offset = 0;
  std::size_t ng = 0;
  std::size_t n_tiles = 0;
  std::size_t tile_box_offset = 0;  // offset in doubles into `tile_boxes` (6 * tile_idx)
};

struct SnkStreamWorker {
  cudaStream_t stream = nullptr;
  cudaEvent_t done_event = nullptr;
  cublasHandle_t cublas = nullptr;

  DevBuf<double> phi_scratch;
  DevBuf<double> phi_w_scratch;
  DevBuf<double> d_cols;
  DevBuf<double> f_chunk;
  DevBuf<double> g_chunk;
  DevBuf<double> f_shell_max;
  DevBuf<double> k_batch;
  DevBuf<double> k_raw_part;

  DevBuf<float> f32_a;
  DevBuf<float> f32_b;
  DevBuf<float> f32_c;

  SnkStreamWorker() = default;
  ~SnkStreamWorker() {
    if (cublas != nullptr) cublasDestroy(cublas);
    if (done_event != nullptr) cudaEventDestroy(done_event);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }
  SnkStreamWorker(const SnkStreamWorker&) = delete;
  SnkStreamWorker& operator=(const SnkStreamWorker&) = delete;

  void init_handles() {
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "snk stream create");
    check_cuda(cudaEventCreateWithFlags(&done_event, cudaEventDisableTiming), "snk event create");
    check_cublas(cublasCreate(&cublas), "cublasCreate (SnkStreamWorker)");
    check_cublas(cublasSetStream(cublas, stream), "cublasSetStream (SnkStreamWorker)");
  }

  // Row-major C_sub[M x N] (ldc >= N) = A[M x K] (lda = K) * B[K x N] (ldb = N)
  void gemm_nn_strided_c(int M, int N, int K,
                         const double* dA, const double* dB,
                         double* dC, int ldc, Precision prec) {
    if (prec == Precision::FP32) {
      const dim3 ga(static_cast<unsigned>((K + 127) / 128), static_cast<unsigned>(M), 1);
      const dim3 gb(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>(K), 1);
      const dim3 gc(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>(M), 1);
      snk_cast_f64_to_f32_2d_kernel<<<ga, 128, 0, stream>>>(
          dA, static_cast<std::size_t>(K), f32_a.data(), static_cast<std::size_t>(K),
          static_cast<std::size_t>(M), static_cast<std::size_t>(K));
      snk_cast_f64_to_f32_2d_kernel<<<gb, 128, 0, stream>>>(
          dB, static_cast<std::size_t>(N), f32_b.data(), static_cast<std::size_t>(N),
          static_cast<std::size_t>(K), static_cast<std::size_t>(N));
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               f32_b.data(), N,
                               f32_a.data(), K,
                               &beta, f32_c.data(), N),
                   "snk cublasSgemm NN");
      snk_cast_f32_to_f64_2d_kernel<<<gc, 128, 0, stream>>>(
          f32_c.data(), static_cast<std::size_t>(N), dC, static_cast<std::size_t>(ldc),
          static_cast<std::size_t>(M), static_cast<std::size_t>(N));
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             dB, N,
                             dA, K,
                             &beta, dC, ldc),
                 "snk cublasDgemm NN");
  }

  // Row-major C[M x N] (ldc = N) = A[M x K] (lda = K) * B_sub[N x K]^T (ldb >= K)
  void gemm_nt_strided_b(int M, int N, int K,
                         const double* dA, const double* dB, int ldb,
                         double* dC, Precision prec) {
    if (prec == Precision::FP32) {
      const dim3 ga(static_cast<unsigned>((K + 127) / 128), static_cast<unsigned>(M), 1);
      const dim3 gb(static_cast<unsigned>((K + 127) / 128), static_cast<unsigned>(N), 1);
      const dim3 gc(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>(M), 1);
      snk_cast_f64_to_f32_2d_kernel<<<ga, 128, 0, stream>>>(
          dA, static_cast<std::size_t>(K), f32_a.data(), static_cast<std::size_t>(K),
          static_cast<std::size_t>(M), static_cast<std::size_t>(K));
      snk_cast_f64_to_f32_2d_kernel<<<gb, 128, 0, stream>>>(
          dB, static_cast<std::size_t>(ldb), f32_b.data(), static_cast<std::size_t>(K),
          static_cast<std::size_t>(N), static_cast<std::size_t>(K));
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               f32_b.data(), K,
                               f32_a.data(), K,
                               &beta, f32_c.data(), N),
                   "snk cublasSgemm NT");
      snk_cast_f32_to_f64_2d_kernel<<<gc, 128, 0, stream>>>(
          f32_c.data(), static_cast<std::size_t>(N), dC, static_cast<std::size_t>(N),
          static_cast<std::size_t>(M), static_cast<std::size_t>(N));
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             dB, ldb,
                             dA, K,
                             &beta, dC, N),
                 "snk cublasDgemm NT");
  }

  // Row-major C[M x N] = A[K x M]^T * B[K x N]
  void gemm_tn(int M, int N, int K, const double* dA, const double* dB, double* dC,
               Precision prec, cudaStream_t target_stream) {
    check_cublas(cublasSetStream(cublas, target_stream), "snk cublasSetStream TN");
    if (prec == Precision::FP32) {
      const dim3 ga(static_cast<unsigned>((M + 127) / 128), static_cast<unsigned>(K), 1);
      const dim3 gb(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>(K), 1);
      const dim3 gc(static_cast<unsigned>((N + 127) / 128), static_cast<unsigned>(M), 1);
      snk_cast_f64_to_f32_2d_kernel<<<ga, 128, 0, target_stream>>>(
          dA, static_cast<std::size_t>(M), f32_a.data(), static_cast<std::size_t>(M),
          static_cast<std::size_t>(K), static_cast<std::size_t>(M));
      snk_cast_f64_to_f32_2d_kernel<<<gb, 128, 0, target_stream>>>(
          dB, static_cast<std::size_t>(N), f32_b.data(), static_cast<std::size_t>(N),
          static_cast<std::size_t>(K), static_cast<std::size_t>(N));
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                               N, M, K, &alpha,
                               f32_b.data(), N,
                               f32_a.data(), M,
                               &beta, f32_c.data(), N),
                   "snk cublasSgemm TN");
      snk_cast_f32_to_f64_2d_kernel<<<gc, 128, 0, target_stream>>>(
          f32_c.data(), static_cast<std::size_t>(N), dC, static_cast<std::size_t>(N),
          static_cast<std::size_t>(M), static_cast<std::size_t>(N));
      check_cublas(cublasSetStream(cublas, stream), "snk restore stream");
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha,
                             dB, N,
                             dA, M,
                             &beta, dC, N),
                 "snk cublasDgemm TN");
    check_cublas(cublasSetStream(cublas, stream), "snk restore stream");
  }
};

struct CudaSnkImpl {
  static constexpr int kMaxStreams = 4;
  static constexpr std::size_t kEspChunkMaxPoints = 4096;

  DevFlatBasis basis;
  SnkOptions opts;
  std::size_t npairs_total = 0;

  // Partitioned shell-pair classes for L = 0..4 and L >= 5
  std::array<DevBuf<SnkShellPair>, 5> pairs_by_l;
  std::array<std::size_t, 5> npairs_by_l{};
  DevBuf<SnkShellPair> pairs_high;
  std::size_t npairs_high = 0;

  DevBuf<SnkPrimPair> prims;
  DevBuf<double> h_coeffs;

  DevBuf<double> coords;
  DevBuf<double> weights;
  std::vector<SnkBatchGpuMeta> batches;
  std::vector<SnkEspChunkMeta> chunks;
  DevBuf<double> tile_boxes;
  DevBuf<int> act_shells;
  DevBuf<int> act_shell_local_ao;
  DevBuf<int> act_aos;

  // Persistent GPU cache of compacted batch Phi and weighted w * Phi
  bool ao_cached = false;
  DevBuf<double> phi_cache;
  DevBuf<double> phi_w_cache;

  DevBuf<double> q_fit;

  int num_workers = 0;
  std::array<SnkStreamWorker, kMaxStreams> workers;
  cudaEvent_t fork_event = nullptr;

  std::size_t max_na = 0;
  std::size_t max_ng = 0;
  std::size_t max_chunk_ng = 0;
  std::size_t max_chunk_tiles = 0;

  DevBuf<double> dm_sym;
  DevBuf<double> k_raw;
  DevBuf<double> qtk;
  DevBuf<double> dm_host_buf;
  DevBuf<double> vk_host_buf;

  CudaSnkImpl(const detail::FlatBasisData& flat_b,
              const MolecularGrid& grid,
              const SnkPairTable& pair_table,
              const std::vector<double>& s_exact,
              const SnkOptions& options,
              std::vector<double>& q_fit_out)
      : basis(flat_b),
        opts(options),
        npairs_total(pair_table.pairs.size()),
        prims(pair_table.prims),
        h_coeffs(pair_table.h_coeffs),
        coords(grid.coords()),
        weights(grid.weights()) {
    check_cuda(cudaEventCreateWithFlags(&fork_event, cudaEventDisableTiming),
               "snk fork_event create");

    // Partition shell pairs by L = la + lb angular class (0..4 vs >= 5)
    std::array<std::vector<SnkShellPair>, 5> host_l;
    std::vector<SnkShellPair> host_high;
    for (const SnkShellPair& sp : pair_table.pairs) {
      const int L = sp.la + sp.lb;
      if (L <= 4) {
        host_l[static_cast<std::size_t>(L)].push_back(sp);
      } else {
        host_high.push_back(sp);
      }
    }
    for (std::size_t c = 0; c < 5; ++c) {
      npairs_by_l[c] = host_l[c].size();
      if (npairs_by_l[c] > 0) pairs_by_l[c].upload(host_l[c].data(), npairs_by_l[c]);
    }
    npairs_high = host_high.size();
    if (npairs_high > 0) pairs_high.upload(host_high.data(), npairs_high);

    // Coalesce spatial batches into ~2048-pt GPU super-batches
    const std::vector<detail::SuperBatch> super_batches =
        detail::build_super_batches(grid, flat_b, 2048, 1.45);

    std::size_t total_phi_elems = 0;
    std::vector<int> h_act_shells;
    std::vector<int> h_act_shell_local_ao;
    std::vector<int> h_act_aos;
    batches.reserve(super_batches.size());

    for (const detail::SuperBatch& sb : super_batches) {
      if (sb.npoints == 0 || sb.active_aos.empty()) continue;
      SnkBatchGpuMeta m;
      m.point_offset = sb.point_offset;
      m.ng = sb.npoints;
      m.na = sb.active_aos.size();
      m.n_act_shells = static_cast<int>(sb.active_shells.size());
      m.shell_list_offset = h_act_shells.size();
      m.ao_list_offset = h_act_aos.size();
      m.phi_cache_offset = total_phi_elems;
      total_phi_elems += m.na * m.ng;

      int local_cursor = 0;
      for (int is : sb.active_shells) {
        h_act_shells.push_back(is);
        h_act_shell_local_ao.push_back(local_cursor);
        local_cursor += 2 * flat_b.shell_l[static_cast<std::size_t>(is)] + 1;
      }
      for (int ao : sb.active_aos) h_act_aos.push_back(ao);

      max_na = std::max(max_na, m.na);
      max_ng = std::max(max_ng, m.ng);
      batches.push_back(m);
    }

    act_shells.upload(h_act_shells.data(), h_act_shells.size());
    act_shell_local_ao.upload(h_act_shell_local_ao.data(), h_act_shell_local_ao.size());
    act_aos.upload(h_act_aos.data(), h_act_aos.size());

    // Group contiguous super-batches into ~4096-pt ESP contraction chunks and precompute
    // 128-point tile bounding boxes ONCE on Host!
    const auto& h_coords = grid.coords();
    std::vector<double> h_tile_boxes;
    for (std::size_t ib = 0; ib < batches.size();) {
      SnkEspChunkMeta ch;
      ch.batch_begin = ib;
      ch.point_offset = batches[ib].point_offset;
      ch.ng = batches[ib].ng;
      std::size_t jb = ib + 1;
      while (jb < batches.size() &&
             batches[jb].point_offset == ch.point_offset + ch.ng &&
             ch.ng + batches[jb].ng <= kEspChunkMaxPoints) {
        ch.ng += batches[jb].ng;
        ++jb;
      }
      ch.batch_end = jb;
      ch.n_tiles = (ch.ng + 127) / 128;
      ch.tile_box_offset = h_tile_boxes.size();

      for (std::size_t t = 0; t < ch.n_tiles; ++t) {
        const std::size_t g0 = ch.point_offset + t * 128;
        const std::size_t g1 = std::min(ch.point_offset + ch.ng, g0 + 128);
        double bmin[3] = {
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
        };
        double bmax[3] = {
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity(),
        };
        for (std::size_t gp = g0; gp < g1; ++gp) {
          for (int d = 0; d < 3; ++d) {
            const double v = h_coords[3 * gp + d];
            bmin[d] = std::min(bmin[d], v);
            bmax[d] = std::max(bmax[d], v);
          }
        }
        for (int d = 0; d < 3; ++d) h_tile_boxes.push_back(bmin[d]);
        for (int d = 0; d < 3; ++d) h_tile_boxes.push_back(bmax[d]);
      }

      max_chunk_ng = std::max(max_chunk_ng, ch.ng);
      max_chunk_tiles = std::max(max_chunk_tiles, ch.n_tiles);
      chunks.push_back(ch);
      ib = jb;
    }
    if (!h_tile_boxes.empty()) {
      tile_boxes.upload(h_tile_boxes.data(), h_tile_boxes.size());
    }

    const std::size_t nao = static_cast<std::size_t>(flat_b.nao);
    const std::size_t nao2 = nao * nao;
    num_workers = static_cast<int>(std::min<std::size_t>(
        kMaxStreams, std::max<std::size_t>(1, chunks.size())));

    // Check VRAM budget for persistent Phi + Phi_w cache (2 * total_phi_elems doubles)
    if (opts.cache_ao_on_device && total_phi_elems > 0 && max_na > 0 && max_ng > 0) {
      std::size_t free_bytes = 0, total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        (void)cudaGetLastError();
        free_bytes = opts.max_ao_cache_mb * (1024ULL * 1024ULL) * 2ULL;
      }
      const std::size_t user_cap = opts.max_ao_cache_mb * (1024ULL * 1024ULL);
      const std::size_t safe_free =
          (free_bytes > (512ULL << 20)) ? ((free_bytes * 3ULL) / 4ULL) : 0ULL;
      if (2ULL * total_phi_elems * sizeof(double) <= std::min(user_cap, safe_free)) {
        phi_cache.resize(total_phi_elems);
        phi_w_cache.resize(total_phi_elems);
        ao_cached = true;
      }
    }

    for (int s = 0; s < num_workers; ++s) {
      workers[s].init_handles();
      if (max_na > 0 && max_ng > 0 && nao > 0) {
        if (!ao_cached) {
          workers[s].phi_scratch.resize(max_na * max_ng);
          workers[s].phi_w_scratch.resize(max_na * max_ng);
        }
        workers[s].d_cols.resize(nao * max_na);
        workers[s].f_chunk.resize(nao * max_chunk_ng);
        workers[s].g_chunk.resize(nao * max_chunk_ng);
        workers[s].f_shell_max.resize(
            static_cast<std::size_t>(flat_b.nshell) * max_chunk_tiles);
        workers[s].k_batch.resize(max_na * nao);
        if (num_workers >= 2) {
          workers[s].k_raw_part.resize(nao2);
        }
        if (opts.precision == Precision::FP32) {
          const std::size_t max_sz = std::max({nao * max_na, nao * max_ng, nao2});
          workers[s].f32_a.resize(max_sz);
          workers[s].f32_b.resize(max_sz);
          workers[s].f32_c.resize(max_sz);
        }
      }
    }

    if (nao2 > 0) {
      dm_sym.resize(nao2);
      k_raw.resize(nao2);
      qtk.resize(nao2);
      dm_host_buf.resize(nao2);
      vk_host_buf.resize(nao2);
    }

    // Precompute cached Phi & Phi_w ONCE across worker streams
    if (ao_cached) {
      const detail::FlatBasisView bview = basis.view();
      for (std::size_t ib = 0; ib < batches.size(); ++ib) {
        const SnkBatchGpuMeta& b = batches[ib];
        cudaStream_t ws_stream = workers[ib % static_cast<std::size_t>(num_workers)].stream;
        double* d_phi = phi_cache.data() + b.phi_cache_offset;
        double* d_phi_w = phi_w_cache.data() + b.phi_cache_offset;

        const dim3 ao_grid(static_cast<unsigned>((b.ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        snk_eval_active_ao_kernel<<<ao_grid, 128, 0, ws_stream>>>(
            bview, coords.data(), b.point_offset, b.ng, b.na,
            act_shells.data() + b.shell_list_offset,
            act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, d_phi);

        const dim3 w_grid(static_cast<unsigned>((b.ng + 127) / 128),
                          static_cast<unsigned>(b.na), 1);
        snk_weight_phi_2d_kernel<<<w_grid, 128, 0, ws_stream>>>(
            b.na, b.ng, weights.data(), b.point_offset, d_phi, d_phi_w);
      }
      for (int s = 0; s < num_workers; ++s) {
        check_cuda(cudaStreamSynchronize(workers[s].stream), "snk ao_cache init sync");
      }
    }

    // Build COSX overlap-fitting matrix Q = S_grid^+ * S_exact on GPU reusing cached Phi & Phi_w!
    if (opts.overlap_fitting && nao > 0 && !batches.empty()) {
      SnkStreamWorker& ws0 = workers[0];
      check_cublas(cublasSetStream(ws0.cublas, nullptr), "snk cublas default stream");
      DevBuf<double> d_sg(nao2);
      check_cuda(cudaMemset(d_sg.data(), 0, nao2 * sizeof(double)), "snk zero d_sg");
      DevBuf<double> d_sg_act(max_na * max_na);
      const double one = 1.0, zero = 0.0;

      for (const SnkBatchGpuMeta& b : batches) {
        const double* d_phi = nullptr;
        const double* d_phi_w = nullptr;
        if (ao_cached) {
          d_phi = phi_cache.data() + b.phi_cache_offset;
          d_phi_w = phi_w_cache.data() + b.phi_cache_offset;
        } else {
          const dim3 ao_grid(static_cast<unsigned>((b.ng + 127) / 128),
                             static_cast<unsigned>(b.n_act_shells), 1);
          snk_eval_active_ao_kernel<<<ao_grid, 128>>>(
              basis.view(), coords.data(), b.point_offset, b.ng, b.na,
              act_shells.data() + b.shell_list_offset,
              act_shell_local_ao.data() + b.shell_list_offset,
              b.n_act_shells, ws0.phi_scratch.data());
          const dim3 w_grid(static_cast<unsigned>((b.ng + 127) / 128),
                            static_cast<unsigned>(b.na), 1);
          snk_weight_phi_2d_kernel<<<w_grid, 128>>>(
              b.na, b.ng, weights.data(), b.point_offset,
              ws0.phi_scratch.data(), ws0.phi_w_scratch.data());
          d_phi = ws0.phi_scratch.data();
          d_phi_w = ws0.phi_w_scratch.data();
        }

        const int i_na = static_cast<int>(b.na);
        const int i_ng = static_cast<int>(b.ng);
        check_cublas(cublasDgemm(ws0.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 i_na, i_na, i_ng, &one,
                                 d_phi_w, i_ng,
                                 d_phi, i_ng,
                                 &zero, d_sg_act.data(), i_na),
                     "cublasDgemm S_grid");

        snk_scatter_submatrix_kernel<<<static_cast<unsigned>((b.na * b.na + 255) / 256), 256>>>(
            d_sg_act.data(), act_aos.data() + b.ao_list_offset, b.na, nao, d_sg.data());
      }

      cusolverDnHandle_t cusolver = nullptr;
      check_cusolver(cusolverDnCreate(&cusolver), "cusolverDnCreate (SNK)");
      const int i_nao = static_cast<int>(nao);
      DevBuf<double> d_eval(nao);
      int lwork = 0;
      check_cusolver(cusolverDnDsyevd_bufferSize(
                         cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                         i_nao, d_sg.data(), i_nao, d_eval.data(), &lwork),
                     "cusolverDnDsyevd_bufferSize");
      DevBuf<double> d_work(static_cast<std::size_t>(std::max(1, lwork)));
      DevBuf<int> d_info(1);
      check_cusolver(cusolverDnDsyevd(
                         cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                         i_nao, d_sg.data(), i_nao, d_eval.data(),
                         d_work.data(), lwork, d_info.data()),
                     "cusolverDnDsyevd");
      cusolverDnDestroy(cusolver);

      DevBuf<double> d_pinv(nao2);
      DevBuf<double> d_null(nao2);
      snk_build_pinv_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256>>>(
          d_sg.data(), d_eval.data(), i_nao, opts.lindep_tol, d_pinv.data(), d_null.data());

      DevBuf<double> d_sexact(s_exact);
      q_fit.resize(nao2);
      check_cublas(cublasDgemm(ws0.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               i_nao, i_nao, i_nao, &one,
                               d_sexact.data(), i_nao,
                               d_pinv.data(), i_nao,
                               &zero, q_fit.data(), i_nao),
                   "cublasDgemm Q_fit");
      snk_add_inplace_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256>>>(
          d_null.data(), q_fit.data(), nao2);
      q_fit.download(q_fit_out.data(), nao2);
      check_cuda(cudaDeviceSynchronize(), "snk Q_fit init sync");
      check_cublas(cublasSetStream(ws0.cublas, ws0.stream), "snk restore ws0 stream");
    } else if (nao2 > 0) {
      q_fit.upload(q_fit_out.data(), nao2);
      check_cuda(cudaDeviceSynchronize(), "snk Q_fit identity upload sync");
    }
  }

  ~CudaSnkImpl() {
    if (fork_event != nullptr) cudaEventDestroy(fork_event);
  }
};

template <int L>
void launch_esp_l_group(const CudaSnkImpl* st,
                        const SnkEspChunkMeta& ch,
                        SnkStreamWorker& ws) {
  const std::size_t np = st->npairs_by_l[static_cast<std::size_t>(L)];
  if (np == 0) return;
  const dim3 grid(static_cast<unsigned>(ch.n_tiles),
                  static_cast<unsigned>(np), 1);
  snk_esp_contract_l_kernel<L><<<grid, 128, 0, ws.stream>>>(
      st->pairs_by_l[static_cast<std::size_t>(L)].data(), np,
      st->prims.data(), st->h_coeffs.data(),
      st->coords.data(), ch.point_offset, ch.ng, ch.n_tiles,
      st->tile_boxes.data() + ch.tile_box_offset,
      ws.f_shell_max.data(), ws.f_chunk.data(), ws.g_chunk.data());
}

}  // namespace

int snk_cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

void snk_esp_matrix_cuda(const detail::FlatBasisData& flat_basis,
                         const std::vector<SnkShellPair>& pairs,
                         double gx, double gy, double gz,
                         double pair_cutoff,
                         double* out_v) {
  const std::size_t nao2 = static_cast<std::size_t>(flat_basis.nao) * flat_basis.nao;
  if (nao2 == 0 || pairs.empty()) return;

  DevFlatBasis d_basis(flat_basis);
  DevBuf<SnkShellPair> d_pairs(pairs);
  DevBuf<double> d_v(nao2);
  check_cuda(cudaMemset(d_v.data(), 0, nao2 * sizeof(double)), "snk_esp_matrix memset");

  const unsigned blocks = static_cast<unsigned>((pairs.size() + 127) / 128);
  esp_single_point_kernel<<<blocks, 128>>>(
      d_basis.view(), d_pairs.data(), pairs.size(),
      gx, gy, gz, pair_cutoff, d_v.data());
  check_cuda(cudaGetLastError(), "esp_single_point_kernel launch");
  d_v.download(out_v, nao2);
  check_cuda(cudaDeviceSynchronize(), "snk_esp_matrix_cuda sync");
}

void* snk_cuda_create(const detail::FlatBasisData& flat_basis,
                      const MolecularGrid& grid,
                      const SnkPairTable& pair_table,
                      const std::vector<double>& s_exact,
                      const SnkOptions& opts,
                      std::vector<double>& q_fit_out) {
  return new CudaSnkImpl(flat_basis, grid, pair_table, s_exact, opts, q_fit_out);
}

void snk_cuda_destroy(void* impl) noexcept {
  delete static_cast<CudaSnkImpl*>(impl);
}

void snk_cuda_compute_k_device(void* raw_impl,
                               const double* d_dm,
                               double* d_vk,
                               void* cuda_stream) {
  auto* st = static_cast<CudaSnkImpl*>(raw_impl);
  cudaStream_t caller_stream = static_cast<cudaStream_t>(cuda_stream);

  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const std::size_t nao2 = nao * nao;
  if (nao == 0) return;

  if (st->chunks.empty() || st->npairs_total == 0) {
    check_cuda(cudaMemsetAsync(d_vk, 0, nao2 * sizeof(double), caller_stream), "snk zero d_vk");
    check_cuda(cudaStreamSynchronize(caller_stream), "snk empty sync");
    return;
  }

  snk_symmetrize_dm_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
      d_dm, nao, st->dm_sym.data());

  const int n_streams = st->num_workers;
  const int i_nao = static_cast<int>(nao);
  const detail::FlatBasisView bview = st->basis.view();

  check_cuda(cudaEventRecord(st->fork_event, caller_stream), "snk fork record");
  for (int s = 0; s < n_streams; ++s) {
    SnkStreamWorker& ws = st->workers[s];
    check_cuda(cudaStreamWaitEvent(ws.stream, st->fork_event, 0), "snk fork wait");
    double* target_k = (n_streams == 1) ? st->k_raw.data() : ws.k_raw_part.data();
    check_cuda(cudaMemsetAsync(target_k, 0, nao2 * sizeof(double), ws.stream),
               "snk zero target_k");
  }

  for (std::size_t ic = 0; ic < st->chunks.size(); ++ic) {
    const int s = static_cast<int>(ic % static_cast<std::size_t>(n_streams));
    SnkStreamWorker& ws = st->workers[s];
    const SnkEspChunkMeta& ch = st->chunks[ic];
    const int i_chunk_ng = static_cast<int>(ch.ng);

    // 1. Half-transform D_cols * Phi_b directly into contiguous chunk columns of `ws.f_chunk`
    for (std::size_t ib = ch.batch_begin; ib < ch.batch_end; ++ib) {
      const SnkBatchGpuMeta& b = st->batches[ib];
      const std::size_t local_off = b.point_offset - ch.point_offset;
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const double* d_phi = nullptr;
      if (st->ao_cached) {
        d_phi = st->phi_cache.data() + b.phi_cache_offset;
      } else {
        const dim3 ao_grid(static_cast<unsigned>((b.ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        snk_eval_active_ao_kernel<<<ao_grid, 128, 0, ws.stream>>>(
            bview, st->coords.data(), b.point_offset, b.ng, b.na,
            st->act_shells.data() + b.shell_list_offset,
            st->act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, ws.phi_scratch.data());
        d_phi = ws.phi_scratch.data();
      }

      snk_gather_dm_cols_kernel<<<static_cast<unsigned>((nao * b.na + 255) / 256), 256, 0, ws.stream>>>(
          st->dm_sym.data(), nao, d_batch_aos, b.na, ws.d_cols.data());

      ws.gemm_nn_strided_c(i_nao, static_cast<int>(b.ng), static_cast<int>(b.na),
                           ws.d_cols.data(), d_phi,
                           ws.f_chunk.data() + local_off, i_chunk_ng,
                           st->opts.precision);
    }

    // 2. Per-tile shell |F| bounds + zero-spill (LA, LB)-specialized 1e ESP contraction
    check_cuda(cudaMemsetAsync(ws.g_chunk.data(), 0, nao * ch.ng * sizeof(double), ws.stream),
               "snk zero g_chunk");

    const dim3 fmax_grid(static_cast<unsigned>(ch.n_tiles),
                         static_cast<unsigned>(st->basis.nshell), 1);
    snk_compute_f_shell_max_kernel<<<fmax_grid, 128, 0, ws.stream>>>(
        bview, ch.ng, ch.n_tiles, ws.f_chunk.data(), ws.f_shell_max.data());

    launch_esp_l_group<0>(st, ch, ws);
    launch_esp_l_group<1>(st, ch, ws);
    launch_esp_l_group<2>(st, ch, ws);
    launch_esp_l_group<3>(st, ch, ws);
    launch_esp_l_group<4>(st, ch, ws);

    if (st->npairs_high > 0) {
      const dim3 high_grid(static_cast<unsigned>(ch.n_tiles),
                           static_cast<unsigned>(st->npairs_high), 1);
      snk_esp_contract_high_l_kernel<<<high_grid, 128, 0, ws.stream>>>(
          bview, st->pairs_high.data(), st->npairs_high, st->prims.data(),
          st->coords.data(), ch.point_offset, ch.ng, ch.n_tiles,
          st->tile_boxes.data() + ch.tile_box_offset,
          ws.f_shell_max.data(), ws.f_chunk.data(), ws.g_chunk.data());
    }

    // 3. Second half-transform K_batch = Phi_w_b * G_sub^T + scatter into stream's k_raw_part
    double* target_k = (n_streams == 1) ? st->k_raw.data() : ws.k_raw_part.data();
    for (std::size_t ib = ch.batch_begin; ib < ch.batch_end; ++ib) {
      const SnkBatchGpuMeta& b = st->batches[ib];
      const std::size_t local_off = b.point_offset - ch.point_offset;
      const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

      const double* d_phi_w = nullptr;
      if (st->ao_cached) {
        d_phi_w = st->phi_w_cache.data() + b.phi_cache_offset;
      } else {
        const dim3 ao_grid(static_cast<unsigned>((b.ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        snk_eval_active_ao_kernel<<<ao_grid, 128, 0, ws.stream>>>(
            bview, st->coords.data(), b.point_offset, b.ng, b.na,
            st->act_shells.data() + b.shell_list_offset,
            st->act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, ws.phi_scratch.data());
        const dim3 w_grid(static_cast<unsigned>((b.ng + 127) / 128),
                          static_cast<unsigned>(b.na), 1);
        snk_weight_phi_2d_kernel<<<w_grid, 128, 0, ws.stream>>>(
            b.na, b.ng, st->weights.data(), b.point_offset,
            ws.phi_scratch.data(), ws.phi_w_scratch.data());
        d_phi_w = ws.phi_w_scratch.data();
      }

      ws.gemm_nt_strided_b(static_cast<int>(b.na), i_nao, static_cast<int>(b.ng),
                           d_phi_w, ws.g_chunk.data() + local_off, i_chunk_ng,
                           ws.k_batch.data(), st->opts.precision);

      snk_scatter_rows_kernel<<<static_cast<unsigned>((b.na * nao + 255) / 256), 256, 0, ws.stream>>>(
          ws.k_batch.data(), d_batch_aos, b.na, nao, target_k);
    }
  }

  for (int s = 0; s < n_streams; ++s) {
    check_cuda(cudaEventRecord(st->workers[s].done_event, st->workers[s].stream),
               "snk join record");
    check_cuda(cudaStreamWaitEvent(caller_stream, st->workers[s].done_event, 0),
               "snk join wait");
  }

  if (n_streams >= 2) {
    snk_reduce_stream_k_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        nao2, n_streams,
        st->workers[0].k_raw_part.data(),
        st->workers[1].k_raw_part.data(),
        (n_streams >= 3) ? st->workers[2].k_raw_part.data() : nullptr,
        (n_streams >= 4) ? st->workers[3].k_raw_part.data() : nullptr,
        st->k_raw.data());
  }

  if (st->opts.overlap_fitting) {
    st->workers[0].gemm_tn(i_nao, i_nao, i_nao,
                           st->q_fit.data(), st->k_raw.data(), st->qtk.data(),
                           st->opts.precision, caller_stream);
    snk_symmetrize_dm_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        st->qtk.data(), nao, d_vk);
  } else {
    snk_symmetrize_dm_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, caller_stream>>>(
        st->k_raw.data(), nao, d_vk);
  }
  check_cuda(cudaStreamSynchronize(caller_stream), "snk_cuda_compute_k_device sync");
}

void snk_cuda_compute_k_host(void* raw_impl,
                             const double* h_dm,
                             double* h_vk) {
  auto* st = static_cast<CudaSnkImpl*>(raw_impl);
  const std::size_t nao2 = static_cast<std::size_t>(st->basis.nao) * st->basis.nao;
  st->dm_host_buf.upload(h_dm, nao2);
  snk_cuda_compute_k_device(raw_impl, st->dm_host_buf.data(), st->vk_host_buf.data(), nullptr);
  st->vk_host_buf.download(h_vk, nao2);
  check_cuda(cudaDeviceSynchronize(), "snk_cuda_compute_k_host download sync");
}

}  // namespace gpu4dft::snk_detail
