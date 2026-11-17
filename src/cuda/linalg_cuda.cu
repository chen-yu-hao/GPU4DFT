#include "gpu4dft/linalg.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu4dft {

namespace {

inline void check_cuda(cudaError_t status, const char* context) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, const char* context) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void check_cusolver(cusolverStatus_t status, const char* context) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(context) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

constexpr std::size_t kCublasWorkspaceBytes = 32ULL * 1024ULL * 1024ULL;
constexpr int kMaxOzakiKChunk = 32768;

// ---------------------------------------------------------------------------
// Custom CUDA Kernels for Matrix Utilities, FP32 Cast, Ozaki INT8, and SP2
// ---------------------------------------------------------------------------

__global__ void scale_or_zero_rect_kernel(int m, int n, double beta, double* C, int ldc) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(m) * n;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  if (beta == 0.0) {
    C[static_cast<std::size_t>(i) * ldc + j] = 0.0;
  } else {
    C[static_cast<std::size_t>(i) * ldc + j] *= beta;
  }
}

__global__ void extract_op_to_fp32_kernel(int rows, int cols, const double* src, int ld_src,
                                          bool trans, float* dst_rowmajor) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / cols);
  const int j = static_cast<int>(idx % cols);
  const double v = trans ? src[static_cast<std::size_t>(j) * ld_src + i]
                         : src[static_cast<std::size_t>(i) * ld_src + j];
  dst_rowmajor[idx] = static_cast<float>(v);
}

__global__ void epilogue_fp32_to_fp64_kernel(int m, int n, double alpha, const float* src_rowmajor,
                                             double beta, double* dst, int ldc) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(m) * n;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  const double val = alpha * static_cast<double>(src_rowmajor[idx]);
  double& out = dst[static_cast<std::size_t>(i) * ldc + j];
  out = (beta == 0.0) ? val : (val + beta * out);
}

// Ozaki INT8 row-wise max-exponent & mantissa slicing for op(A)[m x kc] -> [S, mp, kp].
__global__ void ozaki_slice_rows_kernel(int m, int kc, int k_offset, int mp, int kp, int num_slices,
                                        const double* A, int lda, bool transA, int* exp_A,
                                        std::int8_t* A_slices) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= mp) return;

  __shared__ double s_max[256];
  __shared__ int s_exp;
  __shared__ int s_nonzero;

  if (row >= m) {
    if (threadIdx.x == 0) exp_A[row] = 0;
    for (int p = static_cast<int>(threadIdx.x); p < kp; p += static_cast<int>(blockDim.x)) {
      for (int s = 0; s < num_slices; ++s) {
        A_slices[(static_cast<std::size_t>(s) * mp + row) * kp + p] = 0;
      }
    }
    return;
  }

  double local_max = 0.0;
  for (int p = static_cast<int>(threadIdx.x); p < kc; p += static_cast<int>(blockDim.x)) {
    const int k_idx = k_offset + p;
    const double val = transA ? A[static_cast<std::size_t>(k_idx) * lda + row]
                              : A[static_cast<std::size_t>(row) * lda + k_idx];
    local_max = fmax(local_max, fabs(val));
  }
  s_max[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double amax = s_max[0];
    int e = 0;
    if (amax > 0.0) {
      (void)frexp(amax, &e);
      if (ldexp(amax, 7 - e) >= 127.5) {
        e += 1;
      }
      s_nonzero = 1;
    } else {
      s_nonzero = 0;
    }
    s_exp = e;
    exp_A[row] = e;
  }
  __syncthreads();

  const int e = s_exp;
  const bool nonzero = (s_nonzero != 0);

  for (int p = static_cast<int>(threadIdx.x); p < kp; p += static_cast<int>(blockDim.x)) {
    if (!nonzero || p >= kc) {
      for (int s = 0; s < num_slices; ++s) {
        A_slices[(static_cast<std::size_t>(s) * mp + row) * kp + p] = 0;
      }
    } else {
      const int k_idx = k_offset + p;
      const double val = transA ? A[static_cast<std::size_t>(k_idx) * lda + row]
                                : A[static_cast<std::size_t>(row) * lda + k_idx];
      double rem = ldexp(val, -e);
      for (int s = 1; s <= num_slices; ++s) {
        const double q_d = rint(ldexp(rem, 7 * s));
        int qi = static_cast<int>(q_d);
        qi = max(-127, min(127, qi));
        A_slices[(static_cast<std::size_t>(s - 1) * mp + row) * kp + p] =
            static_cast<std::int8_t>(qi);
        rem -= ldexp(static_cast<double>(qi), -7 * s);
      }
    }
  }
}

// Ozaki INT8 column-wise max-exponent & mantissa slicing for op(B)[kc x n],
// storing transposed slices [S, np, kp] so column j of op(B) is contiguous along kp!
__global__ void ozaki_slice_cols_kernel(int n, int kc, int k_offset, int np, int kp, int num_slices,
                                        const double* B, int ldb, bool transB, int* exp_B,
                                        std::int8_t* B_slices_T) {
  const int col = static_cast<int>(blockIdx.x);
  if (col >= np) return;

  __shared__ double s_max[256];
  __shared__ int s_exp;
  __shared__ int s_nonzero;

  if (col >= n) {
    if (threadIdx.x == 0) exp_B[col] = 0;
    for (int p = static_cast<int>(threadIdx.x); p < kp; p += static_cast<int>(blockDim.x)) {
      for (int t = 0; t < num_slices; ++t) {
        B_slices_T[(static_cast<std::size_t>(t) * np + col) * kp + p] = 0;
      }
    }
    return;
  }

  double local_max = 0.0;
  for (int p = static_cast<int>(threadIdx.x); p < kc; p += static_cast<int>(blockDim.x)) {
    const int k_idx = k_offset + p;
    const double val = transB ? B[static_cast<std::size_t>(col) * ldb + k_idx]
                              : B[static_cast<std::size_t>(k_idx) * ldb + col];
    local_max = fmax(local_max, fabs(val));
  }
  s_max[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double bmax = s_max[0];
    int e = 0;
    if (bmax > 0.0) {
      (void)frexp(bmax, &e);
      if (ldexp(bmax, 7 - e) >= 127.5) {
        e += 1;
      }
      s_nonzero = 1;
    } else {
      s_nonzero = 0;
    }
    s_exp = e;
    exp_B[col] = e;
  }
  __syncthreads();

  const int e = s_exp;
  const bool nonzero = (s_nonzero != 0);

  for (int p = static_cast<int>(threadIdx.x); p < kp; p += static_cast<int>(blockDim.x)) {
    if (!nonzero || p >= kc) {
      for (int t = 0; t < num_slices; ++t) {
        B_slices_T[(static_cast<std::size_t>(t) * np + col) * kp + p] = 0;
      }
    } else {
      const int k_idx = k_offset + p;
      const double val = transB ? B[static_cast<std::size_t>(col) * ldb + k_idx]
                                : B[static_cast<std::size_t>(k_idx) * ldb + col];
      double rem = ldexp(val, -e);
      for (int t = 1; t <= num_slices; ++t) {
        const double q_d = rint(ldexp(rem, 7 * t));
        int qi = static_cast<int>(q_d);
        qi = max(-127, min(127, qi));
        B_slices_T[(static_cast<std::size_t>(t - 1) * np + col) * kp + p] =
            static_cast<std::int8_t>(qi);
        rem -= ldexp(static_cast<double>(qi), -7 * t);
      }
    }
  }
}

__global__ void ozaki_accum_diag_kernel(int m, int n, int np, const std::int32_t* C_i32,
                                        const int* exp_A, const int* exp_B, int bit_shift,
                                        double* C_acc) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(m) * n;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  const std::int32_t c_val = C_i32[static_cast<std::size_t>(i) * np + j];
  if (c_val != 0) {
    C_acc[idx] += ldexp(static_cast<double>(c_val), exp_A[i] + exp_B[j] - bit_shift);
  }
}

__global__ void ozaki_epilogue_kernel(int m, int n, double alpha, const double* C_acc, double beta,
                                      double* C, int ldc) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(m) * n;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  const double val = alpha * C_acc[idx];
  double& out = C[static_cast<std::size_t>(i) * ldc + j];
  out = (beta == 0.0) ? val : (val + beta * out);
}

__global__ void transpose_square_kernel(int n, const double* src, double* dst) {
  __shared__ double tile[32][33];
  const int x = static_cast<int>(blockIdx.x) * 32 + static_cast<int>(threadIdx.x);
  const int y = static_cast<int>(blockIdx.y) * 32 + static_cast<int>(threadIdx.y);
  if (x < n && y < n) {
    tile[threadIdx.y][threadIdx.x] = src[static_cast<std::size_t>(y) * n + x];
  }
  __syncthreads();
  const int tx = static_cast<int>(blockIdx.y) * 32 + static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(blockIdx.x) * 32 + static_cast<int>(threadIdx.y);
  if (tx < n && ty < n) {
    dst[static_cast<std::size_t>(ty) * n + tx] = tile[threadIdx.x][threadIdx.y];
  }
}

__global__ void symmetrize_inplace_kernel(int n, double* A) {
  const int i = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
                static_cast<int>(threadIdx.y);
  const int j = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (i < n && j < n && i < j) {
    const std::size_t ij = static_cast<std::size_t>(i) * n + j;
    const std::size_t ji = static_cast<std::size_t>(j) * n + i;
    const double avg = 0.5 * (A[ij] + A[ji]);
    A[ij] = avg;
    A[ji] = avg;
  }
}

__global__ void slice_and_scale_cols_kernel(int rows, int total_cols, int occ_cols,
                                            const double* C_full, const double* col_scales,
                                            double* C_occ_out) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(rows) * occ_cols;
  if (idx >= total) return;
  const int r = static_cast<int>(idx / occ_cols);
  const int c = static_cast<int>(idx % occ_cols);
  const double s = (col_scales != nullptr) ? col_scales[c] : 1.0;
  C_occ_out[idx] = C_full[static_cast<std::size_t>(r) * total_cols + c] * s;
}

__global__ void axpby_kernel(std::size_t count, double alpha, const double* x, double beta,
                             double* y) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  const double xv = (x != nullptr) ? (alpha * x[idx]) : 0.0;
  y[idx] = (beta == 0.0) ? xv : (xv + beta * y[idx]);
}

__global__ void delta_and_norm_kernel(std::size_t count, const double* d_new, const double* d_old,
                                      double* d_delta, double* d_stats) {
  __shared__ double s_sq[256];
  __shared__ double s_max[256];

  double local_sq = 0.0;
  double local_max = 0.0;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < count; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const double diff = d_new[idx] - d_old[idx];
    if (d_delta != nullptr) {
      d_delta[idx] = diff;
    }
    local_sq += diff * diff;
    local_max = fmax(local_max, fabs(diff));
  }
  s_sq[threadIdx.x] = local_sq;
  s_max[threadIdx.x] = local_max;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_sq[threadIdx.x] += s_sq[threadIdx.x + stride];
      s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && d_stats != nullptr) {
    atomicAdd(&d_stats[0], s_sq[0]);
    // Positive IEEE-754 double bit-pattern is monotonic with magnitude; atomicMax via uint64 CAS.
    auto* max_ull = reinterpret_cast<unsigned long long*>(&d_stats[1]);
    unsigned long long old_bits = *max_ull;
    while (__ull2double_rn(0) >= 0.0) {  // loop until CAS succeeds
      const double cur_val = __longlong_as_double(static_cast<long long>(old_bits));
      if (cur_val >= s_max[0]) break;
      const unsigned long long desired =
          static_cast<unsigned long long>(__double_as_longlong(s_max[0]));
      const unsigned long long prev = atomicCAS(max_ull, old_bits, desired);
      if (prev == old_bits) break;
      old_bits = prev;
    }
  }
}

__global__ void assemble_fock_and_traces_kernel(std::size_t count, const double* D,
                                                const double* H, const double* J, const double* K,
                                                double alpha_x, const double* Vxc, double* F_out,
                                                double* traces3_out) {
  __shared__ double s_trH[256];
  __shared__ double s_trJ[256];
  __shared__ double s_trK[256];

  double trH = 0.0;
  double trJ = 0.0;
  double trK = 0.0;

  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < count; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const double d = D[idx];
    const double h = H[idx];
    const double j = (J != nullptr) ? J[idx] : 0.0;
    const double k = (K != nullptr && alpha_x != 0.0) ? K[idx] : 0.0;
    const double vxc = (Vxc != nullptr) ? Vxc[idx] : 0.0;

    F_out[idx] = h + j - 0.5 * alpha_x * k + vxc;
    trH += d * h;
    trJ += d * j;
    trK += d * k;
  }

  s_trH[threadIdx.x] = trH;
  s_trJ[threadIdx.x] = trJ;
  s_trK[threadIdx.x] = trK;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_trH[threadIdx.x] += s_trH[threadIdx.x + stride];
      s_trJ[threadIdx.x] += s_trJ[threadIdx.x + stride];
      s_trK[threadIdx.x] += s_trK[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && traces3_out != nullptr) {
    atomicAdd(&traces3_out[0], s_trH[0]);
    atomicAdd(&traces3_out[1], s_trJ[0]);
    atomicAdd(&traces3_out[2], s_trK[0]);
  }
}

// ---------------------------------------------------------------------------
// SP2 Density Matrix Purification Kernels (100% CUDA Graph Capturable!)
// ---------------------------------------------------------------------------

struct Sp2DeviceCtrl {
  double trace = 0.0;
  double gamma = 0.0;
  double prev_gamma = 1e300;
  int mode = 0;       // 0 = converged no-op, 1 = P^2, 2 = 2P - P^2
  int converged = 0;  // 1 once converged
  int steps = 0;
};

__global__ void sp2_gershgorin_rows_kernel(int nmo, const double* F_ortho, double* row_min,
                                           double* row_max) {
  const int row = static_cast<int>(blockIdx.x);
  if (row >= nmo) return;

  __shared__ double s_rad[256];
  double rad = 0.0;
  for (int j = static_cast<int>(threadIdx.x); j < nmo; j += static_cast<int>(blockDim.x)) {
    if (j != row) {
      rad += fabs(F_ortho[static_cast<std::size_t>(row) * nmo + j]);
    }
  }
  s_rad[threadIdx.x] = rad;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_rad[threadIdx.x] += s_rad[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double diag = F_ortho[static_cast<std::size_t>(row) * nmo + row];
    row_min[row] = diag - s_rad[0];
    row_max[row] = diag + s_rad[0];
  }
}

__global__ void sp2_gershgorin_reduce_kernel(int nmo, const double* row_min, const double* row_max,
                                             double* bounds2_out, Sp2DeviceCtrl* ctrl_out) {
  __shared__ double s_min[256];
  __shared__ double s_max[256];

  double lmin = 1e300;
  double lmax = -1e300;
  for (int i = static_cast<int>(threadIdx.x); i < nmo; i += static_cast<int>(blockDim.x)) {
    lmin = fmin(lmin, row_min[i]);
    lmax = fmax(lmax, row_max[i]);
  }
  s_min[threadIdx.x] = lmin;
  s_max[threadIdx.x] = lmax;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_min[threadIdx.x] = fmin(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
      s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double span = fmax(s_max[0] - s_min[0], 1e-12);
    const double pad = 1e-10 * span + 1e-12;
    bounds2_out[0] = s_min[0] - pad;
    bounds2_out[1] = s_max[0] + pad;
    ctrl_out->trace = 0.0;
    ctrl_out->gamma = 1e300;
    ctrl_out->prev_gamma = 1e300;
    ctrl_out->mode = 0;
    ctrl_out->converged = 0;
    ctrl_out->steps = 0;
  }
}

__global__ void sp2_init_p0_kernel(int nmo, const double* F_ortho, const double* bounds2,
                                   double* P_out) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(nmo) * nmo;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / nmo);
  const int j = static_cast<int>(idx % nmo);
  const double emin = bounds2[0];
  const double emax = bounds2[1];
  const double inv_span = 1.0 / (emax - emin);
  const double diag_term = (i == j) ? emax : 0.0;
  P_out[idx] = (diag_term - F_ortho[idx]) * inv_span;
}

__global__ void sp2_trace_decide_kernel(int nmo, int n_occ, double idem_tol, double trace_tol,
                                        double stagnation_floor, const double* P,
                                        const double* P2, Sp2DeviceCtrl* ctrl) {
  __shared__ double s_trP[256];
  __shared__ double s_trP2[256];

  if (ctrl->converged != 0) {
    if (threadIdx.x == 0) {
      ctrl->mode = 0;
    }
    return;
  }

  double trP = 0.0;
  double trP2 = 0.0;
  for (int i = static_cast<int>(threadIdx.x); i < nmo; i += static_cast<int>(blockDim.x)) {
    const std::size_t diag_idx = static_cast<std::size_t>(i) * nmo + i;
    trP += P[diag_idx];
    trP2 += P2[diag_idx];
  }
  s_trP[threadIdx.x] = trP;
  s_trP2[threadIdx.x] = trP2;
  __syncthreads();

  for (unsigned int stride = 128; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_trP[threadIdx.x] += s_trP[threadIdx.x + stride];
      s_trP2[threadIdx.x] += s_trP2[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double tau = s_trP[0];
    const double tau2 = s_trP2[0];
    const double gamma = fabs(tau - tau2);
    const double target = static_cast<double>(n_occ);
    const double tr_err = fabs(tau - target);

    if (ctrl->steps >= 2 &&
        ((gamma <= idem_tol && tr_err <= trace_tol) ||
         (gamma <= stagnation_floor && tr_err <= 1e-2 && gamma >= ctrl->prev_gamma * 0.995))) {
      ctrl->trace = tau;
      ctrl->gamma = gamma;
      ctrl->mode = 0;
      ctrl->converged = 1;
    } else {
      const bool branch_up = (tau < target);
      ctrl->mode = branch_up ? 2 : 1;
      ctrl->trace = branch_up ? (2.0 * tau - tau2) : tau2;
      ctrl->prev_gamma = gamma;
      ctrl->gamma = gamma;
      ctrl->steps += 1;
    }
  }
}

__global__ void sp2_apply_update_kernel(std::size_t total, double* P, const double* P2,
                                        const Sp2DeviceCtrl* ctrl) {
  const int mode = ctrl->mode;
  if (mode == 0) return;
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  if (mode == 1) {
    P[idx] = P2[idx];
  } else {
    P[idx] = 2.0 * P[idx] - P2[idx];
  }
}

__global__ void sp2_mcweeny_kernel(std::size_t total, const double* P2, const double* P3,
                                   double* P_out) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  P_out[idx] = 3.0 * P2[idx] - 2.0 * P3[idx];
}

__global__ void set_identity_scaled_kernel(int n, double diag_val, double* A) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(n) * n;
  if (idx >= total) return;
  const int i = static_cast<int>(idx / n);
  const int j = static_cast<int>(idx % n);
  A[idx] = (i == j) ? diag_val : 0.0;
}

}  // namespace

// ---------------------------------------------------------------------------
// GpuLinalgContext Implementation
// ---------------------------------------------------------------------------

struct GpuLinalgContext::Impl {
  int device_id = 0;
  DeviceMemoryPool* pool = nullptr;
  CudaGraphExecutor graph_exec{true};
  cublasHandle_t cublas = nullptr;
  cusolverDnHandle_t cusolver = nullptr;
  StreamHandle current_stream = nullptr;
  void* d_cublas_workspace = nullptr;

  Impl(DeviceMemoryPool* ext_pool, int dev) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
      (void)cudaGetLastError();
      pool = (ext_pool != nullptr) ? ext_pool : &default_device_pool(0);
      return;
    }
    if (dev < 0) {
      check_cuda(cudaGetDevice(&device_id), "GpuLinalgContext cudaGetDevice");
    } else {
      device_id = dev;
      check_cuda(cudaSetDevice(device_id), "GpuLinalgContext cudaSetDevice");
    }
    pool = (ext_pool != nullptr) ? ext_pool : &default_device_pool(device_id);

    check_cublas(cublasCreate(&cublas), "cublasCreate");
    check_cusolver(cusolverDnCreate(&cusolver), "cusolverDnCreate");

    d_cublas_workspace = pool->acquire(kCublasWorkspaceBytes, nullptr);
    check_cublas(cublasSetWorkspace(cublas, d_cublas_workspace, kCublasWorkspaceBytes),
                 "cublasSetWorkspace");
  }

  ~Impl() {
    if (cusolver != nullptr) {
      (void)cusolverDnDestroy(cusolver);
      cusolver = nullptr;
    }
    if (cublas != nullptr) {
      (void)cublasDestroy(cublas);
      cublas = nullptr;
    }
    if (pool != nullptr && d_cublas_workspace != nullptr) {
      pool->release(d_cublas_workspace, nullptr);
      d_cublas_workspace = nullptr;
    }
  }
};

GpuLinalgContext::GpuLinalgContext(DeviceMemoryPool* pool, int device_id)
    : impl_(std::make_unique<Impl>(pool, device_id)) {}

GpuLinalgContext::~GpuLinalgContext() = default;
GpuLinalgContext::GpuLinalgContext(GpuLinalgContext&&) noexcept = default;
GpuLinalgContext& GpuLinalgContext::operator=(GpuLinalgContext&&) noexcept = default;

DeviceMemoryPool& GpuLinalgContext::pool() noexcept { return *impl_->pool; }
CudaGraphExecutor& GpuLinalgContext::graph_executor() noexcept { return impl_->graph_exec; }

void GpuLinalgContext::set_stream(StreamHandle stream) {
  impl_->current_stream = stream;
  if (impl_->cublas != nullptr) {
    check_cublas(cublasSetStream(impl_->cublas, stream), "cublasSetStream");
  }
  if (impl_->cusolver != nullptr) {
    check_cusolver(cusolverDnSetStream(impl_->cusolver, stream), "cusolverDnSetStream");
  }
}

StreamHandle GpuLinalgContext::stream() const noexcept { return impl_->current_stream; }
int GpuLinalgContext::device_id() const noexcept { return impl_->device_id; }
void* GpuLinalgContext::cublas_raw() const noexcept { return impl_->cublas; }
void* GpuLinalgContext::cusolver_raw() const noexcept { return impl_->cusolver; }

// ---------------------------------------------------------------------------
// Device-Resident GEMM (FP64, FP32 Tensor Core, Ozaki INT8 Tensor Core)
// ---------------------------------------------------------------------------

void gemm_device(GpuLinalgContext& ctx, int m, int n, int k, double alpha, const double* d_A,
                 int lda, bool transA, const double* d_B, int ldb, bool transB, double beta,
                 double* d_C, int ldc, GemmMode mode, StreamHandle stream, int ozaki_slices) {
  if (m <= 0 || n <= 0) return;
  auto* cublas = static_cast<cublasHandle_t>(ctx.cublas_raw());
  if (cublas == nullptr) {
    throw std::runtime_error("gemm_device: CUDA/cuBLAS is not available on this system");
  }

  ctx.set_stream(stream);

  if (k <= 0 || alpha == 0.0) {
    const std::size_t total = static_cast<std::size_t>(m) * n;
    const int blocks = static_cast<int>((total + 255) / 256);
    scale_or_zero_rect_kernel<<<blocks, 256, 0, stream>>>(m, n, beta, d_C, ldc);
    check_cuda(cudaGetLastError(), "scale_or_zero_rect_kernel");
    return;
  }

  const cublasOperation_t op_a = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t op_b = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  if (mode == GemmMode::FP64) {
    // Row-major C[m x n] = op(A)[m x k] * op(B)[k x n]
    // <-> Col-major C^T[n x m] = op(B)^T[n x k] * op(A)^T[k x m].
    check_cublas(cublasDgemm(cublas, op_b, op_a, n, m, k, &alpha, d_B, ldb, d_A, lda, &beta, d_C,
                             ldc),
                 "cublasDgemm");
    return;
  }

  if (mode == GemmMode::FP32) {
    auto& pool = ctx.pool();
    PooledBuffer<float> d_Af(pool, static_cast<std::size_t>(m) * k, stream);
    PooledBuffer<float> d_Bf(pool, static_cast<std::size_t>(k) * n, stream);
    PooledBuffer<float> d_Cf(pool, static_cast<std::size_t>(m) * n, stream);

    const int blocks_a = static_cast<int>((static_cast<std::size_t>(m) * k + 255) / 256);
    const int blocks_b = static_cast<int>((static_cast<std::size_t>(k) * n + 255) / 256);
    const int blocks_c = static_cast<int>((static_cast<std::size_t>(m) * n + 255) / 256);

    extract_op_to_fp32_kernel<<<blocks_a, 256, 0, stream>>>(m, k, d_A, lda, transA, d_Af.data());
    extract_op_to_fp32_kernel<<<blocks_b, 256, 0, stream>>>(k, n, d_B, ldb, transB, d_Bf.data());

    const float alpha_f = 1.0f;
    const float beta_f = 0.0f;
    check_cublas(cublasGemmEx(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha_f, d_Bf.data(),
                              CUDA_R_32F, n, d_Af.data(), CUDA_R_32F, k, &beta_f, d_Cf.data(),
                              CUDA_R_32F, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT),
                 "cublasGemmEx FP32 Tensor Core");

    epilogue_fp32_to_fp64_kernel<<<blocks_c, 256, 0, stream>>>(m, n, alpha, d_Cf.data(), beta, d_C,
                                                               ldc);
    check_cuda(cudaGetLastError(), "gemm_device FP32 epilogue");
    return;
  }

  // -------------------------------------------------------------------------
  // GemmMode::OzakiInt8 — Error-free INT8 Tensor Core FP64-Emulated GEMM
  // -------------------------------------------------------------------------
  const int S = std::clamp(ozaki_slices, 2, 10);
  const int mp = ((m + 15) / 16) * 16;
  const int np = ((n + 15) / 16) * 16;
  const int max_kc = std::min(k, kMaxOzakiKChunk);
  const int max_kp = ((max_kc + 15) / 16) * 16;

  auto& pool = ctx.pool();
  PooledBuffer<std::int8_t> d_A_slices(
      pool, static_cast<std::size_t>(S) * mp * max_kp, stream);
  PooledBuffer<std::int8_t> d_B_slices_T(
      pool, static_cast<std::size_t>(S) * np * max_kp, stream);
  PooledBuffer<int> d_exp_A(pool, static_cast<std::size_t>(mp), stream);
  PooledBuffer<int> d_exp_B(pool, static_cast<std::size_t>(np), stream);
  PooledBuffer<std::int32_t> d_C_i32(pool, static_cast<std::size_t>(mp) * np, stream);
  PooledBuffer<double> d_C_acc(pool, static_cast<std::size_t>(m) * n, stream);

  memset_zero_async(d_C_acc.data(), static_cast<std::size_t>(m) * n * sizeof(double), stream);

  const std::size_t total_mn = static_cast<std::size_t>(m) * n;
  const int blocks_mn = static_cast<int>((total_mn + 255) / 256);

  for (int k0 = 0; k0 < k; k0 += kMaxOzakiKChunk) {
    const int kc = std::min(k - k0, kMaxOzakiKChunk);
    const int kp = ((kc + 15) / 16) * 16;

    ozaki_slice_rows_kernel<<<mp, 256, 0, stream>>>(
        m, kc, k0, mp, kp, S, d_A, lda, transA, d_exp_A.data(), d_A_slices.data());
    ozaki_slice_cols_kernel<<<np, 256, 0, stream>>>(
        n, kc, k0, np, kp, S, d_B, ldb, transB, d_exp_B.data(), d_B_slices_T.data());

    // Sum diagonals d = s + t from S + 1 down to 2 (smallest magnitude terms first!).
    // Within each diagonal d, all pairs (s, d - s) share the exact exponent shift -7*d,
    // so we accumulate them directly inside exact int32 Tensor Core registers/buffer!
    for (int d = S + 1; d >= 2; --d) {
      const int s_min = std::max(1, d - S);
      const int s_max = std::min(S, d - 1);
      bool first_in_diag = true;

      for (int s = s_min; s <= s_max; ++s) {
        const int t = d - s;
        const std::int8_t* ptr_A_s =
            d_A_slices.data() + static_cast<std::size_t>(s - 1) * mp * kp;
        const std::int8_t* ptr_B_t =
            d_B_slices_T.data() + static_cast<std::size_t>(t - 1) * np * kp;

        const std::int32_t alpha_i32 = 1;
        const std::int32_t beta_i32 = first_in_diag ? 0 : 1;
        first_in_diag = false;

        // TN layout in column-major hits native Blackwell INT8 Tensor Core MMA directly:
        // B_slices_T is col-major [kp x np], op_T -> [np x kp];
        // A_slices is col-major [kp x mp], op_N -> [kp x mp];
        // Output C_i32 is col-major [np x mp] == row-major [mp x np]!
        check_cublas(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N, np, mp, kp, &alpha_i32,
                                  ptr_B_t, CUDA_R_8I, kp, ptr_A_s, CUDA_R_8I, kp, &beta_i32,
                                  d_C_i32.data(), CUDA_R_32I, np, CUBLAS_COMPUTE_32I,
                                  CUBLAS_GEMM_DEFAULT),
                     "cublasGemmEx Ozaki INT8");
      }

      ozaki_accum_diag_kernel<<<blocks_mn, 256, 0, stream>>>(
          m, n, np, d_C_i32.data(), d_exp_A.data(), d_exp_B.data(), 7 * d, d_C_acc.data());
    }
  }

  ozaki_epilogue_kernel<<<blocks_mn, 256, 0, stream>>>(
      m, n, alpha, d_C_acc.data(), beta, d_C, ldc);
  check_cuda(cudaGetLastError(), "Ozaki INT8 epilogue");
}

// ---------------------------------------------------------------------------
// Symmetric Eigensolver & Roothaan-Hall Diagonalization on GPU
// ---------------------------------------------------------------------------

void syevd_device(GpuLinalgContext& ctx, int n, const double* d_A, double* d_w_out,
                  double* d_U_out, StreamHandle stream) {
  if (n <= 0) return;
  auto* cusolver = static_cast<cusolverDnHandle_t>(ctx.cusolver_raw());
  if (cusolver == nullptr) {
    throw std::runtime_error("syevd_device: cuSOLVER is not available on this system");
  }
  ctx.set_stream(stream);

  auto& pool = ctx.pool();
  PooledBuffer<double> d_work_mat(pool, static_cast<std::size_t>(n) * n, stream);
  copy_d2d_async(d_work_mat.data(), d_A, static_cast<std::size_t>(n) * n * sizeof(double), stream);

  int lwork = 0;
  check_cusolver(cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER, n, d_work_mat.data(), n,
                                             d_w_out, &lwork),
                 "cusolverDnDsyevd_bufferSize");

  PooledBuffer<double> d_work(pool, static_cast<std::size_t>(std::max(1, lwork)), stream);
  PooledBuffer<int> d_info(pool, 1, stream);

  check_cusolver(cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
                                  d_work_mat.data(), n, d_w_out, d_work.data(), lwork,
                                  d_info.data()),
                 "cusolverDnDsyevd");

  int h_info = 0;
  copy_d2h_async(&h_info, d_info.data(), sizeof(int), stream);
  synchronize_stream(stream);
  if (h_info != 0) {
    throw std::runtime_error("cusolverDnDsyevd failed with devInfo = " + std::to_string(h_info));
  }

  // cusolverDnDsyevd stores eigenvector j in column-major column j (contiguous length n),
  // which in row-major is row j; transpose so row-major U[:, j] holds eigenvector j.
  const dim3 grid((n + 31) / 32, (n + 31) / 32);
  const dim3 block(32, 32);
  transpose_square_kernel<<<grid, block, 0, stream>>>(n, d_work_mat.data(), d_U_out);
  check_cuda(cudaGetLastError(), "transpose_square_kernel");
}

void solve_fock_diag_device(GpuLinalgContext& ctx, int nao, int nmo, int n_occ, const double* d_F,
                            const double* d_X, double* d_eps_out, double* d_C_out, double* d_D_out,
                            double* d_W_out, GemmMode gemm_mode, StreamHandle stream,
                            int ozaki_slices) {
  if (nao <= 0 || nmo <= 0) return;
  auto& pool = ctx.pool();

  // 1. F_ortho = X^T * F * X
  PooledBuffer<double> d_FX(pool, static_cast<std::size_t>(nao) * nmo, stream);
  PooledBuffer<double> d_F_ortho(pool, static_cast<std::size_t>(nmo) * nmo, stream);
  PooledBuffer<double> d_C_ortho(pool, static_cast<std::size_t>(nmo) * nmo, stream);

  gemm_device(ctx, nao, nmo, nao, 1.0, d_F, nao, false, d_X, nmo, false, 0.0, d_FX.data(), nmo,
              gemm_mode, stream, ozaki_slices);
  gemm_device(ctx, nmo, nmo, nao, 1.0, d_X, nmo, true, d_FX.data(), nmo, false, 0.0,
              d_F_ortho.data(), nmo, gemm_mode, stream, ozaki_slices);
  symmetrize_matrix_device(nmo, d_F_ortho.data(), stream);

  // 2. Diagonalize F_ortho = C_ortho * diag(eps) * C_ortho^T
  syevd_device(ctx, nmo, d_F_ortho.data(), d_eps_out, d_C_ortho.data(), stream);

  // 3. Back-transform C = X * C_ortho
  gemm_device(ctx, nao, nmo, nmo, 1.0, d_X, nmo, false, d_C_ortho.data(), nmo, false, 0.0, d_C_out,
              nmo, gemm_mode, stream, ozaki_slices);

  // 4. Density matrix D = 2 * C_occ * C_occ^T and Energy-weighted W = 2 * C_occ * diag(eps) * C_occ^T
  if (n_occ <= 0) {
    memset_zero_async(d_D_out, static_cast<std::size_t>(nao) * nao * sizeof(double), stream);
    if (d_W_out != nullptr) {
      memset_zero_async(d_W_out, static_cast<std::size_t>(nao) * nao * sizeof(double), stream);
    }
    return;
  }

  PooledBuffer<double> d_C_occ(pool, static_cast<std::size_t>(nao) * n_occ, stream);
  const int blocks_occ = static_cast<int>((static_cast<std::size_t>(nao) * n_occ + 255) / 256);
  slice_and_scale_cols_kernel<<<blocks_occ, 256, 0, stream>>>(
      nao, nmo, n_occ, d_C_out, nullptr, d_C_occ.data());

  gemm_device(ctx, nao, nao, n_occ, 2.0, d_C_occ.data(), n_occ, false, d_C_occ.data(), n_occ, true,
              0.0, d_D_out, nao, gemm_mode, stream, ozaki_slices);
  symmetrize_matrix_device(nao, d_D_out, stream);

  if (d_W_out != nullptr) {
    PooledBuffer<double> d_C_scaled(pool, static_cast<std::size_t>(nao) * n_occ, stream);
    slice_and_scale_cols_kernel<<<blocks_occ, 256, 0, stream>>>(
        nao, nmo, n_occ, d_C_out, d_eps_out, d_C_scaled.data());
    gemm_device(ctx, nao, nao, n_occ, 2.0, d_C_occ.data(), n_occ, false, d_C_scaled.data(), n_occ,
                true, 0.0, d_W_out, nao, gemm_mode, stream, ozaki_slices);
    symmetrize_matrix_device(nao, d_W_out, stream);
  }
}

// ---------------------------------------------------------------------------
// SP2 Density Matrix Purification on GPU (with CUDA Graph Capture!)
// ---------------------------------------------------------------------------

DeviceSp2Stats solve_fock_sp2_device(GpuLinalgContext& ctx, int nao, int nmo, int n_occ,
                                     const double* d_F, const double* d_X, double* d_P_work,
                                     double* d_D_out, double* d_W_out, const Sp2Options& opts,
                                     StreamHandle stream) {
  DeviceSp2Stats stats{};
  if (nao <= 0 || nmo <= 0) return stats;
  auto& pool = ctx.pool();

  if (n_occ <= 0) {
    memset_zero_async(d_D_out, static_cast<std::size_t>(nao) * nao * sizeof(double), stream);
    if (d_W_out != nullptr) {
      memset_zero_async(d_W_out, static_cast<std::size_t>(nao) * nao * sizeof(double), stream);
    }
    return stats;
  }

  if (n_occ >= nmo) {
    const int blocks_nmo = static_cast<int>((static_cast<std::size_t>(nmo) * nmo + 255) / 256);
    set_identity_scaled_kernel<<<blocks_nmo, 256, 0, stream>>>(nmo, 1.0, d_P_work);
    gemm_device(ctx, nao, nao, nmo, 2.0, d_X, nmo, false, d_X, nmo, true, 0.0, d_D_out, nao,
                opts.gemm_mode, stream, opts.ozaki_slices);
    symmetrize_matrix_device(nao, d_D_out, stream);
    if (d_W_out != nullptr) {
      PooledBuffer<double> d_DF(pool, static_cast<std::size_t>(nao) * nao, stream);
      gemm_device(ctx, nao, nao, nao, 1.0, d_D_out, nao, false, d_F, nao, false, 0.0, d_DF.data(),
                  nao, opts.gemm_mode, stream, opts.ozaki_slices);
      gemm_device(ctx, nao, nao, nao, 0.5, d_DF.data(), nao, false, d_D_out, nao, false, 0.0,
                  d_W_out, nao, opts.gemm_mode, stream, opts.ozaki_slices);
      symmetrize_matrix_device(nao, d_W_out, stream);
    }
    stats.final_trace = static_cast<double>(nmo);
    return stats;
  }

  const std::size_t nmo2 = static_cast<std::size_t>(nmo) * nmo;
  const int blocks_nmo2 = static_cast<int>((nmo2 + 255) / 256);

  // Choose active execution stream (use graph_executor stream directly when capturing if stream is
  // default/matching, or synchronize across streams via CUDA events).
  auto& graph_exec = ctx.graph_executor();
  StreamHandle work_stream = opts.use_cuda_graph ? graph_exec.stream() : stream;

  cudaEvent_t sync_ev = nullptr;
  const bool bridge_streams = (opts.use_cuda_graph && stream != work_stream);
  if (bridge_streams) {
    check_cuda(cudaEventCreateWithFlags(&sync_ev, cudaEventDisableTiming), "cudaEventCreate");
    check_cuda(cudaEventRecord(sync_ev, stream), "cudaEventRecord pre-SP2");
    check_cuda(cudaStreamWaitEvent(work_stream, sync_ev, 0), "cudaStreamWaitEvent pre-SP2");
  }

  PooledBuffer<double> d_FX(pool, static_cast<std::size_t>(nao) * nmo, work_stream);
  PooledBuffer<double> d_F_ortho(pool, nmo2, work_stream);
  PooledBuffer<double> d_P2(pool, nmo2, work_stream);
  PooledBuffer<double> d_row_min(pool, static_cast<std::size_t>(nmo), work_stream);
  PooledBuffer<double> d_row_max(pool, static_cast<std::size_t>(nmo), work_stream);
  PooledBuffer<double> d_bounds2(pool, 2, work_stream);
  PooledBuffer<Sp2DeviceCtrl> d_ctrl(pool, 1, work_stream);

  // 1. Transform F' = 0.5 * (X^T F X + (X^T F X)^T)
  gemm_device(ctx, nao, nmo, nao, 1.0, d_F, nao, false, d_X, nmo, false, 0.0, d_FX.data(), nmo,
              opts.gemm_mode, work_stream, opts.ozaki_slices);
  gemm_device(ctx, nmo, nmo, nao, 1.0, d_X, nmo, true, d_FX.data(), nmo, false, 0.0,
              d_F_ortho.data(), nmo, opts.gemm_mode, work_stream, opts.ozaki_slices);
  symmetrize_matrix_device(nmo, d_F_ortho.data(), work_stream);

  // 2. Gershgorin spectral bounds & initial P_0 = (emax * I - F') / (emax - emin)
  sp2_gershgorin_rows_kernel<<<nmo, 256, 0, work_stream>>>(
      nmo, d_F_ortho.data(), d_row_min.data(), d_row_max.data());
  sp2_gershgorin_reduce_kernel<<<1, 256, 0, work_stream>>>(
      nmo, d_row_min.data(), d_row_max.data(), d_bounds2.data(), d_ctrl.data());
  sp2_init_p0_kernel<<<blocks_nmo2, 256, 0, work_stream>>>(
      nmo, d_F_ortho.data(), d_bounds2.data(), d_P_work);

  // 3. Fixed-topology SP2 polynomial step blocks (captured in CUDA Graph!)
  const int block_steps = std::max(1, opts.graph_block_steps);
  const double idem_tol =
      (opts.gemm_mode == GemmMode::FP32) ? std::max(opts.idempotency_tol, 1e-4) : opts.idempotency_tol;
  const double tr_tol =
      (opts.gemm_mode == GemmMode::FP32) ? std::max(opts.trace_tol, 1e-3) : opts.trace_tol;
  const double stag_floor = (opts.gemm_mode == GemmMode::FP32) ? 5e-2 : 1e-6;

  auto run_sp2_block = [&](StreamHandle s) {
    for (int step = 0; step < block_steps; ++step) {
      gemm_device(ctx, nmo, nmo, nmo, 1.0, d_P_work, nmo, false, d_P_work, nmo, false, 0.0,
                  d_P2.data(), nmo, opts.gemm_mode, s, opts.ozaki_slices);
      sp2_trace_decide_kernel<<<1, 256, 0, s>>>(nmo, n_occ, idem_tol, tr_tol, stag_floor, d_P_work,
                                                d_P2.data(), d_ctrl.data());
      sp2_apply_update_kernel<<<blocks_nmo2, 256, 0, s>>>(nmo2, d_P_work, d_P2.data(),
                                                          d_ctrl.data());
    }
  };

  std::uint64_t key = 0x5350325F424C4B31ULL;  // "SP2_BLK1"
  key = CudaGraphExecutor::hash_combine(key, static_cast<std::uint64_t>(nmo));
  key = CudaGraphExecutor::hash_combine(key, static_cast<std::uint64_t>(n_occ));
  key = CudaGraphExecutor::hash_combine(key, static_cast<std::uint64_t>(block_steps));
  key = CudaGraphExecutor::hash_combine(key, static_cast<std::uint64_t>(opts.gemm_mode));
  key = CudaGraphExecutor::hash_combine(key, static_cast<std::uint64_t>(opts.ozaki_slices));
  key = CudaGraphExecutor::hash_combine(key, reinterpret_cast<std::uintptr_t>(d_P_work));
  key = CudaGraphExecutor::hash_combine(key, reinterpret_cast<std::uintptr_t>(d_P2.data()));
  key = CudaGraphExecutor::hash_combine(key, reinterpret_cast<std::uintptr_t>(d_ctrl.data()));

  Sp2DeviceCtrl h_ctrl{};
  for (int elapsed = 0; elapsed < opts.max_iter; elapsed += block_steps) {
    if (opts.use_cuda_graph) {
      graph_exec.execute(key, run_sp2_block);
    } else {
      run_sp2_block(work_stream);
    }
    copy_d2h_async(&h_ctrl, d_ctrl.data(), sizeof(Sp2DeviceCtrl), work_stream);
    synchronize_stream(work_stream);
    if (h_ctrl.converged != 0) break;
  }

  // 4. Optional McWeeny cubic polish P <- 3 P^2 - 2 P^3 for machine-precision idempotency
  if (opts.mcweeny_polish && opts.gemm_mode != GemmMode::FP32) {
    PooledBuffer<double> d_P3(pool, nmo2, work_stream);
    gemm_device(ctx, nmo, nmo, nmo, 1.0, d_P_work, nmo, false, d_P_work, nmo, false, 0.0,
                d_P2.data(), nmo, opts.gemm_mode, work_stream, opts.ozaki_slices);
    gemm_device(ctx, nmo, nmo, nmo, 1.0, d_P2.data(), nmo, false, d_P_work, nmo, false, 0.0,
                d_P3.data(), nmo, opts.gemm_mode, work_stream, opts.ozaki_slices);
    sp2_mcweeny_kernel<<<blocks_nmo2, 256, 0, work_stream>>>(nmo2, d_P2.data(), d_P3.data(),
                                                             d_P_work);
  }
  symmetrize_matrix_device(nmo, d_P_work, work_stream);

  // 5. Back-transform D = 2 * X * P * X^T
  gemm_device(ctx, nao, nmo, nmo, 1.0, d_X, nmo, false, d_P_work, nmo, false, 0.0, d_FX.data(), nmo,
              opts.gemm_mode, work_stream, opts.ozaki_slices);
  gemm_device(ctx, nao, nao, nmo, 2.0, d_FX.data(), nmo, false, d_X, nmo, true, 0.0, d_D_out, nao,
              opts.gemm_mode, work_stream, opts.ozaki_slices);
  symmetrize_matrix_device(nao, d_D_out, work_stream);

  // 6. Energy-weighted density matrix W = 0.5 * D * F * D (exact without MO orbitals!)
  if (d_W_out != nullptr) {
    PooledBuffer<double> d_DF(pool, static_cast<std::size_t>(nao) * nao, work_stream);
    gemm_device(ctx, nao, nao, nao, 1.0, d_D_out, nao, false, d_F, nao, false, 0.0, d_DF.data(),
                nao, opts.gemm_mode, work_stream, opts.ozaki_slices);
    gemm_device(ctx, nao, nao, nao, 0.5, d_DF.data(), nao, false, d_D_out, nao, false, 0.0,
                d_W_out, nao, opts.gemm_mode, work_stream, opts.ozaki_slices);
    symmetrize_matrix_device(nao, d_W_out, work_stream);
  }

  if (bridge_streams) {
    check_cuda(cudaEventRecord(sync_ev, work_stream), "cudaEventRecord post-SP2");
    check_cuda(cudaStreamWaitEvent(stream, sync_ev, 0), "cudaStreamWaitEvent post-SP2");
    check_cuda(cudaEventDestroy(sync_ev), "cudaEventDestroy");
  }

  stats.iterations = h_ctrl.steps;
  stats.final_trace = h_ctrl.trace;
  stats.idempotency_error = h_ctrl.gamma;
  return stats;
}

// ---------------------------------------------------------------------------
// Device Matrix Primitives
// ---------------------------------------------------------------------------

void symmetrize_matrix_device(int n, double* d_A, StreamHandle stream) {
  if (n <= 1) return;
  const dim3 block(16, 16);
  const dim3 grid((n + 15) / 16, (n + 15) / 16);
  symmetrize_inplace_kernel<<<grid, block, 0, stream>>>(n, d_A);
  check_cuda(cudaGetLastError(), "symmetrize_inplace_kernel");
}

void axpby_device(std::size_t count, double alpha, const double* d_x, double beta, double* d_y,
                  StreamHandle stream) {
  if (count == 0) return;
  const int blocks = static_cast<int>((count + 255) / 256);
  axpby_kernel<<<blocks, 256, 0, stream>>>(count, alpha, d_x, beta, d_y);
  check_cuda(cudaGetLastError(), "axpby_kernel");
}

void compute_delta_and_norm_device(std::size_t count, const double* d_new, const double* d_old,
                                   double* d_delta_out, double* d_norm_stats_out,
                                   StreamHandle stream) {
  if (d_norm_stats_out != nullptr) {
    memset_zero_async(d_norm_stats_out, 2 * sizeof(double), stream);
  }
  if (count == 0) return;
  const int blocks = static_cast<int>(std::min<std::size_t>((count + 255) / 256, 512));
  delta_and_norm_kernel<<<blocks, 256, 0, stream>>>(count, d_new, d_old, d_delta_out,
                                                    d_norm_stats_out);
  check_cuda(cudaGetLastError(), "delta_and_norm_kernel");
}

void assemble_fock_and_traces_device(int nao, const double* d_D, const double* d_H,
                                     const double* d_J, const double* d_K, double alpha_x,
                                     const double* d_Vxc, double* d_F_out,
                                     double* d_traces3_out, StreamHandle stream) {
  const std::size_t count = static_cast<std::size_t>(nao) * nao;
  if (d_traces3_out != nullptr) {
    memset_zero_async(d_traces3_out, 3 * sizeof(double), stream);
  }
  if (count == 0) return;
  const int blocks = static_cast<int>(std::min<std::size_t>((count + 255) / 256, 512));
  assemble_fock_and_traces_kernel<<<blocks, 256, 0, stream>>>(
      count, d_D, d_H, d_J, d_K, alpha_x, d_Vxc, d_F_out, d_traces3_out);
  check_cuda(cudaGetLastError(), "assemble_fock_and_traces_kernel");
}

void apply_level_shift_ao_device(GpuLinalgContext& ctx, int nao, double level_shift,
                                 const double* d_S, const double* d_D, double* d_F_inout,
                                 GemmMode gemm_mode, StreamHandle stream) {
  if (nao <= 0 || level_shift == 0.0) return;
  auto& pool = ctx.pool();
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  PooledBuffer<double> d_SD(pool, n2, stream);
  PooledBuffer<double> d_SDS(pool, n2, stream);

  // Exact AO virtual-space shift: F <- F + level_shift * (S - 0.5 * S * D * S)
  gemm_device(ctx, nao, nao, nao, 1.0, d_S, nao, false, d_D, nao, false, 0.0, d_SD.data(), nao,
              gemm_mode, stream);
  gemm_device(ctx, nao, nao, nao, 1.0, d_SD.data(), nao, false, d_S, nao, false, 0.0,
              d_SDS.data(), nao, gemm_mode, stream);

  axpby_device(n2, level_shift, d_S, 1.0, d_F_inout, stream);
  axpby_device(n2, -0.5 * level_shift, d_SDS.data(), 1.0, d_F_inout, stream);
  symmetrize_matrix_device(nao, d_F_inout, stream);
}

}  // namespace gpu4dft
