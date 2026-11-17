#include "gpu4dft/linalg.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace gpu4dft {

const char* to_string(GemmMode mode) {
  switch (mode) {
    case GemmMode::FP64:
      return "fp64";
    case GemmMode::FP32:
      return "fp32";
    case GemmMode::OzakiInt8:
      return "ozaki_int8";
  }
  return "unknown";
}

namespace {

// ---------------------------------------------------------------------------
// Host Reference / OpenMP Implementations
// ---------------------------------------------------------------------------

void gemm_host_fp64(int m, int n, int k, double alpha, const double* A, int lda, bool transA,
                    const double* B, int ldb, bool transB, double beta, double* C, int ldc) {
  if (m <= 0 || n <= 0) return;

#pragma omp parallel for schedule(static) if (static_cast<long long>(m) * n > 4096)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double& out = C[static_cast<std::size_t>(i) * ldc + j];
      out = (beta == 0.0) ? 0.0 : (beta * out);
    }
  }
  if (k <= 0 || alpha == 0.0) return;

  constexpr int kTile = 64;
#pragma omp parallel for collapse(2) schedule(static) if (static_cast<long long>(m) * n * k > 32768)
  for (int i0 = 0; i0 < m; i0 += kTile) {
    for (int j0 = 0; j0 < n; j0 += kTile) {
      const int imax = std::min(i0 + kTile, m);
      const int jmax = std::min(j0 + kTile, n);
      for (int p0 = 0; p0 < k; p0 += kTile) {
        const int pmax = std::min(p0 + kTile, k);
        for (int i = i0; i < imax; ++i) {
          for (int p = p0; p < pmax; ++p) {
            const double a_ip = alpha * (transA ? A[static_cast<std::size_t>(p) * lda + i]
                                                : A[static_cast<std::size_t>(i) * lda + p]);
            double* c_row = &C[static_cast<std::size_t>(i) * ldc];
            if (!transB) {
              const double* b_row = &B[static_cast<std::size_t>(p) * ldb];
              for (int j = j0; j < jmax; ++j) {
                c_row[j] += a_ip * b_row[j];
              }
            } else {
              for (int j = j0; j < jmax; ++j) {
                c_row[j] += a_ip * B[static_cast<std::size_t>(j) * ldb + p];
              }
            }
          }
        }
      }
    }
  }
}

void gemm_host_fp32(int m, int n, int k, double alpha, const double* A, int lda, bool transA,
                    const double* B, int ldb, bool transB, double beta, double* C, int ldc) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || alpha == 0.0) {
    gemm_host_fp64(m, n, 0, 0.0, A, lda, transA, B, ldb, transB, beta, C, ldc);
    return;
  }

  std::vector<float> Af(static_cast<std::size_t>(m) * k);
  std::vector<float> Bf_T(static_cast<std::size_t>(n) * k);

  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const double v = transA ? A[static_cast<std::size_t>(p) * lda + i]
                              : A[static_cast<std::size_t>(i) * lda + p];
      Af[static_cast<std::size_t>(i) * k + p] = static_cast<float>(v);
    }
  }
  for (int j = 0; j < n; ++j) {
    for (int p = 0; p < k; ++p) {
      const double v = transB ? B[static_cast<std::size_t>(j) * ldb + p]
                              : B[static_cast<std::size_t>(p) * ldb + j];
      Bf_T[static_cast<std::size_t>(j) * k + p] = static_cast<float>(v);
    }
  }

#pragma omp parallel for schedule(static) if (static_cast<long long>(m) * n * k > 32768)
  for (int i = 0; i < m; ++i) {
    const float* a_row = &Af[static_cast<std::size_t>(i) * k];
    for (int j = 0; j < n; ++j) {
      const float* b_col = &Bf_T[static_cast<std::size_t>(j) * k];
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) {
        acc += a_row[p] * b_col[p];
      }
      double& out = C[static_cast<std::size_t>(i) * ldc + j];
      const double scaled = alpha * static_cast<double>(acc);
      out = (beta == 0.0) ? scaled : (scaled + beta * out);
    }
  }
}

void gemm_host_ozaki_int8(int m, int n, int k, double alpha, const double* A, int lda, bool transA,
                          const double* B, int ldb, bool transB, double beta, double* C, int ldc,
                          int ozaki_slices) {
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || alpha == 0.0) {
    gemm_host_fp64(m, n, 0, 0.0, A, lda, transA, B, ldb, transB, beta, C, ldc);
    return;
  }

  const int S = std::clamp(ozaki_slices, 2, 10);
  constexpr int kMaxChunk = 32768;
  std::vector<double> C_acc(static_cast<std::size_t>(m) * n, 0.0);

  for (int k0 = 0; k0 < k; k0 += kMaxChunk) {
    const int kc = std::min(k - k0, kMaxChunk);
    std::vector<int> exp_A(static_cast<std::size_t>(m), 0);
    std::vector<int> exp_B(static_cast<std::size_t>(n), 0);
    std::vector<std::int8_t> A_slices(static_cast<std::size_t>(S) * m * kc, 0);
    std::vector<std::int8_t> B_slices_T(static_cast<std::size_t>(S) * n * kc, 0);

    for (int i = 0; i < m; ++i) {
      double amax = 0.0;
      for (int p = 0; p < kc; ++p) {
        const double v = transA ? A[static_cast<std::size_t>(k0 + p) * lda + i]
                                : A[static_cast<std::size_t>(i) * lda + (k0 + p)];
        amax = std::max(amax, std::abs(v));
      }
      if (amax > 0.0) {
        int e = 0;
        (void)std::frexp(amax, &e);
        if (std::ldexp(amax, 7 - e) >= 127.5) e += 1;
        exp_A[i] = e;
        for (int p = 0; p < kc; ++p) {
          const double v = transA ? A[static_cast<std::size_t>(k0 + p) * lda + i]
                                  : A[static_cast<std::size_t>(i) * lda + (k0 + p)];
          double rem = std::ldexp(v, -e);
          for (int s = 1; s <= S; ++s) {
            int qi = static_cast<int>(std::rint(std::ldexp(rem, 7 * s)));
            qi = std::clamp(qi, -127, 127);
            A_slices[(static_cast<std::size_t>(s - 1) * m + i) * kc + p] =
                static_cast<std::int8_t>(qi);
            rem -= std::ldexp(static_cast<double>(qi), -7 * s);
          }
        }
      }
    }

    for (int j = 0; j < n; ++j) {
      double bmax = 0.0;
      for (int p = 0; p < kc; ++p) {
        const double v = transB ? B[static_cast<std::size_t>(j) * ldb + (k0 + p)]
                                : B[static_cast<std::size_t>(k0 + p) * ldb + j];
        bmax = std::max(bmax, std::abs(v));
      }
      if (bmax > 0.0) {
        int e = 0;
        (void)std::frexp(bmax, &e);
        if (std::ldexp(bmax, 7 - e) >= 127.5) e += 1;
        exp_B[j] = e;
        for (int p = 0; p < kc; ++p) {
          const double v = transB ? B[static_cast<std::size_t>(j) * ldb + (k0 + p)]
                                  : B[static_cast<std::size_t>(k0 + p) * ldb + j];
          double rem = std::ldexp(v, -e);
          for (int t = 1; t <= S; ++t) {
            int qi = static_cast<int>(std::rint(std::ldexp(rem, 7 * t)));
            qi = std::clamp(qi, -127, 127);
            B_slices_T[(static_cast<std::size_t>(t - 1) * n + j) * kc + p] =
                static_cast<std::int8_t>(qi);
            rem -= std::ldexp(static_cast<double>(qi), -7 * t);
          }
        }
      }
    }

#pragma omp parallel for collapse(2) schedule(static) if (static_cast<long long>(m) * n * kc > 8192)
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum_ij = 0.0;
        for (int d = S + 1; d >= 2; --d) {
          const int s_min = std::max(1, d - S);
          const int s_max = std::min(S, d - 1);
          std::int32_t diag_i32 = 0;
          for (int s = s_min; s <= s_max; ++s) {
            const int t = d - s;
            const std::int8_t* a_ptr =
                &A_slices[(static_cast<std::size_t>(s - 1) * m + i) * kc];
            const std::int8_t* b_ptr =
                &B_slices_T[(static_cast<std::size_t>(t - 1) * n + j) * kc];
            for (int p = 0; p < kc; ++p) {
              diag_i32 += static_cast<std::int32_t>(a_ptr[p]) * static_cast<std::int32_t>(b_ptr[p]);
            }
          }
          if (diag_i32 != 0) {
            sum_ij += std::ldexp(static_cast<double>(diag_i32), exp_A[i] + exp_B[j] - 7 * d);
          }
        }
        C_acc[static_cast<std::size_t>(i) * n + j] += sum_ij;
      }
    }
  }

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double& out = C[static_cast<std::size_t>(i) * ldc + j];
      const double val = alpha * C_acc[static_cast<std::size_t>(i) * n + j];
      out = (beta == 0.0) ? val : (val + beta * out);
    }
  }
}

// Householder tridiagonalization + Wilkinson-shifted implicit symmetric QL (`tred2` + `tql2`).
void syevd_host(int n, const double* A, double* w_out, double* U_out) {
  if (n <= 0) return;
  if (n == 1) {
    w_out[0] = A[0];
    U_out[0] = 1.0;
    return;
  }

  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::copy(A, A + nn, U_out);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double avg = 0.5 * (U_out[static_cast<std::size_t>(i) * n + j] +
                                U_out[static_cast<std::size_t>(j) * n + i]);
      U_out[static_cast<std::size_t>(i) * n + j] = avg;
      U_out[static_cast<std::size_t>(j) * n + i] = avg;
    }
  }

  std::vector<double> e(static_cast<std::size_t>(n), 0.0);

  // 1. Householder reduction to symmetric tridiagonal form
  for (int i = n - 1; i >= 1; --i) {
    const int l = i - 1;
    double h = 0.0;
    double scale = 0.0;
    if (l > 0) {
      for (int k = 0; k <= l; ++k) {
        scale += std::abs(U_out[static_cast<std::size_t>(i) * n + k]);
      }
      if (scale == 0.0) {
        e[i] = U_out[static_cast<std::size_t>(i) * n + l];
      } else {
        for (int k = 0; k <= l; ++k) {
          U_out[static_cast<std::size_t>(i) * n + k] /= scale;
          const double v = U_out[static_cast<std::size_t>(i) * n + k];
          h += v * v;
        }
        double f = U_out[static_cast<std::size_t>(i) * n + l];
        double g = (f >= 0.0) ? -std::sqrt(h) : std::sqrt(h);
        e[i] = scale * g;
        h -= f * g;
        U_out[static_cast<std::size_t>(i) * n + l] = f - g;
        f = 0.0;
        for (int j = 0; j <= l; ++j) {
          U_out[static_cast<std::size_t>(j) * n + i] =
              U_out[static_cast<std::size_t>(i) * n + j] / h;
          g = 0.0;
          for (int k = 0; k <= j; ++k) {
            g += U_out[static_cast<std::size_t>(j) * n + k] *
                 U_out[static_cast<std::size_t>(i) * n + k];
          }
          for (int k = j + 1; k <= l; ++k) {
            g += U_out[static_cast<std::size_t>(k) * n + j] *
                 U_out[static_cast<std::size_t>(i) * n + k];
          }
          e[j] = g / h;
          f += e[j] * U_out[static_cast<std::size_t>(i) * n + j];
        }
        const double hh = f / (h + h);
        for (int j = 0; j <= l; ++j) {
          f = U_out[static_cast<std::size_t>(i) * n + j];
          g = e[j] - hh * f;
          e[j] = g;
          for (int k = 0; k <= j; ++k) {
            U_out[static_cast<std::size_t>(j) * n + k] -=
                (f * e[k] + g * U_out[static_cast<std::size_t>(i) * n + k]);
          }
        }
      }
    } else {
      e[i] = U_out[static_cast<std::size_t>(i) * n + l];
    }
    w_out[i] = h;
  }

  w_out[0] = 0.0;
  e[0] = 0.0;
  for (int i = 0; i < n; ++i) {
    const int l = i - 1;
    if (w_out[i] != 0.0) {
      for (int j = 0; j <= l; ++j) {
        double g = 0.0;
        for (int k = 0; k <= l; ++k) {
          g += U_out[static_cast<std::size_t>(i) * n + k] *
               U_out[static_cast<std::size_t>(k) * n + j];
        }
        for (int k = 0; k <= l; ++k) {
          U_out[static_cast<std::size_t>(k) * n + j] -=
              g * U_out[static_cast<std::size_t>(k) * n + i];
        }
      }
    }
    w_out[i] = U_out[static_cast<std::size_t>(i) * n + i];
    U_out[static_cast<std::size_t>(i) * n + i] = 1.0;
    for (int j = 0; j <= l; ++j) {
      U_out[static_cast<std::size_t>(j) * n + i] = 0.0;
      U_out[static_cast<std::size_t>(i) * n + j] = 0.0;
    }
  }

  // 2. Implicit Wilkinson-shifted QL iteration
  for (int i = 1; i < n; ++i) {
    e[i - 1] = e[i];
  }
  e[n - 1] = 0.0;

  for (int l = 0; l < n; ++l) {
    int iter = 0;
    int m = l;
    do {
      for (m = l; m < n - 1; ++m) {
        const double dd = std::abs(w_out[m]) + std::abs(w_out[m + 1]);
        if (std::abs(e[m]) <= 1e-16 * dd) break;
      }
      if (m != l) {
        if (++iter > 120) {
          throw std::runtime_error("syevd_host: implicit QL failed to converge");
        }
        double g = (w_out[l + 1] - w_out[l]) / (2.0 * e[l]);
        double r = std::hypot(g, 1.0);
        const double signed_r = (g >= 0.0) ? r : -r;
        g = w_out[m] - w_out[l] + e[l] / (g + signed_r);
        double s = 1.0;
        double c = 1.0;
        double p = 0.0;
        int i = m - 1;
        for (; i >= l; --i) {
          double f = s * e[i];
          const double b = c * e[i];
          r = std::hypot(f, g);
          e[i + 1] = r;
          if (r == 0.0) {
            w_out[i + 1] -= p;
            e[m] = 0.0;
            break;
          }
          s = f / r;
          c = g / r;
          g = w_out[i + 1] - p;
          r = (w_out[i] - g) * s + 2.0 * c * b;
          p = s * r;
          w_out[i + 1] = g + p;
          g = c * r - b;
          for (int k = 0; k < n; ++k) {
            f = U_out[static_cast<std::size_t>(k) * n + (i + 1)];
            U_out[static_cast<std::size_t>(k) * n + (i + 1)] =
                s * U_out[static_cast<std::size_t>(k) * n + i] + c * f;
            U_out[static_cast<std::size_t>(k) * n + i] =
                c * U_out[static_cast<std::size_t>(k) * n + i] - s * f;
          }
        }
        if (r == 0.0 && i >= l) continue;
        w_out[l] -= p;
        e[l] = g;
        e[m] = 0.0;
      }
    } while (m != l);
  }

  // 3. Sort eigenvalues in ascending order and permute eigenvector columns
  for (int i = 0; i < n - 1; ++i) {
    int k = i;
    double p = w_out[i];
    for (int j = i + 1; j < n; ++j) {
      if (w_out[j] < p) {
        k = j;
        p = w_out[j];
      }
    }
    if (k != i) {
      w_out[k] = w_out[i];
      w_out[i] = p;
      for (int r = 0; r < n; ++r) {
        std::swap(U_out[static_cast<std::size_t>(r) * n + i],
                  U_out[static_cast<std::size_t>(r) * n + k]);
      }
    }
  }
}

void symmetrize_host(int n, std::vector<double>& A) {
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double avg =
          0.5 * (A[static_cast<std::size_t>(i) * n + j] + A[static_cast<std::size_t>(j) * n + i]);
      A[static_cast<std::size_t>(i) * n + j] = avg;
      A[static_cast<std::size_t>(j) * n + i] = avg;
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Public Dispatched API (Host & CUDA)
// ---------------------------------------------------------------------------

void gemm(int m, int n, int k, double alpha, const double* A, int lda, bool transA, const double* B,
          int ldb, bool transB, double beta, double* C, int ldc, Backend backend, GemmMode mode,
          int ozaki_slices) {
  if (m <= 0 || n <= 0) return;

  switch (backend) {
    case Backend::Host:
      if (mode == GemmMode::FP64) {
        gemm_host_fp64(m, n, k, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc);
      } else if (mode == GemmMode::FP32) {
        gemm_host_fp32(m, n, k, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc);
      } else {
        gemm_host_ozaki_int8(m, n, k, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc,
                             ozaki_slices);
      }
      return;

    case Backend::Cuda: {
#if defined(GPU4DFT_HAVE_CUDA)
      GpuLinalgContext ctx;
      auto& pool = ctx.pool();
      const std::size_t rowsA = transA ? static_cast<std::size_t>(k) : static_cast<std::size_t>(m);
      const std::size_t rowsB = transB ? static_cast<std::size_t>(n) : static_cast<std::size_t>(k);
      const std::size_t sizeA = rowsA * static_cast<std::size_t>(lda);
      const std::size_t sizeB = rowsB * static_cast<std::size_t>(ldb);
      const std::size_t sizeC = static_cast<std::size_t>(m) * static_cast<std::size_t>(ldc);

      PooledBuffer<double> d_A(pool, sizeA);
      PooledBuffer<double> d_B(pool, sizeB);
      PooledBuffer<double> d_C(pool, sizeC);

      if (sizeA > 0 && A != nullptr) copy_h2d_async(d_A.data(), A, sizeA * sizeof(double));
      if (sizeB > 0 && B != nullptr) copy_h2d_async(d_B.data(), B, sizeB * sizeof(double));
      if (beta != 0.0 && sizeC > 0) copy_h2d_async(d_C.data(), C, sizeC * sizeof(double));

      gemm_device(ctx, m, n, k, alpha, d_A.data(), lda, transA, d_B.data(), ldb, transB, beta,
                  d_C.data(), ldc, mode, nullptr, ozaki_slices);
      copy_d2h_async(C, d_C.data(), sizeC * sizeof(double));
      synchronize_stream(nullptr);
      return;
#else
      throw std::runtime_error("gemm: CUDA backend requested, but GPU4DFT was built without CUDA");
#endif
    }

    default:
      throw NotImplemented("gemm: requested backend is not compiled into this build");
  }
}

void gemm(int m, int n, int k, double alpha, const double* A, bool transA, const double* B,
          bool transB, double beta, double* C, Backend backend, GemmMode mode, int ozaki_slices) {
  const int lda = transA ? m : k;
  const int ldb = transB ? k : n;
  const int ldc = n;
  gemm(m, n, k, alpha, A, lda, transA, B, ldb, transB, beta, C, ldc, backend, mode, ozaki_slices);
}

GemmBenchmarkResult benchmark_gemm(int n, int warmup_iters, int bench_iters, Backend backend,
                                   int ozaki_slices) {
  if (n <= 0) throw std::invalid_argument("benchmark_gemm: n must be positive");
  warmup_iters = std::max(1, warmup_iters);
  bench_iters = std::max(1, bench_iters);

  const std::size_t nn = static_cast<std::size_t>(n) * n;
  std::mt19937_64 rng(42);
  std::normal_distribution<double> dist(0.0, 1.0);
  std::vector<double> A(nn), B(nn), C_fp64(nn, 0.0), C_ozaki(nn, 0.0), C_fp32(nn, 0.0);
  for (std::size_t i = 0; i < nn; ++i) {
    A[i] = dist(rng);
    B[i] = dist(rng);
  }

  GemmBenchmarkResult res{};
  res.n = n;
  const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    GpuLinalgContext ctx;
    auto& pool = ctx.pool();
    PooledBuffer<double> d_A(pool, nn);
    PooledBuffer<double> d_B(pool, nn);
    PooledBuffer<double> d_C(pool, nn);
    copy_h2d_async(d_A.data(), A.data(), nn * sizeof(double));
    copy_h2d_async(d_B.data(), B.data(), nn * sizeof(double));
    synchronize_stream(nullptr);

    auto time_mode = [&](GemmMode mode, std::vector<double>& h_out) -> double {
      for (int w = 0; w < warmup_iters; ++w) {
        gemm_device(ctx, n, n, n, 1.0, d_A.data(), n, false, d_B.data(), n, false, 0.0, d_C.data(),
                    n, mode, nullptr, ozaki_slices);
      }
      synchronize_stream(nullptr);

      cudaEvent_t ev_start = nullptr;
      cudaEvent_t ev_stop = nullptr;
      cudaEventCreate(&ev_start);
      cudaEventCreate(&ev_stop);
      cudaEventRecord(ev_start, nullptr);
      for (int it = 0; it < bench_iters; ++it) {
        gemm_device(ctx, n, n, n, 1.0, d_A.data(), n, false, d_B.data(), n, false, 0.0, d_C.data(),
                    n, mode, nullptr, ozaki_slices);
      }
      cudaEventRecord(ev_stop, nullptr);
      cudaEventSynchronize(ev_stop);
      float total_ms = 0.0f;
      cudaEventElapsedTime(&total_ms, ev_start, ev_stop);
      cudaEventDestroy(ev_start);
      cudaEventDestroy(ev_stop);

      copy_d2h_async(h_out.data(), d_C.data(), nn * sizeof(double));
      synchronize_stream(nullptr);
      return static_cast<double>(total_ms) / static_cast<double>(bench_iters);
    };

    res.fp64_ms = time_mode(GemmMode::FP64, C_fp64);
    res.ozaki_ms = time_mode(GemmMode::OzakiInt8, C_ozaki);
    res.fp32_ms = time_mode(GemmMode::FP32, C_fp32);
#else
    throw std::runtime_error("benchmark_gemm: CUDA backend not available in this build");
#endif
  } else if (backend == Backend::Host) {
    auto time_host = [&](GemmMode mode, std::vector<double>& h_out) -> double {
      for (int w = 0; w < warmup_iters; ++w) {
        gemm(n, n, n, 1.0, A.data(), false, B.data(), false, 0.0, h_out.data(), Backend::Host, mode,
             ozaki_slices);
      }
      const auto t0 = std::chrono::steady_clock::now();
      for (int it = 0; it < bench_iters; ++it) {
        gemm(n, n, n, 1.0, A.data(), false, B.data(), false, 0.0, h_out.data(), Backend::Host, mode,
             ozaki_slices);
      }
      const auto t1 = std::chrono::steady_clock::now();
      const double ms =
          std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(bench_iters);
      return ms;
    };
    res.fp64_ms = time_host(GemmMode::FP64, C_fp64);
    res.ozaki_ms = time_host(GemmMode::OzakiInt8, C_ozaki);
    res.fp32_ms = time_host(GemmMode::FP32, C_fp32);
  } else {
    throw NotImplemented("benchmark_gemm: unsupported backend");
  }

  res.fp64_tflops = (res.fp64_ms > 0.0) ? (flops / (res.fp64_ms * 1e9)) : 0.0;
  res.ozaki_tflops = (res.ozaki_ms > 0.0) ? (flops / (res.ozaki_ms * 1e9)) : 0.0;
  res.fp32_tflops = (res.fp32_ms > 0.0) ? (flops / (res.fp32_ms * 1e9)) : 0.0;

  double ref_norm_sq = 0.0;
  double ozaki_diff_sq = 0.0;
  double fp32_diff_sq = 0.0;
  double ozaki_max_abs = 0.0;
  for (std::size_t i = 0; i < nn; ++i) {
    const double r = C_fp64[i];
    const double d_oz = C_ozaki[i] - r;
    const double d_32 = C_fp32[i] - r;
    ref_norm_sq += r * r;
    ozaki_diff_sq += d_oz * d_oz;
    fp32_diff_sq += d_32 * d_32;
    ozaki_max_abs = std::max(ozaki_max_abs, std::abs(d_oz));
  }
  const double ref_norm = std::max(std::sqrt(ref_norm_sq), 1e-300);
  res.ozaki_rel_error = std::sqrt(ozaki_diff_sq) / ref_norm;
  res.ozaki_max_abs_error = ozaki_max_abs;
  res.fp32_rel_error = std::sqrt(fp32_diff_sq) / ref_norm;
  return res;
}

void syevd(int n, const double* A, double* w_out, double* U_out, Backend backend) {
  if (n <= 0) return;
  switch (backend) {
    case Backend::Host:
      syevd_host(n, A, w_out, U_out);
      return;
    case Backend::Cuda: {
#if defined(GPU4DFT_HAVE_CUDA)
      GpuLinalgContext ctx;
      auto& pool = ctx.pool();
      const std::size_t nn = static_cast<std::size_t>(n) * n;
      PooledBuffer<double> d_A(pool, nn);
      PooledBuffer<double> d_w(pool, static_cast<std::size_t>(n));
      PooledBuffer<double> d_U(pool, nn);
      copy_h2d_async(d_A.data(), A, nn * sizeof(double));
      syevd_device(ctx, n, d_A.data(), d_w.data(), d_U.data(), nullptr);
      copy_d2h_async(w_out, d_w.data(), static_cast<std::size_t>(n) * sizeof(double));
      copy_d2h_async(U_out, d_U.data(), nn * sizeof(double));
      synchronize_stream(nullptr);
      return;
#else
      throw std::runtime_error("syevd: CUDA backend requested, but GPU4DFT was built without CUDA");
#endif
    }
    default:
      throw NotImplemented("syevd: unsupported backend");
  }
}

CanonicalOrthoResult canonical_orthogonalization(int nao, const double* S, double lindep_tol,
                                                 Backend backend) {
  if (nao <= 0) throw std::invalid_argument("canonical_orthogonalization: nao must be positive");
  std::vector<double> w(static_cast<std::size_t>(nao));
  std::vector<double> U(static_cast<std::size_t>(nao) * nao);
  syevd(nao, S, w.data(), U.data(), backend);

  std::vector<int> kept_cols;
  kept_cols.reserve(static_cast<std::size_t>(nao));
  for (int j = 0; j < nao; ++j) {
    if (w[j] >= lindep_tol) {
      kept_cols.push_back(j);
    }
  }
  if (kept_cols.empty()) {
    throw std::runtime_error("canonical_orthogonalization: all overlap eigenvalues < lindep_tol");
  }

  const int nmo = static_cast<int>(kept_cols.size());
  CanonicalOrthoResult res;
  res.nao = nao;
  res.nmo = nmo;
  res.X.resize(static_cast<std::size_t>(nao) * nmo);
  res.eigenvalues.resize(static_cast<std::size_t>(nmo));

  std::vector<double> inv_sqrt_s(static_cast<std::size_t>(nmo));
  for (int k = 0; k < nmo; ++k) {
    const double s_val = w[kept_cols[k]];
    res.eigenvalues[k] = s_val;
    inv_sqrt_s[k] = 1.0 / std::sqrt(s_val);
  }

  for (int i = 0; i < nao; ++i) {
    for (int k = 0; k < nmo; ++k) {
      const int col = kept_cols[k];
      res.X[static_cast<std::size_t>(i) * nmo + k] =
          U[static_cast<std::size_t>(i) * nao + col] * inv_sqrt_s[k];
    }
  }
  return res;
}

FockDiagResult solve_fock_diag(int nao, int nmo, int n_occ, const double* F, const double* X,
                               Backend backend, GemmMode gemm_mode, int ozaki_slices) {
  if (nao <= 0 || nmo <= 0) {
    throw std::invalid_argument("solve_fock_diag: nao and nmo must be positive");
  }
  if (n_occ < 0 || n_occ > nmo) {
    throw std::invalid_argument("solve_fock_diag: n_occ out of range [0, nmo]");
  }

  FockDiagResult res;
  res.nao = nao;
  res.nmo = nmo;
  res.mo_energies.resize(static_cast<std::size_t>(nmo), 0.0);
  res.mo_coeff.resize(static_cast<std::size_t>(nao) * nmo, 0.0);
  res.dm.resize(static_cast<std::size_t>(nao) * nao, 0.0);
  res.energy_dm.resize(static_cast<std::size_t>(nao) * nao, 0.0);

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    GpuLinalgContext ctx;
    auto& pool = ctx.pool();
    const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
    const std::size_t nm = static_cast<std::size_t>(nao) * nmo;
    PooledBuffer<double> d_F(pool, n2);
    PooledBuffer<double> d_X(pool, nm);
    PooledBuffer<double> d_eps(pool, static_cast<std::size_t>(nmo));
    PooledBuffer<double> d_C(pool, nm);
    PooledBuffer<double> d_D(pool, n2);
    PooledBuffer<double> d_W(pool, n2);

    copy_h2d_async(d_F.data(), F, n2 * sizeof(double));
    copy_h2d_async(d_X.data(), X, nm * sizeof(double));

    solve_fock_diag_device(ctx, nao, nmo, n_occ, d_F.data(), d_X.data(), d_eps.data(), d_C.data(),
                           d_D.data(), d_W.data(), gemm_mode, nullptr, ozaki_slices);

    copy_d2h_async(res.mo_energies.data(), d_eps.data(),
                   static_cast<std::size_t>(nmo) * sizeof(double));
    copy_d2h_async(res.mo_coeff.data(), d_C.data(), nm * sizeof(double));
    copy_d2h_async(res.dm.data(), d_D.data(), n2 * sizeof(double));
    copy_d2h_async(res.energy_dm.data(), d_W.data(), n2 * sizeof(double));
    synchronize_stream(nullptr);
    return res;
#else
    throw std::runtime_error("solve_fock_diag: CUDA backend not available in this build");
#endif
  }

  if (backend != Backend::Host) {
    throw NotImplemented("solve_fock_diag: unsupported backend");
  }

  // Host path
  std::vector<double> FX(static_cast<std::size_t>(nao) * nmo, 0.0);
  std::vector<double> F_ortho(static_cast<std::size_t>(nmo) * nmo, 0.0);
  std::vector<double> C_ortho(static_cast<std::size_t>(nmo) * nmo, 0.0);

  gemm(nao, nmo, nao, 1.0, F, false, X, false, 0.0, FX.data(), Backend::Host, gemm_mode,
       ozaki_slices);
  gemm(nmo, nmo, nao, 1.0, X, true, FX.data(), false, 0.0, F_ortho.data(), Backend::Host,
       gemm_mode, ozaki_slices);
  symmetrize_host(nmo, F_ortho);

  syevd_host(nmo, F_ortho.data(), res.mo_energies.data(), C_ortho.data());

  gemm(nao, nmo, nmo, 1.0, X, false, C_ortho.data(), false, 0.0, res.mo_coeff.data(),
       Backend::Host, gemm_mode, ozaki_slices);

  if (n_occ > 0) {
    std::vector<double> C_occ(static_cast<std::size_t>(nao) * n_occ);
    std::vector<double> C_scaled(static_cast<std::size_t>(nao) * n_occ);
    for (int i = 0; i < nao; ++i) {
      for (int a = 0; a < n_occ; ++a) {
        const double c = res.mo_coeff[static_cast<std::size_t>(i) * nmo + a];
        C_occ[static_cast<std::size_t>(i) * n_occ + a] = c;
        C_scaled[static_cast<std::size_t>(i) * n_occ + a] = c * res.mo_energies[a];
      }
    }
    gemm(nao, nao, n_occ, 2.0, C_occ.data(), false, C_occ.data(), true, 0.0, res.dm.data(),
         Backend::Host, gemm_mode, ozaki_slices);
    gemm(nao, nao, n_occ, 2.0, C_occ.data(), false, C_scaled.data(), true, 0.0,
         res.energy_dm.data(), Backend::Host, gemm_mode, ozaki_slices);
    symmetrize_host(nao, res.dm);
    symmetrize_host(nao, res.energy_dm);
  }
  return res;
}

Sp2Result solve_fock_sp2(int nao, int nmo, int n_occ, const double* F, const double* X,
                         Backend backend, const Sp2Options& opts) {
  if (nao <= 0 || nmo <= 0) {
    throw std::invalid_argument("solve_fock_sp2: nao and nmo must be positive");
  }
  if (n_occ < 0 || n_occ > nmo) {
    throw std::invalid_argument("solve_fock_sp2: n_occ out of range [0, nmo]");
  }

  Sp2Result res;
  res.nao = nao;
  res.nmo = nmo;
  res.dm.resize(static_cast<std::size_t>(nao) * nao, 0.0);
  res.energy_dm.resize(static_cast<std::size_t>(nao) * nao, 0.0);

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    GpuLinalgContext ctx;
    auto& pool = ctx.pool();
    const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
    const std::size_t nm = static_cast<std::size_t>(nao) * nmo;
    const std::size_t m2 = static_cast<std::size_t>(nmo) * nmo;
    PooledBuffer<double> d_F(pool, n2);
    PooledBuffer<double> d_X(pool, nm);
    PooledBuffer<double> d_P(pool, m2);
    PooledBuffer<double> d_D(pool, n2);
    PooledBuffer<double> d_W(pool, n2);

    copy_h2d_async(d_F.data(), F, n2 * sizeof(double));
    copy_h2d_async(d_X.data(), X, nm * sizeof(double));

    const DeviceSp2Stats d_stats =
        solve_fock_sp2_device(ctx, nao, nmo, n_occ, d_F.data(), d_X.data(), d_P.data(), d_D.data(),
                              d_W.data(), opts, nullptr);

    copy_d2h_async(res.dm.data(), d_D.data(), n2 * sizeof(double));
    copy_d2h_async(res.energy_dm.data(), d_W.data(), n2 * sizeof(double));
    synchronize_stream(nullptr);

    res.iterations = d_stats.iterations;
    res.final_trace = d_stats.final_trace;
    res.idempotency_error = d_stats.idempotency_error;
    res.graph_captures = ctx.graph_executor().capture_count();
    res.graph_replays = ctx.graph_executor().replay_count();
    return res;
#else
    throw std::runtime_error("solve_fock_sp2: CUDA backend not available in this build");
#endif
  }

  if (backend != Backend::Host) {
    throw NotImplemented("solve_fock_sp2: unsupported backend");
  }

  // Host SP2 purification
  if (n_occ == 0) return res;
  const std::size_t m2 = static_cast<std::size_t>(nmo) * nmo;
  std::vector<double> FX(static_cast<std::size_t>(nao) * nmo, 0.0);
  std::vector<double> F_ortho(m2, 0.0);
  std::vector<double> P(m2, 0.0);
  std::vector<double> P2(m2, 0.0);

  if (n_occ == nmo) {
    for (int i = 0; i < nmo; ++i) P[static_cast<std::size_t>(i) * nmo + i] = 1.0;
    res.final_trace = static_cast<double>(nmo);
  } else {
    gemm(nao, nmo, nao, 1.0, F, false, X, false, 0.0, FX.data(), Backend::Host, opts.gemm_mode,
         opts.ozaki_slices);
    gemm(nmo, nmo, nao, 1.0, X, true, FX.data(), false, 0.0, F_ortho.data(), Backend::Host,
         opts.gemm_mode, opts.ozaki_slices);
    symmetrize_host(nmo, F_ortho);

    double emin = 1e300;
    double emax = -1e300;
    for (int i = 0; i < nmo; ++i) {
      double rad = 0.0;
      for (int j = 0; j < nmo; ++j) {
        if (j != i) rad += std::abs(F_ortho[static_cast<std::size_t>(i) * nmo + j]);
      }
      const double diag = F_ortho[static_cast<std::size_t>(i) * nmo + i];
      emin = std::min(emin, diag - rad);
      emax = std::max(emax, diag + rad);
    }
    const double span = std::max(emax - emin, 1e-12);
    const double pad = 1e-10 * span + 1e-12;
    emin -= pad;
    emax += pad;
    const double inv_span = 1.0 / (emax - emin);

    for (int i = 0; i < nmo; ++i) {
      for (int j = 0; j < nmo; ++j) {
        const double d_term = (i == j) ? emax : 0.0;
        P[static_cast<std::size_t>(i) * nmo + j] =
            (d_term - F_ortho[static_cast<std::size_t>(i) * nmo + j]) * inv_span;
      }
    }

    const double idem_tol =
        (opts.gemm_mode == GemmMode::FP32) ? std::max(opts.idempotency_tol, 1e-4) : opts.idempotency_tol;
    const double tr_tol =
        (opts.gemm_mode == GemmMode::FP32) ? std::max(opts.trace_tol, 1e-3) : opts.trace_tol;
    const double stag_floor = (opts.gemm_mode == GemmMode::FP32) ? 5e-2 : 1e-6;
    double prev_gamma = 1e300;

    for (int iter = 0; iter < opts.max_iter; ++iter) {
      gemm(nmo, nmo, nmo, 1.0, P.data(), false, P.data(), false, 0.0, P2.data(), Backend::Host,
           opts.gemm_mode, opts.ozaki_slices);
      double tau = 0.0;
      double tau2 = 0.0;
      for (int i = 0; i < nmo; ++i) {
        tau += P[static_cast<std::size_t>(i) * nmo + i];
        tau2 += P2[static_cast<std::size_t>(i) * nmo + i];
      }
      const double gamma = std::abs(tau - tau2);
      const double tr_err = std::abs(tau - static_cast<double>(n_occ));

      res.final_trace = tau;
      res.idempotency_error = gamma;
      if (iter >= 2 &&
          ((gamma <= idem_tol && tr_err <= tr_tol) ||
           (gamma <= stag_floor && tr_err <= 1e-2 && gamma >= prev_gamma * 0.995))) {
        break;
      }

      if (tau < static_cast<double>(n_occ)) {
        for (std::size_t idx = 0; idx < m2; ++idx) P[idx] = 2.0 * P[idx] - P2[idx];
        res.final_trace = 2.0 * tau - tau2;
      } else {
        P = P2;
        res.final_trace = tau2;
      }
      prev_gamma = gamma;
      res.iterations = iter + 1;
    }

    if (opts.mcweeny_polish && opts.gemm_mode != GemmMode::FP32) {
      std::vector<double> P3(m2, 0.0);
      gemm(nmo, nmo, nmo, 1.0, P.data(), false, P.data(), false, 0.0, P2.data(), Backend::Host,
           opts.gemm_mode, opts.ozaki_slices);
      gemm(nmo, nmo, nmo, 1.0, P2.data(), false, P.data(), false, 0.0, P3.data(), Backend::Host,
           opts.gemm_mode, opts.ozaki_slices);
      for (std::size_t idx = 0; idx < m2; ++idx) {
        P[idx] = 3.0 * P2[idx] - 2.0 * P3[idx];
      }
    }
    symmetrize_host(nmo, P);
  }

  gemm(nao, nmo, nmo, 1.0, X, false, P.data(), false, 0.0, FX.data(), Backend::Host, opts.gemm_mode,
       opts.ozaki_slices);
  gemm(nao, nao, nmo, 2.0, FX.data(), false, X, true, 0.0, res.dm.data(), Backend::Host,
       opts.gemm_mode, opts.ozaki_slices);
  symmetrize_host(nao, res.dm);

  std::vector<double> DF(static_cast<std::size_t>(nao) * nao, 0.0);
  gemm(nao, nao, nao, 1.0, res.dm.data(), false, F, false, 0.0, DF.data(), Backend::Host,
       opts.gemm_mode, opts.ozaki_slices);
  gemm(nao, nao, nao, 0.5, DF.data(), false, res.dm.data(), false, 0.0, res.energy_dm.data(),
       Backend::Host, opts.gemm_mode, opts.ozaki_slices);
  symmetrize_host(nao, res.energy_dm);
  return res;
}

// ---------------------------------------------------------------------------
// Host-only Build Stubs when GPU4DFT_HAVE_CUDA is OFF
// ---------------------------------------------------------------------------
#if !defined(GPU4DFT_HAVE_CUDA)

struct DeviceMemoryPool::Impl {
  int device_id = 0;
};
DeviceMemoryPool::DeviceMemoryPool(int dev) : impl_(std::make_unique<Impl>()) {
  impl_->device_id = dev < 0 ? 0 : dev;
}
DeviceMemoryPool::~DeviceMemoryPool() = default;
DeviceMemoryPool::DeviceMemoryPool(DeviceMemoryPool&&) noexcept = default;
DeviceMemoryPool& DeviceMemoryPool::operator=(DeviceMemoryPool&&) noexcept = default;
void* DeviceMemoryPool::acquire(std::size_t bytes, StreamHandle) {
  if (bytes == 0) return nullptr;
  void* ptr = std::malloc(align_bytes(bytes));
  if (!ptr) throw std::bad_alloc();
  return ptr;
}
void DeviceMemoryPool::release(void* ptr, StreamHandle) noexcept { std::free(ptr); }
void DeviceMemoryPool::reserve_arena(std::size_t, StreamHandle) {}
void* DeviceMemoryPool::alloc_arena(std::size_t bytes) { return acquire(bytes, nullptr); }
void DeviceMemoryPool::reset_arena() noexcept {}
void DeviceMemoryPool::trim(StreamHandle) {}
int DeviceMemoryPool::device_id() const noexcept { return impl_ ? impl_->device_id : 0; }
PoolStats DeviceMemoryPool::stats() const { return PoolStats{}; }
DeviceMemoryPool& default_device_pool(int) {
  static DeviceMemoryPool pool(0);
  return pool;
}

void copy_h2d_async(void* dst, const void* src, std::size_t n, StreamHandle) {
  if (n > 0) std::memcpy(dst, src, n);
}
void copy_d2h_async(void* dst, const void* src, std::size_t n, StreamHandle) {
  if (n > 0) std::memcpy(dst, src, n);
}
void copy_d2d_async(void* dst, const void* src, std::size_t n, StreamHandle) {
  if (n > 0 && dst != src) std::memcpy(dst, src, n);
}
void memset_zero_async(void* dst, std::size_t n, StreamHandle) {
  if (n > 0) std::memset(dst, 0, n);
}
void synchronize_stream(StreamHandle) {}

struct CudaGraphExecutor::Impl {
  std::size_t fallbacks = 0;
};
CudaGraphExecutor::CudaGraphExecutor(bool) : impl_(std::make_unique<Impl>()) {}
CudaGraphExecutor::~CudaGraphExecutor() = default;
CudaGraphExecutor::CudaGraphExecutor(CudaGraphExecutor&&) noexcept = default;
CudaGraphExecutor& CudaGraphExecutor::operator=(CudaGraphExecutor&&) noexcept = default;
void CudaGraphExecutor::execute(std::uint64_t, const std::function<void(StreamHandle)>& fn) {
  fn(nullptr);
  impl_->fallbacks += 1;
}
bool CudaGraphExecutor::capture(std::uint64_t, const std::function<void(StreamHandle)>&) {
  return false;
}
void CudaGraphExecutor::launch(std::uint64_t) {
  throw std::runtime_error("CudaGraphExecutor: built without CUDA");
}
bool CudaGraphExecutor::is_captured(std::uint64_t) const { return false; }
void CudaGraphExecutor::invalidate(std::uint64_t) {}
void CudaGraphExecutor::clear() {}
void CudaGraphExecutor::synchronize() {}
StreamHandle CudaGraphExecutor::stream() const noexcept { return nullptr; }
std::size_t CudaGraphExecutor::capture_count() const noexcept { return 0; }
std::size_t CudaGraphExecutor::replay_count() const noexcept { return 0; }
std::size_t CudaGraphExecutor::fallback_count() const noexcept { return impl_->fallbacks; }

struct GpuLinalgContext::Impl {
  DeviceMemoryPool* pool = nullptr;
  CudaGraphExecutor graph_exec{true};
};
GpuLinalgContext::GpuLinalgContext(DeviceMemoryPool* p, int) : impl_(std::make_unique<Impl>()) {
  impl_->pool = p ? p : &default_device_pool(0);
}
GpuLinalgContext::~GpuLinalgContext() = default;
GpuLinalgContext::GpuLinalgContext(GpuLinalgContext&&) noexcept = default;
GpuLinalgContext& GpuLinalgContext::operator=(GpuLinalgContext&&) noexcept = default;
DeviceMemoryPool& GpuLinalgContext::pool() noexcept { return *impl_->pool; }
CudaGraphExecutor& GpuLinalgContext::graph_executor() noexcept { return impl_->graph_exec; }
void GpuLinalgContext::set_stream(StreamHandle) {}
StreamHandle GpuLinalgContext::stream() const noexcept { return nullptr; }
int GpuLinalgContext::device_id() const noexcept { return 0; }
void* GpuLinalgContext::cublas_raw() const noexcept { return nullptr; }
void* GpuLinalgContext::cusolver_raw() const noexcept { return nullptr; }

void gemm_device(GpuLinalgContext&, int, int, int, double, const double*, int, bool, const double*,
                 int, bool, double, double*, int, GemmMode, StreamHandle, int) {
  throw std::runtime_error("gemm_device: built without CUDA");
}
void syevd_device(GpuLinalgContext&, int, const double*, double*, double*, StreamHandle) {
  throw std::runtime_error("syevd_device: built without CUDA");
}
void solve_fock_diag_device(GpuLinalgContext&, int, int, int, const double*, const double*, double*,
                            double*, double*, double*, GemmMode, StreamHandle, int) {
  throw std::runtime_error("solve_fock_diag_device: built without CUDA");
}
DeviceSp2Stats solve_fock_sp2_device(GpuLinalgContext&, int, int, int, const double*, const double*,
                                     double*, double*, double*, const Sp2Options&, StreamHandle) {
  throw std::runtime_error("solve_fock_sp2_device: built without CUDA");
}
void symmetrize_matrix_device(int, double*, StreamHandle) {}
void axpby_device(std::size_t, double, const double*, double, double*, StreamHandle) {}
void compute_delta_and_norm_device(std::size_t, const double*, const double*, double*, double*,
                                   StreamHandle) {}
void assemble_fock_and_traces_device(int, const double*, const double*, const double*,
                                     const double*, double, const double*, double*, double*,
                                     StreamHandle) {}
void apply_level_shift_ao_device(GpuLinalgContext&, int, double, const double*, const double*,
                                 double*, GemmMode, StreamHandle) {}

#endif

}  // namespace gpu4dft
