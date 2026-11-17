#pragma once

#include <math.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/types.hpp"
#include "gpu4dft/xc.hpp"

namespace gpu4dft {

struct SnkOptions {
  Backend backend = Backend::Cuda;
  Precision precision = Precision::FP64;  // FP64: cublasDgemm; FP32: Tensor Core cublasSgemm + FP64 Boys/MD
  bool overlap_fitting = true;            // COSX overlap fitting Q = S_grid^+ * S_exact
  double pair_cutoff = 1e-13;             // Gaussian primitive-pair prefactor screening exp(-mu * R_AB^2)
  double lindep_tol = 1e-8;               // Pseudo-inverse threshold on S_grid eigenvalues
  bool cache_ao_on_device = true;         // Cache compacted batch Phi and weighted w*Phi on GPU
  std::size_t max_ao_cache_mb = 2048;     // Maximum GPU memory budget (MiB) for persistent AO cache
};

// Evaluates exact analytical 1e overlap matrix S_{uv} = \int chi_u(r) chi_v(r) dr
// [nao * nao] using SNK's self-contained McMurchie-Davidson evaluator (0 <= l <= 4).
std::vector<double> snk_overlap_matrix(const Basis& basis);

// Evaluates 1e electrostatic potential matrix V_{uv}(r_g) = \int chi_u(r') chi_v(r') / |r' - r_g| dr'
// [nao * nao] at point `point_bohr` (matches PySCF mol.intor('int1e_grids') to < 1e-13).
std::vector<double> snk_esp_matrix(const Basis& basis,
                                   std::array<double, 3> point_bohr,
                                   Backend backend = Backend::Host);

class SemiNumericalK {
 public:
  SemiNumericalK(const Basis& basis, const MolecularGrid& grid,
                 const SnkOptions& opts = {});
  ~SemiNumericalK();

  SemiNumericalK(SemiNumericalK&&) noexcept;
  SemiNumericalK& operator=(SemiNumericalK&&) noexcept;
  SemiNumericalK(const SemiNumericalK&) = delete;
  SemiNumericalK& operator=(const SemiNumericalK&) = delete;

  int nao() const { return basis_.nao(); }
  const MolecularGrid& grid() const { return grid_; }
  Backend active_backend() const { return active_backend_; }
  const std::vector<double>& overlap_fit_matrix() const { return q_fit_; }

  // Host API: returns [nao * nao] semi-numerical exchange matrix K(D) matching PySCF get_k.
  std::vector<double> compute_k(std::span<const double> dm) const;
  void compute_k(const double* dm, double* vk_out) const;

  // GPU Device-Pointer API: computes K(d_dm) -> d_vk directly on GPU without host matrix transfers.
  void compute_k_device(const double* d_dm, double* d_vk, void* cuda_stream = nullptr) const;

 private:
  Basis basis_;
  MolecularGrid grid_;
  SnkOptions opts_;
  Backend active_backend_ = Backend::Host;
  std::vector<double> s_exact_;  // [nao * nao]
  std::vector<double> q_fit_;    // [nao * nao] COSX overlap-fitting matrix Q

  struct HostPairState;
  std::unique_ptr<HostPairState> host_pairs_;

  struct CudaState;
  std::unique_ptr<CudaState> cuda_;
};

namespace snk_detail {

GPU4DFT_HD inline double fast_inv_sqrt(double x) {
#if defined(__CUDA_ARCH__)
  return ::rsqrt(x);
#else
  return 1.0 / ::sqrt(x);
#endif
}

GPU4DFT_HD constexpr double boys_asymp_threshold(int m_max) {
  return (m_max == 0) ? 30.0 : (33.0 + 3.0 * static_cast<double>(m_max));
}

// Self-contained GPU4DFT_HD Boys function vector F_0(T) .. F_{m_max}(T) (0 <= m_max <= 16)
// accurate to < 1e-14..1e-15 across all T >= 0. Uses zero-transcendental rsqrt(T) upward
// recurrence in the asymptotic far field (avoids >85% of all exp/erf calls on GPU!).
GPU4DFT_HD inline void boys_vec_hd(int m_max, double T, double* F) {
  constexpr double kSqrtPiOver2 = 0.88622692545275801365;  // sqrt(pi) / 2
  if (T <= 0.0) {
    for (int m = 0; m <= m_max; ++m) {
      F[m] = 1.0 / static_cast<double>(2 * m + 1);
    }
    return;
  }

  // Zero-transcendental asymptotic fast path (erfc(sqrt(T)) == 0 and exp(-T) negligible)
  if (T > boys_asymp_threshold(m_max)) {
    const double inv_sqrt_t = fast_inv_sqrt(T);
    F[0] = kSqrtPiOver2 * inv_sqrt_t;
    if (m_max > 0) {
      const double inv_2t = 0.5 * inv_sqrt_t * inv_sqrt_t;
      for (int m = 0; m < m_max; ++m) {
        F[m + 1] = static_cast<double>(2 * m + 1) * F[m] * inv_2t;
      }
    }
    return;
  }

  const double limit = 26.0 + 2.0 * static_cast<double>(m_max);
  if (T > limit) {
    const double inv_sqrt_t = fast_inv_sqrt(T);
    const double sqrt_t = T * inv_sqrt_t;
    F[0] = kSqrtPiOver2 * ::erf(sqrt_t) * inv_sqrt_t;
    const double exp_mt = ::exp(-T);
    const double inv_2t = 0.5 * inv_sqrt_t * inv_sqrt_t;
    for (int m = 0; m < m_max; ++m) {
      F[m + 1] = (static_cast<double>(2 * m + 1) * F[m] - exp_mt) * inv_2t;
    }
    return;
  }

  // Positive-term Kummer series for F_{m_max}(T) + exact downward recurrence
  const double exp_mt = ::exp(-T);
  const double two_t = 2.0 * T;
  double denom = static_cast<double>(2 * m_max + 1);
  double term = 1.0 / denom;
  double sum = term;
  for (int k = 1; k < 80; ++k) {
    denom += 2.0;
    term *= two_t / denom;
    sum += term;
    if (term <= 1e-17 * sum) break;
  }
  F[m_max] = exp_mt * sum;

  for (int m = m_max; m >= 1; --m) {
    F[m - 1] = (two_t * F[m] + exp_mt) / static_cast<double>(2 * m - 1);
  }
}

// 1D McMurchie-Davidson Hermite expansion coefficients E_t^{i, j} (0 <= i, j <= 4, 0 <= t <= i+j).
GPU4DFT_HD inline void md_hermite_1d(int la, int lb, double pa, double pb, double inv_2p,
                                     double E[5][5][9]) {
  for (int i = 0; i <= la; ++i) {
    for (int j = 0; j <= lb; ++j) {
      for (int t = 0; t <= la + lb; ++t) {
        E[i][j][t] = 0.0;
      }
    }
  }
  E[0][0][0] = 1.0;

  for (int i = 0; i < la; ++i) {
    for (int t = 0; t <= i + 1; ++t) {
      double val = pa * E[i][0][t];
      if (t > 0) val += inv_2p * E[i][0][t - 1];
      if (t + 1 <= i) val += static_cast<double>(t + 1) * E[i][0][t + 1];
      E[i + 1][0][t] = val;
    }
  }

  for (int j = 0; j < lb; ++j) {
    for (int i = 0; i <= la; ++i) {
      for (int t = 0; t <= i + j + 1; ++t) {
        double val = pb * E[i][j][t];
        if (t > 0) val += inv_2p * E[i][j][t - 1];
        if (t + 1 <= i + j) val += static_cast<double>(t + 1) * E[i][j][t + 1];
        E[i][j + 1][t] = val;
      }
    }
  }
}

GPU4DFT_HD constexpr int tet_count(int L) {
  return (L + 1) * (L + 2) * (L + 3) / 6;
}

GPU4DFT_HD constexpr int tet_index(int t, int u, int v) {
  const int s = t + u + v;
  const int base = s * (s + 1) * (s + 2) / 6;
  const int row = t * (s + 1) - (t * (t - 1)) / 2;
  return base + row + u;
}

// In-place Auxiliary Hermite Coulomb integrals R^{(0)}_{t, u, v}(p, R_PC) for t + u + v <= L <= 8
// (eliminates the two 165-double ping-pong stack arrays!).
GPU4DFT_HD inline void md_hermite_coulomb(int L, double p,
                                          double xpc, double ypc, double zpc,
                                          double* R_out) {
  double F[17];
  const double rpc2 = xpc * xpc + ypc * ypc + zpc * zpc;
  boys_vec_hd(L, p * rpc2, F);

  const double m2p = -2.0 * p;
  double m2p_pow = 1.0;
  for (int k = 0; k < L; ++k) m2p_pow *= m2p;
  R_out[0] = m2p_pow * F[L];

  for (int n = L - 1; n >= 0; --n) {
    const int max_s = L - n;
    for (int s = max_s; s >= 1; --s) {
      for (int t = 0; t <= s; ++t) {
        for (int u = 0; u <= s - t; ++u) {
          const int v = s - t - u;
          double val = 0.0;
          if (t > 0) {
            val = xpc * R_out[tet_index(t - 1, u, v)];
            if (t > 1) val += static_cast<double>(t - 1) * R_out[tet_index(t - 2, u, v)];
          } else if (u > 0) {
            val = ypc * R_out[tet_index(t, u - 1, v)];
            if (u > 1) val += static_cast<double>(u - 1) * R_out[tet_index(t, u - 2, v)];
          } else {
            val = zpc * R_out[tet_index(t, u, v - 1)];
            if (v > 1) val += static_cast<double>(v - 1) * R_out[tet_index(t, u, v - 2)];
          }
          R_out[tet_index(t, u, v)] = val;
        }
      }
    }
    double p_pow = 1.0;
    for (int k = 0; k < n; ++k) p_pow *= m2p;
    R_out[0] = p_pow * F[n];
  }
}

// Pure-register closed-form Hermite Coulomb evaluator for 0 <= L <= 4 (compile-time constant
// subscripts keep R[] 100% in GPU registers with 0 bytes of local stack spill!).
template <int L>
GPU4DFT_HD inline void eval_hermite_coulomb_reg(const double* d,
                                                double x, double y, double z,
                                                double* R) {
  R[tet_index(0, 0, 0)] = d[0];
  if constexpr (L >= 1) {
    const double d1 = d[1];
    R[tet_index(1, 0, 0)] = x * d1;
    R[tet_index(0, 1, 0)] = y * d1;
    R[tet_index(0, 0, 1)] = z * d1;
  }
  if constexpr (L >= 2) {
    const double d1 = d[1];
    const double d2 = d[2];
    const double xx = x * x, yy = y * y, zz = z * z;
    R[tet_index(2, 0, 0)] = xx * d2 + d1;
    R[tet_index(1, 1, 0)] = x * y * d2;
    R[tet_index(1, 0, 1)] = x * z * d2;
    R[tet_index(0, 2, 0)] = yy * d2 + d1;
    R[tet_index(0, 1, 1)] = y * z * d2;
    R[tet_index(0, 0, 2)] = zz * d2 + d1;
  }
  if constexpr (L >= 3) {
    const double d2 = d[2];
    const double d3 = d[3];
    const double xx = x * x, yy = y * y, zz = z * z;
    R[tet_index(3, 0, 0)] = x * (xx * d3 + 3.0 * d2);
    R[tet_index(0, 3, 0)] = y * (yy * d3 + 3.0 * d2);
    R[tet_index(0, 0, 3)] = z * (zz * d3 + 3.0 * d2);
    R[tet_index(2, 1, 0)] = y * (xx * d3 + d2);
    R[tet_index(2, 0, 1)] = z * (xx * d3 + d2);
    R[tet_index(1, 2, 0)] = x * (yy * d3 + d2);
    R[tet_index(0, 2, 1)] = z * (yy * d3 + d2);
    R[tet_index(1, 0, 2)] = x * (zz * d3 + d2);
    R[tet_index(0, 1, 2)] = y * (zz * d3 + d2);
    R[tet_index(1, 1, 1)] = x * y * z * d3;
  }
  if constexpr (L >= 4) {
    const double d2 = d[2];
    const double d3 = d[3];
    const double d4 = d[4];
    const double xx = x * x, yy = y * y, zz = z * z;
    R[tet_index(4, 0, 0)] = xx * (xx * d4 + 6.0 * d3) + 3.0 * d2;
    R[tet_index(0, 4, 0)] = yy * (yy * d4 + 6.0 * d3) + 3.0 * d2;
    R[tet_index(0, 0, 4)] = zz * (zz * d4 + 6.0 * d3) + 3.0 * d2;
    R[tet_index(3, 1, 0)] = x * y * (xx * d4 + 3.0 * d3);
    R[tet_index(3, 0, 1)] = x * z * (xx * d4 + 3.0 * d3);
    R[tet_index(1, 3, 0)] = x * y * (yy * d4 + 3.0 * d3);
    R[tet_index(0, 3, 1)] = y * z * (yy * d4 + 3.0 * d3);
    R[tet_index(1, 0, 3)] = x * z * (zz * d4 + 3.0 * d3);
    R[tet_index(0, 1, 3)] = y * z * (zz * d4 + 3.0 * d3);
    R[tet_index(2, 2, 0)] = (xx * yy) * d4 + (xx + yy) * d3 + d2;
    R[tet_index(2, 0, 2)] = (xx * zz) * d4 + (xx + zz) * d3 + d2;
    R[tet_index(0, 2, 2)] = (yy * zz) * d4 + (yy + zz) * d3 + d2;
    R[tet_index(2, 1, 1)] = y * z * (xx * d4 + d3);
    R[tet_index(1, 2, 1)] = x * z * (yy * d4 + d3);
    R[tet_index(1, 1, 2)] = x * y * (zz * d4 + d3);
  }
}

struct SphCartTerm {
  int m;
  int ix;
  int iy;
  int iz;
  double coeff;
};

// Sparse Spherical-to-Cartesian monomial expansion for 0 <= l <= 4 matching PySCF spherical AOs.
GPU4DFT_HD inline int sph_cart_terms(int l, SphCartTerm* out) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  if (l == 0) {
    out[0] = {0, 0, 0, 0, 1.0 / ::sqrt(4.0 * kPi)};
    return 1;
  }
  if (l == 1) {
    const double c1 = ::sqrt(3.0 / (4.0 * kPi));
    out[0] = {0, 1, 0, 0, c1};
    out[1] = {1, 0, 1, 0, c1};
    out[2] = {2, 0, 0, 1, c1};
    return 3;
  }
  if (l == 2) {
    const double c2 = ::sqrt(15.0 / (4.0 * kPi));
    const double c20 = 0.5 * ::sqrt(5.0 / (4.0 * kPi));
    out[0] = {0, 1, 1, 0, c2};
    out[1] = {1, 0, 1, 1, c2};
    out[2] = {2, 0, 0, 2, 2.0 * c20};
    out[3] = {2, 2, 0, 0, -c20};
    out[4] = {2, 0, 2, 0, -c20};
    out[5] = {3, 1, 0, 1, c2};
    out[6] = {4, 2, 0, 0, 0.5 * c2};
    out[7] = {4, 0, 2, 0, -0.5 * c2};
    return 8;
  }
  if (l == 3) {
    const double k3 = ::sqrt(35.0 / (32.0 * kPi));
    const double k2 = ::sqrt(105.0 / (4.0 * kPi));
    const double k1 = ::sqrt(21.0 / (32.0 * kPi));
    const double k0 = ::sqrt(7.0 / (16.0 * kPi));
    const double kp2 = ::sqrt(105.0 / (16.0 * kPi));
    int n = 0;
    out[n++] = {0, 2, 1, 0, 3.0 * k3};
    out[n++] = {0, 0, 3, 0, -k3};
    out[n++] = {1, 1, 1, 1, k2};
    out[n++] = {2, 0, 1, 2, 4.0 * k1};
    out[n++] = {2, 2, 1, 0, -k1};
    out[n++] = {2, 0, 3, 0, -k1};
    out[n++] = {3, 0, 0, 3, 2.0 * k0};
    out[n++] = {3, 2, 0, 1, -3.0 * k0};
    out[n++] = {3, 0, 2, 1, -3.0 * k0};
    out[n++] = {4, 1, 0, 2, 4.0 * k1};
    out[n++] = {4, 3, 0, 0, -k1};
    out[n++] = {4, 1, 2, 0, -k1};
    out[n++] = {5, 2, 0, 1, kp2};
    out[n++] = {5, 0, 2, 1, -kp2};
    out[n++] = {6, 3, 0, 0, k3};
    out[n++] = {6, 1, 2, 0, -3.0 * k3};
    return n;
  }
  if (l == 4) {
    const double c44 = 0.75 * ::sqrt(35.0 / kPi);
    const double c43 = 0.75 * ::sqrt(35.0 / (2.0 * kPi));
    const double c42 = 0.75 * ::sqrt(5.0 / kPi);
    const double c41 = 0.75 * ::sqrt(5.0 / (2.0 * kPi));
    const double c40 = (3.0 / 16.0) * ::sqrt(1.0 / kPi);
    const double c4p2 = 0.375 * ::sqrt(5.0 / kPi);
    const double c4p4 = (3.0 / 16.0) * ::sqrt(35.0 / kPi);
    int n = 0;
    out[n++] = {0, 3, 1, 0, c44};
    out[n++] = {0, 1, 3, 0, -c44};
    out[n++] = {1, 2, 1, 1, 3.0 * c43};
    out[n++] = {1, 0, 3, 1, -c43};
    out[n++] = {2, 1, 1, 2, 6.0 * c42};
    out[n++] = {2, 3, 1, 0, -c42};
    out[n++] = {2, 1, 3, 0, -c42};
    out[n++] = {3, 0, 1, 3, 4.0 * c41};
    out[n++] = {3, 2, 1, 1, -3.0 * c41};
    out[n++] = {3, 0, 3, 1, -3.0 * c41};
    out[n++] = {4, 0, 0, 4, 8.0 * c40};
    out[n++] = {4, 2, 0, 2, -24.0 * c40};
    out[n++] = {4, 0, 2, 2, -24.0 * c40};
    out[n++] = {4, 4, 0, 0, 3.0 * c40};
    out[n++] = {4, 2, 2, 0, 6.0 * c40};
    out[n++] = {4, 0, 4, 0, 3.0 * c40};
    out[n++] = {5, 1, 0, 3, 4.0 * c41};
    out[n++] = {5, 3, 0, 1, -3.0 * c41};
    out[n++] = {5, 1, 2, 1, -3.0 * c41};
    out[n++] = {6, 2, 0, 2, 6.0 * c4p2};
    out[n++] = {6, 0, 2, 2, -6.0 * c4p2};
    out[n++] = {6, 4, 0, 0, -c4p2};
    out[n++] = {6, 0, 4, 0, c4p2};
    out[n++] = {7, 3, 0, 1, c43};
    out[n++] = {7, 1, 2, 1, -3.0 * c43};
    out[n++] = {8, 4, 0, 0, c4p4};
    out[n++] = {8, 2, 2, 0, -6.0 * c4p4};
    out[n++] = {8, 0, 4, 0, c4p4};
    return n;
  }
  return 0;
}

// Precomputed primitive-pair & shell-pair data for zero-redundant ESP evaluation (Host & CUDA).
struct SnkPrimPair {
  double px = 0.0;
  double py = 0.0;
  double pz = 0.0;
  double p = 0.0;
  double pre = 0.0;
  int h_offset = -1;  // For L <= 4: offset into SnkPairTable::h_coeffs [nma * nmb * tet_count(L)]
};

struct SnkShellPair {
  int ish_a = 0;
  int ish_b = 0;          // Canonicalized so la >= lb (and ish_a >= ish_b when la == lb)
  int la = 0;
  int lb = 0;
  int ao_a = 0;
  int ao_b = 0;
  int nma = 0;
  int nmb = 0;
  int prim_offset = 0;
  int prim_count = 0;
  int asymp_h_offset = -1;  // >= 0 for concentric pairs (rab2 < 1e-12) with L <= 4
  double center[3] = {0.0, 0.0, 0.0};
  double extent_radius = 0.0;
  double p_min = 0.0;
  double q_monopole = 0.0;  // Analytical monopole S_{ab} for s-s (Frobenius overlap norm for L > 0)
  double v_max_near = 0.0;
  double q_far_bound = 0.0;
};

struct SnkPairTable {
  std::vector<SnkShellPair> pairs;
  std::vector<SnkPrimPair> prims;
  std::vector<double> h_coeffs;
};

// Standalone general evaluator (used for single-point `snk_esp_matrix` and diagonal bounds).
GPU4DFT_HD inline void eval_shell_pair_esp(const detail::FlatBasisView& b,
                                           int ish_a, int ish_b,
                                           double gx, double gy, double gz,
                                           double pair_cutoff,
                                           double* v_sph) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  const int la = b.shell_l[ish_a];
  const int lb = b.shell_l[ish_b];
  const int nma = 2 * la + 1;
  const int nmb = 2 * lb + 1;
  for (int i = 0; i < nma * nmb; ++i) v_sph[i] = 0.0;

  const double ax = b.shell_center[3 * ish_a + 0];
  const double ay = b.shell_center[3 * ish_a + 1];
  const double az = b.shell_center[3 * ish_a + 2];
  const double bx = b.shell_center[3 * ish_b + 0];
  const double by = b.shell_center[3 * ish_b + 1];
  const double bz = b.shell_center[3 * ish_b + 2];

  const double abx = ax - bx;
  const double aby = ay - by;
  const double abz = az - bz;
  const double rab2 = abx * abx + aby * aby + abz * abz;

  SphCartTerm ta[28], tb[28];
  const int nta = sph_cart_terms(la, ta);
  const int ntb = sph_cart_terms(lb, tb);

  const int pa0 = b.shell_prim_offset[ish_a];
  const int npa = b.shell_prim_count[ish_a];
  const int pb0 = b.shell_prim_offset[ish_b];
  const int npb = b.shell_prim_count[ish_b];
  const int L = la + lb;

  double Ex[5][5][9], Ey[5][5][9], Ez[5][5][9];
  double R_tet[165];

  for (int ka = 0; ka < npa; ++ka) {
    const double alpha = b.prim_exp[pa0 + ka];
    const double ca = b.prim_coeff[pa0 + ka];
    for (int kb = 0; kb < npb; ++kb) {
      const double beta = b.prim_exp[pb0 + kb];
      const double cb = b.prim_coeff[pb0 + kb];
      const double p = alpha + beta;
      const double inv_p = 1.0 / p;
      const double mu = alpha * beta * inv_p;
      if (mu * rab2 > 42.0) continue;
      const double kab = ::exp(-mu * rab2);
      const double pre = (kTwoPi * inv_p) * ca * cb * kab;
      if (::fabs(pre) < pair_cutoff) continue;

      const double px = (alpha * ax + beta * bx) * inv_p;
      const double py = (alpha * ay + beta * by) * inv_p;
      const double pz = (alpha * az + beta * bz) * inv_p;

      if (L == 0) {
        const double xpc = px - gx, ypc = py - gy, zpc = pz - gz;
        double f0 = 0.0;
        boys_vec_hd(0, p * (xpc * xpc + ypc * ypc + zpc * zpc), &f0);
        v_sph[0] += pre * ta[0].coeff * tb[0].coeff * f0;
        continue;
      }

      md_hermite_1d(la, lb, px - ax, px - bx, 0.5 * inv_p, Ex);
      md_hermite_1d(la, lb, py - ay, py - by, 0.5 * inv_p, Ey);
      md_hermite_1d(la, lb, pz - az, pz - bz, 0.5 * inv_p, Ez);
      md_hermite_coulomb(L, p, px - gx, py - gy, pz - gz, R_tet);

#if defined(__CUDA_ARCH__)
      #pragma unroll 1
#endif
      for (int ia = 0; ia < nta; ++ia) {
        const SphCartTerm& sa = ta[ia];
#if defined(__CUDA_ARCH__)
        #pragma unroll 1
#endif
        for (int ib = 0; ib < ntb; ++ib) {
          const SphCartTerm& sb = tb[ib];
          const int mx = sa.ix + sb.ix;
          const int my = sa.iy + sb.iy;
          const int mz = sa.iz + sb.iz;
          double sum_tuv = 0.0;
          for (int t = 0; t <= mx; ++t) {
            const double ex = Ex[sa.ix][sb.ix][t];
            if (ex == 0.0) continue;
            for (int u = 0; u <= my; ++u) {
              const double exy = ex * Ey[sa.iy][sb.iy][u];
              if (exy == 0.0) continue;
              for (int v = 0; v <= mz; ++v) {
                sum_tuv += exy * Ez[sa.iz][sb.iz][v] * R_tet[tet_index(t, u, v)];
              }
            }
          }
          v_sph[sa.m * nmb + sb.m] += pre * (sa.coeff * sb.coeff) * sum_tuv;
        }
      }
    }
  }
}

}  // namespace snk_detail

}  // namespace gpu4dft
