#pragma once

#include "gpu4dft/boys.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/math.hpp"

namespace gpu4dft {

namespace detail {

GPU4DFT_HD inline double rys_abs(double x) { return ::fabs(x); }
GPU4DFT_HD inline double rys_copysign(double x, double y) { return ::copysign(x, y); }

// Implicitly shifted symmetric tridiagonal QL algorithm (Golub-Welsch).
// Computes all eigenvalues `d[0..n-1]` (sorted ascending) and the first-row
// components `z0[0..n-1]` of the normalized eigenvectors of the symmetric
// tridiagonal matrix with diagonal `d[0..n-1]` and subdiagonal `e[0..n-2]`
// (`e[n-1]` is scratch, must be initialized to 0).
template <int MaxN>
GPU4DFT_HD inline void sym_tridiag_ql(int n, double* d, double* e, double* z0) {
  for (int i = 0; i < n; ++i) z0[i] = (i == 0) ? 1.0 : 0.0;
  if (n <= 1) return;
  e[n - 1] = 0.0;

  for (int l = 0; l < n; ++l) {
    int iter = 0;
    while (true) {
      int m = l;
      for (; m < n - 1; ++m) {
        const double dd = rys_abs(d[m]) + rys_abs(d[m + 1]);
        if (rys_abs(e[m]) <= 1e-15 * dd) break;
      }
      if (m == l) break;
      if (++iter > 60) break;

      double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
      double r = math::sqrt(g * g + 1.0);
      g = d[m] - d[l] + e[l] / (g + rys_copysign(r, g));
      double s = 1.0;
      double c = 1.0;
      double p = 0.0;
      for (int i = m - 1; i >= l; --i) {
        const double f = s * e[i];
        const double b = c * e[i];
        r = math::sqrt(f * f + g * g);
        e[i + 1] = r;
        if (r == 0.0) {
          d[i + 1] -= p;
          e[m] = 0.0;
          break;
        }
        s = f / r;
        c = g / r;
        g = d[i + 1] - p;
        r = (d[i] - g) * s + 2.0 * c * b;
        p = s * r;
        d[i + 1] = g + p;
        g = c * r - b;

        const double z_next = z0[i + 1];
        const double z_curr = z0[i];
        z0[i + 1] = s * z_curr + c * z_next;
        z0[i] = c * z_curr - s * z_next;
      }
      if (r == 0.0 && e[m] == 0.0) continue;
      d[l] -= p;
      e[l] = g;
      e[m] = 0.0;
    }
  }

  // Sort eigenvalues ascending so roots u_0 < u_1 < ... < u_{n-1}.
  for (int i = 0; i < n - 1; ++i) {
    int k = i;
    double pk = d[i];
    for (int j = i + 1; j < n; ++j) {
      if (d[j] < pk) {
        k = j;
        pk = d[j];
      }
    }
    if (k != i) {
      d[k] = d[i];
      d[i] = pk;
      const double tz = z0[i];
      z0[i] = z0[k];
      z0[k] = tz;
    }
  }
}

// Modified Chebyshev algorithm on shifted Legendre polynomials P_k*(u) = P_k(2u-1)
// on [0, 1] from precomputed Boys moments F_0..F_{2n-1}(T) for small/moderate T,
// paired with Stieltjes/asymptotic paths when Hankel conditioning degrades.
template <int MaxN>
GPU4DFT_HD inline bool rys_chebyshev_legendre(int n, const double* F, double* d, double* e,
                                              double& mu0) {
  const int m2 = 2 * n;
  double row_prev[2 * MaxN] = {0.0};
  double row_curr[2 * MaxN] = {0.0};
  double nu[2 * MaxN] = {0.0};

  // p_0(u) = 1: row_curr[j] = F_j
  for (int j = 0; j < m2; ++j) {
    row_prev[j] = 0.0;
    row_curr[j] = F[j];
  }
  nu[0] = row_curr[0];
  mu0 = F[0];

  for (int l = 1; l < m2; ++l) {
    const double bl = (l == 1) ? 0.0 : double((l - 1) * (l - 1)) / (4.0 * double(4 * (l - 1) * (l - 1) - 1));
    double row_next[2 * MaxN];
    for (int j = 0; j < m2 - l; ++j) {
      row_next[j] = row_curr[j + 1] - 0.5 * row_curr[j] - bl * row_prev[j];
    }
    nu[l] = row_next[0];
    for (int j = 0; j < m2 - l; ++j) {
      row_prev[j] = row_curr[j];
      row_curr[j] = row_next[j];
    }
  }

  // Wheeler's modified Chebyshev algorithm.
  double sig_prev[2 * MaxN] = {0.0};
  double sig_curr[2 * MaxN] = {0.0};
  for (int l = 0; l < m2; ++l) sig_curr[l] = nu[l];

  d[0] = 0.5 + nu[1] / nu[0];
  if (!(d[0] > 0.0 && d[0] < 1.0)) return false;

  for (int k = 1; k < n; ++k) {
    double sig_next[2 * MaxN] = {0.0};
    const double bk_prev = (k == 1) ? 0.0 : e[k - 2] * e[k - 2];
    for (int l = k; l < m2 - k; ++l) {
      const double bl = double(l * l) / (4.0 * double(4 * l * l - 1));
      const double bl_next_a = 0.5;
      sig_next[l] = sig_curr[l + 1] - (d[k - 1] - bl_next_a) * sig_curr[l] -
                    bk_prev * sig_prev[l] + bl * sig_curr[l - 1];
    }
    const double ratio = sig_next[k] / sig_curr[k - 1];
    if (!(ratio > 1e-30)) return false;
    e[k - 1] = math::sqrt(ratio);
    d[k] = 0.5 + sig_next[k + 1] / sig_next[k] - sig_curr[k] / sig_curr[k - 1];
    if (!(d[k] > 0.0 && d[k] < 1.0)) return false;

    for (int l = k; l < m2 - k; ++l) {
      sig_prev[l] = sig_curr[l];
      sig_curr[l] = sig_next[l];
    }
  }
  return true;
}

// Unconditionally stable Stieltjes orthogonalization + Laguerre/Gauss-Hermite
// asymptotic regime for n = 1..12 across all T >= 0.
template <int MaxN>
GPU4DFT_HD inline void rys_solve_double(int n, double T, const double* F_opt, double* roots,
                                        double* weights) {
  if (T < 0.0) T = 0.0;

  // Exact closed form for n = 1.
  if (n == 1) {
    double F[2];
    if (F_opt != nullptr) {
      F[0] = F_opt[0];
      F[1] = F_opt[1];
    } else {
      boys_vec<double>(1, T, F);
    }
    roots[0] = F[1] / F[0];
    weights[0] = F[0];
    return;
  }

  // Exact algebraic 2x2 closed form for n = 2 (zero cancellation for all T >= 0).
  if (n == 2) {
    double F[4];
    if (F_opt != nullptr) {
      for (int k = 0; k < 4; ++k) F[k] = F_opt[k];
    } else {
      boys_vec<double>(3, T, F);
    }
    const double det = F[0] * F[2] - F[1] * F[1];
    const double a0 = F[1] / F[0];
    const double b1 = math::sqrt(det) / F[0];
    const double a1 = (F[0] * F[3] - F[1] * F[2]) / det - a0;
    const double avg = 0.5 * (a0 + a1);
    const double half_diff = 0.5 * (a1 - a0);
    const double rad = math::sqrt(half_diff * half_diff + b1 * b1);
    roots[0] = avg - rad;
    roots[1] = avg + rad;
    // Normalized eigenvector of [[a0, b1], [b1, a1]] for eigenvalue roots[0]:
    // (a0 - roots[0]) * v0 + b1 * v1 = 0, with a0 - roots[0] = rad - half_diff.
    // Avoid cancellation by choosing the larger denominator branch.
    double c2;
    if (half_diff >= 0.0) {
      const double t = b1 / (half_diff + rad);
      c2 = 1.0 / (1.0 + t * t);
      weights[0] = F[0] * c2;
      weights[1] = F[0] - weights[0];
    } else {
      const double t = b1 / (-half_diff + rad);
      c2 = 1.0 / (1.0 + t * t);
      weights[1] = F[0] * c2;
      weights[0] = F[0] - weights[1];
    }
    return;
  }

  double d[MaxN] = {0.0};
  double e[MaxN] = {0.0};
  double z0[MaxN] = {0.0};

  // Large-T asymptotic regime: scaled positive Gauss-Hermite (= Generalized
  // Laguerre alpha = -1/2 Jacobi matrix), exact to < 1e-15 when exp(-T) << eps.
  const double T_asymp = 44.0 + 3.5 * double(n);
  if (T >= T_asymp) {
    for (int k = 0; k < n; ++k) {
      d[k] = 2.0 * double(k) + 0.5;
      if (k + 1 < n) {
        const double kp = double(k + 1);
        e[k] = math::sqrt(kp * (kp - 0.5));
      }
    }
    sym_tridiag_ql<MaxN>(n, d, e, z0);
    const double inv_T = 1.0 / T;
    const double mu0 = 0.88622692545275801365 / math::sqrt(T);  // sqrt(pi) / (2 sqrt(T))
    for (int r = 0; r < n; ++r) {
      roots[r] = d[r] * inv_T;
      weights[r] = mu0 * z0[r] * z0[r];
    }
    return;
  }

  // For n <= 5 and small/moderate T <= 6.0, modified Chebyshev on shifted
  // Legendre polynomials on [0, 1] is well-conditioned and ultra-fast.
  if (n <= 5 && T <= 6.0) {
    double F_local[2 * MaxN];
    const double* F_ptr = F_opt;
    if (F_ptr == nullptr) {
      boys_vec<double>(2 * n - 1, T, F_local);
      F_ptr = F_local;
    }
    double mu0 = 0.0;
    if (rys_chebyshev_legendre<MaxN>(n, F_ptr, d, e, mu0)) {
      sym_tridiag_ql<MaxN>(n, d, e, z0);
      for (int r = 0; r < n; ++r) {
        roots[r] = d[r];
        weights[r] = mu0 * z0[r] * z0[r];
      }
      return;
    }
  }

  // General T in [0, T_asymp): Gautschi discretized Stieltjes orthogonalization
  // on 3 subpanels of [0, 1] (48 Gauss-Legendre points in t = sqrt(u), degree
  // 31 per third-interval -> < 1e-16 moment integration error and condition
  // number 1.000 for all n = 3..12 and all T >= 0).
  constexpr double kGL16Nodes01[16] = {
      0.0052995325041750336, 0.0277124884633837110, 0.0671843988060841226,
      0.1222977958224984830, 0.1910618777986781245, 0.2709916111713863075,
      0.3591982246103705425, 0.4524937450811812796, 0.5475062549188187204,
      0.6408017753896294575, 0.7290083888286136925, 0.8089381222013218755,
      0.8777022041775015170, 0.9328156011939158774, 0.9722875115366162890,
      0.9947004674958249664,
  };
  constexpr double kGL16Weights01[16] = {
      0.0135762297058770474, 0.0311267619693239463, 0.0475792558412463923,
      0.0623144856277669360, 0.0747979944082883660, 0.0845782596975012697,
      0.0913017075224617941, 0.0947253052275342481, 0.0947253052275342481,
      0.0913017075224617941, 0.0845782596975012697, 0.0747979944082883660,
      0.0623144856277669360, 0.0475792558412463923, 0.0311267619693239463,
      0.0135762297058770474,
  };

  constexpr int kPanels = 3;
  constexpr int kPts = kPanels * 16;
  double u_pts[kPts];
  double w_pts[kPts];
  double p_prev[kPts];
  double p_curr[kPts];

  const double h = 1.0 / double(kPanels);
  double s_curr = 0.0;
  for (int panel = 0; panel < kPanels; ++panel) {
    const double t0 = double(panel) * h;
    for (int q = 0; q < 16; ++q) {
      const int idx = panel * 16 + q;
      const double t = t0 + h * kGL16Nodes01[q];
      const double u = t * t;
      const double w = h * kGL16Weights01[q] * math::exp(-T * u);
      u_pts[idx] = u;
      w_pts[idx] = w;
      p_prev[idx] = 0.0;
      p_curr[idx] = 1.0;
      s_curr += w;
    }
  }
  const double mu0 = s_curr;

  for (int k = 0; k < n; ++k) {
    double num = 0.0;
    for (int idx = 0; idx < kPts; ++idx) {
      num += w_pts[idx] * u_pts[idx] * p_curr[idx] * p_curr[idx];
    }
    const double alpha = num / s_curr;
    d[k] = alpha;

    if (k + 1 < n) {
      const double b2_prev = (k == 0) ? 0.0 : (e[k - 1] * e[k - 1]);
      double s_next = 0.0;
      for (int idx = 0; idx < kPts; ++idx) {
        const double p_next = (u_pts[idx] - alpha) * p_curr[idx] - b2_prev * p_prev[idx];
        p_prev[idx] = p_curr[idx];
        p_curr[idx] = p_next;
        s_next += w_pts[idx] * p_next * p_next;
      }
      e[k] = math::sqrt(s_next / s_curr);
      s_curr = s_next;
    }
  }

  sym_tridiag_ql<MaxN>(n, d, e, z0);
  for (int r = 0; r < n; ++r) {
    roots[r] = d[r];
    weights[r] = mu0 * z0[r] * z0[r];
  }
}

}  // namespace detail

// Computes Rys quadrature squared roots u_r = t_r^2 in (0, 1) and weights w_r
// for `n_roots` in 1..12 such that \sum_{r=0}^{n_roots-1} w_r u_r^k == F_k(T).
template <typename R>
GPU4DFT_HD inline void rys_roots_weights(int n_roots, R T, R* roots, R* weights) {
  constexpr int kMaxRoots = 12;
  if (n_roots <= 0 || n_roots > kMaxRoots) return;
  double u_d[kMaxRoots];
  double w_d[kMaxRoots];
  detail::rys_solve_double<kMaxRoots>(n_roots, static_cast<double>(T), nullptr, u_d, w_d);
  for (int r = 0; r < n_roots; ++r) {
    roots[r] = static_cast<R>(u_d[r]);
    weights[r] = static_cast<R>(w_d[r]);
  }
}

template <typename R>
GPU4DFT_HD inline void rys_roots_weights_from_boys(int n_roots, R T, const R* F, R* roots,
                                                   R* weights) {
  constexpr int kMaxRoots = 12;
  if (n_roots <= 0 || n_roots > kMaxRoots) return;
  double F_d[2 * kMaxRoots];
  for (int k = 0; k < 2 * n_roots; ++k) F_d[k] = static_cast<double>(F[k]);
  double u_d[kMaxRoots];
  double w_d[kMaxRoots];
  detail::rys_solve_double<kMaxRoots>(n_roots, static_cast<double>(T), F_d, u_d, w_d);
  for (int r = 0; r < n_roots; ++r) {
    roots[r] = static_cast<R>(u_d[r]);
    weights[r] = static_cast<R>(w_d[r]);
  }
}

// Omega-attenuated Rys quadrature roots u_r and weights w_r for erf(omega * r12) / r12:
//   \sum_{r=0}^{n_roots-1} w_r u_r^k == theta^{k + 1/2} F_k(theta * T)
// where theta = omega^2 / (omega^2 + rho). Delegates to standard rys_roots_weights when omega <= 0.
template <typename R>
GPU4DFT_HD inline void rys_roots_weights_erf(int n_roots, R T, R rho, R omega, R* roots,
                                             R* weights) {
  if (omega <= R(0)) {
    rys_roots_weights<R>(n_roots, T, roots, weights);
    return;
  }
  constexpr int kMaxRoots = 12;
  if (n_roots <= 0 || n_roots > kMaxRoots) return;
  const double theta = omega_theta<double>(static_cast<double>(rho), static_cast<double>(omega));
  const double sqrt_theta = math::sqrt(theta);
  double u_d[kMaxRoots];
  double w_d[kMaxRoots];
  detail::rys_solve_double<kMaxRoots>(n_roots, theta * static_cast<double>(T), nullptr, u_d, w_d);
  for (int r = 0; r < n_roots; ++r) {
    roots[r] = static_cast<R>(theta * u_d[r]);
    weights[r] = static_cast<R>(sqrt_theta * w_d[r]);
  }
}

}  // namespace gpu4dft
