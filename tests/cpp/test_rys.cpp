// Verifies boys_vec<double/float> (m <= 24) against long-double reference
// and rys_roots_weights (n_roots = 1..9) moment reproduction
// \sum_r w_r u_r^k == F_k(T) to < 1e-12 across T >= 0.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu4dft/boys.hpp"
#include "gpu4dft/rys.hpp"

namespace {

void boys_vec_reference(int m_max, long double T, std::vector<long double>& out) {
  out.assign(m_max + 1, 0.0L);
  if (T <= 0.0L) {
    for (int m = 0; m <= m_max; ++m) out[m] = 1.0L / static_cast<long double>(2 * m + 1);
    return;
  }
  if (T <= 80.0L) {
    const long double e_neg = std::exp(-T);
    long double denom = static_cast<long double>(2 * m_max + 1);
    long double term = 1.0L / denom;
    long double sum = term;
    for (int k = 1; k < 250; ++k) {
      denom += 2.0L;
      term *= (2.0L * T) / denom;
      sum += term;
      if (term <= sum * 1e-21L) break;
    }
    out[m_max] = e_neg * sum;
    for (int m = m_max; m >= 1; --m) {
      out[m - 1] = (2.0L * T * out[m] + e_neg) / static_cast<long double>(2 * m - 1);
    }
  } else {
    const long double x = std::sqrt(T);
    const long double e_neg = std::exp(-T);
    out[0] = 0.8862269254527580136490837416705725914L * std::erf(x) / x;
    for (int m = 1; m <= m_max; ++m) {
      out[m] = (static_cast<long double>(2 * m - 1) * out[m - 1] - e_neg) / (2.0L * T);
    }
  }
}

}  // namespace

int main() {
  int failures = 0;

  std::vector<double> Ts = {0.0, 1e-14, 1e-9, 1e-5, 1e-3, 0.05, 0.5,  1.0,  2.5,  5.99,
                            6.0, 6.01,  12.0, 24.9, 25.0, 25.1, 40.0, 49.0, 50.0, 55.0,
                            65.0, 72.0, 80.0, 110.0, 180.0, 300.0};

  // 1. Check boys_vec<double> and boys_vec<float> for m = 0..24.
  for (double T : Ts) {
    double fd[25] = {0.0};
    float ff[25] = {0.0f};
    gpu4dft::boys_vec<double>(24, T, fd);
    gpu4dft::boys_vec<float>(24, static_cast<float>(T), ff);

    std::vector<long double> ref_d, ref_f;
    boys_vec_reference(24, static_cast<long double>(T), ref_d);
    boys_vec_reference(24, static_cast<long double>(static_cast<float>(T)), ref_f);

    for (int m = 0; m <= 24; ++m) {
      const double err_d = static_cast<double>(std::fabs(fd[m] - ref_d[m]));
      const double scale_d = std::max(1e-30, static_cast<double>(std::fabs(ref_d[m])));
      if (err_d / scale_d > 1e-14 && err_d > 1e-15) {
        std::printf("FAIL boys_vec<double> m=%d T=%.6g got=%.17g ref=%.17Lg rel=%.3e\n", m, T,
                    fd[m], ref_d[m], err_d / scale_d);
        ++failures;
      }

      if (T <= 80.0) {
        const double err_f =
            static_cast<double>(std::fabs(static_cast<long double>(ff[m]) - ref_f[m]));
        const double scale_f = std::max(1e-20, static_cast<double>(std::fabs(ref_f[m])));
        if (err_f / scale_f > 1e-6 && err_f > 1e-7) {
          std::printf("FAIL boys_vec<float> m=%d T=%.6g got=%.9g ref=%.9Lg rel=%.3e\n", m, T,
                      double(ff[m]), ref_f[m], err_f / scale_f);
          ++failures;
        }
      }
    }
  }

  // 2. Check Rys quadrature roots and weights for n_roots = 1..9 across all T.
  for (int n = 1; n <= 9; ++n) {
    for (double T : Ts) {
      double u[12] = {0.0};
      double w[12] = {0.0};
      gpu4dft::rys_roots_weights<double>(n, T, u, w);

      for (int r = 0; r < n; ++r) {
        if (!(u[r] > 0.0 && u[r] < 1.0) || !(w[r] > 0.0)) {
          std::printf("FAIL rys bounds n=%d T=%.6g r=%d u=%.17g w=%.17g\n", n, T, r, u[r], w[r]);
          ++failures;
        }
        if (r > 0 && !(u[r] > u[r - 1])) {
          std::printf("FAIL rys ordering n=%d T=%.6g u[%d]=%.17g <= u[%d]=%.17g\n", n, T, r, u[r],
                      r - 1, u[r - 1]);
          ++failures;
        }
      }

      double F[24] = {0.0};
      gpu4dft::boys_vec<double>(2 * n - 1, T, F);
      for (int k = 0; k < 2 * n; ++k) {
        double mom = 0.0;
        for (int r = 0; r < n; ++r) {
          mom += w[r] * std::pow(u[r], k);
        }
        const double err = std::fabs(mom - F[k]);
        const double tol = 1e-12 * std::max(1.0, std::fabs(F[0]));
        if (!(err <= tol)) {
          std::printf("FAIL rys moment n=%d k=%d T=%.6g mom=%.17g F=%.17g err=%.3e\n", n, k, T, mom,
                      F[k], err);
          ++failures;
        }
      }
    }
  }

  if (failures != 0) {
    std::printf("test_rys: %d failures\n", failures);
    return 1;
  }
  std::printf("boys_vec & rys_roots_weights (n=1..9, %zu T points): ALL OK\n", Ts.size());
  return 0;
}
