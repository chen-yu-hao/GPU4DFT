#pragma once

#include "gpu4dft/config.hpp"
#include "gpu4dft/math.hpp"

namespace gpu4dft {

// Boys function of order 0:
//   F_0(T) = \int_0^1 exp(-T t^2) dt = sqrt(pi) / 2 * erf(sqrt(T)) / sqrt(T).
template <typename R>
GPU4DFT_HD inline R boys_f0(R T) {
  // Small-T branch: the erf form is singular at T = 0. The Taylor series
  // truncation error at the threshold is below T^5 / 1320 ~ 1e-18.
  if (T < R(1e-3)) {
    return R(1) - T * (R(1) / R(3) - T * (R(1) / R(10) - T * (R(1) / R(42) - T * (R(1) / R(216)))));
  }
  const R x = math::sqrt(T);
  return R(0.88622692545275801365) * math::erf(x) / x;  // sqrt(pi) / 2
}

// Evaluates F_0(T), F_1(T), ..., F_{m_max}(T) for m_max >= 0 and T >= 0.
// Accurate to < 1e-14 in double and < 1e-6 in float across all T >= 0 for m <= 32:
//   * T > m_max + 25: contractive upward recurrence from F_0(T) via erf(sqrt(T)).
//   * 0 <= T <= m_max + 25: strictly positive Kummer confluent hypergeometric
//     series for F_{m_max}(T) followed by strictly positive downward recurrence
//     F_{m-1}(T) = (2 T F_m(T) + exp(-T)) / (2m - 1) (zero cancellation).
template <typename R>
GPU4DFT_HD inline void boys_vec(int m_max, R T, R* F) {
  if (m_max < 0) return;
  if (T <= R(0)) {
    for (int m = 0; m <= m_max; ++m) {
      F[m] = R(1) / R(2 * m + 1);
    }
    return;
  }
  if (m_max == 0) {
    F[0] = boys_f0<R>(T);
    return;
  }

  const R threshold = R(m_max + 25);
  if (T > threshold) {
    const R x = math::sqrt(T);
    F[0] = R(0.88622692545275801365) * math::erf(x) / x;
    const R exp_neg_T = math::exp(-T);
    const R inv_two_T = R(0.5) / T;
    for (int m = 1; m <= m_max; ++m) {
      F[m] = (R(2 * m - 1) * F[m - 1] - exp_neg_T) * inv_two_T;
    }
    return;
  }

  const R exp_neg_T = math::exp(-T);
  const R two_T = R(2) * T;
  R denom = R(2 * m_max + 1);
  R term = R(1) / denom;
  R sum = term;
  const R eps = (sizeof(R) == sizeof(double)) ? R(1e-16) : R(1e-8);
  for (int k = 1; k < 160; ++k) {
    denom += R(2);
    term *= two_T / denom;
    sum += term;
    if (term <= sum * eps) break;
  }
  F[m_max] = exp_neg_T * sum;
  for (int m = m_max; m >= 1; --m) {
    F[m - 1] = (two_T * F[m] + exp_neg_T) / R(2 * m - 1);
  }
}

// Overflow-safe attenuation factor theta = omega^2 / (omega^2 + rho) in (0, 1]
// for erf(omega * r12) / r12 integrals. Returns 1 when omega <= 0.
template <typename R>
GPU4DFT_HD inline R omega_theta(R rho, R omega) {
  if (omega <= R(0)) return R(1);
  const double rho_d = static_cast<double>(rho);
  const double omega_d = static_cast<double>(omega);
  if (rho_d <= 0.0) return R(1);
  if (omega_d >= rho_d) {
    const double r = (rho_d / omega_d) / omega_d;
    return static_cast<R>(1.0 / (1.0 + r));
  }
  const double x = (omega_d / rho_d) * omega_d;
  return static_cast<R>(x / (1.0 + x));
}

// Attenuated Boys function of order 0 for erf(omega * r12) / r12:
//   F_0^omega(T) = sqrt(theta) * F_0(theta * T)
template <typename R>
GPU4DFT_HD inline R boys_f0_erf(R T, R rho, R omega) {
  if (omega <= R(0)) return boys_f0<R>(T);
  const double theta = omega_theta<double>(static_cast<double>(rho), static_cast<double>(omega));
  return static_cast<R>(math::sqrt(theta) * boys_f0<double>(theta * static_cast<double>(T)));
}

// Attenuated Boys moments F_m^omega(T) = theta^{m + 1/2} * F_m(theta * T) for m = 0..m_max.
template <typename R>
GPU4DFT_HD inline void boys_vec_erf(int m_max, R T, R rho, R omega, R* F) {
  if (m_max < 0) return;
  if (omega <= R(0)) {
    boys_vec<R>(m_max, T, F);
    return;
  }
  const double theta = omega_theta<double>(static_cast<double>(rho), static_cast<double>(omega));
  const double sqrt_theta = math::sqrt(theta);
  if (m_max <= 32) {
    double F_d[33];
    boys_vec<double>(m_max, theta * static_cast<double>(T), F_d);
    double scale = sqrt_theta;
    for (int m = 0; m <= m_max; ++m) {
      F[m] = static_cast<R>(scale * F_d[m]);
      scale *= theta;
    }
  } else {
    boys_vec<R>(m_max, static_cast<R>(theta * static_cast<double>(T)), F);
    double scale = sqrt_theta;
    for (int m = 0; m <= m_max; ++m) {
      F[m] = static_cast<R>(scale * static_cast<double>(F[m]));
      scale *= theta;
    }
  }
}

}  // namespace gpu4dft
