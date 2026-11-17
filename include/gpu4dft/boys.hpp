#pragma once

#include "gpu4dft/config.hpp"
#include "gpu4dft/math.hpp"

namespace gpu4dft {

// Boys function of order 0:
//   F_0(T) = \int_0^1 exp(-T t^2) dt = sqrt(pi) / 2 * erf(sqrt(T)) / sqrt(T).
// Higher orders (needed from (ps|ss) upward) will come with the Rys / MD
// generators.
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

}  // namespace gpu4dft
