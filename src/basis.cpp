#include "gpu4dft/basis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace gpu4dft {
namespace {

// \int_0^inf r^{2l+2} exp(-alpha r^2) dr = Gamma(l + 3/2) / (2 alpha^{l + 3/2})
double radial_moment(int l, double alpha) {
  return std::tgamma(l + 1.5) / (2.0 * std::pow(alpha, l + 1.5));
}

[[noreturn]] void invalid(std::size_t ishell, const std::string& what) {
  throw std::invalid_argument("shell " + std::to_string(ishell) + ": " + what);
}

void validate(const Shell& shell, std::size_t ishell) {
  constexpr int kMaxL = 7;
  if (shell.l < 0 || shell.l > kMaxL) {
    invalid(ishell, "angular momentum " + std::to_string(shell.l) + " outside [0, " +
                        std::to_string(kMaxL) + "]");
  }
  if (shell.exponents.empty()) invalid(ishell, "no primitives");
  if (shell.exponents.size() != shell.coefficients.size()) {
    invalid(ishell, std::to_string(shell.exponents.size()) + " exponents but " +
                        std::to_string(shell.coefficients.size()) + " coefficients");
  }
  for (double a : shell.exponents) {
    if (!(a > 0.0) || !std::isfinite(a)) invalid(ishell, "exponents must be positive and finite");
  }
  for (double c : shell.coefficients) {
    if (!std::isfinite(c)) invalid(ishell, "coefficients must be finite");
  }
  for (double x : shell.center) {
    if (!std::isfinite(x)) invalid(ishell, "center must be finite");
  }
}

void normalize(Shell& shell, std::size_t ishell) {
  const int l = shell.l;
  const auto& a = shell.exponents;
  auto& c = shell.coefficients;
  const std::size_t n = a.size();

  // Coefficients of normalised primitives -> coefficients of raw primitives.
  for (std::size_t k = 0; k < n; ++k) c[k] /= std::sqrt(radial_moment(l, 2.0 * a[k]));

  // Normalise the contraction.
  double norm2 = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t m = 0; m < n; ++m) norm2 += c[k] * c[m] * radial_moment(l, a[k] + a[m]);
  }
  if (!(norm2 > 0.0) || !std::isfinite(norm2)) invalid(ishell, "contraction has zero norm");
  const double scale = 1.0 / std::sqrt(norm2);
  for (double& ck : c) ck *= scale;
}

}  // namespace

Basis::Basis(std::vector<Shell> shells) : shells_(std::move(shells)) {
  ao_offset_.reserve(shells_.size());
  for (std::size_t i = 0; i < shells_.size(); ++i) {
    Shell& shell = shells_[i];
    validate(shell, i);
    normalize(shell, i);
    ao_offset_.push_back(nao_);
    nao_ += 2 * shell.l + 1;
    max_l_ = std::max(max_l_, shell.l);
  }
}

}  // namespace gpu4dft
