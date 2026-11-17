// Checks boys_f0 against direct quadrature of F_0(T) = \int_0^1 exp(-T t^2) dt.

#include <cmath>
#include <cstdio>
#include <vector>

#include "gpu4dft/boys.hpp"

namespace {

// Composite Simpson rule in long double; independent of the erf formula.
long double boys_f0_reference(long double T) {
  constexpr int kIntervals = 20000;
  const long double h = 1.0L / kIntervals;
  long double sum = 1.0L + std::exp(-T);
  for (int i = 1; i < kIntervals; ++i) {
    const long double t = i * h;
    sum += (i % 2 == 1 ? 4.0L : 2.0L) * std::exp(-T * t * t);
  }
  return sum * h / 3.0L;
}

int failures = 0;

template <typename R>
void check(const char* label, double T, double rel_tol) {
  const double got = static_cast<double>(gpu4dft::boys_f0<R>(static_cast<R>(T)));
  // Evaluate the reference at the argument actually seen by boys_f0<R>.
  const long double ref = boys_f0_reference(static_cast<long double>(static_cast<R>(T)));
  const double rel = static_cast<double>(std::fabs((got - ref) / ref));
  if (!(rel <= rel_tol)) {
    std::printf("FAIL %s T=%.17g got=%.17g ref=%.17Lg rel=%.3g tol=%.3g\n", label, T, got, ref,
                rel, rel_tol);
    ++failures;
  }
}

}  // namespace

int main() {
  std::vector<double> Ts = {0.0, 1e-12, 1e-6, 9.999e-4, 1e-3, 1.0001e-3};
  for (double T = 1e-4; T < 200.0; T *= 1.37) Ts.push_back(T);

  for (double T : Ts) {
    check<double>("double", T, 2e-15);
    check<float>("float", T, 2e-6);
  }

  if (failures != 0) {
    std::printf("%d failures\n", failures);
    return 1;
  }
  std::printf("boys_f0: %zu points OK\n", Ts.size());
  return 0;
}
