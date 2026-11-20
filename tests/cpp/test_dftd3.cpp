#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/dftd3.hpp"

namespace {

int g_failures = 0;

void check_near(const char* label, double got, double ref, double tol) {
  const double err = std::fabs(got - ref);
  if (!(err <= tol)) {
    std::printf("FAIL %s: got=%.15g ref=%.15g err=%.3g tol=%.3g\n", label, got, ref, err, tol);
    ++g_failures;
  }
}

void test_parameter_lookup() {
  const auto p_b3lyp = gpu4dft::dftd3::get_d3bj_params("B3LYP");
  check_near("b3lyp.s6", p_b3lyp.s6, 1.0, 1e-14);
  check_near("b3lyp.a1", p_b3lyp.a1, 0.3981, 1e-14);
  check_near("b3lyp.s8", p_b3lyp.s8, 1.9889, 1e-14);
  check_near("b3lyp.a2", p_b3lyp.a2, 4.4211, 1e-14);

  const auto p_cam = gpu4dft::dftd3::get_d3bj_params("cam-b3lyp-d3(bj)");
  check_near("camb3lyp.a1", p_cam.a1, 0.3708, 1e-14);
  check_near("camb3lyp.s8", p_cam.s8, 2.0674, 1e-14);
  check_near("camb3lyp.a2", p_cam.a2, 5.4743, 1e-14);

  const auto p_wb97x = gpu4dft::dftd3::get_d3bj_params("wB97X");
  check_near("wb97x.a1", p_wb97x.a1, 0.0, 1e-14);
  check_near("wb97x.s8", p_wb97x.s8, 0.2641, 1e-14);
  check_near("wb97x.a2", p_wb97x.a2, 5.4959, 1e-14);

  // Do NOT alias wb97xd to wb97x!
  if (gpu4dft::dftd3::has_d3bj_params("wb97x-d") || gpu4dft::dftd3::has_d3bj_params("wb97xd")) {
    std::printf("FAIL wb97xd must not alias wb97x\n");
    ++g_failures;
  }
  bool threw = false;
  try {
    (void)gpu4dft::dftd3::get_d3bj_params("unknown_func_123");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    std::printf("FAIL unknown functional did not throw\n");
    ++g_failures;
  }
}

void test_fd_and_invariances() {
  // Distorted CH3OH (6 atoms -> exercises multi-atom CN chain rule)
  const std::vector<int> z = {6, 8, 1, 1, 1, 1};
  const std::vector<double> coords = {
      -0.662, -0.021, 0.015,
       0.684,  0.032, -0.012,
      -1.080,  0.985, -0.042,
      -1.012, -0.543, -0.875,
      -1.023, -0.511,  0.921,
       1.045, -0.845,  0.033,
  };

  const std::vector<std::string> funcs = {"b3lyp", "pbe", "pbe0", "cam-b3lyp", "wb97x", "hf"};
  const double h = 1e-5;

  for (const std::string& xc : funcs) {
    const auto res = gpu4dft::dftd3::compute_d3bj(z, coords, xc, true);
    if (!(res.energy < 0.0)) {
      std::printf("FAIL %s energy should be negative, got %.6g\n", xc.c_str(), res.energy);
      ++g_failures;
    }

    // 1. Translational invariance
    for (int d = 0; d < 3; ++d) {
      double sum_g = 0.0;
      for (std::size_t a = 0; a < z.size(); ++a) {
        sum_g += res.gradient[3 * a + d];
      }
      check_near((xc + ".trans").c_str(), sum_g, 0.0, 1e-14);
    }

    // 2. Central finite-difference check of analytical gradient (including dCN/dR)
    double max_fd_err = 0.0;
    for (std::size_t i = 0; i < coords.size(); ++i) {
      auto cp = coords;
      auto cm = coords;
      cp[i] += h;
      cm[i] -= h;
      const double ep = gpu4dft::dftd3::compute_d3bj(z, cp, xc, false).energy;
      const double em = gpu4dft::dftd3::compute_d3bj(z, cm, xc, false).energy;
      const double g_fd = (ep - em) / (2.0 * h);
      max_fd_err = std::max(max_fd_err, std::fabs(res.gradient[i] - g_fd));
    }
    check_near((xc + ".fd_grad").c_str(), max_fd_err, 0.0, 1e-9);
  }

  // 3. Single atom -> zero dispersion
  const auto single = gpu4dft::dftd3::compute_d3bj({6}, {0.0, 0.0, 0.0}, "b3lyp", true);
  check_near("single_atom.e", single.energy, 0.0, 1e-16);
  check_near("single_atom.gx", single.gradient[0], 0.0, 1e-16);

  // 4. Cutoffs: disp2_cutoff = 60.0 Bohr
  const auto c2_59 = gpu4dft::dftd3::compute_d3bj({6, 6}, {0.0, 0.0, 0.0, 59.0, 0.0, 0.0}, "b3lyp", false);
  const auto c2_61 = gpu4dft::dftd3::compute_d3bj({6, 6}, {0.0, 0.0, 0.0, 61.0, 0.0, 0.0}, "b3lyp", false);
  if (!(c2_59.energy < -1e-10)) {
    std::printf("FAIL c2_59 energy=%.6g\n", c2_59.energy);
    ++g_failures;
  }
  check_near("c2_61.cutoff", c2_61.energy, 0.0, 1e-16);
}

}  // namespace

int main() {
  test_parameter_lookup();
  test_fd_and_invariances();
  if (g_failures != 0) {
    std::printf("%d failures in test_dftd3\n", g_failures);
    return 1;
  }
  std::printf("test_dftd3: ALL PASSED\n");
  return 0;
}
