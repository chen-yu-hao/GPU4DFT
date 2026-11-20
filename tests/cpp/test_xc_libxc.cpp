#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/kernel_registry.hpp"
#include "gpu4dft/xc.hpp"

namespace {

int g_failures = 0;

void check_near(const char* label, double got, double ref, double tol) {
  const double err = std::fabs(got - ref);
  if (!(err <= tol)) {
    std::printf("FAIL %s: got=%.15g ref=%.15g err=%.3g tol=%.3g\n", label, got, ref, err, tol);
    ++g_failures;
  }
}

void check_vec_near(const char* label, const std::vector<double>& got,
                    const std::vector<double>& ref, double tol) {
  if (got.size() != ref.size()) {
    std::printf("FAIL %s: size mismatch %zu vs %zu\n", label, got.size(), ref.size());
    ++g_failures;
    return;
  }
  double max_err = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
  }
  if (!(max_err <= tol)) {
    std::printf("FAIL %s: max_err=%.3g tol=%.3g\n", label, max_err, tol);
    ++g_failures;
  }
}

gpu4dft::Basis make_h2o_sto3g_basis(const std::vector<gpu4dft::XcAtom>& atoms) {
  std::vector<gpu4dft::Shell> shells;
  // O 1s, 2s, 2p
  shells.push_back(gpu4dft::Shell{0, atoms[0].center,
                                  {130.7093200, 23.8088610, 6.4436083},
                                  {0.15432897, 0.53532814, 0.44463454}});
  shells.push_back(gpu4dft::Shell{0, atoms[0].center,
                                  {5.0331513, 1.1695961, 0.3803890},
                                  {-0.09996723, 0.39951283, 0.70011547}});
  shells.push_back(gpu4dft::Shell{1, atoms[0].center,
                                  {5.0331513, 1.1695961, 0.3803890},
                                  {0.15591627, 0.60768372, 0.39195739}});
  // H1 1s, H2 1s
  for (int h = 1; h <= 2; ++h) {
    shells.push_back(gpu4dft::Shell{0, atoms[h].center,
                                    {3.42525091, 0.62391373, 0.16885540},
                                    {0.15432897, 0.53532814, 0.44463454}});
  }
  return gpu4dft::Basis(std::move(shells));
}

std::vector<double> make_test_dm(int nao, double scale, double asym) {
  std::vector<double> m(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      m[static_cast<std::size_t>(i) * nao + j] =
          std::exp(-0.4 * std::abs(i - j)) *
          (1.0 + asym * ((i + 2 * j) % 3 == 0 ? 0.25 : -0.15));
    }
  }
  std::vector<double> dm(static_cast<std::size_t>(nao) * nao, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      double s = 0.0;
      for (int k = 0; k < nao; ++k) {
        s += m[static_cast<std::size_t>(i) * nao + k] * m[static_cast<std::size_t>(j) * nao + k];
      }
      dm[static_cast<std::size_t>(i) * nao + j] = scale * s;
    }
  }
  return dm;
}

void test_rsh_metadata() {
  const auto cam = gpu4dft::xc_rsh_coefficients("cam-b3lyp");
  check_near("cam_b3lyp.omega", cam.omega, 0.33, 1e-12);
  check_near("cam_b3lyp.alpha", cam.alpha, 0.65, 1e-12);
  check_near("cam_b3lyp.beta", cam.beta, -0.46, 1e-12);
  check_near("cam_b3lyp.c_lr", cam.c_lr(), 0.65, 1e-12);
  check_near("cam_b3lyp.c_sr", cam.c_sr(), 0.19, 1e-12);
  if (!cam.is_rsh()) {
    std::printf("FAIL cam_b3lyp.is_rsh\n");
    ++g_failures;
  }

  const auto wb97x = gpu4dft::xc_rsh_coefficients("wb97x");
  check_near("wb97x.omega", wb97x.omega, 0.30, 1e-12);
  check_near("wb97x.alpha", wb97x.alpha, 1.00, 1e-12);
  check_near("wb97x.beta", wb97x.beta, -0.842294, 1e-12);
  check_near("wb97x.c_sr", wb97x.c_sr(), 0.157706, 1e-12);
  if (!wb97x.is_rsh()) {
    std::printf("FAIL wb97x.is_rsh\n");
    ++g_failures;
  }

  const auto wb97xd = gpu4dft::xc_rsh_coefficients("wb97x-d");
  check_near("wb97xd.omega", wb97xd.omega, 0.20, 1e-12);
  check_near("wb97xd.alpha", wb97xd.alpha, 1.00, 1e-12);

  const auto b3lyp = gpu4dft::xc_rsh_coefficients("b3lyp");
  check_near("b3lyp.omega", b3lyp.omega, 0.0, 1e-12);
  check_near("b3lyp.c_sr", b3lyp.c_sr(), 0.20, 1e-12);
  if (b3lyp.is_rsh()) {
    std::printf("FAIL b3lyp.is_rsh\n");
    ++g_failures;
  }

  const auto pbe0 = gpu4dft::xc_rsh_coefficients("pbe0");
  check_near("pbe0.c_sr", pbe0.c_sr(), 0.25, 1e-12);

  const auto by_id = gpu4dft::xc_rsh_coefficients("433");
  check_near("433.omega", by_id.omega, 0.33, 1e-12);
  check_near("433.c_sr", by_id.c_sr(), 0.19, 1e-12);

  const auto custom = gpu4dft::xc_rsh_coefficients("0.2*HF + 0.8*LDA_X, LDA_C_VWN");
  check_near("custom.c_sr", custom.c_sr(), 0.20, 1e-12);

  const auto blyp_xc = gpu4dft::xc_rsh_coefficients("gga_x_b88,gga_c_lyp");
  check_near("blyp.c_sr", blyp_xc.c_sr(), 0.0, 1e-12);

  bool threw = false;
  try {
    (void)gpu4dft::xc_rsh_coefficients("nonexistent_xc_functional_xyz");
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  if (!threw) {
    std::printf("FAIL invalid functional did not throw\n");
    ++g_failures;
  }
}

void test_uks_identities_and_grid_response(gpu4dft::Backend be) {
  const std::vector<gpu4dft::XcAtom> atoms = {
      {8, {0.0, 0.0, 0.221}},
      {1, {0.0, 1.431, -0.886}},
      {1, {0.09, -1.410, -0.850}},
  };
  const auto basis = make_h2o_sto3g_basis(atoms);
  const int nao = basis.nao();

  gpu4dft::GridOptions gopts;
  gopts.level = 2;
  gopts.backend = be;
  const gpu4dft::MolecularGrid grid(atoms, basis, gopts);
  if (!grid.has_partition_info() || !grid.has_grid_response_data()) {
    std::printf("FAIL native grid missing partition info\n");
    ++g_failures;
  }

  const auto dm_tot = make_test_dm(nao, 0.45, 1.0);
  std::vector<double> dm_half(dm_tot.size());
  for (std::size_t i = 0; i < dm_tot.size(); ++i) dm_half[i] = 0.5 * dm_tot[i];
  const auto dm_a = make_test_dm(nao, 0.30, 1.2);
  const auto dm_b = make_test_dm(nao, 0.18, -0.8);

  const std::vector<std::string> funcs = {
      "svwn", "pbe", "blyp", "b3lyp", "b3lyp5", "pbe0", "tpss", "cam-b3lyp", "wb97x",
  };
  for (const std::string& xc : funcs) {
    gpu4dft::XcOptions xopts;
    xopts.backend = be;
    xopts.spin_polarized = true;
    xopts.use_libxc = true;
    xopts.grid_response = true;
    gpu4dft::XcEngine eng(basis, grid, xc, xopts);

    // 1. Closed-shell identity: UKS(0.5 D, 0.5 D) == RKS(D)
    const auto rks = eng.eval_vxc(dm_tot);
    const auto uks = eng.eval_vxc_uks(dm_half, dm_half);
    check_near((xc + ".cs_exc").c_str(), uks.exc, rks.exc, 1e-13);
    check_near((xc + ".cs_nelec").c_str(), uks.nelec_alpha + uks.nelec_beta, rks.nelec, 1e-13);
    check_vec_near((xc + ".cs_va").c_str(), uks.vxc_alpha, rks.vxc, 1e-12);
    check_vec_near((xc + ".cs_vb").c_str(), uks.vxc_beta, rks.vxc, 1e-12);

    // 2. Spin-swap symmetry
    const auto uks_ab = eng.eval_vxc_uks(dm_a, dm_b);
    const auto uks_ba = eng.eval_vxc_uks(dm_b, dm_a);
    check_near((xc + ".swap_exc").c_str(), uks_ab.exc, uks_ba.exc, 1e-13);
    check_vec_near((xc + ".swap_va").c_str(), uks_ab.vxc_alpha, uks_ba.vxc_beta, 1e-12);
    check_vec_near((xc + ".swap_vb").c_str(), uks_ab.vxc_beta, uks_ba.vxc_alpha, 1e-12);

    // 3. Nuclear gradient closed-shell identity & translational invariance
    const auto g_rks = eng.nuclear_grad(atoms, dm_tot);
    const auto g_uks_cs = eng.nuclear_grad_uks(atoms, dm_half, dm_half);
    check_vec_near((xc + ".grad_cs").c_str(), g_uks_cs, g_rks, 1e-12);

    const auto g_uks_ab = eng.nuclear_grad_uks(atoms, dm_a, dm_b);
    for (int d = 0; d < 3; ++d) {
      const double sum_rks = g_rks[d] + g_rks[3 + d] + g_rks[6 + d];
      const double sum_uks = g_uks_ab[d] + g_uks_ab[3 + d] + g_uks_ab[6 + d];
      check_near((xc + ".trans_rks").c_str(), sum_rks, 0.0, 1e-12);
      check_near((xc + ".trans_uks").c_str(), sum_uks, 0.0, 1e-12);
    }
  }
}

}  // namespace

int main() {
  test_rsh_metadata();
  test_uks_identities_and_grid_response(gpu4dft::Backend::Host);
  if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
    test_uks_identities_and_grid_response(gpu4dft::Backend::Cuda);
  }

  if (g_failures != 0) {
    std::printf("%d failures in test_xc_libxc\n", g_failures);
    return 1;
  }
  std::printf("test_xc_libxc: ALL PASSED\n");
  return 0;
}
