#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/jk_engine.hpp"
#include "gpu4dft/types.hpp"
#include "gpu4dft/workflow.hpp"
#include "gpu4dft/xc.hpp"

namespace {

int g_failures = 0;

void check_true(const char* label, bool cond) {
  if (!cond) {
    std::printf("FAIL %s\n", label);
    ++g_failures;
  }
}

void check_near(const char* label, double got, double ref, double tol) {
  const double err = std::fabs(got - ref);
  if (!(err <= tol) || !std::isfinite(got)) {
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
    if (!std::isfinite(got[i])) {
      std::printf("FAIL %s: non-finite entry at %zu\n", label, i);
      ++g_failures;
      return;
    }
    max_err = std::max(max_err, std::fabs(got[i] - ref[i]));
  }
  if (!(max_err <= tol)) {
    std::printf("FAIL %s: max_err=%.3g tol=%.3g\n", label, max_err, tol);
    ++g_failures;
  }
}

gpu4dft::Basis make_sto3g_basis(const std::vector<gpu4dft::Atom>& atoms) {
  std::vector<gpu4dft::Shell> shells;
  for (const auto& at : atoms) {
    if (at.atomic_number == 1) {
      shells.push_back(gpu4dft::Shell{0, at.center,
                                      {3.42525091, 0.62391373, 0.16885540},
                                      {0.15432897, 0.53532814, 0.44463454}});
    } else if (at.atomic_number == 8) {
      shells.push_back(gpu4dft::Shell{0, at.center,
                                      {130.7093200, 23.8088610, 6.4436083},
                                      {0.15432897, 0.53532814, 0.44463454}});
      shells.push_back(gpu4dft::Shell{0, at.center,
                                      {5.0331513, 1.1695961, 0.3803890},
                                      {-0.09996723, 0.39951283, 0.70011547}});
      shells.push_back(gpu4dft::Shell{1, at.center,
                                      {5.0331513, 1.1695961, 0.3803890},
                                      {0.15591627, 0.60768372, 0.39195739}});
    }
  }
  return gpu4dft::Basis(std::move(shells));
}

void test_validation_guards() {
  const std::vector<gpu4dft::Atom> h_atom = {{1, {0.0, 0.0, 0.0}}};
  const auto basis = make_sto3g_basis(h_atom);

  // Restricted odd-electron must throw
  bool threw_r_odd = false;
  try {
    gpu4dft::WorkflowOptions opts;
    opts.scf.backend = gpu4dft::Backend::Host;
    gpu4dft::MolecularScf m(basis, h_atom, 1, opts);
  } catch (const std::invalid_argument&) {
    threw_r_odd = true;
  }
  check_true("restricted_odd_throws", threw_r_odd);

  // Unrestricted parity / bounds errors must throw
  auto expect_u_throw = [&](const char* label, int nelec, int spin) {
    bool threw = false;
    try {
      gpu4dft::WorkflowOptions opts;
      opts.scf.backend = gpu4dft::Backend::Host;
      opts.unrestricted = true;
      opts.spin = spin;
      gpu4dft::MolecularScf m(basis, h_atom, nelec, opts);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check_true(label, threw);
  };
  expect_u_throw("u_neg_spin", 1, -1);
  expect_u_throw("u_spin_gt_nelec", 1, 3);
  expect_u_throw("u_bad_parity_even", 2, 1);
  expect_u_throw("u_bad_parity_odd", 1, 0);
}

void test_h_atom_nbeta0(gpu4dft::Backend be) {
  const std::vector<gpu4dft::Atom> atoms = {{1, {0.0, 0.0, 0.0}}};
  const auto basis = make_sto3g_basis(atoms);

  for (const std::string& xc : {"hf", "pbe", "b3lyp", "tpss"}) {
    gpu4dft::WorkflowOptions opts;
    opts.xc = xc;
    opts.unrestricted = true;
    opts.spin = 1;
    opts.scf.backend = be;
    opts.scf.conv_tol = 1e-10;
    opts.scf.diis_tol = 1e-8;

    gpu4dft::MolecularScf mscf(basis, atoms, 1, opts);
    const auto wres = mscf.run();
    const auto& sres = wres.scf;
    check_true(("h_atom." + xc + ".conv").c_str(), sres.converged);
    check_true(("h_atom." + xc + ".unres").c_str(), sres.unrestricted);
    check_true(("h_atom." + xc + ".na_nb").c_str(), sres.n_alpha == 1 && sres.n_beta == 0);
    check_near(("h_atom." + xc + ".s2").c_str(), sres.s2, 0.75, 1e-12);
    check_near(("h_atom." + xc + ".mult").c_str(), sres.multiplicity, 2.0, 1e-12);

    double db_max = 0.0;
    for (double v : sres.density_beta) db_max = std::max(db_max, std::fabs(v));
    check_near(("h_atom." + xc + ".db_zero").c_str(), db_max, 0.0, 1e-14);
    if (be == gpu4dft::Backend::Cuda) {
      check_true(("h_atom." + xc + ".steady_allocs").c_str(), sres.steady_state_allocations == 0);
    }

    if (xc == "hf") {
      // One-electron H atom self-interaction 2e gradient must be identically zero
      const auto g2e = gpu4dft::jk_nuclear_grad_uhf(
          basis, atoms, sres.density_alpha.data(), sres.density_beta.data(), 1.0, be, 1e-13);
      check_vec_near("h_atom.hf.grad2e_zero", g2e, {0.0, 0.0, 0.0}, 1e-14);
    }
  }
}

void test_closed_shell_limit(gpu4dft::Backend be) {
  const std::vector<gpu4dft::Atom> atoms = {
      {8, {0.0, 0.0, 0.22}},
      {1, {0.0, 1.43, -0.88}},
      {1, {0.0, -1.43, -0.88}},
  };
  const auto basis = make_sto3g_basis(atoms);

  for (const std::string& xc : {"hf", "b3lyp"}) {
    gpu4dft::WorkflowOptions r_opts;
    r_opts.xc = xc;
    r_opts.scf.backend = be;
    r_opts.scf.conv_tol = 1e-10;
    r_opts.scf.diis_tol = 1e-8;
    const auto r_res = gpu4dft::run_scf(basis, atoms, 10, r_opts);
    const auto r_grad = gpu4dft::run_grad(basis, atoms, r_res.scf, r_opts);

    gpu4dft::WorkflowOptions u_opts = r_opts;
    u_opts.unrestricted = true;
    u_opts.spin = 0;
    const auto u_res = gpu4dft::run_scf(basis, atoms, 10, u_opts);
    const auto u_grad = gpu4dft::run_grad(basis, atoms, u_res.scf, u_opts);

    check_true(("cs." + xc + ".conv").c_str(), r_res.scf.converged && u_res.scf.converged);
    check_near(("cs." + xc + ".etot").c_str(), u_res.scf.total_energy, r_res.scf.total_energy,
               1e-10);
    check_near(("cs." + xc + ".s2").c_str(), u_res.scf.s2, 0.0, 1e-10);
    check_vec_near(("cs." + xc + ".grad").c_str(), u_grad.total_grad, r_grad.total_grad, 1e-10);
  }
}

void test_h2o_cation_doublet_host_cuda_and_fd(bool have_cuda) {
  // Bent C2v H2O+ (^2B1, 9 electrons, spin=1) avoids the linear ^2Pi rotational
  // degeneracy of OH while exercising open-shell HF/GGA/hybrid/mGGA parity and FD.
  const std::vector<gpu4dft::Atom> atoms = {
      {8, {0.0, 0.0, 0.2217}},
      {1, {0.0, 1.4309, -0.8867}},
      {1, {0.0, -1.4309, -0.8867}},
  };
  const auto basis = make_sto3g_basis(atoms);

  for (const std::string& xc : {"hf", "pbe", "b3lyp", "tpss"}) {
    gpu4dft::WorkflowOptions opts;
    opts.xc = xc;
    opts.unrestricted = true;
    opts.spin = 1;
    opts.scf.backend = gpu4dft::Backend::Host;
    opts.scf.conv_tol = 1e-10;
    opts.scf.diis_tol = 1e-8;
    opts.scf.dynamic_precision = false;

    gpu4dft::MolecularScf mscf_h(basis, atoms, 9, opts);
    const auto w_h = mscf_h.run();
    const auto g_h = mscf_h.compute_gradient();
    check_true(("h2op_host." + xc + ".conv").c_str(), w_h.scf.converged);

    // Translational invariance across all 3 atoms
    for (int d = 0; d < 3; ++d) {
      check_near(("h2op_host." + xc + ".trans").c_str(),
                 g_h.total_grad[d] + g_h.total_grad[3 + d] + g_h.total_grad[6 + d], 0.0, 1e-9);
    }

    // run_response must throw NotImplemented on unrestricted
    bool threw_resp = false;
    try {
      (void)mscf_h.run_response();
    } catch (const gpu4dft::NotImplemented&) {
      threw_resp = true;
    }
    check_true(("h2op_host." + xc + ".resp_guard").c_str(), threw_resp);

    if (have_cuda) {
      opts.scf.backend = gpu4dft::Backend::Cuda;
      gpu4dft::MolecularScf mscf_c(basis, atoms, 9, opts);
      const auto w_c = mscf_c.run();
      const auto g_c = mscf_c.compute_gradient();
      check_true(("h2op_cuda." + xc + ".conv").c_str(), w_c.scf.converged);
      check_true(("h2op_cuda." + xc + ".steady").c_str(), w_c.scf.steady_state_allocations == 0);
      check_near(("h2op_parity." + xc + ".etot").c_str(), w_c.scf.total_energy, w_h.scf.total_energy,
                 1e-10);
      check_near(("h2op_parity." + xc + ".s2").c_str(), w_c.scf.s2, w_h.scf.s2, 1e-10);
      check_vec_near(("h2op_parity." + xc + ".grad").c_str(), g_c.total_grad, g_h.total_grad, 1e-9);
    }
  }

  // Central finite-difference check on H2O+ UHF & B3LYP analytical nuclear gradient
  const double h = 1e-4;
  for (const std::string& xc : {"hf", "b3lyp"}) {
    const gpu4dft::Backend be =
        have_cuda ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
    gpu4dft::WorkflowOptions opts;
    opts.xc = xc;
    opts.unrestricted = true;
    opts.spin = 1;
    opts.scf.backend = be;
    opts.scf.conv_tol = 1e-11;
    opts.scf.diis_tol = 1e-9;
    opts.scf.dynamic_precision = false;
    opts.grid.level = 3;

    const auto g0 = gpu4dft::run_grad(basis, atoms, gpu4dft::run_scf(basis, atoms, 9, opts).scf,
                                      opts);

    auto eval_e = [&](double dz_h1) {
      auto at = atoms;
      at[1].center[2] += dz_h1;
      const auto b = make_sto3g_basis(at);
      return gpu4dft::run_scf(b, at, 9, opts).scf.total_energy;
    };
    const double g_fd = (eval_e(h) - eval_e(-h)) / (2.0 * h);
    check_near(("h2op_fd." + xc + ".h1z").c_str(), g0.total_grad[5], g_fd, 1e-6);
  }
}

void test_direct_jk_uhf_spinflip_and_nonpsd_xc(gpu4dft::Backend be) {
  const std::vector<gpu4dft::Atom> atoms = {
      {8, {0.0, 0.0, 0.0}},
      {1, {0.0, 0.0, 1.85}},
  };
  const auto basis = make_sto3g_basis(atoms);
  const int nao = basis.nao();
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;

  // Non-PSD spin density matrices (one negative eigenvalue in alpha) -> verify eval_vxc_uks finite
  std::vector<double> dm_a(n2, 0.0), dm_b(n2, 0.0);
  for (int i = 0; i < nao; ++i) {
    dm_a[i * nao + i] = (i == 0) ? -0.35 : 0.25;
    dm_b[i * nao + i] = 0.22;
    if (i + 1 < nao) {
      dm_a[i * nao + (i + 1)] = dm_a[(i + 1) * nao + i] = 0.12;
      dm_b[i * nao + (i + 1)] = dm_b[(i + 1) * nao + i] = -0.08;
    }
  }

  std::vector<gpu4dft::XcAtom> xatoms;
  for (const auto& a : atoms) xatoms.push_back({a.atomic_number, a.center});
  gpu4dft::GridOptions gopts;
  gopts.level = 2;
  gopts.backend = be;
  gpu4dft::MolecularGrid grid(xatoms, basis, gopts);
  for (const std::string& xc : {"pbe", "b3lyp", "tpss"}) {
    gpu4dft::XcOptions xopts;
    xopts.backend = be;
    xopts.spin_polarized = true;
    xopts.use_libxc = true;
    gpu4dft::XcEngine eng(basis, grid, xc, xopts);
    const auto ures = eng.eval_vxc_uks(dm_a, dm_b);
    check_true(("nonpsd." + xc + ".exc_finite").c_str(), std::isfinite(ures.exc));
    for (double v : ures.vxc_alpha) check_true(("nonpsd." + xc + ".va_finite").c_str(), std::isfinite(v));
    for (double v : ures.vxc_beta) check_true(("nonpsd." + xc + ".vb_finite").c_str(), std::isfinite(v));
  }

  // Pure spin-flip DirectJKEngine::compute_jk_uhf test: dDa = -dDb -> dD_tot == 0, J == 0, Ka == -Kb
  std::vector<double> spin_flip(n2, 0.0), neg_spin_flip(n2, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      const double v = 0.15 * std::exp(-0.5 * std::abs(i - j));
      spin_flip[i * nao + j] = v;
      neg_spin_flip[i * nao + j] = -v;
    }
  }
  gpu4dft::DirectJKEngine djk(basis);
  gpu4dft::JKOptions jopts;
  jopts.backend = be;
  jopts.precision = gpu4dft::Precision::FP64;
  jopts.schwarz_cutoff = 1e-13;
  std::vector<double> vj(n2, 0.0), vka(n2, 0.0), vkb(n2, 0.0);
  djk.compute_jk_uhf(spin_flip.data(), neg_spin_flip.data(), vj.data(), vka.data(), vkb.data(),
                     jopts);
  check_vec_near("spinflip.j_zero", vj, std::vector<double>(n2, 0.0), 1e-13);
  std::vector<double> ka_plus_kb(n2, 0.0);
  for (std::size_t i = 0; i < n2; ++i) ka_plus_kb[i] = vka[i] + vkb[i];
  check_vec_near("spinflip.ka_neg_kb", ka_plus_kb, std::vector<double>(n2, 0.0), 1e-13);
  double ka_norm = 0.0;
  for (double v : vka) ka_norm += std::fabs(v);
  check_true("spinflip.ka_nonzero", ka_norm > 1.0);
}

}  // namespace

int main() {
  test_validation_guards();
  const bool have_cuda = gpu4dft::backend_available(gpu4dft::Backend::Cuda);
  test_h_atom_nbeta0(gpu4dft::Backend::Host);
  test_closed_shell_limit(gpu4dft::Backend::Host);
  test_direct_jk_uhf_spinflip_and_nonpsd_xc(gpu4dft::Backend::Host);
  if (have_cuda) {
    test_h_atom_nbeta0(gpu4dft::Backend::Cuda);
    test_closed_shell_limit(gpu4dft::Backend::Cuda);
    test_direct_jk_uhf_spinflip_and_nonpsd_xc(gpu4dft::Backend::Cuda);
  }
  test_h2o_cation_doublet_host_cuda_and_fd(have_cuda);

  if (g_failures != 0) {
    std::printf("%d failures in test_uhf_uks\n", g_failures);
    return 1;
  }
  std::printf("test_uhf_uks: ALL PASSED\n");
  return 0;
}
