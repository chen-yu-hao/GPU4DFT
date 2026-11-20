#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/jk_engine.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/types.hpp"
#include "gpu4dft/workflow.hpp"

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

std::vector<gpu4dft::Atom> make_h2o_atoms() {
  return {
      gpu4dft::Atom{8, {0.0, 0.0, 0.22166}},
      gpu4dft::Atom{1, {0.0, 1.4309, -0.88665}},
      gpu4dft::Atom{1, {0.0, -1.4309, -0.88665}},
  };
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

gpu4dft::Basis make_aux_basis(const std::vector<gpu4dft::Atom>& atoms, bool include_g = false) {
  std::vector<gpu4dft::Shell> shells;
  for (const auto& at : atoms) {
    if (at.atomic_number == 1) {
      shells.push_back(gpu4dft::Shell{0, at.center, {8.0, 2.0, 0.5}, {0.3, 0.5, 0.3}});
      shells.push_back(gpu4dft::Shell{1, at.center, {1.5, 0.4}, {0.6, 0.5}});
    } else if (at.atomic_number == 8) {
      shells.push_back(gpu4dft::Shell{0, at.center, {80.0, 18.0, 4.5, 1.1}, {0.2, 0.4, 0.4, 0.2}});
      shells.push_back(gpu4dft::Shell{1, at.center, {12.0, 3.0, 0.8}, {0.3, 0.5, 0.3}});
      shells.push_back(gpu4dft::Shell{2, at.center, {4.0, 1.0}, {0.5, 0.6}});
      if (include_g) {
        shells.push_back(gpu4dft::Shell{4, at.center, {1.4}, {1.0}});
      }
    }
  }
  return gpu4dft::Basis(std::move(shells));
}

std::vector<double> make_nonsym_dm(int nao) {
  const std::size_t n = static_cast<std::size_t>(nao);
  std::vector<double> dm(n * n, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      const double sym = 0.08 * std::exp(-0.25 * std::abs(i - j));
      const double skew = (i < j ? 1.0 : (i > j ? -0.4 : 0.0)) * 0.15 *
                          std::sin(0.7 * (i + 1) - 1.3 * (j + 1));
      dm[i * n + j] = sym + skew;
    }
  }
  return dm;
}

void test_no_eager_device_b_and_lazy_builder() {
  const auto atoms = make_h2o_atoms();
  const auto basis = make_sto3g_basis(atoms);
  const auto aux = make_aux_basis(atoms);

  gpu4dft::RIJKEngine ri_h(basis, aux, gpu4dft::Backend::Host, 0.4);
  check_true("lazy.host.no_device_b", !ri_h.has_device_b() && ri_h.device_b_ptr() == nullptr);
  check_true("lazy.host.no_device_bw",
             !ri_h.has_device_b_omega() && ri_h.device_b_omega_ptr() == nullptr);

  if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
    gpu4dft::RIJKEngine ri_c(basis, aux, gpu4dft::Backend::Cuda, 0.4);
    check_true("lazy.cuda.no_device_b", !ri_c.has_device_b() && ri_c.device_b_ptr() == nullptr);
    check_true("lazy.cuda.no_device_bw",
               !ri_c.has_device_b_omega() && ri_c.device_b_omega_ptr() == nullptr);
    check_vec_near("lazy.b_tensor.host_vs_cuda", ri_c.b_tensor(), ri_h.b_tensor(), 1e-11);
    check_vec_near("lazy.b_omega.host_vs_cuda", ri_c.b_omega_tensor(), ri_h.b_omega_tensor(),
                   1e-11);
  }
}

void test_mo_vs_dm_and_fallbacks() {
  const auto atoms = make_h2o_atoms();
  const auto basis = make_sto3g_basis(atoms);
  const auto aux = make_aux_basis(atoms);
  const int nao = basis.nao();
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;

  const int n_occ = 5;
  std::vector<double> c_occ(static_cast<std::size_t>(nao) * n_occ, 0.0);
  for (int u = 0; u < nao; ++u) {
    for (int i = 0; i < n_occ; ++i) {
      c_occ[static_cast<std::size_t>(u) * n_occ + i] =
          0.25 * std::cos(0.6 * (u + 1) + 0.9 * (i + 1));
    }
  }
  std::vector<double> dm(nao2, 0.0);
  gpu4dft::gemm(nao, nao, n_occ, 2.0, c_occ.data(), n_occ, false, c_occ.data(), n_occ, true, 0.0,
                dm.data(), nao, gpu4dft::Backend::Host);

  const int n_alpha = 4, n_beta = 2;
  std::vector<double> ca(static_cast<std::size_t>(nao) * n_alpha, 0.0);
  std::vector<double> cb(static_cast<std::size_t>(nao) * n_beta, 0.0);
  for (int u = 0; u < nao; ++u) {
    for (int i = 0; i < n_alpha; ++i) ca[u * n_alpha + i] = c_occ[u * n_occ + i];
    for (int i = 0; i < n_beta; ++i) cb[u * n_beta + i] = c_occ[u * n_occ + (i + 1)];
  }
  std::vector<double> da(nao2, 0.0), db(nao2, 0.0);
  gpu4dft::gemm(nao, nao, n_alpha, 1.0, ca.data(), n_alpha, false, ca.data(), n_alpha, true, 0.0,
                da.data(), nao, gpu4dft::Backend::Host);
  gpu4dft::gemm(nao, nao, n_beta, 1.0, cb.data(), n_beta, false, cb.data(), n_beta, true, 0.0,
                db.data(), nao, gpu4dft::Backend::Host);

  gpu4dft::RIJKEngine ri_h(basis, aux, gpu4dft::Backend::Host, 0.4);
  gpu4dft::JKOptions opts;
  opts.backend = gpu4dft::Backend::Host;
  opts.compute_j = true;
  opts.compute_k = true;
  opts.k_scale = 1.0;

  std::vector<double> vj_mo_h(nao2), vk_mo_h(nao2), vj_dm_h(nao2), vk_dm_h(nao2);
  ri_h.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_mo_h.data(), vk_mo_h.data(), opts);
  ri_h.compute_jk(dm.data(), vj_dm_h.data(), vk_dm_h.data(), opts);
  check_vec_near("host.mo_vs_dm.J", vj_mo_h, vj_dm_h, 1e-12);
  check_vec_near("host.mo_vs_dm.K", vk_mo_h, vk_dm_h, 1e-11);

  // n_occ == 0 => zero K
  std::vector<double> vj_zero_k(nao2), vk_zero(nao2, 99.0);
  ri_h.compute_jk_mo(dm.data(), nullptr, 0, 2.0, vj_zero_k.data(), vk_zero.data(), opts);
  check_vec_near("host.nocc0.J", vj_zero_k, vj_dm_h, 1e-12);
  check_vec_near("host.nocc0.K", vk_zero, std::vector<double>(nao2, 0.0), 1e-14);

  // RSH Host
  gpu4dft::JKOptions rsh_opts = opts;
  rsh_opts.rsh_mode = true;
  rsh_opts.omega = 0.4;
  rsh_opts.k_sr = 0.25;
  rsh_opts.k_lr = 1.0;
  std::vector<double> vj_rsh_h(nao2), vk_rsh_h(nao2);
  ri_h.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_rsh_h.data(), vk_rsh_h.data(),
                     rsh_opts);

  // Non-symmetric DM Host
  const auto dm_ns = make_nonsym_dm(nao);
  gpu4dft::JKOptions ns_opts = opts;
  ns_opts.symmetrize_dm = false;
  std::vector<double> vj_ns_h(nao2), vk_ns_h(nao2);
  ri_h.compute_jk(dm_ns.data(), vj_ns_h.data(), vk_ns_h.data(), ns_opts);

  // UHF Host (including n_beta == 0)
  std::vector<double> vj_uks_h(nao2), vka_uks_h(nao2), vkb_uks_h(nao2);
  ri_h.compute_jk_uks_mo(da.data(), db.data(), ca.data(), n_alpha, cb.data(), n_beta,
                         vj_uks_h.data(), vka_uks_h.data(), vkb_uks_h.data(), opts);
  std::vector<double> db_zero(nao2, 0.0), vj_nb0_h(nao2), vka_nb0_h(nao2), vkb_nb0_h(nao2, 55.0);
  ri_h.compute_jk_uks_mo(da.data(), db_zero.data(), ca.data(), n_alpha, nullptr, 0,
                         vj_nb0_h.data(), vka_nb0_h.data(), vkb_nb0_h.data(), opts);
  check_vec_near("host.uks_nb0.K_beta", vkb_nb0_h, std::vector<double>(nao2, 0.0), 1e-14);

  if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
    gpu4dft::RIJKEngine ri_c(basis, aux, gpu4dft::Backend::Cuda, 0.4);
    gpu4dft::JKOptions copts = opts;
    copts.backend = gpu4dft::Backend::Cuda;

    std::vector<double> vj_mo_c(nao2), vk_mo_c(nao2), vj_dm_c(nao2), vk_dm_c(nao2);
    ri_c.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_mo_c.data(), vk_mo_c.data(), copts);
    ri_c.compute_jk(dm.data(), vj_dm_c.data(), vk_dm_c.data(), copts);
    check_vec_near("cuda.mo_vs_host.J", vj_mo_c, vj_mo_h, 1e-11);
    check_vec_near("cuda.mo_vs_host.K", vk_mo_c, vk_mo_h, 1e-11);
    check_vec_near("cuda.dm_vs_host.K", vk_dm_c, vk_dm_h, 1e-11);

    // OzakiInt8 & FP32
    gpu4dft::JKOptions oz_opts = copts;
    oz_opts.gemm_mode = gpu4dft::GemmMode::OzakiInt8;
    std::vector<double> vj_oz(nao2), vk_oz(nao2);
    ri_c.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_oz.data(), vk_oz.data(), oz_opts);
    check_vec_near("cuda.ozaki.K", vk_oz, vk_mo_h, 1e-9);

    gpu4dft::JKOptions f32_opts = copts;
    f32_opts.precision = gpu4dft::Precision::FP32;
    f32_opts.gemm_mode = gpu4dft::GemmMode::FP32;
    std::vector<double> vj_f32(nao2), vk_f32(nao2);
    ri_c.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_f32.data(), vk_f32.data(),
                       f32_opts);
    check_vec_near("cuda.fp32.K", vk_f32, vk_mo_h, 1e-4);

    // RSH CUDA
    gpu4dft::JKOptions rsh_c = rsh_opts;
    rsh_c.backend = gpu4dft::Backend::Cuda;
    std::vector<double> vj_rsh_c(nao2), vk_rsh_c(nao2);
    ri_c.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_rsh_c.data(), vk_rsh_c.data(),
                       rsh_c);
    check_vec_near("cuda.rsh.J", vj_rsh_c, vj_rsh_h, 1e-11);
    check_vec_near("cuda.rsh.K", vk_rsh_c, vk_rsh_h, 1e-11);

    // Zero K weight RSH
    gpu4dft::JKOptions rsh_zero = rsh_c;
    rsh_zero.k_sr = 0.0;
    rsh_zero.k_lr = 0.0;
    std::vector<double> vj_rz(nao2), vk_rz(nao2, 77.0);
    ri_c.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj_rz.data(), vk_rz.data(), rsh_zero);
    check_vec_near("cuda.rsh_zero_k", vk_rz, std::vector<double>(nao2, 0.0), 1e-14);

    // Non-symmetric DM CUDA
    gpu4dft::JKOptions ns_c = ns_opts;
    ns_c.backend = gpu4dft::Backend::Cuda;
    std::vector<double> vj_ns_c(nao2), vk_ns_c(nao2);
    ri_c.compute_jk(dm_ns.data(), vj_ns_c.data(), vk_ns_c.data(), ns_c);
    check_vec_near("cuda.nonsym.J", vj_ns_c, vj_ns_h, 1e-11);
    check_vec_near("cuda.nonsym.K", vk_ns_c, vk_ns_h, 1e-11);

    // UHF CUDA (both packed and n_beta == 0)
    std::vector<double> vj_uks_c(nao2), vka_uks_c(nao2), vkb_uks_c(nao2);
    ri_c.compute_jk_uks_mo(da.data(), db.data(), ca.data(), n_alpha, cb.data(), n_beta,
                           vj_uks_c.data(), vka_uks_c.data(), vkb_uks_c.data(), copts);
    check_vec_near("cuda.uks.J", vj_uks_c, vj_uks_h, 1e-11);
    check_vec_near("cuda.uks.Ka", vka_uks_c, vka_uks_h, 1e-11);
    check_vec_near("cuda.uks.Kb", vkb_uks_c, vkb_uks_h, 1e-11);

    std::vector<double> vj_nb0_c(nao2), vka_nb0_c(nao2), vkb_nb0_c(nao2, 55.0);
    ri_c.compute_jk_uks_mo(da.data(), db_zero.data(), ca.data(), n_alpha, nullptr, 0,
                           vj_nb0_c.data(), vka_nb0_c.data(), vkb_nb0_c.data(), copts);
    check_vec_near("cuda.uks_nb0.J", vj_nb0_c, vj_nb0_h, 1e-11);
    check_vec_near("cuda.uks_nb0.Ka", vka_nb0_c, vka_nb0_h, 1e-11);
    check_vec_near("cuda.uks_nb0.Kb", vkb_nb0_c, std::vector<double>(nao2, 0.0), 1e-14);
  }
}

void test_scf_and_gradients() {
  const auto atoms = make_h2o_atoms();
  auto run_case = [&](const std::string& xc, int spin, int charge, gpu4dft::Backend be,
                      gpu4dft::DensitySolver solver = gpu4dft::DensitySolver::Diag) {
    gpu4dft::WorkflowOptions opts;
    opts.jk_mode = gpu4dft::JKMode::RI;
    opts.xc = xc;
    opts.unrestricted = (spin != 0);
    opts.spin = spin;
    opts.scf.backend = be;
    opts.scf.solver = solver;
    opts.scf.unrestricted = (spin != 0);
    opts.scf.spin = spin;
    opts.scf.conv_tol = 1e-10;
    opts.scf.diis_tol = 1e-8;
    opts.scf.dynamic_precision = false;
    opts.scf.gemm_mode = gpu4dft::GemmMode::FP64;
    opts.grid.backend = be;
    opts.grid.level = 3;

    const auto basis = make_sto3g_basis(atoms);
    const auto aux = make_aux_basis(atoms);
    gpu4dft::MolecularScf mol(basis, atoms, 10 - charge, opts, aux);
    const auto res = mol.run();
    const auto gres = mol.compute_gradient(res.scf);
    return std::make_pair(res.scf, gres);
  };

  auto fd_grad = [&](const std::string& xc, int spin, int charge, gpu4dft::Backend be) {
    const double h = 1e-4;
    std::vector<double> g_fd(atoms.size() * 3, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int d = 0; d < 3; ++d) {
        auto ap = atoms;
        auto am = atoms;
        ap[a].center[d] += h;
        am[a].center[d] -= h;

        gpu4dft::WorkflowOptions opts;
        opts.jk_mode = gpu4dft::JKMode::RI;
        opts.xc = xc;
        opts.unrestricted = (spin != 0);
        opts.spin = spin;
        opts.scf.backend = be;
        opts.scf.unrestricted = (spin != 0);
        opts.scf.spin = spin;
        opts.scf.conv_tol = 1e-11;
        opts.scf.diis_tol = 1e-9;
        opts.scf.dynamic_precision = false;
        opts.grid.backend = be;
        opts.grid.level = 3;

        gpu4dft::MolecularScf mp(make_sto3g_basis(ap), ap, 10 - charge, opts, make_aux_basis(ap));
        gpu4dft::MolecularScf mm(make_sto3g_basis(am), am, 10 - charge, opts, make_aux_basis(am));
        const double ep = mp.run().scf.total_energy;
        const double em = mm.run().scf.total_energy;
        g_fd[3 * a + d] = (ep - em) / (2.0 * h);
      }
    }
    return g_fd;
  };

  // Closed-shell HF, CAM-B3LYP, wB97X + Open-shell H2O+ UHF
  struct Case {
    const char* name;
    const char* xc;
    int spin;
    int charge;
  };
  const Case cases[] = {
      {"rhf", "HF", 0, 0},
      {"pbe", "PBE", 0, 0},
      {"cam_b3lyp", "CAM-B3LYP", 0, 0},
      {"wb97x", "wB97X", 0, 0},
      {"uhf_cation", "HF", 1, 1},
  };

  for (const auto& c : cases) {
    const auto [res_h, g_h] = run_case(c.xc, c.spin, c.charge, gpu4dft::Backend::Host);
    check_true((std::string("scf.conv.host.") + c.name).c_str(), res_h.converged);
    const auto g_fd = fd_grad(c.xc, c.spin, c.charge, gpu4dft::Backend::Host);
    check_vec_near((std::string("grad.fd.host.") + c.name).c_str(), g_h.total_grad, g_fd, 1e-5);

    // Translational invariance
    double sum[3] = {0.0, 0.0, 0.0};
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int d = 0; d < 3; ++d) sum[d] += g_h.total_grad[3 * a + d];
    }
    for (int d = 0; d < 3; ++d) {
      check_near((std::string("grad.trans_inv.") + c.name).c_str(), sum[d], 0.0, 1e-8);
    }

    if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
      const auto [res_c, g_c] = run_case(c.xc, c.spin, c.charge, gpu4dft::Backend::Cuda);
      check_true((std::string("scf.conv.cuda.") + c.name).c_str(), res_c.converged);
      check_true((std::string("scf.zero_alloc.") + c.name).c_str(),
                 res_c.steady_state_allocations == 0);
      check_near((std::string("scf.energy.cuda_vs_host.") + c.name).c_str(), res_c.total_energy,
                 res_h.total_energy, 1e-10);
      check_vec_near((std::string("grad.cuda_vs_host.") + c.name).c_str(), g_c.total_grad,
                     g_h.total_grad, 1e-10);
    }
  }

  // DM-only ScfResult gradient fallback & SP2 solver parity
  {
    const auto [res_diag, g_diag] = run_case("HF", 0, 0, gpu4dft::Backend::Host);
    gpu4dft::ScfResult dm_only = res_diag;
    dm_only.mo_coefficients.clear();
    gpu4dft::WorkflowOptions opts;
    opts.scf.backend = gpu4dft::Backend::Host;
    opts.jk_mode = gpu4dft::JKMode::RI;
    opts.xc = "HF";
    gpu4dft::MolecularScf mol(make_sto3g_basis(atoms), atoms, 10, opts, make_aux_basis(atoms));
    const auto g_dm_h = mol.compute_gradient(dm_only);
    check_vec_near("grad.dm_only_fallback.host", g_dm_h.total_grad, g_diag.total_grad, 1e-11);

    if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
      opts.scf.backend = gpu4dft::Backend::Cuda;
      gpu4dft::MolecularScf mol_c(make_sto3g_basis(atoms), atoms, 10, opts, make_aux_basis(atoms));
      const auto g_dm_c = mol_c.compute_gradient(dm_only);
      check_vec_near("grad.dm_only_fallback.cuda", g_dm_c.total_grad, g_diag.total_grad, 1e-10);

      const auto [res_sp2, g_sp2] =
          run_case("HF", 0, 0, gpu4dft::Backend::Cuda, gpu4dft::DensitySolver::SP2);
      check_near("scf.sp2.energy", res_sp2.total_energy, res_diag.total_energy, 1e-9);
      check_vec_near("grad.sp2.cuda", g_sp2.total_grad, g_diag.total_grad, 1e-9);
    }
  }

  // g-shell (l = 4) auxiliary basis 2e RI gradient vs FD of 0.5*Tr(D*(J - 0.5*K))
  {
    const auto basis = make_sto3g_basis(atoms);
    const auto aux_g = make_aux_basis(atoms, /*include_g=*/true);
    const int nao = basis.nao();
    const int n_occ = 5;
    std::vector<double> c_occ(static_cast<std::size_t>(nao) * n_occ, 0.0);
    for (int u = 0; u < nao; ++u) {
      for (int i = 0; i < n_occ; ++i) {
        c_occ[static_cast<std::size_t>(u) * n_occ + i] =
            0.2 * std::sin(0.5 * (u + 1) + 0.7 * (i + 1));
      }
    }
    std::vector<double> dm(static_cast<std::size_t>(nao) * nao, 0.0);
    gpu4dft::gemm(nao, nao, n_occ, 2.0, c_occ.data(), n_occ, false, c_occ.data(), n_occ, true, 0.0,
                  dm.data(), nao, gpu4dft::Backend::Host);

    gpu4dft::RIJKEngine ri_h(basis, aux_g, gpu4dft::Backend::Host, 0.4);
    const auto g_h = ri_h.compute_ri_jk_grad(atoms, dm.data(), c_occ.data(), n_occ, nullptr,
                                              nullptr, 0, false, 1.0, 0.25, 1.0, 0.4,
                                              gpu4dft::Backend::Host);

    std::vector<double> g_fd(atoms.size() * 3, 0.0);
    const double h = 1e-4;
    auto eval_e2 = [&](const std::vector<gpu4dft::Atom>& at_list) {
      const auto b = make_sto3g_basis(at_list);
      const auto ag = make_aux_basis(at_list, true);
      gpu4dft::RIJKEngine eng(b, ag, gpu4dft::Backend::Host, 0.4);
      gpu4dft::JKOptions jko;
      jko.backend = gpu4dft::Backend::Host;
      jko.compute_j = true;
      jko.compute_k = true;
      jko.rsh_mode = true;
      jko.omega = 0.4;
      jko.k_sr = 0.25;
      jko.k_lr = 1.0;
      std::vector<double> vj(dm.size()), vk(dm.size());
      eng.compute_jk_mo(dm.data(), c_occ.data(), n_occ, 2.0, vj.data(), vk.data(), jko);
      double e = 0.0;
      for (std::size_t idx = 0; idx < dm.size(); ++idx) {
        e += 0.5 * dm[idx] * (vj[idx] - 0.5 * vk[idx]);
      }
      return e;
    };
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int d = 0; d < 3; ++d) {
        auto ap = atoms, am = atoms;
        ap[a].center[d] += h;
        am[a].center[d] -= h;
        g_fd[3 * a + d] = (eval_e2(ap) - eval_e2(am)) / (2.0 * h);
      }
    }
    check_vec_near("grad.aux_l4.host_vs_fd", g_h, g_fd, 1e-6);

    if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
      gpu4dft::RIJKEngine ri_c(basis, aux_g, gpu4dft::Backend::Cuda, 0.4);
      const auto g_c = ri_c.compute_ri_jk_grad(atoms, dm.data(), c_occ.data(), n_occ, nullptr,
                                                nullptr, 0, false, 1.0, 0.25, 1.0, 0.4,
                                                gpu4dft::Backend::Cuda);
      check_vec_near("grad.aux_l4.cuda_vs_host", g_c, g_h, 1e-10);
    }
  }
}

void test_indefinite_dm_ri_jk_grad_and_run_grad() {
  const auto atoms = make_h2o_atoms();
  const auto basis = make_sto3g_basis(atoms);
  const auto aux = make_aux_basis(atoms);
  const int nao = basis.nao();
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;

  // Build symmetric indefinite DM (both positive and negative eigenvalues)
  std::vector<double> dm_a(nao2, 0.0), dm_b(nao2, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      dm_a[static_cast<std::size_t>(i) * nao + j] =
          0.35 * std::sin(0.9 * (i + 1) + 0.9 * (j + 1)) + ((i == j) ? ((i % 2 == 0) ? 0.4 : -0.5) : 0.0);
      dm_b[static_cast<std::size_t>(i) * nao + j] =
          0.25 * std::cos(0.7 * (i + 1) + 0.7 * (j + 1)) + ((i == j) ? ((i % 2 == 0) ? -0.45 : 0.35) : 0.0);
    }
  }
  // Confirm dm_a has at least one negative and one positive eigenvalue
  {
    std::vector<double> w(nao, 0.0), U(nao2, 0.0);
    std::vector<double> tmp = dm_a;
    gpu4dft::syevd(nao, tmp.data(), w.data(), U.data(), gpu4dft::Backend::Host);
    check_true("indef_dm.has_neg_and_pos_eig", w.front() < -0.1 && w.back() > 0.1);
  }

  const double h = 1e-4;
  // 1. Restricted indefinite DM (RSH omega=0.4, k_sr=0.25, k_lr=1.0)
  {
    gpu4dft::RIJKEngine ri_h(basis, aux, gpu4dft::Backend::Host, 0.4);
    const auto g_h = ri_h.compute_ri_jk_grad(atoms, dm_a.data(), nullptr, -1, nullptr, nullptr, 0,
                                              false, 1.0, 0.25, 1.0, 0.4, gpu4dft::Backend::Host);

    auto eval_e2 = [&](const std::vector<gpu4dft::Atom>& at_list) {
      gpu4dft::RIJKEngine eng(make_sto3g_basis(at_list), make_aux_basis(at_list),
                              gpu4dft::Backend::Host, 0.4);
      gpu4dft::JKOptions jko;
      jko.backend = gpu4dft::Backend::Host;
      jko.rsh_mode = true;
      jko.omega = 0.4;
      jko.k_sr = 0.25;
      jko.k_lr = 1.0;
      std::vector<double> vj(nao2), vk(nao2);
      eng.compute_jk(dm_a.data(), vj.data(), vk.data(), jko);
      double e = 0.0;
      for (std::size_t idx = 0; idx < nao2; ++idx) {
        e += 0.5 * dm_a[idx] * (vj[idx] - 0.5 * vk[idx]);
      }
      return e;
    };
    std::vector<double> g_fd(atoms.size() * 3, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int d = 0; d < 3; ++d) {
        auto ap = atoms, am = atoms;
        ap[a].center[d] += h;
        am[a].center[d] -= h;
        g_fd[3 * a + d] = (eval_e2(ap) - eval_e2(am)) / (2.0 * h);
      }
    }
    check_vec_near("indef_dm.rsh.host_vs_fd", g_h, g_fd, 1e-7);

    if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
      gpu4dft::RIJKEngine ri_c(basis, aux, gpu4dft::Backend::Cuda, 0.4);
      const auto g_c = ri_c.compute_ri_jk_grad(atoms, dm_a.data(), nullptr, -1, nullptr, nullptr,
                                                0, false, 1.0, 0.25, 1.0, 0.4,
                                                gpu4dft::Backend::Cuda);
      check_vec_near("indef_dm.rsh.cuda_vs_fd", g_c, g_fd, 1e-7);
      check_vec_near("indef_dm.rsh.cuda_vs_host", g_c, g_h, 1e-11);
    }
  }

  // 2. Unrestricted indefinite DM (c_occ_alpha=nullptr, c_occ_beta=nullptr)
  {
    gpu4dft::RIJKEngine ri_h(basis, aux, gpu4dft::Backend::Host, 0.0);
    const auto g_h = ri_h.compute_ri_jk_grad(atoms, dm_a.data(), nullptr, -1, dm_b.data(), nullptr,
                                              -1, true, 1.0, 1.0, 1.0, 0.0, gpu4dft::Backend::Host);

    std::vector<double> dm_tot(nao2, 0.0);
    for (std::size_t idx = 0; idx < nao2; ++idx) dm_tot[idx] = dm_a[idx] + dm_b[idx];

    auto eval_uhf_e2 = [&](const std::vector<gpu4dft::Atom>& at_list) {
      gpu4dft::RIJKEngine eng(make_sto3g_basis(at_list), make_aux_basis(at_list),
                              gpu4dft::Backend::Host, 0.0);
      gpu4dft::JKOptions jko;
      jko.backend = gpu4dft::Backend::Host;
      std::vector<double> vj(nao2), vka(nao2), vkb(nao2);
      eng.compute_jk(dm_tot.data(), vj.data(), nullptr, jko);
      eng.compute_jk(dm_a.data(), nullptr, vka.data(), jko);
      eng.compute_jk(dm_b.data(), nullptr, vkb.data(), jko);
      double e = 0.0;
      for (std::size_t idx = 0; idx < nao2; ++idx) {
        e += 0.5 * (dm_tot[idx] * vj[idx] - dm_a[idx] * vka[idx] - dm_b[idx] * vkb[idx]);
      }
      return e;
    };
    std::vector<double> g_fd(atoms.size() * 3, 0.0);
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      for (int d = 0; d < 3; ++d) {
        auto ap = atoms, am = atoms;
        ap[a].center[d] += h;
        am[a].center[d] -= h;
        g_fd[3 * a + d] = (eval_uhf_e2(ap) - eval_uhf_e2(am)) / (2.0 * h);
      }
    }
    check_vec_near("indef_dm.uhf.host_vs_fd", g_h, g_fd, 1e-7);

    if (gpu4dft::backend_available(gpu4dft::Backend::Cuda)) {
      gpu4dft::RIJKEngine ri_c(basis, aux, gpu4dft::Backend::Cuda, 0.0);
      const auto g_c = ri_c.compute_ri_jk_grad(atoms, dm_a.data(), nullptr, -1, dm_b.data(),
                                                nullptr, -1, true, 1.0, 1.0, 1.0, 0.0,
                                                gpu4dft::Backend::Cuda);
      check_vec_near("indef_dm.uhf.cuda_vs_fd", g_c, g_fd, 1e-7);
      check_vec_near("indef_dm.uhf.cuda_vs_host", g_c, g_h, 1e-11);
    }
  }

  // 3. run_grad respects JKMode::RI when aux_basis is supplied and throws without it
  {
    gpu4dft::WorkflowOptions opts;
    opts.jk_mode = gpu4dft::JKMode::RI;
    opts.xc = "HF";
    opts.scf.backend = gpu4dft::Backend::Host;
    opts.scf.conv_tol = 1e-10;
    gpu4dft::MolecularScf mol(basis, atoms, 10, opts, aux);
    const auto wres = mol.run();
    const auto g_mol = mol.compute_gradient(wres.scf);
    const auto g_run = gpu4dft::run_grad(basis, atoms, wres.scf, opts, {}, {}, &aux);
    check_vec_near("run_grad.ri_mode_matches_mscf", g_run.total_grad, g_mol.total_grad, 1e-11);

    bool threw = false;
    try {
      (void)gpu4dft::run_grad(basis, atoms, wres.scf, opts);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check_true("run_grad.ri_without_aux_throws", threw);
  }
}

}  // namespace

int main() {
  test_no_eager_device_b_and_lazy_builder();
  test_mo_vs_dm_and_fallbacks();
  test_scf_and_gradients();
  test_indefinite_dm_ri_jk_grad_and_run_grad();
  if (g_failures == 0) {
    std::printf("ALL test_ri_jk_grad checks PASSED\n");
    return 0;
  }
  std::printf("FAILED %d checks in test_ri_jk_grad\n", g_failures);
  return 1;
}
