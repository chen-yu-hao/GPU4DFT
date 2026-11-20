#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/boys.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/jk_engine.hpp"
#include "gpu4dft/rys.hpp"
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

gpu4dft::Basis make_aux_basis(const std::vector<gpu4dft::Atom>& atoms) {
  std::vector<gpu4dft::Shell> shells;
  for (const auto& at : atoms) {
    if (at.atomic_number == 1) {
      shells.push_back(gpu4dft::Shell{0, at.center, {8.0, 2.0, 0.5}, {0.3, 0.5, 0.3}});
      shells.push_back(gpu4dft::Shell{1, at.center, {1.5, 0.4}, {0.6, 0.5}});
    } else if (at.atomic_number == 8) {
      shells.push_back(gpu4dft::Shell{0, at.center, {80.0, 18.0, 4.5, 1.1}, {0.2, 0.4, 0.4, 0.2}});
      shells.push_back(gpu4dft::Shell{1, at.center, {12.0, 3.0, 0.8}, {0.3, 0.5, 0.3}});
      shells.push_back(gpu4dft::Shell{2, at.center, {4.0, 1.0}, {0.5, 0.6}});
    }
  }
  return gpu4dft::Basis(std::move(shells));
}

gpu4dft::Basis make_high_l_test_basis() {
  std::vector<gpu4dft::Shell> shells = {
      gpu4dft::Shell{0, {0.0, 0.0, 0.0}, {1.25}, {1.0}},
      gpu4dft::Shell{1, {0.9, -0.3, 0.4}, {0.85}, {1.0}},
      gpu4dft::Shell{2, {-0.5, 0.7, -0.2}, {0.65}, {1.0}},
      gpu4dft::Shell{3, {0.2, 0.4, 0.8}, {0.55}, {1.0}},
      gpu4dft::Shell{4, {-0.4, -0.6, 0.3}, {0.45}, {1.0}},
  };
  return gpu4dft::Basis(std::move(shells));
}

std::vector<double> make_sym_dm(int nao, double scale = 0.08) {
  const std::size_t n = static_cast<std::size_t>(nao);
  std::vector<double> dm(n * n, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      dm[i * n + j] = scale * std::exp(-0.25 * std::abs(i - j));
    }
  }
  return dm;
}

std::vector<double> make_nonsym_dm(int nao, double scale = 0.08, double asym = 0.25) {
  const std::size_t n = static_cast<std::size_t>(nao);
  std::vector<double> dm(n * n, 0.0);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      const double sym_part = scale * std::exp(-0.25 * std::abs(i - j));
      const double skew = (i < j ? 1.0 : (i > j ? -0.4 : 0.0)) * asym *
                          std::sin(0.7 * (i + 1) - 1.3 * (j + 1));
      dm[i * n + j] = sym_part + skew;
    }
  }
  return dm;
}

double trace_prod(int nao, const std::vector<double>& a, const std::vector<double>& b) {
  double tr = 0.0;
  for (std::size_t i = 0; i < static_cast<std::size_t>(nao) * nao; ++i) tr += a[i] * b[i];
  return tr;
}

void test_boys_rys_erf() {
  // Overflow safety for omega_theta
  check_near("theta.omega0", gpu4dft::omega_theta(1.5, 0.0), 1.0, 1e-15);
  check_true("theta.tiny_omega", gpu4dft::omega_theta(1.5, 1e-8) < 1e-15);
  check_near("theta.huge_omega", gpu4dft::omega_theta(1.0, 1e100), 1.0, 1e-15);
  const double tiny_theta = gpu4dft::omega_theta(1e100, 1.0);
  check_true("theta.huge_rho_finite",
             std::isfinite(tiny_theta) && tiny_theta > 0.0 && tiny_theta < 1e-90);

  const double rho = 1.75;
  const double omega = 0.33;
  const double theta = gpu4dft::omega_theta(rho, omega);
  const double sqrt_theta = std::sqrt(theta);

  for (double T : {0.0, 1e-5, 0.5, 4.2, 45.0}) {
    const double f0_erf = gpu4dft::boys_f0_erf(T, rho, omega);
    const double f0_ref = sqrt_theta * gpu4dft::boys_f0(theta * T);
    check_near("boys_f0_erf", f0_erf, f0_ref, 1e-14);

    double f_erf[10]{};
    double f_unatt[10]{};
    gpu4dft::boys_vec_erf(9, T, rho, omega, f_erf);
    gpu4dft::boys_vec(9, theta * T, f_unatt);
    double scale = sqrt_theta;
    for (int m = 0; m <= 9; ++m) {
      check_near("boys_vec_erf", f_erf[m], scale * f_unatt[m], 1e-13);
      scale *= theta;
    }

    double u_erf[5]{}, w_erf[5]{};
    gpu4dft::rys_roots_weights_erf(5, T, rho, omega, u_erf, w_erf);
    for (int n = 0; n < 10; ++n) {
      double quad = 0.0;
      for (int r = 0; r < 5; ++r) {
        double p = 1.0;
        for (int k = 0; k < n; ++k) p *= u_erf[r];
        quad += w_erf[r] * p;
      }
      check_near("rys_erf_moment", quad, f_erf[n], 1e-12);
    }
  }

  // Unattenuated limit: large omega -> unattenuated Boys/Rys
  double u_lim[4]{}, w_lim[4]{}, u_ref[4]{}, w_ref[4]{};
  gpu4dft::rys_roots_weights_erf(4, 2.3, 1.2, 1e8, u_lim, w_lim);
  gpu4dft::rys_roots_weights(4, 2.3, u_ref, w_ref);
  for (int r = 0; r < 4; ++r) {
    check_near("rys_erf_inf_u", u_lim[r], u_ref[r], 1e-12);
    check_near("rys_erf_inf_w", w_lim[r], w_ref[r], 1e-12);
  }
}

void test_eri_omega_high_l(bool have_cuda) {
  const auto basis = make_high_l_test_basis();
  const int nao = basis.nao();
  const double omega = 0.33;

  const auto eri_coul = gpu4dft::eri_tensor(basis, gpu4dft::Backend::Host, gpu4dft::Precision::FP64, 0.0);
  const auto eri_omega_h =
      gpu4dft::eri_tensor(basis, gpu4dft::Backend::Host, gpu4dft::Precision::FP64, omega);
  const auto eri_omega_inf =
      gpu4dft::eri_tensor(basis, gpu4dft::Backend::Host, gpu4dft::Precision::FP64, 1e6);
  const auto eri_omega_zeroish =
      gpu4dft::eri_tensor(basis, gpu4dft::Backend::Host, gpu4dft::Precision::FP64, 1e-8);

  const std::size_t n = static_cast<std::size_t>(nao);
  auto idx = [&](int i, int j, int k, int l) -> std::size_t {
    return (((static_cast<std::size_t>(i) * n + j) * n + k) * n + l);
  };
  double norm_omega = 0.0;
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      for (int k = 0; k < nao; ++k) {
        for (int l = 0; l < nao; ++l) {
          const double v = eri_omega_h[idx(i, j, k, l)];
          norm_omega += std::fabs(v);
          check_near("eri_omega.sym_ij", v, eri_omega_h[idx(j, i, k, l)], 1e-12);
          check_near("eri_omega.sym_kl", v, eri_omega_h[idx(i, j, l, k)], 1e-12);
          check_near("eri_omega.sym_braket", v, eri_omega_h[idx(k, l, i, j)], 1e-12);
        }
      }
    }
  }
  check_true("eri_omega.nonzero", norm_omega > 1.0);
  check_vec_near("eri_omega.inf_limit", eri_omega_inf, eri_coul, 1e-7);
  check_vec_near("eri_omega.zero_limit", eri_omega_zeroish, std::vector<double>(eri_coul.size(), 0.0), 1e-7);

  if (have_cuda) {
    const auto eri_omega_d =
        gpu4dft::eri_tensor(basis, gpu4dft::Backend::Cuda, gpu4dft::Precision::FP64, omega);
    check_vec_near("eri_omega.host_vs_cuda_l4", eri_omega_d, eri_omega_h, 1e-11);
  }
}

void test_direct_and_ri_jk_rsh(bool have_cuda) {
  const std::vector<gpu4dft::Atom> atoms = {
      {8, {0.0, 0.0, 0.0}},
      {1, {0.0, 1.43, -1.11}},
      {1, {0.0, -1.43, -1.11}},
  };
  const auto basis = make_sto3g_basis(atoms);
  const auto aux_basis = make_aux_basis(atoms);
  const auto hl_basis = make_high_l_test_basis();
  const double omega = 0.33;

  // Verify DirectJKEngine pure omega + 2-pass RSH (including k_sr == 0 and |k_eff| > 1) on H2O and high-l
  for (const auto& pair : std::vector<std::pair<std::string, gpu4dft::Basis>>{{"h2o", basis},
                                                                             {"high_l", hl_basis}}) {
    const auto& b = pair.second;
    const int nao = b.nao();
    const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
    const auto dm = make_sym_dm(nao, 0.07);
    gpu4dft::DirectJKEngine djk(b);

    auto run_djk = [&](gpu4dft::Backend be, bool rsh_mode, double w, double k_sr, double k_lr) {
      gpu4dft::JKOptions opts;
      opts.backend = be;
      opts.precision = gpu4dft::Precision::FP64;
      opts.schwarz_cutoff = 1e-14;
      opts.omega = w;
      opts.rsh_mode = rsh_mode;
      opts.k_sr = k_sr;
      opts.k_lr = k_lr;
      std::vector<double> vj(n2, 0.0), vk(n2, 0.0);
      djk.compute_jk(dm.data(), vj.data(), vk.data(), opts);
      return std::make_pair(vj, vk);
    };

    const auto jk0_h = run_djk(gpu4dft::Backend::Host, false, 0.0, 0.0, 0.0);
    const auto jkw_h = run_djk(gpu4dft::Backend::Host, false, omega, 0.0, 0.0);
    const auto rsh_h = run_djk(gpu4dft::Backend::Host, true, omega, 0.19, 0.65);

    std::vector<double> k_expected(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) {
      k_expected[i] = 0.19 * jk0_h.second[i] + (0.65 - 0.19) * jkw_h.second[i];
    }
    check_vec_near((pair.first + ".direct_rsh.j_eq").c_str(), rsh_h.first, jk0_h.first, 1e-12);
    check_vec_near((pair.first + ".direct_rsh.k_eq").c_str(), rsh_h.second, k_expected, 1e-12);

    // k_sr = 0 and |k_eff| > 1.0
    const auto rsh_csr0_h = run_djk(gpu4dft::Backend::Host, true, omega, 0.0, 1.35);
    std::vector<double> k_csr0_exp(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) k_csr0_exp[i] = 1.35 * jkw_h.second[i];
    check_vec_near((pair.first + ".direct_rsh.csr0_bigk").c_str(), rsh_csr0_h.second, k_csr0_exp, 1e-12);

    if (have_cuda) {
      const auto jkw_c = run_djk(gpu4dft::Backend::Cuda, false, omega, 0.0, 0.0);
      const auto rsh_c = run_djk(gpu4dft::Backend::Cuda, true, omega, 0.19, 0.65);
      const auto rsh_csr0_c = run_djk(gpu4dft::Backend::Cuda, true, omega, 0.0, 1.35);
      check_vec_near((pair.first + ".direct_omega.cuda_j").c_str(), jkw_c.first, jkw_h.first, 1e-11);
      check_vec_near((pair.first + ".direct_omega.cuda_k").c_str(), jkw_c.second, jkw_h.second, 1e-11);
      check_vec_near((pair.first + ".direct_rsh.cuda_j").c_str(), rsh_c.first, rsh_h.first, 1e-11);
      check_vec_near((pair.first + ".direct_rsh.cuda_k").c_str(), rsh_c.second, rsh_h.second, 1e-11);
      check_vec_near((pair.first + ".direct_rsh.cuda_csr0").c_str(), rsh_csr0_c.second, rsh_csr0_h.second, 1e-11);
    }

    // UHF spin-flip test with RSH
    std::vector<double> neg_dm(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) neg_dm[i] = -dm[i];
    auto run_uhf = [&](gpu4dft::Backend be) {
      gpu4dft::JKOptions opts;
      opts.backend = be;
      opts.precision = gpu4dft::Precision::FP64;
      opts.schwarz_cutoff = 1e-14;
      opts.omega = omega;
      opts.rsh_mode = true;
      opts.k_sr = 0.19;
      opts.k_lr = 0.65;
      std::vector<double> vj(n2, 0.0), vka(n2, 0.0), vkb(n2, 0.0);
      djk.compute_jk_uhf(dm.data(), neg_dm.data(), vj.data(), vka.data(), vkb.data(), opts);
      return std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>(vj, vka, vkb);
    };
    const auto [uvj_h, uvka_h, uvkb_h] = run_uhf(gpu4dft::Backend::Host);
    check_vec_near((pair.first + ".uhf_rsh.j_zero").c_str(), uvj_h, std::vector<double>(n2, 0.0), 1e-12);
    check_vec_near((pair.first + ".uhf_rsh.ka_eq").c_str(), uvka_h, k_expected, 1e-12);
    std::vector<double> neg_k_exp(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) neg_k_exp[i] = -k_expected[i];
    check_vec_near((pair.first + ".uhf_rsh.kb_eq").c_str(), uvkb_h, neg_k_exp, 1e-12);
    if (have_cuda) {
      const auto [uvj_c, uvka_c, uvkb_c] = run_uhf(gpu4dft::Backend::Cuda);
      check_vec_near((pair.first + ".uhf_rsh.cuda_j").c_str(), uvj_c, uvj_h, 1e-11);
      check_vec_near((pair.first + ".uhf_rsh.cuda_ka").c_str(), uvka_c, uvka_h, 1e-11);
      check_vec_near((pair.first + ".uhf_rsh.cuda_kb").c_str(), uvkb_c, uvkb_h, 1e-11);
    }
  }

  // RIJKEngine dual-cache RSH test (Host vs CUDA)
  {
    const int nao = basis.nao();
    const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
    const auto dm = make_sym_dm(nao, 0.06);
    gpu4dft::RIJKEngine ri_h(basis, aux_basis, gpu4dft::Backend::Host, omega);
    check_near("rijk.rsh_omega", ri_h.rsh_omega(), omega, 1e-15);

    gpu4dft::JKOptions ropts;
    ropts.backend = gpu4dft::Backend::Host;
    ropts.precision = gpu4dft::Precision::FP64;
    ropts.omega = omega;
    ropts.rsh_mode = true;
    ropts.k_sr = 0.19;
    ropts.k_lr = 0.65;
    std::vector<double> vj_h(n2, 0.0), vk_h(n2, 0.0);
    ri_h.compute_jk(dm.data(), vj_h.data(), vk_h.data(), ropts);

    gpu4dft::JKOptions wopts = ropts;
    wopts.rsh_mode = false;
    std::vector<double> vj_w_h(n2, 0.0), vk_w_h(n2, 0.0);
    ri_h.compute_jk(dm.data(), vj_w_h.data(), vk_w_h.data(), wopts);

    gpu4dft::JKOptions c0pts = ropts;
    c0pts.rsh_mode = false;
    c0pts.omega = 0.0;
    std::vector<double> vj_0_h(n2, 0.0), vk_0_h(n2, 0.0);
    ri_h.compute_jk(dm.data(), vj_0_h.data(), vk_0_h.data(), c0pts);

    std::vector<double> k_exp(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) k_exp[i] = 0.19 * vk_0_h[i] + 0.46 * vk_w_h[i];
    check_vec_near("rijk_rsh.j_eq", vj_h, vj_0_h, 1e-12);
    check_vec_near("rijk_rsh.k_eq", vk_h, k_exp, 1e-12);

    if (have_cuda) {
      gpu4dft::RIJKEngine ri_c(basis, aux_basis, gpu4dft::Backend::Cuda, omega);
      check_vec_near("rijk.b_host_vs_cuda", ri_c.b_tensor(), ri_h.b_tensor(), 1e-11);
      check_vec_near("rijk.b_omega_host_vs_cuda", ri_c.b_omega_tensor(), ri_h.b_omega_tensor(), 1e-11);
      ropts.backend = gpu4dft::Backend::Cuda;
      std::vector<double> vj_c(n2, 0.0), vk_c(n2, 0.0);
      ri_c.compute_jk(dm.data(), vj_c.data(), vk_c.data(), ropts);
      check_vec_near("rijk_rsh.cuda_j", vj_c, vj_h, 1e-11);
      check_vec_near("rijk_rsh.cuda_k", vk_c, vk_h, 1e-11);
    }
  }
}

void test_direct_jk_nonsym_rsh_zero_ksr(bool have_cuda) {
  const std::vector<gpu4dft::Atom> atoms = {
      {8, {0.0, 0.0, 0.0}},
      {1, {0.0, 1.43, -1.11}},
      {1, {0.0, -1.43, -1.11}},
  };
  const auto sp_basis = make_sto3g_basis(atoms);
  const auto hl_basis = make_high_l_test_basis();

  for (const auto& pair : std::vector<std::pair<std::string, gpu4dft::Basis>>{{"sp", sp_basis},
                                                                             {"high_l", hl_basis}}) {
    const auto& b = pair.second;
    const int nao = b.nao();
    const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
    const auto dm = make_nonsym_dm(nao, 0.07, 0.28);
    const auto da = make_nonsym_dm(nao, 0.05, 0.22);
    const auto db = make_nonsym_dm(nao, 0.03, -0.17);
    gpu4dft::DirectJKEngine djk(b);

    // Reference pure-J with compute_k=false (always exact for J)
    std::vector<double> vj_pure_ref(n2, 0.0);
    {
      gpu4dft::JKOptions jopts;
      jopts.backend = gpu4dft::Backend::Host;
      jopts.precision = gpu4dft::Precision::FP64;
      jopts.schwarz_cutoff = 1e-14;
      jopts.compute_j = true;
      jopts.compute_k = false;
      jopts.symmetrize_dm = false;
      djk.compute_jk(dm.data(), vj_pure_ref.data(), nullptr, jopts);
    }

    std::vector<double> dtot(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) dtot[i] = da[i] + db[i];
    std::vector<double> vj_uhf_pure_ref(n2, 0.0);
    {
      gpu4dft::JKOptions jopts;
      jopts.backend = gpu4dft::Backend::Host;
      jopts.precision = gpu4dft::Precision::FP64;
      jopts.schwarz_cutoff = 1e-14;
      jopts.compute_j = true;
      jopts.compute_k = false;
      jopts.symmetrize_dm = false;
      djk.compute_jk(dtot.data(), vj_uhf_pure_ref.data(), nullptr, jopts);
    }

    struct Case {
      const char* name;
      double omega;
      double k_sr;
      double k_lr;
      double k_scale;
    };
    const std::vector<Case> cases = {
        {"pure_lr", 0.4, 0.0, 1.0, 1.0},
        {"k_scale_zero", 0.33, 0.19, 0.65, 0.0},
    };

    for (const auto& c : cases) {
      auto make_opts = [&](gpu4dft::Backend be, gpu4dft::Precision prec) {
        gpu4dft::JKOptions opts;
        opts.backend = be;
        opts.precision = prec;
        opts.schwarz_cutoff = 1e-14;
        opts.omega = c.omega;
        opts.rsh_mode = true;
        opts.k_sr = c.k_sr;
        opts.k_lr = c.k_lr;
        opts.k_scale = c.k_scale;
        opts.symmetrize_dm = false;
        opts.compute_j = true;
        opts.compute_k = true;
        return opts;
      };

      std::vector<double> vj_h(n2, 99.0), vk_h(n2, 99.0);
      djk.compute_jk(dm.data(), vj_h.data(), vk_h.data(),
                     make_opts(gpu4dft::Backend::Host, gpu4dft::Precision::FP64));
      check_vec_near((pair.first + "." + c.name + ".rhf.host_j_eq_pure").c_str(), vj_h,
                     vj_pure_ref, 1e-12);

      std::vector<double> uvj_h(n2, 99.0), uvka_h(n2, 99.0), uvkb_h(n2, 99.0);
      djk.compute_jk_uhf(da.data(), db.data(), uvj_h.data(), uvka_h.data(), uvkb_h.data(),
                         make_opts(gpu4dft::Backend::Host, gpu4dft::Precision::FP64));
      check_vec_near((pair.first + "." + c.name + ".uhf.host_j_eq_pure").c_str(), uvj_h,
                     vj_uhf_pure_ref, 1e-12);

      if (c.k_scale == 0.0) {
        const std::vector<double> zeros(n2, 0.0);
        check_vec_near((pair.first + "." + c.name + ".rhf.host_k_zero").c_str(), vk_h, zeros, 1e-14);
        check_vec_near((pair.first + "." + c.name + ".uhf.host_ka_zero").c_str(), uvka_h, zeros, 1e-14);
        check_vec_near((pair.first + "." + c.name + ".uhf.host_kb_zero").c_str(), uvkb_h, zeros, 1e-14);
      }

      if (have_cuda) {
        const std::vector<std::tuple<const char*, gpu4dft::Precision, double>> precs = {
            {"fp64", gpu4dft::Precision::FP64, 1e-11},
            {"mixed", gpu4dft::Precision::Mixed, 1e-8},
            {"fp32", gpu4dft::Precision::FP32, 1e-4},
        };
        for (const auto& [pname, prec, tol] : precs) {
          std::vector<double> vj_c(n2, 99.0), vk_c(n2, 99.0);
          djk.compute_jk(dm.data(), vj_c.data(), vk_c.data(),
                         make_opts(gpu4dft::Backend::Cuda, prec));
          check_vec_near((pair.first + "." + c.name + ".rhf.cuda_" + pname + "_j").c_str(), vj_c,
                         vj_h, tol);
          check_vec_near((pair.first + "." + c.name + ".rhf.cuda_" + pname + "_k").c_str(), vk_c,
                         vk_h, tol);

          std::vector<double> uvj_c(n2, 99.0), uvka_c(n2, 99.0), uvkb_c(n2, 99.0);
          djk.compute_jk_uhf(da.data(), db.data(), uvj_c.data(), uvka_c.data(), uvkb_c.data(),
                             make_opts(gpu4dft::Backend::Cuda, prec));
          check_vec_near((pair.first + "." + c.name + ".uhf.cuda_" + pname + "_j").c_str(), uvj_c,
                         uvj_h, tol);
          check_vec_near((pair.first + "." + c.name + ".uhf.cuda_" + pname + "_ka").c_str(),
                         uvka_c, uvka_h, tol);
          check_vec_near((pair.first + "." + c.name + ".uhf.cuda_" + pname + "_kb").c_str(),
                         uvkb_c, uvkb_h, tol);
        }
      }
    }
  }
}

void test_jk_nuclear_grad_rsh(bool have_cuda) {
  const std::vector<gpu4dft::Atom> atoms = {
      {8, {0.0, 0.0, 0.0}},
      {1, {0.0, 1.43, -1.11}},
      {1, {0.0, -1.43, -1.11}},
  };
  const auto basis0 = make_sto3g_basis(atoms);
  const int nao = basis0.nao();
  const auto dm = make_sym_dm(nao, 0.05);
  const auto da = make_sym_dm(nao, 0.03);
  const auto db = make_sym_dm(nao, 0.02);

  for (const auto& params : std::vector<std::pair<double, double>>{{0.19, 0.65}, {0.0, 1.0}}) {
    const double k_sr = params.first;
    const double k_lr = params.second;
    const double omega = 0.33;

    const auto g_rhf_h = gpu4dft::jk_nuclear_grad_rsh(basis0, atoms, dm.data(), k_sr, k_lr, omega,
                                                      gpu4dft::Backend::Host, 1e-14);
    const auto g_uhf_h = gpu4dft::jk_nuclear_grad_uhf_rsh(
        basis0, atoms, da.data(), db.data(), k_sr, k_lr, omega, gpu4dft::Backend::Host, 1e-14);

    // Translational invariance
    for (int axis = 0; axis < 3; ++axis) {
      check_near("grad_rsh.trans_rhf", g_rhf_h[axis] + g_rhf_h[3 + axis] + g_rhf_h[6 + axis], 0.0, 1e-12);
      check_near("grad_rsh.trans_uhf", g_uhf_h[axis] + g_uhf_h[3 + axis] + g_uhf_h[6 + axis], 0.0, 1e-12);
    }

    // Finite-difference of E_2e(R) at fixed DM
    auto eval_e2e_rhf = [&](double dz_h1) {
      auto at = atoms;
      at[1].center[2] += dz_h1;
      const auto b = make_sto3g_basis(at);
      gpu4dft::DirectJKEngine djk(b);
      const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
      std::vector<double> vj(n2, 0.0), vk(n2, 0.0);
      gpu4dft::JKOptions opts;
      opts.backend = gpu4dft::Backend::Host;
      opts.precision = gpu4dft::Precision::FP64;
      opts.schwarz_cutoff = 1e-14;
      opts.omega = omega;
      opts.rsh_mode = true;
      opts.k_sr = k_sr;
      opts.k_lr = k_lr;
      djk.compute_jk(dm.data(), vj.data(), vk.data(), opts);
      return 0.5 * trace_prod(nao, dm, vj) - 0.25 * trace_prod(nao, dm, vk);
    };
    auto eval_e2e_uhf = [&](double dz_h1) {
      auto at = atoms;
      at[1].center[2] += dz_h1;
      const auto b = make_sto3g_basis(at);
      gpu4dft::DirectJKEngine djk(b);
      const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
      std::vector<double> vj(n2, 0.0), vka(n2, 0.0), vkb(n2, 0.0);
      gpu4dft::JKOptions opts;
      opts.backend = gpu4dft::Backend::Host;
      opts.precision = gpu4dft::Precision::FP64;
      opts.schwarz_cutoff = 1e-14;
      opts.omega = omega;
      opts.rsh_mode = true;
      opts.k_sr = k_sr;
      opts.k_lr = k_lr;
      djk.compute_jk_uhf(da.data(), db.data(), vj.data(), vka.data(), vkb.data(), opts);
      std::vector<double> dtot(n2, 0.0);
      for (std::size_t i = 0; i < n2; ++i) dtot[i] = da[i] + db[i];
      return 0.5 * trace_prod(nao, dtot, vj) -
             0.5 * (trace_prod(nao, da, vka) + trace_prod(nao, db, vkb));
    };

    const double h = 1e-5;
    check_near("grad_rsh.fd_rhf_h1z", g_rhf_h[5], (eval_e2e_rhf(h) - eval_e2e_rhf(-h)) / (2.0 * h), 1e-7);
    check_near("grad_rsh.fd_uhf_h1z", g_uhf_h[5], (eval_e2e_uhf(h) - eval_e2e_uhf(-h)) / (2.0 * h), 1e-7);

    if (have_cuda) {
      const auto g_rhf_c = gpu4dft::jk_nuclear_grad_rsh(basis0, atoms, dm.data(), k_sr, k_lr, omega,
                                                        gpu4dft::Backend::Cuda, 1e-14);
      const auto g_uhf_c = gpu4dft::jk_nuclear_grad_uhf_rsh(
          basis0, atoms, da.data(), db.data(), k_sr, k_lr, omega, gpu4dft::Backend::Cuda, 1e-14);
      check_vec_near("grad_rsh.rhf_host_vs_cuda", g_rhf_c, g_rhf_h, 1e-11);
      check_vec_near("grad_rsh.uhf_host_vs_cuda", g_uhf_c, g_uhf_h, 1e-11);
    }
  }
}

void test_molecular_scf_rsh(bool have_cuda) {
  const std::vector<gpu4dft::Atom> h2o_atoms = {
      {8, {0.0, 0.0, 0.0}},
      {1, {0.0, 1.43, -1.11}},
      {1, {0.0, -1.43, -1.11}},
  };
  const auto h2o_basis = make_sto3g_basis(h2o_atoms);
  const auto h2o_aux = make_aux_basis(h2o_atoms);

  // Guard checks: SnK / RISnK with RSH throws NotImplemented
  for (auto mode : {gpu4dft::JKMode::SnK, gpu4dft::JKMode::RISnK}) {
    gpu4dft::WorkflowOptions opts;
    opts.xc = "cam-b3lyp";
    opts.jk_mode = mode;
    opts.scf.backend = gpu4dft::Backend::Host;
    std::optional<gpu4dft::Basis> aux = (mode == gpu4dft::JKMode::RISnK)
                                            ? std::optional<gpu4dft::Basis>(h2o_aux)
                                            : std::nullopt;
    bool caught = false;
    try {
      gpu4dft::MolecularScf wf(h2o_basis, h2o_atoms, 10, opts, aux);
      wf.setup();
    } catch (const gpu4dft::NotImplemented&) {
      caught = true;
    }
    check_true("guard.snk_rsh_not_impl", caught);
  }

  for (const std::string& xc : {"cam-b3lyp", "wb97x"}) {
    gpu4dft::WorkflowOptions opts;
    opts.xc = xc;
    opts.jk_mode = gpu4dft::JKMode::Direct;
    opts.grid.level = 3;
    opts.scf.backend = gpu4dft::Backend::Host;
    opts.scf.conv_tol = 1e-10;
    opts.scf.diis_tol = 1e-8;

    gpu4dft::MolecularScf wf_h(h2o_basis, h2o_atoms, 10, opts);
    const auto res_h = wf_h.run();
    check_true((xc + ".rks_host.converged").c_str(), res_h.scf.converged);
    check_true((xc + ".rsh_flag").c_str(), wf_h.rsh_coefficients().is_rsh());

    // run_response guard for RSH
    bool resp_caught = false;
    try {
      (void)wf_h.run_response();
    } catch (const gpu4dft::NotImplemented&) {
      resp_caught = true;
    }
    check_true((xc + ".guard.response_not_impl").c_str(), resp_caught);

    // Analytical gradient vs SCF FD
    const auto g0 = wf_h.compute_gradient(res_h.scf);
    const double h = 1e-4;
    auto eval_e = [&](double dz_h1) {
      auto at = h2o_atoms;
      at[1].center[2] += dz_h1;
      gpu4dft::MolecularScf wf_tmp(make_sto3g_basis(at), at, 10, opts);
      return wf_tmp.run().scf.total_energy;
    };
    const double g_fd = (eval_e(h) - eval_e(-h)) / (2.0 * h);
    check_near((xc + ".rks.grad_fd_h1z").c_str(), g0.total_grad[5], g_fd, 1e-5);

    if (have_cuda) {
      auto cuda_opts = opts;
      cuda_opts.scf.backend = gpu4dft::Backend::Cuda;
      gpu4dft::MolecularScf wf_c(h2o_basis, h2o_atoms, 10, cuda_opts);
      const auto res_c = wf_c.run();
      check_true((xc + ".rks_cuda.converged").c_str(), res_c.scf.converged);
      check_true((xc + ".rks_cuda.zero_alloc").c_str(), res_c.scf.steady_state_allocations == 0);
      check_near((xc + ".rks.host_vs_cuda").c_str(), res_c.scf.total_energy, res_h.scf.total_energy, 1e-8);

      auto ri_opts = cuda_opts;
      ri_opts.jk_mode = gpu4dft::JKMode::RI;
      gpu4dft::MolecularScf wf_ri_c(h2o_basis, h2o_atoms, 10, ri_opts, h2o_aux);
      const auto res_ri_c = wf_ri_c.run();
      check_true((xc + ".ri_cuda.converged").c_str(), res_ri_c.scf.converged);
      check_true((xc + ".ri_cuda.zero_alloc").c_str(), res_ri_c.scf.steady_state_allocations == 0);

      auto ri_h_opts = ri_opts;
      ri_h_opts.scf.backend = gpu4dft::Backend::Host;
      gpu4dft::MolecularScf wf_ri_h(h2o_basis, h2o_atoms, 10, ri_h_opts, h2o_aux);
      const auto res_ri_h = wf_ri_h.run();
      check_near((xc + ".ri.host_vs_cuda").c_str(), res_ri_c.scf.total_energy,
                 res_ri_h.scf.total_energy, 1e-8);
    }
  }

  // Open-shell UKS doublet check (H2O+ radical cation)
  const auto oh_basis = make_sto3g_basis(h2o_atoms);
  gpu4dft::WorkflowOptions uopts;
  uopts.xc = "cam-b3lyp";
  uopts.unrestricted = true;
  uopts.spin = 1;
  uopts.jk_mode = gpu4dft::JKMode::Direct;
  uopts.grid.level = 3;
  uopts.scf.backend = gpu4dft::Backend::Host;
  uopts.scf.conv_tol = 1e-11;
  uopts.scf.diis_tol = 1e-9;
  uopts.scf.dynamic_precision = false;
  gpu4dft::MolecularScf wf_uh(oh_basis, h2o_atoms, 9, uopts);
  const auto res_uh = wf_uh.run();
  check_true("h2op.uks_cam_b3lyp.converged", res_uh.scf.converged);
  if (have_cuda) {
    uopts.scf.backend = gpu4dft::Backend::Cuda;
    gpu4dft::MolecularScf wf_uc(oh_basis, h2o_atoms, 9, uopts);
    const auto res_uc = wf_uc.run();
    check_true("h2op.uks_cam_b3lyp.cuda_converged", res_uc.scf.converged);
    check_true("h2op.uks_cam_b3lyp.zero_alloc", res_uc.scf.steady_state_allocations == 0);
    check_near("h2op.uks_cam_b3lyp.host_vs_cuda", res_uc.scf.total_energy, res_uh.scf.total_energy, 1e-9);
    check_near("h2op.uks_cam_b3lyp.s2_host_vs_cuda", res_uc.scf.s2, res_uh.scf.s2, 1e-8);
  }
}

}  // namespace

int main() {
  const bool have_cuda = gpu4dft::backend_available(gpu4dft::Backend::Cuda);
  test_boys_rys_erf();
  test_eri_omega_high_l(have_cuda);
  test_direct_and_ri_jk_rsh(have_cuda);
  test_direct_jk_nonsym_rsh_zero_ksr(have_cuda);
  test_jk_nuclear_grad_rsh(have_cuda);
  test_molecular_scf_rsh(have_cuda);

  if (g_failures != 0) {
    std::printf("%d failures in test_rsh\n", g_failures);
    return 1;
  }
  std::printf("test_rsh: ALL PASSED\n");
  return 0;
}
