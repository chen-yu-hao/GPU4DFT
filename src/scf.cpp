#include "gpu4dft/scf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GPU4DFT_C_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define GPU4DFT_C_EXPORT extern "C"
#endif

namespace gpu4dft {

struct ScfDriver::Impl {
  ScfOptions opts;
  DeviceMemoryPool* ext_pool = nullptr;

  Impl(ScfOptions o, DeviceMemoryPool* p) : opts(std::move(o)), ext_pool(p) {}
};

ScfDriver::ScfDriver(ScfOptions opts, DeviceMemoryPool* pool)
    : impl_(std::make_unique<Impl>(std::move(opts), pool)) {}

ScfDriver::~ScfDriver() = default;
ScfDriver::ScfDriver(ScfDriver&&) noexcept = default;
ScfDriver& ScfDriver::operator=(ScfDriver&&) noexcept = default;

const ScfOptions& ScfDriver::options() const noexcept { return impl_->opts; }
void ScfDriver::set_options(ScfOptions opts) { impl_->opts = std::move(opts); }

namespace {

double dot_host(std::size_t n, const double* a, const double* b) {
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

void symmetrize_host_square(int n, std::vector<double>& A) {
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const double avg =
          0.5 * (A[static_cast<std::size_t>(i) * n + j] + A[static_cast<std::size_t>(j) * n + i]);
      A[static_cast<std::size_t>(i) * n + j] = avg;
      A[static_cast<std::size_t>(j) * n + i] = avg;
    }
  }
}

ScfResult run_scf_host(const ScfOptions& opts, int nao, int n_occ, double e_nuc,
                       const double* S_host, const double* H_core_host,
                       const FockCallbacks& callbacks, const double* initial_dm_host) {
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  const CanonicalOrthoResult ortho =
      canonical_orthogonalization(nao, S_host, opts.lindep_tol, Backend::Host);
  const int nmo = ortho.nmo;
  if (n_occ > nmo) {
    throw std::invalid_argument("ScfDriver::run: n_occ exceeds linearly independent basis size nmo");
  }

  ScfResult res;
  res.nao = nao;
  res.nmo = nmo;
  res.nuclear_repulsion = e_nuc;
  res.density_matrix.assign(n2, 0.0);
  res.energy_weighted_dm.assign(n2, 0.0);
  res.fock_matrix.assign(n2, 0.0);

  if (initial_dm_host != nullptr) {
    std::copy(initial_dm_host, initial_dm_host + n2, res.density_matrix.begin());
  } else if (opts.solver == DensitySolver::Diag) {
    auto init_diag = solve_fock_diag(nao, nmo, n_occ, H_core_host, ortho.X.data(), Backend::Host,
                                     opts.gemm_mode, opts.ozaki_slices);
    res.density_matrix = std::move(init_diag.dm);
  } else {
    Sp2Options sp2_opts = opts.sp2_options;
    sp2_opts.gemm_mode = opts.gemm_mode;
    sp2_opts.ozaki_slices = opts.ozaki_slices;
    auto init_sp2 =
        solve_fock_sp2(nao, nmo, n_occ, H_core_host, ortho.X.data(), Backend::Host, sp2_opts);
    res.density_matrix = std::move(init_sp2.dm);
  }

  DiisOptions diis_opts;
  diis_opts.max_vec = opts.diis_max_vec;
  diis_opts.damping = opts.damping;
  diis_opts.damping_steps = opts.damping_steps;
  diis_opts.level_shift = opts.level_shift;
  diis_opts.use_cuda_graph = false;
  diis_opts.gemm_mode = opts.gemm_mode;
  GpuDiis diis(nao, nmo, diis_opts);

  const bool has_separate_jk = static_cast<bool>(callbacks.build_jk);
  const bool has_vhf = static_cast<bool>(callbacks.build_vhf);
  if (!has_separate_jk && !has_vhf) {
    throw std::invalid_argument(
        "ScfDriver::run: FockCallbacks must provide either build_jk or build_vhf");
  }
  const bool need_k = (opts.exact_exchange_fraction != 0.0);
  const double eff_diis_tol = std::min(opts.diis_tol, std::max(1e-10, opts.conv_tol * 100.0));

  std::vector<double> D_ref(n2, 0.0);
  std::vector<double> delta_D(n2, 0.0);
  std::vector<double> J_acc(n2, 0.0), K_acc(n2, 0.0), Vhf_acc(n2, 0.0);
  std::vector<double> dJ(n2, 0.0), dK(n2, 0.0), dVhf(n2, 0.0);
  std::vector<double> Vxc(n2, 0.0);

  bool in_final_fp64 =
      !opts.dynamic_precision ||
      (opts.early_integral_precision == Precision::FP64 && opts.early_gemm_mode == opts.gemm_mode);
  Precision cur_int_prec = in_final_fp64 ? Precision::FP64 : opts.early_integral_precision;
  GemmMode cur_gemm = in_final_fp64 ? opts.gemm_mode : opts.early_gemm_mode;

  bool force_full_reset = true;
  bool verified_full_build = false;
  int steps_since_reset = 0;
  double prev_energy = 0.0;

  for (int iter = 0; iter < opts.max_iter; ++iter) {
    const bool do_inc = opts.incremental_fock && !force_full_reset && iter > 0 &&
                        (steps_since_reset < std::max(1, opts.fock_reset_interval));

    if (do_inc) {
      for (std::size_t i = 0; i < n2; ++i) {
        delta_D[i] = res.density_matrix[i] - D_ref[i];
      }
      if (has_separate_jk) {
        std::fill(dJ.begin(), dJ.end(), 0.0);
        std::fill(dK.begin(), dK.end(), 0.0);
        callbacks.build_jk(delta_D.data(), true, cur_int_prec, dJ.data(),
                           need_k ? dK.data() : nullptr, nullptr);
        for (std::size_t i = 0; i < n2; ++i) {
          J_acc[i] += dJ[i];
          if (need_k) K_acc[i] += dK[i];
        }
      } else {
        std::fill(dVhf.begin(), dVhf.end(), 0.0);
        callbacks.build_vhf(delta_D.data(), true, cur_int_prec, dVhf.data(), nullptr);
        for (std::size_t i = 0; i < n2; ++i) Vhf_acc[i] += dVhf[i];
      }
      D_ref = res.density_matrix;
      steps_since_reset += 1;
    } else {
      if (has_separate_jk) {
        std::fill(J_acc.begin(), J_acc.end(), 0.0);
        std::fill(K_acc.begin(), K_acc.end(), 0.0);
        callbacks.build_jk(res.density_matrix.data(), false, cur_int_prec, J_acc.data(),
                           need_k ? K_acc.data() : nullptr, nullptr);
      } else {
        std::fill(Vhf_acc.begin(), Vhf_acc.end(), 0.0);
        callbacks.build_vhf(res.density_matrix.data(), false, cur_int_prec, Vhf_acc.data(), nullptr);
      }
      D_ref = res.density_matrix;
      force_full_reset = false;
      steps_since_reset = 0;
    }

    double e_xc = 0.0;
    if (callbacks.build_vxc) {
      std::fill(Vxc.begin(), Vxc.end(), 0.0);
      e_xc = callbacks.build_vxc(res.density_matrix.data(), cur_int_prec, Vxc.data(), nullptr);
    }

    for (std::size_t i = 0; i < n2; ++i) {
      const double v2e = has_separate_jk ? (J_acc[i] - 0.5 * opts.exact_exchange_fraction * K_acc[i])
                                         : Vhf_acc[i];
      res.fock_matrix[i] = H_core_host[i] + v2e + Vxc[i];
    }
    symmetrize_host_square(nao, res.fock_matrix);

    const double e_1e = dot_host(n2, res.density_matrix.data(), H_core_host);
    const double e_2e =
        has_separate_jk
            ? (0.5 * dot_host(n2, res.density_matrix.data(), J_acc.data()) -
               0.25 * opts.exact_exchange_fraction *
                   dot_host(n2, res.density_matrix.data(), K_acc.data()))
            : (0.5 * dot_host(n2, res.density_matrix.data(), Vhf_acc.data()));

    const double e_elec = e_1e + e_2e + e_xc;
    const double e_tot = e_nuc + e_elec;
    const double delta_e = (iter == 0) ? e_tot : (e_tot - prev_energy);
    prev_energy = e_tot;

    diis.set_gemm_mode(cur_gemm);
    const DiisHostResult diis_res =
        diis.update(res.fock_matrix.data(), res.density_matrix.data(), S_host, ortho.X.data(),
                    Backend::Host);
    const double rms_err = diis_res.info.rms_error;

    res.iterations = iter + 1;
    res.total_energy = e_tot;
    res.electronic_energy = e_elec;
    res.one_electron_energy = e_1e;
    res.two_electron_energy = e_2e;
    res.xc_energy = e_xc;
    res.final_diis_rms = rms_err;
    res.history.push_back(
        ScfIterationLog{iter + 1, e_tot, delta_e, rms_err, do_inc, cur_int_prec, cur_gemm});

    if (!in_final_fp64 &&
        (rms_err < opts.fp64_switch_rms ||
         (iter > 0 && std::abs(delta_e) < std::max(100.0 * opts.conv_tol, 1e-6)))) {
      in_final_fp64 = true;
      if (opts.early_gemm_mode == GemmMode::FP32 ||
          opts.early_integral_precision != Precision::FP64) {
        diis.reset();
      }
      cur_int_prec = Precision::FP64;
      cur_gemm = opts.gemm_mode;
      force_full_reset = true;
    } else if (iter >= 1 && in_final_fp64 && std::abs(delta_e) < opts.conv_tol &&
               rms_err < eff_diis_tol) {
      if (opts.incremental_fock && steps_since_reset >= 4 && !verified_full_build) {
        force_full_reset = true;
        verified_full_build = true;
      } else {
        res.converged = true;
        break;
      }
    }

    if (opts.solver == DensitySolver::Diag) {
      auto diag = solve_fock_diag(nao, nmo, n_occ, diis_res.fock_extrapolated.data(),
                                  ortho.X.data(), Backend::Host, cur_gemm, opts.ozaki_slices);
      res.density_matrix = std::move(diag.dm);
      res.energy_weighted_dm = std::move(diag.energy_dm);
      res.mo_energies = std::move(diag.mo_energies);
      res.mo_coefficients = std::move(diag.mo_coeff);
    } else {
      Sp2Options sp2_opts = opts.sp2_options;
      sp2_opts.gemm_mode = cur_gemm;
      sp2_opts.ozaki_slices = opts.ozaki_slices;
      sp2_opts.use_cuda_graph = false;
      auto sp2 = solve_fock_sp2(nao, nmo, n_occ, diis_res.fock_extrapolated.data(), ortho.X.data(),
                                Backend::Host, sp2_opts);
      res.density_matrix = std::move(sp2.dm);
      res.energy_weighted_dm = std::move(sp2.energy_dm);
    }
  }

  // Re-evaluate final orbital energies & unshifted W from the converged raw Fock matrix
  if (opts.solver == DensitySolver::Diag) {
    auto final_diag = solve_fock_diag(nao, nmo, n_occ, res.fock_matrix.data(), ortho.X.data(),
                                      Backend::Host, opts.gemm_mode, opts.ozaki_slices);
    res.density_matrix = std::move(final_diag.dm);
    res.energy_weighted_dm = std::move(final_diag.energy_dm);
    res.mo_energies = std::move(final_diag.mo_energies);
    res.mo_coefficients = std::move(final_diag.mo_coeff);
  } else {
    std::vector<double> DF(n2, 0.0);
    gemm(nao, nao, nao, 1.0, res.density_matrix.data(), false, res.fock_matrix.data(), false, 0.0,
         DF.data(), Backend::Host, opts.gemm_mode, opts.ozaki_slices);
    gemm(nao, nao, nao, 0.5, DF.data(), false, res.density_matrix.data(), false, 0.0,
         res.energy_weighted_dm.data(), Backend::Host, opts.gemm_mode, opts.ozaki_slices);
    symmetrize_host_square(nao, res.energy_weighted_dm);
  }

  return res;
}

}  // namespace

ScfResult ScfDriver::run(int nao, int n_elec, double e_nuc, const double* S_host,
                         const double* H_core_host, const FockCallbacks& callbacks,
                         const double* initial_dm_host) {
  if (nao <= 0) throw std::invalid_argument("ScfDriver::run: nao must be positive");
  if (n_elec < 0 || (n_elec % 2) != 0) {
    throw std::invalid_argument("ScfDriver::run: closed-shell n_elec must be non-negative and even");
  }
  const int n_occ = n_elec / 2;
  const ScfOptions& opts = impl_->opts;

  if (opts.backend == Backend::Host) {
    return run_scf_host(opts, nao, n_occ, e_nuc, S_host, H_core_host, callbacks, initial_dm_host);
  }
  if (opts.backend != Backend::Cuda) {
    throw NotImplemented("ScfDriver::run: requested backend is not compiled into this build");
  }

#if !defined(GPU4DFT_HAVE_CUDA)
  throw std::runtime_error("ScfDriver::run: CUDA backend requested, but GPU4DFT was built without CUDA");
#else
  const bool has_separate_jk = static_cast<bool>(callbacks.build_jk);
  const bool has_vhf = static_cast<bool>(callbacks.build_vhf);
  if (!has_separate_jk && !has_vhf) {
    throw std::invalid_argument(
        "ScfDriver::run: FockCallbacks must provide either build_jk or build_vhf");
  }

  DeviceMemoryPool& pool =
      (impl_->ext_pool != nullptr) ? *impl_->ext_pool : default_device_pool();
  GpuLinalgContext ctx(&pool);
  StreamHandle scf_stream = ctx.graph_executor().stream();

  // 1. One-time Canonical Orthogonalization on GPU
  const CanonicalOrthoResult ortho =
      canonical_orthogonalization(nao, S_host, opts.lindep_tol, Backend::Cuda);
  const int nmo = ortho.nmo;
  if (n_occ > nmo) {
    throw std::invalid_argument("ScfDriver::run: n_occ exceeds linearly independent basis size nmo");
  }

  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  const std::size_t nm = static_cast<std::size_t>(nao) * nmo;
  const std::size_t m2 = static_cast<std::size_t>(nmo) * nmo;
  const bool need_k = (opts.exact_exchange_fraction != 0.0);
  const double eff_diis_tol = std::min(opts.diis_tol, std::max(1e-10, opts.conv_tol * 100.0));

  // 2. Pre-allocate ALL GPU-resident SCF buffers from DeviceMemoryPool BEFORE the SCF loop!
  PooledBuffer<double> d_S(pool, n2, scf_stream);
  PooledBuffer<double> d_H(pool, n2, scf_stream);
  PooledBuffer<double> d_X(pool, nm, scf_stream);
  PooledBuffer<double> d_D(pool, n2, scf_stream);
  PooledBuffer<double> d_D_ref(pool, n2, scf_stream);
  PooledBuffer<double> d_delta_D(pool, n2, scf_stream);
  PooledBuffer<double> d_J(pool, n2, scf_stream);
  PooledBuffer<double> d_K(pool, n2, scf_stream);
  PooledBuffer<double> d_dJ(pool, n2, scf_stream);
  PooledBuffer<double> d_dK(pool, n2, scf_stream);
  PooledBuffer<double> d_Vxc(pool, n2, scf_stream);
  PooledBuffer<double> d_F(pool, n2, scf_stream);
  PooledBuffer<double> d_F_extrap(pool, n2, scf_stream);
  PooledBuffer<double> d_C(pool, nm, scf_stream);
  PooledBuffer<double> d_eps(pool, static_cast<std::size_t>(nmo), scf_stream);
  PooledBuffer<double> d_W(pool, n2, scf_stream);
  PooledBuffer<double> d_P_sp2(pool, m2, scf_stream);
  PooledBuffer<double> d_traces3(pool, 3, scf_stream);
  PooledBuffer<double> d_norm2(pool, 2, scf_stream);

  copy_h2d_async(d_S.data(), S_host, n2 * sizeof(double), scf_stream);
  copy_h2d_async(d_H.data(), H_core_host, n2 * sizeof(double), scf_stream);
  copy_h2d_async(d_X.data(), ortho.X.data(), nm * sizeof(double), scf_stream);
  memset_zero_async(d_J.data(), n2 * sizeof(double), scf_stream);
  memset_zero_async(d_K.data(), n2 * sizeof(double), scf_stream);
  memset_zero_async(d_Vxc.data(), n2 * sizeof(double), scf_stream);

  // Initial density matrix guess on GPU
  if (initial_dm_host != nullptr) {
    copy_h2d_async(d_D.data(), initial_dm_host, n2 * sizeof(double), scf_stream);
  } else if (opts.solver == DensitySolver::Diag) {
    solve_fock_diag_device(ctx, nao, nmo, n_occ, d_H.data(), d_X.data(), d_eps.data(), d_C.data(),
                           d_D.data(), nullptr, opts.gemm_mode, scf_stream, opts.ozaki_slices);
  } else {
    Sp2Options sp2_init = opts.sp2_options;
    sp2_init.gemm_mode = opts.gemm_mode;
    sp2_init.ozaki_slices = opts.ozaki_slices;
    sp2_init.use_cuda_graph = opts.use_cuda_graphs;
    (void)solve_fock_sp2_device(ctx, nao, nmo, n_occ, d_H.data(), d_X.data(), d_P_sp2.data(),
                                d_D.data(), nullptr, sp2_init, scf_stream);
  }

  DiisOptions diis_opts;
  diis_opts.max_vec = opts.diis_max_vec;
  diis_opts.damping = opts.damping;
  diis_opts.damping_steps = opts.damping_steps;
  diis_opts.level_shift = opts.level_shift;
  diis_opts.use_cuda_graph = opts.use_cuda_graphs;
  diis_opts.gemm_mode = opts.gemm_mode;
  GpuDiis diis(nao, nmo, diis_opts, &pool);

  // Host staging buffers ONLY allocated if a callback uses host pointers (`!device_resident`)
  std::vector<double> h_stage_dm;
  std::vector<double> h_stage_vj;
  std::vector<double> h_stage_vk;
  std::vector<double> h_stage_vxc;
  if (!callbacks.device_resident) {
    h_stage_dm.resize(n2, 0.0);
    h_stage_vj.resize(n2, 0.0);
    if (need_k && has_separate_jk) h_stage_vk.resize(n2, 0.0);
  }
  if (callbacks.build_vxc && !callbacks.vxc_device_resident && !callbacks.device_resident) {
    if (h_stage_dm.empty()) h_stage_dm.resize(n2, 0.0);
    h_stage_vxc.resize(n2, 0.0);
  } else if (callbacks.build_vxc && !callbacks.vxc_device_resident) {
    if (h_stage_dm.empty()) h_stage_dm.resize(n2, 0.0);
    h_stage_vxc.resize(n2, 0.0);
  }

  bool in_final_fp64 =
      !opts.dynamic_precision ||
      (opts.early_integral_precision == Precision::FP64 && opts.early_gemm_mode == opts.gemm_mode);
  Precision cur_int_prec = in_final_fp64 ? Precision::FP64 : opts.early_integral_precision;
  GemmMode cur_gemm = in_final_fp64 ? opts.gemm_mode : opts.early_gemm_mode;

  bool force_full_reset = true;
  bool verified_full_build = false;
  int steps_since_reset = 0;
  double prev_energy = 0.0;

  ScfResult res;
  res.nao = nao;
  res.nmo = nmo;
  res.nuclear_repulsion = e_nuc;

  std::size_t allocs_after_warmup_baseline = 0;
  bool baseline_recorded = false;

  // 3. GPU-Resident SCF Main Loop (only scalar energy/residual transfers per iteration!)
  for (int iter = 0; iter < opts.max_iter; ++iter) {
    const bool do_inc = opts.incremental_fock && !force_full_reset && iter > 0 &&
                        (steps_since_reset < std::max(1, opts.fock_reset_interval));

    const double* d_jk_input = d_D.data();
    if (do_inc) {
      compute_delta_and_norm_device(n2, d_D.data(), d_D_ref.data(), d_delta_D.data(), nullptr,
                                    scf_stream);
      d_jk_input = d_delta_D.data();
    }

    // 3a. Two-electron Coulomb / Exchange update (Full or Incremental)
    if (callbacks.device_resident) {
      double* target_j = do_inc ? d_dJ.data() : d_J.data();
      double* target_k = do_inc ? d_dK.data() : d_K.data();
      memset_zero_async(target_j, n2 * sizeof(double), scf_stream);
      if (need_k && has_separate_jk) {
        memset_zero_async(target_k, n2 * sizeof(double), scf_stream);
      }
      if (has_separate_jk) {
        callbacks.build_jk(d_jk_input, do_inc, cur_int_prec, target_j,
                           need_k ? target_k : nullptr, scf_stream);
      } else {
        callbacks.build_vhf(d_jk_input, do_inc, cur_int_prec, target_j, scf_stream);
      }
      if (do_inc) {
        axpby_device(n2, 1.0, d_dJ.data(), 1.0, d_J.data(), scf_stream);
        if (need_k && has_separate_jk) {
          axpby_device(n2, 1.0, d_dK.data(), 1.0, d_K.data(), scf_stream);
        }
      }
    } else {
      copy_d2h_async(h_stage_dm.data(), d_jk_input, n2 * sizeof(double), scf_stream);
      synchronize_stream(scf_stream);
      std::fill(h_stage_vj.begin(), h_stage_vj.end(), 0.0);
      if (has_separate_jk) {
        if (need_k) std::fill(h_stage_vk.begin(), h_stage_vk.end(), 0.0);
        callbacks.build_jk(h_stage_dm.data(), do_inc, cur_int_prec, h_stage_vj.data(),
                           need_k ? h_stage_vk.data() : nullptr, nullptr);
      } else {
        callbacks.build_vhf(h_stage_dm.data(), do_inc, cur_int_prec, h_stage_vj.data(), nullptr);
      }
      if (do_inc) {
        copy_h2d_async(d_dJ.data(), h_stage_vj.data(), n2 * sizeof(double), scf_stream);
        axpby_device(n2, 1.0, d_dJ.data(), 1.0, d_J.data(), scf_stream);
        if (need_k && has_separate_jk) {
          copy_h2d_async(d_dK.data(), h_stage_vk.data(), n2 * sizeof(double), scf_stream);
          axpby_device(n2, 1.0, d_dK.data(), 1.0, d_K.data(), scf_stream);
        }
      } else {
        copy_h2d_async(d_J.data(), h_stage_vj.data(), n2 * sizeof(double), scf_stream);
        if (need_k && has_separate_jk) {
          copy_h2d_async(d_K.data(), h_stage_vk.data(), n2 * sizeof(double), scf_stream);
        }
      }
    }

    copy_d2d_async(d_D_ref.data(), d_D.data(), n2 * sizeof(double), scf_stream);
    if (do_inc) {
      steps_since_reset += 1;
    } else {
      force_full_reset = false;
      steps_since_reset = 0;
    }

    // 3b. Exchange-Correlation V_xc(D) & E_xc (always evaluated on FULL density D)
    double e_xc = 0.0;
    if (callbacks.build_vxc) {
      if (callbacks.vxc_device_resident) {
        memset_zero_async(d_Vxc.data(), n2 * sizeof(double), scf_stream);
        e_xc = callbacks.build_vxc(d_D.data(), cur_int_prec, d_Vxc.data(), scf_stream);
      } else {
        copy_d2h_async(h_stage_dm.data(), d_D.data(), n2 * sizeof(double), scf_stream);
        synchronize_stream(scf_stream);
        std::fill(h_stage_vxc.begin(), h_stage_vxc.end(), 0.0);
        e_xc = callbacks.build_vxc(h_stage_dm.data(), cur_int_prec, h_stage_vxc.data(), nullptr);
        copy_h2d_async(d_Vxc.data(), h_stage_vxc.data(), n2 * sizeof(double), scf_stream);
      }
    }

    // 3c. Fused GPU Fock Assembly & Scalar Energy Traces (transfers ONLY 3 doubles = 24 bytes!)
    const double eff_alpha = has_separate_jk ? opts.exact_exchange_fraction : 0.0;
    assemble_fock_and_traces_device(
        nao, d_D.data(), d_H.data(), d_J.data(),
        (need_k && has_separate_jk) ? d_K.data() : nullptr, eff_alpha,
        callbacks.build_vxc ? d_Vxc.data() : nullptr, d_F.data(), d_traces3.data(), scf_stream);
    symmetrize_matrix_device(nao, d_F.data(), scf_stream);

    double h_traces[3] = {0.0, 0.0, 0.0};
    copy_d2h_async(h_traces, d_traces3.data(), 3 * sizeof(double), scf_stream);
    synchronize_stream(scf_stream);

    const double e_1e = h_traces[0];
    const double e_2e = has_separate_jk ? (0.5 * h_traces[1] - 0.25 * eff_alpha * h_traces[2])
                                        : (0.5 * h_traces[1]);
    const double e_elec = e_1e + e_2e + e_xc;
    const double e_tot = e_nuc + e_elec;
    const double delta_e = (iter == 0) ? e_tot : (e_tot - prev_energy);
    prev_energy = e_tot;

    // 3d. GPU CDIIS Residual, 8x8 Host Pulay Solve, and CUDA-Graph-Captured Extrapolation
    diis.set_gemm_mode(cur_gemm);
    const DiisStepInfo diis_info = diis.update_device(
        ctx, d_F.data(), d_D.data(), d_S.data(), d_X.data(), d_F_extrap.data(), scf_stream);
    const double rms_err = diis_info.rms_error;

    res.iterations = iter + 1;
    res.total_energy = e_tot;
    res.electronic_energy = e_elec;
    res.one_electron_energy = e_1e;
    res.two_electron_energy = e_2e;
    res.xc_energy = e_xc;
    res.final_diis_rms = rms_err;
    res.history.push_back(
        ScfIterationLog{iter + 1, e_tot, delta_e, rms_err, do_inc, cur_int_prec, cur_gemm});

    // 3e. Scalar-only Convergence & Dynamic Precision Promotion Check
    bool promoted_this_step = false;
    if (!in_final_fp64 &&
        (rms_err < opts.fp64_switch_rms ||
         (iter > 0 && std::abs(delta_e) < std::max(100.0 * opts.conv_tol, 1e-6)))) {
      in_final_fp64 = true;
      promoted_this_step = true;
      if (opts.early_gemm_mode == GemmMode::FP32 ||
          opts.early_integral_precision != Precision::FP64) {
        diis.reset(scf_stream);
      }
      cur_int_prec = Precision::FP64;
      cur_gemm = opts.gemm_mode;
      force_full_reset = true;
      // Reset steady-state allocation baseline right after precision switch warms up new GEMM buffers
      baseline_recorded = false;
    } else if (iter >= 1 && in_final_fp64 && std::abs(delta_e) < opts.conv_tol &&
               rms_err < eff_diis_tol) {
      if (opts.incremental_fock && steps_since_reset >= 4 && !verified_full_build) {
        force_full_reset = true;
        verified_full_build = true;
      } else {
        res.converged = true;
        break;
      }
    }

    // 3f. GPU Density Update (cuSOLVER Diag or CUDA-Graph-Captured SP2 Purification)
    const GemmMode step_gemm = promoted_this_step ? opts.gemm_mode : cur_gemm;
    if (opts.solver == DensitySolver::Diag) {
      solve_fock_diag_device(ctx, nao, nmo, n_occ, d_F_extrap.data(), d_X.data(), d_eps.data(),
                             d_C.data(), d_D.data(), nullptr, step_gemm, scf_stream,
                             opts.ozaki_slices);
    } else {
      Sp2Options sp2_opts = opts.sp2_options;
      sp2_opts.gemm_mode = step_gemm;
      sp2_opts.ozaki_slices = opts.ozaki_slices;
      sp2_opts.use_cuda_graph = opts.use_cuda_graphs;
      (void)solve_fock_sp2_device(ctx, nao, nmo, n_occ, d_F_extrap.data(), d_X.data(),
                                  d_P_sp2.data(), d_D.data(), nullptr, sp2_opts, scf_stream);
    }

    if (!baseline_recorded && iter >= 1 && !promoted_this_step) {
      synchronize_stream(scf_stream);
      allocs_after_warmup_baseline = pool.stats().async_allocations;
      baseline_recorded = true;
    }
  }

  // Record steady-state allocation delta BEFORE final host extraction
  synchronize_stream(scf_stream);
  if (baseline_recorded) {
    const std::size_t post_loop_allocs = pool.stats().async_allocations;
    res.steady_state_allocations =
        (post_loop_allocs >= allocs_after_warmup_baseline)
            ? (post_loop_allocs - allocs_after_warmup_baseline)
            : 0;
  }

  // 4. Finalize unshifted MO coefficients/energies and energy-weighted density W on GPU,
  //    then download final results once!
  if (opts.solver == DensitySolver::Diag) {
    solve_fock_diag_device(ctx, nao, nmo, n_occ, d_F.data(), d_X.data(), d_eps.data(), d_C.data(),
                           d_D.data(), d_W.data(), opts.gemm_mode, scf_stream, opts.ozaki_slices);
    res.mo_energies.resize(static_cast<std::size_t>(nmo), 0.0);
    res.mo_coefficients.resize(nm, 0.0);
    copy_d2h_async(res.mo_energies.data(), d_eps.data(),
                   static_cast<std::size_t>(nmo) * sizeof(double), scf_stream);
    copy_d2h_async(res.mo_coefficients.data(), d_C.data(), nm * sizeof(double), scf_stream);
  } else {
    // Exact orbital-free energy-weighted density matrix W = 0.5 * D * F * D
    PooledBuffer<double> d_DF(pool, n2, scf_stream);
    gemm_device(ctx, nao, nao, nao, 1.0, d_D.data(), nao, false, d_F.data(), nao, false, 0.0,
                d_DF.data(), nao, opts.gemm_mode, scf_stream, opts.ozaki_slices);
    gemm_device(ctx, nao, nao, nao, 0.5, d_DF.data(), nao, false, d_D.data(), nao, false, 0.0,
                d_W.data(), nao, opts.gemm_mode, scf_stream, opts.ozaki_slices);
    symmetrize_matrix_device(nao, d_W.data(), scf_stream);
  }

  res.density_matrix.resize(n2, 0.0);
  res.energy_weighted_dm.resize(n2, 0.0);
  res.fock_matrix.resize(n2, 0.0);
  copy_d2h_async(res.density_matrix.data(), d_D.data(), n2 * sizeof(double), scf_stream);
  copy_d2h_async(res.energy_weighted_dm.data(), d_W.data(), n2 * sizeof(double), scf_stream);
  copy_d2h_async(res.fock_matrix.data(), d_F.data(), n2 * sizeof(double), scf_stream);
  synchronize_stream(scf_stream);

  res.cuda_graph_captures = ctx.graph_executor().capture_count();
  res.cuda_graph_replays = ctx.graph_executor().replay_count();
  res.pool_cache_hits = pool.stats().cache_hits;
  return res;
#endif
}

std::vector<double> ScfDriver::compute_nuclear_gradient(
    int natom, const ScfResult& scf_res, const GradCallbacks& grad_callbacks) const {
  if (natom <= 0) return {};
  const std::size_t ncomp = static_cast<std::size_t>(natom) * 3;
  std::vector<double> total_grad(ncomp, 0.0);

  auto accumulate_term = [&](const std::vector<double>& term, const char* name) {
    if (term.empty()) return;
    if (term.size() != ncomp) {
      throw std::runtime_error(std::string("ScfDriver::compute_nuclear_gradient: ") + name +
                               " returned size " + std::to_string(term.size()) + ", expected " +
                               std::to_string(ncomp));
    }
    for (std::size_t i = 0; i < ncomp; ++i) {
      total_grad[i] += term[i];
    }
  };

  if (grad_callbacks.nuclear_repulsion_grad_fn) {
    accumulate_term(grad_callbacks.nuclear_repulsion_grad_fn(), "nuclear_repulsion_grad_fn");
  }
  if (grad_callbacks.one_electron_grad_fn) {
    accumulate_term(grad_callbacks.one_electron_grad_fn(scf_res.density_matrix.data(),
                                                        scf_res.energy_weighted_dm.data()),
                    "one_electron_grad_fn");
  }
  if (grad_callbacks.two_electron_grad_fn) {
    accumulate_term(grad_callbacks.two_electron_grad_fn(scf_res.density_matrix.data(),
                                                        impl_->opts.exact_exchange_fraction),
                    "two_electron_grad_fn");
  }
  if (grad_callbacks.xc_grad_fn) {
    accumulate_term(grad_callbacks.xc_grad_fn(scf_res.density_matrix.data()), "xc_grad_fn");
  }
  return total_grad;
}

// ---------------------------------------------------------------------------
// Standalone GPU-Resident In-Core ERI Callback & Gradient Assembly Helpers
// ---------------------------------------------------------------------------

FockCallbacks make_in_core_eri_callbacks(int nao, const double* h_eri, Backend backend) {
  if (nao <= 0 || h_eri == nullptr) {
    throw std::invalid_argument("make_in_core_eri_callbacks: invalid nao or null ERI pointer");
  }
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;
  const std::size_t n4 = n2 * n2;

  // Permute ERI_K[(u, v), (l, s)] = ERI[(u, l), (v, s)] so both J and K are single dense GEMVs!
  auto eri_j = std::make_shared<std::vector<double>>(h_eri, h_eri + n4);
  auto eri_k = std::make_shared<std::vector<double>>(n4, 0.0);
  for (int u = 0; u < nao; ++u) {
    for (int v = 0; v < nao; ++v) {
      const std::size_t uv = static_cast<std::size_t>(u) * nao + v;
      for (int l = 0; l < nao; ++l) {
        for (int s = 0; s < nao; ++s) {
          const std::size_t ls = static_cast<std::size_t>(l) * nao + s;
          const std::size_t ul = static_cast<std::size_t>(u) * nao + l;
          const std::size_t vs = static_cast<std::size_t>(v) * nao + s;
          (*eri_k)[uv * n2 + ls] = h_eri[ul * n2 + vs];
        }
      }
    }
  }

  FockCallbacks cb;
  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    struct GpuEriState {
      int nao = 0;
      std::size_t n2 = 0;
      GpuLinalgContext ctx;
      PooledBuffer<double> d_eri_j;
      PooledBuffer<double> d_eri_k;
      // Mixed-precision quartet split: |ERI| >= 1e-4 in FP64, |ERI| < 1e-4 in FP32
      PooledBuffer<double> d_eri_j_hi;
      PooledBuffer<double> d_eri_j_lo;
      PooledBuffer<double> d_eri_k_hi;
      PooledBuffer<double> d_eri_k_lo;
      bool mixed_initialized = false;

      GpuEriState(int nao_, const std::vector<double>& hj, const std::vector<double>& hk)
          : nao(nao_),
            n2(static_cast<std::size_t>(nao_) * nao_),
            ctx(&default_device_pool()),
            d_eri_j(ctx.pool(), n2 * n2),
            d_eri_k(ctx.pool(), n2 * n2) {
        copy_h2d_async(d_eri_j.data(), hj.data(), hj.size() * sizeof(double));
        copy_h2d_async(d_eri_k.data(), hk.data(), hk.size() * sizeof(double));
        synchronize_stream(nullptr);
      }

      void ensure_mixed(const std::vector<double>& hj, const std::vector<double>& hk,
                        StreamHandle stream) {
        if (mixed_initialized) return;
        const std::size_t n4 = n2 * n2;
        std::vector<double> j_hi(n4, 0.0), j_lo(n4, 0.0), k_hi(n4, 0.0), k_lo(n4, 0.0);
        constexpr double kSplit = 1e-4;
        for (std::size_t i = 0; i < n4; ++i) {
          if (std::abs(hj[i]) >= kSplit) {
            j_hi[i] = hj[i];
          } else {
            j_lo[i] = hj[i];
          }
          if (std::abs(hk[i]) >= kSplit) {
            k_hi[i] = hk[i];
          } else {
            k_lo[i] = hk[i];
          }
        }
        d_eri_j_hi = PooledBuffer<double>(ctx.pool(), n4, stream);
        d_eri_j_lo = PooledBuffer<double>(ctx.pool(), n4, stream);
        d_eri_k_hi = PooledBuffer<double>(ctx.pool(), n4, stream);
        d_eri_k_lo = PooledBuffer<double>(ctx.pool(), n4, stream);
        copy_h2d_async(d_eri_j_hi.data(), j_hi.data(), n4 * sizeof(double), stream);
        copy_h2d_async(d_eri_j_lo.data(), j_lo.data(), n4 * sizeof(double), stream);
        copy_h2d_async(d_eri_k_hi.data(), k_hi.data(), n4 * sizeof(double), stream);
        copy_h2d_async(d_eri_k_lo.data(), k_lo.data(), n4 * sizeof(double), stream);
        mixed_initialized = true;
      }
    };

    auto state = std::make_shared<GpuEriState>(nao, *eri_j, *eri_k);
    cb.device_resident = true;
    cb.build_jk = [state, eri_j, eri_k](const double* d_dm, bool /*is_inc*/, Precision prec,
                                        double* d_vj, double* d_vk, void* raw_stream) {
      auto s = static_cast<StreamHandle>(raw_stream);
      const int m = static_cast<int>(state->n2);
      if (prec == Precision::FP64) {
        gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_j.data(), m, false, d_dm, 1, false, 0.0,
                    d_vj, 1, GemmMode::FP64, s);
        if (d_vk != nullptr) {
          gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_k.data(), m, false, d_dm, 1, false,
                      0.0, d_vk, 1, GemmMode::FP64, s);
        }
      } else if (prec == Precision::FP32) {
        gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_j.data(), m, false, d_dm, 1, false, 0.0,
                    d_vj, 1, GemmMode::FP32, s);
        if (d_vk != nullptr) {
          gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_k.data(), m, false, d_dm, 1, false,
                      0.0, d_vk, 1, GemmMode::FP32, s);
        }
      } else {
        // Precision::Mixed — large quartets in FP64, small quartets in FP32, accumulated in FP64!
        state->ensure_mixed(*eri_j, *eri_k, s);
        gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_j_hi.data(), m, false, d_dm, 1, false,
                    0.0, d_vj, 1, GemmMode::FP64, s);
        gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_j_lo.data(), m, false, d_dm, 1, false,
                    1.0, d_vj, 1, GemmMode::FP32, s);
        if (d_vk != nullptr) {
          gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_k_hi.data(), m, false, d_dm, 1, false,
                      0.0, d_vk, 1, GemmMode::FP64, s);
          gemm_device(state->ctx, m, 1, m, 1.0, state->d_eri_k_lo.data(), m, false, d_dm, 1, false,
                      1.0, d_vk, 1, GemmMode::FP32, s);
        }
      }
    };
    return cb;
#else
    throw std::runtime_error("make_in_core_eri_callbacks: built without CUDA");
#endif
  }

  // Host ERI callback
  cb.device_resident = false;
  cb.build_jk = [n2, eri_j, eri_k](const double* h_dm, bool /*is_inc*/, Precision prec,
                                   double* h_vj, double* h_vk, void* /*stream*/) {
    const int m = static_cast<int>(n2);
    const GemmMode gmode = (prec == Precision::FP32) ? GemmMode::FP32 : GemmMode::FP64;
    gemm(m, 1, m, 1.0, eri_j->data(), false, h_dm, false, 0.0, h_vj, Backend::Host, gmode);
    if (h_vk != nullptr) {
      gemm(m, 1, m, 1.0, eri_k->data(), false, h_dm, false, 0.0, h_vk, Backend::Host, gmode);
    }
  };
  return cb;
}

std::vector<double> assemble_nuclear_gradient(int natom, int nao, const double* dm_host,
                                              const double* energy_weighted_dm_host,
                                              const double* grad_nuc_3n,
                                              const double* dhcore_dr_3n_nao2,
                                              const double* ds_dr_3n_nao2,
                                              const double* grad_2e_3n,
                                              const double* grad_xc_3n) {
  if (natom <= 0 || nao <= 0) return {};
  const std::size_t ncomp = static_cast<std::size_t>(natom) * 3;
  const std::size_t n2 = static_cast<std::size_t>(nao) * nao;

  GradCallbacks gc;
  if (grad_nuc_3n != nullptr) {
    gc.nuclear_repulsion_grad_fn = [grad_nuc_3n, ncomp]() {
      return std::vector<double>(grad_nuc_3n, grad_nuc_3n + ncomp);
    };
  }
  if (dhcore_dr_3n_nao2 != nullptr || ds_dr_3n_nao2 != nullptr) {
    gc.one_electron_grad_fn = [=](const double* D, const double* W) {
      std::vector<double> g1(ncomp, 0.0);
      for (std::size_t c = 0; c < ncomp; ++c) {
        double val = 0.0;
        if (dhcore_dr_3n_nao2 != nullptr) {
          val += dot_host(n2, D, dhcore_dr_3n_nao2 + c * n2);
        }
        if (ds_dr_3n_nao2 != nullptr) {
          val -= dot_host(n2, W, ds_dr_3n_nao2 + c * n2);
        }
        g1[c] = val;
      }
      return g1;
    };
  }
  if (grad_2e_3n != nullptr) {
    gc.two_electron_grad_fn = [grad_2e_3n, ncomp](const double*, double) {
      return std::vector<double>(grad_2e_3n, grad_2e_3n + ncomp);
    };
  }
  if (grad_xc_3n != nullptr) {
    gc.xc_grad_fn = [grad_xc_3n, ncomp](const double*) {
      return std::vector<double>(grad_xc_3n, grad_xc_3n + ncomp);
    };
  }

  ScfResult dummy;
  dummy.nao = nao;
  dummy.density_matrix.assign(dm_host, dm_host + n2);
  dummy.energy_weighted_dm.assign(energy_weighted_dm_host, energy_weighted_dm_host + n2);

  ScfDriver drv;
  return drv.compute_nuclear_gradient(natom, dummy, gc);
}

}  // namespace gpu4dft

// ===========================================================================
// C ABI Exports for Standalone ctypes / pytest Verification of All Subsystems
// ===========================================================================

namespace {

gpu4dft::Backend c_to_backend(int b) {
  return (b == 1) ? gpu4dft::Backend::Cuda : gpu4dft::Backend::Host;
}

gpu4dft::GemmMode c_to_gemm_mode(int m) {
  if (m == 1) return gpu4dft::GemmMode::FP32;
  if (m == 2) return gpu4dft::GemmMode::OzakiInt8;
  return gpu4dft::GemmMode::FP64;
}

}  // namespace

GPU4DFT_C_EXPORT int gpu4dft_c_cuda_available() {
  return (gpu4dft::MultiGpuManager::visible_device_count() > 0) ? 1 : 0;
}

GPU4DFT_C_EXPORT int gpu4dft_c_gemm(int m, int n, int k, double alpha, const double* A, int transA,
                                    const double* B, int transB, double beta, double* C,
                                    int backend, int gemm_mode, int ozaki_slices) {
  try {
    gpu4dft::gemm(m, n, k, alpha, A, transA != 0, B, transB != 0, beta, C, c_to_backend(backend),
                  c_to_gemm_mode(gemm_mode), ozaki_slices);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_benchmark_gemm(int n, int warmup_iters, int bench_iters, int backend,
                                              int ozaki_slices, double* out9) {
  try {
    const auto r = gpu4dft::benchmark_gemm(n, warmup_iters, bench_iters, c_to_backend(backend),
                                           ozaki_slices);
    out9[0] = r.fp64_ms;
    out9[1] = r.fp64_tflops;
    out9[2] = r.ozaki_ms;
    out9[3] = r.ozaki_tflops;
    out9[4] = r.fp32_ms;
    out9[5] = r.fp32_tflops;
    out9[6] = r.ozaki_rel_error;
    out9[7] = r.ozaki_max_abs_error;
    out9[8] = r.fp32_rel_error;
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_canonical_ortho(int nao, const double* S, double lindep_tol,
                                               int backend, int* nmo_out, double* X_out) {
  try {
    const auto r =
        gpu4dft::canonical_orthogonalization(nao, S, lindep_tol, c_to_backend(backend));
    *nmo_out = r.nmo;
    std::copy(r.X.begin(), r.X.end(), X_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_solve_fock_diag(int nao, int nmo, int n_occ, const double* F,
                                               const double* X, int backend, int gemm_mode,
                                               double* eps_out, double* C_out, double* D_out,
                                               double* W_out) {
  try {
    const auto r = gpu4dft::solve_fock_diag(nao, nmo, n_occ, F, X, c_to_backend(backend),
                                            c_to_gemm_mode(gemm_mode));
    if (eps_out) std::copy(r.mo_energies.begin(), r.mo_energies.end(), eps_out);
    if (C_out) std::copy(r.mo_coeff.begin(), r.mo_coeff.end(), C_out);
    if (D_out) std::copy(r.dm.begin(), r.dm.end(), D_out);
    if (W_out) std::copy(r.energy_dm.begin(), r.energy_dm.end(), W_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_solve_fock_sp2(int nao, int nmo, int n_occ, const double* F,
                                              const double* X, int backend, int gemm_mode,
                                              int use_graph, double* D_out, double* W_out,
                                              double* stats5_out) {
  try {
    gpu4dft::Sp2Options opts;
    opts.gemm_mode = c_to_gemm_mode(gemm_mode);
    opts.use_cuda_graph = (use_graph != 0);
    const auto r =
        gpu4dft::solve_fock_sp2(nao, nmo, n_occ, F, X, c_to_backend(backend), opts);
    if (D_out) std::copy(r.dm.begin(), r.dm.end(), D_out);
    if (W_out) std::copy(r.energy_dm.begin(), r.energy_dm.end(), W_out);
    if (stats5_out) {
      stats5_out[0] = static_cast<double>(r.iterations);
      stats5_out[1] = r.final_trace;
      stats5_out[2] = r.idempotency_error;
      stats5_out[3] = static_cast<double>(r.graph_captures);
      stats5_out[4] = static_cast<double>(r.graph_replays);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_test_pool_and_graphs(int n, int steps, std::size_t* out_stats6) {
  try {
    gpu4dft::DeviceMemoryPool pool;
    gpu4dft::GpuLinalgContext ctx(&pool);
    auto& graph = ctx.graph_executor();
    gpu4dft::StreamHandle s = graph.stream();

    const std::size_t nn = static_cast<std::size_t>(n) * n;
    std::vector<double> h_A(nn, 0.5 / static_cast<double>(n));
    std::vector<double> h_out(nn, 0.0);

    gpu4dft::PooledBuffer<double> d_A(pool, nn, s);
    gpu4dft::PooledBuffer<double> d_B(pool, nn, s);
    gpu4dft::copy_h2d_async(d_A.data(), h_A.data(), nn * sizeof(double), s);
    gpu4dft::memset_zero_async(d_B.data(), nn * sizeof(double), s);

    std::size_t allocs_after_warmup = 0;
    const std::uint64_t key = 0x12345678ABCDEF01ULL;
    for (int it = 0; it < steps; ++it) {
      graph.execute(key, [&](gpu4dft::StreamHandle gs) {
        gpu4dft::PooledBuffer<double> d_tmp(pool, nn, gs);
        gpu4dft::gemm_device(ctx, n, n, n, 1.0, d_A.data(), n, false, d_A.data(), n, false, 0.0,
                             d_tmp.data(), n, gpu4dft::GemmMode::FP64, gs);
        gpu4dft::axpby_device(nn, 1.0, d_tmp.data(), 1.0, d_B.data(), gs);
      });
      if (it == 1) {
        graph.synchronize();
        allocs_after_warmup = pool.stats().async_allocations;
      }
    }
    graph.synchronize();
    gpu4dft::copy_d2h_async(h_out.data(), d_B.data(), nn * sizeof(double), s);
    graph.synchronize();

    // Expected: d_A * d_A has every entry = n * (0.5/n)^2 = 0.25 / n; accumulated `steps` times.
    const double expected = static_cast<double>(steps) * (0.25 / static_cast<double>(n));
    if (std::abs(h_out[0] - expected) > 1e-12) {
      return -2;
    }

    const auto st = pool.stats();
    out_stats6[0] = st.async_allocations - allocs_after_warmup;  // 0 new allocs in steady state!
    out_stats6[1] = st.cache_hits;
    out_stats6[2] = graph.capture_count();
    out_stats6[3] = graph.replay_count();
    out_stats6[4] = st.allocated_bytes;
    out_stats6[5] = st.peak_in_use_bytes;
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_test_multigpu(int num_tasks, const int* weights, double* out_vals,
                                             int* out_meta4) {
  try {
    gpu4dft::MultiGpuManager mgr;
    gpu4dft::DynamicTaskQueue queue;
    const auto block_counts = queue.execute_weighted_tasks(num_tasks, weights, out_vals, 16);

    int total_claimed = 0;
    int active_blocks = 0;
    for (int c : block_counts) {
      total_claimed += c;
      if (c > 0) active_blocks += 1;
    }

    // Also test GPU partition allreduce summation across 4 buffers
    auto& pool = mgr.pool(0);
    constexpr std::size_t kLen = 256;
    std::vector<double> h0(kLen, 1.5), h1(kLen, 2.25), h2(kLen, -0.5), h3(kLen, 4.75);
    gpu4dft::PooledBuffer<double> b0(pool, kLen), b1(pool, kLen), b2(pool, kLen), b3(pool, kLen);
    gpu4dft::copy_h2d_async(b0.data(), h0.data(), kLen * sizeof(double));
    gpu4dft::copy_h2d_async(b1.data(), h1.data(), kLen * sizeof(double));
    gpu4dft::copy_h2d_async(b2.data(), h2.data(), kLen * sizeof(double));
    gpu4dft::copy_h2d_async(b3.data(), h3.data(), kLen * sizeof(double));
    gpu4dft::synchronize_stream(nullptr);

    std::vector<double*> ptrs = {b0.data(), b1.data(), b2.data(), b3.data()};
    mgr.allreduce_local_partitions(ptrs, kLen);
    mgr.allreduce_sum(b0.data(), kLen);

    std::vector<double> check_vec(kLen, 0.0);
    gpu4dft::copy_d2h_async(check_vec.data(), b2.data(), kLen * sizeof(double));
    gpu4dft::synchronize_stream(nullptr);
    for (double v : check_vec) {
      if (std::abs(v - 8.0) > 1e-13) return -3;
    }

    out_meta4[0] = mgr.num_devices();
    out_meta4[1] = total_claimed;
    out_meta4[2] = active_blocks;
    out_meta4[3] = queue.value();
    return 0;
  } catch (...) {
    return -1;
  }
}

using gpu4dft_vxc_cb_t = double (*)(const double* dm_host, double* vxc_host, void* user_data);

GPU4DFT_C_EXPORT int gpu4dft_c_run_scf_eri(
    int nao, int n_elec, double e_nuc, const double* S_host, const double* H_core_host,
    const double* eri_host, const double* init_dm_host, int backend, int solver_mode,
    int gemm_mode, int dynamic_prec, int incremental_fock, int use_graphs, double alpha_x,
    double conv_tol, double damping, double level_shift, gpu4dft_vxc_cb_t vxc_cb, void* vxc_user,
    double* D_out, double* W_out, double* F_out, double* mo_eps_out, double* stats8_out) {
  try {
    gpu4dft::ScfOptions opts;
    opts.backend = c_to_backend(backend);
    opts.solver = (solver_mode == 1) ? gpu4dft::DensitySolver::SP2 : gpu4dft::DensitySolver::Diag;
    opts.gemm_mode = c_to_gemm_mode(gemm_mode);
    opts.dynamic_precision = (dynamic_prec != 0);
    if (opts.dynamic_precision) {
      opts.early_integral_precision = gpu4dft::Precision::Mixed;
      opts.early_gemm_mode = gpu4dft::GemmMode::OzakiInt8;
    }
    opts.incremental_fock = (incremental_fock != 0);
    opts.use_cuda_graphs = (use_graphs != 0);
    opts.exact_exchange_fraction = alpha_x;
    opts.conv_tol = conv_tol;
    opts.damping = damping;
    opts.level_shift = level_shift;

    gpu4dft::FockCallbacks cb =
        gpu4dft::make_in_core_eri_callbacks(nao, eri_host, opts.backend);
    if (vxc_cb != nullptr) {
      cb.vxc_device_resident = false;
      cb.build_vxc = [vxc_cb, vxc_user](const double* dm_h, gpu4dft::Precision, double* vxc_h,
                                        void*) -> double {
        return vxc_cb(dm_h, vxc_h, vxc_user);
      };
    }

    gpu4dft::ScfDriver driver(opts);
    const gpu4dft::ScfResult res =
        driver.run(nao, n_elec, e_nuc, S_host, H_core_host, cb, init_dm_host);

    if (D_out) std::copy(res.density_matrix.begin(), res.density_matrix.end(), D_out);
    if (W_out) std::copy(res.energy_weighted_dm.begin(), res.energy_weighted_dm.end(), W_out);
    if (F_out) std::copy(res.fock_matrix.begin(), res.fock_matrix.end(), F_out);
    if (mo_eps_out && !res.mo_energies.empty()) {
      std::copy(res.mo_energies.begin(), res.mo_energies.end(), mo_eps_out);
    }
    if (stats8_out) {
      stats8_out[0] = res.converged ? 1.0 : 0.0;
      stats8_out[1] = static_cast<double>(res.iterations);
      stats8_out[2] = res.total_energy;
      stats8_out[3] = res.final_diis_rms;
      stats8_out[4] = static_cast<double>(res.cuda_graph_captures);
      stats8_out[5] = static_cast<double>(res.cuda_graph_replays);
      stats8_out[6] = static_cast<double>(res.steady_state_allocations);
      stats8_out[7] = static_cast<double>(res.pool_cache_hits);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_assemble_gradient(int natom, int nao, const double* dm_host,
                                                 const double* W_host, const double* grad_nuc,
                                                 const double* dhcore_dr, const double* ds_dr,
                                                 const double* grad_2e, const double* grad_xc,
                                                 double* grad_out) {
  try {
    const auto g = gpu4dft::assemble_nuclear_gradient(natom, nao, dm_host, W_host, grad_nuc,
                                                      dhcore_dr, ds_dr, grad_2e, grad_xc);
    std::copy(g.begin(), g.end(), grad_out);
    return 0;
  } catch (...) {
    return -1;
  }
}
