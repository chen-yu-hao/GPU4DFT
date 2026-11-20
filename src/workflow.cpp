#include "gpu4dft/workflow.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu4dft/device_pool.hpp"
#include "gpu4dft/dftd3.hpp"
#include "gpu4dft/eri.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/response.hpp"

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace gpu4dft {

const char* to_string(JKMode mode) {
  switch (mode) {
    case JKMode::Direct:
      return "direct";
    case JKMode::RI:
      return "ri";
    case JKMode::SnK:
      return "snk";
    case JKMode::RISnK:
      return "ri_snk";
  }
  return "unknown";
}

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_seconds(SteadyClock::time_point t0, SteadyClock::time_point t1) {
  return std::chrono::duration<double>(t1 - t0).count();
}

void sync_gpu_if_cuda(Backend backend) {
#if defined(GPU4DFT_HAVE_CUDA)
  if (backend == Backend::Cuda) {
    (void)cudaDeviceSynchronize();
  }
#else
  (void)backend;
#endif
}

bool is_hf_functional_name(std::string_view raw) {
  std::string norm;
  norm.reserve(raw.size());
  for (char ch : raw) {
    if (std::isspace(static_cast<unsigned char>(ch)) != 0 || ch == '-' || ch == '_') continue;
    norm.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
  }
  return norm.empty() || norm == "hf" || norm == "rhf" || norm == "uhf";
}

bool parse_dispersion_option(std::string_view raw) {
  std::string norm;
  norm.reserve(raw.size());
  for (char ch : raw) {
    const unsigned char uch = static_cast<unsigned char>(ch);
    if (std::isspace(uch) != 0 || ch == '-' || ch == '_' || ch == '(' || ch == ')') continue;
    norm.push_back(static_cast<char>(std::tolower(uch)));
  }
  if (norm.empty() || norm == "none" || norm == "off" || norm == "false") {
    return false;
  }
  if (norm == "d3bj" || norm == "dftd3bj" || norm == "dftd3") {
    return true;
  }
  throw std::invalid_argument("Unsupported dispersion model '" + std::string(raw) +
                              "'; expected ''/'none' or 'd3bj'");
}

std::vector<XcAtom> to_xc_atoms(std::span<const Atom> atoms) {
  std::vector<XcAtom> out;
  out.reserve(atoms.size());
  for (const Atom& a : atoms) {
    out.push_back(XcAtom{a.atomic_number, a.center});
  }
  return out;
}

double compute_active_ao_ratio(const MolecularGrid& grid) {
  if (grid.nbatches() == 0 || grid.nao() <= 0) return 0.0;
  double sum_ratio = 0.0;
  std::size_t valid_batches = 0;
  for (const SpatialBatch& b : grid.batches()) {
    if (b.npoints == 0) continue;
    sum_ratio += static_cast<double>(b.active_aos.size()) / static_cast<double>(grid.nao());
    valid_batches += 1;
  }
  return (valid_batches > 0) ? (sum_ratio / static_cast<double>(valid_batches)) : 0.0;
}

}  // namespace

struct MolecularScf::Impl {
  Basis basis;
  std::vector<Atom> atoms;
  std::vector<XcAtom> xc_atoms;
  int n_elec = 0;
  WorkflowOptions opts;
  std::optional<Basis> aux_basis;
  std::vector<double> ext_grid_coords;
  std::vector<double> ext_grid_weights;

  bool is_dft_flag = false;
  double exx_fraction = 1.0;
  RshCoefficients rsh_;
  bool dispersion_enabled = false;
  std::optional<dftd3::D3BJResult> d3_cached;
  double dispersion_setup_time_s = 0.0;

  bool initialized = false;
  double e_nuc = 0.0;
  std::vector<double> S_mat;
  std::vector<double> H_core_mat;

  MolecularGrid grid;
  std::unique_ptr<XcEngine> xc_engine;
  std::unique_ptr<DirectJKEngine> direct_jk;
  std::unique_ptr<RIJKEngine> ri_jk;
  std::unique_ptr<SemiNumericalK> snk_engine;

  double setup_time_s = 0.0;
  double one_elec_time_s = 0.0;
  double jk_setup_time_s = 0.0;
  double grid_setup_time_s = 0.0;
  std::size_t grid_npoints = 0;
  std::size_t grid_nbatches = 0;
  double active_ao_ratio = 0.0;

  std::optional<ScfResult> last_scf;

  Impl(Basis b, std::vector<Atom> at, int nelec, WorkflowOptions o, std::optional<Basis> aux,
       std::span<const double> gcoords, std::span<const double> gweights)
      : basis(std::move(b)),
        atoms(std::move(at)),
        xc_atoms(to_xc_atoms(atoms)),
        n_elec(nelec),
        opts(std::move(o)),
        aux_basis(std::move(aux)),
        ext_grid_coords(gcoords.begin(), gcoords.end()),
        ext_grid_weights(gweights.begin(), gweights.end()) {
    const bool is_unrestricted = opts.unrestricted || opts.scf.unrestricted;
    const int spin = (opts.spin != 0) ? opts.spin : opts.scf.spin;
    if (is_unrestricted) {
      opts.unrestricted = true;
      opts.scf.unrestricted = true;
      opts.spin = spin;
      opts.scf.spin = spin;
      if (n_elec < 0 || spin < 0 || spin > n_elec || ((n_elec - spin) % 2) != 0) {
        throw std::invalid_argument(
            "MolecularScf: invalid (n_elec, spin) for unrestricted SCF");
      }
      opts.xc_options.spin_polarized = true;
    } else {
      if (spin != 0) {
        throw std::invalid_argument("MolecularScf: restricted SCF requires spin == 0");
      }
      if (n_elec < 0 || (n_elec % 2) != 0) {
        throw std::invalid_argument(
            "MolecularScf: closed-shell n_elec must be non-negative and even");
      }
    }
    if (!ext_grid_coords.empty() || !ext_grid_weights.empty()) {
      if (ext_grid_coords.size() != ext_grid_weights.size() * 3) {
        throw std::invalid_argument(
            "MolecularScf: external_grid_coords size must be 3 * external_grid_weights.size()");
      }
    }
    if ((opts.jk_mode == JKMode::RI || opts.jk_mode == JKMode::RISnK) && !aux_basis.has_value()) {
      throw std::invalid_argument(
          "MolecularScf: aux_basis is required when jk_mode is JKMode::RI or JKMode::RISnK");
    }

    is_dft_flag = !is_hf_functional_name(opts.xc);
    exx_fraction = is_dft_flag ? xc_exact_exchange_fraction(opts.xc) : 1.0;
    rsh_ = is_dft_flag ? xc_rsh_coefficients(opts.xc) : RshCoefficients{};
    if (rsh_.is_rsh() && (opts.jk_mode == JKMode::SnK || opts.jk_mode == JKMode::RISnK)) {
      throw NotImplemented(
          "MolecularScf: semi-numerical exchange (JKMode::SnK / JKMode::RISnK) is not "
          "implemented for range-separated hybrid functionals");
    }
    opts.scf.exact_exchange_fraction = rsh_.is_rsh() ? 1.0 : exx_fraction;

    dispersion_enabled = parse_dispersion_option(opts.dispersion);
    if (dispersion_enabled) {
      (void)dftd3::get_d3bj_params(opts.xc);  // Validate functional support eagerly
    }

    const Backend be = opts.scf.backend;
    opts.grid.backend = be;
    opts.xc_options.backend = be;
    opts.snk_options.backend = be;
  }

  void ensure_setup() {
    if (initialized) return;
    const Backend be = opts.scf.backend;
    if (!backend_available(be)) {
      throw std::runtime_error(std::string("MolecularScf::setup: backend '") + to_string(be) +
                               "' is not available");
    }

    sync_gpu_if_cuda(be);
    const auto t_setup_start = SteadyClock::now();

    // 1. One-electron matrices S, T, V_nuc and classical nuclear repulsion E_nuc
    const auto t_1e_start = SteadyClock::now();
    e_nuc = nuclear_repulsion_energy(atoms);
    S_mat = overlap_matrix(basis, be);
    std::vector<double> T_mat = kinetic_matrix(basis, be);
    std::vector<double> V_mat = nuclear_attraction_matrix(basis, atoms, be);
    const std::size_t n2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
    H_core_mat.resize(n2, 0.0);
    for (std::size_t i = 0; i < n2; ++i) {
      H_core_mat[i] = T_mat[i] + V_mat[i];
    }
    sync_gpu_if_cuda(be);
    const auto t_1e_end = SteadyClock::now();
    one_elec_time_s = elapsed_seconds(t_1e_start, t_1e_end);

    // 2. Molecular grid & persistent XcEngine (constructed once!)
    const bool need_k = (exx_fraction != 0.0) || rsh_.is_rsh();
    const bool need_snk =
        need_k && (opts.jk_mode == JKMode::SnK || opts.jk_mode == JKMode::RISnK);
    const bool need_grid = is_dft_flag || need_snk;

    if (need_grid) {
      const auto t_grid_start = SteadyClock::now();
      if (!ext_grid_weights.empty()) {
        grid = MolecularGrid::from_points(ext_grid_coords, ext_grid_weights, basis,
                                          opts.grid.max_batch_points, opts.grid.ao_cutoff);
      } else {
        grid = MolecularGrid(xc_atoms, basis, opts.grid);
      }
      grid_npoints = grid.npoints();
      grid_nbatches = grid.nbatches();
      active_ao_ratio = compute_active_ao_ratio(grid);

      if (is_dft_flag) {
        xc_engine = std::make_unique<XcEngine>(basis, grid, opts.xc, opts.xc_options);
      }
      sync_gpu_if_cuda(be);
      const auto t_grid_end = SteadyClock::now();
      grid_setup_time_s = elapsed_seconds(t_grid_start, t_grid_end);
    }

    // 3. Persistent J/K Engines (`DirectJKEngine`, `RIJKEngine`, `SemiNumericalK`) constructed once!
    const auto t_jk_start = SteadyClock::now();
    if (opts.jk_mode == JKMode::Direct) {
      direct_jk = std::make_unique<DirectJKEngine>(basis, opts.prim_cutoff);
    } else if (opts.jk_mode == JKMode::RI) {
      ri_jk = std::make_unique<RIJKEngine>(basis, *aux_basis, be, rsh_.is_rsh() ? rsh_.omega : 0.0);
    } else if (opts.jk_mode == JKMode::SnK) {
      direct_jk = std::make_unique<DirectJKEngine>(basis, opts.prim_cutoff);
      if (need_k) {
        snk_engine = std::make_unique<SemiNumericalK>(basis, grid, opts.snk_options);
      }
    } else if (opts.jk_mode == JKMode::RISnK) {
      ri_jk = std::make_unique<RIJKEngine>(basis, *aux_basis, be);
      if (need_k) {
        snk_engine = std::make_unique<SemiNumericalK>(basis, grid, opts.snk_options);
      }
    }
    sync_gpu_if_cuda(be);
    const auto t_jk_end = SteadyClock::now();
    jk_setup_time_s = elapsed_seconds(t_jk_start, t_jk_end);

    if (dispersion_enabled) {
      const auto t_d3_start = SteadyClock::now();
      d3_cached = dftd3::compute_d3bj(atoms, opts.xc, /*compute_grad=*/true);
      const auto t_d3_end = SteadyClock::now();
      dispersion_setup_time_s = elapsed_seconds(t_d3_start, t_d3_end);
    }

    setup_time_s = elapsed_seconds(t_setup_start, SteadyClock::now());

    initialized = true;
  }
};

MolecularScf::MolecularScf(Basis basis, std::vector<Atom> atoms, int n_elec, WorkflowOptions opts,
                           std::optional<Basis> aux_basis,
                           std::span<const double> external_grid_coords,
                           std::span<const double> external_grid_weights)
    : impl_(std::make_unique<Impl>(std::move(basis), std::move(atoms), n_elec, std::move(opts),
                                   std::move(aux_basis), external_grid_coords,
                                   external_grid_weights)) {}

MolecularScf::~MolecularScf() = default;
MolecularScf::MolecularScf(MolecularScf&&) noexcept = default;
MolecularScf& MolecularScf::operator=(MolecularScf&&) noexcept = default;

int MolecularScf::nao() const noexcept { return impl_->basis.nao(); }
int MolecularScf::natom() const noexcept { return static_cast<int>(impl_->atoms.size()); }
int MolecularScf::n_elec() const noexcept { return impl_->n_elec; }
bool MolecularScf::is_dft() const noexcept { return impl_->is_dft_flag; }
double MolecularScf::exact_exchange_fraction() const noexcept { return impl_->exx_fraction; }
const RshCoefficients& MolecularScf::rsh_coefficients() const noexcept { return impl_->rsh_; }
const WorkflowOptions& MolecularScf::options() const noexcept { return impl_->opts; }

void MolecularScf::set_xc_grid_response(bool enabled) noexcept {
  impl_->opts.xc_options.grid_response = enabled;
  if (impl_->xc_engine != nullptr) {
    impl_->xc_engine->set_grid_response(enabled);
  }
}

void MolecularScf::setup() { impl_->ensure_setup(); }

WorkflowResult MolecularScf::run(std::span<const double> initial_dm_host) {
  impl_->ensure_setup();

  const int n_ao = impl_->basis.nao();
  const std::size_t n2 = static_cast<std::size_t>(n_ao) * n_ao;
  const bool is_unrestricted = impl_->opts.unrestricted;
  if (is_unrestricted) {
    if (!initial_dm_host.empty() && initial_dm_host.size() != n2 &&
        initial_dm_host.size() != 2 * n2) {
      throw std::invalid_argument(
          "MolecularScf::run: unrestricted initial_dm_host size must equal 2 * nao * nao or nao * nao");
    }
  } else {
    if (!initial_dm_host.empty() && initial_dm_host.size() != n2) {
      throw std::invalid_argument("MolecularScf::run: initial_dm_host size must equal nao * nao");
    }
  }

  const Backend be = impl_->opts.scf.backend;

  ScfOptions run_scf_opts = impl_->opts.scf;
  run_scf_opts.unrestricted = is_unrestricted;
  run_scf_opts.spin = impl_->opts.spin;
  run_scf_opts.exact_exchange_fraction =
      impl_->rsh_.is_rsh() ? 1.0 : impl_->exx_fraction;
  // Dense RI-JK contractions do not rely on |Delta D| quartet screening, so evaluating
  // J(D) and K(D) directly avoids delta-density staging and saves the final full-reset step.
  if (impl_->opts.jk_mode == JKMode::RI || impl_->opts.jk_mode == JKMode::RISnK) {
    run_scf_opts.incremental_fock = false;
  }

  FockCallbacks cb;

  // 1. Wire XC Engine (`eval_vxc(_uks)_device` on GPU, `eval_vxc(_uks)` on Host)
  if (impl_->is_dft_flag) {
    XcEngine* xc_ptr = impl_->xc_engine.get();
    if (is_unrestricted) {
      if (be == Backend::Cuda) {
        cb.vxc_device_resident = true;
        cb.build_vxc_uks = [xc_ptr](const double* d_da, const double* d_db, Precision /*prec*/,
                                    double* d_va, double* d_vb, void* raw_stream) -> double {
          return xc_ptr->eval_vxc_uks_device(d_da, d_db, d_va, d_vb, raw_stream);
        };
      } else {
        cb.vxc_device_resident = false;
        cb.build_vxc_uks = [xc_ptr](const double* h_da, const double* h_db, Precision /*prec*/,
                                    double* h_va, double* h_vb, void* /*stream*/) -> double {
          return xc_ptr->eval_vxc_uks(h_da, h_db, h_va, h_vb, nullptr, nullptr);
        };
      }
    } else {
      if (be == Backend::Cuda) {
        cb.vxc_device_resident = true;
        cb.build_vxc = [xc_ptr](const double* d_dm, Precision /*prec*/, double* d_vxc,
                                void* raw_stream) -> double {
          return xc_ptr->eval_vxc_device(d_dm, d_vxc, raw_stream);
        };
      } else {
        cb.vxc_device_resident = false;
        cb.build_vxc = [xc_ptr](const double* h_dm, Precision /*prec*/, double* h_vxc,
                                void* /*stream*/) -> double {
          return xc_ptr->eval_vxc(h_dm, h_vxc, nullptr);
        };
      }
    }
  }

  // 2. Wire Two-Electron J/K Engine (100% Zero-Copy Device-Resident on Backend::Cuda!)
  const RshCoefficients rsh = impl_->rsh_;
  if (impl_->opts.jk_mode == JKMode::Direct) {
    DirectJKEngine* djk = impl_->direct_jk.get();
    const double s_cut = impl_->opts.schwarz_cutoff;
    const double fp32_thr = impl_->opts.fp32_threshold;
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.device_resident = true;
      cb.build_jk = [djk, s_cut, fp32_thr, rsh](const double* d_dm, bool /*is_inc*/, Precision prec,
                                                double* d_vj, double* d_vk, void* raw_stream) {
        JKOptions jk_opts;
        jk_opts.backend = Backend::Cuda;
        jk_opts.precision = prec;
        jk_opts.schwarz_cutoff = s_cut;
        jk_opts.fp32_threshold = fp32_thr;
        jk_opts.compute_j = (d_vj != nullptr);
        jk_opts.compute_k = (d_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = true;
        if (rsh.is_rsh()) {
          jk_opts.omega = rsh.omega;
          jk_opts.rsh_mode = true;
          jk_opts.k_sr = rsh.c_sr();
          jk_opts.k_lr = rsh.c_lr();
        }
        djk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
      };
      cb.build_jk_uks = [djk, s_cut, fp32_thr, rsh](const double* d_da, const double* d_db,
                                                    bool /*is_inc*/, Precision prec, double* d_vj,
                                                    double* d_vka, double* d_vkb,
                                                    void* raw_stream) {
        JKOptions jk_opts;
        jk_opts.backend = Backend::Cuda;
        jk_opts.precision = prec;
        jk_opts.schwarz_cutoff = s_cut;
        jk_opts.fp32_threshold = fp32_thr;
        jk_opts.compute_j = (d_vj != nullptr);
        jk_opts.compute_k = (d_vka != nullptr) || (d_vkb != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = true;
        if (rsh.is_rsh()) {
          jk_opts.omega = rsh.omega;
          jk_opts.rsh_mode = true;
          jk_opts.k_sr = rsh.c_sr();
          jk_opts.k_lr = rsh.c_lr();
        }
        djk->compute_jk_uhf_device(d_da, d_db, d_vj, d_vka, d_vkb, jk_opts, raw_stream);
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [djk, be, s_cut, fp32_thr, rsh](const double* h_dm, bool /*is_inc*/,
                                                    Precision prec, double* h_vj, double* h_vk,
                                                    void* /*stream*/) {
        JKOptions jk_opts;
        jk_opts.backend = be;
        jk_opts.precision = prec;
        jk_opts.schwarz_cutoff = s_cut;
        jk_opts.fp32_threshold = fp32_thr;
        jk_opts.compute_j = (h_vj != nullptr);
        jk_opts.compute_k = (h_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = true;
        if (rsh.is_rsh()) {
          jk_opts.omega = rsh.omega;
          jk_opts.rsh_mode = true;
          jk_opts.k_sr = rsh.c_sr();
          jk_opts.k_lr = rsh.c_lr();
        }
        djk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
      };
      cb.build_jk_uks = [djk, be, s_cut, fp32_thr, rsh](const double* h_da, const double* h_db,
                                                        bool /*is_inc*/, Precision prec,
                                                        double* h_vj, double* h_vka, double* h_vkb,
                                                        void* /*stream*/) {
        JKOptions jk_opts;
        jk_opts.backend = be;
        jk_opts.precision = prec;
        jk_opts.schwarz_cutoff = s_cut;
        jk_opts.fp32_threshold = fp32_thr;
        jk_opts.compute_j = (h_vj != nullptr);
        jk_opts.compute_k = (h_vka != nullptr) || (h_vkb != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = true;
        if (rsh.is_rsh()) {
          jk_opts.omega = rsh.omega;
          jk_opts.rsh_mode = true;
          jk_opts.k_sr = rsh.c_sr();
          jk_opts.k_lr = rsh.c_lr();
        }
        djk->compute_jk_uhf(h_da, h_db, h_vj, h_vka, h_vkb, jk_opts);
      };
    }
  } else if (impl_->opts.jk_mode == JKMode::RI) {
    RIJKEngine* rijk = impl_->ri_jk.get();
    const GemmMode target_gemm = run_scf_opts.gemm_mode;
    const GemmMode early_gemm = run_scf_opts.early_gemm_mode;
    const int oz_slices = run_scf_opts.ozaki_slices;
    auto make_ri_opts = [target_gemm, early_gemm, oz_slices, rsh](
                            Backend b, Precision prec, bool do_j, bool do_k) {
      GemmMode k_mode = target_gemm;
      if (prec == Precision::FP32) {
        k_mode = GemmMode::FP32;
      } else if (prec == Precision::Mixed) {
        k_mode = early_gemm;
      }
      JKOptions jk_opts;
      jk_opts.backend = b;
      jk_opts.precision = (k_mode == GemmMode::FP32) ? Precision::FP32 : Precision::FP64;
      jk_opts.gemm_mode = k_mode;
      jk_opts.ozaki_slices = oz_slices;
      jk_opts.compute_j = do_j;
      jk_opts.compute_k = do_k;
      jk_opts.k_scale = 1.0;
      jk_opts.symmetrize_dm = true;
      if (rsh.is_rsh()) {
        jk_opts.omega = rsh.omega;
        jk_opts.rsh_mode = true;
        jk_opts.k_sr = rsh.c_sr();
        jk_opts.k_lr = rsh.c_lr();
      }
      return jk_opts;
    };
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.device_resident = true;
      cb.build_jk = [rijk, make_ri_opts](const double* d_dm, bool /*is_inc*/, Precision prec,
                                         double* d_vj, double* d_vk, void* raw_stream) {
        const JKOptions jk_opts =
            make_ri_opts(Backend::Cuda, prec, d_vj != nullptr, d_vk != nullptr);
        rijk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
      };
      cb.build_jk_mo = [rijk, make_ri_opts](const double* d_dm, const double* d_c_occ, int n_occ,
                                            bool /*is_inc*/, Precision prec, double* d_vj,
                                            double* d_vk, void* raw_stream) {
        const JKOptions jk_opts =
            make_ri_opts(Backend::Cuda, prec, d_vj != nullptr, d_vk != nullptr);
        rijk->compute_jk_mo_device(d_dm, d_c_occ, n_occ, 2.0, d_vj, d_vk, jk_opts, raw_stream);
      };
      cb.build_jk_uks = [rijk, make_ri_opts](const double* d_da, const double* d_db,
                                             bool /*is_inc*/, Precision prec, double* d_vj,
                                             double* d_vka, double* d_vkb, void* raw_stream) {
        const JKOptions jk_opts = make_ri_opts(
            Backend::Cuda, prec, d_vj != nullptr, (d_vka != nullptr || d_vkb != nullptr));
        rijk->compute_jk_uks_mo_device(d_da, d_db, nullptr, -1, nullptr, -1, d_vj, d_vka, d_vkb,
                                       jk_opts, raw_stream);
      };
      cb.build_jk_uks_mo = [rijk, make_ri_opts](
                               const double* d_da, const double* d_db, const double* d_ca,
                               int n_alpha, const double* d_cb, int n_beta, bool /*is_inc*/,
                               Precision prec, double* d_vj, double* d_vka, double* d_vkb,
                               void* raw_stream) {
        const JKOptions jk_opts = make_ri_opts(
            Backend::Cuda, prec, d_vj != nullptr, (d_vka != nullptr || d_vkb != nullptr));
        rijk->compute_jk_uks_mo_device(d_da, d_db, d_ca, n_alpha, d_cb, n_beta, d_vj, d_vka, d_vkb,
                                       jk_opts, raw_stream);
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [rijk, be, make_ri_opts](const double* h_dm, bool /*is_inc*/, Precision prec,
                                             double* h_vj, double* h_vk, void* /*stream*/) {
        const JKOptions jk_opts = make_ri_opts(be, prec, h_vj != nullptr, h_vk != nullptr);
        rijk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
      };
      cb.build_jk_mo = [rijk, be, make_ri_opts](const double* h_dm, const double* h_c_occ,
                                                int n_occ, bool /*is_inc*/, Precision prec,
                                                double* h_vj, double* h_vk, void* /*stream*/) {
        const JKOptions jk_opts = make_ri_opts(be, prec, h_vj != nullptr, h_vk != nullptr);
        rijk->compute_jk_mo(h_dm, h_c_occ, n_occ, 2.0, h_vj, h_vk, jk_opts);
      };
      cb.build_jk_uks = [rijk, be, make_ri_opts](const double* h_da, const double* h_db,
                                                 bool /*is_inc*/, Precision prec, double* h_vj,
                                                 double* h_vka, double* h_vkb, void* /*stream*/) {
        const JKOptions jk_opts =
            make_ri_opts(be, prec, h_vj != nullptr, (h_vka != nullptr || h_vkb != nullptr));
        rijk->compute_jk_uks_mo(h_da, h_db, nullptr, -1, nullptr, -1, h_vj, h_vka, h_vkb, jk_opts);
      };
      cb.build_jk_uks_mo = [rijk, be, make_ri_opts](
                                   const double* h_da, const double* h_db, const double* h_ca,
                                   int n_alpha, const double* h_cb, int n_beta, bool /*is_inc*/,
                                   Precision prec, double* h_vj, double* h_vka, double* h_vkb,
                                   void* /*stream*/) {
        const JKOptions jk_opts =
            make_ri_opts(be, prec, h_vj != nullptr, (h_vka != nullptr || h_vkb != nullptr));
        rijk->compute_jk_uks_mo(h_da, h_db, h_ca, n_alpha, h_cb, n_beta, h_vj, h_vka, h_vkb,
                                jk_opts);
      };
    }
  } else if (impl_->opts.jk_mode == JKMode::SnK) {
    DirectJKEngine* djk = impl_->direct_jk.get();
    SemiNumericalK* snk = impl_->snk_engine.get();
    const double s_cut = impl_->opts.schwarz_cutoff;
    const double fp32_thr = impl_->opts.fp32_threshold;
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.device_resident = true;
      cb.build_jk = [djk, snk, s_cut, fp32_thr](const double* d_dm, bool /*is_inc*/, Precision prec,
                                                double* d_vj, double* d_vk, void* raw_stream) {
        if (d_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = Backend::Cuda;
          j_opts.precision = prec;
          j_opts.schwarz_cutoff = s_cut;
          j_opts.fp32_threshold = fp32_thr;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          djk->compute_jk_device(d_dm, d_vj, nullptr, j_opts, raw_stream);
        }
        if (d_vk != nullptr && snk != nullptr) {
          snk->compute_k_device(d_dm, d_vk, raw_stream);
        }
      };
      cb.build_jk_uks = [djk, snk, s_cut, fp32_thr](const double* d_da, const double* d_db,
                                                    bool /*is_inc*/, Precision prec, double* d_vj,
                                                    double* d_vka, double* d_vkb,
                                                    void* raw_stream) {
        if (d_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = Backend::Cuda;
          j_opts.precision = prec;
          j_opts.schwarz_cutoff = s_cut;
          j_opts.fp32_threshold = fp32_thr;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          djk->compute_jk_uhf_device(d_da, d_db, d_vj, nullptr, nullptr, j_opts, raw_stream);
        }
        if (snk != nullptr) {
          if (d_vka != nullptr) snk->compute_k_device(d_da, d_vka, raw_stream);
          if (d_vkb != nullptr) snk->compute_k_device(d_db, d_vkb, raw_stream);
        }
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [djk, snk, be, s_cut, fp32_thr](const double* h_dm, bool /*is_inc*/,
                                                    Precision prec, double* h_vj, double* h_vk,
                                                    void* /*stream*/) {
        if (h_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = be;
          j_opts.precision = prec;
          j_opts.schwarz_cutoff = s_cut;
          j_opts.fp32_threshold = fp32_thr;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          djk->compute_jk(h_dm, h_vj, nullptr, j_opts);
        }
        if (h_vk != nullptr && snk != nullptr) {
          snk->compute_k(h_dm, h_vk);
        }
      };
      cb.build_jk_uks = [djk, snk, be, s_cut, fp32_thr](const double* h_da, const double* h_db,
                                                        bool /*is_inc*/, Precision prec,
                                                        double* h_vj, double* h_vka, double* h_vkb,
                                                        void* /*stream*/) {
        if (h_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = be;
          j_opts.precision = prec;
          j_opts.schwarz_cutoff = s_cut;
          j_opts.fp32_threshold = fp32_thr;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          djk->compute_jk_uhf(h_da, h_db, h_vj, nullptr, nullptr, j_opts);
        }
        if (snk != nullptr) {
          if (h_vka != nullptr) snk->compute_k(h_da, h_vka);
          if (h_vkb != nullptr) snk->compute_k(h_db, h_vkb);
        }
      };
    }
  } else if (impl_->opts.jk_mode == JKMode::RISnK) {
    RIJKEngine* rijk = impl_->ri_jk.get();
    SemiNumericalK* snk = impl_->snk_engine.get();
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.device_resident = true;
      cb.build_jk = [rijk, snk](const double* d_dm, bool /*is_inc*/, Precision /*prec*/,
                                double* d_vj, double* d_vk, void* raw_stream) {
        if (d_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = Backend::Cuda;
          j_opts.precision = Precision::FP64;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          rijk->compute_jk_device(d_dm, d_vj, nullptr, j_opts, raw_stream);
        }
        if (d_vk != nullptr && snk != nullptr) {
          snk->compute_k_device(d_dm, d_vk, raw_stream);
        }
      };
      cb.build_jk_uks = [rijk, snk, n2](const double* d_da, const double* d_db, bool /*is_inc*/,
                                        Precision /*prec*/, double* d_vj, double* d_vka,
                                        double* d_vkb, void* raw_stream) {
        auto s = static_cast<StreamHandle>(raw_stream);
        if (d_vj != nullptr) {
          PooledBuffer<double> d_dtot(default_device_pool(), n2, s);
          copy_d2d_async(d_dtot.data(), d_da, n2 * sizeof(double), s);
          axpby_device(n2, 1.0, d_db, 1.0, d_dtot.data(), s);
          JKOptions j_opts;
          j_opts.backend = Backend::Cuda;
          j_opts.precision = Precision::FP64;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          rijk->compute_jk_device(d_dtot.data(), d_vj, nullptr, j_opts, raw_stream);
        }
        if (snk != nullptr) {
          if (d_vka != nullptr) snk->compute_k_device(d_da, d_vka, raw_stream);
          if (d_vkb != nullptr) snk->compute_k_device(d_db, d_vkb, raw_stream);
        }
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [rijk, snk, be](const double* h_dm, bool /*is_inc*/, Precision /*prec*/,
                                    double* h_vj, double* h_vk, void* /*stream*/) {
        if (h_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = be;
          j_opts.precision = Precision::FP64;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          rijk->compute_jk(h_dm, h_vj, nullptr, j_opts);
        }
        if (h_vk != nullptr && snk != nullptr) {
          snk->compute_k(h_dm, h_vk);
        }
      };
      cb.build_jk_uks = [rijk, snk, be, n2](const double* h_da, const double* h_db, bool /*is_inc*/,
                                            Precision /*prec*/, double* h_vj, double* h_vka,
                                            double* h_vkb, void* /*stream*/) {
        if (h_vj != nullptr) {
          std::vector<double> h_dtot(n2, 0.0);
          for (std::size_t i = 0; i < n2; ++i) h_dtot[i] = h_da[i] + h_db[i];
          JKOptions j_opts;
          j_opts.backend = be;
          j_opts.precision = Precision::FP64;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          j_opts.symmetrize_dm = true;
          rijk->compute_jk(h_dtot.data(), h_vj, nullptr, j_opts);
        }
        if (snk != nullptr) {
          if (h_vka != nullptr) snk->compute_k(h_da, h_vka);
          if (h_vkb != nullptr) snk->compute_k(h_db, h_vkb);
        }
      };
    }
  }

  ScfDriver driver(run_scf_opts);

  sync_gpu_if_cuda(be);
  const auto t_scf_start = SteadyClock::now();
  ScfResult scf_res;
  if (is_unrestricted) {
    if (initial_dm_host.empty()) {
      scf_res = driver.run(n_ao, impl_->n_elec, impl_->e_nuc, impl_->S_mat.data(),
                           impl_->H_core_mat.data(), cb, nullptr, nullptr);
    } else if (initial_dm_host.size() == 2 * n2) {
      scf_res = driver.run(n_ao, impl_->n_elec, impl_->e_nuc, impl_->S_mat.data(),
                           impl_->H_core_mat.data(), cb, initial_dm_host.data(),
                           initial_dm_host.data() + n2);
    } else {
      scf_res = driver.run(n_ao, impl_->n_elec, impl_->e_nuc, impl_->S_mat.data(),
                           impl_->H_core_mat.data(), cb, initial_dm_host.data());
    }
  } else {
    scf_res = driver.run(n_ao, impl_->n_elec, impl_->e_nuc, impl_->S_mat.data(),
                         impl_->H_core_mat.data(), cb,
                         initial_dm_host.empty() ? nullptr : initial_dm_host.data());
  }
  sync_gpu_if_cuda(be);
  if (impl_->ri_jk != nullptr) {
    impl_->ri_jk->release_workspaces();
  }
  const auto t_scf_end = SteadyClock::now();

  const double e_disp =
      (impl_->dispersion_enabled && impl_->d3_cached.has_value()) ? impl_->d3_cached->energy : 0.0;
  scf_res.dispersion_energy = e_disp;
  scf_res.total_energy += e_disp;

  impl_->last_scf = scf_res;

  WorkflowResult out;
  out.scf = std::move(scf_res);
  out.setup_time_s = impl_->setup_time_s;
  out.one_electron_time_s = impl_->one_elec_time_s;
  out.jk_setup_time_s = impl_->jk_setup_time_s;
  out.grid_setup_time_s = impl_->grid_setup_time_s;
  out.scf_time_s = elapsed_seconds(t_scf_start, t_scf_end);
  out.grid_npoints = impl_->grid_npoints;
  out.grid_nbatches = impl_->grid_nbatches;
  out.active_ao_ratio = impl_->active_ao_ratio;
  out.dispersion_energy = e_disp;
  out.overlap_matrix = impl_->S_mat;
  out.hcore_matrix = impl_->H_core_mat;
  return out;
}

GradientResult MolecularScf::compute_gradient(const ScfResult& scf_res) {
  impl_->ensure_setup();

  const int n_atom = static_cast<int>(impl_->atoms.size());
  const std::size_t ncomp = static_cast<std::size_t>(n_atom) * 3;
  const std::size_t n2 = static_cast<std::size_t>(impl_->basis.nao()) * impl_->basis.nao();
  if (scf_res.density_matrix.size() != n2 || scf_res.energy_weighted_dm.size() != n2) {
    throw std::invalid_argument(
        "MolecularScf::compute_gradient: density_matrix and energy_weighted_dm must have size nao^2");
  }
  if (scf_res.unrestricted &&
      (scf_res.density_alpha.size() != n2 || scf_res.density_beta.size() != n2)) {
    throw std::invalid_argument(
        "MolecularScf::compute_gradient: unrestricted density_alpha and density_beta must have size nao^2");
  }

  const Backend be = impl_->opts.scf.backend;
  const double grad_schwarz = std::min(impl_->opts.schwarz_cutoff, 1e-13);

  GradientResult gres;
  gres.natom = n_atom;
  gres.xc_grad.assign(ncomp, 0.0);
  gres.dispersion_grad.assign(ncomp, 0.0);

  sync_gpu_if_cuda(be);
  const auto t_tot_start = SteadyClock::now();

  gres.nuclear_repulsion_grad = nuclear_repulsion_grad(impl_->atoms);

  const auto t_1e_start = SteadyClock::now();
  gres.one_electron_grad =
      one_electron_grad(impl_->basis, impl_->atoms, scf_res.density_matrix.data(),
                        scf_res.energy_weighted_dm.data(), be);
  sync_gpu_if_cuda(be);
  const auto t_1e_end = SteadyClock::now();
  gres.one_electron_time_s = elapsed_seconds(t_1e_start, t_1e_end);

  const auto t_2e_start = SteadyClock::now();
  if (impl_->opts.jk_mode == JKMode::RI) {
    if (impl_->ri_jk == nullptr) {
      throw std::runtime_error("MolecularScf::compute_gradient: RIJKEngine is not initialized");
    }
    const int nao = impl_->basis.nao();
    auto slice_occ = [&](const std::vector<double>& mo, int nmo, int n_occ) {
      std::vector<double> out;
      if (mo.empty() || nmo < n_occ || n_occ <= 0 || nao <= 0) return out;
      out.resize(static_cast<std::size_t>(nao) * n_occ);
      for (int r = 0; r < nao; ++r) {
        const double* src = mo.data() + static_cast<std::size_t>(r) * nmo;
        double* dst = out.data() + static_cast<std::size_t>(r) * n_occ;
        std::copy(src, src + n_occ, dst);
      }
      return out;
    };
    const double j_frac = 1.0;
    const double k_sr = impl_->rsh_.is_rsh() ? impl_->rsh_.c_sr() : impl_->exx_fraction;
    const double k_lr = impl_->rsh_.is_rsh() ? impl_->rsh_.c_lr() : impl_->exx_fraction;
    const double omega = impl_->rsh_.is_rsh() ? impl_->rsh_.omega : 0.0;
    if (scf_res.unrestricted) {
      const int n_a = scf_res.n_alpha;
      const int n_b = scf_res.n_beta;
      const std::vector<double> c_a = slice_occ(scf_res.mo_coefficients, scf_res.nmo, n_a);
      const std::vector<double> c_b = slice_occ(scf_res.mo_coefficients_beta, scf_res.nmo, n_b);
      const int n_a_arg =
          !c_a.empty() ? n_a : ((n_a == 0 && (scf_res.nmo > 0 || scf_res.converged)) ? 0 : -1);
      const int n_b_arg =
          !c_b.empty() ? n_b : ((n_b == 0 && (scf_res.nmo > 0 || scf_res.converged)) ? 0 : -1);
      gres.two_electron_grad = impl_->ri_jk->compute_ri_jk_grad(
          impl_->atoms, scf_res.density_alpha.data(), c_a.empty() ? nullptr : c_a.data(), n_a_arg,
          scf_res.density_beta.data(), c_b.empty() ? nullptr : c_b.data(), n_b_arg,
          /*unrestricted=*/true, j_frac, k_sr, k_lr, omega, be);
    } else {
      const int n_occ = (scf_res.n_alpha > 0) ? scf_res.n_alpha : (impl_->n_elec / 2);
      const std::vector<double> c_a = slice_occ(scf_res.mo_coefficients, scf_res.nmo, n_occ);
      const int n_a_arg =
          !c_a.empty() ? n_occ : ((n_occ == 0 && (scf_res.nmo > 0 || scf_res.converged)) ? 0 : -1);
      gres.two_electron_grad = impl_->ri_jk->compute_ri_jk_grad(
          impl_->atoms, scf_res.density_matrix.data(), c_a.empty() ? nullptr : c_a.data(), n_a_arg,
          nullptr, nullptr, 0, /*unrestricted=*/false, j_frac, k_sr, k_lr, omega, be);
    }
  } else if (impl_->rsh_.is_rsh()) {
    if (scf_res.unrestricted) {
      gres.two_electron_grad = jk_nuclear_grad_uhf_rsh(
          impl_->basis, impl_->atoms, scf_res.density_alpha.data(), scf_res.density_beta.data(),
          impl_->rsh_.c_sr(), impl_->rsh_.c_lr(), impl_->rsh_.omega, be, grad_schwarz);
    } else {
      gres.two_electron_grad = jk_nuclear_grad_rsh(
          impl_->basis, impl_->atoms, scf_res.density_matrix.data(), impl_->rsh_.c_sr(),
          impl_->rsh_.c_lr(), impl_->rsh_.omega, be, grad_schwarz);
    }
  } else if (scf_res.unrestricted) {
    gres.two_electron_grad =
        jk_nuclear_grad_uhf(impl_->basis, impl_->atoms, scf_res.density_alpha.data(),
                            scf_res.density_beta.data(), impl_->exx_fraction, be, grad_schwarz);
  } else {
    gres.two_electron_grad =
        jk_nuclear_grad(impl_->basis, impl_->atoms, scf_res.density_matrix.data(),
                        impl_->exx_fraction, be, grad_schwarz);
  }
  sync_gpu_if_cuda(be);
  const auto t_2e_end = SteadyClock::now();
  gres.two_electron_time_s = elapsed_seconds(t_2e_start, t_2e_end);

  if (impl_->is_dft_flag && impl_->xc_engine != nullptr) {
    const auto t_xc_start = SteadyClock::now();
    if (scf_res.unrestricted) {
      gres.xc_grad = impl_->xc_engine->nuclear_grad_uks(impl_->xc_atoms, scf_res.density_alpha,
                                                        scf_res.density_beta);
    } else {
      gres.xc_grad = impl_->xc_engine->nuclear_grad(impl_->xc_atoms, scf_res.density_matrix);
    }
    sync_gpu_if_cuda(be);
    const auto t_xc_end = SteadyClock::now();
    gres.xc_time_s = elapsed_seconds(t_xc_start, t_xc_end);
  }

  if (impl_->dispersion_enabled) {
    const auto t_d3_start = SteadyClock::now();
    if (impl_->d3_cached.has_value() && impl_->d3_cached->gradient.size() == ncomp) {
      gres.dispersion_grad = impl_->d3_cached->gradient;
    } else {
      gres.dispersion_grad = dftd3::compute_d3bj(impl_->atoms, impl_->opts.xc, true).gradient;
    }
    const auto t_d3_end = SteadyClock::now();
    gres.dispersion_time_s = elapsed_seconds(t_d3_start, t_d3_end);
  }

  const auto t_tot_end = SteadyClock::now();
  gres.total_time_s = elapsed_seconds(t_tot_start, t_tot_end);

  gres.total_grad.assign(ncomp, 0.0);
  for (std::size_t i = 0; i < ncomp; ++i) {
    gres.total_grad[i] = gres.nuclear_repulsion_grad[i] + gres.one_electron_grad[i] +
                         gres.two_electron_grad[i] + gres.xc_grad[i] + gres.dispersion_grad[i];
  }
  return gres;
}

GradientResult MolecularScf::compute_gradient() {
  if (!impl_->last_scf.has_value()) {
    throw std::runtime_error("MolecularScf::compute_gradient: run() has not been executed yet");
  }
  return compute_gradient(*impl_->last_scf);
}

std::array<double, 3> MolecularScf::compute_dipole(const ScfResult& scf_res,
                                                   std::array<double, 3> origin) {
  impl_->ensure_setup();
  const std::size_t n2 = static_cast<std::size_t>(impl_->basis.nao()) * impl_->basis.nao();
  if (scf_res.density_matrix.size() != n2) {
    throw std::invalid_argument(
        "MolecularScf::compute_dipole: scf_res.density_matrix size must equal nao^2");
  }
  return molecular_dipole_moment(impl_->basis, impl_->atoms, scf_res.density_matrix.data(), origin,
                                 impl_->opts.scf.backend);
}

std::array<double, 3> MolecularScf::compute_dipole(std::array<double, 3> origin) {
  if (!impl_->last_scf.has_value()) {
    throw std::runtime_error("MolecularScf::compute_dipole: run() has not been executed yet");
  }
  return compute_dipole(*impl_->last_scf, origin);
}

ResponseResult MolecularScf::run_response(const ResponseOptions& resp_opts) {
  impl_->ensure_setup();
  if (!impl_->last_scf.has_value()) {
    throw std::runtime_error("MolecularScf::run_response: run() has not been executed yet");
  }

  const ScfResult& scf_res = *impl_->last_scf;
  if (impl_->opts.unrestricted || scf_res.unrestricted) {
    throw NotImplemented(
        "MolecularScf::run_response: unrestricted TDDFT/CIS is not implemented");
  }
  if (impl_->rsh_.is_rsh()) {
    throw NotImplemented(
        "MolecularScf::run_response: range-separated hybrid TDDFT is not implemented");
  }
  const int n_ao = scf_res.nao;
  const int n_occ = impl_->n_elec / 2;

  ResponseOptions eff_opts = resp_opts;
  eff_opts.exact_exchange_fraction = impl_->exx_fraction;
  const Backend be = eff_opts.backend;
  if (!backend_available(be)) {
    throw std::runtime_error(std::string("MolecularScf::run_response: backend '") + to_string(be) +
                             "' is not available");
  }

  sync_gpu_if_cuda(be);
  const auto t_resp_start = SteadyClock::now();

  // If SCF used SP2 purification without canonical MOs, extract canonical MOs from converged Fock.
  std::vector<double> local_mo_e;
  std::vector<double> local_mo_c;
  int n_mo = scf_res.nmo;
  const double* mo_e_ptr = scf_res.mo_energies.data();
  const double* mo_c_ptr = scf_res.mo_coefficients.data();
  if (scf_res.mo_energies.empty() || scf_res.mo_coefficients.empty()) {
    const auto ortho =
        canonical_orthogonalization(n_ao, impl_->S_mat.data(), impl_->opts.scf.lindep_tol, be);
    n_mo = ortho.nmo;
    auto diag = solve_fock_diag(n_ao, n_mo, n_occ, scf_res.fock_matrix.data(), ortho.X.data(), be,
                                eff_opts.gemm_mode, eff_opts.ozaki_slices);
    local_mo_e = std::move(diag.mo_energies);
    local_mo_c = std::move(diag.mo_coeff);
    mo_e_ptr = local_mo_e.data();
    mo_c_ptr = local_mo_c.data();
  }

  const bool need_k = (std::abs(eff_opts.exact_exchange_fraction) > 0.0);
  const bool use_ri =
      (impl_->opts.jk_mode == JKMode::RI) || (impl_->opts.jk_mode == JKMode::RISnK && !need_k);

  ResponseCallbacks cb;

  // 1. Wire Zero-Copy Non-Symmetric J/K Response Callbacks (`symmetrize_dm = false`)
  if (use_ri && impl_->ri_jk != nullptr) {
    RIJKEngine* rijk = impl_->ri_jk.get();
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      const GemmMode final_mode = eff_opts.gemm_mode;
      const int oz_slices = eff_opts.ozaki_slices;
      cb.device_resident = true;
      cb.build_jk = [rijk, final_mode, oz_slices](const double* d_dm, double* d_vj, double* d_vk,
                                                  void* raw_stream) {
        JKOptions jk_opts;
        jk_opts.backend = Backend::Cuda;
        jk_opts.precision = (final_mode == GemmMode::FP32) ? Precision::FP32 : Precision::FP64;
        jk_opts.gemm_mode = final_mode;
        jk_opts.ozaki_slices = oz_slices;
        jk_opts.compute_j = (d_vj != nullptr);
        jk_opts.compute_k = (d_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = false;
        rijk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [rijk](const double* h_dm, double* h_vj, double* h_vk, void* /*stream*/) {
        JKOptions jk_opts;
        jk_opts.backend = Backend::Host;
        jk_opts.precision = Precision::FP64;
        jk_opts.compute_j = (h_vj != nullptr);
        jk_opts.compute_k = (h_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = false;
        rijk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
      };
    }
  } else {
    if (!impl_->direct_jk) {
      impl_->direct_jk = std::make_unique<DirectJKEngine>(impl_->basis, impl_->opts.prim_cutoff);
    }
    DirectJKEngine* djk = impl_->direct_jk.get();
    const double s_cut = impl_->opts.schwarz_cutoff;
    const double fp32_thr = impl_->opts.fp32_threshold;
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.device_resident = true;
      cb.build_jk = [djk, s_cut, fp32_thr](const double* d_dm, double* d_vj, double* d_vk,
                                           void* raw_stream) {
        JKOptions jk_opts;
        jk_opts.backend = Backend::Cuda;
        jk_opts.precision = Precision::FP64;
        jk_opts.schwarz_cutoff = s_cut;
        jk_opts.fp32_threshold = fp32_thr;
        jk_opts.compute_j = (d_vj != nullptr);
        jk_opts.compute_k = (d_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = false;
        djk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [djk, s_cut, fp32_thr](const double* h_dm, double* h_vj, double* h_vk,
                                           void* /*stream*/) {
        JKOptions jk_opts;
        jk_opts.backend = Backend::Host;
        jk_opts.precision = Precision::FP64;
        jk_opts.schwarz_cutoff = s_cut;
        jk_opts.fp32_threshold = fp32_thr;
        jk_opts.compute_j = (h_vj != nullptr);
        jk_opts.compute_k = (h_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = false;
        djk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
      };
    }
  }

  // 2. Wire Zero-Copy 2nd-Derivative XC Kernel Contraction (`eval_fxc_device` / `eval_fxc`)
  std::shared_ptr<XcEngine> alt_xc;
  if (impl_->is_dft_flag) {
    XcEngine* xc_ptr = impl_->xc_engine.get();
    if (xc_ptr == nullptr || xc_ptr->active_backend() != be) {
      XcOptions xopts = impl_->opts.xc_options;
      xopts.backend = be;
      xopts.precision = Precision::FP64;
      alt_xc = std::make_shared<XcEngine>(impl_->basis, impl_->grid, impl_->opts.xc, xopts);
      xc_ptr = alt_xc.get();
    }
    xc_ptr->set_reference_density(scf_res.density_matrix);
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.fxc_device_resident = true;
      cb.build_fxc = [xc_ptr, alt_xc](const double* d_sym_dm, double* d_fxc_out, void* raw_stream) {
        (void)alt_xc;
        xc_ptr->eval_fxc_device(d_sym_dm, d_fxc_out, raw_stream);
      };
    } else
#endif
    {
      cb.fxc_device_resident = false;
      cb.build_fxc = [xc_ptr, alt_xc](const double* h_sym_dm, double* h_fxc_out, void* /*stream*/) {
        (void)alt_xc;
        xc_ptr->eval_fxc(h_sym_dm, h_fxc_out);
      };
    }
  }

  // 3. Evaluate AO electric dipole matrices & solve Block-Davidson linear response
  const auto ao_dip = dipole_matrices(impl_->basis, {0.0, 0.0, 0.0}, be);
  LinearResponseDriver drv(eff_opts);
  ResponseResult res = drv.run(n_ao, n_mo, n_occ, mo_e_ptr, mo_c_ptr, cb, &ao_dip);
  sync_gpu_if_cuda(be);
  res.wall_time_s = elapsed_seconds(t_resp_start, SteadyClock::now());
  return res;
}

ResponseResult MolecularScf::run_response() {
  ResponseOptions opts;
  opts.backend = impl_->opts.scf.backend;
  opts.gemm_mode = impl_->opts.scf.gemm_mode;
  opts.ozaki_slices = impl_->opts.scf.ozaki_slices;
  return run_response(opts);
}

VibrationalResult MolecularScf::run_hessian(const HessianOptions& hess_opts,
                                            std::span<const double> custom_masses_amu) {
  impl_->ensure_setup();
  if (!impl_->last_scf.has_value()) {
    (void)this->run();
  }

  const Backend be = hess_opts.backend;
  if (!backend_available(be)) {
    throw std::runtime_error(std::string("MolecularScf::run_hessian: backend '") + to_string(be) +
                             "' is not available");
  }

  const int n_atom = static_cast<int>(impl_->atoms.size());
  std::vector<int> shell_parent(impl_->basis.nshell(), -1);
  for (std::size_t s = 0; s < impl_->basis.nshell(); ++s) {
    const auto& sc = impl_->basis.shell(s).center;
    for (int a = 0; a < n_atom; ++a) {
      const double dx = sc[0] - impl_->atoms[static_cast<std::size_t>(a)].center[0];
      const double dy = sc[1] - impl_->atoms[static_cast<std::size_t>(a)].center[1];
      const double dz = sc[2] - impl_->atoms[static_cast<std::size_t>(a)].center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-10) {
        shell_parent[s] = a;
        break;
      }
    }
  }

  // Recover unnormalized contraction coefficients so Basis(disp_shells) matches equilibrium radial norms
  std::vector<Shell> raw_template_shells;
  raw_template_shells.reserve(impl_->basis.nshell());
  for (std::size_t s = 0; s < impl_->basis.nshell(); ++s) {
    const Shell& sh = impl_->basis.shell(s);
    Shell raw;
    raw.l = sh.l;
    raw.center = sh.center;
    raw.exponents = sh.exponents;
    raw.coefficients.resize(sh.coefficients.size());
    for (std::size_t k = 0; k < sh.exponents.size(); ++k) {
      Basis single({Shell{sh.l, {0.0, 0.0, 0.0}, {sh.exponents[k]}, {1.0}}});
      raw.coefficients[k] = sh.coefficients[k] / single.shell(0).coefficients[0];
    }
    raw_template_shells.push_back(std::move(raw));
  }

  WorkflowOptions disp_opts = impl_->opts;
  disp_opts.jk_mode = JKMode::Direct;
  disp_opts.scf.backend = be;
  disp_opts.scf.solver = DensitySolver::Diag;
  disp_opts.scf.gemm_mode = GemmMode::FP64;
  disp_opts.scf.conv_tol = std::min(disp_opts.scf.conv_tol, 1e-11);
  disp_opts.scf.diis_tol = std::min(disp_opts.scf.diis_tol, 1e-8);
  disp_opts.scf.dynamic_precision = false;
  disp_opts.scf.incremental_fock = false;
  disp_opts.grid.backend = be;
  disp_opts.xc_options.backend = be;
  disp_opts.xc_options.precision = Precision::FP64;
  disp_opts.xc_options.grid_response = !hess_opts.fixed_xc_grid;

  std::vector<double> fixed_coords;
  std::vector<double> fixed_weights;
  if (impl_->is_dft_flag && hess_opts.fixed_xc_grid) {
    if (!impl_->ext_grid_weights.empty()) {
      fixed_coords = impl_->ext_grid_coords;
      fixed_weights = impl_->ext_grid_weights;
    } else if (impl_->grid.npoints() > 0) {
      fixed_coords = impl_->grid.coords();
      fixed_weights = impl_->grid.weights();
    }
  }

  std::vector<double> eq_dm = impl_->last_scf->density_matrix;
  if (impl_->opts.unrestricted && impl_->last_scf->unrestricted) {
    eq_dm.resize(impl_->last_scf->density_alpha.size() + impl_->last_scf->density_beta.size());
    std::copy(impl_->last_scf->density_alpha.begin(), impl_->last_scf->density_alpha.end(),
              eq_dm.begin());
    std::copy(impl_->last_scf->density_beta.begin(), impl_->last_scf->density_beta.end(),
              eq_dm.begin() + static_cast<std::ptrdiff_t>(impl_->last_scf->density_alpha.size()));
  }

  auto grad_fn = [&](std::span<const Atom> disp_atoms) -> std::vector<double> {
    std::vector<Shell> disp_shells = raw_template_shells;
    for (std::size_t s = 0; s < disp_shells.size(); ++s) {
      const int pa = shell_parent[s];
      if (pa >= 0) {
        const auto upa = static_cast<std::size_t>(pa);
        for (int c = 0; c < 3; ++c) {
          const double shift = disp_atoms[upa].center[c] - impl_->atoms[upa].center[c];
          disp_shells[s].center[c] = raw_template_shells[s].center[c] + shift;
        }
      }
    }
    Basis disp_basis(std::move(disp_shells));
    std::vector<Atom> at_vec(disp_atoms.begin(), disp_atoms.end());
    MolecularScf mscf(std::move(disp_basis), std::move(at_vec), impl_->n_elec, disp_opts,
                      std::nullopt, fixed_coords, fixed_weights);
    const WorkflowResult wres = mscf.run(eq_dm);
    if (!wres.scf.converged) {
      throw std::runtime_error("MolecularScf::run_hessian: displaced SCF failed to converge");
    }
    return mscf.compute_gradient(wres.scf).total_grad;
  };

  return compute_nuclear_hessian(impl_->atoms, grad_fn, custom_masses_amu, hess_opts);
}

VibrationalResult MolecularScf::run_hessian() {
  HessianOptions opts;
  opts.backend = impl_->opts.scf.backend;
  return run_hessian(opts);
}

WorkflowResult run_scf(const Basis& basis, std::span<const Atom> atoms, int n_elec,
                       const WorkflowOptions& opts, std::span<const double> initial_dm_host,
                       const Basis* aux_basis, std::span<const double> external_grid_coords,
                       std::span<const double> external_grid_weights) {
  std::optional<Basis> opt_aux;
  if (aux_basis != nullptr) {
    opt_aux = *aux_basis;
  }
  MolecularScf runner(basis, std::vector<Atom>(atoms.begin(), atoms.end()), n_elec, opts,
                      std::move(opt_aux), external_grid_coords, external_grid_weights);
  return runner.run(initial_dm_host);
}

GradientResult run_grad(const Basis& basis, std::span<const Atom> atoms, const ScfResult& scf_res,
                        const WorkflowOptions& opts, std::span<const double> external_grid_coords,
                        std::span<const double> external_grid_weights, const Basis* aux_basis) {
  WorkflowOptions grad_opts = opts;
  std::optional<Basis> opt_aux;
  if (aux_basis != nullptr) {
    opt_aux = *aux_basis;
  } else if (grad_opts.jk_mode == JKMode::RI) {
    throw std::invalid_argument(
        "run_grad: JKMode::RI requires an explicit aux_basis (or use MolecularScf::compute_gradient())");
  }
  int n_elec = 0;
  if (scf_res.unrestricted) {
    n_elec = scf_res.n_alpha + scf_res.n_beta;
  } else if (scf_res.n_alpha > 0) {
    n_elec = 2 * scf_res.n_alpha;
  } else if (scf_res.density_matrix.size() ==
             static_cast<std::size_t>(basis.nao()) * basis.nao()) {
    const std::vector<double> S = overlap_matrix(basis, Backend::Host);
    double tr = 0.0;
    for (std::size_t i = 0; i < S.size(); ++i) tr += scf_res.density_matrix[i] * S[i];
    n_elec = std::max(0, static_cast<int>(std::llround(tr)));
  }
  if (scf_res.unrestricted) {
    grad_opts.unrestricted = true;
    grad_opts.scf.unrestricted = true;
    grad_opts.spin = scf_res.n_alpha - scf_res.n_beta;
    grad_opts.scf.spin = grad_opts.spin;
  }
  MolecularScf runner(basis, std::vector<Atom>(atoms.begin(), atoms.end()), n_elec, grad_opts,
                      std::move(opt_aux), external_grid_coords, external_grid_weights);
  return runner.compute_gradient(scf_res);
}

}  // namespace gpu4dft
