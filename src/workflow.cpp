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
#include "gpu4dft/eri.hpp"
#include "gpu4dft/linalg.hpp"

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
  return norm.empty() || norm == "hf" || norm == "rhf";
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

#if defined(GPU4DFT_HAVE_CUDA)
// Persistent GPU-resident state for 3-center RI-J and RI-K contractions.
// Uploads B_{P, uv} [naux x nao^2] (and B_perm [nao*naux x nao] when K is needed)
// ONCE during setup and contracts directly on GPU device pointers via `gemm_device`
// (supporting FP64, FP32 Tensor Core, and Ozaki INT8 Tensor Core modes).
struct GpuRIJKDeviceState {
  static constexpr int kAuxChunk = 192;

  int nao = 0;
  int naux = 0;
  bool has_k = false;
  GpuLinalgContext ctx;
  PooledBuffer<double> d_B;         // [naux * nao * nao]
  PooledBuffer<double> d_v_aux;     // [naux]
  PooledBuffer<double> d_B_chunks;  // [naux * nao * nao] packed in chunks of [nao, bc, nao]
  PooledBuffer<double> d_T_chunk;   // [nao * min(naux, kAuxChunk) * nao] bounded workspace

  GpuRIJKDeviceState(int nao_, int naux_, const std::vector<double>& h_B, bool need_k)
      : nao(nao_),
        naux(naux_),
        has_k(need_k),
        ctx(&default_device_pool()),
        d_B(ctx.pool(), static_cast<std::size_t>(naux_) * nao_ * nao_),
        d_v_aux(ctx.pool(), static_cast<std::size_t>(naux_)) {
    copy_h2d_async(d_B.data(), h_B.data(), h_B.size() * sizeof(double));
    if (has_k && naux_ > 0 && nao_ > 0) {
      const std::size_t total = static_cast<std::size_t>(nao_) * naux_ * nao_;
      std::vector<double> h_B_chunks(total, 0.0);
      for (int p0 = 0; p0 < naux_; p0 += kAuxChunk) {
        const int bc = std::min(kAuxChunk, naux_ - p0);
        double* dst_chunk =
            h_B_chunks.data() + static_cast<std::size_t>(p0) * nao_ * nao_;
        for (int pl = 0; pl < bc; ++pl) {
          const int p = p0 + pl;
          for (int u = 0; u < nao_; ++u) {
            for (int v = 0; v < nao_; ++v) {
              dst_chunk[(static_cast<std::size_t>(u) * bc + pl) * nao_ + v] =
                  h_B[(static_cast<std::size_t>(p) * nao_ + u) * nao_ + v];
            }
          }
        }
      }
      const std::size_t max_chunk_elems =
          static_cast<std::size_t>(nao_) * std::min(naux_, kAuxChunk) * nao_;
      d_B_chunks = PooledBuffer<double>(ctx.pool(), total);
      d_T_chunk = PooledBuffer<double>(ctx.pool(), max_chunk_elems);
      copy_h2d_async(d_B_chunks.data(), h_B_chunks.data(), total * sizeof(double));
    }
    synchronize_stream(nullptr);
  }

  void compute_j_device(const double* d_dm, double* d_vj, StreamHandle stream) {
    const int nao2 = nao * nao;
    gemm_device(ctx, naux, 1, nao2, 1.0, d_B.data(), nao2, false, d_dm, 1, false, 0.0,
                d_v_aux.data(), 1, GemmMode::FP64, stream);
    gemm_device(ctx, nao2, 1, naux, 1.0, d_B.data(), nao2, true, d_v_aux.data(), 1, false, 0.0,
                d_vj, 1, GemmMode::FP64, stream);
  }

  void compute_k_device(const double* d_dm, double* d_vk, GemmMode mode, int ozaki_slices,
                        StreamHandle stream) {
    if (!has_k || d_vk == nullptr) return;
    for (int p0 = 0; p0 < naux; p0 += kAuxChunk) {
      const int bc = std::min(kAuxChunk, naux - p0);
      const int k_chunk = nao * bc;
      const double* d_Bc =
          d_B_chunks.data() + static_cast<std::size_t>(p0) * nao * nao;
      const double beta = (p0 == 0) ? 0.0 : 1.0;
      // Step 1: T_chunk [(nao * bc) x nao] = B_chunk [(nao * bc) x nao] * D [nao x nao]
      gemm_device(ctx, k_chunk, nao, nao, 1.0, d_Bc, nao, false, d_dm, nao, false, 0.0,
                  d_T_chunk.data(), nao, mode, stream, ozaki_slices);
      // Step 2: K [nao x nao] += T_chunk [nao x (bc * nao)] * B_chunk^T [(bc * nao) x nao]
      gemm_device(ctx, nao, nao, k_chunk, 1.0, d_T_chunk.data(), k_chunk, false, d_Bc, k_chunk,
                  true, beta, d_vk, nao, mode, stream, ozaki_slices);
    }
    symmetrize_matrix_device(nao, d_vk, stream);
  }
};
#endif

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

  bool initialized = false;
  double e_nuc = 0.0;
  std::vector<double> S_mat;
  std::vector<double> H_core_mat;

  MolecularGrid grid;
  std::unique_ptr<XcEngine> xc_engine;
  std::unique_ptr<DirectJKEngine> direct_jk;
  std::unique_ptr<RIJKEngine> ri_jk;
  std::unique_ptr<SemiNumericalK> snk_engine;
#if defined(GPU4DFT_HAVE_CUDA)
  std::unique_ptr<GpuRIJKDeviceState> gpu_rijk;
#endif

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
    if (n_elec < 0 || (n_elec % 2) != 0) {
      throw std::invalid_argument("MolecularScf: closed-shell n_elec must be non-negative and even");
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
    opts.scf.exact_exchange_fraction = exx_fraction;

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
    const bool need_k = (exx_fraction != 0.0);
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
      ri_jk = std::make_unique<RIJKEngine>(basis, *aux_basis, be);
#if defined(GPU4DFT_HAVE_CUDA)
      if (be == Backend::Cuda) {
        gpu_rijk = std::make_unique<GpuRIJKDeviceState>(basis.nao(), ri_jk->naux(),
                                                        ri_jk->b_tensor(), need_k);
      }
#endif
    } else if (opts.jk_mode == JKMode::SnK) {
      direct_jk = std::make_unique<DirectJKEngine>(basis, opts.prim_cutoff);
      if (need_k) {
        snk_engine = std::make_unique<SemiNumericalK>(basis, grid, opts.snk_options);
      }
    } else if (opts.jk_mode == JKMode::RISnK) {
      ri_jk = std::make_unique<RIJKEngine>(basis, *aux_basis, be);
#if defined(GPU4DFT_HAVE_CUDA)
      if (be == Backend::Cuda) {
        gpu_rijk = std::make_unique<GpuRIJKDeviceState>(basis.nao(), ri_jk->naux(),
                                                        ri_jk->b_tensor(), /*need_k=*/false);
      }
#endif
      if (need_k) {
        snk_engine = std::make_unique<SemiNumericalK>(basis, grid, opts.snk_options);
      }
    }
    sync_gpu_if_cuda(be);
    const auto t_jk_end = SteadyClock::now();
    jk_setup_time_s = elapsed_seconds(t_jk_start, t_jk_end);
    setup_time_s = elapsed_seconds(t_setup_start, t_jk_end);

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
const WorkflowOptions& MolecularScf::options() const noexcept { return impl_->opts; }

void MolecularScf::setup() { impl_->ensure_setup(); }

WorkflowResult MolecularScf::run(std::span<const double> initial_dm_host) {
  impl_->ensure_setup();

  const int n_ao = impl_->basis.nao();
  const std::size_t n2 = static_cast<std::size_t>(n_ao) * n_ao;
  if (!initial_dm_host.empty() && initial_dm_host.size() != n2) {
    throw std::invalid_argument("MolecularScf::run: initial_dm_host size must equal nao * nao");
  }

  const Backend be = impl_->opts.scf.backend;
  const bool need_k = (impl_->exx_fraction != 0.0);

  ScfOptions run_scf_opts = impl_->opts.scf;
  run_scf_opts.exact_exchange_fraction = impl_->exx_fraction;
  // Dense RI-JK contractions do not rely on |Delta D| quartet screening, so evaluating
  // J(D) and K(D) directly avoids delta-density staging and saves the final full-reset step.
  if (impl_->opts.jk_mode == JKMode::RI || impl_->opts.jk_mode == JKMode::RISnK) {
    run_scf_opts.incremental_fock = false;
  }

  FockCallbacks cb;

  // 1. Wire XC Engine (`eval_vxc_device` on GPU, `eval_vxc` on Host)
  if (impl_->is_dft_flag) {
    XcEngine* xc_ptr = impl_->xc_engine.get();
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

  // 2. Wire Two-Electron J/K Engine (`DirectJKEngine` / `RIJKEngine` / `SemiNumericalK`)
  if (impl_->opts.jk_mode == JKMode::Direct) {
    DirectJKEngine* djk = impl_->direct_jk.get();
    const double s_cut = impl_->opts.schwarz_cutoff;
    const double fp32_thr = impl_->opts.fp32_threshold;
    cb.device_resident = false;
    cb.build_jk = [djk, be, s_cut, fp32_thr](const double* h_dm, bool /*is_inc*/, Precision prec,
                                             double* h_vj, double* h_vk, void* /*stream*/) {
      JKOptions jk_opts;
      jk_opts.backend = be;
      jk_opts.precision = prec;
      jk_opts.schwarz_cutoff = s_cut;
      jk_opts.fp32_threshold = fp32_thr;
      jk_opts.compute_j = (h_vj != nullptr);
      jk_opts.compute_k = (h_vk != nullptr);
      jk_opts.k_scale = 1.0;
      djk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
    };
  } else if (impl_->opts.jk_mode == JKMode::RI) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      GpuRIJKDeviceState* g_ri = impl_->gpu_rijk.get();
      const GemmMode target_gemm = run_scf_opts.gemm_mode;
      const GemmMode early_gemm = run_scf_opts.early_gemm_mode;
      const int oz_slices = run_scf_opts.ozaki_slices;
      cb.device_resident = true;
      cb.build_jk = [g_ri, target_gemm, early_gemm, oz_slices](
                        const double* d_dm, bool /*is_inc*/, Precision prec, double* d_vj,
                        double* d_vk, void* raw_stream) {
        auto s = static_cast<StreamHandle>(raw_stream);
        if (d_vj != nullptr) {
          g_ri->compute_j_device(d_dm, d_vj, s);
        }
        if (d_vk != nullptr) {
          GemmMode k_mode = target_gemm;
          if (prec == Precision::FP32) {
            k_mode = GemmMode::FP32;
          } else if (prec == Precision::Mixed) {
            k_mode = early_gemm;
          }
          g_ri->compute_k_device(d_dm, d_vk, k_mode, oz_slices, s);
        }
      };
    } else
#endif
    {
      RIJKEngine* rijk = impl_->ri_jk.get();
      cb.device_resident = false;
      cb.build_jk = [rijk, be](const double* h_dm, bool /*is_inc*/, Precision prec, double* h_vj,
                               double* h_vk, void* /*stream*/) {
        JKOptions jk_opts;
        jk_opts.backend = be;
        jk_opts.precision = (prec == Precision::FP32) ? Precision::FP32 : Precision::FP64;
        jk_opts.compute_j = (h_vj != nullptr);
        jk_opts.compute_k = (h_vk != nullptr);
        jk_opts.k_scale = 1.0;
        rijk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
      };
    }
  } else if (impl_->opts.jk_mode == JKMode::SnK) {
    DirectJKEngine* djk = impl_->direct_jk.get();
    SemiNumericalK* snk = impl_->snk_engine.get();
    const double s_cut = impl_->opts.schwarz_cutoff;
    const double fp32_thr = impl_->opts.fp32_threshold;
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      auto h_dm_stage = std::make_shared<std::vector<double>>(n2, 0.0);
      auto h_vj_stage = std::make_shared<std::vector<double>>(n2, 0.0);
      cb.device_resident = true;
      cb.build_jk = [djk, snk, n2, s_cut, fp32_thr, h_dm_stage, h_vj_stage](
                        const double* d_dm, bool /*is_inc*/, Precision prec, double* d_vj,
                        double* d_vk, void* raw_stream) {
        auto s = static_cast<StreamHandle>(raw_stream);
        if (d_vj != nullptr) {
          copy_d2h_async(h_dm_stage->data(), d_dm, n2 * sizeof(double), s);
          synchronize_stream(s);
          std::fill(h_vj_stage->begin(), h_vj_stage->end(), 0.0);
          JKOptions j_opts;
          j_opts.backend = Backend::Cuda;
          j_opts.precision = prec;
          j_opts.schwarz_cutoff = s_cut;
          j_opts.fp32_threshold = fp32_thr;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          djk->compute_jk(h_dm_stage->data(), h_vj_stage->data(), nullptr, j_opts);
          copy_h2d_async(d_vj, h_vj_stage->data(), n2 * sizeof(double), s);
        }
        if (d_vk != nullptr && snk != nullptr) {
          snk->compute_k_device(d_dm, d_vk, raw_stream);
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
          djk->compute_jk(h_dm, h_vj, nullptr, j_opts);
        }
        if (h_vk != nullptr && snk != nullptr) {
          snk->compute_k(h_dm, h_vk);
        }
      };
    }
  } else if (impl_->opts.jk_mode == JKMode::RISnK) {
    SemiNumericalK* snk = impl_->snk_engine.get();
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      GpuRIJKDeviceState* g_ri = impl_->gpu_rijk.get();
      cb.device_resident = true;
      cb.build_jk = [g_ri, snk](const double* d_dm, bool /*is_inc*/, Precision /*prec*/,
                                double* d_vj, double* d_vk, void* raw_stream) {
        auto s = static_cast<StreamHandle>(raw_stream);
        if (d_vj != nullptr) {
          g_ri->compute_j_device(d_dm, d_vj, s);
        }
        if (d_vk != nullptr && snk != nullptr) {
          snk->compute_k_device(d_dm, d_vk, raw_stream);
        }
      };
    } else
#endif
    {
      RIJKEngine* rijk = impl_->ri_jk.get();
      cb.device_resident = false;
      cb.build_jk = [rijk, snk, be](const double* h_dm, bool /*is_inc*/, Precision /*prec*/,
                                    double* h_vj, double* h_vk, void* /*stream*/) {
        if (h_vj != nullptr) {
          JKOptions j_opts;
          j_opts.backend = be;
          j_opts.precision = Precision::FP64;
          j_opts.compute_j = true;
          j_opts.compute_k = false;
          rijk->compute_jk(h_dm, h_vj, nullptr, j_opts);
        }
        if (h_vk != nullptr && snk != nullptr) {
          snk->compute_k(h_dm, h_vk);
        }
      };
    }
  }

  (void)need_k;
  ScfDriver driver(run_scf_opts);

  sync_gpu_if_cuda(be);
  const auto t_scf_start = SteadyClock::now();
  ScfResult scf_res =
      driver.run(n_ao, impl_->n_elec, impl_->e_nuc, impl_->S_mat.data(), impl_->H_core_mat.data(),
                 cb, initial_dm_host.empty() ? nullptr : initial_dm_host.data());
  sync_gpu_if_cuda(be);
  const auto t_scf_end = SteadyClock::now();

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

  const Backend be = impl_->opts.scf.backend;
  const double grad_schwarz = std::min(impl_->opts.schwarz_cutoff, 1e-13);

  GradientResult gres;
  gres.natom = n_atom;
  gres.xc_grad.assign(ncomp, 0.0);

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
  gres.two_electron_grad =
      jk_nuclear_grad(impl_->basis, impl_->atoms, scf_res.density_matrix.data(),
                      impl_->exx_fraction, be, grad_schwarz);
  sync_gpu_if_cuda(be);
  const auto t_2e_end = SteadyClock::now();
  gres.two_electron_time_s = elapsed_seconds(t_2e_start, t_2e_end);

  if (impl_->is_dft_flag && impl_->xc_engine != nullptr) {
    const auto t_xc_start = SteadyClock::now();
    gres.xc_grad = impl_->xc_engine->nuclear_grad(impl_->xc_atoms, scf_res.density_matrix);
    sync_gpu_if_cuda(be);
    const auto t_xc_end = SteadyClock::now();
    gres.xc_time_s = elapsed_seconds(t_xc_start, t_xc_end);
  }

  const auto t_tot_end = SteadyClock::now();
  gres.total_time_s = elapsed_seconds(t_tot_start, t_tot_end);

  gres.total_grad.assign(ncomp, 0.0);
  for (std::size_t i = 0; i < ncomp; ++i) {
    gres.total_grad[i] = gres.nuclear_repulsion_grad[i] + gres.one_electron_grad[i] +
                         gres.two_electron_grad[i] + gres.xc_grad[i];
  }
  return gres;
}

GradientResult MolecularScf::compute_gradient() {
  if (!impl_->last_scf.has_value()) {
    throw std::runtime_error("MolecularScf::compute_gradient: run() has not been executed yet");
  }
  return compute_gradient(*impl_->last_scf);
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
                        std::span<const double> external_grid_weights) {
  WorkflowOptions grad_opts = opts;
  // Analytical 2e gradient uses exact 4-center Rys/MD derivative engine regardless of SCF JK mode,
  // so no auxiliary basis is needed when constructing a standalone gradient evaluator.
  grad_opts.jk_mode = JKMode::Direct;
  MolecularScf runner(basis, std::vector<Atom>(atoms.begin(), atoms.end()), /*n_elec=*/0, grad_opts,
                      std::nullopt, external_grid_coords, external_grid_weights);
  return runner.compute_gradient(scf_res);
}

}  // namespace gpu4dft
