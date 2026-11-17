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
// Zero-copy GPU-resident state for OzakiInt8 Tensor Core RI-J and chunked RI-K contractions.
// Borrows `RIJKEngine::device_b_ptr()` directly whenever `ri_jk.has_device_b()` is true,
// completely avoiding any D2H `b_tensor()` download, duplicate `d_B` VRAM allocation, or
// host-side `B_perm` permutation loop!
struct GpuRIJKDeviceState {
  static constexpr int kAuxChunk = 192;

  int nao = 0;
  int naux = 0;
  GpuLinalgContext ctx;
  const double* d_B_ptr = nullptr;   // Borrowed from RIJKEngine::device_b_ptr()
  PooledBuffer<double> d_B_owned;    // Fallback only if RIJKEngine was built without device B
  PooledBuffer<double> d_v_aux;      // [naux]
  PooledBuffer<double> d_T_raw;      // [min(naux, kAuxChunk) * nao * nao]
  PooledBuffer<double> d_Bp_chunk;   // [min(naux, kAuxChunk) * nao * nao]
  PooledBuffer<double> d_Tp_chunk;   // [min(naux, kAuxChunk) * nao * nao]

  GpuRIJKDeviceState(const RIJKEngine& ri_jk, bool prealloc_k)
      : nao(ri_jk.nao()),
        naux(ri_jk.naux()),
        ctx(&default_device_pool()),
        d_v_aux(ctx.pool(), static_cast<std::size_t>(std::max(1, ri_jk.naux()))) {
    if (ri_jk.has_device_b()) {
      d_B_ptr = ri_jk.device_b_ptr();
    } else if (nao > 0 && naux > 0) {
      const auto& h_B = ri_jk.b_tensor();
      d_B_owned = PooledBuffer<double>(ctx.pool(), h_B.size());
      copy_h2d_async(d_B_owned.data(), h_B.data(), h_B.size() * sizeof(double));
      d_B_ptr = d_B_owned.data();
    }
    if (prealloc_k && naux > 0 && nao > 0) {
      ensure_k_buffers(nullptr);
    }
    synchronize_stream(nullptr);
  }

  void ensure_k_buffers(StreamHandle stream) {
    if (d_T_raw.size() > 0 || naux <= 0 || nao <= 0) return;
    const std::size_t max_chunk_elems =
        static_cast<std::size_t>(std::min(naux, kAuxChunk)) * nao * nao;
    d_T_raw = PooledBuffer<double>(ctx.pool(), max_chunk_elems, stream);
    d_Bp_chunk = PooledBuffer<double>(ctx.pool(), max_chunk_elems, stream);
    d_Tp_chunk = PooledBuffer<double>(ctx.pool(), max_chunk_elems, stream);
  }

  void compute_j_device(const double* d_dm, double* d_vj, StreamHandle stream) {
    if (d_vj == nullptr || nao == 0 || naux == 0 || d_B_ptr == nullptr) return;
    const int nao2 = nao * nao;
    gemm_device(ctx, naux, 1, nao2, 1.0, d_B_ptr, nao2, false, d_dm, 1, false, 0.0,
                d_v_aux.data(), 1, GemmMode::FP64, stream);
    gemm_device(ctx, nao2, 1, naux, 1.0, d_B_ptr, nao2, true, d_v_aux.data(), 1, false, 0.0, d_vj,
                1, GemmMode::FP64, stream);
  }

  void compute_k_device(const double* d_dm, double* d_vk, GemmMode mode, int ozaki_slices,
                        bool symmetrize_out, StreamHandle stream) {
    if (d_vk == nullptr || nao == 0 || naux == 0 || d_B_ptr == nullptr) return;
    ensure_k_buffers(stream);
    const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
    for (int p0 = 0; p0 < naux; p0 += kAuxChunk) {
      const int bc = std::min(kAuxChunk, naux - p0);
      const int k_chunk = bc * nao;
      const double* d_Bc = d_B_ptr + static_cast<std::size_t>(p0) * nao2;
      const double beta = (p0 == 0) ? 0.0 : 1.0;
      // Step 1: T_raw [(bc * nao) x nao] = B_c [(bc * nao) x nao] * d_dm [nao x nao]
      gemm_device(ctx, k_chunk, nao, nao, 1.0, d_Bc, nao, false, d_dm, nao, false, 0.0,
                  d_T_raw.data(), nao, mode, stream, ozaki_slices);
      // Coalesced GPU permute [bc, nao, nao] -> [nao, bc, nao] for both B_c and T_raw
      permute_ri_chunk_pair_device(bc, nao, d_Bc, d_T_raw.data(), d_Bp_chunk.data(),
                                   d_Tp_chunk.data(), stream);
      // Step 2: K [nao x nao] += T_perm [nao x (bc * nao)] * B_perm^T [(bc * nao) x nao]
      gemm_device(ctx, nao, nao, k_chunk, 1.0, d_Tp_chunk.data(), k_chunk, false,
                  d_Bp_chunk.data(), k_chunk, true, beta, d_vk, nao, mode, stream, ozaki_slices);
    }
    if (symmetrize_out) {
      symmetrize_matrix_device(nao, d_vk, stream);
    }
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
      if (be == Backend::Cuda && need_k &&
          (opts.scf.gemm_mode == GemmMode::OzakiInt8 ||
           (opts.scf.dynamic_precision && opts.scf.early_gemm_mode == GemmMode::OzakiInt8))) {
        gpu_rijk = std::make_unique<GpuRIJKDeviceState>(*ri_jk, /*prealloc_k=*/true);
      }
#endif
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

  // 2. Wire Two-Electron J/K Engine (100% Zero-Copy Device-Resident on Backend::Cuda!)
  if (impl_->opts.jk_mode == JKMode::Direct) {
    DirectJKEngine* djk = impl_->direct_jk.get();
    const double s_cut = impl_->opts.schwarz_cutoff;
    const double fp32_thr = impl_->opts.fp32_threshold;
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      cb.device_resident = true;
      cb.build_jk = [djk, s_cut, fp32_thr](const double* d_dm, bool /*is_inc*/, Precision prec,
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
        djk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
      };
    } else
#endif
    {
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
        jk_opts.symmetrize_dm = true;
        djk->compute_jk(h_dm, h_vj, h_vk, jk_opts);
      };
    }
  } else if (impl_->opts.jk_mode == JKMode::RI) {
    RIJKEngine* rijk = impl_->ri_jk.get();
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == Backend::Cuda) {
      const GemmMode target_gemm = run_scf_opts.gemm_mode;
      const GemmMode early_gemm = run_scf_opts.early_gemm_mode;
      const int oz_slices = run_scf_opts.ozaki_slices;
      if (need_k &&
          (target_gemm == GemmMode::OzakiInt8 ||
           (run_scf_opts.dynamic_precision && early_gemm == GemmMode::OzakiInt8)) &&
          !impl_->gpu_rijk) {
        impl_->gpu_rijk = std::make_unique<GpuRIJKDeviceState>(*rijk, /*prealloc_k=*/true);
      }
      GpuRIJKDeviceState* g_ri = impl_->gpu_rijk.get();
      cb.device_resident = true;
      cb.build_jk = [rijk, g_ri, target_gemm, early_gemm, oz_slices](
                        const double* d_dm, bool /*is_inc*/, Precision prec, double* d_vj,
                        double* d_vk, void* raw_stream) {
        GemmMode k_mode = target_gemm;
        if (prec == Precision::FP32) {
          k_mode = GemmMode::FP32;
        } else if (prec == Precision::Mixed) {
          k_mode = early_gemm;
        }

        if (d_vk != nullptr && k_mode == GemmMode::OzakiInt8 && g_ri != nullptr) {
          if (d_vj != nullptr) {
            JKOptions j_opts;
            j_opts.backend = Backend::Cuda;
            j_opts.precision = Precision::FP64;
            j_opts.compute_j = true;
            j_opts.compute_k = false;
            j_opts.symmetrize_dm = true;
            rijk->compute_jk_device(d_dm, d_vj, nullptr, j_opts, raw_stream);
          }
          g_ri->compute_k_device(d_dm, d_vk, GemmMode::OzakiInt8, oz_slices,
                                 /*symmetrize_out=*/true, static_cast<StreamHandle>(raw_stream));
        } else {
          JKOptions jk_opts;
          jk_opts.backend = Backend::Cuda;
          jk_opts.precision = (k_mode == GemmMode::FP32) ? Precision::FP32 : Precision::FP64;
          jk_opts.compute_j = (d_vj != nullptr);
          jk_opts.compute_k = (d_vk != nullptr);
          jk_opts.k_scale = 1.0;
          jk_opts.symmetrize_dm = true;
          rijk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
        }
      };
    } else
#endif
    {
      cb.device_resident = false;
      cb.build_jk = [rijk, be](const double* h_dm, bool /*is_inc*/, Precision prec, double* h_vj,
                               double* h_vk, void* /*stream*/) {
        JKOptions jk_opts;
        jk_opts.backend = be;
        jk_opts.precision = (prec == Precision::FP32) ? Precision::FP32 : Precision::FP64;
        jk_opts.compute_j = (h_vj != nullptr);
        jk_opts.compute_k = (h_vk != nullptr);
        jk_opts.k_scale = 1.0;
        jk_opts.symmetrize_dm = true;
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
    }
  }

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
      const bool want_ozaki_k =
          need_k &&
          (eff_opts.gemm_mode == GemmMode::OzakiInt8 ||
           (eff_opts.dynamic_precision && eff_opts.early_gemm_mode == GemmMode::OzakiInt8));
      if (want_ozaki_k && !impl_->gpu_rijk) {
        impl_->gpu_rijk = std::make_unique<GpuRIJKDeviceState>(*rijk, /*prealloc_k=*/true);
      }
      GpuRIJKDeviceState* g_ri = impl_->gpu_rijk.get();
      const GemmMode final_mode = eff_opts.gemm_mode;
      const int oz_slices = eff_opts.ozaki_slices;
      cb.device_resident = true;
      cb.build_jk = [rijk, g_ri, final_mode, oz_slices](const double* d_dm, double* d_vj,
                                                        double* d_vk, void* raw_stream) {
        if (d_vk != nullptr && final_mode == GemmMode::OzakiInt8 && g_ri != nullptr) {
          if (d_vj != nullptr) {
            JKOptions j_opts;
            j_opts.backend = Backend::Cuda;
            j_opts.precision = Precision::FP64;
            j_opts.compute_j = true;
            j_opts.compute_k = false;
            j_opts.symmetrize_dm = false;
            rijk->compute_jk_device(d_dm, d_vj, nullptr, j_opts, raw_stream);
          }
          g_ri->compute_k_device(d_dm, d_vk, GemmMode::OzakiInt8, oz_slices,
                                 /*symmetrize_out=*/false, static_cast<StreamHandle>(raw_stream));
        } else {
          JKOptions jk_opts;
          jk_opts.backend = Backend::Cuda;
          jk_opts.precision =
              (final_mode == GemmMode::FP32) ? Precision::FP32 : Precision::FP64;
          jk_opts.compute_j = (d_vj != nullptr);
          jk_opts.compute_k = (d_vk != nullptr);
          jk_opts.k_scale = 1.0;
          jk_opts.symmetrize_dm = false;
          rijk->compute_jk_device(d_dm, d_vj, d_vk, jk_opts, raw_stream);
        }
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

  const std::vector<double> eq_dm = impl_->last_scf->density_matrix;

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
