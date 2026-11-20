#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "gpu4dft/device_pool.hpp"
#include "gpu4dft/diis.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/multigpu.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

enum class DensitySolver {
  Diag,  // Canonical orthogonalization + cuSOLVER / Host syevd
  SP2,   // Niklasson second-order trace-resetting density matrix purification (pure GEMMs)
};

struct ScfOptions {
  Backend backend = Backend::Cuda;
  DensitySolver solver = DensitySolver::Diag;
  GemmMode gemm_mode = GemmMode::FP64;              // Target/final GEMM mode (FP64, FP32, OzakiInt8)
  GemmMode early_gemm_mode = GemmMode::OzakiInt8;   // Early-stage GEMM mode when dynamic_precision=true
  Precision early_integral_precision = Precision::Mixed;  // Early-stage integral precision

  int max_iter = 100;
  double conv_tol = 1e-9;               // |Delta E| convergence threshold (Hartree)
  double diis_tol = 1e-6;               // CDIIS commutator RMS convergence threshold
  int diis_max_vec = 8;
  double damping = 0.0;                 // Early-step linear Fock damping factor in [0, 0.95)
  int damping_steps = 2;
  double level_shift = 0.0;             // Virtual-orbital level shift (Hartree)
  double lindep_tol = 1e-7;             // Canonical orthogonalization eigenvalue cutoff

  bool incremental_fock = true;         // Incremental Delta J(Delta D), Delta K(Delta D) builds
  int fock_reset_interval = 10;         // Periodic full Fock rebuild interval
  bool dynamic_precision = true;        // Multi-stage Mixed/FP32/Ozaki -> full FP64 promotion
  double fp64_switch_rms = 1e-4;        // DIIS RMS threshold triggering full FP64 Fock reset
  bool use_cuda_graphs = true;          // Capture CDIIS & SP2 sequences in CUDA Graphs

  double exact_exchange_fraction = 1.0; // RHF = 1.0, LDA/GGA/MGGA = 0.0, B3LYP = 0.20, PBE0 = 0.25
  int ozaki_slices = 8;
  Sp2Options sp2_options{};

  bool unrestricted = false;            // false = RHF/RKS, true = UHF/UKS
  int spin = 0;                         // 2*S = n_alpha - n_beta (matching PySCF mol.spin)
};

// Decoupled Fock callbacks supporting BOTH GPU device-pointer hot paths
// (`device_resident = true`) and Host staging paths (`device_resident = false`),
// and supporting either separate `(J, K)` callbacks (`build_jk`) or combined
// `V_hf = J - 0.5 * alpha_x * K` (`build_vhf`), plus spin-resolved UKS/UHF callbacks.
struct FockCallbacks {
  // Option 1: Separate Coulomb J(D) and Exact Exchange K(D) [nao x nao].
  // Note: When `compute_k` is not needed (`exact_exchange_fraction == 0.0`), `vk_out_ptr` may be nullptr.
  std::function<void(const double* dm_ptr, bool is_incremental, Precision prec, double* vj_out_ptr,
                     double* vk_out_ptr, void* stream)>
      build_jk;

  // Optional occupied-MO-accelerated J(D) and K(C_occ) [nao x nao] where `c_occ_ptr` is
  // contiguous row-major [nao x n_occ] (or nullptr when unavailable, e.g. iter 0 minao or SP2).
  std::function<void(const double* dm_ptr, const double* c_occ_ptr, int n_occ, bool is_incremental,
                     Precision prec, double* vj_out_ptr, double* vk_out_ptr, void* stream)>
      build_jk_mo;

  // Option 2: Combined two-electron HF potential V_hf(D) = J(D) - 0.5 * alpha_x * K(D) [nao x nao].
  std::function<void(const double* dm_ptr, bool is_incremental, Precision prec,
                     double* vhf_out_ptr, void* stream)>
      build_vhf;

  // Optional XC potential V_xc(D) [nao x nao] returning scalar E_xc (leave empty for RHF).
  std::function<double(const double* dm_ptr, Precision prec, double* vxc_out_ptr, void* stream)>
      build_vxc;

  // Unrestricted J(D_alpha + D_beta), K(D_alpha), K(D_beta) [nao x nao].
  // Note: `vk_alpha_out_ptr` and `vk_beta_out_ptr` may be nullptr when `exact_exchange_fraction == 0.0`.
  std::function<void(const double* dm_alpha_ptr, const double* dm_beta_ptr, bool is_incremental,
                     Precision prec, double* vj_tot_out_ptr, double* vk_alpha_out_ptr,
                     double* vk_beta_out_ptr, void* stream)>
      build_jk_uks;

  // Optional occupied-MO-accelerated unrestricted J(D_a + D_b), K(C_a), K(C_b) [nao x nao].
  std::function<void(const double* dm_alpha_ptr, const double* dm_beta_ptr,
                     const double* c_occ_alpha_ptr, int n_alpha, const double* c_occ_beta_ptr,
                     int n_beta, bool is_incremental, Precision prec, double* vj_tot_out_ptr,
                     double* vk_alpha_out_ptr, double* vk_beta_out_ptr, void* stream)>
      build_jk_uks_mo;

  // Optional unrestricted XC potentials V_xc_alpha, V_xc_beta [nao x nao] returning scalar E_xc.
  std::function<double(const double* dm_alpha_ptr, const double* dm_beta_ptr, Precision prec,
                       double* vxc_alpha_out_ptr, double* vxc_beta_out_ptr, void* stream)>
      build_vxc_uks;

  // When true, `dm_ptr`, `vj_out_ptr`, `vk_out_ptr`, `vhf_out_ptr` passed to `build_jk`/`build_vhf`
  // are GPU device pointers on `Backend::Cuda` (zero host matrix copies!).
  bool device_resident = false;

  // When true, `dm_ptr` and `vxc_out_ptr` passed to `build_vxc` are GPU device pointers.
  bool vxc_device_resident = false;
};

// Decoupled analytical nuclear gradient callbacks returning [natom * 3] vectors in Hartree / Bohr.
struct GradCallbacks {
  std::function<std::vector<double>()> nuclear_repulsion_grad_fn;
  std::function<std::vector<double>(const double* dm_host, const double* energy_weighted_dm_host)>
      one_electron_grad_fn;
  std::function<std::vector<double>(const double* dm_host, double exact_exchange_fraction)>
      two_electron_grad_fn;
  std::function<std::vector<double>(const double* dm_host)> xc_grad_fn;
  std::function<std::vector<double>(const double* dm_alpha_host, const double* dm_beta_host,
                                    double exact_exchange_fraction)>
      two_electron_grad_uks_fn;
  std::function<std::vector<double>(const double* dm_alpha_host, const double* dm_beta_host)>
      xc_grad_uks_fn;
};

struct ScfIterationLog {
  int iteration = 0;
  double total_energy = 0.0;
  double delta_energy = 0.0;
  double diis_rms = 0.0;
  bool incremental_build = false;
  Precision integral_precision = Precision::FP64;
  GemmMode gemm_mode = GemmMode::FP64;
};

struct ScfResult {
  bool converged = false;
  int iterations = 0;
  int nao = 0;
  int nmo = 0;
  double total_energy = 0.0;
  double electronic_energy = 0.0;
  double one_electron_energy = 0.0;
  double two_electron_energy = 0.0;
  double xc_energy = 0.0;
  double dispersion_energy = 0.0;
  double nuclear_repulsion = 0.0;
  double final_diis_rms = 0.0;

  std::size_t cuda_graph_captures = 0;
  std::size_t cuda_graph_replays = 0;
  std::size_t pool_cache_hits = 0;
  std::size_t steady_state_allocations = 0;  // New GPU allocations after warmup iterations (0!)

  std::vector<double> density_matrix;      // [nao * nao] total D (Tr(D*S) = n_elec)
  std::vector<double> energy_weighted_dm;  // [nao * nao] total W
  std::vector<double> fock_matrix;         // [nao * nao] converged AO Fock matrix (alpha if UHF/UKS)
  std::vector<double> mo_energies;         // [nmo] orbital energies (alpha if UHF/UKS, Diag mode)
  std::vector<double> mo_coefficients;     // [nao * nmo] row-major MO coefficients (alpha if UHF/UKS)
  std::vector<ScfIterationLog> history;

  // Open-shell UHF/UKS fields
  bool unrestricted = false;
  int n_alpha = 0;
  int n_beta = 0;
  double s2 = 0.0;
  double multiplicity = 1.0;
  std::vector<double> density_alpha;             // [nao * nao]
  std::vector<double> density_beta;              // [nao * nao]
  std::vector<double> energy_weighted_dm_alpha;  // [nao * nao]
  std::vector<double> energy_weighted_dm_beta;   // [nao * nao]
  std::vector<double> fock_alpha;                // [nao * nao]
  std::vector<double> fock_beta;                 // [nao * nao]
  std::vector<double> mo_energies_beta;          // [nmo]
  std::vector<double> mo_coefficients_beta;      // [nao * nmo]
  std::vector<double> mo_occ_alpha;              // [nmo]
  std::vector<double> mo_occ_beta;               // [nmo]
};

class ScfDriver {
 public:
  explicit ScfDriver(ScfOptions opts = {}, DeviceMemoryPool* pool = nullptr);
  ~ScfDriver();

  ScfDriver(const ScfDriver&) = delete;
  ScfDriver& operator=(const ScfDriver&) = delete;
  ScfDriver(ScfDriver&&) noexcept;
  ScfDriver& operator=(ScfDriver&&) noexcept;

  const ScfOptions& options() const noexcept;
  void set_options(ScfOptions opts);

  // Execute closed-shell RHF / RKS (or unrestricted with optional total/split guess).
  // If `initial_dm_host` is nullptr, constructs initial density from H_core.
  ScfResult run(int nao, int n_elec, double e_nuc, const double* S_host, const double* H_core_host,
                const FockCallbacks& callbacks, const double* initial_dm_host = nullptr);

  // Execute SCF with explicit spin-resolved initial density guesses [nao * nao] each.
  ScfResult run(int nao, int n_elec, double e_nuc, const double* S_host, const double* H_core_host,
                const FockCallbacks& callbacks, const double* initial_dm_alpha_host,
                const double* initial_dm_beta_host);

  // Assemble total analytical nuclear gradient [natom * 3] (Hartree / Bohr) from converged ScfResult.
  std::vector<double> compute_nuclear_gradient(int natom, const ScfResult& scf_res,
                                               const GradCallbacks& grad_callbacks) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Standalone helper creating GPU-resident (on Backend::Cuda) or Host (on Backend::Host)
// FockCallbacks from an in-core 4-index ERI tensor `h_eri` of shape [nao, nao, nao, nao].
// Contracts J and K via cuBLAS GEMM on GPU with zero host matrix copies per iteration!
FockCallbacks make_in_core_eri_callbacks(int nao, const double* h_eri,
                                         Backend backend = Backend::Cuda);

// Standalone helper assembling total analytical nuclear gradient [natom * 3] directly from
// converged `D`, `W`, nuclear repulsion gradient `grad_nuc` [natom * 3], AO derivative tensors
// `dhcore_dr` [natom * 3, nao, nao] and `ds_dr` [natom * 3, nao, nao], 2e gradient `grad_vhf`
// [natom * 3], and optional XC gradient `grad_xc` [natom * 3].
std::vector<double> assemble_nuclear_gradient(int natom, int nao, const double* dm_host,
                                              const double* energy_weighted_dm_host,
                                              const double* grad_nuc_3n,
                                              const double* dhcore_dr_3n_nao2,
                                              const double* ds_dr_3n_nao2,
                                              const double* grad_2e_3n = nullptr,
                                              const double* grad_xc_3n = nullptr);

}  // namespace gpu4dft
