#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/jk_engine.hpp"
#include "gpu4dft/one_electron.hpp"
#include "gpu4dft/scf.hpp"
#include "gpu4dft/snk.hpp"
#include "gpu4dft/types.hpp"
#include "gpu4dft/xc.hpp"

namespace gpu4dft {

// Two-electron Coulomb/Exchange evaluation strategy for molecular RHF / RKS / Hybrid DFT.
enum class JKMode {
  Direct,  // Analytical 4-center Direct J/K (Hermite J-engine + Rys 2D-VRR + Schwarz/Mixed)
  RI,      // Density-fitting 3-center RI-J / RI-K (GPU-resident cuBLAS / OzakiInt8 / FP32 GEMMs)
  SnK,     // Analytical Direct-J (Hermite J-engine) + Semi-Numerical Exchange (sn-K / COSX)
  RISnK,   // Density-fitting RI-J + Semi-Numerical Exchange (GPU-resident RIJCOSX)
};

const char* to_string(JKMode mode);

struct WorkflowOptions {
  // Exchange-correlation functional name:
  //   "hf" / "rhf" / "" -> Restricted Hartree-Fock (exact_exchange_fraction = 1.0, no XC grid)
  //   "svwn" / "lda", "pbe", "blyp", "b3lyp", "b3lyp5", "pbe0", "tpss", ... -> RKS / Hybrid DFT
  std::string xc = "hf";

  JKMode jk_mode = JKMode::Direct;
  ScfOptions scf{};

  // Direct J/K & nuclear gradient screening parameters
  double schwarz_cutoff = 1e-13;
  double fp32_threshold = 1e-6;
  double prim_cutoff = 1e-14;

  // Molecular quadrature grid, XC engine, and semi-numerical exchange options
  GridOptions grid{};
  XcOptions xc_options{};
  SnkOptions snk_options{};
};

struct WorkflowResult {
  ScfResult scf{};

  // Wall-clock timing breakdown (seconds)
  double setup_time_s = 0.0;
  double one_electron_time_s = 0.0;
  double jk_setup_time_s = 0.0;
  double grid_setup_time_s = 0.0;
  double scf_time_s = 0.0;

  // Grid & spatial active-AO sparsity diagnostics (populated when DFT or sn-K is active)
  std::size_t grid_npoints = 0;
  std::size_t grid_nbatches = 0;
  double active_ao_ratio = 0.0;

  // Cached 1-electron matrices [nao * nao] row-major
  std::vector<double> overlap_matrix;
  std::vector<double> hcore_matrix;
};

struct GradientResult {
  int natom = 0;
  std::vector<double> total_grad;              // [natom * 3] row-major (Hartree / Bohr)
  std::vector<double> nuclear_repulsion_grad;  // [natom * 3]
  std::vector<double> one_electron_grad;       // [natom * 3]
  std::vector<double> two_electron_grad;       // [natom * 3]
  std::vector<double> xc_grad;                 // [natom * 3]

  double total_time_s = 0.0;
  double one_electron_time_s = 0.0;
  double two_electron_time_s = 0.0;
  double xc_time_s = 0.0;
};

// End-to-end C++20 molecular SCF & analytical nuclear gradient workflow.
//
// Constructs all persistent GPU/Host engines (`DirectJKEngine`, `RIJKEngine`,
// `SemiNumericalK`, `MolecularGrid`, `XcEngine`, and 1e matrices `S`, `H_core`)
// ONCE during `setup()` and executes the entire self-consistent field loop inside
// `ScfDriver` with ZERO Python callbacks in the SCF hot path.
class MolecularScf {
 public:
  MolecularScf(Basis basis, std::vector<Atom> atoms, int n_elec,
               WorkflowOptions opts = {},
               std::optional<Basis> aux_basis = std::nullopt,
               std::span<const double> external_grid_coords = {},
               std::span<const double> external_grid_weights = {});
  ~MolecularScf();

  MolecularScf(MolecularScf&&) noexcept;
  MolecularScf& operator=(MolecularScf&&) noexcept;
  MolecularScf(const MolecularScf&) = delete;
  MolecularScf& operator=(const MolecularScf&) = delete;

  int nao() const noexcept;
  int natom() const noexcept;
  int n_elec() const noexcept;
  bool is_dft() const noexcept;
  double exact_exchange_fraction() const noexcept;
  const WorkflowOptions& options() const noexcept;

  // Precomputes 1e integrals and constructs persistent J/K, grid, and XC engines once.
  // Automatically invoked on the first `run()` if not called explicitly.
  void setup();

  // Runs the full C++20 / CUDA SCF loop. If `initial_dm_host` is non-empty ([nao * nao]),
  // uses it as initial density matrix D_0 (e.g. PySCF minao guess); otherwise uses H_core guess.
  WorkflowResult run(std::span<const double> initial_dm_host = {});

  // Evaluates analytical nuclear gradients [natom * 3] (Hartree / Bohr) from a converged
  // SCF result (or the most recent `run()` result), reusing any prebuilt `XcEngine`/`MolecularGrid`.
  GradientResult compute_gradient(const ScfResult& scf_res);
  GradientResult compute_gradient();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Convenience single-call C++20 SCF entry point.
WorkflowResult run_scf(const Basis& basis, std::span<const Atom> atoms, int n_elec,
                       const WorkflowOptions& opts = {},
                       std::span<const double> initial_dm_host = {},
                       const Basis* aux_basis = nullptr,
                       std::span<const double> external_grid_coords = {},
                       std::span<const double> external_grid_weights = {});

// Convenience single-call C++20 analytical nuclear gradient entry point.
GradientResult run_grad(const Basis& basis, std::span<const Atom> atoms,
                        const ScfResult& scf_res, const WorkflowOptions& opts = {},
                        std::span<const double> external_grid_coords = {},
                        std::span<const double> external_grid_weights = {});

}  // namespace gpu4dft
