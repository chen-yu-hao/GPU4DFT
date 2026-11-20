#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "gpu4dft/device_pool.hpp"
#include "gpu4dft/linalg.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

struct DiisOptions {
  int max_vec = 8;
  int n_spin = 1;              // 1 for restricted (RHF/RKS), 2 for joint alpha/beta (UHF/UKS)
  double damping = 0.0;        // Linear Fock mixing factor in [0, 1) on early iterations
  int damping_steps = 2;       // Apply damping when step_index < damping_steps
  double level_shift = 0.0;    // Virtual-orbital level shift (Hartree) applied after extrapolation
  bool use_cuda_graph = true;  // Capture commutator residual & Fock extrapolation in CUDA Graphs
  GemmMode gemm_mode = GemmMode::FP64;
};

struct DiisStepInfo {
  int subspace_size = 0;
  double rms_error = 0.0;
  double max_abs_coeff = 0.0;
  std::vector<double> coefficients;
};

struct DiisHostResult {
  DiisStepInfo info;
  std::vector<double> fock_extrapolated;  // [nao x nao]
  std::vector<double> residual;           // [nmo x nmo] X^T (F D S - S D F) X
};

struct DiisHostUksResult {
  DiisStepInfo info;
  std::vector<double> fock_alpha_extrapolated;  // [nao x nao]
  std::vector<double> fock_beta_extrapolated;   // [nao x nao]
  std::vector<double> residual_alpha;           // [nmo x nmo]
  std::vector<double> residual_beta;            // [nmo x nmo]
};

// Solve Pulay's (m + 1) x (m + 1) augmented linear system given an [m x m]
// row-major error inner-product matrix B_{ij} = <e_i, e_j>, ordered from oldest
// to newest vector (or arbitrary order if `order_oldest_to_newest` is supplied).
std::vector<double> solve_pulay_weights(int m, const double* B_rowmajor,
                                        const std::vector<int>& order_oldest_to_newest = {});

// GPU-resident Commutator DIIS (CDIIS) accelerator with ring-buffered device Fock
// and residual storage, 8x8 scalar host Pulay solver, CUDA Graph capture of both
// residual construction and Fock extrapolation, plus early-step damping & level shift.
class GpuDiis {
 public:
  GpuDiis(int nao, int nmo, DiisOptions opts = {}, DeviceMemoryPool* pool = nullptr);
  ~GpuDiis();

  GpuDiis(const GpuDiis&) = delete;
  GpuDiis& operator=(const GpuDiis&) = delete;
  GpuDiis(GpuDiis&&) noexcept;
  GpuDiis& operator=(GpuDiis&&) noexcept;

  // Reset subspace history while keeping pre-allocated GPU ring buffers intact
  // (zero allocations, preserves CUDA Graph pointer validity across precision switches).
  void reset(StreamHandle stream = nullptr);

  void set_damping(double damping, int damping_steps = 2) noexcept;
  void set_level_shift(double level_shift) noexcept;
  void set_gemm_mode(GemmMode mode) noexcept;

  // Zero-host-matrix-copy GPU update:
  // 1. Computes orthonormalized residual d_err = X^T (F D S - S D F) X on GPU.
  // 2. Updates ring buffer slot and computes B_{slot, j} = <e_slot, e_j> on GPU.
  // 3. Transfers ONLY <= 8 doubles to host, solves Pulay weights, uploads <= 8 weights.
  // 4. Evaluates d_F_out = sum_k c_k d_F[k] (+ optional damping & AO level shift) on GPU.
  DiisStepInfo update_device(GpuLinalgContext& ctx, const double* d_F_in, const double* d_D,
                             const double* d_S, const double* d_X, double* d_F_out,
                             StreamHandle stream = nullptr);

  // Joint alpha/beta CDIIS update on GPU (packs [F_a|F_b] and [e_a|e_b] with shared Pulay weights).
  DiisStepInfo update_device_uks(GpuLinalgContext& ctx, const double* d_Fa_in,
                                 const double* d_Fb_in, const double* d_Da, const double* d_Db,
                                 const double* d_S, const double* d_X, double* d_Fa_out,
                                 double* d_Fb_out, StreamHandle stream = nullptr);

  // Host/Backend convenience API (runs on GPU when backend == Backend::Cuda, or CPU on Host).
  DiisHostResult update(const double* F_in, const double* D, const double* S, const double* X,
                        Backend backend = Backend::Cuda);

  DiisHostUksResult update_uks(const double* Fa_in, const double* Fb_in, const double* Da,
                               const double* Db, const double* S, const double* X,
                               Backend backend = Backend::Cuda);

  int nao() const noexcept;
  int nmo() const noexcept;
  int max_vec() const noexcept;
  int current_size() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

using CdiisAccelerator = GpuDiis;

}  // namespace gpu4dft
