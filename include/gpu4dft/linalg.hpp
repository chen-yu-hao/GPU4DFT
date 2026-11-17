#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "gpu4dft/device_pool.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

enum class GemmMode {
  FP64,
  FP32,
  OzakiInt8,
};

const char* to_string(GemmMode mode);

struct GemmBenchmarkResult {
  int n = 0;
  double fp64_ms = 0.0;
  double fp64_tflops = 0.0;
  double ozaki_ms = 0.0;
  double ozaki_tflops = 0.0;
  double fp32_ms = 0.0;
  double fp32_tflops = 0.0;
  double ozaki_rel_error = 0.0;
  double ozaki_max_abs_error = 0.0;
  double fp32_rel_error = 0.0;
};

struct CanonicalOrthoResult {
  int nao = 0;
  int nmo = 0;
  std::vector<double> X;            // [nao x nmo] row-major: S^{-1/2} (X^T S X = I_nmo)
  std::vector<double> eigenvalues;  // kept overlap eigenvalues (ascending)
};

struct FockDiagResult {
  int nao = 0;
  int nmo = 0;
  std::vector<double> mo_energies;  // [nmo] ascending orbital energies
  std::vector<double> mo_coeff;     // [nao x nmo] row-major MO coefficients C (C[:, p] is MO p)
  std::vector<double> dm;           // [nao x nao] D = 2 C_occ C_occ^T
  std::vector<double> energy_dm;    // [nao x nao] W = 2 C_occ diag(eps_occ) C_occ^T
};

struct Sp2Options {
  GemmMode gemm_mode = GemmMode::FP64;
  int max_iter = 100;
  int graph_block_steps = 10;
  double idempotency_tol = 1e-12;
  double trace_tol = 1e-10;
  bool use_cuda_graph = true;
  bool mcweeny_polish = true;
  int ozaki_slices = 8;
};

struct Sp2Result {
  int nao = 0;
  int nmo = 0;
  int iterations = 0;
  double final_trace = 0.0;
  double idempotency_error = 0.0;
  std::size_t graph_captures = 0;
  std::size_t graph_replays = 0;
  std::vector<double> dm;         // [nao x nao] D = 2 X P X^T
  std::vector<double> energy_dm;  // [nao x nao] W = 0.5 D F D
};

// Persistent GPU handle owning cuBLAS + cuSOLVER handles, pre-allocated cuBLAS
// workspace (making cuBLAS 100% safe inside CUDA Graph capture), a stream-ordered
// DeviceMemoryPool reference, and an optional CudaGraphExecutor.
class GpuLinalgContext {
 public:
  explicit GpuLinalgContext(DeviceMemoryPool* pool = nullptr, int device_id = -1);
  ~GpuLinalgContext();

  GpuLinalgContext(const GpuLinalgContext&) = delete;
  GpuLinalgContext& operator=(const GpuLinalgContext&) = delete;
  GpuLinalgContext(GpuLinalgContext&&) noexcept;
  GpuLinalgContext& operator=(GpuLinalgContext&&) noexcept;

  DeviceMemoryPool& pool() noexcept;
  CudaGraphExecutor& graph_executor() noexcept;
  void set_stream(StreamHandle stream);
  StreamHandle stream() const noexcept;
  int device_id() const noexcept;

  void* cublas_raw() const noexcept;
  void* cusolver_raw() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Host / Backend-dispatched Linear Algebra API (row-major double matrices)
// ---------------------------------------------------------------------------

// Row-major C[m x n] = alpha * op(A)[m x k] * op(B)[k x n] + beta * C[m x n].
// Stored shape of A is [m x k] if !transA else [k x m]; B is [k x n] if !transB else [n x k].
void gemm(int m, int n, int k, double alpha, const double* A, int lda, bool transA, const double* B,
          int ldb, bool transB, double beta, double* C, int ldc, Backend backend = Backend::Cuda,
          GemmMode mode = GemmMode::FP64, int ozaki_slices = 8);

// Contiguous row-major convenience overload.
void gemm(int m, int n, int k, double alpha, const double* A, bool transA, const double* B,
          bool transB, double beta, double* C, Backend backend = Backend::Cuda,
          GemmMode mode = GemmMode::FP64, int ozaki_slices = 8);

// Benchmark FP64 vs OzakiInt8 vs FP32 GEMM TFLOPS and relative Frobenius error vs FP64.
GemmBenchmarkResult benchmark_gemm(int n, int warmup_iters = 2, int bench_iters = 5,
                                   Backend backend = Backend::Cuda, int ozaki_slices = 8);

// Symmetric matrix eigendecomposition A = U * diag(w) * U^T.
// Input `A` is [n x n] row-major symmetric; outputs `w` [n] ascending and `U` [n x n] row-major
// (column j of U is eigenvector j: sum_k A[i, k] * U[k, j] = w[j] * U[i, j]).
void syevd(int n, const double* A, double* w_out, double* U_out, Backend backend = Backend::Cuda);

// Canonical orthogonalization X = U * s^{-1/2}, dropping eigenvalues < lindep_tol.
CanonicalOrthoResult canonical_orthogonalization(int nao, const double* S,
                                                 double lindep_tol = 1e-7,
                                                 Backend backend = Backend::Cuda);

// Solve Roothaan-Hall equations F C = S C eps via canonical orthogonalizer X [nao x nmo],
// building D = 2 C_occ C_occ^T and W = 2 C_occ diag(eps_occ) C_occ^T.
FockDiagResult solve_fock_diag(int nao, int nmo, int n_occ, const double* F, const double* X,
                               Backend backend = Backend::Cuda,
                               GemmMode gemm_mode = GemmMode::FP64, int ozaki_slices = 8);

// Niklasson's Second-Order Trace-Resetting Density Matrix Purification (SP2).
// Maps F' = X^T F X via Gershgorin bounds into [0, 1], iterates P_{k+1} = P_k^2 or 2 P_k - P_k^2
// using pure GEMMs (FP64, FP32, or OzakiInt8, with CUDA Graph capture on GPU), and forms
// D = 2 X P X^T and W = 0.5 D F D.
Sp2Result solve_fock_sp2(int nao, int nmo, int n_occ, const double* F, const double* X,
                         Backend backend = Backend::Cuda, const Sp2Options& opts = {});

// ---------------------------------------------------------------------------
// GPU Device-Pointer API (zero host transfers inside GPU-resident SCF)
// ---------------------------------------------------------------------------

void gemm_device(GpuLinalgContext& ctx, int m, int n, int k, double alpha, const double* d_A,
                 int lda, bool transA, const double* d_B, int ldb, bool transB, double beta,
                 double* d_C, int ldc, GemmMode mode = GemmMode::FP64,
                 StreamHandle stream = nullptr, int ozaki_slices = 8);

void syevd_device(GpuLinalgContext& ctx, int n, const double* d_A, double* d_w_out,
                  double* d_U_out, StreamHandle stream = nullptr);

// Transforms F' = X^T F X, diagonalizes via cusolverDnDsyevd, back-transforms C = X C',
// and constructs D = 2 C_occ C_occ^T and (if d_W_out != nullptr) W = 2 C_occ diag(eps_occ) C_occ^T.
void solve_fock_diag_device(GpuLinalgContext& ctx, int nao, int nmo, int n_occ, const double* d_F,
                            const double* d_X, double* d_eps_out, double* d_C_out, double* d_D_out,
                            double* d_W_out = nullptr, GemmMode gemm_mode = GemmMode::FP64,
                            StreamHandle stream = nullptr, int ozaki_slices = 8);

// GPU-resident SP2 purification with CUDA Graph capture of fixed SP2 polynomial step blocks.
struct DeviceSp2Stats {
  int iterations = 0;
  double final_trace = 0.0;
  double idempotency_error = 0.0;
};

DeviceSp2Stats solve_fock_sp2_device(GpuLinalgContext& ctx, int nao, int nmo, int n_occ,
                                     const double* d_F, const double* d_X, double* d_P_work,
                                     double* d_D_out, double* d_W_out = nullptr,
                                     const Sp2Options& opts = {}, StreamHandle stream = nullptr);

// Device matrix utility primitives used by GPU-resident SCF and CDIIS.
void symmetrize_matrix_device(int n, double* d_A, StreamHandle stream = nullptr);
void axpby_device(std::size_t count, double alpha, const double* d_x, double beta, double* d_y,
                  StreamHandle stream = nullptr);
void compute_delta_and_norm_device(std::size_t count, const double* d_new, const double* d_old,
                                   double* d_delta_out, double* d_norm_stats_out,
                                   StreamHandle stream = nullptr);
void assemble_fock_and_traces_device(int nao, const double* d_D, const double* d_H,
                                     const double* d_J, const double* d_K, double alpha_x,
                                     const double* d_Vxc, double* d_F_out,
                                     double* d_traces3_out, StreamHandle stream = nullptr);
void apply_level_shift_ao_device(GpuLinalgContext& ctx, int nao, double level_shift,
                                 const double* d_S, const double* d_D, double* d_F_inout,
                                 GemmMode gemm_mode = GemmMode::FP64,
                                 StreamHandle stream = nullptr);

}  // namespace gpu4dft
