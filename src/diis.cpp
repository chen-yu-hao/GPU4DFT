#include "gpu4dft/diis.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
namespace gpu4dft::detail {
void diis_antisymmetrize_cuda(int n, const double* d_M, double* d_R_out, StreamHandle stream);
void diis_dot_row_cuda(std::size_t err_len, int num_vecs, const double* d_err_ring, int slot,
                       double* d_row_out, StreamHandle stream);
void diis_extrapolate_cuda(std::size_t fock_len, int max_vec, const double* d_F_ring,
                           const double* d_coeffs, double* d_F_out, StreamHandle stream);
}  // namespace gpu4dft::detail
#endif

namespace gpu4dft {

namespace {

bool solve_linear_system_pivoted(int n, std::vector<double>& A, std::vector<double>& rhs,
                                 std::vector<double>& sol) {
  sol.assign(static_cast<std::size_t>(n), 0.0);
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    double max_val = std::abs(A[static_cast<std::size_t>(col) * n + col]);
    for (int row = col + 1; row < n; ++row) {
      const double v = std::abs(A[static_cast<std::size_t>(row) * n + col]);
      if (v > max_val) {
        max_val = v;
        pivot = row;
      }
    }
    if (max_val < 1e-15) {
      return false;
    }
    if (pivot != col) {
      for (int j = col; j < n; ++j) {
        std::swap(A[static_cast<std::size_t>(col) * n + j],
                  A[static_cast<std::size_t>(pivot) * n + j]);
      }
      std::swap(rhs[col], rhs[pivot]);
    }
    const double diag = A[static_cast<std::size_t>(col) * n + col];
    for (int row = col + 1; row < n; ++row) {
      const double factor = A[static_cast<std::size_t>(row) * n + col] / diag;
      A[static_cast<std::size_t>(row) * n + col] = 0.0;
      for (int j = col + 1; j < n; ++j) {
        A[static_cast<std::size_t>(row) * n + j] -=
            factor * A[static_cast<std::size_t>(col) * n + j];
      }
      rhs[row] -= factor * rhs[col];
    }
  }

  for (int row = n - 1; row >= 0; --row) {
    double sum = rhs[row];
    for (int j = row + 1; j < n; ++j) {
      sum -= A[static_cast<std::size_t>(row) * n + j] * sol[j];
    }
    sol[row] = sum / A[static_cast<std::size_t>(row) * n + row];
  }
  return true;
}

}  // namespace

std::vector<double> solve_pulay_weights(int m, const double* B_rowmajor,
                                        const std::vector<int>& order_oldest_to_newest) {
  if (m <= 0) return {};
  std::vector<int> order = order_oldest_to_newest;
  if (static_cast<int>(order.size()) != m) {
    order.resize(static_cast<std::size_t>(m));
    std::iota(order.begin(), order.end(), 0);
  }

  std::vector<double> weights(static_cast<std::size_t>(m), 0.0);
  if (m == 1) {
    weights[order.back()] = 1.0;
    return weights;
  }

  // Iteratively drop the oldest vector if the Pulay matrix is rank-deficient or produces
  // runaway extrapolation coefficients (|c_k| > 50).
  for (int drop = 0; drop < m - 1; ++drop) {
    const int k_sub = m - drop;
    double max_diag = 0.0;
    for (int a = 0; a < k_sub; ++a) {
      const int ia = order[drop + a];
      max_diag = std::max(max_diag, std::abs(B_rowmajor[static_cast<std::size_t>(ia) * m + ia]));
    }
    if (max_diag < 1e-30) {
      weights.assign(static_cast<std::size_t>(m), 0.0);
      weights[order.back()] = 1.0;
      return weights;
    }

    const int aug = k_sub + 1;
    std::vector<double> A(static_cast<std::size_t>(aug) * aug, 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(aug), 0.0);
    for (int a = 0; a < k_sub; ++a) {
      const int ia = order[drop + a];
      for (int b = 0; b < k_sub; ++b) {
        const int ib = order[drop + b];
        double val = B_rowmajor[static_cast<std::size_t>(ia) * m + ib] / max_diag;
        if (a == b) val += 1e-14;  // tiny Tikhonov regularization
        A[static_cast<std::size_t>(a) * aug + b] = val;
      }
      A[static_cast<std::size_t>(a) * aug + k_sub] = -1.0;
      A[static_cast<std::size_t>(k_sub) * aug + a] = -1.0;
    }
    A[static_cast<std::size_t>(k_sub) * aug + k_sub] = 0.0;
    rhs[k_sub] = -1.0;

    std::vector<double> sol;
    if (solve_linear_system_pivoted(aug, A, rhs, sol)) {
      double max_c = 0.0;
      double sum_c = 0.0;
      for (int a = 0; a < k_sub; ++a) {
        max_c = std::max(max_c, std::abs(sol[a]));
        sum_c += sol[a];
      }
      if (std::isfinite(max_c) && max_c <= 50.0 && std::abs(sum_c - 1.0) < 1e-6) {
        weights.assign(static_cast<std::size_t>(m), 0.0);
        for (int a = 0; a < k_sub; ++a) {
          weights[order[drop + a]] = sol[a] / sum_c;
        }
        return weights;
      }
    }
  }

  weights.assign(static_cast<std::size_t>(m), 0.0);
  weights[order.back()] = 1.0;
  return weights;
}

// ---------------------------------------------------------------------------
// GpuDiis Implementation
// ---------------------------------------------------------------------------

struct GpuDiis::Impl {
  int nao = 0;
  int nmo = 0;
  DiisOptions opts;
  DeviceMemoryPool* pool = nullptr;

  int count = 0;
  int head = 0;
  int step_counter = 0;
  std::vector<int> slot_age_order;   // ring slots ordered from oldest to newest
  std::vector<double> h_B;           // [max_vec x max_vec] cached inner-product matrix on host
  std::vector<double> h_coeffs_pad;  // [max_vec] padded coefficients

  // Host ring buffers (used only when Backend::Host is called)
  std::vector<std::vector<double>> h_F_ring;
  std::vector<std::vector<double>> h_err_ring;
  std::vector<double> h_F_prev;
  bool h_has_prev = false;

  // GPU device buffers (allocated lazily once from DeviceMemoryPool on first GPU use)
  bool gpu_initialized = false;
  double* d_F_ring = nullptr;     // [max_vec * nao * nao]
  double* d_err_ring = nullptr;   // [max_vec * nmo * nmo]
  double* d_err_tmp = nullptr;    // [nmo * nmo] fixed target for CUDA Graph residual capture
  double* d_T1 = nullptr;         // [nao * nao]
  double* d_M = nullptr;          // [nao * nao]
  double* d_R = nullptr;          // [nao * nao]
  double* d_RX = nullptr;         // [nao * nmo]
  double* d_F_extrap = nullptr;   // [nao * nao] fixed target for CUDA Graph extrapolation capture
  double* d_F_prev = nullptr;     // [nao * nao] for early-step damping
  double* d_coeffs = nullptr;     // [max_vec]
  double* d_dot_row = nullptr;    // [max_vec]
  bool d_has_prev = false;

  Impl(int nao_, int nmo_, DiisOptions o, DeviceMemoryPool* p)
      : nao(nao_), nmo(nmo_), opts(o), pool(p) {
    if (nao <= 0 || nmo <= 0) {
      throw std::invalid_argument("GpuDiis: nao and nmo must be positive");
    }
    opts.max_vec = std::max(1, opts.max_vec);
    h_B.assign(static_cast<std::size_t>(opts.max_vec) * opts.max_vec, 0.0);
    h_coeffs_pad.assign(static_cast<std::size_t>(opts.max_vec), 0.0);
  }

  void ensure_gpu_buffers(DeviceMemoryPool& active_pool, StreamHandle stream) {
    if (gpu_initialized) return;
    pool = &active_pool;
    const std::size_t fock_len = static_cast<std::size_t>(nao) * nao;
    const std::size_t err_len = static_cast<std::size_t>(nmo) * nmo;
    const std::size_t nm_len = static_cast<std::size_t>(nao) * nmo;
    const std::size_t mv = static_cast<std::size_t>(opts.max_vec);

    d_F_ring = pool->acquire_typed<double>(mv * fock_len, stream);
    d_err_ring = pool->acquire_typed<double>(mv * err_len, stream);
    d_err_tmp = pool->acquire_typed<double>(err_len, stream);
    d_T1 = pool->acquire_typed<double>(fock_len, stream);
    d_M = pool->acquire_typed<double>(fock_len, stream);
    d_R = pool->acquire_typed<double>(fock_len, stream);
    d_RX = pool->acquire_typed<double>(nm_len, stream);
    d_F_extrap = pool->acquire_typed<double>(fock_len, stream);
    d_F_prev = pool->acquire_typed<double>(fock_len, stream);
    d_coeffs = pool->acquire_typed<double>(mv, stream);
    d_dot_row = pool->acquire_typed<double>(mv, stream);

    memset_zero_async(d_F_ring, mv * fock_len * sizeof(double), stream);
    memset_zero_async(d_err_ring, mv * err_len * sizeof(double), stream);
    memset_zero_async(d_coeffs, mv * sizeof(double), stream);
    gpu_initialized = true;
  }

  void release_gpu_buffers() noexcept {
    if (!gpu_initialized || pool == nullptr) return;
    pool->release(d_dot_row, nullptr);
    pool->release(d_coeffs, nullptr);
    pool->release(d_F_prev, nullptr);
    pool->release(d_F_extrap, nullptr);
    pool->release(d_RX, nullptr);
    pool->release(d_R, nullptr);
    pool->release(d_M, nullptr);
    pool->release(d_T1, nullptr);
    pool->release(d_err_tmp, nullptr);
    pool->release(d_err_ring, nullptr);
    pool->release(d_F_ring, nullptr);
    gpu_initialized = false;
  }

  void record_slot(int slot) {
    auto it = std::find(slot_age_order.begin(), slot_age_order.end(), slot);
    if (it != slot_age_order.end()) {
      slot_age_order.erase(it);
    }
    slot_age_order.push_back(slot);
  }

  ~Impl() { release_gpu_buffers(); }
};

GpuDiis::GpuDiis(int nao, int nmo, DiisOptions opts, DeviceMemoryPool* pool)
    : impl_(std::make_unique<Impl>(nao, nmo, opts, pool)) {}

GpuDiis::~GpuDiis() = default;
GpuDiis::GpuDiis(GpuDiis&&) noexcept = default;
GpuDiis& GpuDiis::operator=(GpuDiis&&) noexcept = default;

void GpuDiis::reset(StreamHandle stream) {
  impl_->count = 0;
  impl_->head = 0;
  impl_->slot_age_order.clear();
  std::fill(impl_->h_B.begin(), impl_->h_B.end(), 0.0);
  std::fill(impl_->h_coeffs_pad.begin(), impl_->h_coeffs_pad.end(), 0.0);
  impl_->h_F_ring.clear();
  impl_->h_err_ring.clear();
  if (impl_->gpu_initialized) {
    const std::size_t fock_len = static_cast<std::size_t>(impl_->nao) * impl_->nao;
    const std::size_t err_len = static_cast<std::size_t>(impl_->nmo) * impl_->nmo;
    const std::size_t mv = static_cast<std::size_t>(impl_->opts.max_vec);
    memset_zero_async(impl_->d_F_ring, mv * fock_len * sizeof(double), stream);
    memset_zero_async(impl_->d_err_ring, mv * err_len * sizeof(double), stream);
    memset_zero_async(impl_->d_coeffs, mv * sizeof(double), stream);
  }
}

void GpuDiis::set_damping(double damping, int damping_steps) noexcept {
  impl_->opts.damping = damping;
  impl_->opts.damping_steps = damping_steps;
}

void GpuDiis::set_level_shift(double level_shift) noexcept {
  impl_->opts.level_shift = level_shift;
}

void GpuDiis::set_gemm_mode(GemmMode mode) noexcept { impl_->opts.gemm_mode = mode; }

int GpuDiis::nao() const noexcept { return impl_->nao; }
int GpuDiis::nmo() const noexcept { return impl_->nmo; }
int GpuDiis::max_vec() const noexcept { return impl_->opts.max_vec; }
int GpuDiis::current_size() const noexcept { return impl_->count; }

DiisStepInfo GpuDiis::update_device(GpuLinalgContext& ctx, const double* d_F_in, const double* d_D,
                                    const double* d_S, const double* d_X, double* d_F_out,
                                    StreamHandle stream) {
#if defined(GPU4DFT_HAVE_CUDA)
  const int nao = impl_->nao;
  const int nmo = impl_->nmo;
  const int mv = impl_->opts.max_vec;
  const std::size_t fock_len = static_cast<std::size_t>(nao) * nao;
  const std::size_t err_len = static_cast<std::size_t>(nmo) * nmo;

  auto& graph_exec = ctx.graph_executor();
  const bool use_graph = impl_->opts.use_cuda_graph;
  StreamHandle work_stream = use_graph ? graph_exec.stream() : stream;

  cudaEvent_t sync_ev = nullptr;
  const bool bridge_streams = (use_graph && stream != work_stream);
  if (bridge_streams) {
    cudaEventCreateWithFlags(&sync_ev, cudaEventDisableTiming);
    cudaEventRecord(sync_ev, stream);
    cudaStreamWaitEvent(work_stream, sync_ev, 0);
  }

  impl_->ensure_gpu_buffers(ctx.pool(), work_stream);

  // 1. Compute orthonormalized commutator residual d_err_tmp = X^T (F D S - S D F) X on GPU
  //    (captured in CUDA Graph when pointer addresses are fixed across SCF steps!).
  auto compute_residual_seq = [&](StreamHandle s) {
    gemm_device(ctx, nao, nao, nao, 1.0, d_D, nao, false, d_S, nao, false, 0.0, impl_->d_T1, nao,
                impl_->opts.gemm_mode, s);
    gemm_device(ctx, nao, nao, nao, 1.0, d_F_in, nao, false, impl_->d_T1, nao, false, 0.0,
                impl_->d_M, nao, impl_->opts.gemm_mode, s);
    detail::diis_antisymmetrize_cuda(nao, impl_->d_M, impl_->d_R, s);
    gemm_device(ctx, nao, nmo, nao, 1.0, impl_->d_R, nao, false, d_X, nmo, false, 0.0, impl_->d_RX,
                nmo, impl_->opts.gemm_mode, s);
    gemm_device(ctx, nmo, nmo, nao, 1.0, d_X, nmo, true, impl_->d_RX, nmo, false, 0.0,
                impl_->d_err_tmp, nmo, impl_->opts.gemm_mode, s);
  };

  if (use_graph) {
    std::uint64_t res_key = 0x444949535F524553ULL;  // "DIIS_RES"
    res_key = CudaGraphExecutor::hash_combine(res_key, static_cast<std::uint64_t>(nao));
    res_key = CudaGraphExecutor::hash_combine(res_key, static_cast<std::uint64_t>(nmo));
    res_key = CudaGraphExecutor::hash_combine(res_key, static_cast<std::uint64_t>(impl_->opts.gemm_mode));
    res_key = CudaGraphExecutor::hash_combine(res_key, reinterpret_cast<std::uintptr_t>(d_F_in));
    res_key = CudaGraphExecutor::hash_combine(res_key, reinterpret_cast<std::uintptr_t>(d_D));
    res_key = CudaGraphExecutor::hash_combine(res_key, reinterpret_cast<std::uintptr_t>(d_S));
    res_key = CudaGraphExecutor::hash_combine(res_key, reinterpret_cast<std::uintptr_t>(d_X));
    res_key = CudaGraphExecutor::hash_combine(res_key, reinterpret_cast<std::uintptr_t>(impl_->d_err_tmp));
    graph_exec.execute(res_key, compute_residual_seq);
  } else {
    compute_residual_seq(work_stream);
  }

  // 2. Write F_in and err_tmp into active ring buffer slot
  const int slot = impl_->head;
  impl_->head = (impl_->head + 1) % mv;
  if (impl_->count < mv) impl_->count += 1;
  impl_->record_slot(slot);

  copy_d2d_async(impl_->d_F_ring + static_cast<std::size_t>(slot) * fock_len, d_F_in,
                 fock_len * sizeof(double), work_stream);
  copy_d2d_async(impl_->d_err_ring + static_cast<std::size_t>(slot) * err_len, impl_->d_err_tmp,
                 err_len * sizeof(double), work_stream);

  // 3. Compute B_{slot, 0..count-1} on GPU and download ONLY <= 8 doubles!
  detail::diis_dot_row_cuda(err_len, impl_->count, impl_->d_err_ring, slot, impl_->d_dot_row,
                            work_stream);

  std::vector<double> h_row(static_cast<std::size_t>(impl_->count), 0.0);
  copy_d2h_async(h_row.data(), impl_->d_dot_row,
                 static_cast<std::size_t>(impl_->count) * sizeof(double), work_stream);
  synchronize_stream(work_stream);

  for (int j = 0; j < impl_->count; ++j) {
    impl_->h_B[static_cast<std::size_t>(slot) * mv + j] = h_row[j];
    impl_->h_B[static_cast<std::size_t>(j) * mv + slot] = h_row[j];
  }

  const double norm_sq = std::max(0.0, h_row[slot]);
  const double rms_err = std::sqrt(norm_sq / static_cast<double>(err_len));

  // 4. Solve Pulay system on host (<= 9 x 9!)
  std::vector<double> B_active(static_cast<std::size_t>(impl_->count) * impl_->count, 0.0);
  for (int i = 0; i < impl_->count; ++i) {
    for (int j = 0; j < impl_->count; ++j) {
      B_active[static_cast<std::size_t>(i) * impl_->count + j] =
          impl_->h_B[static_cast<std::size_t>(i) * mv + j];
    }
  }
  const std::vector<double> sub_weights =
      solve_pulay_weights(impl_->count, B_active.data(), impl_->slot_age_order);

  std::fill(impl_->h_coeffs_pad.begin(), impl_->h_coeffs_pad.end(), 0.0);
  double max_abs_c = 0.0;
  for (int k = 0; k < impl_->count; ++k) {
    impl_->h_coeffs_pad[k] = sub_weights[k];
    max_abs_c = std::max(max_abs_c, std::abs(sub_weights[k]));
  }
  copy_h2d_async(impl_->d_coeffs, impl_->h_coeffs_pad.data(),
                 static_cast<std::size_t>(mv) * sizeof(double), work_stream);

  // 5. Extrapolate d_F_extrap = sum_k c_k d_F_ring[k] on GPU (captured in CUDA Graph!)
  auto extrapolate_seq = [&](StreamHandle s) {
    detail::diis_extrapolate_cuda(fock_len, mv, impl_->d_F_ring, impl_->d_coeffs, impl_->d_F_extrap,
                                  s);
  };

  if (use_graph) {
    std::uint64_t ext_key = 0x444949535F455854ULL;  // "DIIS_EXT"
    ext_key = CudaGraphExecutor::hash_combine(ext_key, fock_len);
    ext_key = CudaGraphExecutor::hash_combine(ext_key, static_cast<std::uint64_t>(mv));
    ext_key = CudaGraphExecutor::hash_combine(ext_key, reinterpret_cast<std::uintptr_t>(impl_->d_F_ring));
    ext_key = CudaGraphExecutor::hash_combine(ext_key, reinterpret_cast<std::uintptr_t>(impl_->d_coeffs));
    ext_key = CudaGraphExecutor::hash_combine(ext_key, reinterpret_cast<std::uintptr_t>(impl_->d_F_extrap));
    graph_exec.execute(ext_key, extrapolate_seq);
  } else {
    extrapolate_seq(work_stream);
  }

  copy_d2d_async(d_F_out, impl_->d_F_extrap, fock_len * sizeof(double), work_stream);

  // 6. Optional early-step linear damping & AO virtual level shifting
  const double beta_damp = std::clamp(impl_->opts.damping, 0.0, 0.95);
  if (beta_damp > 0.0 && impl_->d_has_prev && impl_->step_counter < impl_->opts.damping_steps) {
    axpby_device(fock_len, beta_damp, impl_->d_F_prev, 1.0 - beta_damp, d_F_out, work_stream);
  }
  copy_d2d_async(impl_->d_F_prev, d_F_out, fock_len * sizeof(double), work_stream);
  impl_->d_has_prev = true;

  if (impl_->opts.level_shift != 0.0) {
    apply_level_shift_ao_device(ctx, nao, impl_->opts.level_shift, d_S, d_D, d_F_out,
                                impl_->opts.gemm_mode, work_stream);
  }
  symmetrize_matrix_device(nao, d_F_out, work_stream);

  if (bridge_streams) {
    cudaEventRecord(sync_ev, work_stream);
    cudaStreamWaitEvent(stream, sync_ev, 0);
    cudaEventDestroy(sync_ev);
  }

  impl_->step_counter += 1;

  DiisStepInfo info;
  info.subspace_size = impl_->count;
  info.rms_error = rms_err;
  info.max_abs_coeff = max_abs_c;
  info.coefficients.resize(static_cast<std::size_t>(impl_->count));
  for (int idx = 0; idx < impl_->count; ++idx) {
    info.coefficients[idx] = sub_weights[impl_->slot_age_order[idx]];
  }
  return info;
#else
  (void)ctx;
  (void)d_F_in;
  (void)d_D;
  (void)d_S;
  (void)d_X;
  (void)d_F_out;
  (void)stream;
  throw std::runtime_error("GpuDiis::update_device: built without CUDA");
#endif
}

DiisHostResult GpuDiis::update(const double* F_in, const double* D, const double* S,
                               const double* X, Backend backend) {
  const int nao = impl_->nao;
  const int nmo = impl_->nmo;
  const std::size_t fock_len = static_cast<std::size_t>(nao) * nao;
  const std::size_t err_len = static_cast<std::size_t>(nmo) * nmo;

  DiisHostResult out;
  out.fock_extrapolated.resize(fock_len, 0.0);
  out.residual.resize(err_len, 0.0);

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    DeviceMemoryPool& pool = (impl_->pool != nullptr) ? *impl_->pool : default_device_pool();
    GpuLinalgContext ctx(&pool);
    PooledBuffer<double> d_F_in(pool, fock_len);
    PooledBuffer<double> d_D(pool, fock_len);
    PooledBuffer<double> d_S(pool, fock_len);
    PooledBuffer<double> d_X(pool, static_cast<std::size_t>(nao) * nmo);
    PooledBuffer<double> d_F_out(pool, fock_len);

    copy_h2d_async(d_F_in.data(), F_in, fock_len * sizeof(double));
    copy_h2d_async(d_D.data(), D, fock_len * sizeof(double));
    copy_h2d_async(d_S.data(), S, fock_len * sizeof(double));
    copy_h2d_async(d_X.data(), X, static_cast<std::size_t>(nao) * nmo * sizeof(double));

    out.info =
        update_device(ctx, d_F_in.data(), d_D.data(), d_S.data(), d_X.data(), d_F_out.data());

    copy_d2h_async(out.fock_extrapolated.data(), d_F_out.data(), fock_len * sizeof(double));
    copy_d2h_async(out.residual.data(), impl_->d_err_tmp, err_len * sizeof(double));
    synchronize_stream(nullptr);
    return out;
#else
    throw std::runtime_error("GpuDiis::update: CUDA backend requested, built without CUDA");
#endif
  }

  if (backend != Backend::Host) {
    throw NotImplemented("GpuDiis::update: unsupported backend");
  }

  // Host reference CDIIS
  const int mv = impl_->opts.max_vec;
  if (impl_->h_F_ring.empty()) {
    impl_->h_F_ring.assign(static_cast<std::size_t>(mv), std::vector<double>(fock_len, 0.0));
    impl_->h_err_ring.assign(static_cast<std::size_t>(mv), std::vector<double>(err_len, 0.0));
  }

  std::vector<double> DS(fock_len, 0.0);
  std::vector<double> FDS(fock_len, 0.0);
  std::vector<double> R(fock_len, 0.0);
  std::vector<double> RX(static_cast<std::size_t>(nao) * nmo, 0.0);

  gemm(nao, nao, nao, 1.0, D, false, S, false, 0.0, DS.data(), Backend::Host,
       impl_->opts.gemm_mode);
  gemm(nao, nao, nao, 1.0, F_in, false, DS.data(), false, 0.0, FDS.data(), Backend::Host,
       impl_->opts.gemm_mode);
  for (int i = 0; i < nao; ++i) {
    for (int j = 0; j < nao; ++j) {
      R[static_cast<std::size_t>(i) * nao + j] =
          FDS[static_cast<std::size_t>(i) * nao + j] - FDS[static_cast<std::size_t>(j) * nao + i];
    }
  }
  gemm(nao, nmo, nao, 1.0, R.data(), false, X, false, 0.0, RX.data(), Backend::Host,
       impl_->opts.gemm_mode);
  gemm(nmo, nmo, nao, 1.0, X, true, RX.data(), false, 0.0, out.residual.data(), Backend::Host,
       impl_->opts.gemm_mode);

  const int slot = impl_->head;
  impl_->head = (impl_->head + 1) % mv;
  if (impl_->count < mv) impl_->count += 1;
  impl_->record_slot(slot);

  std::copy(F_in, F_in + fock_len, impl_->h_F_ring[slot].begin());
  impl_->h_err_ring[slot] = out.residual;

  for (int j = 0; j < impl_->count; ++j) {
    double dot = 0.0;
    for (std::size_t idx = 0; idx < err_len; ++idx) {
      dot += impl_->h_err_ring[slot][idx] * impl_->h_err_ring[j][idx];
    }
    impl_->h_B[static_cast<std::size_t>(slot) * mv + j] = dot;
    impl_->h_B[static_cast<std::size_t>(j) * mv + slot] = dot;
  }

  const double norm_sq = std::max(0.0, impl_->h_B[static_cast<std::size_t>(slot) * mv + slot]);
  const double rms_err = std::sqrt(norm_sq / static_cast<double>(err_len));

  std::vector<double> B_active(static_cast<std::size_t>(impl_->count) * impl_->count, 0.0);
  for (int i = 0; i < impl_->count; ++i) {
    for (int j = 0; j < impl_->count; ++j) {
      B_active[static_cast<std::size_t>(i) * impl_->count + j] =
          impl_->h_B[static_cast<std::size_t>(i) * mv + j];
    }
  }
  const std::vector<double> sub_weights =
      solve_pulay_weights(impl_->count, B_active.data(), impl_->slot_age_order);

  std::fill(out.fock_extrapolated.begin(), out.fock_extrapolated.end(), 0.0);
  double max_abs_c = 0.0;
  for (int k = 0; k < impl_->count; ++k) {
    const double c = sub_weights[k];
    max_abs_c = std::max(max_abs_c, std::abs(c));
    if (c != 0.0) {
      for (std::size_t idx = 0; idx < fock_len; ++idx) {
        out.fock_extrapolated[idx] += c * impl_->h_F_ring[k][idx];
      }
    }
  }

  const double beta_damp = std::clamp(impl_->opts.damping, 0.0, 0.95);
  if (beta_damp > 0.0 && impl_->h_has_prev && impl_->step_counter < impl_->opts.damping_steps) {
    for (std::size_t idx = 0; idx < fock_len; ++idx) {
      out.fock_extrapolated[idx] =
          (1.0 - beta_damp) * out.fock_extrapolated[idx] + beta_damp * impl_->h_F_prev[idx];
    }
  }
  impl_->h_F_prev = out.fock_extrapolated;
  impl_->h_has_prev = true;

  if (impl_->opts.level_shift != 0.0) {
    std::vector<double> SD(fock_len, 0.0);
    std::vector<double> SDS(fock_len, 0.0);
    gemm(nao, nao, nao, 1.0, S, false, D, false, 0.0, SD.data(), Backend::Host,
         impl_->opts.gemm_mode);
    gemm(nao, nao, nao, 1.0, SD.data(), false, S, false, 0.0, SDS.data(), Backend::Host,
         impl_->opts.gemm_mode);
    const double ls = impl_->opts.level_shift;
    for (std::size_t idx = 0; idx < fock_len; ++idx) {
      out.fock_extrapolated[idx] += ls * (S[idx] - 0.5 * SDS[idx]);
    }
  }
  for (int i = 0; i < nao; ++i) {
    for (int j = i + 1; j < nao; ++j) {
      const double avg = 0.5 * (out.fock_extrapolated[static_cast<std::size_t>(i) * nao + j] +
                                out.fock_extrapolated[static_cast<std::size_t>(j) * nao + i]);
      out.fock_extrapolated[static_cast<std::size_t>(i) * nao + j] = avg;
      out.fock_extrapolated[static_cast<std::size_t>(j) * nao + i] = avg;
    }
  }

  impl_->step_counter += 1;
  out.info.subspace_size = impl_->count;
  out.info.rms_error = rms_err;
  out.info.max_abs_coeff = max_abs_c;
  out.info.coefficients.resize(static_cast<std::size_t>(impl_->count));
  for (int idx = 0; idx < impl_->count; ++idx) {
    out.info.coefficients[idx] = sub_weights[impl_->slot_age_order[idx]];
  }
  return out;
}

}  // namespace gpu4dft
