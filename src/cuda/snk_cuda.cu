#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/snk.hpp"

namespace gpu4dft::snk_detail {

// Declare SnkShellPair matching src/snk.cpp
struct SnkShellPair {
  int ish_a = 0;
  int ish_b = 0;
};

namespace {

inline void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuBLAS status " +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void check_cusolver(cusolverStatus_t status, const char* what) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template <typename T>
class DevBuf {
 public:
  DevBuf() = default;
  explicit DevBuf(std::size_t count) { resize(count); }
  DevBuf(const T* src, std::size_t count) {
    resize(count);
    upload(src, count);
  }
  explicit DevBuf(const std::vector<T>& vec) : DevBuf(vec.data(), vec.size()) {}

  ~DevBuf() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept : ptr_(o.ptr_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.size_ = 0;
  }
  DevBuf& operator=(DevBuf&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = o.ptr_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  void resize(std::size_t count) {
    if (count <= size_) return;
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    size_ = count;
    if (size_ > 0) {
      check_cuda(cudaMalloc(&ptr_, size_ * sizeof(T)), "snk cudaMalloc");
    }
  }

  void upload(const T* src, std::size_t count, cudaStream_t stream = nullptr) {
    if (count == 0) return;
    resize(count);
    check_cuda(cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream),
               "snk cudaMemcpyAsync H2D");
  }

  void download(T* dst, std::size_t count, cudaStream_t stream = nullptr) const {
    if (count == 0) return;
    check_cuda(cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream),
               "snk cudaMemcpyAsync D2H");
  }

  T* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  std::size_t size_ = 0;
};

struct DevFlatBasis {
  int nshell = 0;
  int nao = 0;
  DevBuf<int> shell_l;
  DevBuf<int> shell_ao_offset;
  DevBuf<int> shell_prim_offset;
  DevBuf<int> shell_prim_count;
  DevBuf<double> shell_center;
  DevBuf<double> prim_exp;
  DevBuf<double> prim_coeff;

  explicit DevFlatBasis(const detail::FlatBasisData& h)
      : nshell(h.nshell),
        nao(h.nao),
        shell_l(h.shell_l),
        shell_ao_offset(h.shell_ao_offset),
        shell_prim_offset(h.shell_prim_offset),
        shell_prim_count(h.shell_prim_count),
        shell_center(h.shell_center),
        prim_exp(h.prim_exp),
        prim_coeff(h.prim_coeff) {}

  detail::FlatBasisView view() const {
    return detail::FlatBasisView{
        nshell,
        nao,
        shell_l.data(),
        shell_ao_offset.data(),
        shell_prim_offset.data(),
        shell_prim_count.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
    };
  }
};

__global__ void esp_single_point_kernel(detail::FlatBasisView bview,
                                        const SnkShellPair* __restrict__ pairs,
                                        std::size_t npairs,
                                        double gx, double gy, double gz,
                                        double pair_cutoff,
                                        double* __restrict__ out_v) {
  const std::size_t ip = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (ip >= npairs) return;

  const int ia = pairs[ip].ish_a;
  const int ib = pairs[ip].ish_b;
  const int nma = 2 * bview.shell_l[ia] + 1;
  const int nmb = 2 * bview.shell_l[ib] + 1;
  const std::size_t ao_a = static_cast<std::size_t>(bview.shell_ao_offset[ia]);
  const std::size_t ao_b = static_cast<std::size_t>(bview.shell_ao_offset[ib]);
  const std::size_t nao = static_cast<std::size_t>(bview.nao);

  double block[81];
  eval_shell_pair_esp(bview, ia, ib, gx, gy, gz, pair_cutoff, block);

  for (int ma = 0; ma < nma; ++ma) {
    for (int mb = 0; mb < nmb; ++mb) {
      const double val = block[ma * nmb + mb];
      const std::size_t u = ao_a + static_cast<std::size_t>(ma);
      const std::size_t v = ao_b + static_cast<std::size_t>(mb);
      out_v[u * nao + v] = val;
      out_v[v * nao + u] = val;
    }
  }
}

__global__ void snk_eval_active_ao_kernel(detail::FlatBasisView bview,
                                          const double* __restrict__ coords,
                                          std::size_t point_offset,
                                          std::size_t ng,
                                          std::size_t na,
                                          const int* __restrict__ act_shells,
                                          const int* __restrict__ act_shell_local_ao,
                                          int n_act_shells,
                                          double* __restrict__ phi) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int s_idx = static_cast<int>(blockIdx.y);
  if (g >= ng || s_idx >= n_act_shells) return;

  const int is = act_shells[s_idx];
  const std::size_t local_ao0 = static_cast<std::size_t>(act_shell_local_ao[s_idx]);
  const std::size_t gp = point_offset + g;

  detail::eval_shell_at_point(bview, is,
                              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                              0, phi + local_ao0 * ng + g, na * ng, ng);
}

__global__ void snk_weight_phi_kernel(std::size_t na,
                                      std::size_t ng,
                                      const double* __restrict__ weights,
                                      std::size_t point_offset,
                                      const double* __restrict__ phi,
                                      double* __restrict__ phi_w) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * ng) return;
  const std::size_t g = idx % ng;
  phi_w[idx] = weights[point_offset + g] * phi[idx];
}

__global__ void snk_scatter_submatrix_kernel(const double* __restrict__ sg_act,
                                             const int* __restrict__ act_aos,
                                             std::size_t na,
                                             std::size_t nao,
                                             double* __restrict__ sg_full) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * na) return;
  const std::size_t u = idx / na;
  const std::size_t v = idx % na;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  const std::size_t gv = static_cast<std::size_t>(act_aos[v]);
  sg_full[gu * nao + gv] += sg_act[idx];
}

__global__ void snk_symmetrize_dm_kernel(const double* __restrict__ dm,
                                         std::size_t nao,
                                         double* __restrict__ dm_sym) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nao * nao) return;
  const std::size_t i = idx / nao;
  const std::size_t j = idx % nao;
  dm_sym[idx] = 0.5 * (dm[i * nao + j] + dm[j * nao + i]);
}

__global__ void snk_gather_dm_cols_kernel(const double* __restrict__ dm_sym,
                                          std::size_t nao,
                                          const int* __restrict__ act_aos,
                                          std::size_t na,
                                          double* __restrict__ d_cols) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nao * na) return;
  const std::size_t v = idx / na;
  const std::size_t u = idx % na;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  d_cols[idx] = dm_sym[v * nao + gu];
}

// Coalesced GPU semi-numerical 1e ESP contraction:
//   G[v, g] += sum_s A_{vs}(r_g) * F[s, g]
// Thread x = grid point g (coalesced, zero intra-warp atomic conflicts on G[u * ng + g]!),
// Block y = shell pair ip in [0, npairs).
__global__ void snk_esp_contract_batch_kernel(detail::FlatBasisView bview,
                                              const SnkShellPair* __restrict__ pairs,
                                              std::size_t npairs,
                                              const double* __restrict__ coords,
                                              std::size_t point_offset,
                                              std::size_t ng,
                                              double pair_cutoff,
                                              const double* __restrict__ f_mat,
                                              double* __restrict__ g_mat) {
  const std::size_t g = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t ip = static_cast<std::size_t>(blockIdx.y);
  if (g >= ng || ip >= npairs) return;

  const int ia = pairs[ip].ish_a;
  const int ib = pairs[ip].ish_b;
  const int nma = 2 * bview.shell_l[ia] + 1;
  const int nmb = 2 * bview.shell_l[ib] + 1;
  const std::size_t ao_a = static_cast<std::size_t>(bview.shell_ao_offset[ia]);
  const std::size_t ao_b = static_cast<std::size_t>(bview.shell_ao_offset[ib]);
  const std::size_t gp = point_offset + g;

  double v_block[81];
  eval_shell_pair_esp(bview, ia, ib,
                      coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                      pair_cutoff, v_block);

  for (int ma = 0; ma < nma; ++ma) {
    const std::size_t ua = ao_a + static_cast<std::size_t>(ma);
    double acc_a = 0.0;
    for (int mb = 0; mb < nmb; ++mb) {
      const std::size_t ub = ao_b + static_cast<std::size_t>(mb);
      const double a_uv = v_block[ma * nmb + mb];
      acc_a += a_uv * f_mat[ub * ng + g];
      if (ia != ib) {
        atomicAdd(&g_mat[ub * ng + g], a_uv * f_mat[ua * ng + g]);
      }
    }
    atomicAdd(&g_mat[ua * ng + g], acc_a);
  }
}

__global__ void snk_scatter_rows_kernel(const double* __restrict__ k_batch,
                                        const int* __restrict__ act_aos,
                                        std::size_t na,
                                        std::size_t nao,
                                        double* __restrict__ k_raw) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= na * nao) return;
  const std::size_t u = idx / nao;
  const std::size_t v = idx % nao;
  const std::size_t gu = static_cast<std::size_t>(act_aos[u]);
  k_raw[gu * nao + v] += k_batch[idx];
}

__global__ void snk_build_pinv_kernel(const double* __restrict__ evec,
                                      const double* __restrict__ eval,
                                      int nao,
                                      double lindep_tol,
                                      double* __restrict__ sg_pinv,
                                      double* __restrict__ p_null) {
  const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  if (idx >= nao * nao) return;
  const int i = idx / nao;
  const int j = idx % nao;

  double pinv_ij = 0.0;
  double null_ij = 0.0;
  for (int k = 0; k < nao; ++k) {
    const double lam = eval[k];
    const double vik = evec[k * nao + i];
    const double vjk = evec[k * nao + j];
    if (lam > lindep_tol) {
      pinv_ij += (vik * vjk) / lam;
    } else {
      null_ij += vik * vjk;
    }
  }
  sg_pinv[idx] = pinv_ij;
  p_null[idx] = null_ij;
}

__global__ void snk_add_inplace_kernel(const double* __restrict__ src,
                                       double* __restrict__ dst,
                                       std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] += src[i];
}

__global__ void snk_cast_f64_to_f32_kernel(const double* __restrict__ src,
                                           float* __restrict__ dst,
                                           std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<float>(src[i]);
}

__global__ void snk_cast_f32_to_f64_kernel(const float* __restrict__ src,
                                           double* __restrict__ dst,
                                           std::size_t n) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = static_cast<double>(src[i]);
}

struct SnkBatchGpuMeta {
  std::size_t point_offset = 0;
  std::size_t ng = 0;
  std::size_t na = 0;
  int n_act_shells = 0;
  std::size_t shell_list_offset = 0;
  std::size_t ao_list_offset = 0;
};

struct CudaSnkImpl {
  DevFlatBasis basis;
  SnkOptions opts;
  std::size_t npairs = 0;

  DevBuf<SnkShellPair> pairs;
  DevBuf<double> coords;
  DevBuf<double> weights;
  std::vector<SnkBatchGpuMeta> batches;
  DevBuf<int> act_shells;
  DevBuf<int> act_shell_local_ao;
  DevBuf<int> act_aos;

  DevBuf<double> q_fit;  // [nao * nao] COSX overlap-fitting matrix on GPU

  // Persistent GPU scratch buffers (zero cudaMalloc in SCF hot path!)
  std::size_t max_na = 0;
  std::size_t max_ng = 0;
  DevBuf<double> phi;
  DevBuf<double> phi_w;
  DevBuf<double> d_cols;
  DevBuf<double> f_mat;
  DevBuf<double> g_mat;
  DevBuf<double> k_batch;
  DevBuf<double> dm_sym;
  DevBuf<double> k_raw;
  DevBuf<double> qtk;
  DevBuf<double> dm_host_buf;
  DevBuf<double> vk_host_buf;

  // FP32 Tensor Core scratch buffers
  DevBuf<float> f32_a;
  DevBuf<float> f32_b;
  DevBuf<float> f32_c;

  cublasHandle_t cublas = nullptr;

  CudaSnkImpl(const detail::FlatBasisData& flat_b,
              const MolecularGrid& grid,
              const std::vector<SnkShellPair>& h_pairs,
              const std::vector<double>& s_exact,
              const SnkOptions& options,
              std::vector<double>& q_fit_out)
      : basis(flat_b),
        opts(options),
        npairs(h_pairs.size()),
        pairs(h_pairs),
        coords(grid.coords()),
        weights(grid.weights()) {
    check_cublas(cublasCreate(&cublas), "cublasCreate (SNK)");

    std::vector<int> h_act_shells;
    std::vector<int> h_act_shell_local_ao;
    std::vector<int> h_act_aos;
    batches.reserve(grid.batches().size());

    for (const SpatialBatch& sb : grid.batches()) {
      if (sb.npoints == 0 || sb.active_aos.empty()) continue;
      SnkBatchGpuMeta m;
      m.point_offset = sb.point_offset;
      m.ng = sb.npoints;
      m.na = sb.active_aos.size();
      m.n_act_shells = static_cast<int>(sb.active_shells.size());
      m.shell_list_offset = h_act_shells.size();
      m.ao_list_offset = h_act_aos.size();

      int local_cursor = 0;
      for (int is : sb.active_shells) {
        h_act_shells.push_back(is);
        h_act_shell_local_ao.push_back(local_cursor);
        local_cursor += 2 * flat_b.shell_l[static_cast<std::size_t>(is)] + 1;
      }
      for (int ao : sb.active_aos) h_act_aos.push_back(ao);

      max_na = std::max(max_na, m.na);
      max_ng = std::max(max_ng, m.ng);
      batches.push_back(m);
    }

    act_shells.upload(h_act_shells.data(), h_act_shells.size());
    act_shell_local_ao.upload(h_act_shell_local_ao.data(), h_act_shell_local_ao.size());
    act_aos.upload(h_act_aos.data(), h_act_aos.size());

    const std::size_t nao = static_cast<std::size_t>(flat_b.nao);
    const std::size_t nao2 = nao * nao;

    if (max_na > 0 && max_ng > 0 && nao > 0) {
      phi.resize(max_na * max_ng);
      phi_w.resize(max_na * max_ng);
      d_cols.resize(nao * max_na);
      f_mat.resize(nao * max_ng);
      g_mat.resize(nao * max_ng);
      k_batch.resize(max_na * nao);

      if (opts.precision == Precision::FP32) {
        const std::size_t max_sz = std::max({nao * max_na, nao * max_ng, nao2});
        f32_a.resize(max_sz);
        f32_b.resize(max_sz);
        f32_c.resize(max_sz);
      }
    }

    if (nao2 > 0) {
      dm_sym.resize(nao2);
      k_raw.resize(nao2);
      qtk.resize(nao2);
      dm_host_buf.resize(nao2);
      vk_host_buf.resize(nao2);
    }

    // Build COSX overlap-fitting matrix Q = S_grid^+ * S_exact on GPU via cuBLAS + cuSOLVER!
    if (opts.overlap_fitting && nao > 0 && !batches.empty()) {
      DevBuf<double> d_sg(nao2);
      check_cuda(cudaMemset(d_sg.data(), 0, nao2 * sizeof(double)), "snk zero d_sg");
      DevBuf<double> d_sg_act(max_na * max_na);
      const double one = 1.0, zero = 0.0;

      for (const SnkBatchGpuMeta& b : batches) {
        const dim3 ao_grid(static_cast<unsigned>((b.ng + 127) / 128),
                           static_cast<unsigned>(b.n_act_shells), 1);
        snk_eval_active_ao_kernel<<<ao_grid, 128>>>(
            basis.view(), coords.data(), b.point_offset, b.ng, b.na,
            act_shells.data() + b.shell_list_offset,
            act_shell_local_ao.data() + b.shell_list_offset,
            b.n_act_shells, phi.data());

        snk_weight_phi_kernel<<<static_cast<unsigned>((b.na * b.ng + 255) / 256), 256>>>(
            b.na, b.ng, weights.data(), b.point_offset, phi.data(), phi_w.data());

        const int i_na = static_cast<int>(b.na);
        const int i_ng = static_cast<int>(b.ng);
        check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                 i_na, i_na, i_ng, &one,
                                 phi_w.data(), i_ng,
                                 phi.data(), i_ng,
                                 &zero, d_sg_act.data(), i_na),
                     "cublasDgemm S_grid");

        snk_scatter_submatrix_kernel<<<static_cast<unsigned>((b.na * b.na + 255) / 256), 256>>>(
            d_sg_act.data(), act_aos.data() + b.ao_list_offset, b.na, nao, d_sg.data());
      }

      // Symmetric eigendecomposition of S_grid via cusolverDnDsyevd
      cusolverDnHandle_t cusolver = nullptr;
      check_cusolver(cusolverDnCreate(&cusolver), "cusolverDnCreate (SNK)");
      const int i_nao = static_cast<int>(nao);
      DevBuf<double> d_eval(nao);
      int lwork = 0;
      check_cusolver(cusolverDnDsyevd_bufferSize(
                         cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                         i_nao, d_sg.data(), i_nao, d_eval.data(), &lwork),
                     "cusolverDnDsyevd_bufferSize");
      DevBuf<double> d_work(static_cast<std::size_t>(std::max(1, lwork)));
      DevBuf<int> d_info(1);
      check_cusolver(cusolverDnDsyevd(
                         cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                         i_nao, d_sg.data(), i_nao, d_eval.data(),
                         d_work.data(), lwork, d_info.data()),
                     "cusolverDnDsyevd");
      cusolverDnDestroy(cusolver);

      DevBuf<double> d_pinv(nao2);
      DevBuf<double> d_null(nao2);
      snk_build_pinv_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256>>>(
          d_sg.data(), d_eval.data(), i_nao, opts.lindep_tol, d_pinv.data(), d_null.data());

      DevBuf<double> d_sexact(s_exact);
      q_fit.resize(nao2);
      // Q = S_grid^+ * S_exact + P_null (row-major NN)
      check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               i_nao, i_nao, i_nao, &one,
                               d_sexact.data(), i_nao,
                               d_pinv.data(), i_nao,
                               &zero, q_fit.data(), i_nao),
                   "cublasDgemm Q_fit");
      snk_add_inplace_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256>>>(
          d_null.data(), q_fit.data(), nao2);
      q_fit.download(q_fit_out.data(), nao2);
      check_cuda(cudaDeviceSynchronize(), "snk Q_fit init sync");
    } else if (nao2 > 0) {
      q_fit.upload(q_fit_out.data(), nao2);
      check_cuda(cudaDeviceSynchronize(), "snk Q_fit identity upload sync");
    }
  }

  ~CudaSnkImpl() {
    if (cublas != nullptr) cublasDestroy(cublas);
  }

  // Row-major C[M x N] = A[M x K] * B[K x N]
  void gemm_nn(int M, int N, int K, const double* dA, const double* dB, double* dC,
               cudaStream_t stream) {
    if (opts.precision == Precision::FP32) {
      const std::size_t szA = static_cast<std::size_t>(M) * K;
      const std::size_t szB = static_cast<std::size_t>(K) * N;
      const std::size_t szC = static_cast<std::size_t>(M) * N;
      snk_cast_f64_to_f32_kernel<<<static_cast<unsigned>((szA + 255) / 256), 256, 0, stream>>>(
          dA, f32_a.data(), szA);
      snk_cast_f64_to_f32_kernel<<<static_cast<unsigned>((szB + 255) / 256), 256, 0, stream>>>(
          dB, f32_b.data(), szB);
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               f32_b.data(), N,
                               f32_a.data(), K,
                               &beta, f32_c.data(), N),
                   "snk cublasSgemm NN");
      snk_cast_f32_to_f64_kernel<<<static_cast<unsigned>((szC + 255) / 256), 256, 0, stream>>>(
          f32_c.data(), dC, szC);
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             dB, N,
                             dA, K,
                             &beta, dC, N),
                 "snk cublasDgemm NN");
  }

  // Row-major C[M x N] = A[M x K] * B[N x K]^T
  void gemm_nt(int M, int N, int K, const double* dA, const double* dB, double* dC,
               cudaStream_t stream) {
    if (opts.precision == Precision::FP32) {
      const std::size_t szA = static_cast<std::size_t>(M) * K;
      const std::size_t szB = static_cast<std::size_t>(N) * K;
      const std::size_t szC = static_cast<std::size_t>(M) * N;
      snk_cast_f64_to_f32_kernel<<<static_cast<unsigned>((szA + 255) / 256), 256, 0, stream>>>(
          dA, f32_a.data(), szA);
      snk_cast_f64_to_f32_kernel<<<static_cast<unsigned>((szB + 255) / 256), 256, 0, stream>>>(
          dB, f32_b.data(), szB);
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                               N, M, K, &alpha,
                               f32_b.data(), K,
                               f32_a.data(), K,
                               &beta, f32_c.data(), N),
                   "snk cublasSgemm NT");
      snk_cast_f32_to_f64_kernel<<<static_cast<unsigned>((szC + 255) / 256), 256, 0, stream>>>(
          f32_c.data(), dC, szC);
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             N, M, K, &alpha,
                             dB, K,
                             dA, K,
                             &beta, dC, N),
                 "snk cublasDgemm NT");
  }

  // Row-major C[M x N] = A[K x M]^T * B[K x N]
  void gemm_tn(int M, int N, int K, const double* dA, const double* dB, double* dC,
               cudaStream_t stream) {
    if (opts.precision == Precision::FP32) {
      const std::size_t szA = static_cast<std::size_t>(K) * M;
      const std::size_t szB = static_cast<std::size_t>(K) * N;
      const std::size_t szC = static_cast<std::size_t>(M) * N;
      snk_cast_f64_to_f32_kernel<<<static_cast<unsigned>((szA + 255) / 256), 256, 0, stream>>>(
          dA, f32_a.data(), szA);
      snk_cast_f64_to_f32_kernel<<<static_cast<unsigned>((szB + 255) / 256), 256, 0, stream>>>(
          dB, f32_b.data(), szB);
      const float alpha = 1.0f, beta = 0.0f;
      check_cublas(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                               N, M, K, &alpha,
                               f32_b.data(), N,
                               f32_a.data(), M,
                               &beta, f32_c.data(), N),
                   "snk cublasSgemm TN");
      snk_cast_f32_to_f64_kernel<<<static_cast<unsigned>((szC + 255) / 256), 256, 0, stream>>>(
          f32_c.data(), dC, szC);
      return;
    }
    const double alpha = 1.0, beta = 0.0;
    check_cublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                             N, M, K, &alpha,
                             dB, N,
                             dA, M,
                             &beta, dC, N),
                 "snk cublasDgemm TN");
  }
};

}  // namespace

int snk_cuda_device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

void snk_esp_matrix_cuda(const detail::FlatBasisData& flat_basis,
                         const std::vector<SnkShellPair>& pairs,
                         double gx, double gy, double gz,
                         double pair_cutoff,
                         double* out_v) {
  const std::size_t nao2 = static_cast<std::size_t>(flat_basis.nao) * flat_basis.nao;
  if (nao2 == 0 || pairs.empty()) return;

  DevFlatBasis d_basis(flat_basis);
  DevBuf<SnkShellPair> d_pairs(pairs);
  DevBuf<double> d_v(nao2);
  check_cuda(cudaMemset(d_v.data(), 0, nao2 * sizeof(double)), "snk_esp_matrix memset");

  const unsigned blocks = static_cast<unsigned>((pairs.size() + 127) / 128);
  esp_single_point_kernel<<<blocks, 128>>>(
      d_basis.view(), d_pairs.data(), pairs.size(),
      gx, gy, gz, pair_cutoff, d_v.data());
  check_cuda(cudaGetLastError(), "esp_single_point_kernel launch");
  d_v.download(out_v, nao2);
  check_cuda(cudaDeviceSynchronize(), "snk_esp_matrix_cuda sync");
}

void* snk_cuda_create(const detail::FlatBasisData& flat_basis,
                      const MolecularGrid& grid,
                      const std::vector<SnkShellPair>& pairs,
                      const std::vector<double>& s_exact,
                      const SnkOptions& opts,
                      std::vector<double>& q_fit_out) {
  return new CudaSnkImpl(flat_basis, grid, pairs, s_exact, opts, q_fit_out);
}

void snk_cuda_destroy(void* impl) noexcept {
  delete static_cast<CudaSnkImpl*>(impl);
}

void snk_cuda_compute_k_device(void* raw_impl,
                               const double* d_dm,
                               double* d_vk,
                               void* cuda_stream) {
  auto* st = static_cast<CudaSnkImpl*>(raw_impl);
  cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  check_cublas(cublasSetStream(st->cublas, stream), "snk cublasSetStream");

  const std::size_t nao = static_cast<std::size_t>(st->basis.nao);
  const std::size_t nao2 = nao * nao;
  if (nao == 0) return;

  if (st->batches.empty() || st->npairs == 0) {
    check_cuda(cudaMemsetAsync(d_vk, 0, nao2 * sizeof(double), stream), "snk zero d_vk");
    return;
  }

  snk_symmetrize_dm_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, stream>>>(
      d_dm, nao, st->dm_sym.data());
  check_cuda(cudaMemsetAsync(st->k_raw.data(), 0, nao2 * sizeof(double), stream),
             "snk zero k_raw");

  const int i_nao = static_cast<int>(nao);
  const detail::FlatBasisView bview = st->basis.view();

  for (const SnkBatchGpuMeta& b : st->batches) {
    const std::size_t na = b.na;
    const std::size_t ng = b.ng;
    const int i_na = static_cast<int>(na);
    const int i_ng = static_cast<int>(ng);
    const int* d_batch_aos = st->act_aos.data() + b.ao_list_offset;

    // 1. Evaluate active AOs Phi [na x ng]
    const dim3 ao_grid(static_cast<unsigned>((ng + 127) / 128),
                       static_cast<unsigned>(b.n_act_shells), 1);
    snk_eval_active_ao_kernel<<<ao_grid, 128, 0, stream>>>(
        bview, st->coords.data(), b.point_offset, ng, na,
        st->act_shells.data() + b.shell_list_offset,
        st->act_shell_local_ao.data() + b.shell_list_offset,
        b.n_act_shells, st->phi.data());

    // 2. Gather D_cols [nao x na] and GEMM F [nao x ng] = D_cols * Phi
    snk_gather_dm_cols_kernel<<<static_cast<unsigned>((nao * na + 255) / 256), 256, 0, stream>>>(
        st->dm_sym.data(), nao, d_batch_aos, na, st->d_cols.data());
    st->gemm_nn(i_nao, i_ng, i_na, st->d_cols.data(), st->phi.data(), st->f_mat.data(), stream);

    // 3. Coalesced GPU 1e ESP contraction: G [nao x ng] = sum_s A_{vs}(r_g) * F[s, g]
    check_cuda(cudaMemsetAsync(st->g_mat.data(), 0, nao * ng * sizeof(double), stream),
               "snk zero g_mat");
    const dim3 esp_grid(static_cast<unsigned>((ng + 63) / 64),
                        static_cast<unsigned>(st->npairs), 1);
    snk_esp_contract_batch_kernel<<<esp_grid, 64, 0, stream>>>(
        bview, st->pairs.data(), st->npairs,
        st->coords.data(), b.point_offset, ng,
        st->opts.pair_cutoff, st->f_mat.data(), st->g_mat.data());

    // 4. Weight Phi_w = Phi * diag(w) and GEMM K_batch [na x nao] = Phi_w * G^T
    snk_weight_phi_kernel<<<static_cast<unsigned>((na * ng + 255) / 256), 256, 0, stream>>>(
        na, ng, st->weights.data(), b.point_offset, st->phi.data(), st->phi_w.data());
    st->gemm_nt(i_na, i_nao, i_ng, st->phi_w.data(), st->g_mat.data(), st->k_batch.data(), stream);

    // 5. Scatter active rows into k_raw [nao x nao]
    snk_scatter_rows_kernel<<<static_cast<unsigned>((na * nao + 255) / 256), 256, 0, stream>>>(
        st->k_batch.data(), d_batch_aos, na, nao, st->k_raw.data());
  }

  if (st->opts.overlap_fitting) {
    // Q^T * K_raw -> qtk, then symmetrize 0.5 * (Q^T K_raw + K_raw^T Q) -> d_vk
    st->gemm_tn(i_nao, i_nao, i_nao, st->q_fit.data(), st->k_raw.data(), st->qtk.data(), stream);
    snk_symmetrize_dm_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, stream>>>(
        st->qtk.data(), nao, d_vk);
  } else {
    snk_symmetrize_dm_kernel<<<static_cast<unsigned>((nao2 + 255) / 256), 256, 0, stream>>>(
        st->k_raw.data(), nao, d_vk);
  }
  check_cuda(cudaStreamSynchronize(stream), "snk_cuda_compute_k_device sync");
}

void snk_cuda_compute_k_host(void* raw_impl,
                             const double* h_dm,
                             double* h_vk) {
  auto* st = static_cast<CudaSnkImpl*>(raw_impl);
  const std::size_t nao2 = static_cast<std::size_t>(st->basis.nao) * st->basis.nao;
  st->dm_host_buf.upload(h_dm, nao2);
  snk_cuda_compute_k_device(raw_impl, st->dm_host_buf.data(), st->vk_host_buf.data(), nullptr);
  st->vk_host_buf.download(h_vk, nao2);
  check_cuda(cudaDeviceSynchronize(), "snk_cuda_compute_k_host download sync");
}

}  // namespace gpu4dft::snk_detail
