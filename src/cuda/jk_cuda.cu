#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/eri_deriv1.hpp"
#include "gpu4dft/jk_engine.hpp"

namespace gpu4dft::detail {
namespace {

void check_cuda_jk(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

void check_cublas_jk(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + ": cublasStatus=" + std::to_string(int(status)));
  }
}

template <typename T>
class DevVecJK {
 public:
  DevVecJK() = default;
  explicit DevVecJK(std::size_t n) { resize(n); }
  DevVecJK(const T* host, std::size_t n) { upload(host, n); }
  explicit DevVecJK(const std::vector<T>& v) { upload(v.data(), v.size()); }
  ~DevVecJK() { reset(); }

  DevVecJK(DevVecJK&& o) noexcept : ptr_(o.ptr_), cap_(o.cap_), size_(o.size_) {
    o.ptr_ = nullptr;
    o.cap_ = o.size_ = 0;
  }
  DevVecJK& operator=(DevVecJK&& o) noexcept {
    if (this != &o) {
      reset();
      ptr_ = o.ptr_;
      cap_ = o.cap_;
      size_ = o.size_;
      o.ptr_ = nullptr;
      o.cap_ = o.size_ = 0;
    }
    return *this;
  }
  DevVecJK(const DevVecJK&) = delete;
  DevVecJK& operator=(const DevVecJK&) = delete;

  void reset() {
    if (ptr_ != nullptr) cudaFree(ptr_);
    ptr_ = nullptr;
    cap_ = size_ = 0;
  }

  void resize(std::size_t n) {
    if (n > cap_) {
      reset();
      check_cuda_jk(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc (jk)");
      cap_ = n;
    }
    size_ = n;
  }

  void upload(const T* host, std::size_t n) {
    resize(n);
    if (n > 0) {
      check_cuda_jk(cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D (jk)");
    }
  }

  void zero() {
    if (size_ > 0) {
      check_cuda_jk(cudaMemset(ptr_, 0, size_ * sizeof(T)), "cudaMemset (jk)");
    }
  }

  void download(T* host, std::size_t n) const {
    if (n > 0) {
      check_cuda_jk(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H (jk)");
    }
  }

  T* data() const { return ptr_; }
  std::size_t size() const { return size_; }

 private:
  T* ptr_ = nullptr;
  std::size_t cap_ = 0;
  std::size_t size_ = 0;
};

struct DevShellPairList {
  int la = 0;
  int lb = 0;
  int npair = 0;
  DevVecJK<int> prim_offset;
  DevVecJK<double> exponent;
  DevVecJK<double> Px, Py, Pz;
  DevVecJK<double> prefactor;
  DevVecJK<double> Ax, Ay, Az;
  DevVecJK<double> Bx, By, Bz;
  DevVecJK<double> alpha_a, alpha_b;
  DevVecJK<double> schwarz_bound;
  DevVecJK<int> ish, jsh, ao_i, ao_j;

  explicit DevShellPairList(const ShellPairData& p)
      : la(p.la),
        lb(p.lb),
        npair(p.npair()),
        prim_offset(p.prim_offset),
        exponent(p.exponent),
        Px(p.Px),
        Py(p.Py),
        Pz(p.Pz),
        prefactor(p.prefactor),
        Ax(p.Ax),
        Ay(p.Ay),
        Az(p.Az),
        Bx(p.Bx),
        By(p.By),
        Bz(p.Bz),
        alpha_a(p.alpha_a),
        alpha_b(p.alpha_b),
        schwarz_bound(p.schwarz_bound),
        ish(p.ish),
        jsh(p.jsh),
        ao_i(p.ao_i),
        ao_j(p.ao_j) {}

  ShellPairView view() const {
    return ShellPairView{
        npair,
        prim_offset.data(),
        exponent.data(),
        Px.data(),
        Py.data(),
        Pz.data(),
        prefactor.data(),
        Ax.data(),
        Ay.data(),
        Az.data(),
        Bx.data(),
        By.data(),
        Bz.data(),
        alpha_a.data(),
        alpha_b.data(),
        schwarz_bound.data(),
        ish.data(),
        jsh.data(),
        ao_i.data(),
        ao_j.data(),
    };
  }
};

struct DevPairClassMeta {
  int la;
  int lb;
  ShellPairView view;
};

__device__ inline double dev_quartet_dmax(const double* d_sh, int nsh, int i, int j, int k, int l,
                                          bool do_j, bool do_k) {
  double m = 0.0;
  if (do_j) {
    const double d_ij = 4.0 * d_sh[i * nsh + j];
    const double d_kl = 4.0 * d_sh[k * nsh + l];
    if (d_ij > m) m = d_ij;
    if (d_kl > m) m = d_kl;
  }
  if (do_k) {
    const double d1 = d_sh[i * nsh + k];
    const double d2 = d_sh[i * nsh + l];
    const double d3 = d_sh[j * nsh + k];
    const double d4 = d_sh[j * nsh + l];
    if (d1 > m) m = d1;
    if (d2 > m) m = d2;
    if (d3 > m) m = d3;
    if (d4 > m) m = d4;
  }
  return m;
}

// Dynamic atomic work-queue Direct J/K CUDA kernel with per-quartet FP64/FP32
// mixed-precision dispatch and Hermite J-engine fast path.
__global__ void direct_jk_dynamic_kernel(
    const DevPairClassMeta* __restrict__ classes, const JKTaskTile* __restrict__ tasks,
    int num_tasks, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const double* __restrict__ d_cart, const double* __restrict__ d_shell_max, double global_scale,
    int nsh, int ncart_tot, bool do_j, bool do_k, Precision precision, double schwarz_cutoff,
    double fp32_threshold, double* __restrict__ j_cart_out, double* __restrict__ k_cart_out) {
  auto accum_j = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&j_cart_out[r * ncart_tot + c], val);
  };
  auto accum_k = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&k_cart_out[r * ncart_tot + c], val);
  };

  while (true) {
    const int tid = atomicAdd(task_counter, 1);
    if (tid >= num_tasks) break;

    const JKTaskTile t = tasks[tid];
    const DevPairClassMeta bra_m = classes[t.bp];
    const DevPairClassMeta ket_m = classes[t.kp];
    const int P = t.P;
    const int i = bra_m.view.ish[P];
    const int j = bra_m.view.jsh[P];
    const double deg_ij = (i == j) ? 1.0 : 2.0;
    const double q_ab = bra_m.view.schwarz_bound[P];
    const int off_a = shell_cart_off[i];
    const int off_b = shell_cart_off[j];

    for (int Q = t.Q_begin; Q < t.Q_end; ++Q) {
      const double q_cd = ket_m.view.schwarz_bound[Q];
      if (q_ab * q_cd * global_scale < schwarz_cutoff) break;
      const int k = ket_m.view.ish[Q];
      const int l = ket_m.view.jsh[Q];
      const double dmax = dev_quartet_dmax(d_shell_max, nsh, i, j, k, l, do_j, do_k);
      const double est = q_ab * q_cd * dmax;
      if (est < schwarz_cutoff) continue;

      const bool same_PQ = (t.bp == t.kp) && (P == Q);
      const double deg_kl = (k == l) ? 1.0 : 2.0;
      const double s_PQ = same_PQ ? 1.0 : 2.0;
      const double w_J = deg_ij * deg_kl;
      const double w_K = 0.25 * deg_ij * deg_kl * s_PQ;
      const int off_c = shell_cart_off[k];
      const int off_d = shell_cart_off[l];

      const bool use_fp64 =
          (precision == Precision::FP64) ||
          (precision == Precision::Mixed && est >= fp32_threshold);
      if (use_fp64) {
        eval_direct_jk_quartet<double>(bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P,
                                       ket_m.view, Q, same_PQ, do_j, do_k, w_J, w_K, d_cart,
                                       ncart_tot, off_a, off_b, off_c, off_d, accum_j, accum_k);
      } else {
        eval_direct_jk_quartet<float>(bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P,
                                      ket_m.view, Q, same_PQ, do_j, do_k, w_J, w_K, d_cart,
                                      ncart_tot, off_a, off_b, off_c, off_d, accum_j, accum_k);
      }
    }
  }
}

__global__ void cast_d2f_kernel(const double* __restrict__ src, float* __restrict__ dst,
                                std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    dst[i] = static_cast<float>(src[i]);
  }
}

__global__ void cast_f2d_sym_scale_kernel(const float* __restrict__ src, double scale, int nao,
                                          double* __restrict__ dst) {
  const int total = nao * nao;
  const int stride = gridDim.x * blockDim.x;
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
    const int r = idx / nao;
    const int c = idx - r * nao;
    dst[idx] = scale * 0.5 *
               (static_cast<double>(src[r * nao + c]) + static_cast<double>(src[c * nao + r]));
  }
}

__global__ void jk_nuclear_grad_dynamic_kernel(
    const DevPairClassMeta* __restrict__ classes, const GradQuartetTask* __restrict__ tasks,
    int num_tasks, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const int* __restrict__ shell_atom, const double* __restrict__ d_cart, int ncart_tot,
    double k_fraction, double* __restrict__ grad_out) {
  while (true) {
    const int tid = atomicAdd(task_counter, 1);
    if (tid >= num_tasks) break;

    const GradQuartetTask t = tasks[tid];
    const DevPairClassMeta bra = classes[t.bp];
    const DevPairClassMeta ket = classes[t.kp];
    const int i = bra.view.ish[t.P];
    const int j = bra.view.jsh[t.P];
    const int k = ket.view.ish[t.Q];
    const int l = ket.view.jsh[t.Q];

    double g12[12];
    generated::eri_quartet_grad_direct_dm<double>(
        bra.la, bra.lb, ket.la, ket.lb, bra.view, t.P, ket.view, t.Q, d_cart, ncart_tot,
        shell_cart_off[i], shell_cart_off[j], shell_cart_off[k], shell_cart_off[l], t.w_sym,
        k_fraction, g12);

    const int atoms4[4] = {shell_atom[i], shell_atom[j], shell_atom[k], shell_atom[l]};
    for (int c = 0; c < 4; ++c) {
      const int at = atoms4[c];
      if (at >= 0) {
        atomicAdd(&grad_out[3 * at + 0], g12[3 * c + 0]);
        atomicAdd(&grad_out[3 * at + 1], g12[3 * c + 1]);
        atomicAdd(&grad_out[3 * at + 2], g12[3 * c + 2]);
      }
    }
  }
}

}  // namespace

struct DirectJKCudaCache {
  std::vector<DevShellPairList> dev_lists;
  DevVecJK<DevPairClassMeta> d_classes;
  DevVecJK<int> d_shell_cart_off;
  DevVecJK<JKTaskTile> d_tasks;
  DevVecJK<int> d_counter;
  DevVecJK<double> d_dm_cart;
  DevVecJK<double> d_shell_max;
  DevVecJK<double> d_j_cart;
  DevVecJK<double> d_k_cart;
};

DirectJKCudaCache* create_direct_jk_cuda_cache(
    const Basis& /*basis*/, const std::vector<ShellPairData>& pair_lists,
    const std::vector<int>& shell_cart_offset, int /*ncart_total*/) {
  auto cache = std::make_unique<DirectJKCudaCache>();
  cache->dev_lists.reserve(pair_lists.size());
  std::vector<DevPairClassMeta> metas;
  metas.reserve(pair_lists.size());
  for (const ShellPairData& pd : pair_lists) {
    cache->dev_lists.emplace_back(pd);
    metas.push_back(DevPairClassMeta{pd.la, pd.lb, cache->dev_lists.back().view()});
  }
  cache->d_classes.upload(metas.data(), metas.size());
  cache->d_shell_cart_off.upload(shell_cart_offset.data(), shell_cart_offset.size());
  cache->d_counter.resize(1);
  return cache.release();
}

void destroy_direct_jk_cuda_cache(DirectJKCudaCache* p) { delete p; }

void run_direct_jk_cuda(DirectJKCudaCache& cache, const std::vector<JKTaskTile>& tasks,
                        const std::vector<double>& d_cart_host,
                        const std::vector<double>& d_shell_max_host, double global_scale,
                        int nshell, int ncart_total, const JKOptions& opts,
                        std::vector<double>& j_cart_out, std::vector<double>& k_cart_out) {
  cache.d_tasks.upload(tasks.data(), tasks.size());
  cache.d_counter.zero();
  cache.d_dm_cart.upload(d_cart_host.data(), d_cart_host.size());
  cache.d_shell_max.upload(d_shell_max_host.data(), d_shell_max_host.size());

  const std::size_t nc2 = static_cast<std::size_t>(ncart_total) * ncart_total;
  if (opts.compute_j) {
    cache.d_j_cart.resize(nc2);
    cache.d_j_cart.zero();
  }
  if (opts.compute_k) {
    cache.d_k_cart.resize(nc2);
    cache.d_k_cart.zero();
  }

  constexpr int kBlock = 64;
  const int blocks = std::min(static_cast<int>((tasks.size() + kBlock - 1) / kBlock), 2048);
  direct_jk_dynamic_kernel<<<blocks, kBlock>>>(
      cache.d_classes.data(), cache.d_tasks.data(), static_cast<int>(tasks.size()),
      cache.d_counter.data(), cache.d_shell_cart_off.data(), cache.d_dm_cart.data(),
      cache.d_shell_max.data(), global_scale, nshell, ncart_total, opts.compute_j, opts.compute_k,
      opts.precision, opts.schwarz_cutoff, opts.fp32_threshold,
      opts.compute_j ? cache.d_j_cart.data() : nullptr,
      opts.compute_k ? cache.d_k_cart.data() : nullptr);
  check_cuda_jk(cudaGetLastError(), "direct_jk_dynamic_kernel launch");
  check_cuda_jk(cudaDeviceSynchronize(), "direct_jk_dynamic_kernel sync");

  if (opts.compute_j) cache.d_j_cart.download(j_cart_out.data(), nc2);
  if (opts.compute_k) cache.d_k_cart.download(k_cart_out.data(), nc2);
}

struct RIJKCudaCache {
  int nao = 0;
  int naux = 0;
  cublasHandle_t handle = nullptr;
  DevVecJK<double> d_B;       // [naux * nao * nao]
  DevVecJK<double> d_B_perm;  // [nao * naux * nao]
  DevVecJK<float> d_B_perm_f;
  DevVecJK<double> d_dm;
  DevVecJK<double> d_v_aux;
  DevVecJK<double> d_vj;
  DevVecJK<double> d_vk;
  DevVecJK<double> d_T;
  DevVecJK<double> d_T_perm;
  DevVecJK<float> d_dm_f;
  DevVecJK<float> d_T_perm_f;
  DevVecJK<float> d_vk_f;

  ~RIJKCudaCache() {
    if (handle != nullptr) cublasDestroy(handle);
  }
};

RIJKCudaCache* create_rijk_cuda_cache(int nao, int naux, const std::vector<double>& b_tensor,
                                      const std::vector<double>& b_perm_tensor) {
  auto c = std::make_unique<RIJKCudaCache>();
  c->nao = nao;
  c->naux = naux;
  check_cublas_jk(cublasCreate(&c->handle), "cublasCreate (RIJK)");
  c->d_B.upload(b_tensor.data(), b_tensor.size());
  c->d_B_perm.upload(b_perm_tensor.data(), b_perm_tensor.size());
  return c.release();
}

void destroy_rijk_cuda_cache(RIJKCudaCache* p) { delete p; }

void run_rijk_cuda(RIJKCudaCache& cache, const double* dm, double* vj_out, double* vk_out,
                   const JKOptions& opts) {
  const int nao = cache.nao;
  const int naux = cache.naux;
  const int nao2 = nao * nao;
  cache.d_dm.upload(dm, static_cast<std::size_t>(nao2));

  if (vj_out != nullptr) {
    cache.d_v_aux.resize(naux);
    cache.d_vj.resize(nao2);
    const double one = 1.0, zero = 0.0;
    // Row-major B [naux x nao2] is col-major B_col [nao2 x naux]:
    //   v_aux = B_col^T * dm
    check_cublas_jk(cublasDgemv(cache.handle, CUBLAS_OP_T, nao2, naux, &one, cache.d_B.data(),
                                nao2, cache.d_dm.data(), 1, &zero, cache.d_v_aux.data(), 1),
                    "cublasDgemv (RI-J step 1)");
    //   vj = B_col * v_aux
    check_cublas_jk(cublasDgemv(cache.handle, CUBLAS_OP_N, nao2, naux, &one, cache.d_B.data(),
                                nao2, cache.d_v_aux.data(), 1, &zero, cache.d_vj.data(), 1),
                    "cublasDgemv (RI-J step 2)");
    cache.d_vj.download(vj_out, static_cast<std::size_t>(nao2));
  }

  if (vk_out != nullptr) {
    const int k_big = naux * nao;
    const std::size_t total = static_cast<std::size_t>(naux) * nao * nao;
    cache.d_vk.resize(nao2);

    if (opts.precision == Precision::FP32) {
      if (cache.d_B_perm_f.size() != total) {
        cache.d_B_perm_f.resize(total);
        cast_d2f_kernel<<<256, 256>>>(cache.d_B_perm.data(), cache.d_B_perm_f.data(), total);
      }
      cache.d_dm_f.resize(nao2);
      cache.d_T_perm_f.resize(total);
      cache.d_vk_f.resize(nao2);
      cast_d2f_kernel<<<64, 256>>>(cache.d_dm.data(), cache.d_dm_f.data(), nao2);

      const float f_one = 1.0f, f_zero = 0.0f;
      // Row-major T_perm [(nao * naux) x nao] = B_perm [(nao * naux) x nao] * D [nao x nao].
      // In col-major: T_perm_col [nao x k_big] = D_col [nao x nao] * B_perm_col [nao x k_big].
      check_cublas_jk(cublasSgemm(cache.handle, CUBLAS_OP_N, CUBLAS_OP_N, nao, k_big, nao, &f_one,
                                  cache.d_dm_f.data(), nao, cache.d_B_perm_f.data(), nao, &f_zero,
                                  cache.d_T_perm_f.data(), nao),
                      "cublasSgemm (RI-K step 1)");
      // Row-major K [nao x nao] = T_perm [nao x k_big] * B_perm^T [k_big x nao].
      // In col-major: K_col [nao x nao] = B_perm_col^T [nao x k_big] * T_perm_col [k_big x nao].
      check_cublas_jk(cublasSgemm(cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, k_big, &f_one,
                                  cache.d_B_perm_f.data(), k_big, cache.d_T_perm_f.data(), k_big,
                                  &f_zero, cache.d_vk_f.data(), nao),
                      "cublasSgemm (RI-K step 2)");
      cast_f2d_sym_scale_kernel<<<64, 256>>>(cache.d_vk_f.data(), opts.k_scale, nao,
                                             cache.d_vk.data());
    } else {
      cache.d_T_perm.resize(total);
      const double one = 1.0, zero = 0.0, k_sc = opts.k_scale;
      // GEMM 1: T_perm [(nao * naux) x nao] = B_perm [(nao * naux) x nao] * D [nao x nao].
      check_cublas_jk(cublasDgemm(cache.handle, CUBLAS_OP_N, CUBLAS_OP_N, nao, k_big, nao, &one,
                                  cache.d_dm.data(), nao, cache.d_B_perm.data(), nao, &zero,
                                  cache.d_T_perm.data(), nao),
                      "cublasDgemm (RI-K step 1)");
      // GEMM 2: K [nao x nao] = k_scale * T_perm [nao x k_big] * B_perm^T [k_big x nao].
      check_cublas_jk(cublasDgemm(cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, k_big, &k_sc,
                                  cache.d_B_perm.data(), k_big, cache.d_T_perm.data(), k_big,
                                  &zero, cache.d_vk.data(), nao),
                      "cublasDgemm (RI-K step 2)");
    }
    check_cuda_jk(cudaDeviceSynchronize(), "RI-K CUDA sync");
    cache.d_vk.download(vk_out, static_cast<std::size_t>(nao2));
  }
}

void run_jk_nuclear_grad_cuda(const std::vector<ShellPairData>& pair_lists,
                              const std::vector<GradQuartetTask>& tasks,
                              const std::vector<int>& shell_cart_offset,
                              const std::vector<int>& shell_atom, int ncart_total, int natom,
                              const std::vector<double>& d_cart_host, double k_fraction,
                              std::vector<double>& grad_out) {
  std::vector<DevShellPairList> dev_lists;
  dev_lists.reserve(pair_lists.size());
  std::vector<DevPairClassMeta> metas;
  metas.reserve(pair_lists.size());
  for (const ShellPairData& pd : pair_lists) {
    dev_lists.emplace_back(pd);
    metas.push_back(DevPairClassMeta{pd.la, pd.lb, dev_lists.back().view()});
  }

  DevVecJK<DevPairClassMeta> d_classes(metas);
  DevVecJK<GradQuartetTask> d_tasks(tasks);
  DevVecJK<int> d_counter(1);
  d_counter.zero();
  DevVecJK<int> d_sh_off(shell_cart_offset);
  DevVecJK<int> d_sh_atom(shell_atom);
  DevVecJK<double> d_dm_cart(d_cart_host);
  DevVecJK<double> d_grad(static_cast<std::size_t>(natom) * 3);
  d_grad.zero();

  constexpr int kBlock = 64;
  const int blocks = std::min(static_cast<int>((tasks.size() + kBlock - 1) / kBlock), 2048);
  jk_nuclear_grad_dynamic_kernel<<<blocks, kBlock>>>(
      d_classes.data(), d_tasks.data(), static_cast<int>(tasks.size()), d_counter.data(),
      d_sh_off.data(), d_sh_atom.data(), d_dm_cart.data(), ncart_total, k_fraction, d_grad.data());
  check_cuda_jk(cudaGetLastError(), "jk_nuclear_grad_dynamic_kernel launch");
  check_cuda_jk(cudaDeviceSynchronize(), "jk_nuclear_grad_dynamic_kernel sync");
  d_grad.download(grad_out.data(), grad_out.size());
}

}  // namespace gpu4dft::detail
