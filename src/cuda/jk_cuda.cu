#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/eri_deriv1.hpp"
#include "gpu4dft/jk_engine.hpp"

namespace gpu4dft::detail {
namespace {

constexpr int kGpuKetTile = 256;
constexpr int kRiAuxChunk = 128;

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

void check_cusolver_jk(cusolverStatus_t status, const char* what) {
  if (status != CUSOLVER_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) +
                             ": cusolverStatus=" + std::to_string(int(status)));
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
      if (n > 0) {
        check_cuda_jk(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc (jk)");
      }
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

  void upload_async(const T* host, std::size_t n, cudaStream_t stream) {
    resize(n);
    if (n > 0) {
      check_cuda_jk(cudaMemcpyAsync(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice, stream),
                    "cudaMemcpyAsync H2D (jk)");
    }
  }

  void zero() {
    if (size_ > 0) {
      check_cuda_jk(cudaMemset(ptr_, 0, size_ * sizeof(T)), "cudaMemset (jk)");
    }
  }

  void zero_async(cudaStream_t stream) {
    if (size_ > 0) {
      check_cuda_jk(cudaMemsetAsync(ptr_, 0, size_ * sizeof(T), stream), "cudaMemsetAsync (jk)");
    }
  }

  void download(T* host, std::size_t n) const {
    if (n > 0) {
      check_cuda_jk(cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H (jk)");
    }
  }

  void download_async(T* host, std::size_t n, cudaStream_t stream) const {
    if (n > 0) {
      check_cuda_jk(cudaMemcpyAsync(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost, stream),
                    "cudaMemcpyAsync D2H (jk)");
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

// ---------------------------------------------------------------------------
// Device-Resident Direct J/K Kernels (Sph<->Cart + 2D Tiered Work-Stealing)
// ---------------------------------------------------------------------------

__device__ inline void atomic_max_nonneg_double(double* addr, double val) {
  if (val <= 0.0) return;
  auto* ull_addr = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old_bits = *ull_addr;
  const unsigned long long desired = static_cast<unsigned long long>(__double_as_longlong(val));
  while (old_bits < desired) {
    const unsigned long long prev = atomicCAS(ull_addr, old_bits, desired);
    if (prev == old_bits) break;
    old_bits = prev;
  }
}

__global__ void prepare_cartesian_density_kernel(
    int nsh, int nao, int ncart_tot, bool symmetrize_dm, const int* __restrict__ shell_l,
    const int* __restrict__ shell_ao_off, const int* __restrict__ shell_cart_off,
    const double* __restrict__ harm_c, const double* __restrict__ d_dm_sph,
    double* __restrict__ d_dm_cart, double* __restrict__ d_shell_max,
    double* __restrict__ d_global_dmax) {
  __shared__ double s_max[128];

  const int total_pairs = nsh * nsh;
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  double local_max = 0.0;

  if (tid < total_pairs) {
    const int i = tid / nsh;
    const int j = tid - i * nsh;
    const int li = shell_l[i];
    const int lj = shell_l[j];
    const int nci = n_cart(li), nsi = n_sph(li);
    const int ncj = n_cart(lj), nsj = n_sph(lj);
    const int aoi = shell_ao_off[i];
    const int aoj = shell_ao_off[j];
    const int ci0 = shell_cart_off[i];
    const int cj0 = shell_cart_off[j];
    const double* Ci = harm_c + (li * 9) * 15;
    const double* Cj = harm_c + (lj * 9) * 15;

    double blk_max = 0.0;
    double tmp[9 * 15] = {0.0};
    for (int mi = 0; mi < nsi; ++mi) {
      const int u = aoi + mi;
      for (int mj = 0; mj < nsj; ++mj) {
        const int v = aoj + mj;
        const double duv = d_dm_sph[u * nao + v];
        const double dvu = d_dm_sph[v * nao + u];
        double d_eff = duv;
        if (symmetrize_dm) {
          d_eff = 0.5 * (duv + dvu);
          const double ad = fabs(d_eff);
          if (ad > blk_max) blk_max = ad;
        } else {
          const double ad = fmax(fabs(duv), fabs(dvu));
          if (ad > blk_max) blk_max = ad;
        }
        const double* cj_row = Cj + mj * 15;
        double* tmp_row = tmp + mi * ncj;
        for (int cj = 0; cj < ncj; ++cj) {
          tmp_row[cj] += d_eff * cj_row[cj];
        }
      }
    }
    d_shell_max[tid] = blk_max;
    local_max = blk_max;

    for (int ci = 0; ci < nci; ++ci) {
      for (int cj = 0; cj < ncj; ++cj) {
        double acc = 0.0;
        for (int mi = 0; mi < nsi; ++mi) {
          acc += Ci[mi * 15 + ci] * tmp[mi * ncj + cj];
        }
        d_dm_cart[(ci0 + ci) * ncart_tot + (cj0 + cj)] = acc;
      }
    }
  }

  s_max[threadIdx.x] = local_max;
  __syncthreads();
  for (unsigned int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_max[threadIdx.x] = fmax(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && s_max[0] > 0.0) {
    atomic_max_nonneg_double(d_global_dmax, s_max[0]);
  }
}

__global__ void finalize_cartesian_fock_kernel(
    int nsh, int nao, int ncart_tot, bool symmetrize_out, double scale,
    const int* __restrict__ shell_l, const int* __restrict__ shell_ao_off,
    const int* __restrict__ shell_cart_off, const double* __restrict__ harm_c,
    const double* __restrict__ m_cart_raw, double* __restrict__ m_sph_out) {
  const int total_pairs = nsh * nsh;
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  if (tid >= total_pairs) return;

  const int i = tid / nsh;
  const int j = tid - i * nsh;
  if (symmetrize_out && j > i) return;

  const int li = shell_l[i];
  const int lj = shell_l[j];
  const int nci = n_cart(li), nsi = n_sph(li);
  const int ncj = n_cart(lj), nsj = n_sph(lj);
  const int aoi = shell_ao_off[i];
  const int aoj = shell_ao_off[j];
  const int ci0 = shell_cart_off[i];
  const int cj0 = shell_cart_off[j];
  const double* Ci = harm_c + (li * 9) * 15;
  const double* Cj = harm_c + (lj * 9) * 15;

  double tmp[15 * 9] = {0.0};
  for (int ci = 0; ci < nci; ++ci) {
    const int r = ci0 + ci;
    for (int cj = 0; cj < ncj; ++cj) {
      const int c = cj0 + cj;
      const double v_cart =
          symmetrize_out ? 0.5 * (m_cart_raw[r * ncart_tot + c] + m_cart_raw[c * ncart_tot + r])
                         : m_cart_raw[r * ncart_tot + c];
      for (int mj = 0; mj < nsj; ++mj) {
        tmp[ci * nsj + mj] += v_cart * Cj[mj * 15 + cj];
      }
    }
  }

  for (int mi = 0; mi < nsi; ++mi) {
    for (int mj = 0; mj < nsj; ++mj) {
      double acc = 0.0;
      for (int ci = 0; ci < nci; ++ci) {
        acc += Ci[mi * 15 + ci] * tmp[ci * nsj + mj];
      }
      const double v = scale * acc;
      m_sph_out[(aoi + mi) * nao + (aoj + mj)] = v;
      if (symmetrize_out && i != j) {
        m_sph_out[(aoj + mj) * nao + (aoi + mi)] = v;
      }
    }
  }
}

__device__ inline double dev_quartet_dmax(const double* __restrict__ d_sh, int nsh, int i, int j,
                                          int k, int l, bool do_j, bool do_k) {
  double m = 0.0;
  if (do_j) {
    const double d_ij = 4.0 * d_sh[i * nsh + j];
    const double d_kl = 4.0 * d_sh[k * nsh + l];
    if (d_ij > m) m = d_ij;
    if (d_kl > m) m = d_kl;
  }
  if (do_k) {
    const double d1 = 2.0 * d_sh[i * nsh + k];
    const double d2 = 2.0 * d_sh[i * nsh + l];
    const double d3 = 2.0 * d_sh[j * nsh + k];
    const double d4 = 2.0 * d_sh[j * nsh + l];
    if (d1 > m) m = d1;
    if (d2 > m) m = d2;
    if (d3 > m) m = d3;
    if (d4 > m) m = d4;
  }
  return m;
}

// Warp-cooperative __ballot_sync density-compacted pure-J Hermite kernel for `sp` tiles
// (MaxL == 1, where la+lb+lc+ld <= 4 always holds and SymmetrizeDM == true is always exact).
template <typename R>
__global__ void __launch_bounds__(64, 2) direct_j_hermite_sp_warp_kernel(
    const DevPairClassMeta* __restrict__ classes, const GpuJKTile* __restrict__ tiles,
    int num_tiles, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const double* __restrict__ d_cart, const double* __restrict__ d_shell_max,
    const double* __restrict__ d_global_dmax, int nsh, int ncart_tot, double est_min,
    double est_max, double early_exit_cutoff, double* __restrict__ j_cart_out) {
  const double global_d_max = *d_global_dmax;
  const double global_scale = 4.0 * global_d_max;
  if (global_scale <= 0.0) return;

  __shared__ int s_q_queue[2][64];
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  int* w_queue = s_q_queue[warp_in_block];

  auto accum_j_ket = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&j_cart_out[r * ncart_tot + c], val);
  };

  while (true) {
    int tid = 0;
    if (lane == 0) tid = atomicAdd(task_counter, 1);
    tid = __shfl_sync(0xffffffffu, tid, 0);
    if (tid >= num_tiles) break;

    const GpuJKTile t = tiles[tid];
    if (t.max_schwarz * global_scale < early_exit_cutoff) break;

    const DevPairClassMeta bra_m = classes[t.bp];
    const DevPairClassMeta ket_m = classes[t.kp];
    const int P = t.P;
    const int i = bra_m.view.ish[P];
    const int j = bra_m.view.jsh[P];
    const double deg_ij = (i == j) ? 1.0 : 2.0;
    const double q_ab = bra_m.view.schwarz_bound[P];
    const int off_a = shell_cart_off[i];
    const int off_b = shell_cart_off[j];
    const int nca = n_cart(bra_m.la);
    const int ncb = n_cart(bra_m.lb);

    double j_ab_lane[9] = {0.0};
    auto accum_j_bra = [&](int ia, int ib_c, double val) {
      j_ab_lane[ia * ncb + ib_c] += val;
    };

    int q_count = 0;
    for (int Q_base = t.Q_begin; Q_base < t.Q_end; Q_base += 32) {
      if (q_ab * ket_m.view.schwarz_bound[Q_base] * global_scale < est_min) break;

      const int Q_cand = Q_base + lane;
      bool pass = false;
      if (Q_cand < t.Q_end) {
        const double q_cd = ket_m.view.schwarz_bound[Q_cand];
        const int k = ket_m.view.ish[Q_cand];
        const int l = ket_m.view.jsh[Q_cand];
        const double dmax = dev_quartet_dmax(d_shell_max, nsh, i, j, k, l, true, false);
        const double est = q_ab * q_cd * dmax;
        pass = (est >= est_min) && (est < est_max);
      }

      const unsigned mask = __ballot_sync(0xffffffffu, pass);
      if (mask != 0u) {
        const int prefix = __popc(mask & ((1u << lane) - 1u));
        if (pass) w_queue[q_count + prefix] = Q_cand;
        q_count += __popc(mask);
      }
      __syncwarp(0xffffffffu);

      if (q_count >= 32) {
        const int Q = w_queue[lane];
        const int k = ket_m.view.ish[Q];
        const int l = ket_m.view.jsh[Q];
        const bool same_PQ = (t.bp == t.kp) && (P == Q);
        const double deg_kl = (k == l) ? 1.0 : 2.0;
        eval_hermite_j_quartet_tiered<1, R, true>(
            bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ,
            deg_ij * deg_kl, d_cart, ncart_tot, off_a, off_b, shell_cart_off[k], shell_cart_off[l],
            accum_j_bra, accum_j_ket);
        __syncwarp(0xffffffffu);
        q_count -= 32;
        if (lane < q_count) w_queue[lane] = w_queue[lane + 32];
        __syncwarp(0xffffffffu);
      }
    }

    if (lane < q_count) {
      const int Q = w_queue[lane];
      const int k = ket_m.view.ish[Q];
      const int l = ket_m.view.jsh[Q];
      const bool same_PQ = (t.bp == t.kp) && (P == Q);
      const double deg_kl = (k == l) ? 1.0 : 2.0;
      eval_hermite_j_quartet_tiered<1, R, true>(
          bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ,
          deg_ij * deg_kl, d_cart, ncart_tot, off_a, off_b, shell_cart_off[k], shell_cart_off[l],
          accum_j_bra, accum_j_ket);
    }
    __syncwarp(0xffffffffu);

    const int n_ab = nca * ncb;
    for (int idx = 0; idx < n_ab; ++idx) {
      double v = j_ab_lane[idx];
      for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
      }
      if (lane == 0 && v != 0.0) {
        const int ia = idx / ncb;
        const int ib_c = idx - ia * ncb;
        atomicAdd(&j_cart_out[(off_a + ia) * ncart_tot + (off_b + ib_c)], v);
      }
    }
  }
}

// Warp-cooperative __ballot_sync density-compacted Rys Direct J+K kernel, instantiated
// cleanly per `(MaxL, R, SymmetrizeDM)` so `<double>`, `<float>`, and `Hermite` never
// bloat one another's register allocation!
template <int MaxL, typename R, bool SymmetrizeDM>
__global__ void __launch_bounds__(64, MaxL <= 2 ? 2 : 1) direct_jk_rys_warp_kernel(
    const DevPairClassMeta* __restrict__ classes, const GpuJKTile* __restrict__ tiles,
    int num_tiles, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const double* __restrict__ d_cart, const double* __restrict__ d_shell_max,
    const double* __restrict__ d_global_dmax, int nsh, int ncart_tot, bool do_j, bool do_k,
    double est_min, double est_max, double early_exit_cutoff, double* __restrict__ j_cart_out,
    double* __restrict__ k_cart_out) {
  const double global_d_max = *d_global_dmax;
  const double global_scale = (do_j ? 4.0 : 2.0) * global_d_max;
  if (global_scale <= 0.0) return;

  constexpr int kMaxCart = (MaxL + 1) * (MaxL + 2) / 2;
  constexpr int kMaxCart2 = kMaxCart * kMaxCart;

  __shared__ int s_q_queue[2][64];
  const int warp_in_block = static_cast<int>(threadIdx.x) >> 5;
  const int lane = static_cast<int>(threadIdx.x) & 31;
  int* w_queue = s_q_queue[warp_in_block];

  auto accum_j_ket = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&j_cart_out[r * ncart_tot + c], val);
  };
  auto accum_k = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&k_cart_out[r * ncart_tot + c], val);
  };

  while (true) {
    int tid = 0;
    if (lane == 0) tid = atomicAdd(task_counter, 1);
    tid = __shfl_sync(0xffffffffu, tid, 0);
    if (tid >= num_tiles) break;

    const GpuJKTile t = tiles[tid];
    // Prebuilt `tiles` are sorted in descending `max_schwarz` order: as soon as
    // `max_schwarz * global_scale < early_exit_cutoff`, all remaining tiles can be skipped!
    if (t.max_schwarz * global_scale < early_exit_cutoff) break;

    const DevPairClassMeta bra_m = classes[t.bp];
    const DevPairClassMeta ket_m = classes[t.kp];
    const int P = t.P;
    const int i = bra_m.view.ish[P];
    const int j = bra_m.view.jsh[P];
    const double deg_ij = (i == j) ? 1.0 : 2.0;
    const double q_ab = bra_m.view.schwarz_bound[P];
    const int off_a = shell_cart_off[i];
    const int off_b = shell_cart_off[j];
    const int nca = n_cart(bra_m.la);
    const int ncb = n_cart(bra_m.lb);

    double j_ab_lane[kMaxCart2] = {0.0};
    auto accum_j_bra = [&](int ia, int ib_c, double val) {
      j_ab_lane[ia * ncb + ib_c] += val;
    };

    int q_count = 0;
    for (int Q_base = t.Q_begin; Q_base < t.Q_end; Q_base += 32) {
      if (q_ab * ket_m.view.schwarz_bound[Q_base] * global_scale < est_min) break;

      const int Q_cand = Q_base + lane;
      bool pass = false;
      if (Q_cand < t.Q_end) {
        const double q_cd = ket_m.view.schwarz_bound[Q_cand];
        const int k = ket_m.view.ish[Q_cand];
        const int l = ket_m.view.jsh[Q_cand];
        const double dmax = dev_quartet_dmax(d_shell_max, nsh, i, j, k, l, do_j, do_k);
        const double est = q_ab * q_cd * dmax;
        pass = (est >= est_min) && (est < est_max);
      }

      const unsigned mask = __ballot_sync(0xffffffffu, pass);
      if (mask != 0u) {
        const int prefix = __popc(mask & ((1u << lane) - 1u));
        if (pass) w_queue[q_count + prefix] = Q_cand;
        q_count += __popc(mask);
      }
      __syncwarp(0xffffffffu);

      if (q_count >= 32) {
        const int Q = w_queue[lane];
        const int k = ket_m.view.ish[Q];
        const int l = ket_m.view.jsh[Q];
        const bool same_PQ = (t.bp == t.kp) && (P == Q);
        const double deg_kl = (k == l) ? 1.0 : 2.0;
        const double s_PQ = same_PQ ? 1.0 : 2.0;
        const double w_J = deg_ij * deg_kl;
        const double w_K =
            SymmetrizeDM ? (0.25 * deg_ij * deg_kl * s_PQ) : (0.25 * deg_ij * deg_kl);
        eval_rys_jk_quartet_tiered<MaxL, R, SymmetrizeDM>(
            bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ, do_j,
            do_k, w_J, w_K, d_cart, ncart_tot, off_a, off_b, shell_cart_off[k], shell_cart_off[l],
            accum_j_bra, accum_j_ket, accum_k);
        __syncwarp(0xffffffffu);
        q_count -= 32;
        if (lane < q_count) w_queue[lane] = w_queue[lane + 32];
        __syncwarp(0xffffffffu);
      }
    }

    if (lane < q_count) {
      const int Q = w_queue[lane];
      const int k = ket_m.view.ish[Q];
      const int l = ket_m.view.jsh[Q];
      const bool same_PQ = (t.bp == t.kp) && (P == Q);
      const double deg_kl = (k == l) ? 1.0 : 2.0;
      const double s_PQ = same_PQ ? 1.0 : 2.0;
      const double w_J = deg_ij * deg_kl;
      const double w_K =
          SymmetrizeDM ? (0.25 * deg_ij * deg_kl * s_PQ) : (0.25 * deg_ij * deg_kl);
      eval_rys_jk_quartet_tiered<MaxL, R, SymmetrizeDM>(
          bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ, do_j,
          do_k, w_J, w_K, d_cart, ncart_tot, off_a, off_b, shell_cart_off[k], shell_cart_off[l],
          accum_j_bra, accum_j_ket, accum_k);
    }
    __syncwarp(0xffffffffu);

    if (do_j) {
      const int n_ab = nca * ncb;
      for (int idx = 0; idx < n_ab; ++idx) {
        double v = j_ab_lane[idx];
        for (int offset = 16; offset > 0; offset >>= 1) {
          v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0 && v != 0.0) {
          const int ia = idx / ncb;
          const int ib_c = idx - ia * ncb;
          atomicAdd(&j_cart_out[(off_a + ia) * ncart_tot + (off_b + ib_c)], v);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 100% GPU-Native 2-Center & 3-Center RI-JK Integral & Contraction Kernels
// ---------------------------------------------------------------------------

__device__ inline void rys_2c_1d(int lP, int lQ, double C00, double D00, double B10, double B01,
                                 double B00, double G[5][5]) {
  G[0][0] = 1.0;
  for (int n = 0; n < lP; ++n) {
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? static_cast<double>(n) * B10 * G[n - 1][0] : 0.0);
  }
  for (int m = 0; m < lQ; ++m) {
    for (int n = 0; n <= lP; ++n) {
      double val = D00 * G[n][m];
      if (m > 0) val += static_cast<double>(m) * B01 * G[n][m - 1];
      if (n > 0) val += static_cast<double>(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }
  }
}

__global__ void __launch_bounds__(128) ri_2c_metric_kernel(
    int lP, int lQ, bool same_class, ShellPairView bra_aux, ShellPairView ket_aux, int naux,
    double* __restrict__ d_metric) {
  const long long nket = ket_aux.npair;
  const long long total = static_cast<long long>(bra_aux.npair) * nket;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

  const int ncP = n_cart(lP), ncQ = n_cart(lQ);
  const int nsP = n_sph(lP), nsQ = n_sph(lQ);
  CartTriple tP[15], tQ[15];
  for (int i = 0; i < ncP; ++i) tP[i] = cart_triple(lP, i);
  for (int j = 0; j < ncQ; ++j) tQ[j] = cart_triple(lQ, j);

  const int n_roots = (lP + lQ) / 2 + 1;

  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int P = static_cast<int>(idx / nket);
    const int Q = static_cast<int>(idx - static_cast<long long>(P) * nket);
    if (same_class && Q > P) continue;

    const int p0 = bra_aux.ao_i[P];
    const int q0 = ket_aux.ao_i[Q];
    const int bra_begin = bra_aux.prim_offset[P], bra_end = bra_aux.prim_offset[P + 1];
    const int ket_begin = ket_aux.prim_offset[Q], ket_end = ket_aux.prim_offset[Q + 1];

    const double Ax = bra_aux.Ax[P], Ay = bra_aux.Ay[P], Az = bra_aux.Az[P];
    const double Cx = ket_aux.Ax[Q], Cy = ket_aux.Ay[Q], Cz = ket_aux.Az[Q];
    const double PQx = Ax - Cx, PQy = Ay - Cy, PQz = Az - Cz;
    const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;

    if (lP == 0 && lQ == 0) {
      double val = 0.0;
      for (int ip = bra_begin; ip < bra_end; ++ip) {
        const double p = bra_aux.exponent[ip];
        const double cP = bra_aux.prefactor[ip];
        for (int kp = ket_begin; kp < ket_end; ++kp) {
          const double q = ket_aux.exponent[kp];
          const double cQ = ket_aux.prefactor[kp];
          const double rho = p * q / (p + q);
          val += cP * cQ * kJKTwoOverSqrtPi * sqrt(rho) * boys_f0(rho * PQ2);
        }
      }
      d_metric[static_cast<std::size_t>(p0) * naux + q0] = val;
      d_metric[static_cast<std::size_t>(q0) * naux + p0] = val;
      continue;
    }

    double cart[225] = {0.0};
    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const double p = bra_aux.exponent[ip];
      const double cP = bra_aux.prefactor[ip];
      const double inv_2p = 0.5 / p;
      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const double q = ket_aux.exponent[kp];
        const double cQ = ket_aux.prefactor[kp];
        const double pq_sum = p + q;
        const double rho = p * q / pq_sum;
        const double T = rho * PQ2;
        const double scale = cP * cQ * kJKTwoOverSqrtPi * sqrt(rho);

        double u_roots[5], w_weights[5];
        rys_roots_weights<double>(n_roots, T, u_roots, w_weights);

        const double inv_2q = 0.5 / q;
        const double inv_2pq = 0.5 / pq_sum;
        const double rho_over_p = rho / p;
        const double rho_over_q = rho / q;

        for (int r = 0; r < n_roots; ++r) {
          const double u = u_roots[r];
          const double Wr = scale * w_weights[r];
          const double B00 = inv_2pq * u;
          const double B10 = inv_2p * (1.0 - rho_over_p * u);
          const double B01 = inv_2q * (1.0 - rho_over_q * u);

          double Gx[5][5], Gy[5][5], Gz[5][5];
          rys_2c_1d(lP, lQ, -rho_over_p * u * PQx, rho_over_q * u * PQx, B10, B01, B00, Gx);
          rys_2c_1d(lP, lQ, -rho_over_p * u * PQy, rho_over_q * u * PQy, B10, B01, B00, Gy);
          rys_2c_1d(lP, lQ, -rho_over_p * u * PQz, rho_over_q * u * PQz, B10, B01, B00, Gz);

          for (int i = 0; i < ncP; ++i) {
            for (int j = 0; j < ncQ; ++j) {
              cart[i * ncQ + j] +=
                  Wr * Gx[tP[i].lx][tQ[j].lx] * Gy[tP[i].ly][tQ[j].ly] * Gz[tP[i].lz][tQ[j].lz];
            }
          }
        }
      }
    }

    double sph[81];
    harmonics::cart_to_sph_2d<double>(lP, lQ, cart, sph);
    for (int mp = 0; mp < nsP; ++mp) {
      for (int mq = 0; mq < nsQ; ++mq) {
        const double v = sph[mp * nsQ + mq];
        d_metric[static_cast<std::size_t>(p0 + mp) * naux + (q0 + mq)] = v;
        d_metric[static_cast<std::size_t>(q0 + mq) * naux + (p0 + mp)] = v;
      }
    }
  }
}

template <int MaxOrbL>
__device__ inline void rys_3c_1d(int la, int lb, int lP, double C00, double D00, double B10,
                                 double B01, double B00, double AB,
                                 double H[MaxOrbL + 1][MaxOrbL + 1][5]) {
  constexpr int kLab = 2 * MaxOrbL + 1;
  const int lab = la + lb;
  double G[kLab][5];
  G[0][0] = 1.0;
  for (int n = 0; n < lab; ++n) {
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? static_cast<double>(n) * B10 * G[n - 1][0] : 0.0);
  }
  for (int m = 0; m < lP; ++m) {
    for (int n = 0; n <= lab; ++n) {
      double val = D00 * G[n][m];
      if (m > 0) val += static_cast<double>(m) * B01 * G[n][m - 1];
      if (n > 0) val += static_cast<double>(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }
  }

  for (int j = 0; j <= lb; ++j) {
    for (int i = 0; i <= la; ++i) {
      for (int m = 0; m <= lP; ++m) H[i][j][m] = G[i][m];
    }
    if (j < lb) {
      for (int i = 0; i <= lab - j - 1; ++i) {
        for (int m = 0; m <= lP; ++m) G[i][m] = G[i + 1][m] + AB * G[i][m];
      }
    }
  }
}

// Fast high-occupancy 3-center `(ab | P)` kernel for `s, p, d` orbital shells (`la, lb <= 2`)
// and arbitrary auxiliary shells (`0 <= lP <= 4`), scattering directly into GPU `d_B`.
__global__ void __launch_bounds__(128) ri_3c_spd_kernel(
    int la, int lb, int lP, ShellPairView orb, ShellPairView aux, int nao,
    double* __restrict__ d_B) {
  const long long naux_p = aux.npair;
  const long long total = static_cast<long long>(orb.npair) * naux_p;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;

  const int nca = n_cart(la), ncb = n_cart(lb), ncP = n_cart(lP);
  const int nsa = n_sph(la), nsb = n_sph(lb), nsP = n_sph(lP);
  CartTriple ta[6], tb[6], tP[15];
  for (int i = 0; i < nca; ++i) ta[i] = cart_triple(la, i);
  for (int j = 0; j < ncb; ++j) tb[j] = cart_triple(lb, j);
  for (int k = 0; k < ncP; ++k) tP[k] = cart_triple(lP, k);

  const int n_roots = (la + lb + lP) / 2 + 1;

  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int ab = static_cast<int>(idx / naux_p);
    const int P = static_cast<int>(idx - static_cast<long long>(ab) * naux_p);

    const int u0 = orb.ao_i[ab];
    const int v0 = orb.ao_j[ab];
    const int p0 = aux.ao_i[P];
    const int bra_begin = orb.prim_offset[ab], bra_end = orb.prim_offset[ab + 1];
    const int ket_begin = aux.prim_offset[P], ket_end = aux.prim_offset[P + 1];

    const double Cx = aux.Ax[P], Cy = aux.Ay[P], Cz = aux.Az[P];

    // Exact scalar fast path for (ss|s)
    if (la == 0 && lb == 0 && lP == 0) {
      double val = 0.0;
      for (int ip = bra_begin; ip < bra_end; ++ip) {
        const double p = orb.exponent[ip];
        const double PQx = orb.Px[ip] - Cx;
        const double PQy = orb.Py[ip] - Cy;
        const double PQz = orb.Pz[ip] - Cz;
        const double Cab = orb.prefactor[ip];
        const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;
        for (int kp = ket_begin; kp < ket_end; ++kp) {
          const double q = aux.exponent[kp];
          const double cP = aux.prefactor[kp];
          const double rho = p * q / (p + q);
          val += Cab * cP * kJKTwoOverSqrtPi * sqrt(rho) * boys_f0(rho * PQ2);
        }
      }
      double* row_ptr = d_B + static_cast<std::size_t>(p0) * nao2;
      row_ptr[static_cast<std::size_t>(u0) * nao + v0] = val;
      row_ptr[static_cast<std::size_t>(v0) * nao + u0] = val;
      continue;
    }

    const double Ax = orb.Ax[ab], Ay = orb.Ay[ab], Az = orb.Az[ab];
    const double Bx = orb.Bx[ab], By = orb.By[ab], Bz = orb.Bz[ab];
    const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;

    const int n_comp = nca * ncb * ncP;
    double cart[540];
    for (int c = 0; c < n_comp; ++c) cart[c] = 0.0;

    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const double p = orb.exponent[ip];
      const double Px = orb.Px[ip], Py = orb.Py[ip], Pz = orb.Pz[ip];
      const double Cab = orb.prefactor[ip];
      const double PAx = Px - Ax, PAy = Py - Ay, PAz = Pz - Az;
      const double PQx = Px - Cx, PQy = Py - Cy, PQz = Pz - Cz;
      const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;
      const double inv_2p = 0.5 / p;

      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const double q = aux.exponent[kp];
        const double cP = aux.prefactor[kp];
        const double pq_sum = p + q;
        const double rho = p * q / pq_sum;
        const double T = rho * PQ2;
        const double scale = Cab * cP * kJKTwoOverSqrtPi * sqrt(rho);

        double u_roots[5], w_weights[5];
        rys_roots_weights<double>(n_roots, T, u_roots, w_weights);

        const double inv_2q = 0.5 / q;
        const double inv_2pq = 0.5 / pq_sum;
        const double rho_over_p = rho / p;
        const double rho_over_q = rho / q;

        for (int r = 0; r < n_roots; ++r) {
          const double u = u_roots[r];
          const double Wr = scale * w_weights[r];
          const double B00 = inv_2pq * u;
          const double B10 = inv_2p * (1.0 - rho_over_p * u);
          const double B01 = inv_2q * (1.0 - rho_over_q * u);

          double Hx[3][3][5], Hy[3][3][5], Hz[3][3][5];
          rys_3c_1d<2>(la, lb, lP, PAx - rho_over_p * u * PQx, rho_over_q * u * PQx, B10, B01, B00,
                       ABx, Hx);
          rys_3c_1d<2>(la, lb, lP, PAy - rho_over_p * u * PQy, rho_over_q * u * PQy, B10, B01, B00,
                       ABy, Hy);
          rys_3c_1d<2>(la, lb, lP, PAz - rho_over_p * u * PQz, rho_over_q * u * PQz, B10, B01, B00,
                       ABz, Hz);

          int out_i = 0;
          for (int ia = 0; ia < nca; ++ia) {
            for (int ib_c = 0; ib_c < ncb; ++ib_c) {
              for (int iP = 0; iP < ncP; ++iP) {
                cart[out_i++] += Wr * Hx[ta[ia].lx][tb[ib_c].lx][tP[iP].lx] *
                                 Hy[ta[ia].ly][tb[ib_c].ly][tP[iP].ly] *
                                 Hz[ta[ia].lz][tb[ib_c].lz][tP[iP].lz];
              }
            }
          }
        }
      }
    }

    // Sequential 1D Cart->Sph along P, then b, then a (executed ONCE per contracted triplet!)
    double tmp_ab_sP[6 * 6 * 9];
    for (int iab = 0; iab < nca * ncb; ++iab) {
      harmonics::cart_to_sph_axis<double>(lP, cart + iab * ncP, 1, tmp_ab_sP + iab * nsP, 1);
    }
    double tmp_a_sb_sP[6 * 5 * 9];
    for (int ia = 0; ia < nca; ++ia) {
      for (int mp = 0; mp < nsP; ++mp) {
        harmonics::cart_to_sph_axis<double>(lb, tmp_ab_sP + ia * ncb * nsP + mp, nsP,
                                            tmp_a_sb_sP + ia * nsb * nsP + mp, nsP);
      }
    }
    double sph[5 * 5 * 9];
    for (int mbp = 0; mbp < nsb * nsP; ++mbp) {
      harmonics::cart_to_sph_axis<double>(la, tmp_a_sb_sP + mbp, nsb * nsP, sph + mbp, nsb * nsP);
    }

    for (int mp = 0; mp < nsP; ++mp) {
      double* row_ptr = d_B + static_cast<std::size_t>(p0 + mp) * nao2;
      for (int ma = 0; ma < nsa; ++ma) {
        const int u = u0 + ma;
        for (int mb = 0; mb < nsb; ++mb) {
          const int v = v0 + mb;
          const double val = sph[(ma * nsb + mb) * nsP + mp];
          row_ptr[static_cast<std::size_t>(u) * nao + v] = val;
          row_ptr[static_cast<std::size_t>(v) * nao + u] = val;
        }
      }
    }
  }
}

// General 3-center `(ab | P)` kernel for `f, g` orbital shells (`la >= 3 || lb >= 3`).
__global__ void __launch_bounds__(64) ri_3c_high_l_kernel(
    int la, int lb, int lP, ShellPairView orb, ShellPairView aux, int nao,
    double* __restrict__ d_B) {
  const long long naux_p = aux.npair;
  const long long total = static_cast<long long>(orb.npair) * naux_p;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;

  const int nca = n_cart(la), ncb = n_cart(lb), ncP = n_cart(lP);
  const int nsa = n_sph(la), nsb = n_sph(lb), nsP = n_sph(lP);
  CartTriple ta[15], tb[15], tP[15];
  for (int i = 0; i < nca; ++i) ta[i] = cart_triple(la, i);
  for (int j = 0; j < ncb; ++j) tb[j] = cart_triple(lb, j);
  for (int k = 0; k < ncP; ++k) tP[k] = cart_triple(lP, k);

  double ca_tab[15][9] = {{0.0}};
  double cb_tab[15][9] = {{0.0}};
  double unit[15] = {0.0};
  for (int ia = 0; ia < nca; ++ia) {
    unit[ia] = 1.0;
    harmonics::cart_to_sph_axis<double>(la, unit, 1, ca_tab[ia], 1);
    unit[ia] = 0.0;
  }
  for (int ib_c = 0; ib_c < ncb; ++ib_c) {
    unit[ib_c] = 1.0;
    harmonics::cart_to_sph_axis<double>(lb, unit, 1, cb_tab[ib_c], 1);
    unit[ib_c] = 0.0;
  }

  const int n_roots = (la + lb + lP) / 2 + 1;
  const int n_sph_tot = nsa * nsb * nsP;

  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int ab = static_cast<int>(idx / naux_p);
    const int P = static_cast<int>(idx - static_cast<long long>(ab) * naux_p);

    const int u0 = orb.ao_i[ab];
    const int v0 = orb.ao_j[ab];
    const int p0 = aux.ao_i[P];
    const int bra_begin = orb.prim_offset[ab], bra_end = orb.prim_offset[ab + 1];
    const int ket_begin = aux.prim_offset[P], ket_end = aux.prim_offset[P + 1];

    const double Ax = orb.Ax[ab], Ay = orb.Ay[ab], Az = orb.Az[ab];
    const double Bx = orb.Bx[ab], By = orb.By[ab], Bz = orb.Bz[ab];
    const double Cx = aux.Ax[P], Cy = aux.Ay[P], Cz = aux.Az[P];
    const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;

    double sph[9 * 9 * 9];
    for (int s = 0; s < n_sph_tot; ++s) sph[s] = 0.0;

    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const double p = orb.exponent[ip];
      const double Px = orb.Px[ip], Py = orb.Py[ip], Pz = orb.Pz[ip];
      const double Cab = orb.prefactor[ip];
      const double PAx = Px - Ax, PAy = Py - Ay, PAz = Pz - Az;
      const double PQx = Px - Cx, PQy = Py - Cy, PQz = Pz - Cz;
      const double PQ2 = PQx * PQx + PQy * PQy + PQz * PQz;
      const double inv_2p = 0.5 / p;

      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const double q = aux.exponent[kp];
        const double cP = aux.prefactor[kp];
        const double pq_sum = p + q;
        const double rho = p * q / pq_sum;
        const double T = rho * PQ2;
        const double scale = Cab * cP * kJKTwoOverSqrtPi * sqrt(rho);

        double u_roots[7], w_weights[7];
        rys_roots_weights<double>(n_roots, T, u_roots, w_weights);

        const double inv_2q = 0.5 / q;
        const double inv_2pq = 0.5 / pq_sum;
        const double rho_over_p = rho / p;
        const double rho_over_q = rho / q;

        double Hx_r[7][5][5][5], Hy_r[7][5][5][5], Hz_r[7][5][5][5];
        double W_r[7];
        for (int r = 0; r < n_roots; ++r) {
          const double u = u_roots[r];
          W_r[r] = scale * w_weights[r];
          const double B00 = inv_2pq * u;
          const double B10 = inv_2p * (1.0 - rho_over_p * u);
          const double B01 = inv_2q * (1.0 - rho_over_q * u);
          rys_3c_1d<4>(la, lb, lP, PAx - rho_over_p * u * PQx, rho_over_q * u * PQx, B10, B01, B00,
                       ABx, Hx_r[r]);
          rys_3c_1d<4>(la, lb, lP, PAy - rho_over_p * u * PQy, rho_over_q * u * PQy, B10, B01, B00,
                       ABy, Hy_r[r]);
          rys_3c_1d<4>(la, lb, lP, PAz - rho_over_p * u * PQz, rho_over_q * u * PQz, B10, B01, B00,
                       ABz, Hz_r[r]);
        }

        double cart_P[15], sph_P[9];
        for (int ia = 0; ia < nca; ++ia) {
          for (int ib_c = 0; ib_c < ncb; ++ib_c) {
            for (int iP = 0; iP < ncP; ++iP) {
              double sum_r = 0.0;
              for (int r = 0; r < n_roots; ++r) {
                sum_r += W_r[r] * Hx_r[r][ta[ia].lx][tb[ib_c].lx][tP[iP].lx] *
                         Hy_r[r][ta[ia].ly][tb[ib_c].ly][tP[iP].ly] *
                         Hz_r[r][ta[ia].lz][tb[ib_c].lz][tP[iP].lz];
              }
              cart_P[iP] = sum_r;
            }
            harmonics::cart_to_sph_axis<double>(lP, cart_P, 1, sph_P, 1);
            for (int ma = 0; ma < nsa; ++ma) {
              const double wa = ca_tab[ia][ma];
              if (wa == 0.0) continue;
              for (int mb = 0; mb < nsb; ++mb) {
                const double wab = wa * cb_tab[ib_c][mb];
                if (wab == 0.0) continue;
                double* dst = sph + (ma * nsb + mb) * nsP;
                for (int mp = 0; mp < nsP; ++mp) dst[mp] += wab * sph_P[mp];
              }
            }
          }
        }
      }
    }

    for (int mp = 0; mp < nsP; ++mp) {
      double* row_ptr = d_B + static_cast<std::size_t>(p0 + mp) * nao2;
      for (int ma = 0; ma < nsa; ++ma) {
        const int u = u0 + ma;
        for (int mb = 0; mb < nsb; ++mb) {
          const int v = v0 + mb;
          const double val = sph[(ma * nsb + mb) * nsP + mp];
          row_ptr[static_cast<std::size_t>(u) * nao + v] = val;
          row_ptr[static_cast<std::size_t>(v) * nao + u] = val;
        }
      }
    }
  }
}

__global__ void check_cholesky_diag_kernel(int n, const double* __restrict__ d_L, double min_diag,
                                           int* __restrict__ d_bad_flag) {
  const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (i < n) {
    const double d = d_L[static_cast<std::size_t>(i) * n + i];
    if (!(d > min_diag)) {
      atomicExch(d_bad_flag, 1);
    }
  }
}

__global__ void scale_eigvecs_inv_fourth_root_kernel(int n, const double* __restrict__ d_w,
                                                     double eig_cutoff,
                                                     double* __restrict__ d_U_col) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(n) * n;
  if (idx >= total) return;
  const std::size_t k = idx / static_cast<std::size_t>(n);
  const double ev = d_w[k];
  const double sc = (ev > eig_cutoff) ? pow(ev, -0.25) : 0.0;
  d_U_col[idx] *= sc;
}

__global__ void copy_or_symmetrize_dm_kernel(int nao, bool symmetrize,
                                             const double* __restrict__ src,
                                             double* __restrict__ dst) {
  const int total = nao * nao;
  const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
  for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
       idx < total; idx += stride) {
    if (!symmetrize) {
      dst[idx] = src[idx];
    } else {
      const int r = idx / nao;
      const int c = idx - r * nao;
      dst[idx] = 0.5 * (src[r * nao + c] + src[c * nao + r]);
    }
  }
}

__global__ void symmetrize_inplace_jk_kernel(int nao, double* __restrict__ mat) {
  const int i = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) +
                static_cast<int>(threadIdx.y);
  const int j = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                static_cast<int>(threadIdx.x);
  if (i < nao && j < nao && i < j) {
    const int ij = i * nao + j;
    const int ji = j * nao + i;
    const double avg = 0.5 * (mat[ij] + mat[ji]);
    mat[ij] = avg;
    mat[ji] = avg;
  }
}

// Coalesced fused permutation `[bc, nao, nao] -> [nao, bc, nao]` along outer two axes
// while preserving contiguous inner `s` (`0..nao-1`) dimension for both `B_c` and `T_raw`!
template <typename T>
__global__ void permute_bc_pair_kernel(int bc, int nao, const T* __restrict__ src_a,
                                       const T* __restrict__ src_b, T* __restrict__ dst_a,
                                       T* __restrict__ dst_b) {
  const std::size_t total = static_cast<std::size_t>(bc) * nao * nao;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t nao_u = static_cast<std::size_t>(nao);
  const std::size_t nao2_u = nao_u * nao_u;

  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t pl = idx / nao2_u;
    const std::size_t rem = idx - pl * nao2_u;
    const std::size_t u = rem / nao_u;
    const std::size_t s = rem - u * nao_u;
    const std::size_t dst_idx = (u * static_cast<std::size_t>(bc) + pl) * nao_u + s;
    dst_a[dst_idx] = src_a[idx];
    dst_b[dst_idx] = src_b[idx];
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

__global__ void cast_f2d_scale_epilogue_kernel(const float* __restrict__ src, double scale, int nao,
                                               bool symmetrize_out, double* __restrict__ dst) {
  const int total = nao * nao;
  const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
  for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
       idx < total; idx += stride) {
    if (symmetrize_out) {
      const int r = idx / nao;
      const int c = idx - r * nao;
      dst[idx] = scale * 0.5 *
                 (static_cast<double>(src[r * nao + c]) + static_cast<double>(src[c * nao + r]));
    } else {
      dst[idx] = scale * static_cast<double>(src[idx]);
    }
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

template <int MaxL>
void launch_direct_jk_tier(int num_sms, const DevPairClassMeta* d_classes,
                           const DevVecJK<GpuJKTile>& d_tiles, int* d_counter_fp64,
                           int* d_counter_fp32, const int* d_shell_cart_off,
                           const double* d_dm_cart, const double* d_shell_max,
                           const double* d_global_dmax, int nsh, int ncart_tot,
                           const JKOptions& opts, double* d_j_cart, double* d_k_cart,
                           cudaStream_t stream) {
  const int num_tiles = static_cast<int>(d_tiles.size());
  if (num_tiles <= 0) return;

  constexpr int kBlock = 64;  // 2 warps per block
  constexpr int kWarpsPerBlock = kBlock / 32;
  const int max_persistent_blocks = std::max(32, num_sms * (MaxL <= 2 ? 4 : 2));
  const int blocks =
      std::min((num_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock, max_persistent_blocks);
  constexpr double kInf = 1e300;

  if constexpr (MaxL == 1) {
    if (opts.compute_j && !opts.compute_k) {
      if (opts.precision == Precision::FP64) {
        direct_j_hermite_sp_warp_kernel<double><<<blocks, kBlock, 0, stream>>>(
            d_classes, d_tiles.data(), num_tiles, d_counter_fp64, d_shell_cart_off, d_dm_cart,
            d_shell_max, d_global_dmax, nsh, ncart_tot, opts.schwarz_cutoff, kInf,
            opts.schwarz_cutoff, d_j_cart);
      } else if (opts.precision == Precision::FP32) {
        direct_j_hermite_sp_warp_kernel<float><<<blocks, kBlock, 0, stream>>>(
            d_classes, d_tiles.data(), num_tiles, d_counter_fp32, d_shell_cart_off, d_dm_cart,
            d_shell_max, d_global_dmax, nsh, ncart_tot, opts.schwarz_cutoff, kInf,
            opts.schwarz_cutoff, d_j_cart);
      } else {
        const double thr = std::max(opts.fp32_threshold, opts.schwarz_cutoff);
        direct_j_hermite_sp_warp_kernel<double><<<blocks, kBlock, 0, stream>>>(
            d_classes, d_tiles.data(), num_tiles, d_counter_fp64, d_shell_cart_off, d_dm_cart,
            d_shell_max, d_global_dmax, nsh, ncart_tot, thr, kInf, thr, d_j_cart);
        if (opts.fp32_threshold > opts.schwarz_cutoff) {
          direct_j_hermite_sp_warp_kernel<float><<<blocks, kBlock, 0, stream>>>(
              d_classes, d_tiles.data(), num_tiles, d_counter_fp32, d_shell_cart_off, d_dm_cart,
              d_shell_max, d_global_dmax, nsh, ncart_tot, opts.schwarz_cutoff, opts.fp32_threshold,
              opts.schwarz_cutoff, d_j_cart);
        }
      }
      check_cuda_jk(cudaGetLastError(), "direct_j_hermite_sp_warp_kernel launch");
      return;
    }
  }

  auto launch_rys = [&](auto prec_tag, int* cnt, double e_min, double e_max, double early_cut) {
    using R = decltype(prec_tag);
    if (opts.symmetrize_dm) {
      direct_jk_rys_warp_kernel<MaxL, R, true><<<blocks, kBlock, 0, stream>>>(
          d_classes, d_tiles.data(), num_tiles, cnt, d_shell_cart_off, d_dm_cart, d_shell_max,
          d_global_dmax, nsh, ncart_tot, opts.compute_j, opts.compute_k, e_min, e_max, early_cut,
          d_j_cart, d_k_cart);
    } else {
      direct_jk_rys_warp_kernel<MaxL, R, false><<<blocks, kBlock, 0, stream>>>(
          d_classes, d_tiles.data(), num_tiles, cnt, d_shell_cart_off, d_dm_cart, d_shell_max,
          d_global_dmax, nsh, ncart_tot, opts.compute_j, opts.compute_k, e_min, e_max, early_cut,
          d_j_cart, d_k_cart);
    }
  };

  if (opts.precision == Precision::FP64) {
    launch_rys(double(0), d_counter_fp64, opts.schwarz_cutoff, kInf, opts.schwarz_cutoff);
  } else if (opts.precision == Precision::FP32) {
    launch_rys(float(0), d_counter_fp32, opts.schwarz_cutoff, kInf, opts.schwarz_cutoff);
  } else {
    const double thr = std::max(opts.fp32_threshold, opts.schwarz_cutoff);
    launch_rys(double(0), d_counter_fp64, thr, kInf, thr);
    if (opts.fp32_threshold > opts.schwarz_cutoff) {
      launch_rys(float(0), d_counter_fp32, opts.schwarz_cutoff, opts.fp32_threshold,
                 opts.schwarz_cutoff);
    }
  }
  check_cuda_jk(cudaGetLastError(), "direct_jk_rys_warp_kernel launch");
}

}  // namespace

// ---------------------------------------------------------------------------
// DirectJKCudaCache Implementation
// ---------------------------------------------------------------------------

struct DirectJKCudaCache {
  int nao = 0;
  int nsh = 0;
  int ncart_total = 0;
  int num_sms = 64;

  std::vector<DevShellPairList> dev_lists;
  DevVecJK<DevPairClassMeta> d_classes;
  DevVecJK<int> d_shell_l;
  DevVecJK<int> d_shell_ao_off;
  DevVecJK<int> d_shell_cart_off;
  DevVecJK<double> d_harm_c;
  DevVecJK<GpuJKTile> d_tier_tiles[4];

  // Persistent pre-allocated device buffers (zero cudaMalloc per SCF step!)
  DevVecJK<int> d_counters;
  DevVecJK<double> d_global_dmax;
  DevVecJK<double> d_dm_cart;
  DevVecJK<double> d_shell_max;
  DevVecJK<double> d_j_cart;
  DevVecJK<double> d_k_cart;

  // Host-pointer staging buffers (used only by `compute_jk` host wrapper)
  DevVecJK<double> d_dm_sph;
  DevVecJK<double> d_vj_sph;
  DevVecJK<double> d_vk_sph;
};

DirectJKCudaCache* create_direct_jk_cuda_cache(
    const Basis& basis, const std::vector<ShellPairData>& pair_lists,
    const std::vector<int>& shell_cart_offset, int ncart_total) {
  auto cache = std::make_unique<DirectJKCudaCache>();
  cache->nao = basis.nao();
  cache->nsh = static_cast<int>(basis.nshell());
  cache->ncart_total = ncart_total;

  int dev = 0;
  if (cudaGetDevice(&dev) == cudaSuccess) {
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        sms > 0) {
      cache->num_sms = sms;
    }
  }

  // 1. Upload shell metadata + reduced solid-harmonic table [5 x 9 x 15].
  std::vector<int> h_shell_l(cache->nsh, 0);
  std::vector<int> h_shell_ao_off(cache->nsh, 0);
  for (int s = 0; s < cache->nsh; ++s) {
    h_shell_l[s] = basis.shell(s).l;
    h_shell_ao_off[s] = basis.ao_offset(s);
  }
  cache->d_shell_l.upload(h_shell_l.data(), h_shell_l.size());
  cache->d_shell_ao_off.upload(h_shell_ao_off.data(), h_shell_ao_off.size());
  cache->d_shell_cart_off.upload(shell_cart_offset.data(), shell_cart_offset.size());

  std::vector<double> h_harm_c(5 * 9 * 15, 0.0);
  for (int l = 0; l <= 4; ++l) {
    const int nc = n_cart(l);
    const int ns = n_sph(l);
    double unit[15] = {0.0};
    double col[9] = {0.0};
    for (int ic = 0; ic < nc; ++ic) {
      unit[ic] = 1.0;
      harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
      for (int m = 0; m < ns; ++m) {
        h_harm_c[(l * 9 + m) * 15 + ic] = col[m];
      }
      unit[ic] = 0.0;
    }
  }
  cache->d_harm_c.upload(h_harm_c.data(), h_harm_c.size());

  // 2. Upload SoA shell-pair lists.
  cache->dev_lists.reserve(pair_lists.size());
  std::vector<DevPairClassMeta> metas;
  metas.reserve(pair_lists.size());
  for (const ShellPairData& pd : pair_lists) {
    cache->dev_lists.emplace_back(pd);
    metas.push_back(DevPairClassMeta{pd.la, pd.lb, cache->dev_lists.back().view()});
  }
  cache->d_classes.upload(metas.data(), metas.size());

  // 3. Prebuild 2D `(bra P, ket Q0..Q1)` tiles partitioned into 4 angular-momentum tiers
  //    (MaxL = 1, 2, 3, 4), each sorted in descending `max_schwarz` order!
  std::vector<GpuJKTile> tier_tiles_host[4];
  constexpr double kStaticSchwarzFloor = 1e-18;

  for (int bp = 0; bp < static_cast<int>(pair_lists.size()); ++bp) {
    const ShellPairData& bra = pair_lists[bp];
    for (int kp = 0; kp <= bp; ++kp) {
      const ShellPairData& ket = pair_lists[kp];
      const int max_l = std::max({1, bra.la, bra.lb, ket.la, ket.lb});
      const int tier = std::min(max_l, 4) - 1;
      auto& dst_vec = tier_tiles_host[tier];

      for (int P = 0; P < bra.npair(); ++P) {
        const double q_ab = bra.schwarz_bound[P];
        if (q_ab <= 0.0) break;
        const int q_limit = (bp == kp) ? (P + 1) : ket.npair();
        for (int Q0 = 0; Q0 < q_limit; Q0 += kGpuKetTile) {
          const double max_s = q_ab * ket.schwarz_bound[Q0];
          if (max_s < kStaticSchwarzFloor) break;
          const int Q1 = std::min(Q0 + kGpuKetTile, q_limit);
          dst_vec.push_back(GpuJKTile{bp, P, kp, Q0, Q1, max_s});
        }
      }
    }
  }

  for (int t = 0; t < 4; ++t) {
    auto& v = tier_tiles_host[t];
    std::sort(v.begin(), v.end(),
              [](const GpuJKTile& a, const GpuJKTile& b) { return a.max_schwarz > b.max_schwarz; });
    cache->d_tier_tiles[t].upload(v.data(), v.size());
  }

  // 4. Pre-allocate persistent working buffers once!
  const std::size_t nc2 = static_cast<std::size_t>(ncart_total) * ncart_total;
  const std::size_t nsh2 = static_cast<std::size_t>(cache->nsh) * cache->nsh;
  const std::size_t nao2 = static_cast<std::size_t>(cache->nao) * cache->nao;
  cache->d_counters.resize(8);
  cache->d_global_dmax.resize(1);
  cache->d_dm_cart.resize(nc2);
  cache->d_shell_max.resize(nsh2);
  cache->d_j_cart.resize(nc2);
  cache->d_k_cart.resize(nc2);
  cache->d_dm_sph.resize(nao2);
  cache->d_vj_sph.resize(nao2);
  cache->d_vk_sph.resize(nao2);

  return cache.release();
}

void destroy_direct_jk_cuda_cache(DirectJKCudaCache* p) { delete p; }

void run_direct_jk_cuda_device(DirectJKCudaCache& cache, const double* d_dm, double* d_vj,
                               double* d_vk, const JKOptions& opts, void* raw_stream) {
  auto stream = static_cast<cudaStream_t>(raw_stream);
  const bool do_j = opts.compute_j && (d_vj != nullptr);
  const bool do_k = opts.compute_k && (d_vk != nullptr);
  if ((!do_j && !do_k) || cache.nao == 0) return;

  // When only J is requested, symmetrizing D is always algebraically exact!
  const bool eff_sym_dm = opts.symmetrize_dm || !do_k;

  cache.d_counters.zero_async(stream);
  cache.d_global_dmax.zero_async(stream);
  if (do_j) cache.d_j_cart.zero_async(stream);
  if (do_k) cache.d_k_cart.zero_async(stream);

  const int nsh2 = cache.nsh * cache.nsh;
  const int prep_blocks = (nsh2 + 127) / 128;
  prepare_cartesian_density_kernel<<<prep_blocks, 128, 0, stream>>>(
      cache.nsh, cache.nao, cache.ncart_total, eff_sym_dm, cache.d_shell_l.data(),
      cache.d_shell_ao_off.data(), cache.d_shell_cart_off.data(), cache.d_harm_c.data(), d_dm,
      cache.d_dm_cart.data(), cache.d_shell_max.data(), cache.d_global_dmax.data());
  check_cuda_jk(cudaGetLastError(), "prepare_cartesian_density_kernel launch");

  JKOptions k_opts = opts;
  k_opts.compute_j = do_j;
  k_opts.compute_k = do_k;
  k_opts.symmetrize_dm = eff_sym_dm;

  double* j_ptr = do_j ? cache.d_j_cart.data() : nullptr;
  double* k_ptr = do_k ? cache.d_k_cart.data() : nullptr;

  launch_direct_jk_tier<1>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[0],
                           cache.d_counters.data() + 0, cache.d_counters.data() + 4,
                           cache.d_shell_cart_off.data(), cache.d_dm_cart.data(),
                           cache.d_shell_max.data(), cache.d_global_dmax.data(), cache.nsh,
                           cache.ncart_total, k_opts, j_ptr, k_ptr, stream);
  launch_direct_jk_tier<2>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[1],
                           cache.d_counters.data() + 1, cache.d_counters.data() + 5,
                           cache.d_shell_cart_off.data(), cache.d_dm_cart.data(),
                           cache.d_shell_max.data(), cache.d_global_dmax.data(), cache.nsh,
                           cache.ncart_total, k_opts, j_ptr, k_ptr, stream);
  launch_direct_jk_tier<3>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[2],
                           cache.d_counters.data() + 2, cache.d_counters.data() + 6,
                           cache.d_shell_cart_off.data(), cache.d_dm_cart.data(),
                           cache.d_shell_max.data(), cache.d_global_dmax.data(), cache.nsh,
                           cache.ncart_total, k_opts, j_ptr, k_ptr, stream);
  launch_direct_jk_tier<4>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[3],
                           cache.d_counters.data() + 3, cache.d_counters.data() + 7,
                           cache.d_shell_cart_off.data(), cache.d_dm_cart.data(),
                           cache.d_shell_max.data(), cache.d_global_dmax.data(), cache.nsh,
                           cache.ncart_total, k_opts, j_ptr, k_ptr, stream);

  if (do_j) {
    finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
        cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/true, 1.0,
        cache.d_shell_l.data(), cache.d_shell_ao_off.data(), cache.d_shell_cart_off.data(),
        cache.d_harm_c.data(), cache.d_j_cart.data(), d_vj);
    check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (J) launch");
  }
  if (do_k) {
    finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
        cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/opts.symmetrize_dm,
        opts.k_scale, cache.d_shell_l.data(), cache.d_shell_ao_off.data(),
        cache.d_shell_cart_off.data(), cache.d_harm_c.data(), cache.d_k_cart.data(), d_vk);
    check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (K) launch");
  }
}

void run_direct_jk_cuda_host(DirectJKCudaCache& cache, const double* h_dm, double* h_vj,
                             double* h_vk, const JKOptions& opts) {
  const std::size_t nao2 = static_cast<std::size_t>(cache.nao) * cache.nao;
  const bool do_j = opts.compute_j && (h_vj != nullptr);
  const bool do_k = opts.compute_k && (h_vk != nullptr);
  if ((!do_j && !do_k) || nao2 == 0) return;

  cache.d_dm_sph.upload_async(h_dm, nao2, nullptr);
  run_direct_jk_cuda_device(cache, cache.d_dm_sph.data(), do_j ? cache.d_vj_sph.data() : nullptr,
                            do_k ? cache.d_vk_sph.data() : nullptr, opts, nullptr);
  if (do_j) cache.d_vj_sph.download_async(h_vj, nao2, nullptr);
  if (do_k) cache.d_vk_sph.download_async(h_vk, nao2, nullptr);
  check_cuda_jk(cudaStreamSynchronize(nullptr), "run_direct_jk_cuda_host sync");
}

// ---------------------------------------------------------------------------
// RIJKCudaCache Implementation (100% GPU-Native Setup + Chunked Contraction)
// ---------------------------------------------------------------------------

struct RIJKCudaCache {
  int nao = 0;
  int naux = 0;
  cublasHandle_t handle = nullptr;
  DevVecJK<double> d_B;  // [naux * nao * nao] resident on GPU!

  // Persistent reusable contraction workspaces (zero duplicate full `d_B_perm`!)
  DevVecJK<double> d_dm_work;
  DevVecJK<double> d_v_aux;
  DevVecJK<double> d_T_raw;
  DevVecJK<double> d_Bp_chunk;
  DevVecJK<double> d_Tp_chunk;

  // FP32 workspaces (allocated on demand if Precision::FP32 is requested)
  DevVecJK<float> d_dm_f;
  DevVecJK<float> d_Bc_f;
  DevVecJK<float> d_T_raw_f;
  DevVecJK<float> d_Bp_chunk_f;
  DevVecJK<float> d_Tp_chunk_f;
  DevVecJK<float> d_vk_f;

  // Host-pointer staging buffers (used only by `compute_jk` host wrapper)
  DevVecJK<double> d_dm_stage;
  DevVecJK<double> d_vj_stage;
  DevVecJK<double> d_vk_stage;

  ~RIJKCudaCache() {
    if (handle != nullptr) cublasDestroy(handle);
  }
};

namespace {

void init_rijk_workspaces(RIJKCudaCache& c) {
  const std::size_t nao2 = static_cast<std::size_t>(c.nao) * c.nao;
  const int bc_max = std::min(c.naux, kRiAuxChunk);
  const std::size_t chunk_elems = static_cast<std::size_t>(bc_max) * nao2;
  c.d_dm_work.resize(nao2);
  c.d_v_aux.resize(static_cast<std::size_t>(c.naux));
  c.d_T_raw.resize(chunk_elems);
  c.d_Bp_chunk.resize(chunk_elems);
  c.d_Tp_chunk.resize(chunk_elems);
  c.d_dm_stage.resize(nao2);
  c.d_vj_stage.resize(nao2);
  c.d_vk_stage.resize(nao2);
}

}  // namespace

RIJKCudaCache* build_rijk_cuda_cache_native(int nao, int naux,
                                            const std::vector<ShellPairData>& orb_lists,
                                            const std::vector<ShellPairData>& aux_lists) {
  auto c = std::make_unique<RIJKCudaCache>();
  c->nao = nao;
  c->naux = naux;
  check_cublas_jk(cublasCreate(&c->handle), "cublasCreate (RIJK)");
  if (nao == 0 || naux == 0) return c.release();

  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const std::size_t total_b = static_cast<std::size_t>(naux) * nao2;
  const std::size_t naux2 = static_cast<std::size_t>(naux) * naux;

  c->d_B.resize(total_b);
  c->d_B.zero();
  DevVecJK<double> d_metric(naux2);
  d_metric.zero();

  // Upload shell-pair lists once for 2c and 3c kernel evaluation.
  std::vector<DevShellPairList> dev_aux;
  dev_aux.reserve(aux_lists.size());
  for (const auto& ap : aux_lists) dev_aux.emplace_back(ap);

  std::vector<DevShellPairList> dev_orb;
  dev_orb.reserve(orb_lists.size());
  for (const auto& op : orb_lists) dev_orb.emplace_back(op);

  // 1. Evaluate & scatter 2-center metric V_{PQ} = (P | Q) directly on GPU.
  for (std::size_t ip = 0; ip < dev_aux.size(); ++ip) {
    for (std::size_t iq = 0; iq <= ip; ++iq) {
      const long long n_pq =
          static_cast<long long>(dev_aux[ip].npair) * static_cast<long long>(dev_aux[iq].npair);
      if (n_pq == 0) continue;
      const int blocks = static_cast<int>(std::min((n_pq + 127LL) / 128LL, 4096LL));
      ri_2c_metric_kernel<<<blocks, 128>>>(dev_aux[ip].la, dev_aux[iq].la, ip == iq,
                                           dev_aux[ip].view(), dev_aux[iq].view(), naux,
                                           d_metric.data());
      check_cuda_jk(cudaGetLastError(), "ri_2c_metric_kernel launch");
    }
  }

  // 2. Evaluate & scatter 3-center integrals I_{P, uv} = (uv | P) directly into `d_B`!
  for (std::size_t io = 0; io < dev_orb.size(); ++io) {
    const auto& orb = dev_orb[io];
    const bool is_spd = (orb.la <= 2 && orb.lb <= 2);
    for (std::size_t ia = 0; ia < dev_aux.size(); ++ia) {
      const auto& aux = dev_aux[ia];
      const long long n_trip =
          static_cast<long long>(orb.npair) * static_cast<long long>(aux.npair);
      if (n_trip == 0) continue;
      if (is_spd) {
        const int blocks = static_cast<int>(std::min((n_trip + 127LL) / 128LL, 8192LL));
        ri_3c_spd_kernel<<<blocks, 128>>>(orb.la, orb.lb, aux.la, orb.view(), aux.view(), nao,
                                          c->d_B.data());
        check_cuda_jk(cudaGetLastError(), "ri_3c_spd_kernel launch");
      } else {
        const int blocks = static_cast<int>(std::min((n_trip + 63LL) / 64LL, 4096LL));
        ri_3c_high_l_kernel<<<blocks, 64>>>(orb.la, orb.lb, aux.la, orb.view(), aux.view(), nao,
                                            c->d_B.data());
        check_cuda_jk(cudaGetLastError(), "ri_3c_high_l_kernel launch");
      }
    }
  }

  // Free temporary shell-pair device lists before metric factorization.
  dev_orb.clear();
  dev_aux.clear();

  // 3. GPU Cholesky factorization `cusolverDnDpotrf` + in-place `cublasDtrsm`
  //    (with `cusolverDnDsyevd` inverse-sqrt fallback for near-linearly-dependent aux bases).
  DevVecJK<double> d_metric_bak(naux2);
  check_cuda_jk(cudaMemcpy(d_metric_bak.data(), d_metric.data(), naux2 * sizeof(double),
                           cudaMemcpyDeviceToDevice),
                "cudaMemcpy D2D metric backup");

  cusolverDnHandle_t cusolver = nullptr;
  check_cusolver_jk(cusolverDnCreate(&cusolver), "cusolverDnCreate (RIJK)");
  struct CusolverGuard {
    cusolverDnHandle_t h;
    ~CusolverGuard() {
      if (h != nullptr) cusolverDnDestroy(h);
    }
  } guard{cusolver};

  int lwork = 0;
  check_cusolver_jk(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, naux,
                                                d_metric.data(), naux, &lwork),
                    "cusolverDnDpotrf_bufferSize");
  DevVecJK<double> d_work(static_cast<std::size_t>(std::max(1, lwork)));
  DevVecJK<int> d_info(2);
  d_info.zero();

  check_cusolver_jk(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, naux, d_metric.data(), naux,
                                     d_work.data(), lwork, d_info.data()),
                    "cusolverDnDpotrf");
  check_cholesky_diag_kernel<<<(naux + 255) / 256, 256>>>(naux, d_metric.data(),
                                                          3.1622776601683795e-6, d_info.data() + 1);
  check_cuda_jk(cudaGetLastError(), "check_cholesky_diag_kernel");

  int h_info[2] = {0, 0};
  d_info.download(h_info, 2);

  const double one = 1.0, zero = 0.0;
  const int nao2_i = static_cast<int>(nao2);

  if (h_info[0] == 0 && h_info[1] == 0) {
    // Row-major `L * B = I_3c` (`B` is [naux x nao2]) <-> Col-major `B_col * L_col^T = I_3c_col`
    // (`B_col` is [nao2 x naux], solved in-place in `c->d_B`!).
    check_cublas_jk(cublasDtrsm(c->handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, nao2_i, naux, &one, d_metric.data(), naux,
                                c->d_B.data(), nao2_i),
                    "cublasDtrsm (RI metric solve)");
  } else {
    // Near-singular auxiliary metric fallback: GPU eigendecomposition V = U diag(w) U^T -> V^{-1/2}.
    DevVecJK<double> d_w(static_cast<std::size_t>(naux));
    int lwork_ev = 0;
    check_cusolver_jk(
        cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                    naux, d_metric_bak.data(), naux, d_w.data(), &lwork_ev),
        "cusolverDnDsyevd_bufferSize (RI fallback)");
    d_work.resize(static_cast<std::size_t>(std::max(1, lwork_ev)));
    check_cusolver_jk(cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                       naux, d_metric_bak.data(), naux, d_w.data(), d_work.data(),
                                       lwork_ev, d_info.data()),
                      "cusolverDnDsyevd (RI fallback)");

    const int blocks_u = static_cast<int>((naux2 + 255) / 256);
    scale_eigvecs_inv_fourth_root_kernel<<<blocks_u, 256>>>(naux, d_w.data(), 1e-10,
                                                            d_metric_bak.data());
    check_cublas_jk(cublasDgemm(c->handle, CUBLAS_OP_N, CUBLAS_OP_T, naux, naux, naux, &one,
                                d_metric_bak.data(), naux, d_metric_bak.data(), naux, &zero,
                                d_metric.data(), naux),
                    "cublasDgemm (RI V^{-1/2})");

    DevVecJK<double> d_eri3c_tmp(total_b);
    check_cuda_jk(cudaMemcpy(d_eri3c_tmp.data(), c->d_B.data(), total_b * sizeof(double),
                             cudaMemcpyDeviceToDevice),
                  "cudaMemcpy D2D eri3c backup");
    check_cublas_jk(cublasDgemm(c->handle, CUBLAS_OP_N, CUBLAS_OP_N, nao2_i, naux, naux, &one,
                                d_eri3c_tmp.data(), nao2_i, d_metric.data(), naux, &zero,
                                c->d_B.data(), nao2_i),
                    "cublasDgemm (RI V^{-1/2} * I_3c)");
  }

  init_rijk_workspaces(*c);
  check_cuda_jk(cudaDeviceSynchronize(), "build_rijk_cuda_cache_native sync");
  return c.release();
}

RIJKCudaCache* create_rijk_cuda_cache_from_host(int nao, int naux,
                                                const std::vector<double>& b_tensor) {
  auto c = std::make_unique<RIJKCudaCache>();
  c->nao = nao;
  c->naux = naux;
  check_cublas_jk(cublasCreate(&c->handle), "cublasCreate (RIJK from host)");
  c->d_B.upload(b_tensor.data(), b_tensor.size());
  if (nao > 0 && naux > 0) {
    init_rijk_workspaces(*c);
  }
  return c.release();
}

void destroy_rijk_cuda_cache(RIJKCudaCache* p) { delete p; }

const double* rijk_cuda_device_b_ptr(const RIJKCudaCache& cache) noexcept {
  return cache.d_B.data();
}

void download_rijk_b_tensor(const RIJKCudaCache& cache, std::vector<double>& b_out) {
  const std::size_t total = static_cast<std::size_t>(cache.naux) * cache.nao * cache.nao;
  b_out.resize(total);
  cache.d_B.download(b_out.data(), total);
}

void run_rijk_cuda_device(RIJKCudaCache& cache, const double* d_dm, double* d_vj, double* d_vk,
                          const JKOptions& opts, void* raw_stream) {
  auto stream = static_cast<cudaStream_t>(raw_stream);
  const int nao = cache.nao;
  const int naux = cache.naux;
  const int nao2 = nao * nao;
  const bool do_j = opts.compute_j && (d_vj != nullptr);
  const bool do_k = opts.compute_k && (d_vk != nullptr);
  if ((!do_j && !do_k) || nao == 0 || naux == 0) return;

  check_cublas_jk(cublasSetStream(cache.handle, stream), "cublasSetStream (RIJK)");

  const int dm_blocks = std::min((nao2 + 255) / 256, 1024);
  copy_or_symmetrize_dm_kernel<<<dm_blocks, 256, 0, stream>>>(nao, opts.symmetrize_dm, d_dm,
                                                              cache.d_dm_work.data());
  check_cuda_jk(cudaGetLastError(), "copy_or_symmetrize_dm_kernel");

  const double one = 1.0, zero = 0.0;

  if (do_j) {
    // Row-major B [naux x nao2] is col-major B_col [nao2 x naux]:
    //   v_aux = B_col^T * dm_eff,  vj = B_col * v_aux
    check_cublas_jk(cublasDgemv(cache.handle, CUBLAS_OP_T, nao2, naux, &one, cache.d_B.data(),
                                nao2, cache.d_dm_work.data(), 1, &zero, cache.d_v_aux.data(), 1),
                    "cublasDgemv (RI-J step 1)");
    check_cublas_jk(cublasDgemv(cache.handle, CUBLAS_OP_N, nao2, naux, &one, cache.d_B.data(),
                                nao2, cache.d_v_aux.data(), 1, &zero, d_vj, 1),
                    "cublasDgemv (RI-J step 2)");
  }

  if (do_k) {
    if (opts.k_scale == 0.0) {
      check_cuda_jk(cudaMemsetAsync(d_vk, 0, static_cast<std::size_t>(nao2) * sizeof(double), stream),
                    "cudaMemsetAsync (RI-K zero k_scale)");
      return;
    }

    const int bc_max = std::min(naux, kRiAuxChunk);
    const std::size_t max_chunk_elems = static_cast<std::size_t>(bc_max) * nao2;

    if (opts.precision == Precision::FP32) {
      cache.d_dm_f.resize(static_cast<std::size_t>(nao2));
      cache.d_Bc_f.resize(max_chunk_elems);
      cache.d_T_raw_f.resize(max_chunk_elems);
      cache.d_Bp_chunk_f.resize(max_chunk_elems);
      cache.d_Tp_chunk_f.resize(max_chunk_elems);
      cache.d_vk_f.resize(static_cast<std::size_t>(nao2));

      cast_d2f_kernel<<<dm_blocks, 256, 0, stream>>>(cache.d_dm_work.data(), cache.d_dm_f.data(),
                                                     static_cast<std::size_t>(nao2));
      const float f_one = 1.0f;

      for (int p0 = 0; p0 < naux; p0 += kRiAuxChunk) {
        const int bc = std::min(kRiAuxChunk, naux - p0);
        const int m1 = bc * nao;
        const int k_c = bc * nao;
        const std::size_t chunk_n = static_cast<std::size_t>(bc) * nao2;
        const double* d_Bc = cache.d_B.data() + static_cast<std::size_t>(p0) * nao2;
        const int c_blocks = static_cast<int>(std::min<std::size_t>((chunk_n + 255) / 256, 4096));

        cast_d2f_kernel<<<c_blocks, 256, 0, stream>>>(d_Bc, cache.d_Bc_f.data(), chunk_n);
        const float f_zero = 0.0f;
        const float f_beta = (p0 == 0) ? 0.0f : 1.0f;
        check_cublas_jk(cublasSgemm(cache.handle, CUBLAS_OP_N, CUBLAS_OP_N, nao, m1, nao, &f_one,
                                    cache.d_dm_f.data(), nao, cache.d_Bc_f.data(), nao, &f_zero,
                                    cache.d_T_raw_f.data(), nao),
                        "cublasSgemm (RI-K chunk step 1)");
        permute_bc_pair_kernel<float><<<c_blocks, 256, 0, stream>>>(
            bc, nao, cache.d_Bc_f.data(), cache.d_T_raw_f.data(), cache.d_Bp_chunk_f.data(),
            cache.d_Tp_chunk_f.data());
        check_cublas_jk(cublasSgemm(cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, k_c, &f_one,
                                    cache.d_Bp_chunk_f.data(), k_c, cache.d_Tp_chunk_f.data(), k_c,
                                    &f_beta, cache.d_vk_f.data(), nao),
                        "cublasSgemm (RI-K chunk step 2)");
      }
      cast_f2d_scale_epilogue_kernel<<<dm_blocks, 256, 0, stream>>>(
          cache.d_vk_f.data(), opts.k_scale, nao, opts.symmetrize_dm, d_vk);
      check_cuda_jk(cudaGetLastError(), "cast_f2d_scale_epilogue_kernel");
    } else {
      const double alpha = opts.k_scale;
      for (int p0 = 0; p0 < naux; p0 += kRiAuxChunk) {
        const int bc = std::min(kRiAuxChunk, naux - p0);
        const int m1 = bc * nao;
        const int k_c = bc * nao;
        const std::size_t chunk_n = static_cast<std::size_t>(bc) * nao2;
        const double* d_Bc = cache.d_B.data() + static_cast<std::size_t>(p0) * nao2;
        const double beta = (p0 == 0) ? 0.0 : 1.0;

        // Step 1: row-major T_raw [(bc*nao) x nao] = B_c [(bc*nao) x nao] * D_eff [nao x nao]
        check_cublas_jk(cublasDgemm(cache.handle, CUBLAS_OP_N, CUBLAS_OP_N, nao, m1, nao, &one,
                                    cache.d_dm_work.data(), nao, d_Bc, nao, &zero,
                                    cache.d_T_raw.data(), nao),
                        "cublasDgemm (RI-K chunk step 1)");

        // Fused coalesced permute `[bc, nao, nao] -> [nao, bc, nao]` for both `B_c` and `T_raw`
        const int p_blocks = static_cast<int>(std::min<std::size_t>((chunk_n + 255) / 256, 4096));
        permute_bc_pair_kernel<double><<<p_blocks, 256, 0, stream>>>(
            bc, nao, d_Bc, cache.d_T_raw.data(), cache.d_Bp_chunk.data(), cache.d_Tp_chunk.data());

        // Step 2: row-major K [nao x nao] = alpha * T_perm [nao x k_c] * B_perm^T [k_c x nao] + beta * K
        check_cublas_jk(cublasDgemm(cache.handle, CUBLAS_OP_T, CUBLAS_OP_N, nao, nao, k_c, &alpha,
                                    cache.d_Bp_chunk.data(), k_c, cache.d_Tp_chunk.data(), k_c,
                                    &beta, d_vk, nao),
                        "cublasDgemm (RI-K chunk step 2)");
      }
      if (opts.symmetrize_dm) {
        dim3 threads(16, 16);
        dim3 blocks((nao + 15) / 16, (nao + 15) / 16);
        symmetrize_inplace_jk_kernel<<<blocks, threads, 0, stream>>>(nao, d_vk);
        check_cuda_jk(cudaGetLastError(), "symmetrize_inplace_jk_kernel");
      }
    }
  }
}

void run_rijk_cuda(RIJKCudaCache& cache, const double* h_dm, double* h_vj, double* h_vk,
                   const JKOptions& opts) {
  const std::size_t nao2 = static_cast<std::size_t>(cache.nao) * cache.nao;
  const bool do_j = opts.compute_j && (h_vj != nullptr);
  const bool do_k = opts.compute_k && (h_vk != nullptr);
  if ((!do_j && !do_k) || nao2 == 0 || cache.naux == 0) return;

  cache.d_dm_stage.upload_async(h_dm, nao2, nullptr);
  run_rijk_cuda_device(cache, cache.d_dm_stage.data(), do_j ? cache.d_vj_stage.data() : nullptr,
                       do_k ? cache.d_vk_stage.data() : nullptr, opts, nullptr);
  if (do_j) cache.d_vj_stage.download_async(h_vj, nao2, nullptr);
  if (do_k) cache.d_vk_stage.download_async(h_vk, nao2, nullptr);
  check_cuda_jk(cudaStreamSynchronize(nullptr), "run_rijk_cuda sync");
}

// ---------------------------------------------------------------------------
// Analytical 2-Electron Nuclear Gradient CUDA Driver
// ---------------------------------------------------------------------------

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
