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

__global__ void add_matrices_kernel(std::size_t n, const double* __restrict__ a,
                                    const double* __restrict__ b, double* __restrict__ out) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void prepare_cartesian_density_uhf_kernel(
    int nsh, int nao, int ncart_tot, const int* __restrict__ shell_l,
    const int* __restrict__ shell_ao_off, const int* __restrict__ shell_cart_off,
    const double* __restrict__ harm_c, const double* __restrict__ d_dm_a_sph,
    const double* __restrict__ d_dm_b_sph, double* __restrict__ d_dm_tot_cart,
    double* __restrict__ d_dm_a_cart, double* __restrict__ d_dm_b_cart,
    double* __restrict__ d_sh_tot, double* __restrict__ d_sh_a, double* __restrict__ d_sh_b,
    double* __restrict__ d_sh_spin, double* __restrict__ d_global_dmax4) {
  __shared__ double s_max_tot[128];
  __shared__ double s_max_a[128];
  __shared__ double s_max_b[128];
  __shared__ double s_max_spin[128];

  const int total_pairs = nsh * nsh;
  const int tid = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
  double l_tot = 0.0, l_a = 0.0, l_b = 0.0, l_spin = 0.0;

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

    double b_tot = 0.0, b_a = 0.0, b_b = 0.0;
    double tmp_a[9 * 15] = {0.0};
    double tmp_b[9 * 15] = {0.0};
    for (int mi = 0; mi < nsi; ++mi) {
      const int u = aoi + mi;
      for (int mj = 0; mj < nsj; ++mj) {
        const int v = aoj + mj;
        const double da = 0.5 * (d_dm_a_sph[u * nao + v] + d_dm_a_sph[v * nao + u]);
        const double db = 0.5 * (d_dm_b_sph[u * nao + v] + d_dm_b_sph[v * nao + u]);
        const double dt = da + db;
        b_a = fmax(b_a, fabs(da));
        b_b = fmax(b_b, fabs(db));
        b_tot = fmax(b_tot, fabs(dt));
        const double* cj_row = Cj + mj * 15;
        double* ta_row = tmp_a + mi * ncj;
        double* tb_row = tmp_b + mi * ncj;
        for (int cj = 0; cj < ncj; ++cj) {
          ta_row[cj] += da * cj_row[cj];
          tb_row[cj] += db * cj_row[cj];
        }
      }
    }
    const double b_spin = fmax(b_a, b_b);
    d_sh_tot[tid] = b_tot;
    d_sh_a[tid] = b_a;
    d_sh_b[tid] = b_b;
    d_sh_spin[tid] = b_spin;
    l_tot = b_tot;
    l_a = b_a;
    l_b = b_b;
    l_spin = b_spin;

    for (int ci = 0; ci < nci; ++ci) {
      for (int cj = 0; cj < ncj; ++cj) {
        double acc_a = 0.0, acc_b = 0.0;
        for (int mi = 0; mi < nsi; ++mi) {
          const double c_mi = Ci[mi * 15 + ci];
          acc_a += c_mi * tmp_a[mi * ncj + cj];
          acc_b += c_mi * tmp_b[mi * ncj + cj];
        }
        const std::size_t idx =
            static_cast<std::size_t>(ci0 + ci) * ncart_tot + static_cast<std::size_t>(cj0 + cj);
        d_dm_a_cart[idx] = acc_a;
        d_dm_b_cart[idx] = acc_b;
        d_dm_tot_cart[idx] = acc_a + acc_b;
      }
    }
  }

  s_max_tot[threadIdx.x] = l_tot;
  s_max_a[threadIdx.x] = l_a;
  s_max_b[threadIdx.x] = l_b;
  s_max_spin[threadIdx.x] = l_spin;
  __syncthreads();
  for (unsigned int stride = 64; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_max_tot[threadIdx.x] = fmax(s_max_tot[threadIdx.x], s_max_tot[threadIdx.x + stride]);
      s_max_a[threadIdx.x] = fmax(s_max_a[threadIdx.x], s_max_a[threadIdx.x + stride]);
      s_max_b[threadIdx.x] = fmax(s_max_b[threadIdx.x], s_max_b[threadIdx.x + stride]);
      s_max_spin[threadIdx.x] = fmax(s_max_spin[threadIdx.x], s_max_spin[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (s_max_tot[0] > 0.0) atomic_max_nonneg_double(&d_global_dmax4[0], s_max_tot[0]);
    if (s_max_a[0] > 0.0) atomic_max_nonneg_double(&d_global_dmax4[1], s_max_a[0]);
    if (s_max_b[0] > 0.0) atomic_max_nonneg_double(&d_global_dmax4[2], s_max_b[0]);
    if (s_max_spin[0] > 0.0) atomic_max_nonneg_double(&d_global_dmax4[3], s_max_spin[0]);
  }
}

__global__ void finalize_cartesian_fock_kernel(
    int nsh, int nao, int ncart_tot, bool symmetrize_out, bool accumulate, double scale,
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
      if (accumulate) {
        m_sph_out[(aoi + mi) * nao + (aoj + mj)] += v;
        if (symmetrize_out && i != j) {
          m_sph_out[(aoj + mj) * nao + (aoi + mi)] += v;
        }
      } else {
        m_sph_out[(aoi + mi) * nao + (aoj + mj)] = v;
        if (symmetrize_out && i != j) {
          m_sph_out[(aoj + mj) * nao + (aoi + mi)] = v;
        }
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

__device__ inline double dev_quartet_dmax_uhf(const double* __restrict__ d_sh_tot,
                                              const double* __restrict__ d_sh_spin, int nsh, int i,
                                              int j, int k, int l, bool do_j, bool do_k) {
  double m = 0.0;
  if (do_j) {
    const double d_ij = 4.0 * d_sh_tot[i * nsh + j];
    const double d_kl = 4.0 * d_sh_tot[k * nsh + l];
    if (d_ij > m) m = d_ij;
    if (d_kl > m) m = d_kl;
  }
  if (do_k) {
    const double d1 = 2.0 * d_sh_spin[i * nsh + k];
    const double d2 = 2.0 * d_sh_spin[i * nsh + l];
    const double d3 = 2.0 * d_sh_spin[j * nsh + k];
    const double d4 = 2.0 * d_sh_spin[j * nsh + l];
    if (d1 > m) m = d1;
    if (d2 > m) m = d2;
    if (d3 > m) m = d3;
    if (d4 > m) m = d4;
  }
  return m;
}

// Warp-cooperative __ballot_sync density-compacted pure-J Hermite kernel for `sp` tiles
// (MaxL == 1, where la+lb+lc+ld <= 4 always holds).
template <typename R, bool SymmetrizeDM>
__global__ void __launch_bounds__(64, 2) direct_j_hermite_sp_warp_kernel(
    const DevPairClassMeta* __restrict__ classes, const GpuJKTile* __restrict__ tiles,
    int num_tiles, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const double* __restrict__ d_cart, const double* __restrict__ d_shell_max,
    const double* __restrict__ d_global_dmax, int nsh, int ncart_tot, double est_min,
    double est_max, double early_exit_cutoff, double omega, double* __restrict__ j_cart_out) {
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
        eval_hermite_j_quartet_tiered<1, R, SymmetrizeDM>(
            bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ,
            deg_ij * deg_kl, d_cart, ncart_tot, off_a, off_b, shell_cart_off[k], shell_cart_off[l],
            accum_j_bra, accum_j_ket, omega);
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
      eval_hermite_j_quartet_tiered<1, R, SymmetrizeDM>(
          bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ,
          deg_ij * deg_kl, d_cart, ncart_tot, off_a, off_b, shell_cart_off[k], shell_cart_off[l],
          accum_j_bra, accum_j_ket, omega);
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
    double est_min, double est_max, double early_exit_cutoff, double omega,
    double* __restrict__ j_cart_out, double* __restrict__ k_cart_out) {
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
            accum_j_bra, accum_j_ket, accum_k, omega);
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
          accum_j_bra, accum_j_ket, accum_k, omega);
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

template <int MaxL, typename R>
__global__ void __launch_bounds__(64, 2) direct_jk_rys_uhf_warp_kernel(
    const DevPairClassMeta* __restrict__ classes, const GpuJKTile* __restrict__ tiles,
    int num_tiles, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const double* __restrict__ d_tot_cart, const double* __restrict__ d_a_cart,
    const double* __restrict__ d_b_cart, const double* __restrict__ d_sh_tot,
    const double* __restrict__ d_sh_spin, const double* __restrict__ d_global_dmax4, int nsh,
    int ncart_tot, bool do_j, bool do_ka, bool do_kb, double est_min, double est_max,
    double early_exit_cutoff, double omega, double* __restrict__ j_cart_out,
    double* __restrict__ ka_cart_out, double* __restrict__ kb_cart_out) {
  const double g_tot = d_global_dmax4[0];
  const double g_spin = d_global_dmax4[3];
  const bool do_k = do_ka || do_kb;
  const double global_scale = fmax(do_j ? 4.0 * g_tot : 0.0, do_k ? 2.0 * g_spin : 0.0);
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
  auto accum_ka = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&ka_cart_out[r * ncart_tot + c], val);
  };
  auto accum_kb = [&](int r, int c, double val) {
    if (val != 0.0) atomicAdd(&kb_cart_out[r * ncart_tot + c], val);
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
        const double dmax =
            dev_quartet_dmax_uhf(d_sh_tot, d_sh_spin, nsh, i, j, k, l, do_j, do_k);
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
        const double w_K = 0.25 * deg_ij * deg_kl * s_PQ;
        eval_rys_jk_quartet_tiered_uhf<MaxL, R>(
            bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ, do_j,
            do_ka, do_kb, w_J, w_K, d_tot_cart, d_a_cart, d_b_cart, ncart_tot, off_a, off_b,
            shell_cart_off[k], shell_cart_off[l], accum_j_bra, accum_j_ket, accum_ka, accum_kb,
            omega);
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
      const double w_K = 0.25 * deg_ij * deg_kl * s_PQ;
      eval_rys_jk_quartet_tiered_uhf<MaxL, R>(
          bra_m.la, bra_m.lb, ket_m.la, ket_m.lb, bra_m.view, P, ket_m.view, Q, same_PQ, do_j,
          do_ka, do_kb, w_J, w_K, d_tot_cart, d_a_cart, d_b_cart, ncart_tot, off_a, off_b,
          shell_cart_off[k], shell_cart_off[l], accum_j_bra, accum_j_ket, accum_ka, accum_kb,
          omega);
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
    double omega, double* __restrict__ d_metric) {
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
          val += cP * cQ * kJKTwoOverSqrtPi * sqrt(rho) * boys_f0_erf(rho * PQ2, rho, omega);
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
        rys_roots_weights_erf<double>(n_roots, T, rho, omega, u_roots, w_weights);

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
// and arbitrary auxiliary shells (`0 <= lP <= 4`), scattering chunk `[p_beg, p_end)` directly into `d_I_chunk`.
__global__ void __launch_bounds__(128) ri_3c_spd_kernel(
    int la, int lb, int lP, ShellPairView orb, ShellPairView aux, int p_beg, int p_end,
    int q_offset, int nao, double omega, double* __restrict__ d_I_chunk) {
  const long long naux_p = p_end - p_beg;
  if (naux_p <= 0) return;
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
    const int P = p_beg + static_cast<int>(idx - static_cast<long long>(ab) * naux_p);

    const int u0 = orb.ao_i[ab];
    const int v0 = orb.ao_j[ab];
    const int p0 = aux.ao_i[P] - q_offset;
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
          val += Cab * cP * kJKTwoOverSqrtPi * sqrt(rho) * boys_f0_erf(rho * PQ2, rho, omega);
        }
      }
      double* row_ptr = d_I_chunk + static_cast<std::size_t>(p0) * nao2;
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
        rys_roots_weights_erf<double>(n_roots, T, rho, omega, u_roots, w_weights);

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
      double* row_ptr = d_I_chunk + static_cast<std::size_t>(p0 + mp) * nao2;
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
    int la, int lb, int lP, ShellPairView orb, ShellPairView aux, int p_beg, int p_end,
    int q_offset, int nao, double omega, double* __restrict__ d_I_chunk) {
  const long long naux_p = p_end - p_beg;
  if (naux_p <= 0) return;
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
    const int P = p_beg + static_cast<int>(idx - static_cast<long long>(ab) * naux_p);

    const int u0 = orb.ao_i[ab];
    const int v0 = orb.ao_j[ab];
    const int p0 = aux.ao_i[P] - q_offset;
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
        rys_roots_weights_erf<double>(n_roots, T, rho, omega, u_roots, w_weights);

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
      double* row_ptr = d_I_chunk + static_cast<std::size_t>(p0 + mp) * nao2;
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

__global__ void jk_nuclear_grad_dynamic_kernel(
    const DevPairClassMeta* __restrict__ classes, const GradQuartetTask* __restrict__ tasks,
    int num_tasks, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const int* __restrict__ shell_atom, const double* __restrict__ d_cart, int ncart_tot,
    double j_fraction, double k_fraction, double omega, double* __restrict__ grad_out) {
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
        k_fraction, g12, omega, j_fraction);

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

__global__ void jk_nuclear_grad_uhf_dynamic_kernel(
    const DevPairClassMeta* __restrict__ classes, const GradQuartetTask* __restrict__ tasks,
    int num_tasks, int* __restrict__ task_counter, const int* __restrict__ shell_cart_off,
    const int* __restrict__ shell_atom, const double* __restrict__ d_tot_cart,
    const double* __restrict__ d_alpha_cart, const double* __restrict__ d_beta_cart,
    int ncart_tot, double j_fraction, double k_fraction, double omega,
    double* __restrict__ grad_out) {
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
    generated::eri_quartet_grad_direct_dm_uhf<double>(
        bra.la, bra.lb, ket.la, ket.lb, bra.view, t.P, ket.view, t.Q, d_tot_cart, d_alpha_cart,
        d_beta_cart, ncart_tot, shell_cart_off[i], shell_cart_off[j], shell_cart_off[k],
        shell_cart_off[l], t.w_sym, k_fraction, g12, omega, j_fraction);

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
void launch_direct_jk_uhf_tier(int num_sms, const DevPairClassMeta* d_classes,
                               const DevVecJK<GpuJKTile>& d_tiles, int* d_counter_fp64,
                               int* d_counter_fp32, const int* d_shell_cart_off,
                               const double* d_tot_cart, const double* d_a_cart,
                               const double* d_b_cart, const double* d_sh_tot,
                               const double* d_sh_spin, const double* d_global_dmax4, int nsh,
                               int ncart_tot, bool do_j, bool do_ka, bool do_kb,
                               const JKOptions& opts, double* d_j_cart, double* d_ka_cart,
                               double* d_kb_cart, cudaStream_t stream) {
  const int num_tiles = static_cast<int>(d_tiles.size());
  if (num_tiles <= 0) return;

  constexpr int kBlock = 64;
  constexpr int kWarpsPerBlock = kBlock / 32;
  const int max_persistent_blocks = std::max(32, num_sms * 4);
  const int blocks =
      std::min((num_tiles + kWarpsPerBlock - 1) / kWarpsPerBlock, max_persistent_blocks);
  constexpr double kInf = 1e300;

  auto launch_rys = [&](auto prec_tag, int* cnt, double e_min, double e_max, double early_cut) {
    using R = decltype(prec_tag);
    direct_jk_rys_uhf_warp_kernel<MaxL, R><<<blocks, kBlock, 0, stream>>>(
        d_classes, d_tiles.data(), num_tiles, cnt, d_shell_cart_off, d_tot_cart, d_a_cart,
        d_b_cart, d_sh_tot, d_sh_spin, d_global_dmax4, nsh, ncart_tot, do_j, do_ka, do_kb, e_min,
        e_max, early_cut, opts.omega, d_j_cart, d_ka_cart, d_kb_cart);
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
  check_cuda_jk(cudaGetLastError(), "direct_jk_rys_uhf_warp_kernel launch");
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
      auto launch_hermite = [&](auto prec_tag, int* cnt, double e_min, double e_max,
                                double early_cut) {
        using R = decltype(prec_tag);
        if (opts.symmetrize_dm) {
          direct_j_hermite_sp_warp_kernel<R, true><<<blocks, kBlock, 0, stream>>>(
              d_classes, d_tiles.data(), num_tiles, cnt, d_shell_cart_off, d_dm_cart,
              d_shell_max, d_global_dmax, nsh, ncart_tot, e_min, e_max, early_cut, opts.omega,
              d_j_cart);
        } else {
          direct_j_hermite_sp_warp_kernel<R, false><<<blocks, kBlock, 0, stream>>>(
              d_classes, d_tiles.data(), num_tiles, cnt, d_shell_cart_off, d_dm_cart,
              d_shell_max, d_global_dmax, nsh, ncart_tot, e_min, e_max, early_cut, opts.omega,
              d_j_cart);
        }
      };
      if (opts.precision == Precision::FP64) {
        launch_hermite(double(0), d_counter_fp64, opts.schwarz_cutoff, kInf, opts.schwarz_cutoff);
      } else if (opts.precision == Precision::FP32) {
        launch_hermite(float(0), d_counter_fp32, opts.schwarz_cutoff, kInf, opts.schwarz_cutoff);
      } else {
        const double thr = std::max(opts.fp32_threshold, opts.schwarz_cutoff);
        launch_hermite(double(0), d_counter_fp64, thr, kInf, thr);
        if (opts.fp32_threshold > opts.schwarz_cutoff) {
          launch_hermite(float(0), d_counter_fp32, opts.schwarz_cutoff, opts.fp32_threshold,
                         opts.schwarz_cutoff);
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
          opts.omega, d_j_cart, d_k_cart);
    } else {
      direct_jk_rys_warp_kernel<MaxL, R, false><<<blocks, kBlock, 0, stream>>>(
          d_classes, d_tiles.data(), num_tiles, cnt, d_shell_cart_off, d_dm_cart, d_shell_max,
          d_global_dmax, nsh, ncart_tot, opts.compute_j, opts.compute_k, e_min, e_max, early_cut,
          opts.omega, d_j_cart, d_k_cart);
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

  // Lazy open-shell UHF/UKS buffers (allocated once on first UHF call)
  bool uhf_initialized = false;
  DevVecJK<int> d_counters_uhf;
  DevVecJK<double> d_global_dmax_uhf;
  DevVecJK<double> d_dm_cart_tot;
  DevVecJK<double> d_dm_cart_a;
  DevVecJK<double> d_dm_cart_b;
  DevVecJK<double> d_shell_max_tot;
  DevVecJK<double> d_shell_max_a;
  DevVecJK<double> d_shell_max_b;
  DevVecJK<double> d_shell_max_spin;
  DevVecJK<double> d_ka_cart;
  DevVecJK<double> d_kb_cart;
  DevVecJK<double> d_dm_sph_tot;
  DevVecJK<double> d_dm_sph_b;
  DevVecJK<double> d_vkb_sph;

  void ensure_uhf_buffers() {
    if (uhf_initialized) return;
    const std::size_t nc2 = static_cast<std::size_t>(ncart_total) * ncart_total;
    const std::size_t nsh2 = static_cast<std::size_t>(nsh) * nsh;
    const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
    d_counters_uhf.resize(24);
    d_global_dmax_uhf.resize(4);
    d_dm_cart_tot.resize(nc2);
    d_dm_cart_a.resize(nc2);
    d_dm_cart_b.resize(nc2);
    d_shell_max_tot.resize(nsh2);
    d_shell_max_a.resize(nsh2);
    d_shell_max_b.resize(nsh2);
    d_shell_max_spin.resize(nsh2);
    d_ka_cart.resize(nc2);
    d_kb_cart.resize(nc2);
    d_dm_sph_tot.resize(nao2);
    d_dm_sph_b.resize(nao2);
    d_vkb_sph.resize(nao2);
    uhf_initialized = true;
  }
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

  cache.d_global_dmax.zero_async(stream);
  const int nsh2 = cache.nsh * cache.nsh;
  const int prep_blocks = (nsh2 + 127) / 128;
  prepare_cartesian_density_kernel<<<prep_blocks, 128, 0, stream>>>(
      cache.nsh, cache.nao, cache.ncart_total, eff_sym_dm, cache.d_shell_l.data(),
      cache.d_shell_ao_off.data(), cache.d_shell_cart_off.data(), cache.d_harm_c.data(), d_dm,
      cache.d_dm_cart.data(), cache.d_shell_max.data(), cache.d_global_dmax.data());
  check_cuda_jk(cudaGetLastError(), "prepare_cartesian_density_kernel launch");

  auto run_pass = [&](bool pass_j, bool pass_k, double pass_omega, double k_coeff, bool acc_k) {
    cache.d_counters.zero_async(stream);
    if (pass_j) cache.d_j_cart.zero_async(stream);
    if (pass_k) cache.d_k_cart.zero_async(stream);

    const double k_amp = pass_k ? std::max(1.0, std::abs(k_coeff)) : 1.0;
    JKOptions k_opts = opts;
    k_opts.compute_j = pass_j;
    k_opts.compute_k = pass_k;
    k_opts.symmetrize_dm = eff_sym_dm;
    k_opts.omega = pass_omega;
    k_opts.schwarz_cutoff = opts.schwarz_cutoff / k_amp;
    k_opts.fp32_threshold = opts.fp32_threshold / k_amp;

    double* j_ptr = pass_j ? cache.d_j_cart.data() : nullptr;
    double* k_ptr = pass_k ? cache.d_k_cart.data() : nullptr;

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

    if (pass_j) {
      finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
          cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/true, /*accumulate=*/false,
          1.0, cache.d_shell_l.data(), cache.d_shell_ao_off.data(), cache.d_shell_cart_off.data(),
          cache.d_harm_c.data(), cache.d_j_cart.data(), d_vj);
      check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (J) launch");
    }
    if (pass_k) {
      finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
          cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/opts.symmetrize_dm,
          /*accumulate=*/acc_k, k_coeff, cache.d_shell_l.data(), cache.d_shell_ao_off.data(),
          cache.d_shell_cart_off.data(), cache.d_harm_c.data(), cache.d_k_cart.data(), d_vk);
      check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (K) launch");
    }
  };

  if (!opts.rsh_mode) {
    run_pass(do_j, do_k, opts.omega, opts.k_scale, /*acc_k=*/false);
  } else {
    const double k1 = opts.k_sr * opts.k_scale;
    const bool pass1_k = do_k && (k1 != 0.0);
    if (do_j || pass1_k) {
      run_pass(do_j, pass1_k, /*pass_omega=*/0.0, k1, /*acc_k=*/false);
    }
    const double k2 = (opts.k_lr - opts.k_sr) * opts.k_scale;
    const bool pass2_k = do_k && (opts.omega > 0.0) && (k2 != 0.0);
    if (pass2_k) {
      run_pass(/*pass_j=*/false, /*pass_k=*/true, opts.omega, k2, /*acc_k=*/pass1_k);
    }
    if (do_k && !pass1_k && !pass2_k) {
      const std::size_t nao2 = static_cast<std::size_t>(cache.nao) * cache.nao;
      check_cuda_jk(cudaMemsetAsync(d_vk, 0, nao2 * sizeof(double), stream),
                    "cudaMemsetAsync (RSH zero K)");
    }
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

void run_direct_jk_uhf_cuda_device(DirectJKCudaCache& cache, const double* d_dm_alpha,
                                   const double* d_dm_beta, double* d_vj_tot, double* d_vk_alpha,
                                   double* d_vk_beta, const JKOptions& opts, void* raw_stream) {
  auto stream = static_cast<cudaStream_t>(raw_stream);
  const bool do_j = opts.compute_j && (d_vj_tot != nullptr);
  const bool do_ka = opts.compute_k && (d_vk_alpha != nullptr);
  const bool do_kb = opts.compute_k && (d_vk_beta != nullptr);
  const bool do_k = do_ka || do_kb;
  if ((!do_j && !do_k) || cache.nao == 0) return;

  cache.ensure_uhf_buffers();
  const std::size_t nao2 = static_cast<std::size_t>(cache.nao) * cache.nao;
  const int add_blocks = static_cast<int>((nao2 + 255) / 256);

  // Pure-J path: sum D_tot = D_a + D_b into persistent buffer and run Hermite J-engine!
  if (!do_k) {
    add_matrices_kernel<<<add_blocks, 256, 0, stream>>>(
        nao2, d_dm_alpha, d_dm_beta, cache.d_dm_sph_tot.data());
    check_cuda_jk(cudaGetLastError(), "add_matrices_kernel");
    JKOptions j_opts = opts;
    j_opts.compute_j = true;
    j_opts.compute_k = false;
    if (opts.rsh_mode) {
      j_opts.rsh_mode = false;
      j_opts.omega = 0.0;
    }
    run_direct_jk_cuda_device(cache, cache.d_dm_sph_tot.data(), d_vj_tot, nullptr, j_opts,
                              raw_stream);
    return;
  }

  // Non-symmetric fallback if ever requested
  if (!opts.symmetrize_dm) {
    if (do_j) {
      add_matrices_kernel<<<add_blocks, 256, 0, stream>>>(
          nao2, d_dm_alpha, d_dm_beta, cache.d_dm_sph_tot.data());
      check_cuda_jk(cudaGetLastError(), "add_matrices_kernel");
      JKOptions j_opts = opts;
      j_opts.compute_j = true;
      j_opts.compute_k = false;
      if (opts.rsh_mode) {
        j_opts.rsh_mode = false;
        j_opts.omega = 0.0;
      }
      run_direct_jk_cuda_device(cache, cache.d_dm_sph_tot.data(), d_vj_tot, nullptr, j_opts,
                                raw_stream);
    }
    JKOptions k_opts = opts;
    k_opts.compute_j = false;
    k_opts.compute_k = true;
    if (do_ka) run_direct_jk_cuda_device(cache, d_dm_alpha, nullptr, d_vk_alpha, k_opts, raw_stream);
    if (do_kb) run_direct_jk_cuda_device(cache, d_dm_beta, nullptr, d_vk_beta, k_opts, raw_stream);
    return;
  }

  cache.d_global_dmax_uhf.zero_async(stream);
  const int nsh2 = cache.nsh * cache.nsh;
  const int prep_blocks = (nsh2 + 127) / 128;
  prepare_cartesian_density_uhf_kernel<<<prep_blocks, 128, 0, stream>>>(
      cache.nsh, cache.nao, cache.ncart_total, cache.d_shell_l.data(), cache.d_shell_ao_off.data(),
      cache.d_shell_cart_off.data(), cache.d_harm_c.data(), d_dm_alpha, d_dm_beta,
      cache.d_dm_cart_tot.data(), cache.d_dm_cart_a.data(), cache.d_dm_cart_b.data(),
      cache.d_shell_max_tot.data(), cache.d_shell_max_a.data(), cache.d_shell_max_b.data(),
      cache.d_shell_max_spin.data(), cache.d_global_dmax_uhf.data());
  check_cuda_jk(cudaGetLastError(), "prepare_cartesian_density_uhf_kernel launch");

  auto run_uhf_pass = [&](bool pass_j, bool pass_ka, bool pass_kb, double pass_omega,
                          double k_coeff, bool acc_k) {
    const bool pass_k = pass_ka || pass_kb;
    cache.d_counters_uhf.zero_async(stream);
    if (pass_j) cache.d_j_cart.zero_async(stream);
    if (pass_ka) cache.d_ka_cart.zero_async(stream);
    if (pass_kb) cache.d_kb_cart.zero_async(stream);

    const double k_amp = pass_k ? std::max(1.0, std::abs(k_coeff)) : 1.0;
    JKOptions pass_opts = opts;
    pass_opts.omega = pass_omega;
    pass_opts.schwarz_cutoff = opts.schwarz_cutoff / k_amp;
    pass_opts.fp32_threshold = opts.fp32_threshold / k_amp;

    double* j_ptr = pass_j ? cache.d_j_cart.data() : nullptr;
    double* ka_ptr = pass_ka ? cache.d_ka_cart.data() : nullptr;
    double* kb_ptr = pass_kb ? cache.d_kb_cart.data() : nullptr;

    if (pass_j && !pass_k) {
      JKOptions j_opts = pass_opts;
      j_opts.backend = Backend::Cuda;
      j_opts.compute_j = true;
      j_opts.compute_k = false;
      j_opts.symmetrize_dm = true;
      launch_direct_jk_tier<1>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[0],
                               cache.d_counters_uhf.data() + 0, cache.d_counters_uhf.data() + 4,
                               cache.d_shell_cart_off.data(), cache.d_dm_cart_tot.data(),
                               cache.d_shell_max_tot.data(), cache.d_global_dmax_uhf.data() + 0,
                               cache.nsh, cache.ncart_total, j_opts, j_ptr, nullptr, stream);
      launch_direct_jk_uhf_tier<2>(
          cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[1],
          cache.d_counters_uhf.data() + 1, cache.d_counters_uhf.data() + 5,
          cache.d_shell_cart_off.data(), cache.d_dm_cart_tot.data(), cache.d_dm_cart_a.data(),
          cache.d_dm_cart_b.data(), cache.d_shell_max_tot.data(), cache.d_shell_max_spin.data(),
          cache.d_global_dmax_uhf.data(), cache.nsh, cache.ncart_total, true, false, false,
          pass_opts, j_ptr, nullptr, nullptr, stream);
    } else {
      launch_direct_jk_uhf_tier<1>(
          cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[0],
          cache.d_counters_uhf.data() + 0, cache.d_counters_uhf.data() + 4,
          cache.d_shell_cart_off.data(), cache.d_dm_cart_tot.data(), cache.d_dm_cart_a.data(),
          cache.d_dm_cart_b.data(), cache.d_shell_max_tot.data(), cache.d_shell_max_spin.data(),
          cache.d_global_dmax_uhf.data(), cache.nsh, cache.ncart_total, pass_j, pass_ka, pass_kb,
          pass_opts, j_ptr, ka_ptr, kb_ptr, stream);
      launch_direct_jk_uhf_tier<2>(
          cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[1],
          cache.d_counters_uhf.data() + 1, cache.d_counters_uhf.data() + 5,
          cache.d_shell_cart_off.data(), cache.d_dm_cart_tot.data(), cache.d_dm_cart_a.data(),
          cache.d_dm_cart_b.data(), cache.d_shell_max_tot.data(), cache.d_shell_max_spin.data(),
          cache.d_global_dmax_uhf.data(), cache.nsh, cache.ncart_total, pass_j, pass_ka, pass_kb,
          pass_opts, j_ptr, ka_ptr, kb_ptr, stream);
    }

    // For MaxL >= 3 (f, g): reuse spill-free single-DM Tier 2/3 kernels cleanly
    if (cache.d_tier_tiles[2].size() > 0 || cache.d_tier_tiles[3].size() > 0) {
      if (pass_j) {
        JKOptions j_opts = pass_opts;
        j_opts.backend = Backend::Cuda;
        j_opts.compute_j = true;
        j_opts.compute_k = false;
        j_opts.symmetrize_dm = true;
        launch_direct_jk_tier<3>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[2],
                                 cache.d_counters_uhf.data() + 2, cache.d_counters_uhf.data() + 6,
                                 cache.d_shell_cart_off.data(), cache.d_dm_cart_tot.data(),
                                 cache.d_shell_max_tot.data(), cache.d_global_dmax_uhf.data() + 0,
                                 cache.nsh, cache.ncart_total, j_opts, j_ptr, nullptr, stream);
        launch_direct_jk_tier<4>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[3],
                                 cache.d_counters_uhf.data() + 3, cache.d_counters_uhf.data() + 7,
                                 cache.d_shell_cart_off.data(), cache.d_dm_cart_tot.data(),
                                 cache.d_shell_max_tot.data(), cache.d_global_dmax_uhf.data() + 0,
                                 cache.nsh, cache.ncart_total, j_opts, j_ptr, nullptr, stream);
      }
      if (pass_ka) {
        JKOptions k_opts = pass_opts;
        k_opts.backend = Backend::Cuda;
        k_opts.compute_j = false;
        k_opts.compute_k = true;
        k_opts.symmetrize_dm = true;
        launch_direct_jk_tier<3>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[2],
                                 cache.d_counters_uhf.data() + 10, cache.d_counters_uhf.data() + 14,
                                 cache.d_shell_cart_off.data(), cache.d_dm_cart_a.data(),
                                 cache.d_shell_max_a.data(), cache.d_global_dmax_uhf.data() + 1,
                                 cache.nsh, cache.ncart_total, k_opts, nullptr, ka_ptr, stream);
        launch_direct_jk_tier<4>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[3],
                                 cache.d_counters_uhf.data() + 11, cache.d_counters_uhf.data() + 15,
                                 cache.d_shell_cart_off.data(), cache.d_dm_cart_a.data(),
                                 cache.d_shell_max_a.data(), cache.d_global_dmax_uhf.data() + 1,
                                 cache.nsh, cache.ncart_total, k_opts, nullptr, ka_ptr, stream);
      }
      if (pass_kb) {
        JKOptions k_opts = pass_opts;
        k_opts.backend = Backend::Cuda;
        k_opts.compute_j = false;
        k_opts.compute_k = true;
        k_opts.symmetrize_dm = true;
        launch_direct_jk_tier<3>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[2],
                                 cache.d_counters_uhf.data() + 18, cache.d_counters_uhf.data() + 22,
                                 cache.d_shell_cart_off.data(), cache.d_dm_cart_b.data(),
                                 cache.d_shell_max_b.data(), cache.d_global_dmax_uhf.data() + 2,
                                 cache.nsh, cache.ncart_total, k_opts, nullptr, kb_ptr, stream);
        launch_direct_jk_tier<4>(cache.num_sms, cache.d_classes.data(), cache.d_tier_tiles[3],
                                 cache.d_counters_uhf.data() + 19, cache.d_counters_uhf.data() + 23,
                                 cache.d_shell_cart_off.data(), cache.d_dm_cart_b.data(),
                                 cache.d_shell_max_b.data(), cache.d_global_dmax_uhf.data() + 2,
                                 cache.nsh, cache.ncart_total, k_opts, nullptr, kb_ptr, stream);
      }
    }

    if (pass_j) {
      finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
          cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/true, /*accumulate=*/false,
          1.0, cache.d_shell_l.data(), cache.d_shell_ao_off.data(), cache.d_shell_cart_off.data(),
          cache.d_harm_c.data(), cache.d_j_cart.data(), d_vj_tot);
      check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (UHF J) launch");
    }
    if (pass_ka) {
      finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
          cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/true, /*accumulate=*/acc_k,
          k_coeff, cache.d_shell_l.data(), cache.d_shell_ao_off.data(),
          cache.d_shell_cart_off.data(), cache.d_harm_c.data(), cache.d_ka_cart.data(), d_vk_alpha);
      check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (UHF Ka) launch");
    }
    if (pass_kb) {
      finalize_cartesian_fock_kernel<<<prep_blocks, 128, 0, stream>>>(
          cache.nsh, cache.nao, cache.ncart_total, /*symmetrize_out=*/true, /*accumulate=*/acc_k,
          k_coeff, cache.d_shell_l.data(), cache.d_shell_ao_off.data(),
          cache.d_shell_cart_off.data(), cache.d_harm_c.data(), cache.d_kb_cart.data(), d_vk_beta);
      check_cuda_jk(cudaGetLastError(), "finalize_cartesian_fock_kernel (UHF Kb) launch");
    }
  };

  if (!opts.rsh_mode) {
    run_uhf_pass(do_j, do_ka, do_kb, opts.omega, opts.k_scale, /*acc_k=*/false);
  } else {
    const double k1 = opts.k_sr * opts.k_scale;
    const bool pass1_k = (k1 != 0.0);
    if (do_j || pass1_k) {
      run_uhf_pass(do_j, pass1_k && do_ka, pass1_k && do_kb, /*pass_omega=*/0.0, k1,
                   /*acc_k=*/false);
    }
    const double k2 = (opts.k_lr - opts.k_sr) * opts.k_scale;
    const bool pass2_k = (opts.omega > 0.0) && (k2 != 0.0);
    if (pass2_k) {
      run_uhf_pass(/*pass_j=*/false, do_ka, do_kb, opts.omega, k2, /*acc_k=*/pass1_k);
    }
    if (!pass1_k && !pass2_k) {
      if (do_ka) {
        check_cuda_jk(cudaMemsetAsync(d_vk_alpha, 0, nao2 * sizeof(double), stream),
                      "cudaMemsetAsync (RSH UHF zero Ka)");
      }
      if (do_kb) {
        check_cuda_jk(cudaMemsetAsync(d_vk_beta, 0, nao2 * sizeof(double), stream),
                      "cudaMemsetAsync (RSH UHF zero Kb)");
      }
    }
  }
}

void run_direct_jk_uhf_cuda_host(DirectJKCudaCache& cache, const double* h_dm_alpha,
                                 const double* h_dm_beta, double* h_vj_tot, double* h_vk_alpha,
                                 double* h_vk_beta, const JKOptions& opts) {
  const std::size_t nao2 = static_cast<std::size_t>(cache.nao) * cache.nao;
  const bool do_j = opts.compute_j && (h_vj_tot != nullptr);
  const bool do_ka = opts.compute_k && (h_vk_alpha != nullptr);
  const bool do_kb = opts.compute_k && (h_vk_beta != nullptr);
  if ((!do_j && !do_ka && !do_kb) || nao2 == 0) return;

  cache.ensure_uhf_buffers();
  cache.d_dm_sph.upload_async(h_dm_alpha, nao2, nullptr);
  cache.d_dm_sph_b.upload_async(h_dm_beta, nao2, nullptr);
  run_direct_jk_uhf_cuda_device(cache, cache.d_dm_sph.data(), cache.d_dm_sph_b.data(),
                                do_j ? cache.d_vj_sph.data() : nullptr,
                                do_ka ? cache.d_vk_sph.data() : nullptr,
                                do_kb ? cache.d_vkb_sph.data() : nullptr, opts, nullptr);
  if (do_j) cache.d_vj_sph.download_async(h_vj_tot, nao2, nullptr);
  if (do_ka) cache.d_vk_sph.download_async(h_vk_alpha, nao2, nullptr);
  if (do_kb) cache.d_vkb_sph.download_async(h_vk_beta, nao2, nullptr);
  check_cuda_jk(cudaStreamSynchronize(nullptr), "run_direct_jk_uhf_cuda_host sync");
}

// ---------------------------------------------------------------------------
// RIJKCudaCache Implementation (Bounded-VRAM Streamed F5 RI-K + F6 RI-JK Grad)
// ---------------------------------------------------------------------------

struct RIJKCudaCache {
  int nao = 0;
  int naux = 0;
  int ncart_orb = 0;
  int ncart_aux = 0;
  double rsh_omega = 0.0;

  Basis basis{std::vector<Shell>{}};
  Basis aux_basis{std::vector<Shell>{}};
  std::vector<int> orb_shell_cart_off;
  std::vector<int> aux_shell_cart_off;
  std::vector<std::vector<int>> aux_ish_lists;
  std::vector<DevShellPairList> dev_orb;
  std::vector<DevShellPairList> dev_aux;
  std::vector<RIAuxChunk> chunks_scf;

  // Persistent 2-center metric factors ([naux x naux] ONLY — NO dense [naux, nao, nao] tensor!).
  // If chol_ok_*: stores col-major lower triangular L from cusolverDnDpotrf.
  // If !chol_ok_*: stores symmetric V^{-1/2} from cusolverDnDsyevd (ev > 1e-10).
  bool chol_ok_0 = true;
  bool chol_ok_w = true;
  DevVecJK<double> d_metric_0;
  DevVecJK<double> d_metric_w;

  GpuLinalgContext linalg{&default_device_pool()};

  // Bounded-VRAM PooledBuffer streaming workspaces for SCF (steady_state_allocations == 0).
  bool j_workspaces_allocated = false;
  bool k_workspaces_allocated = false;
  bool k_workspaces_ozaki = false;
  int cap_b_Q = 0;
  int cap_b_P = 64;
  int cap_b_i = 0;
  PooledBuffer<double> d_dm_work;
  PooledBuffer<double> d_v_aux;
  PooledBuffer<double> d_I_chunk;
  PooledBuffer<double> d_Y_blk;
  PooledBuffer<double> d_L_pchunk;
  PooledBuffer<double> d_L_perm_a;
  PooledBuffer<double> d_L_perm_b;
  PooledBuffer<double> d_C_tile;

  // Fixed-capacity buffers for DM eigendecomposition fallback (when C_occ is absent)
  PooledBuffer<double> d_dm_w;
  PooledBuffer<double> d_dm_u;
  PooledBuffer<int> d_dm_cols;
  PooledBuffer<double> d_dm_scales;

  // Host-pointer API staging buffers (allocated only if host-pointer overloads are called)
  DevVecJK<double> d_dm_stage;
  DevVecJK<double> d_vj_stage;
  DevVecJK<double> d_vk_stage;
  DevVecJK<double> d_c_stage;
  DevVecJK<double> d_da_stage;
  DevVecJK<double> d_db_stage;
  DevVecJK<double> d_ca_stage;
  DevVecJK<double> d_cb_stage;
  DevVecJK<double> d_ka_stage;
  DevVecJK<double> d_kb_stage;
};

namespace {

__global__ void permute_ri_mo_chunk_kernel(int bp, int nao, int bi, int i_off, int bi_tot,
                                           const double* __restrict__ src,
                                           double* __restrict__ dst) {
  const std::size_t total = static_cast<std::size_t>(bp) * nao * bi;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t nao_bi = static_cast<std::size_t>(nao) * bi;

  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t pl = idx / nao_bi;
    const std::size_t rem = idx - pl * nao_bi;
    const std::size_t u = rem / bi;
    const std::size_t i_loc = rem - u * bi;
    const std::size_t src_idx =
        ((pl * static_cast<std::size_t>(nao) + u) * bi_tot) + (i_off + i_loc);
    const std::size_t dst_idx = ((u * static_cast<std::size_t>(bp) + pl) * bi) + i_loc;
    dst[dst_idx] = src[src_idx];
  }
}

__global__ void extract_col_tile_kernel(int rows, int total_cols, int c0, int bc, int dst_off,
                                        int dst_ld, const double* __restrict__ src,
                                        double* __restrict__ dst) {
  const std::size_t total = static_cast<std::size_t>(rows) * bc;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int r = static_cast<int>(idx / bc);
    const int c = static_cast<int>(idx - static_cast<std::size_t>(r) * bc);
    dst[static_cast<std::size_t>(r) * dst_ld + (dst_off + c)] =
        src[static_cast<std::size_t>(r) * total_cols + (c0 + c)];
  }
}

__global__ void extract_cols_from_u_scaled_kernel(int nao, int ncols, int dst_off, int dst_ld,
                                                  const int* __restrict__ cols,
                                                  const double* __restrict__ scales,
                                                  const double* __restrict__ U,
                                                  double* __restrict__ dst) {
  const std::size_t total = static_cast<std::size_t>(nao) * ncols;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int r = static_cast<int>(idx / ncols);
    const int j = static_cast<int>(idx - static_cast<std::size_t>(r) * ncols);
    const int src_col = cols[j];
    const double sc = scales[j];
    dst[static_cast<std::size_t>(r) * dst_ld + (dst_off + j)] =
        sc * U[static_cast<std::size_t>(r) * nao + src_col];
  }
}

__global__ void set_identity_col_tile_kernel(int nao, int c0, int bc, int dst_off, int dst_ld,
                                             double* __restrict__ dst) {
  const std::size_t total = static_cast<std::size_t>(nao) * bc;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int r = static_cast<int>(idx / bc);
    const int c = static_cast<int>(idx - static_cast<std::size_t>(r) * bc);
    dst[static_cast<std::size_t>(r) * dst_ld + (dst_off + c)] = (r == (c0 + c)) ? 1.0 : 0.0;
  }
}

__global__ void add_dm_pair_kernel(int nao, bool symmetrize, const double* __restrict__ da,
                                   const double* __restrict__ db, double* __restrict__ out) {
  const int total = nao * nao;
  const int stride = static_cast<int>(gridDim.x) * static_cast<int>(blockDim.x);
  for (int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                 static_cast<int>(threadIdx.x);
       idx < total; idx += stride) {
    if (!symmetrize) {
      out[idx] = da[idx] + db[idx];
    } else {
      const int r = idx / nao;
      const int c = idx - r * nao;
      const int ji = c * nao + r;
      out[idx] = 0.5 * ((da[idx] + db[idx]) + (da[ji] + db[ji]));
    }
  }
}

__global__ void outer_product_scaled_kernel(int naux, double scale,
                                            const double* __restrict__ rho,
                                            double* __restrict__ gamma_2c) {
  const std::size_t total = static_cast<std::size_t>(naux) * naux;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int p = static_cast<int>(idx / naux);
    const int q = static_cast<int>(idx - static_cast<std::size_t>(p) * naux);
    gamma_2c[idx] = scale * rho[p] * rho[q];
  }
}

__global__ void apply_grad_channel_signs_batch_kernel(int naux, int i0, int bi, int nc,
                                                      const double* __restrict__ signs,
                                                      double* __restrict__ d_M_batch) {
  const std::size_t bnc = static_cast<std::size_t>(bi) * nc;
  const std::size_t total = static_cast<std::size_t>(naux) * bnc;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t rem = idx % bnc;
    const int i_loc = static_cast<int>(rem / nc);
    const int j = static_cast<int>(rem - static_cast<std::size_t>(i_loc) * nc);
    d_M_batch[idx] *= signs[i0 + i_loc] * signs[j];
  }
}

__global__ void copy_apply_signs_col_blk_kernel(int naux, int i0, int nc, int src_ld, int src_c0,
                                                int ij0, int bw, const double* __restrict__ signs,
                                                const double* __restrict__ d_M_src,
                                                double* __restrict__ d_M_blk_signed) {
  const std::size_t total = static_cast<std::size_t>(naux) * bw;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int p = static_cast<int>(idx / bw);
    const int k_loc = static_cast<int>(idx - static_cast<std::size_t>(p) * bw);
    const int ij = ij0 + k_loc;
    const int i_loc = ij / nc;
    const int j = ij - i_loc * nc;
    const double v = d_M_src[static_cast<std::size_t>(p) * src_ld + (src_c0 + k_loc)];
    d_M_blk_signed[idx] = v * (signs[i0 + i_loc] * signs[j]);
  }
}

__global__ void init_gamma_3c_coulomb_kernel(int bc_cart, std::size_t ncart2, double j_frac,
                                             const double* __restrict__ d_rho_cart_chunk,
                                             const double* __restrict__ d_D_tot_cart,
                                             double* __restrict__ d_Gamma_chunk) {
  const std::size_t total = static_cast<std::size_t>(bc_cart) * ncart2;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const std::size_t p_loc = idx / ncart2;
    const std::size_t ab = idx - p_loc * ncart2;
    d_Gamma_chunk[idx] = j_frac * d_rho_cart_chunk[p_loc] * d_D_tot_cart[ab];
  }
}

__global__ void __launch_bounds__(64) ri_2c_grad_kernel(
    int lP, int lQ, bool same_class, ShellPairView bra_aux, ShellPairView ket_aux,
    const int* __restrict__ d_aux_sh_off, const int* __restrict__ d_aux_sh_atom, int ncart_aux,
    double omega, const double* __restrict__ d_Gamma_2c_cart, double* __restrict__ d_grad) {
  const long long n_ket = ket_aux.npair;
  const long long total = static_cast<long long>(bra_aux.npair) * n_ket;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;

  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int P = static_cast<int>(idx / n_ket);
    const int Q = static_cast<int>(idx - static_cast<long long>(P) * n_ket);
    if (same_class && Q >= P) continue;

    const int sP = bra_aux.ish[P], sQ = ket_aux.ish[Q];
    const int atP = d_aux_sh_atom[sP], atQ = d_aux_sh_atom[sQ];
    if (atP == atQ) continue;

    const int pc0 = d_aux_sh_off[sP], qc0 = d_aux_sh_off[sQ];
    auto gamma_fn = [&](int ip_c, int iq_c) -> double {
      return 2.0 * d_Gamma_2c_cart[static_cast<std::size_t>(pc0 + ip_c) * ncart_aux + (qc0 + iq_c)];
    };

    double gP[3];
    generated::eri_2c_grad_contract_fn<double>(lP, lQ, bra_aux, P, ket_aux, Q, gamma_fn, gP,
                                               omega);
    if (atP >= 0) {
      atomicAdd(&d_grad[3 * atP + 0], gP[0]);
      atomicAdd(&d_grad[3 * atP + 1], gP[1]);
      atomicAdd(&d_grad[3 * atP + 2], gP[2]);
    }
    if (atQ >= 0) {
      atomicAdd(&d_grad[3 * atQ + 0], -gP[0]);
      atomicAdd(&d_grad[3 * atQ + 1], -gP[1]);
      atomicAdd(&d_grad[3 * atQ + 2], -gP[2]);
    }
  }
}

template <int MaxOrbL>
__global__ void __launch_bounds__(64) ri_3c_grad_kernel(
    int la, int lb, int lP, ShellPairView orb, ShellPairView aux, int p_beg, int p_end,
    int pc0_chunk, const int* __restrict__ d_orb_sh_off, const int* __restrict__ d_aux_sh_off,
    const int* __restrict__ d_orb_sh_atom, const int* __restrict__ d_aux_sh_atom, int ncart_orb,
    double omega, const double* __restrict__ d_Gamma_chunk, double* __restrict__ d_grad) {
  const long long naux_p = p_end - p_beg;
  if (naux_p <= 0) return;
  const long long total = static_cast<long long>(orb.npair) * naux_p;
  const long long stride = static_cast<long long>(gridDim.x) * blockDim.x;
  const std::size_t ncart2 = static_cast<std::size_t>(ncart_orb) * ncart_orb;

  for (long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; idx < total;
       idx += stride) {
    const int ab = static_cast<int>(idx / naux_p);
    const int P = p_beg + static_cast<int>(idx - static_cast<long long>(ab) * naux_p);

    const int i = orb.ish[ab], j = orb.jsh[ab], sP = aux.ish[P];
    const int atA = d_orb_sh_atom[i], atB = d_orb_sh_atom[j], atP = d_aux_sh_atom[sP];
    if (atA == atB && atA == atP) continue;

    const int ca0 = d_orb_sh_off[i], cb0 = d_orb_sh_off[j];
    const int pc_loc = d_aux_sh_off[sP] - pc0_chunk;

    auto gamma_fn = [&](int ia, int ib_c, int iP_c) -> double {
      const double* g_slice =
          d_Gamma_chunk + static_cast<std::size_t>(pc_loc + iP_c) * ncart2;
      if (i == j) {
        return g_slice[static_cast<std::size_t>(ca0 + ia) * ncart_orb + (cb0 + ib_c)];
      }
      return g_slice[static_cast<std::size_t>(ca0 + ia) * ncart_orb + (cb0 + ib_c)] +
             g_slice[static_cast<std::size_t>(cb0 + ib_c) * ncart_orb + (ca0 + ia)];
    };

    double g9[9];
    generated::eri_3c_grad_contract_fn<MaxOrbL, double>(la, lb, lP, orb, ab, aux, P, gamma_fn, g9,
                                                        omega);
    const int atoms3[3] = {atA, atB, atP};
    for (int c = 0; c < 3; ++c) {
      const int at = atoms3[c];
      if (at >= 0) {
        atomicAdd(&d_grad[3 * at + 0], g9[3 * c + 0]);
        atomicAdd(&d_grad[3 * at + 1], g9[3 * c + 1]);
        atomicAdd(&d_grad[3 * at + 2], g9[3 * c + 2]);
      }
    }
  }
}

std::vector<RIAuxChunk> build_ri_aux_chunks_cuda(const Basis& aux_basis,
                                                 const std::vector<std::vector<int>>& aux_ish_lists,
                                                 const std::vector<int>& aux_shell_cart_off,
                                                 int max_sph_width, int max_cart_width) {
  std::vector<RIAuxChunk> chunks;
  const int nsh_aux = static_cast<int>(aux_basis.nshell());
  if (nsh_aux == 0) return chunks;
  const int num_l = static_cast<int>(aux_ish_lists.size());

  std::vector<int> l_cursor(num_l, 0);
  int s = 0;
  while (s < nsh_aux) {
    RIAuxChunk c;
    c.s0 = s;
    c.q0 = aux_basis.ao_offset(s);
    c.pc0 = aux_shell_cart_off[s];
    c.l_beg = l_cursor;

    int cur_sph = 0;
    int cur_cart = 0;
    while (s < nsh_aux) {
      const int l = aux_basis.shell(s).l;
      const int ns = n_sph(l);
      const int nc = n_cart(l);
      if (s > c.s0 && (cur_sph + ns > max_sph_width || cur_cart + nc > max_cart_width)) {
        break;
      }
      cur_sph += ns;
      cur_cart += nc;
      if (l < num_l && l_cursor[l] < static_cast<int>(aux_ish_lists[l].size()) &&
          aux_ish_lists[l][l_cursor[l]] == s) {
        l_cursor[l] += 1;
      }
      ++s;
    }
    c.s1 = s;
    c.q1 = c.q0 + cur_sph;
    c.pc1 = c.pc0 + cur_cart;
    c.l_end = l_cursor;
    chunks.push_back(std::move(c));
  }
  return chunks;
}

void factorize_cuda_ri_metric(int naux, const std::vector<DevShellPairList>& dev_aux, double omega,
                              GpuLinalgContext& linalg, DevVecJK<double>& d_factor_out,
                              bool& is_chol_out) {
  if (naux == 0) {
    is_chol_out = true;
    d_factor_out.reset();
    return;
  }
  linalg.set_stream(nullptr);
  auto* cublas = static_cast<cublasHandle_t>(linalg.cublas_raw());
  auto* cusolver = static_cast<cusolverDnHandle_t>(linalg.cusolver_raw());
  const std::size_t naux2 = static_cast<std::size_t>(naux) * naux;

  d_factor_out.resize(naux2);
  d_factor_out.zero();
  for (std::size_t ip = 0; ip < dev_aux.size(); ++ip) {
    if (dev_aux[ip].npair == 0) continue;
    for (std::size_t iq = 0; iq <= ip; ++iq) {
      if (dev_aux[iq].npair == 0) continue;
      const long long n_pq =
          static_cast<long long>(dev_aux[ip].npair) * static_cast<long long>(dev_aux[iq].npair);
      const int blocks = static_cast<int>(std::min((n_pq + 127LL) / 128LL, 4096LL));
      ri_2c_metric_kernel<<<blocks, 128>>>(dev_aux[ip].la, dev_aux[iq].la, ip == iq,
                                           dev_aux[ip].view(), dev_aux[iq].view(), naux, omega,
                                           d_factor_out.data());
      check_cuda_jk(cudaGetLastError(), "ri_2c_metric_kernel launch");
    }
  }

  DevVecJK<double> d_metric_bak(naux2);
  check_cuda_jk(cudaMemcpy(d_metric_bak.data(), d_factor_out.data(), naux2 * sizeof(double),
                           cudaMemcpyDeviceToDevice),
                "cudaMemcpy D2D metric backup");

  int lwork = 0;
  check_cusolver_jk(cusolverDnDpotrf_bufferSize(cusolver, CUBLAS_FILL_MODE_LOWER, naux,
                                                d_factor_out.data(), naux, &lwork),
                    "cusolverDnDpotrf_bufferSize");
  DevVecJK<double> d_work(static_cast<std::size_t>(std::max(1, lwork)));
  DevVecJK<int> d_info(2);
  d_info.zero();

  check_cusolver_jk(cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER, naux, d_factor_out.data(),
                                     naux, d_work.data(), lwork, d_info.data()),
                    "cusolverDnDpotrf");
  check_cholesky_diag_kernel<<<(naux + 255) / 256, 256>>>(
      naux, d_factor_out.data(), 3.1622776601683795e-6, d_info.data() + 1);
  check_cuda_jk(cudaGetLastError(), "check_cholesky_diag_kernel");

  int h_info[2] = {0, 0};
  d_info.download(h_info, 2);

  if (h_info[0] == 0 && h_info[1] == 0) {
    is_chol_out = true;
    return;
  }

  // Syevd V^{-1/2} fallback (matches Host cutoff 1e-10)
  is_chol_out = false;
  DevVecJK<double> d_w(static_cast<std::size_t>(naux));
  int lwork_ev = 0;
  check_cusolver_jk(
      cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, naux,
                                  d_metric_bak.data(), naux, d_w.data(), &lwork_ev),
      "cusolverDnDsyevd_bufferSize (RI fallback)");
  d_work.resize(static_cast<std::size_t>(std::max(1, lwork_ev)));
  check_cusolver_jk(cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                                     naux, d_metric_bak.data(), naux, d_w.data(), d_work.data(),
                                     lwork_ev, d_info.data()),
                    "cusolverDnDsyevd (RI fallback)");

  const int blocks_u = static_cast<int>((naux2 + 255) / 256);
  scale_eigvecs_inv_fourth_root_kernel<<<blocks_u, 256>>>(naux, d_w.data(), 1e-10,
                                                          d_metric_bak.data());
  const double one = 1.0, zero = 0.0;
  check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, naux, naux, naux, &one,
                              d_metric_bak.data(), naux, d_metric_bak.data(), naux, &zero,
                              d_factor_out.data(), naux),
                  "cublasDgemm (RI V^{-1/2})");
}

void ensure_rijk_workspaces(RIJKCudaCache& c, bool need_k, GemmMode k_gemm_mode,
                            cudaStream_t stream) {
  const int nao = c.nao;
  const int naux = c.naux;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  if (nao == 0 || naux == 0) return;

  auto& pool = c.linalg.pool();
  if (!c.j_workspaces_allocated) {
    c.d_dm_work = PooledBuffer<double>(pool, nao2, stream);
    c.d_v_aux = PooledBuffer<double>(pool, static_cast<std::size_t>(naux), stream);
    c.d_I_chunk = PooledBuffer<double>(pool, static_cast<std::size_t>(c.cap_b_Q) * nao2, stream);
    c.j_workspaces_allocated = true;
  }

  if (!need_k) return;

  const bool is_ozaki = (k_gemm_mode == GemmMode::OzakiInt8);
  if (c.k_workspaces_allocated && is_ozaki && !c.k_workspaces_ozaki) {
    c.d_Y_blk.reset();
    c.d_L_pchunk.reset();
    c.d_L_perm_a.reset();
    c.d_L_perm_b.reset();
    c.d_C_tile.reset();
    c.k_workspaces_allocated = false;
    c.k_workspaces_ozaki = false;
    pool.trim(stream);
  }
  if (c.k_workspaces_allocated) return;

  std::size_t free_bytes = 0, total_bytes = 0;
  check_cuda_jk(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo (RIJK workspaces)");

  c.cap_b_P = std::min(naux, 64);
  const std::size_t bytes_diis_diag = 32ULL * nao2 * sizeof(double);
  const std::size_t bytes_ozaki = is_ozaki ? (768ULL << 20) : 0ULL;
  const std::size_t headroom =
      std::min<std::size_t>(free_bytes / 3, (1024ULL << 20) + bytes_diis_diag + bytes_ozaki);
  const std::size_t raw_avail =
      (free_bytes > headroom) ? (free_bytes - headroom) : (free_bytes / 2);
  const std::size_t pct = is_ozaki ? 75ULL : 80ULL;
  const std::size_t avail = (raw_avail * pct) / 100ULL;
  const std::size_t per_col =
      (static_cast<std::size_t>(naux) + 3ULL * c.cap_b_P + 1ULL) * nao * sizeof(double);
  const int max_bi = static_cast<int>(avail / std::max<std::size_t>(1, per_col));
  c.cap_b_i = std::clamp(max_bi, 1, std::max(1, nao));

  c.d_Y_blk = PooledBuffer<double>(
      pool, static_cast<std::size_t>(naux) * nao * c.cap_b_i, stream);
  c.d_L_pchunk = PooledBuffer<double>(
      pool, static_cast<std::size_t>(c.cap_b_P) * nao * c.cap_b_i, stream);
  c.d_L_perm_a = PooledBuffer<double>(
      pool, static_cast<std::size_t>(nao) * c.cap_b_P * c.cap_b_i, stream);
  c.d_L_perm_b = PooledBuffer<double>(
      pool, static_cast<std::size_t>(nao) * c.cap_b_P * c.cap_b_i, stream);
  c.d_C_tile = PooledBuffer<double>(pool, static_cast<std::size_t>(nao) * c.cap_b_i, stream);
  c.k_workspaces_allocated = true;
  c.k_workspaces_ozaki = is_ozaki;
}

void eval_cuda_ri_3c_chunk(const RIJKCudaCache& cache, const RIAuxChunk& c, double omega,
                           double* d_I_chunk, cudaStream_t stream) {
  const int bc = c.q1 - c.q0;
  if (bc <= 0) return;
  const std::size_t nao2 = static_cast<std::size_t>(cache.nao) * cache.nao;
  check_cuda_jk(cudaMemsetAsync(d_I_chunk, 0, static_cast<std::size_t>(bc) * nao2 * sizeof(double),
                                stream),
                "cudaMemsetAsync (d_I_chunk)");

  for (std::size_t io = 0; io < cache.dev_orb.size(); ++io) {
    const auto& orb = cache.dev_orb[io];
    if (orb.npair == 0) continue;
    const bool is_spd = (orb.la <= 2 && orb.lb <= 2);
    for (std::size_t lP = 0; lP < cache.dev_aux.size(); ++lP) {
      const int pb = c.l_beg[lP];
      const int pe = c.l_end[lP];
      if (pb >= pe) continue;
      const auto& aux = cache.dev_aux[lP];
      const long long n_trip = static_cast<long long>(orb.npair) * (pe - pb);
      if (is_spd) {
        const int blocks = static_cast<int>(std::min((n_trip + 127LL) / 128LL, 8192LL));
        ri_3c_spd_kernel<<<blocks, 128, 0, stream>>>(
            orb.la, orb.lb, aux.la, orb.view(), aux.view(), pb, pe, c.q0, cache.nao, omega,
            d_I_chunk);
        check_cuda_jk(cudaGetLastError(), "ri_3c_spd_kernel launch");
      } else {
        const int blocks = static_cast<int>(std::min((n_trip + 63LL) / 64LL, 4096LL));
        ri_3c_high_l_kernel<<<blocks, 64, 0, stream>>>(
            orb.la, orb.lb, aux.la, orb.view(), aux.view(), pb, pe, c.q0, cache.nao, omega,
            d_I_chunk);
        check_cuda_jk(cudaGetLastError(), "ri_3c_high_l_kernel launch");
      }
    }
  }
}

void solve_ri_metric_v_gpu(int naux, int ncols, const RIJKCudaCache& cache, double omega,
                           double* d_Y, GpuLinalgContext& linalg, cudaStream_t stream) {
  if (naux == 0 || ncols == 0) return;
  linalg.set_stream(stream);
  auto* cublas = static_cast<cublasHandle_t>(linalg.cublas_raw());
  const bool use_omega = (omega > 0.0);
  const bool is_chol = use_omega ? cache.chol_ok_w : cache.chol_ok_0;
  const double* d_fac = use_omega ? cache.d_metric_w.data() : cache.d_metric_0.data();
  const double one = 1.0, zero = 0.0;
  if (is_chol) {
    check_cublas_jk(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, ncols, naux, &one, d_fac, naux, d_Y, ncols),
                    "cublasDtrsm forward V solve");
    check_cublas_jk(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT, ncols, naux, &one, d_fac, naux, d_Y, ncols),
                    "cublasDtrsm backward V solve");
  } else {
    PooledBuffer<double> d_tmp(linalg.pool(), static_cast<std::size_t>(naux) * ncols, stream);
    check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, ncols, naux, naux, &one, d_Y,
                                ncols, d_fac, naux, &zero, d_tmp.data(), ncols),
                    "cublasDgemm V^{-1/2} step 1");
    check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, ncols, naux, naux, &one,
                                d_tmp.data(), ncols, d_fac, naux, &zero, d_Y, ncols),
                    "cublasDgemm V^{-1/2} step 2");
  }
}

struct CudaFactorDesc {
  enum Kind { SubCols, ScaledUCols, IdentityNonsym } kind = SubCols;
  const double* d_A = nullptr;
  int total_cols = 0;
  int ncols = 0;
  double weight = 1.0;
  int count = 0;  // for ScaledUCols
};

void build_dm_factors_cuda(RIJKCudaCache& cache, const double* d_dm_sym, bool symmetrize_dm,
                           std::vector<CudaFactorDesc>& factors_out,
                           std::vector<int>& h_pos_cols, std::vector<double>& h_pos_scales,
                           std::vector<int>& h_neg_cols, std::vector<double>& h_neg_scales,
                           std::vector<int>& d_cols_offset_out, cudaStream_t stream) {
  const int nao = cache.nao;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  if (!symmetrize_dm) {
    factors_out.push_back(
        CudaFactorDesc{CudaFactorDesc::IdentityNonsym, d_dm_sym, nao, nao, 1.0, 0});
    return;
  }

  auto& pool = cache.linalg.pool();
  if (cache.d_dm_w.size() < static_cast<std::size_t>(nao)) {
    cache.d_dm_w = PooledBuffer<double>(pool, static_cast<std::size_t>(nao), stream);
    cache.d_dm_u = PooledBuffer<double>(pool, nao2, stream);
    cache.d_dm_cols = PooledBuffer<int>(pool, static_cast<std::size_t>(nao), stream);
    cache.d_dm_scales = PooledBuffer<double>(pool, static_cast<std::size_t>(nao), stream);
  }
  syevd_device(cache.linalg, nao, d_dm_sym, cache.d_dm_w.data(), cache.d_dm_u.data(), stream);
  std::vector<double> h_w(static_cast<std::size_t>(nao));
  check_cuda_jk(cudaMemcpyAsync(h_w.data(), cache.d_dm_w.data(),
                                static_cast<std::size_t>(nao) * sizeof(double),
                                cudaMemcpyDeviceToHost, stream),
                "cudaMemcpyAsync DM eigenvalues");
  check_cuda_jk(cudaStreamSynchronize(stream), "cudaStreamSynchronize DM eigenvalues");

  h_pos_cols.clear();
  h_pos_scales.clear();
  h_neg_cols.clear();
  h_neg_scales.clear();
  for (int k = 0; k < nao; ++k) {
    if (h_w[k] > 1e-14) {
      h_pos_cols.push_back(k);
      h_pos_scales.push_back(std::sqrt(h_w[k]));
    } else if (h_w[k] < -1e-14) {
      h_neg_cols.push_back(k);
      h_neg_scales.push_back(std::sqrt(-h_w[k]));
    }
  }

  std::vector<int> all_cols;
  std::vector<double> all_scales;
  d_cols_offset_out.clear();
  if (!h_pos_cols.empty()) {
    const int off = static_cast<int>(all_cols.size());
    all_cols.insert(all_cols.end(), h_pos_cols.begin(), h_pos_cols.end());
    all_scales.insert(all_scales.end(), h_pos_scales.begin(), h_pos_scales.end());
    factors_out.push_back(CudaFactorDesc{CudaFactorDesc::ScaledUCols, cache.d_dm_u.data(), nao,
                                         static_cast<int>(h_pos_cols.size()), 1.0,
                                         static_cast<int>(h_pos_cols.size())});
    d_cols_offset_out.push_back(off);
  }
  if (!h_neg_cols.empty()) {
    const int off = static_cast<int>(all_cols.size());
    all_cols.insert(all_cols.end(), h_neg_cols.begin(), h_neg_cols.end());
    all_scales.insert(all_scales.end(), h_neg_scales.begin(), h_neg_scales.end());
    factors_out.push_back(CudaFactorDesc{CudaFactorDesc::ScaledUCols, cache.d_dm_u.data(), nao,
                                         static_cast<int>(h_neg_cols.size()), -1.0,
                                         static_cast<int>(h_neg_cols.size())});
    d_cols_offset_out.push_back(off);
  }
  if (!all_cols.empty()) {
    check_cuda_jk(cudaMemcpyAsync(cache.d_dm_cols.data(), all_cols.data(),
                                  all_cols.size() * sizeof(int), cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync DM cols");
    check_cuda_jk(cudaMemcpyAsync(cache.d_dm_scales.data(), all_scales.data(),
                                  all_scales.size() * sizeof(double), cudaMemcpyHostToDevice,
                                  stream),
                  "cudaMemcpyAsync DM scales");
  }
}

__global__ void cast_d2f_rect_kernel(int rows, int cols, int ld, bool trans,
                                     const double* __restrict__ src, float* __restrict__ dst) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int r = static_cast<int>(idx / cols);
    const int c = static_cast<int>(idx - static_cast<std::size_t>(r) * cols);
    const std::size_t src_idx =
        trans ? (static_cast<std::size_t>(c) * ld + r) : (static_cast<std::size_t>(r) * ld + c);
    dst[idx] = static_cast<float>(src[src_idx]);
  }
}

__global__ void epilogue_f2d_rect_kernel(int rows, int cols, double alpha,
                                         const float* __restrict__ cf, double beta,
                                         double* __restrict__ dc, int ldc) {
  const std::size_t total = static_cast<std::size_t>(rows) * cols;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += stride) {
    const int r = static_cast<int>(idx / cols);
    const int c = static_cast<int>(idx - static_cast<std::size_t>(r) * cols);
    const std::size_t dst_idx = static_cast<std::size_t>(r) * ldc + c;
    const double v = alpha * static_cast<double>(cf[idx]);
    dc[dst_idx] = (beta == 0.0) ? v : (beta * dc[dst_idx] + v);
  }
}

// IEEE-754 24-bit FP32 cublasSgemm for Precision::FP32 (avoiding TF32 10-bit truncation),
// delegating FP64 and OzakiInt8 to gemm_device.
void rijk_gemm_device(GpuLinalgContext& ctx, int m, int n, int k, double alpha, const double* d_A,
                      int lda, bool transA, const double* d_B, int ldb, bool transB, double beta,
                      double* d_C, int ldc, GemmMode mode, cudaStream_t stream, int ozaki_slices) {
  if (mode != GemmMode::FP32) {
    gemm_device(ctx, m, n, k, alpha, d_A, lda, transA, d_B, ldb, transB, beta, d_C, ldc, mode,
                stream, ozaki_slices);
    return;
  }
  if (m <= 0 || n <= 0) return;
  if (k <= 0 || alpha == 0.0) {
    gemm_device(ctx, m, n, k, alpha, d_A, lda, transA, d_B, ldb, transB, beta, d_C, ldc,
                GemmMode::FP64, stream, ozaki_slices);
    return;
  }
  ctx.set_stream(stream);
  auto* cublas = static_cast<cublasHandle_t>(ctx.cublas_raw());
  auto& pool = ctx.pool();
  const std::size_t n_a = static_cast<std::size_t>(m) * k;
  const std::size_t n_b = static_cast<std::size_t>(k) * n;
  const std::size_t n_c = static_cast<std::size_t>(m) * n;
  PooledBuffer<float> d_Af(pool, n_a, stream);
  PooledBuffer<float> d_Bf(pool, n_b, stream);
  PooledBuffer<float> d_Cf(pool, n_c, stream);

  const int ba = static_cast<int>(std::min<std::size_t>((n_a + 255) / 256, 4096));
  const int bb = static_cast<int>(std::min<std::size_t>((n_b + 255) / 256, 4096));
  const int bc = static_cast<int>(std::min<std::size_t>((n_c + 255) / 256, 4096));
  cast_d2f_rect_kernel<<<ba, 256, 0, stream>>>(m, k, lda, transA, d_A, d_Af.data());
  cast_d2f_rect_kernel<<<bb, 256, 0, stream>>>(k, n, ldb, transB, d_B, d_Bf.data());

  const float f_one = 1.0f, f_zero = 0.0f;
  check_cublas_jk(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &f_one, d_Bf.data(), n,
                              d_Af.data(), k, &f_zero, d_Cf.data(), n),
                  "cublasSgemm (RI-JK IEEE FP32)");
  epilogue_f2d_rect_kernel<<<bc, 256, 0, stream>>>(m, n, alpha, d_Cf.data(), beta, d_C, ldc);
  check_cuda_jk(cudaGetLastError(), "epilogue_f2d_rect_kernel");
}

void run_rijk_single_pass_cuda(RIJKCudaCache& cache, double pass_omega, bool pass_j, bool pass_k,
                               double k_coeff, bool acc_k, bool symmetrize_dm,
                               GemmMode k_gemm_mode, int ozaki_slices,
                               const std::vector<CudaFactorDesc>& factors,
                               const std::vector<int>& u_col_offsets, double* d_vj, double* d_vk,
                               cudaStream_t stream) {
  const int nao = cache.nao;
  const int naux = cache.naux;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const int nao2_i = static_cast<int>(nao2);
  cache.linalg.set_stream(stream);
  auto* cublas = static_cast<cublasHandle_t>(cache.linalg.cublas_raw());
  const double one = 1.0, zero = 0.0;

  if (pass_j && (!pass_k || factors.empty() || k_coeff == 0.0)) {
    for (const auto& c : cache.chunks_scf) {
      const int bc = c.q1 - c.q0;
      eval_cuda_ri_3c_chunk(cache, c, pass_omega, cache.d_I_chunk.data(), stream);
      // v_aux[q0:q1] = I_chunk[bc x nao2] * dm_work[nao2]
      check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, nao2_i, bc, &one, cache.d_I_chunk.data(),
                                  nao2_i, cache.d_dm_work.data(), 1, &zero,
                                  cache.d_v_aux.data() + c.q0, 1),
                      "cublasDgemv RI-J pass 1");
    }
    solve_ri_metric_v_gpu(naux, 1, cache, pass_omega, cache.d_v_aux.data(), cache.linalg, stream);
    for (std::size_t ci = 0; ci < cache.chunks_scf.size(); ++ci) {
      const auto& c = cache.chunks_scf[ci];
      const int bc = c.q1 - c.q0;
      const double beta_j = (ci == 0) ? 0.0 : 1.0;
      eval_cuda_ri_3c_chunk(cache, c, pass_omega, cache.d_I_chunk.data(), stream);
      check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_N, nao2_i, bc, &one, cache.d_I_chunk.data(),
                                  nao2_i, cache.d_v_aux.data() + c.q0, 1, &beta_j, d_vj, 1),
                      "cublasDgemv RI-J pass 2");
    }
    return;
  }

  if (!pass_k || k_coeff == 0.0 || factors.empty()) return;

  const bool is_chol = (pass_omega > 0.0) ? cache.chol_ok_w : cache.chol_ok_0;
  const double* d_fac =
      (pass_omega > 0.0) ? cache.d_metric_w.data() : cache.d_metric_0.data();

  int total_tiles = 0;
  for (const auto& f : factors) {
    const int step =
        (f.kind == CudaFactorDesc::IdentityNonsym) ? std::max(1, cache.cap_b_i / 2) : cache.cap_b_i;
    total_tiles += (f.ncols + step - 1) / step;
  }

  bool j_pass1_done = false;
  bool j_pass2_done = !pass_j;
  bool first_k_write = !acc_k;
  int u_fac_cursor = 0;

  for (std::size_t fi = 0; fi < factors.size(); ++fi) {
    const auto& f = factors[fi];
    const bool nonsym_pair = (f.kind == CudaFactorDesc::IdentityNonsym);
    const int step = nonsym_pair ? std::max(1, cache.cap_b_i / 2) : cache.cap_b_i;
    const int u_base = (f.kind == CudaFactorDesc::ScaledUCols) ? u_col_offsets[u_fac_cursor++] : 0;

    for (int i0 = 0; i0 < f.ncols; i0 += step) {
      const int bi = std::min(step, f.ncols - i0);
      const int bi_tot = nonsym_pair ? (2 * bi) : bi;
      const int col_blocks =
          static_cast<int>(std::min<std::size_t>((static_cast<std::size_t>(nao) * bi + 255) / 256,
                                                 4096));
      if (f.kind == CudaFactorDesc::SubCols) {
        extract_col_tile_kernel<<<col_blocks, 256, 0, stream>>>(
            nao, f.total_cols, i0, bi, 0, bi_tot, f.d_A, cache.d_C_tile.data());
      } else if (f.kind == CudaFactorDesc::ScaledUCols) {
        extract_cols_from_u_scaled_kernel<<<col_blocks, 256, 0, stream>>>(
            nao, bi, 0, bi_tot, cache.d_dm_cols.data() + u_base + i0,
            cache.d_dm_scales.data() + u_base + i0, f.d_A, cache.d_C_tile.data());
      } else {
        extract_col_tile_kernel<<<col_blocks, 256, 0, stream>>>(
            nao, f.total_cols, i0, bi, 0, bi_tot, f.d_A, cache.d_C_tile.data());
        set_identity_col_tile_kernel<<<col_blocks, 256, 0, stream>>>(
            nao, i0, bi, bi, bi_tot, cache.d_C_tile.data());
      }
      check_cuda_jk(cudaGetLastError(), "C_tile prepare kernel");

      const bool collect_j_now = pass_j && !j_pass1_done;
      const bool apply_j_now = pass_j && j_pass1_done && !j_pass2_done;

      for (std::size_t ci = 0; ci < cache.chunks_scf.size(); ++ci) {
        const auto& c = cache.chunks_scf[ci];
        const int bc = c.q1 - c.q0;
        eval_cuda_ri_3c_chunk(cache, c, pass_omega, cache.d_I_chunk.data(), stream);

        if (collect_j_now) {
          check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, nao2_i, bc, &one, cache.d_I_chunk.data(),
                                      nao2_i, cache.d_dm_work.data(), 1, &zero,
                                      cache.d_v_aux.data() + c.q0, 1),
                          "cublasDgemv RI-J pass 1 (piggyback)");
        }
        if (apply_j_now) {
          const double beta_j = (ci == 0) ? 0.0 : 1.0;
          check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_N, nao2_i, bc, &one, cache.d_I_chunk.data(),
                                      nao2_i, cache.d_v_aux.data() + c.q0, 1, &beta_j, d_vj, 1),
                          "cublasDgemv RI-J pass 2 (piggyback)");
        }

        // Half-transform d_Y_blk[q0:q1, :, :] = d_I_chunk[bc * nao, nao] * d_C_tile[nao, bi_tot]
        rijk_gemm_device(cache.linalg, bc * nao, bi_tot, nao, 1.0, cache.d_I_chunk.data(), nao,
                         false, cache.d_C_tile.data(), bi_tot, false, 0.0,
                         cache.d_Y_blk.data() + static_cast<std::size_t>(c.q0) * nao * bi_tot,
                         bi_tot, k_gemm_mode, stream, ozaki_slices);
      }

      if (collect_j_now) {
        solve_ri_metric_v_gpu(naux, 1, cache, pass_omega, cache.d_v_aux.data(), cache.linalg,
                              stream);
        j_pass1_done = true;
        if (total_tiles <= 1) {
          for (std::size_t ci = 0; ci < cache.chunks_scf.size(); ++ci) {
            const auto& c = cache.chunks_scf[ci];
            const int bc = c.q1 - c.q0;
            const double beta_j = (ci == 0) ? 0.0 : 1.0;
            eval_cuda_ri_3c_chunk(cache, c, pass_omega, cache.d_I_chunk.data(), stream);
            check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_N, nao2_i, bc, &one,
                                        cache.d_I_chunk.data(), nao2_i, cache.d_v_aux.data() + c.q0,
                                        1, &beta_j, d_vj, 1),
                            "cublasDgemv RI-J pass 2");
          }
          j_pass2_done = true;
        }
      } else if (apply_j_now) {
        j_pass2_done = true;
      }

      const int nao_bi_tot = nao * bi_tot;
      if (is_chol) {
        check_cublas_jk(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                    CUBLAS_DIAG_NON_UNIT, nao_bi_tot, naux, &one, d_fac, naux,
                                    cache.d_Y_blk.data(), nao_bi_tot),
                        "cublasDtrsm forward Y_blk");
      }

      const double w_eff = k_coeff * f.weight;
      for (int p0 = 0; p0 < naux; p0 += cache.cap_b_P) {
        const int bp = std::min(cache.cap_b_P, naux - p0);
        const double* L_src = nullptr;
        if (is_chol) {
          L_src = cache.d_Y_blk.data() + static_cast<std::size_t>(p0) * nao_bi_tot;
        } else {
          // Row-major L_pchunk[bp, nao_bi_tot] = fac[p0:p1, naux] * Y_blk[naux, nao_bi_tot]
          check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nao_bi_tot, bp, naux, &one,
                                      cache.d_Y_blk.data(), nao_bi_tot,
                                      d_fac + static_cast<std::size_t>(p0) * naux, naux, &zero,
                                      cache.d_L_pchunk.data(), nao_bi_tot),
                          "cublasDgemm V^{-1/2} P-chunk");
          L_src = cache.d_L_pchunk.data();
        }

        const std::size_t perm_n = static_cast<std::size_t>(bp) * nao * bi;
        const int p_blocks = static_cast<int>(std::min<std::size_t>((perm_n + 255) / 256, 4096));
        permute_ri_mo_chunk_kernel<<<p_blocks, 256, 0, stream>>>(
            bp, nao, bi, 0, bi_tot, L_src, cache.d_L_perm_a.data());
        check_cuda_jk(cudaGetLastError(), "permute_ri_mo_chunk_kernel a");
        if (nonsym_pair) {
          permute_ri_mo_chunk_kernel<<<p_blocks, 256, 0, stream>>>(
              bp, nao, bi, bi, bi_tot, L_src, cache.d_L_perm_b.data());
          check_cuda_jk(cudaGetLastError(), "permute_ri_mo_chunk_kernel b");
        }

        const double beta_k = first_k_write ? 0.0 : 1.0;
        first_k_write = false;
        const double* d_right =
            nonsym_pair ? cache.d_L_perm_b.data() : cache.d_L_perm_a.data();
        rijk_gemm_device(cache.linalg, nao, nao, bp * bi, w_eff, cache.d_L_perm_a.data(), bp * bi,
                         false, d_right, bp * bi, true, beta_k, d_vk, nao, k_gemm_mode, stream,
                         ozaki_slices);
      }
    }
  }

  if (symmetrize_dm) {
    dim3 threads(16, 16);
    dim3 blocks((nao + 15) / 16, (nao + 15) / 16);
    symmetrize_inplace_jk_kernel<<<blocks, threads, 0, stream>>>(nao, d_vk);
    check_cuda_jk(cudaGetLastError(), "symmetrize_inplace_jk_kernel (RI-K)");
  }
}

void run_rijk_packed_uks_pass_cuda(RIJKCudaCache& cache, double pass_omega, bool pass_j,
                                   bool pass_ka, bool pass_kb, double k_coeff, bool acc_k,
                                   GemmMode k_gemm_mode, int ozaki_slices,
                                   const double* d_c_alpha, int n_alpha, const double* d_c_beta,
                                   int n_beta, double* d_vj_tot, double* d_vk_alpha,
                                   double* d_vk_beta, cudaStream_t stream) {
  const int nao = cache.nao;
  const int naux = cache.naux;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const int nao2_i = static_cast<int>(nao2);
  cache.linalg.set_stream(stream);
  auto* cublas = static_cast<cublasHandle_t>(cache.linalg.cublas_raw());
  const double one = 1.0, zero = 0.0;

  const int bi_tot = n_alpha + n_beta;
  {
    const std::size_t na_tot = static_cast<std::size_t>(nao) * n_alpha;
    const std::size_t nb_tot = static_cast<std::size_t>(nao) * n_beta;
    const int ba = static_cast<int>(std::min<std::size_t>((na_tot + 255) / 256, 4096));
    const int bb = static_cast<int>(std::min<std::size_t>((nb_tot + 255) / 256, 4096));
    extract_col_tile_kernel<<<ba, 256, 0, stream>>>(
        nao, n_alpha, 0, n_alpha, 0, bi_tot, d_c_alpha, cache.d_C_tile.data());
    extract_col_tile_kernel<<<bb, 256, 0, stream>>>(
        nao, n_beta, 0, n_beta, n_alpha, bi_tot, d_c_beta, cache.d_C_tile.data());
    check_cuda_jk(cudaGetLastError(), "pack C_alpha|C_beta kernel");
  }

  for (const auto& c : cache.chunks_scf) {
    const int bc = c.q1 - c.q0;
    eval_cuda_ri_3c_chunk(cache, c, pass_omega, cache.d_I_chunk.data(), stream);
    if (pass_j) {
      check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, nao2_i, bc, &one, cache.d_I_chunk.data(),
                                  nao2_i, cache.d_dm_work.data(), 1, &zero,
                                  cache.d_v_aux.data() + c.q0, 1),
                      "cublasDgemv RI-J packed UKS pass 1");
    }
    rijk_gemm_device(cache.linalg, bc * nao, bi_tot, nao, 1.0, cache.d_I_chunk.data(), nao, false,
                     cache.d_C_tile.data(), bi_tot, false, 0.0,
                     cache.d_Y_blk.data() + static_cast<std::size_t>(c.q0) * nao * bi_tot, bi_tot,
                     k_gemm_mode, stream, ozaki_slices);
  }

  if (pass_j) {
    solve_ri_metric_v_gpu(naux, 1, cache, pass_omega, cache.d_v_aux.data(), cache.linalg, stream);
    for (std::size_t ci = 0; ci < cache.chunks_scf.size(); ++ci) {
      const auto& c = cache.chunks_scf[ci];
      const int bc = c.q1 - c.q0;
      const double beta_j = (ci == 0) ? 0.0 : 1.0;
      eval_cuda_ri_3c_chunk(cache, c, pass_omega, cache.d_I_chunk.data(), stream);
      check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_N, nao2_i, bc, &one, cache.d_I_chunk.data(),
                                  nao2_i, cache.d_v_aux.data() + c.q0, 1, &beta_j, d_vj_tot, 1),
                      "cublasDgemv RI-J packed UKS pass 2");
    }
  }

  if ((!pass_ka && !pass_kb) || k_coeff == 0.0) return;

  const bool is_chol = (pass_omega > 0.0) ? cache.chol_ok_w : cache.chol_ok_0;
  const double* d_fac =
      (pass_omega > 0.0) ? cache.d_metric_w.data() : cache.d_metric_0.data();
  const int nao_bi_tot = nao * bi_tot;
  if (is_chol) {
    check_cublas_jk(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, nao_bi_tot, naux, &one, d_fac, naux,
                                cache.d_Y_blk.data(), nao_bi_tot),
                    "cublasDtrsm forward Y_blk (UKS)");
  }

  for (int p0 = 0; p0 < naux; p0 += cache.cap_b_P) {
    const int bp = std::min(cache.cap_b_P, naux - p0);
    const double* L_src = nullptr;
    if (is_chol) {
      L_src = cache.d_Y_blk.data() + static_cast<std::size_t>(p0) * nao_bi_tot;
    } else {
      check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nao_bi_tot, bp, naux, &one,
                                  cache.d_Y_blk.data(), nao_bi_tot,
                                  d_fac + static_cast<std::size_t>(p0) * naux, naux, &zero,
                                  cache.d_L_pchunk.data(), nao_bi_tot),
                      "cublasDgemm V^{-1/2} P-chunk (UKS)");
      L_src = cache.d_L_pchunk.data();
    }

    const double beta_k = (p0 == 0 && !acc_k) ? 0.0 : 1.0;
    if (pass_ka) {
      const std::size_t perm_a = static_cast<std::size_t>(bp) * nao * n_alpha;
      const int p_blocks = static_cast<int>(std::min<std::size_t>((perm_a + 255) / 256, 4096));
      permute_ri_mo_chunk_kernel<<<p_blocks, 256, 0, stream>>>(
          bp, nao, n_alpha, 0, bi_tot, L_src, cache.d_L_perm_a.data());
      rijk_gemm_device(cache.linalg, nao, nao, bp * n_alpha, k_coeff, cache.d_L_perm_a.data(),
                       bp * n_alpha, false, cache.d_L_perm_a.data(), bp * n_alpha, true, beta_k,
                       d_vk_alpha, nao, k_gemm_mode, stream, ozaki_slices);
    }
    if (pass_kb) {
      const std::size_t perm_b = static_cast<std::size_t>(bp) * nao * n_beta;
      const int p_blocks = static_cast<int>(std::min<std::size_t>((perm_b + 255) / 256, 4096));
      permute_ri_mo_chunk_kernel<<<p_blocks, 256, 0, stream>>>(
          bp, nao, n_beta, n_alpha, bi_tot, L_src, cache.d_L_perm_b.data());
      rijk_gemm_device(cache.linalg, nao, nao, bp * n_beta, k_coeff, cache.d_L_perm_b.data(),
                       bp * n_beta, false, cache.d_L_perm_b.data(), bp * n_beta, true, beta_k,
                       d_vk_beta, nao, k_gemm_mode, stream, ozaki_slices);
    }
  }

  dim3 threads(16, 16);
  dim3 blocks((nao + 15) / 16, (nao + 15) / 16);
  if (pass_ka) {
    symmetrize_inplace_jk_kernel<<<blocks, threads, 0, stream>>>(nao, d_vk_alpha);
  }
  if (pass_kb) {
    symmetrize_inplace_jk_kernel<<<blocks, threads, 0, stream>>>(nao, d_vk_beta);
  }
  check_cuda_jk(cudaGetLastError(), "symmetrize_inplace_jk_kernel (UKS RI-K)");
}

}  // namespace

RIJKCudaCache* build_rijk_cuda_cache_native(
    const Basis& basis, const Basis& aux_basis, const std::vector<ShellPairData>& orb_lists,
    const std::vector<ShellPairData>& aux_lists, const std::vector<int>& orb_shell_cart_off,
    const std::vector<int>& aux_shell_cart_off, int ncart_orb, int ncart_aux,
    double rsh_omega) {
  auto c = std::make_unique<RIJKCudaCache>();
  c->nao = basis.nao();
  c->naux = aux_basis.nao();
  c->ncart_orb = ncart_orb;
  c->ncart_aux = ncart_aux;
  c->rsh_omega = (rsh_omega > 0.0) ? rsh_omega : 0.0;
  c->basis = basis;
  c->aux_basis = aux_basis;
  c->orb_shell_cart_off = orb_shell_cart_off;
  c->aux_shell_cart_off = aux_shell_cart_off;
  if (c->nao == 0 || c->naux == 0) return c.release();

  c->aux_ish_lists.resize(aux_lists.size());
  c->dev_aux.reserve(aux_lists.size());
  for (std::size_t i = 0; i < aux_lists.size(); ++i) {
    c->aux_ish_lists[i] = aux_lists[i].ish;
    c->dev_aux.emplace_back(aux_lists[i]);
  }

  c->dev_orb.reserve(orb_lists.size());
  for (const auto& op : orb_lists) c->dev_orb.emplace_back(op);

  factorize_cuda_ri_metric(c->naux, c->dev_aux, 0.0, c->linalg, c->d_metric_0, c->chol_ok_0);
  if (c->rsh_omega > 0.0) {
    factorize_cuda_ri_metric(c->naux, c->dev_aux, c->rsh_omega, c->linalg, c->d_metric_w,
                             c->chol_ok_w);
  }

  const std::size_t nao2 = static_cast<std::size_t>(c->nao) * c->nao;
  const int max_sph_scf =
      (static_cast<std::size_t>(64) * nao2 * sizeof(double) <= (1280ULL << 20))
          ? 64
          : std::max(16, static_cast<int>((1024ULL << 20) /
                                          std::max<std::size_t>(1, nao2 * sizeof(double))));
  c->chunks_scf = build_ri_aux_chunks_cuda(c->aux_basis, c->aux_ish_lists, c->aux_shell_cart_off,
                                           max_sph_scf, 256);
  int max_bq = 0;
  for (const auto& ch : c->chunks_scf) max_bq = std::max(max_bq, ch.q1 - ch.q0);
  c->cap_b_Q = std::max(1, max_bq);

  check_cuda_jk(cudaDeviceSynchronize(), "build_rijk_cuda_cache_native sync");
  return c.release();
}

void destroy_rijk_cuda_cache(RIJKCudaCache* p) {
  if (p == nullptr) return;
  DeviceMemoryPool& pool = p->linalg.pool();
  delete p;
  pool.trim();
}

void release_rijk_workspaces(RIJKCudaCache& cache) {
  cache.d_dm_work.reset();
  cache.d_v_aux.reset();
  cache.d_I_chunk.reset();
  cache.d_Y_blk.reset();
  cache.d_L_pchunk.reset();
  cache.d_L_perm_a.reset();
  cache.d_L_perm_b.reset();
  cache.d_C_tile.reset();
  cache.d_dm_w.reset();
  cache.d_dm_u.reset();
  cache.d_dm_cols.reset();
  cache.d_dm_scales.reset();
  cache.j_workspaces_allocated = false;
  cache.k_workspaces_allocated = false;
  cache.k_workspaces_ozaki = false;
  cache.linalg.pool().trim();
}

bool download_rijk_metric_factor(const RIJKCudaCache& cache, double omega, bool& is_cholesky_out,
                                 std::vector<double>& factor_out) {
  const int n = cache.naux;
  if (n == 0) {
    is_cholesky_out = true;
    factor_out.clear();
    return true;
  }
  const DevVecJK<double>* src = nullptr;
  if (omega <= 0.0) {
    src = &cache.d_metric_0;
    is_cholesky_out = cache.chol_ok_0;
  } else if (cache.rsh_omega > 0.0 && std::abs(cache.rsh_omega - omega) <= 1e-12) {
    src = &cache.d_metric_w;
    is_cholesky_out = cache.chol_ok_w;
  } else {
    return false;
  }
  if (src->data() == nullptr) return false;

  const std::size_t n2 = static_cast<std::size_t>(n) * n;
  std::vector<double> tmp(n2);
  src->download(tmp.data(), n2);
  if (is_cholesky_out) {
    // d_metric_* stores col-major lower triangular L (L_col(i, j) at j * n + i).
    // Transpose into row-major lower-triangular L for Host RIHostMetric.
    factor_out.assign(n2, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        factor_out[static_cast<std::size_t>(i) * n + j] =
            tmp[static_cast<std::size_t>(j) * n + i];
      }
    }
  } else {
    factor_out = std::move(tmp);
  }
  return true;
}

void run_rijk_mo_cuda_device(RIJKCudaCache& cache, const double* d_dm, const double* d_c_occ,
                             int n_occ, double occ_scale, double* d_vj, double* d_vk,
                             const JKOptions& opts, void* raw_stream) {
  auto stream = static_cast<cudaStream_t>(raw_stream);
  const int nao = cache.nao;
  const int naux = cache.naux;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const bool do_j = opts.compute_j && (d_vj != nullptr);
  const bool do_k = opts.compute_k && (d_vk != nullptr);
  if ((!do_j && !do_k) || nao == 0 || naux == 0) {
    if (do_j) check_cuda_jk(cudaMemsetAsync(d_vj, 0, nao2 * sizeof(double), stream), "zero vj");
    if (do_k) check_cuda_jk(cudaMemsetAsync(d_vk, 0, nao2 * sizeof(double), stream), "zero vk");
    return;
  }

  const GemmMode k_gemm_mode =
      (opts.precision == Precision::FP32) ? GemmMode::FP32 : opts.gemm_mode;
  const bool need_any_k =
      do_k && (opts.k_scale != 0.0) && (n_occ != 0) &&
      (!opts.rsh_mode || opts.k_sr != 0.0 || (opts.omega > 0.0 && opts.k_lr != opts.k_sr));

  ensure_rijk_workspaces(cache, need_any_k, k_gemm_mode, stream);
  cache.linalg.set_stream(stream);

  const int dm_blocks = std::min(static_cast<int>((nao2 + 255) / 256), 1024);
  if (d_dm != nullptr) {
    copy_or_symmetrize_dm_kernel<<<dm_blocks, 256, 0, stream>>>(nao, opts.symmetrize_dm, d_dm,
                                                                cache.d_dm_work.data());
    check_cuda_jk(cudaGetLastError(), "copy_or_symmetrize_dm_kernel");
  } else if (d_c_occ != nullptr && n_occ > 0) {
    gemm_device(cache.linalg, nao, nao, n_occ, occ_scale, d_c_occ, n_occ, false, d_c_occ, n_occ,
                true, 0.0, cache.d_dm_work.data(), nao, GemmMode::FP64, stream);
  } else {
    check_cuda_jk(cudaMemsetAsync(cache.d_dm_work.data(), 0, nao2 * sizeof(double), stream),
                  "zero d_dm_work");
  }

  std::vector<CudaFactorDesc> factors;
  std::vector<int> pos_cols, neg_cols, u_col_offsets;
  std::vector<double> pos_scales, neg_scales;

  if (need_any_k) {
    if (d_c_occ != nullptr && n_occ > 0) {
      factors.push_back(
          CudaFactorDesc{CudaFactorDesc::SubCols, d_c_occ, n_occ, n_occ, occ_scale, 0});
    } else {
      build_dm_factors_cuda(cache, cache.d_dm_work.data(), opts.symmetrize_dm, factors, pos_cols,
                            pos_scales, neg_cols, neg_scales, u_col_offsets, stream);
    }
  }

  if (do_k && (!need_any_k || factors.empty())) {
    check_cuda_jk(cudaMemsetAsync(d_vk, 0, nao2 * sizeof(double), stream), "zero d_vk (empty)");
  }

  if (!opts.rsh_mode) {
    const bool pass_k = need_any_k && !factors.empty() && (opts.k_scale != 0.0);
    run_rijk_single_pass_cuda(cache, opts.omega, do_j, pass_k, opts.k_scale, /*acc_k=*/false,
                              opts.symmetrize_dm, k_gemm_mode, opts.ozaki_slices, factors,
                              u_col_offsets, d_vj, d_vk, stream);
  } else {
    const double k1 = opts.k_sr * opts.k_scale;
    const double k2 = (opts.k_lr - opts.k_sr) * opts.k_scale;
    const bool pass1_k = need_any_k && !factors.empty() && (k1 != 0.0);
    const bool pass2_k = need_any_k && !factors.empty() && (opts.omega > 0.0) && (k2 != 0.0);
    if (do_j || pass1_k) {
      run_rijk_single_pass_cuda(cache, /*pass_omega=*/0.0, do_j, pass1_k, k1, /*acc_k=*/false,
                                opts.symmetrize_dm, k_gemm_mode, opts.ozaki_slices, factors,
                                u_col_offsets, d_vj, d_vk, stream);
    }
    if (pass2_k) {
      run_rijk_single_pass_cuda(cache, opts.omega, /*pass_j=*/false, /*pass_k=*/true, k2,
                                /*acc_k=*/pass1_k, opts.symmetrize_dm, k_gemm_mode,
                                opts.ozaki_slices, factors, u_col_offsets, d_vj, d_vk, stream);
    }
  }
}

void run_rijk_mo_cuda(RIJKCudaCache& cache, const double* h_dm, const double* h_c_occ, int n_occ,
                      double occ_scale, double* h_vj, double* h_vk, const JKOptions& opts) {
  const int nao = cache.nao;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const bool do_j = opts.compute_j && (h_vj != nullptr);
  const bool do_k = opts.compute_k && (h_vk != nullptr);
  if ((!do_j && !do_k) || nao2 == 0 || cache.naux == 0) {
    if (do_j) std::fill(h_vj, h_vj + nao2, 0.0);
    if (do_k) std::fill(h_vk, h_vk + nao2, 0.0);
    return;
  }

  if (h_dm != nullptr) {
    cache.d_dm_stage.upload_async(h_dm, nao2, nullptr);
  }
  const double* d_c_ptr = nullptr;
  if (h_c_occ != nullptr && n_occ > 0) {
    cache.d_c_stage.upload_async(h_c_occ, static_cast<std::size_t>(nao) * n_occ, nullptr);
    d_c_ptr = cache.d_c_stage.data();
  }
  if (do_j) cache.d_vj_stage.resize(nao2);
  if (do_k) cache.d_vk_stage.resize(nao2);

  run_rijk_mo_cuda_device(cache, (h_dm != nullptr) ? cache.d_dm_stage.data() : nullptr, d_c_ptr,
                          n_occ, occ_scale, do_j ? cache.d_vj_stage.data() : nullptr,
                          do_k ? cache.d_vk_stage.data() : nullptr, opts, nullptr);
  if (do_j) cache.d_vj_stage.download_async(h_vj, nao2, nullptr);
  if (do_k) cache.d_vk_stage.download_async(h_vk, nao2, nullptr);
  check_cuda_jk(cudaStreamSynchronize(nullptr), "run_rijk_mo_cuda sync");
}

void run_rijk_uks_mo_cuda_device(RIJKCudaCache& cache, const double* d_dm_alpha,
                                 const double* d_dm_beta, const double* d_c_occ_alpha, int n_alpha,
                                 const double* d_c_occ_beta, int n_beta, double* d_vj_tot,
                                 double* d_vk_alpha, double* d_vk_beta, const JKOptions& opts,
                                 void* raw_stream) {
  auto stream = static_cast<cudaStream_t>(raw_stream);
  const int nao = cache.nao;
  const int naux = cache.naux;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const bool do_j = opts.compute_j && (d_vj_tot != nullptr);
  const bool do_ka = opts.compute_k && (d_vk_alpha != nullptr);
  const bool do_kb = opts.compute_k && (d_vk_beta != nullptr);
  if ((!do_j && !do_ka && !do_kb) || nao == 0 || naux == 0) {
    if (do_j) check_cuda_jk(cudaMemsetAsync(d_vj_tot, 0, nao2 * sizeof(double), stream), "zero j");
    if (do_ka) check_cuda_jk(cudaMemsetAsync(d_vk_alpha, 0, nao2 * sizeof(double), stream), "z ka");
    if (do_kb) check_cuda_jk(cudaMemsetAsync(d_vk_beta, 0, nao2 * sizeof(double), stream), "z kb");
    return;
  }

  const GemmMode k_gemm_mode =
      (opts.precision == Precision::FP32) ? GemmMode::FP32 : opts.gemm_mode;
  const bool need_any_k =
      (do_ka || do_kb) && (opts.k_scale != 0.0) &&
      (!opts.rsh_mode || opts.k_sr != 0.0 || (opts.omega > 0.0 && opts.k_lr != opts.k_sr));

  ensure_rijk_workspaces(cache, need_any_k, k_gemm_mode, stream);
  cache.linalg.set_stream(stream);

  const bool can_pack = need_any_k && do_ka && do_kb && (d_c_occ_alpha != nullptr) &&
                        (d_c_occ_beta != nullptr) && (n_alpha > 0) && (n_beta > 0) &&
                        (n_alpha + n_beta <= cache.cap_b_i);

  if (can_pack) {
    if (do_j) {
      const int dm_blocks = std::min(static_cast<int>((nao2 + 255) / 256), 1024);
      add_dm_pair_kernel<<<dm_blocks, 256, 0, stream>>>(nao, opts.symmetrize_dm, d_dm_alpha,
                                                        d_dm_beta, cache.d_dm_work.data());
      check_cuda_jk(cudaGetLastError(), "add_dm_pair_kernel (UKS packed)");
    }

    if (!opts.rsh_mode) {
      run_rijk_packed_uks_pass_cuda(cache, opts.omega, do_j, do_ka, do_kb, opts.k_scale,
                                    /*acc_k=*/false, k_gemm_mode, opts.ozaki_slices, d_c_occ_alpha,
                                    n_alpha, d_c_occ_beta, n_beta, d_vj_tot, d_vk_alpha, d_vk_beta,
                                    stream);
    } else {
      const double k1 = opts.k_sr * opts.k_scale;
      const double k2 = (opts.k_lr - opts.k_sr) * opts.k_scale;
      const bool pass1_k = (k1 != 0.0);
      const bool pass2_k = (opts.omega > 0.0) && (k2 != 0.0);
      if (do_j || pass1_k) {
        run_rijk_packed_uks_pass_cuda(cache, /*pass_omega=*/0.0, do_j, pass1_k && do_ka,
                                      pass1_k && do_kb, k1, /*acc_k=*/false, k_gemm_mode,
                                      opts.ozaki_slices, d_c_occ_alpha, n_alpha, d_c_occ_beta,
                                      n_beta, d_vj_tot, d_vk_alpha, d_vk_beta, stream);
      }
      if (pass2_k) {
        run_rijk_packed_uks_pass_cuda(cache, opts.omega, /*pass_j=*/false, do_ka, do_kb, k2,
                                      /*acc_k=*/pass1_k, k_gemm_mode, opts.ozaki_slices,
                                      d_c_occ_alpha, n_alpha, d_c_occ_beta, n_beta, d_vj_tot,
                                      d_vk_alpha, d_vk_beta, stream);
      }
      if (!pass1_k && !pass2_k) {
        check_cuda_jk(cudaMemsetAsync(d_vk_alpha, 0, nao2 * sizeof(double), stream), "zero ka");
        check_cuda_jk(cudaMemsetAsync(d_vk_beta, 0, nao2 * sizeof(double), stream), "zero kb");
      }
    }
    return;
  }

  if (do_j) {
    const int dm_blocks = std::min(static_cast<int>((nao2 + 255) / 256), 1024);
    add_dm_pair_kernel<<<dm_blocks, 256, 0, stream>>>(nao, opts.symmetrize_dm, d_dm_alpha,
                                                      d_dm_beta, cache.d_dm_work.data());
    check_cuda_jk(cudaGetLastError(), "add_dm_pair_kernel (UKS)");
    const double j_omega = opts.rsh_mode ? 0.0 : opts.omega;
    std::vector<CudaFactorDesc> empty_fac;
    std::vector<int> dummy_off;
    run_rijk_single_pass_cuda(cache, j_omega, /*pass_j=*/true, /*pass_k=*/false, 0.0, false,
                              opts.symmetrize_dm, k_gemm_mode, opts.ozaki_slices, empty_fac,
                              dummy_off, d_vj_tot, nullptr, stream);
  }
  JKOptions k_opts = opts;
  k_opts.compute_j = false;
  k_opts.compute_k = true;
  if (do_ka) {
    run_rijk_mo_cuda_device(cache, d_dm_alpha, d_c_occ_alpha, n_alpha, 1.0, nullptr, d_vk_alpha,
                            k_opts, raw_stream);
  }
  if (do_kb) {
    run_rijk_mo_cuda_device(cache, d_dm_beta, d_c_occ_beta, n_beta, 1.0, nullptr, d_vk_beta,
                            k_opts, raw_stream);
  }
}

void run_rijk_uks_mo_cuda(RIJKCudaCache& cache, const double* h_dm_alpha, const double* h_dm_beta,
                          const double* h_c_occ_alpha, int n_alpha, const double* h_c_occ_beta,
                          int n_beta, double* h_vj_tot, double* h_vk_alpha, double* h_vk_beta,
                          const JKOptions& opts) {
  const int nao = cache.nao;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const bool do_j = opts.compute_j && (h_vj_tot != nullptr);
  const bool do_ka = opts.compute_k && (h_vk_alpha != nullptr);
  const bool do_kb = opts.compute_k && (h_vk_beta != nullptr);
  if ((!do_j && !do_ka && !do_kb) || nao2 == 0 || cache.naux == 0) {
    if (do_j) std::fill(h_vj_tot, h_vj_tot + nao2, 0.0);
    if (do_ka) std::fill(h_vk_alpha, h_vk_alpha + nao2, 0.0);
    if (do_kb) std::fill(h_vk_beta, h_vk_beta + nao2, 0.0);
    return;
  }

  cache.d_da_stage.upload_async(h_dm_alpha, nao2, nullptr);
  cache.d_db_stage.upload_async(h_dm_beta, nao2, nullptr);
  const double* d_ca = nullptr;
  const double* d_cb = nullptr;
  if (h_c_occ_alpha != nullptr && n_alpha > 0) {
    cache.d_ca_stage.upload_async(h_c_occ_alpha, static_cast<std::size_t>(nao) * n_alpha, nullptr);
    d_ca = cache.d_ca_stage.data();
  }
  if (h_c_occ_beta != nullptr && n_beta > 0) {
    cache.d_cb_stage.upload_async(h_c_occ_beta, static_cast<std::size_t>(nao) * n_beta, nullptr);
    d_cb = cache.d_cb_stage.data();
  }
  if (do_j) cache.d_vj_stage.resize(nao2);
  if (do_ka) cache.d_ka_stage.resize(nao2);
  if (do_kb) cache.d_kb_stage.resize(nao2);

  run_rijk_uks_mo_cuda_device(cache, cache.d_da_stage.data(), cache.d_db_stage.data(), d_ca,
                              n_alpha, d_cb, n_beta, do_j ? cache.d_vj_stage.data() : nullptr,
                              do_ka ? cache.d_ka_stage.data() : nullptr,
                              do_kb ? cache.d_kb_stage.data() : nullptr, opts, nullptr);
  if (do_j) cache.d_vj_stage.download_async(h_vj_tot, nao2, nullptr);
  if (do_ka) cache.d_ka_stage.download_async(h_vk_alpha, nao2, nullptr);
  if (do_kb) cache.d_kb_stage.download_async(h_vk_beta, nao2, nullptr);
  check_cuda_jk(cudaStreamSynchronize(nullptr), "run_rijk_uks_mo_cuda sync");
}

void run_ri_jk_grad_cuda(RIJKCudaCache& cache, std::span<const Atom> atoms,
                         const double* h_dm_alpha, const double* h_c_occ_alpha, int n_alpha,
                         const double* h_dm_beta, const double* h_c_occ_beta, int n_beta,
                         bool unrestricted, double j_fraction, double k_sr, double k_lr,
                         double omega, std::vector<double>& grad_out) {
  grad_out.assign(atoms.size() * 3, 0.0);
  const int nao = cache.nao;
  const int naux = cache.naux;
  const int ncart_orb = cache.ncart_orb;
  const int ncart_aux = cache.ncart_aux;
  if (atoms.empty() || nao == 0 || naux == 0) return;

  release_rijk_workspaces(cache);
  cache.linalg.set_stream(nullptr);
  auto* cublas = static_cast<cublasHandle_t>(cache.linalg.cublas_raw());

  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const std::size_t naux2 = static_cast<std::size_t>(naux) * naux;
  const std::size_t ncart2 = static_cast<std::size_t>(ncart_orb) * ncart_orb;
  const std::size_t ncart_aux2 = static_cast<std::size_t>(ncart_aux) * ncart_aux;

  auto map_shell_atoms = [&](const Basis& b) {
    std::vector<int> out(b.nshell(), -1);
    for (std::size_t s = 0; s < b.nshell(); ++s) {
      const auto& sc = b.shell(s).center;
      for (std::size_t a = 0; a < atoms.size(); ++a) {
        const double dx = sc[0] - atoms[a].center[0];
        const double dy = sc[1] - atoms[a].center[1];
        const double dz = sc[2] - atoms[a].center[2];
        if (dx * dx + dy * dy + dz * dz < 1e-12) {
          out[s] = static_cast<int>(a);
          break;
        }
      }
    }
    return out;
  };
  const std::vector<int> orb_atom = map_shell_atoms(cache.basis);
  const std::vector<int> aux_atom = map_shell_atoms(cache.aux_basis);

  DevVecJK<int> d_orb_sh_off(cache.orb_shell_cart_off);
  DevVecJK<int> d_aux_sh_off(cache.aux_shell_cart_off);
  DevVecJK<int> d_orb_sh_atom(orb_atom);
  DevVecJK<int> d_aux_sh_atom(aux_atom);
  DevVecJK<double> d_grad(atoms.size() * 3);
  d_grad.zero();

  auto build_sph_to_cart = [](const Basis& b, const std::vector<int>& sh_off, int ncart) {
    const int n_sph_tot = b.nao();
    std::vector<double> C(static_cast<std::size_t>(n_sph_tot) * ncart, 0.0);
    for (int s = 0; s < static_cast<int>(b.nshell()); ++s) {
      const int l = b.shell(s).l;
      if (l < 0 || l > 4) {
        throw std::runtime_error("RI-JK gradient supports up to g shells (l <= 4)");
      }
      const int ns = std::min(n_sph(l), 9);
      const int nc = std::min(n_cart(l), 15);
      const int ao0 = b.ao_offset(s);
      const int c0 = sh_off[s];
      double unit[15] = {0.0};
      double col[9] = {0.0};
      for (int ic = 0; ic < nc; ++ic) {
        unit[ic] = 1.0;
        harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
        for (int m = 0; m < ns; ++m) {
          C[static_cast<std::size_t>(ao0 + m) * ncart + (c0 + ic)] = col[m];
        }
        unit[ic] = 0.0;
      }
    }
    return C;
  };

  std::vector<double> dm_tot(nao2, 0.0);
  for (std::size_t i = 0; i < nao2; ++i) {
    dm_tot[i] = h_dm_alpha[i] + (unrestricted ? h_dm_beta[i] : 0.0);
  }
  for (int u = 0; u < nao; ++u) {
    for (int v = u + 1; v < nao; ++v) {
      const double avg = 0.5 * (dm_tot[static_cast<std::size_t>(u) * nao + v] +
                                dm_tot[static_cast<std::size_t>(v) * nao + u]);
      dm_tot[static_cast<std::size_t>(u) * nao + v] = avg;
      dm_tot[static_cast<std::size_t>(v) * nao + u] = avg;
    }
  }
  DevVecJK<double> d_D_tot(dm_tot);
  DevVecJK<double> d_D_tot_cart(ncart2);
  DevVecJK<double> d_C_orb(build_sph_to_cart(cache.basis, cache.orb_shell_cart_off, ncart_orb));
  const double one = 1.0, zero = 0.0;
  {
    DevVecJK<double> d_tmp(static_cast<std::size_t>(nao) * ncart_orb);
    check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, ncart_orb, nao, nao, &one,
                                d_C_orb.data(), ncart_orb, d_D_tot.data(), nao, &zero,
                                d_tmp.data(), ncart_orb),
                    "cublasDgemm D_tot_cart step 1");
    check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, ncart_orb, ncart_orb, nao, &one,
                                d_tmp.data(), ncart_orb, d_C_orb.data(), ncart_orb, &zero,
                                d_D_tot_cart.data(), ncart_orb),
                    "cublasDgemm D_tot_cart step 2");
  }

  struct GradChannelCuda {
    DevVecJK<double> d_C_sph;
    DevVecJK<double> d_C_cart;
    std::vector<double> signs;
    bool has_neg_signs = false;
    int ncols = 0;
    double occ_weight = 1.0;
  };
  std::vector<GradChannelCuda> channels;

  const double k_lr_delta = (omega > 0.0) ? (k_lr - k_sr) : 0.0;
  const bool need_any_k = (k_sr != 0.0) || (omega > 0.0 && k_lr_delta != 0.0);
  if (need_any_k) {
    auto add_channel = [&](const double* h_dm_s, const double* h_c_occ_s, int n_occ_s,
                           double occ_w) {
      if (n_occ_s == 0) return;
      GradChannelCuda ch;
      if (h_c_occ_s != nullptr && n_occ_s > 0) {
        ch.ncols = n_occ_s;
        ch.occ_weight = occ_w;
        ch.d_C_sph.upload(h_c_occ_s, static_cast<std::size_t>(nao) * n_occ_s);
      } else {
        std::vector<double> d_sym(nao2);
        for (int u = 0; u < nao; ++u) {
          for (int v = 0; v < nao; ++v) {
            d_sym[static_cast<std::size_t>(u) * nao + v] =
                0.5 * (h_dm_s[static_cast<std::size_t>(u) * nao + v] +
                       h_dm_s[static_cast<std::size_t>(v) * nao + u]);
          }
        }
        DevVecJK<double> d_dm_in(d_sym);
        DevVecJK<double> d_w(static_cast<std::size_t>(nao));
        DevVecJK<double> d_U(nao2);
        syevd_device(cache.linalg, nao, d_dm_in.data(), d_w.data(), d_U.data(), nullptr);
        std::vector<double> h_w(static_cast<std::size_t>(nao));
        d_w.download(h_w.data(), static_cast<std::size_t>(nao));
        std::vector<int> cols;
        std::vector<double> scales;
        for (int k = 0; k < nao; ++k) {
          if (std::abs(h_w[k]) > 1e-14) {
            cols.push_back(k);
            scales.push_back(std::sqrt(std::abs(h_w[k])));
            const double sg = (h_w[k] >= 0.0) ? 1.0 : -1.0;
            ch.signs.push_back(sg);
            if (sg < 0.0) ch.has_neg_signs = true;
          }
        }
        ch.ncols = static_cast<int>(cols.size());
        if (ch.ncols == 0) return;
        ch.occ_weight = unrestricted ? 1.0 : 0.5;
        ch.d_C_sph.resize(static_cast<std::size_t>(nao) * ch.ncols);
        DevVecJK<int> d_cols(cols);
        DevVecJK<double> d_scales(scales);
        const std::size_t tot = static_cast<std::size_t>(nao) * ch.ncols;
        const int blocks = static_cast<int>(std::min<std::size_t>((tot + 255) / 256, 4096));
        extract_cols_from_u_scaled_kernel<<<blocks, 256>>>(
            nao, ch.ncols, 0, ch.ncols, d_cols.data(), d_scales.data(), d_U.data(),
            ch.d_C_sph.data());
        check_cuda_jk(cudaGetLastError(), "extract_cols_from_u_scaled_kernel (grad)");
      }
      ch.d_C_cart.resize(static_cast<std::size_t>(ncart_orb) * ch.ncols);
      check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, ch.ncols, ncart_orb, nao, &one,
                                  ch.d_C_sph.data(), ch.ncols, d_C_orb.data(), ncart_orb, &zero,
                                  ch.d_C_cart.data(), ch.ncols),
                      "cublasDgemm C_cart (grad)");
      channels.push_back(std::move(ch));
    };

    if (!unrestricted) {
      add_channel(h_dm_alpha, h_c_occ_alpha, n_alpha, 2.0);
    } else {
      add_channel(h_dm_alpha, h_c_occ_alpha, n_alpha, 1.0);
      add_channel(h_dm_beta, h_c_occ_beta, n_beta, 1.0);
    }
  }
  d_C_orb.reset();

  std::size_t free_vram = 0, total_vram = 0;
  check_cuda_jk(cudaMemGetInfo(&free_vram, &total_vram), "cudaMemGetInfo (RI-JK grad)");
  const std::size_t gamma_chunk_cap = std::min<std::size_t>(1024ULL << 20, free_vram / 4);
  const int max_cart_grad = std::clamp<int>(
      static_cast<int>(gamma_chunk_cap / std::max<std::size_t>(1, ncart2 * sizeof(double))), 16,
      64);
  const int max_sph_grad = std::min(32, max_cart_grad);
  const std::vector<RIAuxChunk> grad_chunks = build_ri_aux_chunks_cuda(
      cache.aux_basis, cache.aux_ish_lists, cache.aux_shell_cart_off, max_sph_grad, max_cart_grad);

  int max_bc_sph = 1, max_bc_cart = 1;
  for (const auto& c : grad_chunks) {
    max_bc_sph = std::max(max_bc_sph, c.q1 - c.q0);
    max_bc_cart = std::max(max_bc_cart, c.pc1 - c.pc0);
  }

  auto build_aux_chunk_sph_to_cart = [&](const RIAuxChunk& c) {
    const int bc_sph = c.q1 - c.q0;
    const int bc_cart = c.pc1 - c.pc0;
    std::vector<double> H(static_cast<std::size_t>(bc_cart) * bc_sph, 0.0);
    for (int s = c.s0; s < c.s1; ++s) {
      const int l = cache.aux_basis.shell(s).l;
      if (l < 0 || l > 4) {
        throw std::runtime_error("RI-JK gradient supports up to g aux shells (l <= 4)");
      }
      const int ns = std::min(n_sph(l), 9);
      const int nc = std::min(n_cart(l), 15);
      const int p_loc = cache.aux_basis.ao_offset(s) - c.q0;
      const int pc_loc = cache.aux_shell_cart_off[s] - c.pc0;
      double unit[15] = {0.0};
      double col[9] = {0.0};
      for (int ic = 0; ic < nc; ++ic) {
        unit[ic] = 1.0;
        harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
        for (int m = 0; m < ns; ++m) {
          H[static_cast<std::size_t>(pc_loc + ic) * bc_sph + (p_loc + m)] = col[m];
        }
        unit[ic] = 0.0;
      }
    }
    return H;
  };

  auto run_grad_pass = [&](double pass_omega, double j_frac, double k_frac) {
    const bool do_k = (k_frac != 0.0) && !channels.empty();
    if (j_frac == 0.0 && !do_k) return;

    DevVecJK<double> d_local_fac;
    const double* d_fac = nullptr;
    bool is_chol = true;
    if (pass_omega <= 0.0) {
      d_fac = cache.d_metric_0.data();
      is_chol = cache.chol_ok_0;
    } else if (cache.rsh_omega > 0.0 && std::abs(pass_omega - cache.rsh_omega) <= 1e-12) {
      d_fac = cache.d_metric_w.data();
      is_chol = cache.chol_ok_w;
    } else {
      factorize_cuda_ri_metric(naux, cache.dev_aux, pass_omega, cache.linalg, d_local_fac, is_chol);
      d_fac = d_local_fac.data();
    }

    auto solve_v_fac = [&](int ncols_v, double* d_Y) {
      if (naux == 0 || ncols_v == 0) return;
      if (is_chol) {
        check_cublas_jk(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                    CUBLAS_DIAG_NON_UNIT, ncols_v, naux, &one, d_fac, naux, d_Y,
                                    ncols_v),
                        "cublasDtrsm forward grad V solve");
        check_cublas_jk(cublasDtrsm(cublas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT, ncols_v, naux, &one, d_fac, naux, d_Y,
                                    ncols_v),
                        "cublasDtrsm backward grad V solve");
      } else {
        const int col_blk_v = std::clamp<int>(
            static_cast<int>((64ULL << 20) /
                             std::max<std::size_t>(1, static_cast<std::size_t>(naux) *
                                                          sizeof(double))),
            1, ncols_v);
        DevVecJK<double> d_tmp(static_cast<std::size_t>(naux) * col_blk_v);
        for (int c0 = 0; c0 < ncols_v; c0 += col_blk_v) {
          const int bw = std::min(col_blk_v, ncols_v - c0);
          check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bw, naux, naux, &one,
                                      d_Y + c0, ncols_v, d_fac, naux, &zero, d_tmp.data(), bw),
                          "cublasDgemm grad V^{-1/2} step 1");
          check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bw, naux, naux, &one,
                                      d_tmp.data(), bw, d_fac, naux, &zero, d_Y + c0, ncols_v),
                          "cublasDgemm grad V^{-1/2} step 2");
        }
      }
    };

    DevVecJK<double> d_rho_sph(j_frac != 0.0 ? static_cast<std::size_t>(naux) : 0);
    DevVecJK<double> d_Gamma_2c_sph(naux2);
    d_Gamma_2c_sph.zero();

    auto init_j_gamma_2c = [&]() {
      solve_v_fac(1, d_rho_sph.data());
      const int blocks = static_cast<int>(std::min<std::size_t>((naux2 + 255) / 256, 4096));
      outer_product_scaled_kernel<<<blocks, 256>>>(naux, -0.5 * j_frac, d_rho_sph.data(),
                                                   d_Gamma_2c_sph.data());
      check_cuda_jk(cudaGetLastError(), "outer_product_scaled_kernel");
    };

    if (!do_k) {
      // Pure-Coulomb gradient path (e.g., pure GGA PBE/BLYP): single Pass 1 + Pass 3
      {
        DevVecJK<double> d_I_chunk(static_cast<std::size_t>(max_bc_sph) * nao2);
        const int nao2_i = static_cast<int>(nao2);
        for (const auto& c : grad_chunks) {
          const int bc = c.q1 - c.q0;
          eval_cuda_ri_3c_chunk(cache, c, pass_omega, d_I_chunk.data(), nullptr);
          check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, nao2_i, bc, &one, d_I_chunk.data(),
                                      nao2_i, d_D_tot.data(), 1, &zero, d_rho_sph.data() + c.q0, 1),
                          "cublasDgemv grad t_P (J-only)");
        }
      }
      init_j_gamma_2c();

      {
        DevVecJK<double> d_Gamma_chunk(static_cast<std::size_t>(max_bc_cart) * ncart2);
        DevVecJK<double> d_H_aux(static_cast<std::size_t>(max_bc_cart) * max_bc_sph);
        DevVecJK<double> d_rho_cart_chunk(static_cast<std::size_t>(max_bc_cart));

        for (const auto& c : grad_chunks) {
          const int bc_sph = c.q1 - c.q0;
          const int bc_cart = c.pc1 - c.pc0;
          const std::vector<double> h_H = build_aux_chunk_sph_to_cart(c);
          check_cuda_jk(cudaMemcpy(d_H_aux.data(), h_H.data(), h_H.size() * sizeof(double),
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpy H2D H_aux_chunk (J-only)");
          check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, bc_sph, bc_cart, &one, d_H_aux.data(),
                                      bc_sph, d_rho_sph.data() + c.q0, 1, &zero,
                                      d_rho_cart_chunk.data(), 1),
                          "cublasDgemv rho_cart_chunk (J-only)");
          const std::size_t tot = static_cast<std::size_t>(bc_cart) * ncart2;
          const int blocks = static_cast<int>(std::min<std::size_t>((tot + 255) / 256, 4096));
          init_gamma_3c_coulomb_kernel<<<blocks, 256>>>(bc_cart, ncart2, j_frac,
                                                        d_rho_cart_chunk.data(),
                                                        d_D_tot_cart.data(), d_Gamma_chunk.data());
          check_cuda_jk(cudaGetLastError(), "init_gamma_3c_coulomb_kernel (J-only)");

          for (std::size_t io = 0; io < cache.dev_orb.size(); ++io) {
            const auto& orb = cache.dev_orb[io];
            if (orb.npair == 0) continue;
            const bool is_spd = (orb.la <= 2 && orb.lb <= 2);
            for (std::size_t lP = 0; lP < cache.dev_aux.size(); ++lP) {
              const int pb = c.l_beg[lP];
              const int pe = c.l_end[lP];
              if (pb >= pe) continue;
              const auto& aux = cache.dev_aux[lP];
              const long long total = static_cast<long long>(orb.npair) * (pe - pb);
              const int blk = static_cast<int>(std::min((total + 63LL) / 64LL, 8192LL));
              if (is_spd) {
                ri_3c_grad_kernel<2><<<blk, 64>>>(
                    orb.la, orb.lb, aux.la, orb.view(), aux.view(), pb, pe, c.pc0,
                    d_orb_sh_off.data(), d_aux_sh_off.data(), d_orb_sh_atom.data(),
                    d_aux_sh_atom.data(), ncart_orb, pass_omega, d_Gamma_chunk.data(),
                    d_grad.data());
              } else {
                ri_3c_grad_kernel<4><<<blk, 64>>>(
                    orb.la, orb.lb, aux.la, orb.view(), aux.view(), pb, pe, c.pc0,
                    d_orb_sh_off.data(), d_aux_sh_off.data(), d_orb_sh_atom.data(),
                    d_aux_sh_atom.data(), ncart_orb, pass_omega, d_Gamma_chunk.data(),
                    d_grad.data());
              }
              check_cuda_jk(cudaGetLastError(), "ri_3c_grad_kernel launch (J-only)");
            }
          }
        }
      }
    } else {
      bool did_j_pass1 = false;
      bool did_j_pass3 = false;

      for (std::size_t s = 0; s < channels.size(); ++s) {
        const int nc = channels[s].ncols;
        const double w_2c = 0.5 * channels[s].occ_weight * k_frac;
        const double w_3c = -channels[s].occ_weight * k_frac;

        std::size_t free_now = 0, total_now = 0;
        check_cuda_jk(cudaMemGetInfo(&free_now, &total_now), "cudaMemGetInfo grad channel");
        const std::size_t chunk_peak_bytes =
            std::max(static_cast<std::size_t>(max_bc_sph) * nao2,
                     static_cast<std::size_t>(max_bc_cart) * ncart2) *
                sizeof(double) +
            (512ULL << 20);
        const std::size_t gpu_M_budget =
            (free_now > chunk_peak_bytes) ? (((free_now - chunk_peak_bytes) * 75) / 100) : 0;
        const std::size_t per_i_bytes =
            static_cast<std::size_t>(naux) * nc * sizeof(double);
        const std::size_t per_i_work_bytes =
            (static_cast<std::size_t>(max_bc_sph) * nao +
             static_cast<std::size_t>(max_bc_cart) * nc +
             static_cast<std::size_t>(max_bc_cart) * ncart_orb) *
            sizeof(double);

        const bool host_stage_M = (gpu_M_budget < per_i_bytes + per_i_work_bytes);
        const int bi_cap =
            host_stage_M
                ? std::clamp<int>(
                      static_cast<int>((1536ULL << 20) / std::max<std::size_t>(1, per_i_bytes)),
                      1, nc)
                : std::clamp<int>(
                      static_cast<int>(gpu_M_budget /
                                       std::max<std::size_t>(1, per_i_bytes + per_i_work_bytes)),
                      1, nc);

        DevVecJK<double> d_sg;
        if (channels[s].has_neg_signs) {
          d_sg.upload(channels[s].signs.data(), channels[s].signs.size());
        }

        DevVecJK<double> d_M_batch(
            !host_stage_M ? static_cast<std::size_t>(naux) * bi_cap * nc : 0);
        std::vector<double> h_M_host_batch(
            host_stage_M ? static_cast<std::size_t>(naux) * bi_cap * nc : 0);

        for (int i0 = 0; i0 < nc; i0 += bi_cap) {
          const int bi = std::min(bi_cap, nc - i0);
          const int bnc = bi * nc;

          // Pass 1: Streamed 3c integrals -> M_batch (and d_rho_sph on first batch if j_frac != 0)
          {
            DevVecJK<double> d_I_chunk(static_cast<std::size_t>(max_bc_sph) * nao2);
            DevVecJK<double> d_T_half(static_cast<std::size_t>(max_bc_sph) * nao * bi);
            DevVecJK<double> d_M_tmp(
                host_stage_M ? static_cast<std::size_t>(max_bc_sph) * bnc : 0);

            const bool collect_j_now = (!did_j_pass1 && j_frac != 0.0);
            for (const auto& c : grad_chunks) {
              const int bc = c.q1 - c.q0;
              eval_cuda_ri_3c_chunk(cache, c, pass_omega, d_I_chunk.data(), nullptr);
              if (collect_j_now) {
                const int nao2_i = static_cast<int>(nao2);
                check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, nao2_i, bc, &one,
                                            d_I_chunk.data(), nao2_i, d_D_tot.data(), 1, &zero,
                                            d_rho_sph.data() + c.q0, 1),
                                "cublasDgemv grad t_P");
              }
              check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bi, bc * nao, nao,
                                          &one, channels[s].d_C_sph.data() + i0, nc,
                                          d_I_chunk.data(), nao, &zero, d_T_half.data(), bi),
                              "cublasDgemm grad T_half");
              double* d_M_dst =
                  host_stage_M ? d_M_tmp.data()
                               : (d_M_batch.data() + static_cast<std::size_t>(c.q0) * bnc);
              check_cublas_jk(
                  cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_T, nc, bi, nao, &one,
                                           channels[s].d_C_sph.data(), nc, 0LL, d_T_half.data(),
                                           bi, static_cast<long long>(nao) * bi, &zero, d_M_dst, nc,
                                           static_cast<long long>(bnc), bc),
                  "cublasDgemmStridedBatched grad M_I_MO");
              if (host_stage_M) {
                check_cuda_jk(
                    cudaMemcpy(h_M_host_batch.data() + static_cast<std::size_t>(c.q0) * bnc,
                               d_M_dst, static_cast<std::size_t>(bc) * bnc * sizeof(double),
                               cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H host_stage_M Pass 1");
              }
            }
            if (collect_j_now) {
              init_j_gamma_2c();
              did_j_pass1 = true;
            }
          }

          // Pass 2: Metric solve + accumulate Gamma_2c_sph (preserving sg[i]*sg[j] without squaring!)
          if (!host_stage_M) {
            solve_v_fac(bnc, d_M_batch.data());
            if (!channels[s].has_neg_signs) {
              check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, naux, naux, bnc, &w_2c,
                                          d_M_batch.data(), bnc, d_M_batch.data(), bnc, &one,
                                          d_Gamma_2c_sph.data(), naux),
                              "cublasDgemm Gamma_2c_sph");
            } else {
              const int col_blk = std::clamp<int>(
                  static_cast<int>((64ULL << 20) /
                                   std::max<std::size_t>(1, static_cast<std::size_t>(naux) *
                                                                sizeof(double))),
                  1, bnc);
              DevVecJK<double> d_M_blk(static_cast<std::size_t>(naux) * col_blk);
              for (int c0 = 0; c0 < bnc; c0 += col_blk) {
                const int bw = std::min(col_blk, bnc - c0);
                const std::size_t tot = static_cast<std::size_t>(naux) * bw;
                const int blocks =
                    static_cast<int>(std::min<std::size_t>((tot + 255) / 256, 4096));
                copy_apply_signs_col_blk_kernel<<<blocks, 256>>>(
                    naux, i0, nc, bnc, c0, c0, bw, d_sg.data(), d_M_batch.data(), d_M_blk.data());
                check_cuda_jk(cudaGetLastError(), "copy_apply_signs_col_blk_kernel");
                check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, naux, naux, bw,
                                            &w_2c, d_M_blk.data(), bw, d_M_batch.data() + c0, bnc,
                                            &one, d_Gamma_2c_sph.data(), naux),
                                "cublasDgemm Gamma_2c_sph (signed)");
              }
              const std::size_t tot = static_cast<std::size_t>(naux) * bnc;
              const int blocks = static_cast<int>(std::min<std::size_t>((tot + 255) / 256, 4096));
              apply_grad_channel_signs_batch_kernel<<<blocks, 256>>>(
                  naux, i0, bi, nc, d_sg.data(), d_M_batch.data());
              check_cuda_jk(cudaGetLastError(), "apply_grad_channel_signs_batch_kernel");
            }
          } else {
            const int col_blk = std::clamp<int>(
                static_cast<int>((64ULL << 20) /
                                 std::max<std::size_t>(1, static_cast<std::size_t>(naux) *
                                                              sizeof(double))),
                1, bnc);
            DevVecJK<double> d_M_col(static_cast<std::size_t>(naux) * col_blk);
            DevVecJK<double> d_M_col_signed(
                channels[s].has_neg_signs ? (static_cast<std::size_t>(naux) * col_blk) : 0);
            std::vector<double> h_col(static_cast<std::size_t>(naux) * col_blk);
            for (int c0 = 0; c0 < bnc; c0 += col_blk) {
              const int bw = std::min(col_blk, bnc - c0);
              for (int p = 0; p < naux; ++p) {
                const double* src =
                    h_M_host_batch.data() + static_cast<std::size_t>(p) * bnc + c0;
                std::copy(src, src + bw, h_col.data() + static_cast<std::size_t>(p) * bw);
              }
              check_cuda_jk(cudaMemcpy(d_M_col.data(), h_col.data(),
                                       static_cast<std::size_t>(naux) * bw * sizeof(double),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy H2D M_col");
              solve_v_fac(bw, d_M_col.data());
              if (!channels[s].has_neg_signs) {
                check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, naux, naux, bw,
                                            &w_2c, d_M_col.data(), bw, d_M_col.data(), bw, &one,
                                            d_Gamma_2c_sph.data(), naux),
                                "cublasDgemm Gamma_2c_sph col_blk");
                check_cuda_jk(cudaMemcpy(h_col.data(), d_M_col.data(),
                                         static_cast<std::size_t>(naux) * bw * sizeof(double),
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy D2H M_col");
              } else {
                const std::size_t tot = static_cast<std::size_t>(naux) * bw;
                const int blocks =
                    static_cast<int>(std::min<std::size_t>((tot + 255) / 256, 4096));
                copy_apply_signs_col_blk_kernel<<<blocks, 256>>>(
                    naux, i0, nc, bw, 0, c0, bw, d_sg.data(), d_M_col.data(),
                    d_M_col_signed.data());
                check_cuda_jk(cudaGetLastError(), "copy_apply_signs_col_blk_kernel (host_stage)");
                check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, naux, naux, bw,
                                            &w_2c, d_M_col_signed.data(), bw, d_M_col.data(), bw,
                                            &one, d_Gamma_2c_sph.data(), naux),
                                "cublasDgemm Gamma_2c_sph col_blk (signed)");
                check_cuda_jk(cudaMemcpy(h_col.data(), d_M_col_signed.data(),
                                         static_cast<std::size_t>(naux) * bw * sizeof(double),
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy D2H M_col (signed)");
              }
              for (int p = 0; p < naux; ++p) {
                double* dst = h_M_host_batch.data() + static_cast<std::size_t>(p) * bnc + c0;
                const double* src = h_col.data() + static_cast<std::size_t>(p) * bw;
                std::copy(src, src + bw, dst);
              }
            }
          }

          // Pass 3: Chunked 3-center Cartesian AO nuclear derivatives for this (s, i0, bi) batch
          {
            const bool apply_j_now = (!did_j_pass3 && j_frac != 0.0);
            DevVecJK<double> d_Gamma_chunk(static_cast<std::size_t>(max_bc_cart) * ncart2);
            DevVecJK<double> d_H_aux(static_cast<std::size_t>(max_bc_cart) * max_bc_sph);
            DevVecJK<double> d_rho_cart_chunk(
                apply_j_now ? static_cast<std::size_t>(max_bc_cart) : 0);
            DevVecJK<double> d_M_sph_chunk(
                host_stage_M ? static_cast<std::size_t>(max_bc_sph) * bnc : 0);
            DevVecJK<double> d_M_cart_chunk(static_cast<std::size_t>(max_bc_cart) * bnc);
            DevVecJK<double> d_Y_cart_chunk(
                static_cast<std::size_t>(max_bc_cart) * bi * ncart_orb);

            for (const auto& c : grad_chunks) {
              const int bc_sph = c.q1 - c.q0;
              const int bc_cart = c.pc1 - c.pc0;
              const std::vector<double> h_H = build_aux_chunk_sph_to_cart(c);
              check_cuda_jk(cudaMemcpy(d_H_aux.data(), h_H.data(), h_H.size() * sizeof(double),
                                       cudaMemcpyHostToDevice),
                            "cudaMemcpy H2D H_aux_chunk");

              if (apply_j_now) {
                check_cublas_jk(cublasDgemv(cublas, CUBLAS_OP_T, bc_sph, bc_cart, &one,
                                            d_H_aux.data(), bc_sph, d_rho_sph.data() + c.q0, 1,
                                            &zero, d_rho_cart_chunk.data(), 1),
                                "cublasDgemv rho_cart_chunk");
                const std::size_t tot = static_cast<std::size_t>(bc_cart) * ncart2;
                const int blocks =
                    static_cast<int>(std::min<std::size_t>((tot + 255) / 256, 4096));
                init_gamma_3c_coulomb_kernel<<<blocks, 256>>>(
                    bc_cart, ncart2, j_frac, d_rho_cart_chunk.data(), d_D_tot_cart.data(),
                    d_Gamma_chunk.data());
                check_cuda_jk(cudaGetLastError(), "init_gamma_3c_coulomb_kernel");
              } else {
                check_cuda_jk(cudaMemset(d_Gamma_chunk.data(), 0,
                                         static_cast<std::size_t>(bc_cart) * ncart2 *
                                             sizeof(double)),
                              "cudaMemset d_Gamma_chunk");
              }

              const double* d_M_sph_ptr = nullptr;
              if (!host_stage_M) {
                d_M_sph_ptr = d_M_batch.data() + static_cast<std::size_t>(c.q0) * bnc;
              } else {
                check_cuda_jk(
                    cudaMemcpy(d_M_sph_chunk.data(),
                               h_M_host_batch.data() + static_cast<std::size_t>(c.q0) * bnc,
                               static_cast<std::size_t>(bc_sph) * bnc * sizeof(double),
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy H2D M_sph_chunk Pass 3");
                d_M_sph_ptr = d_M_sph_chunk.data();
              }

              check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, bnc, bc_cart, bc_sph,
                                          &one, d_M_sph_ptr, bnc, d_H_aux.data(), bc_sph, &zero,
                                          d_M_cart_chunk.data(), bnc),
                              "cublasDgemm M_cart_chunk");
              check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, ncart_orb,
                                          bc_cart * bi, nc, &one, channels[s].d_C_cart.data(), nc,
                                          d_M_cart_chunk.data(), nc, &zero, d_Y_cart_chunk.data(),
                                          ncart_orb),
                              "cublasDgemm Y_cart_chunk");
              check_cublas_jk(
                  cublasDgemmStridedBatched(cublas, CUBLAS_OP_N, CUBLAS_OP_N, ncart_orb, ncart_orb,
                                           bi, &w_3c, d_Y_cart_chunk.data(), ncart_orb,
                                           static_cast<long long>(bi) * ncart_orb,
                                           channels[s].d_C_cart.data() + i0, nc, 0LL, &one,
                                           d_Gamma_chunk.data(), ncart_orb,
                                           static_cast<long long>(ncart2), bc_cart),
                  "cublasDgemmStridedBatched Gamma_3c");

              for (std::size_t io = 0; io < cache.dev_orb.size(); ++io) {
                const auto& orb = cache.dev_orb[io];
                if (orb.npair == 0) continue;
                const bool is_spd = (orb.la <= 2 && orb.lb <= 2);
                for (std::size_t lP = 0; lP < cache.dev_aux.size(); ++lP) {
                  const int pb = c.l_beg[lP];
                  const int pe = c.l_end[lP];
                  if (pb >= pe) continue;
                  const auto& aux = cache.dev_aux[lP];
                  const long long total = static_cast<long long>(orb.npair) * (pe - pb);
                  const int blocks = static_cast<int>(std::min((total + 63LL) / 64LL, 8192LL));
                  if (is_spd) {
                    ri_3c_grad_kernel<2><<<blocks, 64>>>(
                        orb.la, orb.lb, aux.la, orb.view(), aux.view(), pb, pe, c.pc0,
                        d_orb_sh_off.data(), d_aux_sh_off.data(), d_orb_sh_atom.data(),
                        d_aux_sh_atom.data(), ncart_orb, pass_omega, d_Gamma_chunk.data(),
                        d_grad.data());
                  } else {
                    ri_3c_grad_kernel<4><<<blocks, 64>>>(
                        orb.la, orb.lb, aux.la, orb.view(), aux.view(), pb, pe, c.pc0,
                        d_orb_sh_off.data(), d_aux_sh_off.data(), d_orb_sh_atom.data(),
                        d_aux_sh_atom.data(), ncart_orb, pass_omega, d_Gamma_chunk.data(),
                        d_grad.data());
                  }
                  check_cuda_jk(cudaGetLastError(), "ri_3c_grad_kernel launch");
                }
              }
            }
            if (apply_j_now) {
              did_j_pass3 = true;
            }
          }
        }
      }
    }

    // 2-center metric nuclear derivatives (once per pass, after all channel/i-batches)
    {
      DevVecJK<double> d_Gamma_2c_cart(ncart_aux2);
      {
        DevVecJK<double> d_C_aux(
            build_sph_to_cart(cache.aux_basis, cache.aux_shell_cart_off, ncart_aux));
        DevVecJK<double> d_tmp_2c(static_cast<std::size_t>(naux) * ncart_aux);
        check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N, ncart_aux, naux, naux, &one,
                                    d_C_aux.data(), ncart_aux, d_Gamma_2c_sph.data(), naux, &zero,
                                    d_tmp_2c.data(), ncart_aux),
                        "cublasDgemm Gamma_2c_cart step 1");
        check_cublas_jk(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T, ncart_aux, ncart_aux, naux,
                                    &one, d_tmp_2c.data(), ncart_aux, d_C_aux.data(), ncart_aux,
                                    &zero, d_Gamma_2c_cart.data(), ncart_aux),
                        "cublasDgemm Gamma_2c_cart step 2");
      }
      d_Gamma_2c_sph.reset();

      for (std::size_t ip = 0; ip < cache.dev_aux.size(); ++ip) {
        if (cache.dev_aux[ip].npair == 0) continue;
        for (std::size_t iq = 0; iq <= ip; ++iq) {
          if (cache.dev_aux[iq].npair == 0) continue;
          const long long n_pq =
              static_cast<long long>(cache.dev_aux[ip].npair) * cache.dev_aux[iq].npair;
          const int blocks = static_cast<int>(std::min((n_pq + 63LL) / 64LL, 4096LL));
          ri_2c_grad_kernel<<<blocks, 64>>>(
              cache.dev_aux[ip].la, cache.dev_aux[iq].la, ip == iq, cache.dev_aux[ip].view(),
              cache.dev_aux[iq].view(), d_aux_sh_off.data(), d_aux_sh_atom.data(), ncart_aux,
              pass_omega, d_Gamma_2c_cart.data(), d_grad.data());
          check_cuda_jk(cudaGetLastError(), "ri_2c_grad_kernel launch");
        }
      }
    }
  };

  run_grad_pass(0.0, j_fraction, k_sr);
  if (omega > 0.0 && k_lr_delta != 0.0) {
    run_grad_pass(omega, 0.0, k_lr_delta);
  }
  check_cuda_jk(cudaDeviceSynchronize(), "run_ri_jk_grad_cuda sync");
  d_grad.download(grad_out.data(), grad_out.size());
}

// ---------------------------------------------------------------------------
// Analytical 2-Electron Nuclear Gradient CUDA Driver
// ---------------------------------------------------------------------------

void run_jk_nuclear_grad_cuda(const std::vector<ShellPairData>& pair_lists,
                              const std::vector<GradQuartetTask>& tasks_sr,
                              const std::vector<GradQuartetTask>& tasks_lr,
                              const std::vector<int>& shell_cart_offset,
                              const std::vector<int>& shell_atom, int ncart_total, int natom,
                              const std::vector<double>& d_cart_host, double j_fraction,
                              double k_sr, double k_lr_delta, double omega,
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
  DevVecJK<int> d_counter(1);
  DevVecJK<int> d_sh_off(shell_cart_offset);
  DevVecJK<int> d_sh_atom(shell_atom);
  DevVecJK<double> d_dm_cart(d_cart_host);
  DevVecJK<double> d_grad(static_cast<std::size_t>(natom) * 3);
  d_grad.zero();

  constexpr int kBlock = 64;
  if (!tasks_sr.empty()) {
    DevVecJK<GradQuartetTask> d_tasks(tasks_sr);
    d_counter.zero();
    const int blocks = std::min(static_cast<int>((tasks_sr.size() + kBlock - 1) / kBlock), 2048);
    jk_nuclear_grad_dynamic_kernel<<<blocks, kBlock>>>(
        d_classes.data(), d_tasks.data(), static_cast<int>(tasks_sr.size()), d_counter.data(),
        d_sh_off.data(), d_sh_atom.data(), d_dm_cart.data(), ncart_total, j_fraction, k_sr, 0.0,
        d_grad.data());
    check_cuda_jk(cudaGetLastError(), "jk_nuclear_grad_dynamic_kernel (SR) launch");
  }
  if (!tasks_lr.empty() && omega > 0.0 && k_lr_delta != 0.0) {
    DevVecJK<GradQuartetTask> d_tasks_lr(tasks_lr);
    d_counter.zero();
    const int blocks = std::min(static_cast<int>((tasks_lr.size() + kBlock - 1) / kBlock), 2048);
    jk_nuclear_grad_dynamic_kernel<<<blocks, kBlock>>>(
        d_classes.data(), d_tasks_lr.data(), static_cast<int>(tasks_lr.size()), d_counter.data(),
        d_sh_off.data(), d_sh_atom.data(), d_dm_cart.data(), ncart_total, 0.0, k_lr_delta, omega,
        d_grad.data());
    check_cuda_jk(cudaGetLastError(), "jk_nuclear_grad_dynamic_kernel (LR) launch");
  }
  check_cuda_jk(cudaDeviceSynchronize(), "jk_nuclear_grad_dynamic_kernel sync");
  d_grad.download(grad_out.data(), grad_out.size());
}

void run_jk_nuclear_grad_uhf_cuda(
    const std::vector<ShellPairData>& pair_lists, const std::vector<GradQuartetTask>& tasks_sr,
    const std::vector<GradQuartetTask>& tasks_lr, const std::vector<int>& shell_cart_offset,
    const std::vector<int>& shell_atom, int ncart_total, int natom,
    const std::vector<double>& d_tot_cart_host, const std::vector<double>& d_alpha_cart_host,
    const std::vector<double>& d_beta_cart_host, double j_fraction, double k_sr, double k_lr_delta,
    double omega, std::vector<double>& grad_out) {
  std::vector<DevShellPairList> dev_lists;
  dev_lists.reserve(pair_lists.size());
  std::vector<DevPairClassMeta> metas;
  metas.reserve(pair_lists.size());
  for (const ShellPairData& pd : pair_lists) {
    dev_lists.emplace_back(pd);
    metas.push_back(DevPairClassMeta{pd.la, pd.lb, dev_lists.back().view()});
  }

  DevVecJK<DevPairClassMeta> d_classes(metas);
  DevVecJK<int> d_counter(1);
  DevVecJK<int> d_sh_off(shell_cart_offset);
  DevVecJK<int> d_sh_atom(shell_atom);
  DevVecJK<double> d_tot_cart(d_tot_cart_host);
  DevVecJK<double> d_a_cart(d_alpha_cart_host);
  DevVecJK<double> d_b_cart(d_beta_cart_host);
  DevVecJK<double> d_grad(static_cast<std::size_t>(natom) * 3);
  d_grad.zero();

  constexpr int kBlock = 64;
  if (!tasks_sr.empty()) {
    DevVecJK<GradQuartetTask> d_tasks(tasks_sr);
    d_counter.zero();
    const int blocks = std::min(static_cast<int>((tasks_sr.size() + kBlock - 1) / kBlock), 2048);
    jk_nuclear_grad_uhf_dynamic_kernel<<<blocks, kBlock>>>(
        d_classes.data(), d_tasks.data(), static_cast<int>(tasks_sr.size()), d_counter.data(),
        d_sh_off.data(), d_sh_atom.data(), d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(),
        ncart_total, j_fraction, k_sr, 0.0, d_grad.data());
    check_cuda_jk(cudaGetLastError(), "jk_nuclear_grad_uhf_dynamic_kernel (SR) launch");
  }
  if (!tasks_lr.empty() && omega > 0.0 && k_lr_delta != 0.0) {
    DevVecJK<GradQuartetTask> d_tasks_lr(tasks_lr);
    d_counter.zero();
    const int blocks = std::min(static_cast<int>((tasks_lr.size() + kBlock - 1) / kBlock), 2048);
    jk_nuclear_grad_uhf_dynamic_kernel<<<blocks, kBlock>>>(
        d_classes.data(), d_tasks_lr.data(), static_cast<int>(tasks_lr.size()), d_counter.data(),
        d_sh_off.data(), d_sh_atom.data(), d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(),
        ncart_total, 0.0, k_lr_delta, omega, d_grad.data());
    check_cuda_jk(cudaGetLastError(), "jk_nuclear_grad_uhf_dynamic_kernel (LR) launch");
  }
  check_cuda_jk(cudaDeviceSynchronize(), "jk_nuclear_grad_uhf_dynamic_kernel sync");
  d_grad.download(grad_out.data(), grad_out.size());
}

}  // namespace gpu4dft::detail
