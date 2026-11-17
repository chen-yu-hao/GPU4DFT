#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/xc_kernels.hpp"
#include "gpu4dft/response.hpp"

namespace gpu4dft::detail {
namespace {

inline void check_cuda_resp(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DevBufResp {
 public:
  DevBufResp() = default;
  explicit DevBufResp(std::size_t n) : n_(n) {
    if (n_ > 0) check_cuda_resp(cudaMalloc(&ptr_, n_ * sizeof(T)), "cudaMalloc (resp)");
  }
  DevBufResp(const T* host, std::size_t n) : DevBufResp(n) {
    if (n_ > 0 && host != nullptr) {
      check_cuda_resp(cudaMemcpy(ptr_, host, n_ * sizeof(T), cudaMemcpyHostToDevice),
                      "cudaMemcpy H2D (resp)");
    }
  }
  ~DevBufResp() {
    if (ptr_ != nullptr) cudaFree(ptr_);
  }

  DevBufResp(const DevBufResp&) = delete;
  DevBufResp& operator=(const DevBufResp&) = delete;

  DevBufResp(DevBufResp&& o) noexcept : ptr_(o.ptr_), n_(o.n_) {
    o.ptr_ = nullptr;
    o.n_ = 0;
  }
  DevBufResp& operator=(DevBufResp&& o) noexcept {
    if (this != &o) {
      if (ptr_ != nullptr) cudaFree(ptr_);
      ptr_ = o.ptr_;
      n_ = o.n_;
      o.ptr_ = nullptr;
      o.n_ = 0;
    }
    return *this;
  }

  T* data() const noexcept { return ptr_; }
  std::size_t size() const noexcept { return n_; }

 private:
  T* ptr_ = nullptr;
  std::size_t n_ = 0;
};

// ===========================================================================
// 1. CUDA AO Electric Dipole Integrals Kernel
// ===========================================================================

struct UploadedBasisDipole {
  DevBufResp<int> shell_l;
  DevBufResp<int> shell_ao;
  DevBufResp<int> prim_offset;
  DevBufResp<double> shell_center;
  DevBufResp<double> prim_exp;
  DevBufResp<double> prim_coeff;
  BasisDeviceView view;

  explicit UploadedBasisDipole(const BasisDeviceView& h)
      : shell_l(h.shell_l, static_cast<std::size_t>(h.nshell)),
        shell_ao(h.shell_ao, static_cast<std::size_t>(h.nshell)),
        prim_offset(h.prim_offset, static_cast<std::size_t>(h.nshell + 1)),
        shell_center(h.shell_center, static_cast<std::size_t>(h.nshell) * 3),
        prim_exp(h.prim_exp, static_cast<std::size_t>(h.prim_offset[h.nshell])),
        prim_coeff(h.prim_coeff, static_cast<std::size_t>(h.prim_offset[h.nshell])) {
    view = BasisDeviceView{
        h.nshell,
        h.nao,
        0,
        shell_l.data(),
        shell_ao.data(),
        nullptr,
        prim_offset.data(),
        shell_center.data(),
        prim_exp.data(),
        prim_coeff.data(),
        nullptr,
        nullptr,
    };
  }
};

__global__ void dipole_matrix_kernel(BasisDeviceView b, const int* __restrict__ pair_i,
                                     const int* __restrict__ pair_j, int npairs,
                                     double ox, double oy, double oz,
                                     double* __restrict__ mx, double* __restrict__ my,
                                     double* __restrict__ mz) {
  const int stride = static_cast<int>(gridDim.x * blockDim.x);
  for (int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); idx < npairs;
       idx += stride) {
    eval_dipole_pair(b, pair_i[idx], pair_j[idx], ox, oy, oz, mx, my, mz);
  }
}

// ===========================================================================
// 2. CUDA In-Core ERI Non-Symmetric JK Contraction Kernels
// ===========================================================================

// Permutes (u l | v s) -> eri_k[(u * nao + v) * nao^2 + (l * nao + s)] on GPU
// so K(dm)_{uv} = \sum_{ls} (ul|vs) dm_{ls} is a single dense GEMM!
__global__ void permute_eri_for_k_kernel(int nao, const double* __restrict__ eri_j,
                                         double* __restrict__ eri_k) {
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  const std::size_t total = nao2 * nao2;
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const std::size_t uv = idx / nao2;
  const std::size_t ls = idx % nao2;
  const std::size_t u = uv / static_cast<std::size_t>(nao);
  const std::size_t v = uv % static_cast<std::size_t>(nao);
  const std::size_t l = ls / static_cast<std::size_t>(nao);
  const std::size_t s = ls % static_cast<std::size_t>(nao);

  const std::size_t src = ((u * nao + l) * nao + v) * nao + s;
  eri_k[idx] = eri_j[src];
}

// ===========================================================================
// 3. CUDA Direct Non-Symmetric Shell-Pair Rys JK Kernels
// ===========================================================================

struct DevicePairListMeta {
  ShellPairView view{};
  int la = 0;
  int lb = 0;
};

struct DeviceShellPairList {
  int la = 0;
  int lb = 0;
  int npair = 0;
  DevBufResp<int> ish;
  DevBufResp<int> jsh;
  DevBufResp<int> ao_i;
  DevBufResp<int> ao_j;
  DevBufResp<double> Ax, Ay, Az;
  DevBufResp<double> Bx, By, Bz;
  DevBufResp<double> schwarz_bound;
  DevBufResp<int> prim_offset;
  DevBufResp<double> exponent;
  DevBufResp<double> Px, Py, Pz;
  DevBufResp<double> prefactor;
  DevBufResp<double> alpha_a, alpha_b;
  DevicePairListMeta meta{};

  explicit DeviceShellPairList(const ShellPairData& h)
      : la(h.la),
        lb(h.lb),
        npair(h.npair()),
        ish(h.ish.data(), h.ish.size()),
        jsh(h.jsh.data(), h.jsh.size()),
        ao_i(h.ao_i.data(), h.ao_i.size()),
        ao_j(h.ao_j.data(), h.ao_j.size()),
        Ax(h.Ax.data(), h.Ax.size()),
        Ay(h.Ay.data(), h.Ay.size()),
        Az(h.Az.data(), h.Az.size()),
        Bx(h.Bx.data(), h.Bx.size()),
        By(h.By.data(), h.By.size()),
        Bz(h.Bz.data(), h.Bz.size()),
        schwarz_bound(h.schwarz_bound.data(), h.schwarz_bound.size()),
        prim_offset(h.prim_offset.data(), h.prim_offset.size()),
        exponent(h.exponent.data(), h.exponent.size()),
        Px(h.Px.data(), h.Px.size()),
        Py(h.Py.data(), h.Py.size()),
        Pz(h.Pz.data(), h.Pz.size()),
        prefactor(h.prefactor.data(), h.prefactor.size()),
        alpha_a(h.alpha_a.data(), h.alpha_a.size()),
        alpha_b(h.alpha_b.data(), h.alpha_b.size()) {
    meta.la = la;
    meta.lb = lb;
    meta.view = ShellPairView{
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

__global__ void compute_nonsym_shell_max_kernel(int nshell, int nao,
                                                const int* __restrict__ shell_ao,
                                                const int* __restrict__ shell_l,
                                                const double* __restrict__ d_dm,
                                                double* __restrict__ d_shell_max) {
  const int total = nshell * nshell;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;

  const int i = idx / nshell;
  const int j = idx % nshell;
  const int u0 = shell_ao[i];
  const int v0 = shell_ao[j];
  const int nu = 2 * shell_l[i] + 1;
  const int nv = 2 * shell_l[j] + 1;

  double blk_max = 0.0;
  for (int mu = 0; mu < nu; ++mu) {
    for (int mv = 0; mv < nv; ++mv) {
      const double a1 = fabs(d_dm[(u0 + mu) * nao + (v0 + mv)]);
      const double a2 = fabs(d_dm[(v0 + mv) * nao + (u0 + mu)]);
      blk_max = fmax(blk_max, fmax(a1, a2));
    }
  }
  d_shell_max[idx] = blk_max;
}

__device__ inline double nonsym_quartet_dmax(const double* __restrict__ d_sh, int nsh, int i, int j,
                                             int k, int l, bool do_j, bool do_k) {
  double m = 0.0;
  if (do_j) {
    m = fmax(m, 4.0 * d_sh[i * nsh + j]);
    m = fmax(m, 4.0 * d_sh[k * nsh + l]);
  }
  if (do_k) {
    m = fmax(m, d_sh[i * nsh + k]);
    m = fmax(m, d_sh[i * nsh + l]);
    m = fmax(m, d_sh[j * nsh + k]);
    m = fmax(m, d_sh[j * nsh + l]);
  }
  return m;
}

template <int MaxL>
__global__ void direct_nonsym_jk_kernel(const DevicePairListMeta* __restrict__ pair_lists,
                                        const RespJKTaskTile* __restrict__ tasks, int num_tasks,
                                        const int* __restrict__ shell_cart_offset, int nshell,
                                        int ncart_tot, double schwarz_cutoff, bool do_j, bool do_k,
                                        const double* __restrict__ d_cart,
                                        const double* __restrict__ d_shell_max,
                                        int* __restrict__ d_queue_counter,
                                        double* __restrict__ j_cart_out,
                                        double* __restrict__ k_cart_out) {
  auto accum_j = [&](int r, int c, double val) {
    atomicAdd(&j_cart_out[r * ncart_tot + c], val);
  };
  auto accum_k = [&](int r, int c, double val) {
    atomicAdd(&k_cart_out[r * ncart_tot + c], val);
  };

  while (true) {
    const int tid = atomicAdd(d_queue_counter, 1);
    if (tid >= num_tasks) break;

    const RespJKTaskTile t = tasks[tid];
    const DevicePairListMeta bm = pair_lists[t.bp];
    const DevicePairListMeta km = pair_lists[t.kp];
    const ShellPairView bra = bm.view;
    const ShellPairView ket = km.view;
    const int P = t.P;
    const int i = bra.ish[P], j = bra.jsh[P];
    const bool same_ij = (i == j);
    const double q_ab = bra.schwarz_bound[P];
    const int off_a = shell_cart_offset[i], off_b = shell_cart_offset[j];

    for (int Q = t.Q_begin; Q < t.Q_end; ++Q) {
      const double q_cd = ket.schwarz_bound[Q];
      const int k = ket.ish[Q], l = ket.jsh[Q];
      const double dmax = nonsym_quartet_dmax(d_shell_max, nshell, i, j, k, l, do_j, do_k);
      if (q_ab * q_cd * dmax < schwarz_cutoff) continue;

      const bool same_kl = (k == l);
      const bool same_PQ = (t.bp == t.kp) && (P == Q);
      const int off_c = shell_cart_offset[k], off_d = shell_cart_offset[l];

      eval_nonsym_jk_quartet<MaxL>(bm.la, bm.lb, km.la, km.lb, bra, P, ket, Q, same_ij, same_kl,
                                   same_PQ, do_j, do_k, d_cart, ncart_tot, off_a, off_b, off_c,
                                   off_d, accum_j, accum_k);
    }
  }
}

// ===========================================================================
// 4. CUDA Second-Derivative XC Hessian (`F_xc`) Kernels
// ===========================================================================

struct FxcPointHessian {
  double wg = 0.0;
  double gx0 = 0.0;
  double gy0 = 0.0;
  double gz0 = 0.0;
  double vsigma = 0.0;
  double v2rho2 = 0.0;
  double v2rhosigma = 0.0;
  double v2sigma2 = 0.0;
  double v2rhotau = 0.0;
  double v2sigmatau = 0.0;
  double v2tau2 = 0.0;
};

__global__ void gather_sym_dm_resp_kernel(int na, int nao, const int* __restrict__ active_aos,
                                          const double* __restrict__ d_dm,
                                          double* __restrict__ d_act) {
  const int total = na * na;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;
  const int u = idx / na;
  const int v = idx % na;
  const int gu = active_aos[u];
  const int gv = active_aos[v];
  d_act[idx] = 0.5 * (d_dm[gu * nao + gv] + d_dm[gv * nao + gu]);
}

__global__ void fxc_point_weights_kernel(int na, int ng, int xc_family,
                                         const FxcPointHessian* __restrict__ pts,
                                         const double* __restrict__ phi,
                                         const double* __restrict__ t_mat,
                                         double* __restrict__ wv0_out,
                                         double* __restrict__ wvx_out,
                                         double* __restrict__ wvy_out,
                                         double* __restrict__ wvz_out,
                                         double* __restrict__ wvtau_out) {
  const int g = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (g >= ng) return;

  const std::size_t comp_stride = static_cast<std::size_t>(na) * ng;
  const double* phi0 = phi;
  const double* phix = (xc_family >= 1) ? (phi + 1 * comp_stride) : nullptr;
  const double* phiy = (xc_family >= 1) ? (phi + 2 * comp_stride) : nullptr;
  const double* phiz = (xc_family >= 1) ? (phi + 3 * comp_stride) : nullptr;
  const double* t0 = t_mat;

  double drho = 0.0;
  double dgx = 0.0, dgy = 0.0, dgz = 0.0;
  double dtau = 0.0;

  for (int u = 0; u < na; ++u) {
    const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
    const double t0_ug = t0[ug];
    drho += phi0[ug] * t0_ug;
    if (xc_family >= 1) {
      dgx += 2.0 * phix[ug] * t0_ug;
      dgy += 2.0 * phiy[ug] * t0_ug;
      dgz += 2.0 * phiz[ug] * t0_ug;
    }
    if (xc_family == 2) {
      dtau += 0.5 * (phix[ug] * t_mat[1 * comp_stride + ug] +
                     phiy[ug] * t_mat[2 * comp_stride + ug] +
                     phiz[ug] * t_mat[3 * comp_stride + ug]);
    }
  }

  const FxcPointHessian p = pts[g];
  const double wg = p.wg;
  const double dsigma =
      (xc_family >= 1) ? (2.0 * (p.gx0 * dgx + p.gy0 * dgy + p.gz0 * dgz)) : 0.0;

  const double dvrho = p.v2rho2 * drho + p.v2rhosigma * dsigma + p.v2rhotau * dtau;
  wv0_out[g] = 0.5 * wg * dvrho;

  if (xc_family >= 1) {
    const double dvsigma = p.v2rhosigma * drho + p.v2sigma2 * dsigma + p.v2sigmatau * dtau;
    wvx_out[g] = 2.0 * wg * (dvsigma * p.gx0 + p.vsigma * dgx);
    wvy_out[g] = 2.0 * wg * (dvsigma * p.gy0 + p.vsigma * dgy);
    wvz_out[g] = 2.0 * wg * (dvsigma * p.gz0 + p.vsigma * dgz);
  }
  if (xc_family == 2) {
    const double dvtau = p.v2rhotau * drho + p.v2sigmatau * dsigma + p.v2tau2 * dtau;
    wvtau_out[g] = 0.5 * wg * dvtau;
  }
}

__global__ void fxc_build_w_2d_kernel(int na, int ng, int xc_family,
                                      const double* __restrict__ phi,
                                      const double* __restrict__ wv0,
                                      const double* __restrict__ wvx,
                                      const double* __restrict__ wvy,
                                      const double* __restrict__ wvz,
                                      double* __restrict__ w_mat) {
  const int g = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int u = static_cast<int>(blockIdx.y);
  if (g >= ng || u >= na) return;

  const std::size_t comp_stride = static_cast<std::size_t>(na) * ng;
  const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
  double val = wv0[g] * phi[ug];
  if (xc_family >= 1) {
    val += wvx[g] * phi[1 * comp_stride + ug] +
           wvy[g] * phi[2 * comp_stride + ug] +
           wvz[g] * phi[3 * comp_stride + ug];
  }
  w_mat[ug] = val;
}

__global__ void fxc_scale_tau_2d_kernel(int na, int ng, const double* __restrict__ phib,
                                        const double* __restrict__ wvtau,
                                        double* __restrict__ w_mat) {
  const int g = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int u = static_cast<int>(blockIdx.y);
  if (g >= ng || u >= na) return;
  const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
  w_mat[ug] = 0.5 * wvtau[g] * phib[ug];
}

__global__ void scatter_sym_fxc_kernel(int na, int nao, const int* __restrict__ active_aos,
                                       const double* __restrict__ f_half,
                                       double* __restrict__ d_fxc_out) {
  const int total = na * na;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;
  const int u = idx / na;
  const int v = idx % na;
  const int gu = active_aos[u];
  const int gv = active_aos[v];
  atomicAdd(&d_fxc_out[gu * nao + gv], f_half[u * na + v] + f_half[v * na + u]);
}

// ===========================================================================
// 5. GPU Linear-Response Matrix / Vector Utility Kernels
// ===========================================================================

__global__ void form_sym_sum_transpose_kernel(int nao, const double* __restrict__ M,
                                              double* __restrict__ sym_out) {
  const int total = nao * nao;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;
  const int u = idx / nao;
  const int v = idx % nao;
  sym_out[idx] = M[u * nao + v] + M[v * nao + u];
}

__global__ void assemble_tda_ao_potential_kernel(int nao, const double* __restrict__ J,
                                                 const double* __restrict__ Fxc, double alpha_x,
                                                 const double* __restrict__ K,
                                                 double* __restrict__ V_out) {
  const int total = nao * nao;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;
  double val = 0.0;
  if (J != nullptr) val += 2.0 * J[idx];
  if (Fxc != nullptr) val += Fxc[idx];
  if (K != nullptr && alpha_x != 0.0) val -= alpha_x * K[idx];
  V_out[idx] = val;
}

__global__ void assemble_rpa_ao_potentials_kernel(int nao, const double* __restrict__ J,
                                                  const double* __restrict__ Fxc, double alpha_x,
                                                  const double* __restrict__ K,
                                                  double* __restrict__ Vp_out,
                                                  double* __restrict__ Vm_out) {
  const int total = nao * nao;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= total) return;
  const int u = idx / nao;
  const int v = idx % nao;
  const int tr = v * nao + u;

  double vp = 0.0;
  if (J != nullptr) vp += 4.0 * J[idx];
  if (Fxc != nullptr) vp += 2.0 * Fxc[idx];
  if (K != nullptr && alpha_x != 0.0) {
    vp -= alpha_x * (K[idx] + K[tr]);
  }
  Vp_out[idx] = vp;

  if (Vm_out != nullptr) {
    double vm = 0.0;
    if (K != nullptr && alpha_x != 0.0) {
      vm = -alpha_x * (K[idx] - K[tr]);
    }
    Vm_out[idx] = vm;
  }
}

__global__ void add_diag_eps_mo_kernel(int n_ov, const double* __restrict__ eps,
                                       const double* __restrict__ x, double beta,
                                       double* __restrict__ y) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n_ov) return;
  const double diag_term = eps[i] * x[i];
  y[i] = (beta == 0.0) ? diag_term : (diag_term + beta * y[i]);
}

__global__ void elemwise_mul_kernel(int n, const double* __restrict__ scale,
                                    const double* __restrict__ x, double* __restrict__ y) {
  const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  y[i] = scale[i] * x[i];
}

}  // namespace

// ===========================================================================
// Exported Detail Implementations
// ===========================================================================

void run_dipole_matrices_cuda(const BasisDeviceView& host_view, std::array<double, 3> origin,
                              std::array<std::vector<double>, 3>& out) {
  UploadedBasisDipole dev(host_view);
  const int nsh = host_view.nshell;
  std::vector<int> pi, pj;
  pi.reserve(static_cast<std::size_t>(nsh) * (nsh + 1) / 2);
  pj.reserve(pi.capacity());
  for (int i = 0; i < nsh; ++i) {
    for (int j = 0; j <= i; ++j) {
      pi.push_back(i);
      pj.push_back(j);
    }
  }
  const int npairs = static_cast<int>(pi.size());
  if (npairs == 0) return;

  const std::size_t n2 = out[0].size();
  DevBufResp<int> d_pi(pi.data(), pi.size());
  DevBufResp<int> d_pj(pj.data(), pj.size());
  DevBufResp<double> d_mx(n2), d_my(n2), d_mz(n2);
  check_cuda_resp(cudaMemset(d_mx.data(), 0, n2 * sizeof(double)), "cudaMemset mx");
  check_cuda_resp(cudaMemset(d_my.data(), 0, n2 * sizeof(double)), "cudaMemset my");
  check_cuda_resp(cudaMemset(d_mz.data(), 0, n2 * sizeof(double)), "cudaMemset mz");

  constexpr int kBlock = 64;
  const int blocks = std::min((npairs + kBlock - 1) / kBlock, 4096);
  dipole_matrix_kernel<<<blocks, kBlock>>>(dev.view, d_pi.data(), d_pj.data(), npairs, origin[0],
                                           origin[1], origin[2], d_mx.data(), d_my.data(),
                                           d_mz.data());
  check_cuda_resp(cudaGetLastError(), "dipole_matrix_kernel launch");
  check_cuda_resp(cudaDeviceSynchronize(), "dipole_matrix_kernel sync");

  check_cuda_resp(cudaMemcpy(out[0].data(), d_mx.data(), n2 * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy mx");
  check_cuda_resp(cudaMemcpy(out[1].data(), d_my.data(), n2 * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy my");
  check_cuda_resp(cudaMemcpy(out[2].data(), d_mz.data(), n2 * sizeof(double), cudaMemcpyDeviceToHost),
                  "cudaMemcpy mz");
}

struct CudaInCoreResponseJK {
  int nao = 0;
  bool has_k = false;
  GpuLinalgContext ctx;
  PooledBuffer<double> d_eri_j;
  PooledBuffer<double> d_eri_k;

  CudaInCoreResponseJK(int nao_, const double* h_eri, bool need_k_)
      : nao(nao_),
        has_k(need_k_),
        ctx(&default_device_pool()),
        d_eri_j(ctx.pool(), static_cast<std::size_t>(nao_) * nao_ * nao_ * nao_) {
    const std::size_t n4 = static_cast<std::size_t>(nao_) * nao_ * nao_ * nao_;
    copy_h2d_async(d_eri_j.data(), h_eri, n4 * sizeof(double));
    if (has_k && n4 > 0) {
      d_eri_k = PooledBuffer<double>(ctx.pool(), n4);
      constexpr int kBlock = 256;
      const unsigned int blocks = static_cast<unsigned int>((n4 + kBlock - 1) / kBlock);
      permute_eri_for_k_kernel<<<blocks, kBlock>>>(nao_, d_eri_j.data(), d_eri_k.data());
      check_cuda_resp(cudaGetLastError(), "permute_eri_for_k_kernel");
    }
    synchronize_stream(nullptr);
  }
};

std::shared_ptr<CudaInCoreResponseJK> create_cuda_incore_response_jk(
    int nao, const double* h_eri, bool need_k) {
  return std::make_shared<CudaInCoreResponseJK>(nao, h_eri, need_k);
}

void eval_cuda_incore_response_jk(CudaInCoreResponseJK& state, const double* d_dm,
                                  double* d_vj, double* d_vk, GemmMode mode,
                                  int ozaki_slices, StreamHandle stream) {
  const int nao2 = state.nao * state.nao;
  if (nao2 == 0) return;
  if (d_vj != nullptr) {
    gemm_device(state.ctx, nao2, 1, nao2, 1.0, state.d_eri_j.data(), nao2, false, d_dm, 1, false,
                0.0, d_vj, 1, mode, stream, ozaki_slices);
  }
  if (d_vk != nullptr && state.has_k) {
    gemm_device(state.ctx, nao2, 1, nao2, 1.0, state.d_eri_k.data(), nao2, false, d_dm, 1, false,
                0.0, d_vk, 1, mode, stream, ozaki_slices);
  }
}

struct CudaDirectResponseJK {
  int nao = 0;
  int nshell = 0;
  int ncart = 0;
  int max_l = 0;
  double schwarz_cutoff = 1e-12;
  GpuLinalgContext ctx;
  std::vector<DeviceShellPairList> dev_lists;
  DevBufResp<DevicePairListMeta> d_list_views;
  DevBufResp<RespJKTaskTile> d_tasks;
  int num_tasks = 0;
  DevBufResp<int> d_shell_ao;
  DevBufResp<int> d_shell_l;
  DevBufResp<int> d_shell_cart_off;
  DevBufResp<double> d_C_sph2cart;  // [nao * ncart]
  PooledBuffer<double> d_tmp_nao_ncart;
  PooledBuffer<double> d_dm_cart;
  PooledBuffer<double> d_j_cart;
  PooledBuffer<double> d_k_cart;
  PooledBuffer<double> d_shell_max;
  DevBufResp<int> d_queue_counter;

  CudaDirectResponseJK(const Basis& basis, double cutoff)
      : nao(basis.nao()),
        nshell(static_cast<int>(basis.nshell())),
        max_l(basis.max_l()),
        schwarz_cutoff(cutoff),
        ctx(&default_device_pool()),
        d_queue_counter(1) {
    std::vector<int> h_shell_ao(nshell), h_shell_l(nshell), h_shell_cart_off(nshell);
    ncart = 0;
    for (int s = 0; s < nshell; ++s) {
      h_shell_ao[s] = basis.ao_offset(s);
      h_shell_l[s] = basis.shell(s).l;
      h_shell_cart_off[s] = ncart;
      ncart += n_cart(h_shell_l[s]);
    }

    std::vector<double> h_C(static_cast<std::size_t>(nao) * ncart, 0.0);
    for (int s = 0; s < nshell; ++s) {
      const int l = h_shell_l[s];
      const int nc = n_cart(l);
      const int ns = n_sph(l);
      const int ao0 = h_shell_ao[s];
      const int c0 = h_shell_cart_off[s];
      double unit[15] = {0.0};
      double col[9] = {0.0};
      for (int ic = 0; ic < nc; ++ic) {
        unit[ic] = 1.0;
        harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
        for (int m = 0; m < ns; ++m) {
          h_C[static_cast<std::size_t>(ao0 + m) * ncart + (c0 + ic)] = col[m];
        }
        unit[ic] = 0.0;
      }
    }

    std::vector<ShellPairData> host_lists;
    const int max_l = basis.max_l();
    for (int la = 0; la <= max_l; ++la) {
      for (int lb = 0; lb <= la; ++lb) {
        ShellPairData pd = build_shell_pairs(basis, la, lb, 1e-14);
        if (pd.npair() > 0) host_lists.push_back(std::move(pd));
      }
    }

    constexpr int kTileKet = 8;
    std::vector<RespJKTaskTile> h_tasks;
    for (int bp = 0; bp < static_cast<int>(host_lists.size()); ++bp) {
      const ShellPairData& bra = host_lists[bp];
      for (int P = 0; P < bra.npair(); ++P) {
        const double q_ab = bra.schwarz_bound[P];
        if (q_ab < 1e-10) continue;
        for (int kp = 0; kp <= bp; ++kp) {
          const ShellPairData& ket = host_lists[kp];
          const int q_lim = (bp == kp) ? (P + 1) : ket.npair();
          for (int Q0 = 0; Q0 < q_lim; Q0 += kTileKet) {
            if (q_ab * ket.schwarz_bound[Q0] < schwarz_cutoff * 1e-3) break;
            const int Q1 = std::min(Q0 + kTileKet, q_lim);
            h_tasks.push_back(RespJKTaskTile{bp, P, kp, Q0, Q1});
          }
        }
      }
    }

    dev_lists.reserve(host_lists.size());
    std::vector<DevicePairListMeta> h_views;
    h_views.reserve(host_lists.size());
    for (const auto& pd : host_lists) {
      dev_lists.emplace_back(pd);
      h_views.push_back(dev_lists.back().meta);
    }

    d_list_views = DevBufResp<DevicePairListMeta>(h_views.data(), h_views.size());
    num_tasks = static_cast<int>(h_tasks.size());
    if (num_tasks > 0) {
      d_tasks = DevBufResp<RespJKTaskTile>(h_tasks.data(), h_tasks.size());
    }
    d_shell_ao = DevBufResp<int>(h_shell_ao.data(), h_shell_ao.size());
    d_shell_l = DevBufResp<int>(h_shell_l.data(), h_shell_l.size());
    d_shell_cart_off = DevBufResp<int>(h_shell_cart_off.data(), h_shell_cart_off.size());
    d_C_sph2cart = DevBufResp<double>(h_C.data(), h_C.size());

    d_tmp_nao_ncart = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(nao) * ncart);
    d_dm_cart = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(ncart) * ncart);
    d_j_cart = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(ncart) * ncart);
    d_k_cart = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(ncart) * ncart);
    d_shell_max = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(nshell) * nshell);
  }
};

std::shared_ptr<CudaDirectResponseJK> create_cuda_direct_response_jk(
    const Basis& basis, bool /*need_k*/, double schwarz_cutoff) {
  return std::make_shared<CudaDirectResponseJK>(basis, schwarz_cutoff);
}

void eval_cuda_direct_response_jk(CudaDirectResponseJK& s, const double* d_dm,
                                  double* d_vj, double* d_vk, StreamHandle stream) {
  const bool do_j = (d_vj != nullptr);
  const bool do_k = (d_vk != nullptr);
  if ((!do_j && !do_k) || s.nao == 0) return;

  const std::size_t nao2 = static_cast<std::size_t>(s.nao) * s.nao;
  const std::size_t nc2 = static_cast<std::size_t>(s.ncart) * s.ncart;
  if (s.num_tasks == 0) {
    if (do_j) memset_zero_async(d_vj, nao2 * sizeof(double), stream);
    if (do_k) memset_zero_async(d_vk, nao2 * sizeof(double), stream);
    return;
  }

  // 1. Non-symmetric D_cart = C^T * D_sph * C on GPU
  gemm_device(s.ctx, s.nao, s.ncart, s.nao, 1.0, d_dm, s.nao, false, s.d_C_sph2cart.data(), s.ncart,
              false, 0.0, s.d_tmp_nao_ncart.data(), s.ncart, GemmMode::FP64, stream);
  gemm_device(s.ctx, s.ncart, s.ncart, s.nao, 1.0, s.d_C_sph2cart.data(), s.ncart, true,
              s.d_tmp_nao_ncart.data(), s.ncart, false, 0.0, s.d_dm_cart.data(), s.ncart,
              GemmMode::FP64, stream);

  // 2. Per-shell max |D_sph| for Schwarz screening
  {
    const int total = s.nshell * s.nshell;
    const int blocks = (total + 127) / 128;
    compute_nonsym_shell_max_kernel<<<blocks, 128, 0, stream>>>(
        s.nshell, s.nao, s.d_shell_ao.data(), s.d_shell_l.data(), d_dm, s.d_shell_max.data());
  }

  // 3. Zero Cartesian accumulators and atomic queue counter
  if (do_j) memset_zero_async(s.d_j_cart.data(), nc2 * sizeof(double), stream);
  if (do_k) memset_zero_async(s.d_k_cart.data(), nc2 * sizeof(double), stream);
  memset_zero_async(s.d_queue_counter.data(), sizeof(int), stream);

  // 4. Launch tiered dynamic-queue Rys quartet kernel (zero local-memory spill for s/p/d!)
  const int blocks = std::min((s.num_tasks + 3) / 4, 2048);
  if (s.max_l <= 1) {
    direct_nonsym_jk_kernel<1><<<blocks, 64, 0, stream>>>(
        s.d_list_views.data(), s.d_tasks.data(), s.num_tasks, s.d_shell_cart_off.data(), s.nshell,
        s.ncart, s.schwarz_cutoff, do_j, do_k, s.d_dm_cart.data(), s.d_shell_max.data(),
        s.d_queue_counter.data(), s.d_j_cart.data(), s.d_k_cart.data());
  } else if (s.max_l <= 2) {
    direct_nonsym_jk_kernel<2><<<blocks, 64, 0, stream>>>(
        s.d_list_views.data(), s.d_tasks.data(), s.num_tasks, s.d_shell_cart_off.data(), s.nshell,
        s.ncart, s.schwarz_cutoff, do_j, do_k, s.d_dm_cart.data(), s.d_shell_max.data(),
        s.d_queue_counter.data(), s.d_j_cart.data(), s.d_k_cart.data());
  } else {
    direct_nonsym_jk_kernel<4><<<blocks, 64, 0, stream>>>(
        s.d_list_views.data(), s.d_tasks.data(), s.num_tasks, s.d_shell_cart_off.data(), s.nshell,
        s.ncart, s.schwarz_cutoff, do_j, do_k, s.d_dm_cart.data(), s.d_shell_max.data(),
        s.d_queue_counter.data(), s.d_j_cart.data(), s.d_k_cart.data());
  }
  check_cuda_resp(cudaGetLastError(), "direct_nonsym_jk_kernel");

  // 5. Transform Cartesian -> Spherical on GPU WITHOUT symmetrizing K!
  if (do_j) {
    gemm_device(s.ctx, s.nao, s.ncart, s.ncart, 1.0, s.d_C_sph2cart.data(), s.ncart, false,
                s.d_j_cart.data(), s.ncart, false, 0.0, s.d_tmp_nao_ncart.data(), s.ncart,
                GemmMode::FP64, stream);
    gemm_device(s.ctx, s.nao, s.nao, s.ncart, 1.0, s.d_tmp_nao_ncart.data(), s.ncart, false,
                s.d_C_sph2cart.data(), s.ncart, true, 0.0, d_vj, s.nao, GemmMode::FP64, stream);
    symmetrize_matrix_device(s.nao, d_vj, stream);
  }
  if (do_k) {
    gemm_device(s.ctx, s.nao, s.ncart, s.ncart, 1.0, s.d_C_sph2cart.data(), s.ncart, false,
                s.d_k_cart.data(), s.ncart, false, 0.0, s.d_tmp_nao_ncart.data(), s.ncart,
                GemmMode::FP64, stream);
    gemm_device(s.ctx, s.nao, s.nao, s.ncart, 1.0, s.d_tmp_nao_ncart.data(), s.ncart, false,
                s.d_C_sph2cart.data(), s.ncart, true, 0.0, d_vk, s.nao, GemmMode::FP64, stream);
  }
}

// ===========================================================================
// GPU Cached Second-Derivative XC Hessian Engine (`CudaFxcEngine`)
// ===========================================================================

struct CudaFxcBatch {
  int na = 0;
  int ng = 0;
  std::size_t ao_offset = 0;
  std::size_t phi_offset = 0;
  std::size_t pt_offset = 0;
};

struct CudaFxcEngine {
  int nao = 0;
  int xc_family = 0;
  int ncomp = 1;
  GpuLinalgContext ctx;
  std::vector<CudaFxcBatch> batches;
  DevBufResp<int> d_all_active_aos;
  DevBufResp<double> d_all_phi;
  DevBufResp<FxcPointHessian> d_all_pts;

  PooledBuffer<double> d_dm_act;
  PooledBuffer<double> d_t_mat;
  PooledBuffer<double> d_w_mat;
  PooledBuffer<double> d_f_half;
  PooledBuffer<double> d_wv0, d_wvx, d_wvy, d_wvz, d_wvtau;

  CudaFxcEngine(const FlatBasisData& flat, const MolecularGrid& grid, int xc_func_id,
                int family, const double* dm0_host, double rho_cutoff)
      : nao(flat.nao),
        xc_family(family),
        ncomp((family == 0) ? 1 : 4),
        ctx(&default_device_pool()) {
    const FlatBasisView bview = flat.view();
    const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id);
    const int deriv = (xc_family == 0) ? 0 : 1;
    const auto& raw_batches = grid.batches();
    const auto& coords = grid.coords();
    const auto& weights = grid.weights();

    // Merge tiny octree leaf batches (e.g. 256 pts) into high-occupancy GPU super-batches
    // (~4096 pts, or a single full-grid batch when dense Phi <= 256 MiB) to eliminate
    // thousands of tiny kernel/cuBLAS launch overheads per Davidson iteration!
    std::vector<SpatialBatch> merged_storage;
    const std::vector<SpatialBatch>* work_batches = &raw_batches;
    const std::size_t dense_bytes =
        static_cast<std::size_t>(ncomp) * nao * grid.npoints() * sizeof(double);
    if (grid.npoints() > 0 && dense_bytes <= (256ULL * 1024ULL * 1024ULL)) {
      SpatialBatch full;
      full.point_offset = 0;
      full.npoints = grid.npoints();
      full.active_shells.resize(static_cast<std::size_t>(bview.nshell));
      for (int s = 0; s < bview.nshell; ++s) full.active_shells[static_cast<std::size_t>(s)] = s;
      full.active_aos.resize(static_cast<std::size_t>(nao));
      for (int u = 0; u < nao; ++u) full.active_aos[static_cast<std::size_t>(u)] = u;
      merged_storage.push_back(std::move(full));
      work_batches = &merged_storage;
    } else if (!raw_batches.empty()) {
      constexpr std::size_t kTargetSuperPts = 4096;
      std::size_t idx = 0;
      while (idx < raw_batches.size()) {
        if (raw_batches[idx].npoints == 0) {
          ++idx;
          continue;
        }
        SpatialBatch sb;
        sb.point_offset = raw_batches[idx].point_offset;
        sb.npoints = 0;
        std::vector<int> sh_mask(static_cast<std::size_t>(bview.nshell), 0);
        while (idx < raw_batches.size()) {
          const SpatialBatch& cur = raw_batches[idx];
          if (cur.npoints > 0 &&
              cur.point_offset == sb.point_offset + sb.npoints &&
              (sb.npoints == 0 || sb.npoints + cur.npoints <= kTargetSuperPts)) {
            sb.npoints += cur.npoints;
            for (int sh : cur.active_shells) sh_mask[static_cast<std::size_t>(sh)] = 1;
            ++idx;
          } else {
            break;
          }
        }
        for (int s = 0; s < bview.nshell; ++s) {
          if (sh_mask[static_cast<std::size_t>(s)] != 0) {
            sb.active_shells.push_back(s);
            const int ao0 = bview.shell_ao_offset[s];
            const int nm = 2 * bview.shell_l[s] + 1;
            for (int m = 0; m < nm; ++m) sb.active_aos.push_back(ao0 + m);
          }
        }
        merged_storage.push_back(std::move(sb));
      }
      work_batches = &merged_storage;
    }
    const auto& gbatches = *work_batches;

    std::vector<int> h_all_aos;
    std::vector<double> h_all_phi;
    std::vector<FxcPointHessian> h_all_pts;

    int max_na = 1;
    int max_ng = 1;

    for (const SpatialBatch& b : gbatches) {
      const int ng = static_cast<int>(b.npoints);
      const int na = static_cast<int>(b.active_aos.size());
      if (ng == 0 || na == 0) continue;
      max_na = std::max(max_na, na);
      max_ng = std::max(max_ng, ng);

      CudaFxcBatch cb;
      cb.na = na;
      cb.ng = ng;
      cb.ao_offset = h_all_aos.size();
      cb.phi_offset = h_all_phi.size();
      cb.pt_offset = h_all_pts.size();
      batches.push_back(cb);

      h_all_aos.insert(h_all_aos.end(), b.active_aos.begin(), b.active_aos.end());

      const std::size_t comp_stride = static_cast<std::size_t>(na) * ng;
      std::vector<double> phi_loc(static_cast<std::size_t>(ncomp) * comp_stride, 0.0);
      std::size_t cursor = 0;
      for (int is : b.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (int g = 0; g < ng; ++g) {
          const std::size_t gp = b.point_offset + static_cast<std::size_t>(g);
          eval_shell_at_point(bview, is, coords[3 * gp + 0], coords[3 * gp + 1],
                              coords[3 * gp + 2], deriv, phi_loc.data() + cursor * ng + g,
                              comp_stride, static_cast<std::size_t>(ng));
        }
        cursor += static_cast<std::size_t>(nm);
      }

      // Evaluate ground-state density & 2nd XC derivatives for this batch
      std::vector<double> d0_act(static_cast<std::size_t>(na) * na, 0.0);
      for (int u = 0; u < na; ++u) {
        const int gu = b.active_aos[static_cast<std::size_t>(u)];
        for (int v = 0; v < na; ++v) {
          const int gv = b.active_aos[static_cast<std::size_t>(v)];
          d0_act[static_cast<std::size_t>(u) * na + v] =
              0.5 * (dm0_host[gu * nao + gv] + dm0_host[gv * nao + gu]);
        }
      }

      const int n_tcomp = (xc_family == 2) ? 4 : 1;
      std::vector<double> t0_mat(static_cast<std::size_t>(n_tcomp) * comp_stride, 0.0);
      for (int c = 0; c < n_tcomp; ++c) {
        const double* phic = phi_loc.data() + static_cast<std::size_t>(c) * comp_stride;
        double* tc = t0_mat.data() + static_cast<std::size_t>(c) * comp_stride;
        for (int u = 0; u < na; ++u) {
          for (int k = 0; k < na; ++k) {
            const double d_uk = d0_act[static_cast<std::size_t>(u) * na + k];
            if (d_uk == 0.0) continue;
            for (int g = 0; g < ng; ++g) {
              tc[static_cast<std::size_t>(u) * ng + g] +=
                  d_uk * phic[static_cast<std::size_t>(k) * ng + g];
            }
          }
        }
      }

      const double* phi0 = phi_loc.data();
      const double* phix = (xc_family >= 1) ? (phi_loc.data() + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family >= 1) ? (phi_loc.data() + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family >= 1) ? (phi_loc.data() + 3 * comp_stride) : nullptr;

      for (int g = 0; g < ng; ++g) {
        double rho0 = 0.0, gx0 = 0.0, gy0 = 0.0, gz0 = 0.0, tau0 = 0.0;
        for (int u = 0; u < na; ++u) {
          const std::size_t ug = static_cast<std::size_t>(u) * ng + g;
          const double t_ug = t0_mat[ug];
          rho0 += phi0[ug] * t_ug;
          if (xc_family >= 1) {
            gx0 += 2.0 * phix[ug] * t_ug;
            gy0 += 2.0 * phiy[ug] * t_ug;
            gz0 += 2.0 * phiz[ug] * t_ug;
          }
          if (xc_family == 2) {
            tau0 += 0.5 * (phix[ug] * t0_mat[1 * comp_stride + ug] +
                           phiy[ug] * t0_mat[2 * comp_stride + ug] +
                           phiz[ug] * t0_mat[3 * comp_stride + ug]);
          }
        }
        const double sig0 = (xc_family >= 1) ? (gx0 * gx0 + gy0 * gy0 + gz0 * gz0) : 0.0;
        xc_gen::XcPointOut pt_out;
        xc_gen::eval_xc_point(fid, rho0, sig0, tau0, 2, pt_out, rho_cutoff);

        FxcPointHessian fh;
        fh.wg = weights[b.point_offset + static_cast<std::size_t>(g)];
        fh.gx0 = gx0;
        fh.gy0 = gy0;
        fh.gz0 = gz0;
        fh.vsigma = pt_out.vsigma;
        fh.v2rho2 = pt_out.v2rho2;
        fh.v2rhosigma = pt_out.v2rhosigma;
        fh.v2sigma2 = pt_out.v2sigma2;
        fh.v2rhotau = pt_out.v2rhotau;
        fh.v2sigmatau = pt_out.v2sigmatau;
        fh.v2tau2 = pt_out.v2tau2;
        h_all_pts.push_back(fh);
      }

      h_all_phi.insert(h_all_phi.end(), phi_loc.begin(), phi_loc.end());
    }

    if (!batches.empty()) {
      d_all_active_aos = DevBufResp<int>(h_all_aos.data(), h_all_aos.size());
      d_all_phi = DevBufResp<double>(h_all_phi.data(), h_all_phi.size());
      d_all_pts = DevBufResp<FxcPointHessian>(h_all_pts.data(), h_all_pts.size());

      const std::size_t max_comp = static_cast<std::size_t>(max_na) * max_ng;
      const std::size_t n_tcomp = (xc_family == 2) ? 4 : 1;
      d_dm_act = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_na) * max_na);
      d_t_mat = PooledBuffer<double>(ctx.pool(), n_tcomp * max_comp);
      d_w_mat = PooledBuffer<double>(ctx.pool(), max_comp);
      d_f_half = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_na) * max_na);
      d_wv0 = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_ng));
      d_wvx = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_ng));
      d_wvy = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_ng));
      d_wvz = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_ng));
      d_wvtau = PooledBuffer<double>(ctx.pool(), static_cast<std::size_t>(max_ng));
    }
  }
};

std::shared_ptr<CudaFxcEngine> create_cuda_fxc_engine(
    const FlatBasisData& flat_basis, const MolecularGrid& grid, int xc_func_id, int xc_family,
    const double* dm0_host, double rho_cutoff) {
  return std::make_shared<CudaFxcEngine>(flat_basis, grid, xc_func_id, xc_family, dm0_host,
                                         rho_cutoff);
}

void eval_cuda_fxc_device(CudaFxcEngine& eng, const double* d_sym_dm, double* d_fxc_out,
                          StreamHandle stream) {
  const std::size_t nao2 = static_cast<std::size_t>(eng.nao) * eng.nao;
  if (nao2 == 0) return;
  memset_zero_async(d_fxc_out, nao2 * sizeof(double), stream);
  if (eng.batches.empty()) return;

  for (const CudaFxcBatch& b : eng.batches) {
    const int na = b.na;
    const int ng = b.ng;
    const std::size_t comp_stride = static_cast<std::size_t>(na) * ng;
    const int* d_aos = eng.d_all_active_aos.data() + b.ao_offset;
    const double* d_phi = eng.d_all_phi.data() + b.phi_offset;
    const FxcPointHessian* d_pts = eng.d_all_pts.data() + b.pt_offset;

    // 1. Gather active symmetric perturbation density block
    {
      const int n2 = na * na;
      gather_sym_dm_resp_kernel<<<(n2 + 127) / 128, 128, 0, stream>>>(
          na, eng.nao, d_aos, d_sym_dm, eng.d_dm_act.data());
    }

    // 2. Half-transform T = d_dm_act * Phi
    gemm_device(eng.ctx, na, ng, na, 1.0, eng.d_dm_act.data(), na, false, d_phi, ng, false, 0.0,
                eng.d_t_mat.data(), ng, GemmMode::FP64, stream);
    if (eng.xc_family == 2) {
      for (int c = 1; c <= 3; ++c) {
        gemm_device(eng.ctx, na, ng, na, 1.0, eng.d_dm_act.data(), na, false,
                    d_phi + static_cast<std::size_t>(c) * comp_stride, ng, false, 0.0,
                    eng.d_t_mat.data() + static_cast<std::size_t>(c) * comp_stride, ng,
                    GemmMode::FP64, stream);
      }
    }

    // 3. Evaluate per-point 2nd-derivative XC response weights
    fxc_point_weights_kernel<<<(ng + 127) / 128, 128, 0, stream>>>(
        na, ng, eng.xc_family, d_pts, d_phi, eng.d_t_mat.data(), eng.d_wv0.data(),
        eng.d_wvx.data(), eng.d_wvy.data(), eng.d_wvz.data(), eng.d_wvtau.data());

    // 4. Coalesced 2D (point_block, ao) build of W[na x ng]
    {
      const dim3 grid_2d(static_cast<unsigned int>((ng + 127) / 128),
                         static_cast<unsigned int>(na));
      fxc_build_w_2d_kernel<<<grid_2d, 128, 0, stream>>>(
          na, ng, eng.xc_family, d_phi, eng.d_wv0.data(), eng.d_wvx.data(), eng.d_wvy.data(),
          eng.d_wvz.data(), eng.d_w_mat.data());
    }

    // 5. F_half = Phi0 * W^T (+ MGGA tau contribution)
    gemm_device(eng.ctx, na, na, ng, 1.0, d_phi, ng, false, eng.d_w_mat.data(), ng, true, 0.0,
                eng.d_f_half.data(), na, GemmMode::FP64, stream);
    if (eng.xc_family == 2) {
      const dim3 grid_2d(static_cast<unsigned int>((ng + 127) / 128),
                         static_cast<unsigned int>(na));
      for (int c = 1; c <= 3; ++c) {
        const double* d_phib = d_phi + static_cast<std::size_t>(c) * comp_stride;
        fxc_scale_tau_2d_kernel<<<grid_2d, 128, 0, stream>>>(
            na, ng, d_phib, eng.d_wvtau.data(), eng.d_w_mat.data());
        gemm_device(eng.ctx, na, na, ng, 1.0, d_phib, ng, false, eng.d_w_mat.data(), ng, true, 1.0,
                    eng.d_f_half.data(), na, GemmMode::FP64, stream);
      }
    }

    // 6. Symmetrized scatter into d_fxc_out
    {
      const int n2 = na * na;
      scatter_sym_fxc_kernel<<<(n2 + 127) / 128, 128, 0, stream>>>(
          na, eng.nao, d_aos, eng.d_f_half.data(), d_fxc_out);
    }
  }
}

void form_sym_sum_transpose_device(int nao, const double* d_M, double* d_sym, StreamHandle stream) {
  const int n2 = nao * nao;
  if (n2 <= 0) return;
  form_sym_sum_transpose_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(nao, d_M, d_sym);
}

void assemble_tda_ao_potential_device(int nao, const double* d_J, const double* d_Fxc,
                                      double alpha_x, const double* d_K, double* d_V_out,
                                      StreamHandle stream) {
  const int n2 = nao * nao;
  if (n2 <= 0) return;
  assemble_tda_ao_potential_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(nao, d_J, d_Fxc, alpha_x,
                                                                         d_K, d_V_out);
}

void assemble_rpa_ao_potentials_device(int nao, const double* d_J, const double* d_Fxc,
                                       double alpha_x, const double* d_K, double* d_Vp_out,
                                       double* d_Vm_out, StreamHandle stream) {
  const int n2 = nao * nao;
  if (n2 <= 0) return;
  assemble_rpa_ao_potentials_kernel<<<(n2 + 255) / 256, 256, 0, stream>>>(
      nao, d_J, d_Fxc, alpha_x, d_K, d_Vp_out, d_Vm_out);
}

void add_diag_eps_mo_device(int n_ov, const double* d_eps, const double* d_x, double beta,
                            double* d_y, StreamHandle stream) {
  if (n_ov <= 0) return;
  add_diag_eps_mo_kernel<<<(n_ov + 255) / 256, 256, 0, stream>>>(n_ov, d_eps, d_x, beta, d_y);
}

void elemwise_mul_device(int n, const double* d_scale, const double* d_x, double* d_y,
                         StreamHandle stream) {
  if (n <= 0) return;
  elemwise_mul_kernel<<<(n + 255) / 256, 256, 0, stream>>>(n, d_scale, d_x, d_y);
}

}  // namespace gpu4dft::detail
