#include "gpu4dft/xc.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/generated/xc_kernels.hpp"

namespace gpu4dft {

namespace detail {
#if defined(GPU4DFT_HAVE_CUDA)
int xc_cuda_device_count();
void eval_xc_points_cuda(int func_id,
                         std::size_t n,
                         const double* rho,
                         const double* sigma,
                         const double* tau,
                         int deriv,
                         double rho_cutoff,
                         XcPointwiseResult& out);

void eval_ao_grid_cuda(const FlatBasisData& flat_basis,
                       const double* coords_xyz,
                       std::size_t npts,
                       int deriv,
                       double* out_ao);
#endif
}  // namespace detail

struct XcEngine::HostFxcState {
  bool ready = false;
  std::vector<double> v2rho2;
  std::vector<double> g0x;
  std::vector<double> g0y;
  std::vector<double> g0z;
  std::vector<double> vsigma;
  std::vector<double> v2rhosigma;
  std::vector<double> v2sigma2;
  std::vector<double> v2rhotau;
  std::vector<double> v2sigmatau;
  std::vector<double> v2tau2;
  bool ao_cached = false;
  std::vector<std::size_t> batch_phi_offset;
  std::vector<double> phi_cache;
};

#if defined(GPU4DFT_HAVE_CUDA)
struct XcEngine::CudaState {
  void* impl = nullptr;
};

namespace detail {
void* xc_cuda_engine_create(const FlatBasisData& flat_basis,
                            const MolecularGrid& grid,
                            int func_id,
                            int family,
                            const XcOptions& opts);
void xc_cuda_engine_destroy(void* impl) noexcept;
double xc_cuda_eval_vxc_device(void* impl,
                               const double* d_dm,
                               double* d_vxc,
                               double* nelec_out,
                               void* cuda_stream);
double xc_cuda_eval_vxc_host(void* impl,
                             const double* h_dm,
                             double* h_vxc,
                             double* nelec_out);
void xc_cuda_set_reference_density_device(void* impl,
                                          const double* d_dm,
                                          void* cuda_stream);
void xc_cuda_set_reference_density_host(void* impl,
                                        const double* h_dm);
bool xc_cuda_has_reference_density(const void* impl) noexcept;
void xc_cuda_eval_fxc_device(void* impl,
                             const double* d_delta_dm,
                             double* d_fxc_out,
                             void* cuda_stream);
void xc_cuda_eval_fxc_host(void* impl,
                           const double* h_delta_dm,
                           double* h_fxc_out);
std::vector<double> xc_cuda_nuclear_grad(void* impl,
                                         const int* ao_to_atom,
                                         int natom,
                                         const double* h_dm);
}  // namespace detail
#else
struct XcEngine::CudaState {};
#endif

double xc_exact_exchange_fraction(std::string_view xc_name) {
  const auto id = xc_gen::parse_xc_functional(xc_name);
  return xc_gen::xc_exact_exchange_fraction_by_id(id);
}

XcPointwiseResult eval_xc_points(std::string_view xc_name,
                                 std::span<const double> rho,
                                 std::span<const double> sigma,
                                 std::span<const double> tau,
                                 int deriv,
                                 Backend backend,
                                 double rho_cutoff) {
  if (deriv < 0 || deriv > 2) {
    throw std::invalid_argument("eval_xc_points: deriv must be in {0, 1, 2}");
  }
  const std::size_t n = rho.size();
  if (!sigma.empty() && sigma.size() != n) {
    throw std::invalid_argument("eval_xc_points: sigma size mismatch");
  }
  if (!tau.empty() && tau.size() != n) {
    throw std::invalid_argument("eval_xc_points: tau size mismatch");
  }

  const auto id = xc_gen::parse_xc_functional(xc_name);
  XcPointwiseResult res;
  res.zk.assign(n, 0.0);
  if (deriv >= 1) {
    res.vrho.assign(n, 0.0);
    res.vsigma.assign(n, 0.0);
    res.vtau.assign(n, 0.0);
  }
  if (deriv >= 2) {
    res.v2rho2.assign(n, 0.0);
    res.v2rhosigma.assign(n, 0.0);
    res.v2sigma2.assign(n, 0.0);
    res.v2rhotau.assign(n, 0.0);
    res.v2sigmatau.assign(n, 0.0);
    res.v2tau2.assign(n, 0.0);
  }
  if (n == 0) return res;

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::xc_cuda_device_count() == 0) {
      throw std::runtime_error("eval_xc_points: no CUDA device available");
    }
    std::vector<double> sig_buf;
    std::vector<double> tau_buf;
    const double* sig_ptr = sigma.data();
    const double* tau_ptr = tau.data();
    if (sigma.empty()) {
      sig_buf.assign(n, 0.0);
      sig_ptr = sig_buf.data();
    }
    if (tau.empty()) {
      tau_buf.assign(n, 0.0);
      tau_ptr = tau_buf.data();
    }
    detail::eval_xc_points_cuda(static_cast<int>(id), n, rho.data(), sig_ptr, tau_ptr,
                                deriv, rho_cutoff, res);
    return res;
#else
    throw std::runtime_error("eval_xc_points: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("eval_xc_points: unsupported backend");
  }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n); ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    const double r = rho[idx];
    const double s = sigma.empty() ? 0.0 : sigma[idx];
    const double t = tau.empty() ? 0.0 : tau[idx];
    xc_gen::XcPointOut pt;
    xc_gen::eval_xc_point(id, r, s, t, deriv, pt, rho_cutoff);
    res.zk[idx] = pt.zk;
    if (deriv >= 1) {
      res.vrho[idx] = pt.vrho;
      res.vsigma[idx] = pt.vsigma;
      res.vtau[idx] = pt.vtau;
    }
    if (deriv >= 2) {
      res.v2rho2[idx] = pt.v2rho2;
      res.v2rhosigma[idx] = pt.v2rhosigma;
      res.v2sigma2[idx] = pt.v2sigma2;
      res.v2rhotau[idx] = pt.v2rhotau;
      res.v2sigmatau[idx] = pt.v2sigmatau;
      res.v2tau2[idx] = pt.v2tau2;
    }
  }
  return res;
}

std::vector<double> eval_ao_grid(const Basis& basis,
                                 std::span<const double> coords_xyz,
                                 int deriv,
                                 Backend backend) {
  if (deriv < 0 || deriv > 2) {
    throw std::invalid_argument("eval_ao_grid: deriv must be in {0, 1, 2}");
  }
  if (coords_xyz.size() % 3 != 0) {
    throw std::invalid_argument("eval_ao_grid: coords_xyz length must be divisible by 3");
  }
  const std::size_t npts = coords_xyz.size() / 3;
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  const std::size_t ncomp = (deriv == 0) ? 1 : ((deriv == 1) ? 4 : 10);
  std::vector<double> out(ncomp * npts * nao, 0.0);
  if (npts == 0 || nao == 0) return out;

  const detail::FlatBasisData flat(basis);
  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::xc_cuda_device_count() == 0) {
      throw std::runtime_error("eval_ao_grid: no CUDA device available");
    }
    detail::eval_ao_grid_cuda(flat, coords_xyz.data(), npts, deriv, out.data());
    return out;
#else
    throw std::runtime_error("eval_ao_grid: gpu4dft was built without CUDA");
#endif
  }
  if (backend != Backend::Host) {
    throw std::runtime_error("eval_ao_grid: unsupported backend");
  }

  const detail::FlatBasisView bview = flat.view();
  const std::size_t comp_stride = npts * nao;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t ig = 0; ig < static_cast<std::ptrdiff_t>(npts); ++ig) {
    const std::size_t g = static_cast<std::size_t>(ig);
    const double gx = coords_xyz[3 * g + 0];
    const double gy = coords_xyz[3 * g + 1];
    const double gz = coords_xyz[3 * g + 2];
    for (int is = 0; is < bview.nshell; ++is) {
      const std::size_t ao0 = static_cast<std::size_t>(bview.shell_ao_offset[is]);
      detail::eval_shell_at_point(bview, is, gx, gy, gz, deriv,
                                  out.data() + g * nao + ao0, comp_stride, 1);
    }
  }
  return out;
}

namespace {

// Cache-friendly Host blocked GEMM C[M x N] = A[M x K] * B[K x N] (or B^T when trans_b=true: B[N x K]).
void host_gemm_nn(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  }
  for (int i = 0; i < M; ++i) {
    double* ci = C + static_cast<std::size_t>(i) * N;
    const double* ai = A + static_cast<std::size_t>(i) * K;
    for (int k = 0; k < K; ++k) {
      const double aik = ai[k];
      if (aik == 0.0) continue;
      const double* bk = B + static_cast<std::size_t>(k) * N;
      for (int j = 0; j < N; ++j) {
        ci[j] += aik * bk[j];
      }
    }
  }
}

void host_gemm_nt(int M, int N, int K, const double* A, const double* B, double* C,
                  bool accum = false) {
  if (!accum) {
    std::fill_n(C, static_cast<std::size_t>(M) * N, 0.0);
  }
  for (int i = 0; i < M; ++i) {
    const double* ai = A + static_cast<std::size_t>(i) * K;
    double* ci = C + static_cast<std::size_t>(i) * N;
    for (int j = 0; j < N; ++j) {
      const double* bj = B + static_cast<std::size_t>(j) * K;
      double sum = 0.0;
      for (int k = 0; k < K; ++k) {
        sum += ai[k] * bj[k];
      }
      ci[j] += sum;
    }
  }
}

std::vector<int> build_ao_to_atom(const Basis& basis, std::span<const XcAtom> atoms) {
  std::vector<int> ao_to_atom(static_cast<std::size_t>(basis.nao()), -1);
  for (std::size_t is = 0; is < basis.nshell(); ++is) {
    const Shell& sh = basis.shell(is);
    int matched = -1;
    for (std::size_t ia = 0; ia < atoms.size(); ++ia) {
      const double dx = sh.center[0] - atoms[ia].center[0];
      const double dy = sh.center[1] - atoms[ia].center[1];
      const double dz = sh.center[2] - atoms[ia].center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-10) {
        matched = static_cast<int>(ia);
        break;
      }
    }
    const int ao0 = basis.ao_offset(is);
    const int nm = 2 * sh.l + 1;
    for (int m = 0; m < nm; ++m) {
      ao_to_atom[static_cast<std::size_t>(ao0 + m)] = matched;
    }
  }
  return ao_to_atom;
}

}  // namespace

XcEngine::XcEngine(const Basis& basis, const MolecularGrid& grid,
                   std::string_view xc_name, const XcOptions& opts)
    : basis_(basis), grid_(grid), xc_name_(xc_name), opts_(opts) {
  if (grid_.nao() != basis_.nao() && grid_.npoints() > 0) {
    throw std::invalid_argument("XcEngine: grid.nao() does not match basis.nao()");
  }
  const auto fid = xc_gen::parse_xc_functional(xc_name_);
  xc_func_id_ = static_cast<int>(fid);
  xc_family_ = static_cast<int>(xc_gen::xc_functional_family(fid));
  exx_fraction_ = xc_gen::xc_exact_exchange_fraction_by_id(fid);

  if (opts_.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (detail::xc_cuda_device_count() > 0) {
      const detail::FlatBasisData flat(basis_);
      cuda_ = std::make_unique<CudaState>();
      cuda_->impl = detail::xc_cuda_engine_create(flat, grid_, xc_func_id_, xc_family_, opts_);
      active_backend_ = Backend::Cuda;
    } else {
      throw std::runtime_error("XcEngine: Backend::Cuda requested, but no CUDA device is available");
    }
#else
    throw std::runtime_error("XcEngine: gpu4dft was built without CUDA");
#endif
  } else if (opts_.backend == Backend::Host) {
    active_backend_ = Backend::Host;
  } else {
    throw std::runtime_error("XcEngine: unsupported backend");
  }
}

XcEngine::~XcEngine() {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_ && cuda_->impl) {
    detail::xc_cuda_engine_destroy(cuda_->impl);
    cuda_->impl = nullptr;
  }
#endif
}

XcEngine::XcEngine(XcEngine&&) noexcept = default;
XcEngine& XcEngine::operator=(XcEngine&&) noexcept = default;

XcResult XcEngine::eval_vxc(std::span<const double> dm) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (dm.size() != n * n) {
    throw std::invalid_argument("XcEngine::eval_vxc: dm size must equal nao * nao");
  }
  XcResult res;
  res.vxc.assign(n * n, 0.0);
  res.exc = eval_vxc(dm.data(), res.vxc.data(), &res.nelec);
  return res;
}

double XcEngine::eval_vxc(const double* dm, double* vxc_out, double* nelec_out) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  std::fill_n(vxc_out, nao * nao, 0.0);
  if (nelec_out) *nelec_out = 0.0;
  if (nao == 0 || grid_.npoints() == 0) return 0.0;

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return detail::xc_cuda_eval_vxc_host(cuda_->impl, dm, vxc_out, nelec_out);
#endif
  }

  // Host OpenMP spatially batched active-AO reference implementation
  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id_);
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();

  double total_exc = 0.0;
  double total_nelec = 0.0;

#if defined(_OPENMP)
#pragma omp parallel reduction(+ : total_exc, total_nelec)
#endif
  {
    std::vector<double> local_vxc(nao * nao, 0.0);
    std::vector<double> phi;
    std::vector<double> d_act;
    std::vector<double> t_mat;
    std::vector<double> w_mat;
    std::vector<double> v_half;
    std::vector<double> rho_g, gx_g, gy_g, gz_g, tau_g, vtau_w;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      phi.assign(ncomp * comp_stride, 0.0);

      // Evaluate active shells into compact [ncomp][na][ng] row-major buffer
      std::size_t local_ao_cursor = 0;
      for (int is : batch.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = batch.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              deriv,
              phi.data() + local_ao_cursor * ng + g,
              comp_stride, ng);
        }
        local_ao_cursor += static_cast<std::size_t>(nm);
      }

      // Gather symmetric active density block D_act [na x na]
      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (dm[gu * nao + gv] + dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat.resize(n_tcomp * comp_stride);

      // T0 = D_act * Phi0
      host_gemm_nn(i_na, i_ng, i_na, d_act.data(), phi.data(), t_mat.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                       phi.data() + b * comp_stride,
                       t_mat.data() + b * comp_stride, false);
        }
      }

      rho_g.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        gx_g.assign(ng, 0.0);
        gy_g.assign(ng, 0.0);
        gz_g.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        tau_g.assign(ng, 0.0);
        vtau_w.assign(ng, 0.0);
      }

      const double* phi0 = phi.data();
      const double* phix = (xc_family_ >= 1) ? (phi.data() + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi.data() + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi.data() + 3 * comp_stride) : nullptr;
      const double* t0 = t_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const double t0_ug = t0[u * ng + g];
          rho_g[g] += phi0[u * ng + g] * t0_ug;
          if (xc_family_ >= 1) {
            gx_g[g] += 2.0 * phix[u * ng + g] * t0_ug;
            gy_g[g] += 2.0 * phiy[u * ng + g] * t0_ug;
            gz_g[g] += 2.0 * phiz[u * ng + g] * t0_ug;
          }
          if (xc_family_ == 2) {
            const double tx = t_mat[1 * comp_stride + u * ng + g];
            const double ty = t_mat[2 * comp_stride + u * ng + g];
            const double tz = t_mat[3 * comp_stride + u * ng + g];
            tau_g[g] += 0.5 * (phix[u * ng + g] * tx + phiy[u * ng + g] * ty +
                               phiz[u * ng + g] * tz);
          }
        }
      }

      // Evaluate generated symbolic XC kernel and build half-weighted W [na x ng]
      w_mat.assign(comp_stride, 0.0);
      for (std::size_t g = 0; g < ng; ++g) {
        const double wg = weights[batch.point_offset + g];
        const double r = rho_g[g];
        const double s = (xc_family_ >= 1)
                             ? (gx_g[g] * gx_g[g] + gy_g[g] * gy_g[g] + gz_g[g] * gz_g[g])
                             : 0.0;
        const double t = (xc_family_ == 2) ? tau_g[g] : 0.0;

        if (r > 0.0) {
          total_nelec += wg * r;
        }
        xc_gen::XcPointOut out_pt;
        xc_gen::eval_xc_point(fid, r, s, t, 1, out_pt, opts_.rho_cutoff);
        total_exc += wg * r * out_pt.zk;

        const double wv0 = 0.5 * wg * out_pt.vrho;
        const double wvx = (xc_family_ >= 1) ? (2.0 * wg * out_pt.vsigma * gx_g[g]) : 0.0;
        const double wvy = (xc_family_ >= 1) ? (2.0 * wg * out_pt.vsigma * gy_g[g]) : 0.0;
        const double wvz = (xc_family_ >= 1) ? (2.0 * wg * out_pt.vsigma * gz_g[g]) : 0.0;
        if (xc_family_ == 2) {
          vtau_w[g] = 0.5 * wg * out_pt.vtau;
        }

        for (std::size_t u = 0; u < na; ++u) {
          double w_ug = wv0 * phi0[u * ng + g];
          if (xc_family_ >= 1) {
            w_ug += wvx * phix[u * ng + g] + wvy * phiy[u * ng + g] + wvz * phiz[u * ng + g];
          }
          w_mat[u * ng + g] = w_ug;
        }
      }

      // Symmetric GEMM V_half = Phi0 * W^T, V_act = V_half + V_half^T
      v_half.resize(na * na);
      host_gemm_nt(i_na, i_na, i_ng, phi0, w_mat.data(), v_half.data(), false);

      if (xc_family_ == 2) {
        // Add MGGA tau potential: 0.5 * sum_b Phi_b * diag(w * vtau) * Phi_b^T
        // Since V_act = V_half + V_half^T adds both transposes, scale by 0.5 here so 0.5*(0.5 + 0.5) = 0.5!
        for (std::size_t b = 1; b <= 3; ++b) {
          const double* phib = phi.data() + b * comp_stride;
          for (std::size_t u = 0; u < na; ++u) {
            for (std::size_t g = 0; g < ng; ++g) {
              w_mat[u * ng + g] = 0.5 * vtau_w[g] * phib[u * ng + g];
            }
          }
          host_gemm_nt(i_na, i_na, i_ng, phib, w_mat.data(), v_half.data(), true);
        }
      }

      // Scatter symmetrized active V_xc into local_vxc [nao x nao]
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          local_vxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t idx = 0; idx < nao * nao; ++idx) {
        vxc_out[idx] += local_vxc[idx];
      }
    }
  }

  if (nelec_out) *nelec_out = total_nelec;
  return total_exc;
}

double XcEngine::eval_vxc_device(const double* d_dm, double* d_vxc, void* cuda_stream) const {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::eval_vxc_device requires Backend::Cuda");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  return detail::xc_cuda_eval_vxc_device(cuda_->impl, d_dm, d_vxc, nullptr, cuda_stream);
#else
  (void)d_dm;
  (void)d_vxc;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void XcEngine::set_reference_density(std::span<const double> h_dm) {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (h_dm.size() != n * n) {
    throw std::invalid_argument("XcEngine::set_reference_density: dm size must equal nao * nao");
  }
  set_reference_density(h_dm.data());
}

void XcEngine::set_reference_density(const double* h_dm) {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (nao > 0 && h_dm == nullptr) {
    throw std::invalid_argument("XcEngine::set_reference_density: h_dm must not be null");
  }

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::xc_cuda_set_reference_density_host(cuda_->impl, h_dm);
    return;
#endif
  }

  if (!host_fxc_) {
    host_fxc_ = std::make_unique<HostFxcState>();
  }
  HostFxcState& st = *host_fxc_;
  const std::size_t npts = grid_.npoints();
  st.v2rho2.assign(npts, 0.0);
  if (xc_family_ >= 1) {
    st.g0x.assign(npts, 0.0);
    st.g0y.assign(npts, 0.0);
    st.g0z.assign(npts, 0.0);
    st.vsigma.assign(npts, 0.0);
    st.v2rhosigma.assign(npts, 0.0);
    st.v2sigma2.assign(npts, 0.0);
  }
  if (xc_family_ == 2) {
    st.v2rhotau.assign(npts, 0.0);
    st.v2sigmatau.assign(npts, 0.0);
    st.v2tau2.assign(npts, 0.0);
  }
  if (nao == 0 || npts == 0) {
    st.ready = true;
    return;
  }

  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id_);
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();

  // Cache host active-batch Phi if within 512 MiB so repeated TDDFT eval_fxc calls skip AO eval
  if (!st.ao_cached) {
    std::size_t total_elems = 0;
    st.batch_phi_offset.resize(batches.size(), 0);
    for (std::size_t ib = 0; ib < batches.size(); ++ib) {
      st.batch_phi_offset[ib] = total_elems;
      total_elems += ncomp * batches[ib].active_aos.size() * batches[ib].npoints;
    }
    if (total_elems > 0 && total_elems * sizeof(double) <= (512ULL << 20)) {
      st.phi_cache.assign(total_elems, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
      for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
        const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
        const std::size_t ng = batch.npoints;
        const std::size_t na = batch.active_aos.size();
        if (ng == 0 || na == 0) continue;
        const std::size_t comp_stride = na * ng;
        double* dst = st.phi_cache.data() + st.batch_phi_offset[static_cast<std::size_t>(ib)];
        std::size_t local_ao_cursor = 0;
        for (int is : batch.active_shells) {
          const int nm = 2 * bview.shell_l[is] + 1;
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t gp = batch.point_offset + g;
            detail::eval_shell_at_point(
                bview, is,
                coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                deriv, dst + local_ao_cursor * ng + g, comp_stride, ng);
          }
          local_ao_cursor += static_cast<std::size_t>(nm);
        }
      }
      st.ao_cached = true;
    } else {
      st.batch_phi_offset.clear();
    }
  }

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> phi_local;
    std::vector<double> d_act;
    std::vector<double> t_mat;
    std::vector<double> rho_g, gx_g, gy_g, gz_g, tau_g;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const std::size_t uib = static_cast<std::size_t>(ib);
      const SpatialBatch& batch = batches[uib];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      const double* phi_ptr = nullptr;
      if (st.ao_cached) {
        phi_ptr = st.phi_cache.data() + st.batch_phi_offset[uib];
      } else {
        phi_local.assign(ncomp * comp_stride, 0.0);
        std::size_t local_ao_cursor = 0;
        for (int is : batch.active_shells) {
          const int nm = 2 * bview.shell_l[is] + 1;
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t gp = batch.point_offset + g;
            detail::eval_shell_at_point(
                bview, is,
                coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                deriv, phi_local.data() + local_ao_cursor * ng + g, comp_stride, ng);
          }
          local_ao_cursor += static_cast<std::size_t>(nm);
        }
        phi_ptr = phi_local.data();
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (h_dm[gu * nao + gv] + h_dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat.resize(n_tcomp * comp_stride);
      host_gemm_nn(i_na, i_ng, i_na, d_act.data(), phi_ptr, t_mat.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                       phi_ptr + b * comp_stride,
                       t_mat.data() + b * comp_stride, false);
        }
      }

      rho_g.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        gx_g.assign(ng, 0.0);
        gy_g.assign(ng, 0.0);
        gz_g.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        tau_g.assign(ng, 0.0);
      }

      const double* phi0 = phi_ptr;
      const double* phix = (xc_family_ >= 1) ? (phi_ptr + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi_ptr + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi_ptr + 3 * comp_stride) : nullptr;
      const double* t0 = t_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double t0_ug = t0[off];
          rho_g[g] += phi0[off] * t0_ug;
          if (xc_family_ >= 1) {
            gx_g[g] += 2.0 * phix[off] * t0_ug;
            gy_g[g] += 2.0 * phiy[off] * t0_ug;
            gz_g[g] += 2.0 * phiz[off] * t0_ug;
          }
          if (xc_family_ == 2) {
            tau_g[g] += 0.5 * (phix[off] * t_mat[1 * comp_stride + off] +
                               phiy[off] * t_mat[2 * comp_stride + off] +
                               phiz[off] * t_mat[3 * comp_stride + off]);
          }
        }
      }

      for (std::size_t g = 0; g < ng; ++g) {
        const std::size_t gp = batch.point_offset + g;
        const double r = rho_g[g];
        const double s = (xc_family_ >= 1)
                             ? (gx_g[g] * gx_g[g] + gy_g[g] * gy_g[g] + gz_g[g] * gz_g[g])
                             : 0.0;
        const double t = (xc_family_ == 2) ? tau_g[g] : 0.0;

        xc_gen::XcPointOut pt;
        xc_gen::eval_xc_point(fid, r, s, t, 2, pt, opts_.rho_cutoff);
        st.v2rho2[gp] = pt.v2rho2;
        if (xc_family_ >= 1) {
          st.g0x[gp] = gx_g[g];
          st.g0y[gp] = gy_g[g];
          st.g0z[gp] = gz_g[g];
          st.vsigma[gp] = pt.vsigma;
          st.v2rhosigma[gp] = pt.v2rhosigma;
          st.v2sigma2[gp] = pt.v2sigma2;
        }
        if (xc_family_ == 2) {
          st.v2rhotau[gp] = pt.v2rhotau;
          st.v2sigmatau[gp] = pt.v2sigmatau;
          st.v2tau2[gp] = pt.v2tau2;
        }
      }
    }
  }
  st.ready = true;
}

void XcEngine::set_reference_density_device(const double* d_dm, void* cuda_stream) {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::set_reference_density_device requires Backend::Cuda");
  }
  if (basis_.nao() > 0 && d_dm == nullptr) {
    throw std::invalid_argument("XcEngine::set_reference_density_device: d_dm must not be null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  detail::xc_cuda_set_reference_density_device(cuda_->impl, d_dm, cuda_stream);
#else
  (void)d_dm;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

bool XcEngine::has_reference_density() const noexcept {
  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return cuda_ && cuda_->impl && detail::xc_cuda_has_reference_density(cuda_->impl);
#else
    return false;
#endif
  }
  return host_fxc_ && host_fxc_->ready;
}

std::vector<double> XcEngine::eval_fxc(std::span<const double> h_delta_dm) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  if (h_delta_dm.size() != n * n) {
    throw std::invalid_argument("XcEngine::eval_fxc: delta_dm size must equal nao * nao");
  }
  return eval_fxc(h_delta_dm.data());
}

std::vector<double> XcEngine::eval_fxc(const double* h_delta_dm) const {
  const std::size_t n = static_cast<std::size_t>(basis_.nao());
  std::vector<double> fxc(n * n, 0.0);
  eval_fxc(h_delta_dm, fxc.data());
  return fxc;
}

void XcEngine::eval_fxc(const double* h_delta_dm, double* h_fxc_out) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (nao > 0 && h_fxc_out == nullptr) {
    throw std::invalid_argument("XcEngine::eval_fxc: h_fxc_out must not be null");
  }
  if (nao > 0 && h_delta_dm == nullptr) {
    throw std::invalid_argument("XcEngine::eval_fxc: h_delta_dm must not be null");
  }
  if (!has_reference_density()) {
    throw std::logic_error(
        "XcEngine::eval_fxc: reference density not set; call set_reference_density first");
  }
  std::fill_n(h_fxc_out, nao * nao, 0.0);
  if (nao == 0 || grid_.npoints() == 0) return;

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::xc_cuda_eval_fxc_host(cuda_->impl, h_delta_dm, h_fxc_out);
    return;
#endif
  }

  const HostFxcState& st = *host_fxc_;
  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const int deriv = (xc_family_ == 0) ? 0 : 1;
  const std::size_t ncomp = (deriv == 0) ? 1 : 4;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_fxc(nao * nao, 0.0);
    std::vector<double> phi_local;
    std::vector<double> d_act;
    std::vector<double> t_mat;
    std::vector<double> w_mat;
    std::vector<double> v_half;
    std::vector<double> drho, dgx, dgy, dgz, dtau, vtau_w;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const std::size_t uib = static_cast<std::size_t>(ib);
      const SpatialBatch& batch = batches[uib];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      const double* phi_ptr = nullptr;
      if (st.ao_cached) {
        phi_ptr = st.phi_cache.data() + st.batch_phi_offset[uib];
      } else {
        phi_local.assign(ncomp * comp_stride, 0.0);
        std::size_t local_ao_cursor = 0;
        for (int is : batch.active_shells) {
          const int nm = 2 * bview.shell_l[is] + 1;
          for (std::size_t g = 0; g < ng; ++g) {
            const std::size_t gp = batch.point_offset + g;
            detail::eval_shell_at_point(
                bview, is,
                coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
                deriv, phi_local.data() + local_ao_cursor * ng + g, comp_stride, ng);
          }
          local_ao_cursor += static_cast<std::size_t>(nm);
        }
        phi_ptr = phi_local.data();
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (h_delta_dm[gu * nao + gv] + h_delta_dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_tcomp = (xc_family_ == 2) ? 4 : 1;
      t_mat.resize(n_tcomp * comp_stride);
      host_gemm_nn(i_na, i_ng, i_na, d_act.data(), phi_ptr, t_mat.data(), false);
      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                       phi_ptr + b * comp_stride,
                       t_mat.data() + b * comp_stride, false);
        }
      }

      drho.assign(ng, 0.0);
      if (xc_family_ >= 1) {
        dgx.assign(ng, 0.0);
        dgy.assign(ng, 0.0);
        dgz.assign(ng, 0.0);
      }
      if (xc_family_ == 2) {
        dtau.assign(ng, 0.0);
        vtau_w.assign(ng, 0.0);
      }

      const double* phi0 = phi_ptr;
      const double* phix = (xc_family_ >= 1) ? (phi_ptr + 1 * comp_stride) : nullptr;
      const double* phiy = (xc_family_ >= 1) ? (phi_ptr + 2 * comp_stride) : nullptr;
      const double* phiz = (xc_family_ >= 1) ? (phi_ptr + 3 * comp_stride) : nullptr;
      const double* t0 = t_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t off = u * ng + g;
          const double t0_ug = t0[off];
          drho[g] += phi0[off] * t0_ug;
          if (xc_family_ >= 1) {
            dgx[g] += 2.0 * phix[off] * t0_ug;
            dgy[g] += 2.0 * phiy[off] * t0_ug;
            dgz[g] += 2.0 * phiz[off] * t0_ug;
          }
          if (xc_family_ == 2) {
            dtau[g] += 0.5 * (phix[off] * t_mat[1 * comp_stride + off] +
                              phiy[off] * t_mat[2 * comp_stride + off] +
                              phiz[off] * t_mat[3 * comp_stride + off]);
          }
        }
      }

      w_mat.assign(comp_stride, 0.0);
      for (std::size_t g = 0; g < ng; ++g) {
        const std::size_t gp = batch.point_offset + g;
        const double wg = weights[gp];
        const double dr = drho[g];

        double d_vrho = st.v2rho2[gp] * dr;
        double wvx = 0.0, wvy = 0.0, wvz = 0.0;
        if (xc_family_ >= 1) {
          const double gx0 = st.g0x[gp];
          const double gy0 = st.g0y[gp];
          const double gz0 = st.g0z[gp];
          const double dsig = 2.0 * (gx0 * dgx[g] + gy0 * dgy[g] + gz0 * dgz[g]);
          d_vrho += st.v2rhosigma[gp] * dsig;
          double d_vsigma = st.v2rhosigma[gp] * dr + st.v2sigma2[gp] * dsig;
          if (xc_family_ == 2) {
            const double dt = dtau[g];
            d_vrho += st.v2rhotau[gp] * dt;
            d_vsigma += st.v2sigmatau[gp] * dt;
            const double d_vtau =
                st.v2rhotau[gp] * dr + st.v2sigmatau[gp] * dsig + st.v2tau2[gp] * dt;
            vtau_w[g] = 0.5 * wg * d_vtau;
          }
          const double vs0 = st.vsigma[gp];
          wvx = 2.0 * wg * (d_vsigma * gx0 + vs0 * dgx[g]);
          wvy = 2.0 * wg * (d_vsigma * gy0 + vs0 * dgy[g]);
          wvz = 2.0 * wg * (d_vsigma * gz0 + vs0 * dgz[g]);
        }
        const double wv0 = 0.5 * wg * d_vrho;

        for (std::size_t u = 0; u < na; ++u) {
          const std::size_t off = u * ng + g;
          double w_ug = wv0 * phi0[off];
          if (xc_family_ >= 1) {
            w_ug += wvx * phix[off] + wvy * phiy[off] + wvz * phiz[off];
          }
          w_mat[off] = w_ug;
        }
      }

      v_half.resize(na * na);
      host_gemm_nt(i_na, i_na, i_ng, phi0, w_mat.data(), v_half.data(), false);

      if (xc_family_ == 2) {
        for (std::size_t b = 1; b <= 3; ++b) {
          const double* phib = phi_ptr + b * comp_stride;
          for (std::size_t u = 0; u < na; ++u) {
            for (std::size_t g = 0; g < ng; ++g) {
              w_mat[u * ng + g] = 0.5 * vtau_w[g] * phib[u * ng + g];
            }
          }
          host_gemm_nt(i_na, i_na, i_ng, phib, w_mat.data(), v_half.data(), true);
        }
      }

      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          local_fxc[gu * nao + gv] += v_half[u * na + v] + v_half[v * na + u];
        }
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t idx = 0; idx < nao * nao; ++idx) {
        h_fxc_out[idx] += local_fxc[idx];
      }
    }
  }
}

void XcEngine::eval_fxc_device(const double* d_delta_dm, double* d_fxc_out,
                               void* cuda_stream) const {
  if (active_backend_ != Backend::Cuda) {
    throw std::runtime_error("XcEngine::eval_fxc_device requires Backend::Cuda");
  }
  if (!has_reference_density()) {
    throw std::logic_error(
        "XcEngine::eval_fxc_device: reference density not set; call set_reference_density_device first");
  }
  if (basis_.nao() > 0 && (d_delta_dm == nullptr || d_fxc_out == nullptr)) {
    throw std::invalid_argument("XcEngine::eval_fxc_device: device pointers must not be null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  detail::xc_cuda_eval_fxc_device(cuda_->impl, d_delta_dm, d_fxc_out, cuda_stream);
#else
  (void)d_delta_dm;
  (void)d_fxc_out;
  (void)cuda_stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

std::vector<double> XcEngine::nuclear_grad(std::span<const XcAtom> atoms,
                                           std::span<const double> dm) const {
  const std::size_t nao = static_cast<std::size_t>(basis_.nao());
  if (dm.size() != nao * nao) {
    throw std::invalid_argument("XcEngine::nuclear_grad: dm size must equal nao * nao");
  }
  const int natom = static_cast<int>(atoms.size());
  std::vector<double> grad(static_cast<std::size_t>(natom) * 3, 0.0);
  if (natom == 0 || nao == 0 || grid_.npoints() == 0) return grad;

  const std::vector<int> ao_to_atom = build_ao_to_atom(basis_, atoms);

  if (active_backend_ == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    return detail::xc_cuda_nuclear_grad(cuda_->impl, ao_to_atom.data(), natom, dm.data());
#endif
  }

  // Host OpenMP analytical XC nuclear gradient (LDA, GGA, Hybrid XC, Meta-GGA)
  const detail::FlatBasisData flat(basis_);
  const detail::FlatBasisView bview = flat.view();
  const auto fid = static_cast<xc_gen::XcFunctionalId>(xc_func_id_);
  const int deriv = (xc_family_ == 0) ? 1 : 2;
  const std::size_t ncomp = (deriv == 1) ? 4 : 10;
  const auto& batches = grid_.batches();
  const auto& coords = grid_.coords();
  const auto& weights = grid_.weights();

  static constexpr int kIdx2[3][3] = {
      {4, 5, 6},
      {5, 7, 8},
      {6, 8, 9},
  };

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<double> local_grad(static_cast<std::size_t>(natom) * 3, 0.0);
    std::vector<double> phi;
    std::vector<double> d_act;
    std::vector<double> f_mat;
    std::vector<double> rho_g, gx_g, gy_g, gz_g, tau_g;
    std::vector<double> wv0, wvx, wvy, wvz, wvt;

#if defined(_OPENMP)
#pragma omp for schedule(dynamic)
#endif
    for (std::ptrdiff_t ib = 0; ib < static_cast<std::ptrdiff_t>(batches.size()); ++ib) {
      const SpatialBatch& batch = batches[static_cast<std::size_t>(ib)];
      const std::size_t ng = batch.npoints;
      const std::size_t na = batch.active_aos.size();
      if (ng == 0 || na == 0) continue;

      const std::size_t comp_stride = na * ng;
      phi.assign(ncomp * comp_stride, 0.0);

      std::size_t local_ao_cursor = 0;
      for (int is : batch.active_shells) {
        const int nm = 2 * bview.shell_l[is] + 1;
        for (std::size_t g = 0; g < ng; ++g) {
          const std::size_t gp = batch.point_offset + g;
          detail::eval_shell_at_point(
              bview, is,
              coords[3 * gp + 0], coords[3 * gp + 1], coords[3 * gp + 2],
              deriv,
              phi.data() + local_ao_cursor * ng + g,
              comp_stride, ng);
        }
        local_ao_cursor += static_cast<std::size_t>(nm);
      }

      d_act.resize(na * na);
      for (std::size_t u = 0; u < na; ++u) {
        const std::size_t gu = static_cast<std::size_t>(batch.active_aos[u]);
        for (std::size_t v = 0; v < na; ++v) {
          const std::size_t gv = static_cast<std::size_t>(batch.active_aos[v]);
          d_act[u * na + v] = 0.5 * (dm[gu * nao + gv] + dm[gv * nao + gu]);
        }
      }

      const int i_na = static_cast<int>(na);
      const int i_ng = static_cast<int>(ng);
      const std::size_t n_fcomp = (xc_family_ >= 1) ? 4 : 1;
      f_mat.resize(n_fcomp * comp_stride);
      for (std::size_t b = 0; b < n_fcomp; ++b) {
        host_gemm_nn(i_na, i_ng, i_na, d_act.data(),
                     phi.data() + b * comp_stride,
                     f_mat.data() + b * comp_stride, false);
      }

      rho_g.assign(ng, 0.0);
      gx_g.assign(ng, 0.0);
      gy_g.assign(ng, 0.0);
      gz_g.assign(ng, 0.0);
      tau_g.assign(ng, 0.0);

      const double* phi0 = phi.data();
      const double* phix = phi.data() + 1 * comp_stride;
      const double* phiy = phi.data() + 2 * comp_stride;
      const double* phiz = phi.data() + 3 * comp_stride;
      const double* f0 = f_mat.data();

      for (std::size_t u = 0; u < na; ++u) {
        for (std::size_t g = 0; g < ng; ++g) {
          const double f0_ug = f0[u * ng + g];
          rho_g[g] += phi0[u * ng + g] * f0_ug;
          if (xc_family_ >= 1) {
            gx_g[g] += 2.0 * phix[u * ng + g] * f0_ug;
            gy_g[g] += 2.0 * phiy[u * ng + g] * f0_ug;
            gz_g[g] += 2.0 * phiz[u * ng + g] * f0_ug;
          }
          if (xc_family_ == 2) {
            tau_g[g] += 0.5 * (phix[u * ng + g] * f_mat[1 * comp_stride + u * ng + g] +
                               phiy[u * ng + g] * f_mat[2 * comp_stride + u * ng + g] +
                               phiz[u * ng + g] * f_mat[3 * comp_stride + u * ng + g]);
          }
        }
      }

      wv0.resize(ng);
      wvx.assign(ng, 0.0);
      wvy.assign(ng, 0.0);
      wvz.assign(ng, 0.0);
      wvt.assign(ng, 0.0);

      for (std::size_t g = 0; g < ng; ++g) {
        const double wg = weights[batch.point_offset + g];
        const double r = rho_g[g];
        const double s = (xc_family_ >= 1)
                             ? (gx_g[g] * gx_g[g] + gy_g[g] * gy_g[g] + gz_g[g] * gz_g[g])
                             : 0.0;
        const double t = (xc_family_ == 2) ? tau_g[g] : 0.0;
        xc_gen::XcPointOut pt;
        xc_gen::eval_xc_point(fid, r, s, t, 1, pt, opts_.rho_cutoff);
        wv0[g] = wg * pt.vrho;
        if (xc_family_ >= 1) {
          wvx[g] = 2.0 * wg * pt.vsigma * gx_g[g];
          wvy[g] = 2.0 * wg * pt.vsigma * gy_g[g];
          wvz[g] = 2.0 * wg * pt.vsigma * gz_g[g];
        }
        if (xc_family_ == 2) {
          wvt[g] = wg * pt.vtau;
        }
      }

      for (std::size_t u = 0; u < na; ++u) {
        const int ia = ao_to_atom[static_cast<std::size_t>(batch.active_aos[u])];
        if (ia < 0) continue;

        double gx_acc = 0.0, gy_acc = 0.0, gz_acc = 0.0;
        for (std::size_t g = 0; g < ng; ++g) {
          const double f0_ug = f0[u * ng + g];
          double w0_ug = wv0[g] * f0_ug;
          if (xc_family_ >= 1) {
            const double fx_ug = f_mat[1 * comp_stride + u * ng + g];
            const double fy_ug = f_mat[2 * comp_stride + u * ng + g];
            const double fz_ug = f_mat[3 * comp_stride + u * ng + g];
            w0_ug += wvx[g] * fx_ug + wvy[g] * fy_ug + wvz[g] * fz_ug;

            const double wb_x = 2.0 * wvx[g] * f0_ug + ((xc_family_ == 2) ? wvt[g] * fx_ug : 0.0);
            const double wb_y = 2.0 * wvy[g] * f0_ug + ((xc_family_ == 2) ? wvt[g] * fy_ug : 0.0);
            const double wb_z = 2.0 * wvz[g] * f0_ug + ((xc_family_ == 2) ? wvt[g] * fz_ug : 0.0);
            const double wb[3] = {wb_x, wb_y, wb_z};

            double sec[3] = {0.0, 0.0, 0.0};
            for (int c = 0; c < 3; ++c) {
              for (int b = 0; b < 3; ++b) {
                sec[c] += phi[static_cast<std::size_t>(kIdx2[b][c]) * comp_stride + u * ng + g] * wb[b];
              }
            }
            gx_acc -= sec[0];
            gy_acc -= sec[1];
            gz_acc -= sec[2];
          }
          gx_acc -= 2.0 * phix[u * ng + g] * w0_ug;
          gy_acc -= 2.0 * phiy[u * ng + g] * w0_ug;
          gz_acc -= 2.0 * phiz[u * ng + g] * w0_ug;
        }
        local_grad[static_cast<std::size_t>(ia) * 3 + 0] += gx_acc;
        local_grad[static_cast<std::size_t>(ia) * 3 + 1] += gy_acc;
        local_grad[static_cast<std::size_t>(ia) * 3 + 2] += gz_acc;
      }
    }

#if defined(_OPENMP)
#pragma omp critical
#endif
    {
      for (std::size_t k = 0; k < local_grad.size(); ++k) {
        grad[k] += local_grad[k];
      }
    }
  }
  return grad;
}

std::vector<double> xc_nuclear_grad(const Basis& basis,
                                    const MolecularGrid& grid,
                                    std::span<const XcAtom> atoms,
                                    std::string_view xc_name,
                                    const double* dm,
                                    Backend backend) {
  XcOptions opts;
  opts.backend = backend;
  XcEngine engine(basis, grid, xc_name, opts);
  const std::size_t nao = static_cast<std::size_t>(basis.nao());
  return engine.nuclear_grad(atoms, std::span<const double>(dm, nao * nao));
}

}  // namespace gpu4dft
