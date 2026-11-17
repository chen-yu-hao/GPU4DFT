#include "gpu4dft/jk_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/eri.hpp"
#include "gpu4dft/generated/eri_deriv1.hpp"

#if defined(GPU4DFT_HAVE_CUDA)
#include <cuda_runtime.h>
#endif

namespace gpu4dft {
namespace {

constexpr int kMaxSupportedL = 4;
constexpr int kHostTileKetSize = 16;
constexpr int kHostRiAuxChunk = 64;

// Prepares spherical density [nao x nao] (symmetrizing if `symmetrize_dm == true`,
// or keeping raw `dm` if `symmetrize_dm == false`) and transforms to Cartesian
// [ncart_total x ncart_total]: D_cart = C^T D_eff C. Also computes per-shell
// maximum |D| norms D_sh[ish * nshell + jsh] and global_d_max.
void prepare_cartesian_density(const Basis& basis, const std::vector<int>& shell_cart_off,
                               int ncart_tot, const double* dm, std::vector<double>& d_cart,
                               std::vector<double>& d_shell_max, double& global_d_max,
                               bool symmetrize_dm = true) {
  const int nao = basis.nao();
  const int nsh = static_cast<int>(basis.nshell());
  d_cart.assign(static_cast<std::size_t>(ncart_tot) * ncart_tot, 0.0);
  d_shell_max.assign(static_cast<std::size_t>(nsh) * nsh, 0.0);
  global_d_max = 0.0;

  std::vector<std::vector<double>> c_mats(nsh);
  for (int s = 0; s < nsh; ++s) {
    const int l = basis.shell(s).l;
    const int nc = n_cart(l);
    const int ns = n_sph(l);
    c_mats[s].assign(static_cast<std::size_t>(ns) * nc, 0.0);
    double unit[15] = {0.0};
    double col[9] = {0.0};
    for (int ic = 0; ic < nc; ++ic) {
      unit[ic] = 1.0;
      harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
      for (int m = 0; m < ns; ++m) c_mats[s][m * nc + ic] = col[m];
      unit[ic] = 0.0;
    }
  }

  for (int i = 0; i < nsh; ++i) {
    const int li = basis.shell(i).l;
    const int nci = n_cart(li), nsi = n_sph(li);
    const int aoi = basis.ao_offset(i);
    const int ci0 = shell_cart_off[i];
    const auto& Ci = c_mats[i];

    for (int j = 0; j < nsh; ++j) {
      const int lj = basis.shell(j).l;
      const int ncj = n_cart(lj), nsj = n_sph(lj);
      const int aoj = basis.ao_offset(j);
      const int cj0 = shell_cart_off[j];
      const auto& Cj = c_mats[j];

      double blk_max = 0.0;
      double tmp[9 * 15] = {0.0};
      for (int mi = 0; mi < nsi; ++mi) {
        for (int mj = 0; mj < nsj; ++mj) {
          const int u = aoi + mi;
          const int v = aoj + mj;
          const double duv = dm[u * nao + v];
          const double dvu = dm[v * nao + u];
          double d_eff = duv;
          if (symmetrize_dm) {
            d_eff = 0.5 * (duv + dvu);
            const double ad = std::abs(d_eff);
            if (ad > blk_max) blk_max = ad;
          } else {
            const double ad = std::max(std::abs(duv), std::abs(dvu));
            if (ad > blk_max) blk_max = ad;
          }
          for (int cj = 0; cj < ncj; ++cj) {
            tmp[mi * ncj + cj] += d_eff * Cj[mj * ncj + cj];
          }
        }
      }
      d_shell_max[i * nsh + j] = blk_max;
      if (blk_max > global_d_max) global_d_max = blk_max;

      for (int ci = 0; ci < nci; ++ci) {
        for (int cj = 0; cj < ncj; ++cj) {
          double acc = 0.0;
          for (int mi = 0; mi < nsi; ++mi) {
            acc += Ci[mi * nci + ci] * tmp[mi * ncj + cj];
          }
          d_cart[(ci0 + ci) * ncart_tot + (cj0 + cj)] = acc;
        }
      }
    }
  }
}

// Transforms Cartesian Fock/J/K [ncart_tot x ncart_tot] -> Spherical [nao x nao],
// symmetrizing `0.5 * (M_raw + M_raw^T)` when `symmetrize_out == true` or
// preserving general non-symmetric structure when `symmetrize_out == false`.
void finalize_cartesian_fock(const Basis& basis, const std::vector<int>& shell_cart_off,
                             int ncart_tot, const std::vector<double>& m_cart_raw, double scale,
                             double* m_sph_out, bool symmetrize_out = true) {
  const int nao = basis.nao();
  const int nsh = static_cast<int>(basis.nshell());

  for (int i = 0; i < nsh; ++i) {
    const int li = basis.shell(i).l;
    const int nci = n_cart(li), nsi = n_sph(li);
    const int aoi = basis.ao_offset(i);
    const int ci0 = shell_cart_off[i];
    const int j_end = symmetrize_out ? (i + 1) : nsh;

    for (int j = 0; j < j_end; ++j) {
      const int lj = basis.shell(j).l;
      const int ncj = n_cart(lj), nsj = n_sph(lj);
      const int aoj = basis.ao_offset(j);
      const int cj0 = shell_cart_off[j];

      double blk_cart[225];
      for (int a = 0; a < nci; ++a) {
        const int r = ci0 + a;
        for (int b = 0; b < ncj; ++b) {
          const int c = cj0 + b;
          blk_cart[a * ncj + b] =
              symmetrize_out
                  ? 0.5 * (m_cart_raw[r * ncart_tot + c] + m_cart_raw[c * ncart_tot + r])
                  : m_cart_raw[r * ncart_tot + c];
        }
      }

      double blk_sph[81];
      harmonics::cart_to_sph_2d<double>(li, lj, blk_cart, blk_sph);

      for (int mi = 0; mi < nsi; ++mi) {
        for (int mj = 0; mj < nsj; ++mj) {
          const double v = scale * blk_sph[mi * nsj + mj];
          m_sph_out[(aoi + mi) * nao + (aoj + mj)] = v;
          if (symmetrize_out && i != j) {
            m_sph_out[(aoj + mj) * nao + (aoi + mi)] = v;
          }
        }
      }
    }
  }
}

double quartet_d_max(const double* d_sh, int nsh, int i, int j, int k, int l, bool do_j,
                     bool do_k) {
  double m = 0.0;
  if (do_j) {
    m = std::max(m, 4.0 * d_sh[i * nsh + j]);
    m = std::max(m, 4.0 * d_sh[k * nsh + l]);
  }
  if (do_k) {
    m = std::max(m, 2.0 * d_sh[i * nsh + k]);
    m = std::max(m, 2.0 * d_sh[i * nsh + l]);
    m = std::max(m, 2.0 * d_sh[j * nsh + k]);
    m = std::max(m, 2.0 * d_sh[j * nsh + l]);
  }
  return m;
}

template <bool SymmetrizeDM>
void eval_host_direct_jk_tiles(const std::vector<ShellPairData>& pair_lists,
                               const std::vector<detail::JKTaskTile>& tasks,
                               const std::vector<int>& shell_cart_off,
                               const std::vector<double>& d_cart,
                               const std::vector<double>& d_shell_max, double global_scale,
                               int nsh, int ncart_tot, bool do_j, bool do_k, const JKOptions& opts,
                               std::vector<double>& j_cart_raw, std::vector<double>& k_cart_raw) {
  const std::size_t nc2 = static_cast<std::size_t>(ncart_tot) * ncart_tot;
  #if defined(_OPENMP)
  #pragma omp parallel
  #endif
  {
    std::vector<double> j_loc(do_j ? nc2 : 0, 0.0);
    std::vector<double> k_loc(do_k ? nc2 : 0, 0.0);
    auto accum_j_ket = [&](int r, int c, double val) { j_loc[r * ncart_tot + c] += val; };
    auto accum_k = [&](int r, int c, double val) { k_loc[r * ncart_tot + c] += val; };

    #if defined(_OPENMP)
    #pragma omp for schedule(dynamic, 4)
    #endif
    for (std::ptrdiff_t tid = 0; tid < static_cast<std::ptrdiff_t>(tasks.size()); ++tid) {
      const auto& t = tasks[tid];
      const ShellPairData& bra = pair_lists[t.bp];
      const ShellPairData& ket = pair_lists[t.kp];
      const ShellPairView bv = bra.view();
      const ShellPairView kv = ket.view();
      const int P = t.P;
      const int i = bra.ish[P], j = bra.jsh[P];
      const double deg_ij = (i == j) ? 1.0 : 2.0;
      const double q_ab = bra.schwarz_bound[P];
      const int off_a = shell_cart_off[i], off_b = shell_cart_off[j];
      const int nca = n_cart(bra.la), ncb = n_cart(bra.lb);
      const int max_l = std::max({1, bra.la, bra.lb, ket.la, ket.lb});

      double j_ab_tile[225] = {0.0};
      auto accum_j_bra = [&](int ia, int ib_c, double val) {
        j_ab_tile[ia * ncb + ib_c] += val;
      };

      for (int Q = t.Q_begin; Q < t.Q_end; ++Q) {
        const double q_cd = ket.schwarz_bound[Q];
        if (q_ab * q_cd * global_scale < opts.schwarz_cutoff) break;
        const int k = ket.ish[Q], l = ket.jsh[Q];
        const double dmax = quartet_d_max(d_shell_max.data(), nsh, i, j, k, l, do_j, do_k);
        const double est = q_ab * q_cd * dmax;
        if (est < opts.schwarz_cutoff) continue;

        const bool same_PQ = (t.bp == t.kp) && (P == Q);
        const double deg_kl = (k == l) ? 1.0 : 2.0;
        const double s_PQ = same_PQ ? 1.0 : 2.0;
        const double w_J = deg_ij * deg_kl;
        const double w_K =
            SymmetrizeDM ? (0.25 * deg_ij * deg_kl * s_PQ) : (0.25 * deg_ij * deg_kl);
        const int off_c = shell_cart_off[k], off_d = shell_cart_off[l];

        const bool use_fp64 =
            (opts.precision == Precision::FP64) ||
            (opts.precision == Precision::Mixed && est >= opts.fp32_threshold);
        if (max_l <= 1) {
          if (use_fp64) {
            detail::eval_direct_jk_quartet_tiered<1, double, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k);
          } else {
            detail::eval_direct_jk_quartet_tiered<1, float, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k);
          }
        } else if (max_l == 2) {
          if (use_fp64) {
            detail::eval_direct_jk_quartet_tiered<2, double, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k);
          } else {
            detail::eval_direct_jk_quartet_tiered<2, float, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k);
          }
        } else {
          if (use_fp64) {
            detail::eval_direct_jk_quartet_tiered<4, double, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k);
          } else {
            detail::eval_direct_jk_quartet_tiered<4, float, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k);
          }
        }
      }

      if (do_j) {
        for (int ia = 0; ia < nca; ++ia) {
          for (int ib_c = 0; ib_c < ncb; ++ib_c) {
            j_loc[(off_a + ia) * ncart_tot + (off_b + ib_c)] += j_ab_tile[ia * ncb + ib_c];
          }
        }
      }
    }

    #if defined(_OPENMP)
    #pragma omp critical
    #endif
    {
      if (do_j) {
        for (std::size_t idx = 0; idx < nc2; ++idx) j_cart_raw[idx] += j_loc[idx];
      }
      if (do_k) {
        for (std::size_t idx = 0; idx < nc2; ++idx) k_cart_raw[idx] += k_loc[idx];
      }
    }
  }
}

// Cholesky factorization V = L L^T with fallback to Jacobi symmetric inverse-square-root
// if auxiliary metric has near-linear dependence, solving B = V^{-1/2} I_3c [naux x nao^2].
void solve_ri_metric(int naux, std::size_t nao2, const std::vector<double>& metric,
                     const std::vector<double>& eri_3c, std::vector<double>& b_out) {
  b_out.assign(static_cast<std::size_t>(naux) * nao2, 0.0);
  std::vector<double> L = metric;
  bool chol_ok = true;
  for (int i = 0; i < naux; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = L[i * naux + j];
      for (int k = 0; k < j; ++k) sum -= L[i * naux + k] * L[j * naux + k];
      if (i == j) {
        if (!(sum > 1e-11)) {
          chol_ok = false;
          break;
        }
        L[i * naux + i] = std::sqrt(sum);
      } else {
        L[i * naux + j] = sum / L[j * naux + j];
      }
    }
    if (!chol_ok) break;
  }

  if (chol_ok) {
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t uv = 0; uv < static_cast<std::ptrdiff_t>(nao2); ++uv) {
      for (int i = 0; i < naux; ++i) {
        double sum = eri_3c[static_cast<std::size_t>(i) * nao2 + uv];
        for (int k = 0; k < i; ++k) {
          sum -= L[i * naux + k] * b_out[static_cast<std::size_t>(k) * nao2 + uv];
        }
        b_out[static_cast<std::size_t>(i) * nao2 + uv] = sum / L[i * naux + i];
      }
    }
    return;
  }

  // Fallback: cyclic Jacobi eigendecomposition V = U diag(w) U^T -> V^{-1/2} = U diag(w^{-1/2}) U^T.
  std::vector<double> A = metric;
  std::vector<double> U(static_cast<std::size_t>(naux) * naux, 0.0);
  for (int i = 0; i < naux; ++i) U[i * naux + i] = 1.0;

  for (int sweep = 0; sweep < 80; ++sweep) {
    double max_off = 0.0;
    for (int p = 0; p < naux; ++p)
      for (int q = p + 1; q < naux; ++q) max_off = std::max(max_off, std::abs(A[p * naux + q]));
    if (max_off < 1e-14) break;

    for (int p = 0; p < naux; ++p) {
      for (int q = p + 1; q < naux; ++q) {
        const double apq = A[p * naux + q];
        if (std::abs(apq) < 1e-15) continue;
        const double tau = (A[q * naux + q] - A[p * naux + p]) / (2.0 * apq);
        const double t = (tau >= 0.0) ? (1.0 / (tau + std::sqrt(1.0 + tau * tau)))
                                      : (-1.0 / (-tau + std::sqrt(1.0 + tau * tau)));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;
        A[p * naux + p] -= t * apq;
        A[q * naux + q] += t * apq;
        A[p * naux + q] = A[q * naux + p] = 0.0;
        for (int r = 0; r < naux; ++r) {
          if (r != p && r != q) {
            const double arp = A[r * naux + p];
            const double arq = A[r * naux + q];
            A[r * naux + p] = A[p * naux + r] = c * arp - s * arq;
            A[r * naux + q] = A[q * naux + r] = s * arp + c * arq;
          }
          const double urp = U[r * naux + p];
          const double urq = U[r * naux + q];
          U[r * naux + p] = c * urp - s * urq;
          U[r * naux + q] = s * urp + c * urq;
        }
      }
    }
  }

  std::vector<double> inv_sqrt(static_cast<std::size_t>(naux) * naux, 0.0);
  for (int k = 0; k < naux; ++k) {
    const double ev = A[k * naux + k];
    if (ev > 1e-10) {
      const double sc = 1.0 / std::sqrt(ev);
      for (int i = 0; i < naux; ++i)
        for (int j = 0; j < naux; ++j)
          inv_sqrt[i * naux + j] += U[i * naux + k] * sc * U[j * naux + k];
    }
  }

  #if defined(_OPENMP)
  #pragma omp parallel for schedule(static)
  #endif
  for (int p = 0; p < naux; ++p) {
    for (int q = 0; q < naux; ++q) {
      const double w = inv_sqrt[p * naux + q];
      if (w == 0.0) continue;
      for (std::size_t uv = 0; uv < nao2; ++uv) {
        b_out[static_cast<std::size_t>(p) * nao2 + uv] +=
            w * eri_3c[static_cast<std::size_t>(q) * nao2 + uv];
      }
    }
  }
}

std::vector<Atom> infer_atoms_from_basis(const Basis& basis) {
  std::vector<Atom> atoms;
  for (std::size_t i = 0; i < basis.nshell(); ++i) {
    const auto& c = basis.shell(i).center;
    bool found = false;
    for (const Atom& a : atoms) {
      const double dx = c[0] - a.center[0];
      const double dy = c[1] - a.center[1];
      const double dz = c[2] - a.center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-12) {
        found = true;
        break;
      }
    }
    if (!found) atoms.push_back(Atom{0, c});
  }
  return atoms;
}

}  // namespace

DirectJKEngine::DirectJKEngine(const Basis& basis, double prim_cutoff) : basis_(basis) {
  if (basis_.max_l() > kMaxSupportedL) {
    throw NotImplemented("DirectJKEngine supports up to g shells (l <= 4); basis has l = " +
                         std::to_string(basis_.max_l()));
  }
  shell_cart_offset_.reserve(basis_.nshell());
  ncart_total_ = 0;
  for (std::size_t i = 0; i < basis_.nshell(); ++i) {
    shell_cart_offset_.push_back(ncart_total_);
    ncart_total_ += n_cart(basis_.shell(i).l);
  }

  const int max_l = basis_.max_l();
  for (int la = 0; la <= max_l; ++la) {
    for (int lb = 0; lb <= la; ++lb) {
      ShellPairData pd = build_shell_pairs(basis_, la, lb, prim_cutoff);
      if (pd.npair() > 0) pair_lists_.push_back(std::move(pd));
    }
  }

#if defined(GPU4DFT_HAVE_CUDA)
  if (basis_.nao() > 0 && backend_available(Backend::Cuda)) {
    cuda_cache_ = detail::create_direct_jk_cuda_cache(basis_, pair_lists_, shell_cart_offset_,
                                                      ncart_total_);
  }
#endif
}

DirectJKEngine::~DirectJKEngine() {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_cache_ != nullptr) {
    detail::destroy_direct_jk_cuda_cache(cuda_cache_);
    cuda_cache_ = nullptr;
  }
#endif
}

DirectJKEngine::DirectJKEngine(DirectJKEngine&& o) noexcept
    : basis_(std::move(o.basis_)),
      pair_lists_(std::move(o.pair_lists_)),
      shell_cart_offset_(std::move(o.shell_cart_offset_)),
      ncart_total_(o.ncart_total_),
      cuda_cache_(o.cuda_cache_) {
  o.cuda_cache_ = nullptr;
}

DirectJKEngine& DirectJKEngine::operator=(DirectJKEngine&& o) noexcept {
  if (this != &o) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (cuda_cache_ != nullptr) detail::destroy_direct_jk_cuda_cache(cuda_cache_);
#endif
    basis_ = std::move(o.basis_);
    pair_lists_ = std::move(o.pair_lists_);
    shell_cart_offset_ = std::move(o.shell_cart_offset_);
    ncart_total_ = o.ncart_total_;
    cuda_cache_ = o.cuda_cache_;
    o.cuda_cache_ = nullptr;
  }
  return *this;
}

void DirectJKEngine::compute_jk_device(const double* d_dm, double* d_vj, double* d_vk,
                                       const JKOptions& opts, void* stream) const {
  if (!backend_available(Backend::Cuda)) {
    throw std::runtime_error("DirectJKEngine::compute_jk_device: backend 'cuda' is not available");
  }
  if (d_dm == nullptr) {
    throw std::invalid_argument("DirectJKEngine::compute_jk_device: d_dm is null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  const bool do_j = opts.compute_j && (d_vj != nullptr);
  const bool do_k = opts.compute_k && (d_vk != nullptr);
  if ((!do_j && !do_k) || nao() == 0) return;

  if (!cuda_cache_) {
    cuda_cache_ = detail::create_direct_jk_cuda_cache(basis_, pair_lists_, shell_cart_offset_,
                                                      ncart_total_);
  }
  JKOptions eff_opts = opts;
  eff_opts.backend = Backend::Cuda;
  eff_opts.compute_j = do_j;
  eff_opts.compute_k = do_k;
  detail::run_direct_jk_cuda_device(*cuda_cache_, d_dm, do_j ? d_vj : nullptr,
                                    do_k ? d_vk : nullptr, eff_opts, stream);
#else
  (void)d_vj;
  (void)d_vk;
  (void)opts;
  (void)stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void DirectJKEngine::compute_jk(const double* dm, double* vj_out, double* vk_out,
                                const JKOptions& opts) const {
  if (!backend_available(opts.backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(opts.backend) +
                             "' is not available");
  }
  if (dm == nullptr) throw std::invalid_argument("DirectJKEngine::compute_jk: dm is null");

  const bool do_j = opts.compute_j && (vj_out != nullptr);
  const bool do_k = opts.compute_k && (vk_out != nullptr);
  const std::size_t nao2 = static_cast<std::size_t>(nao()) * nao();

  if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if ((!do_j && !do_k) || nao() == 0) {
      if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
      if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
      return;
    }
    if (!cuda_cache_) {
      cuda_cache_ = detail::create_direct_jk_cuda_cache(basis_, pair_lists_, shell_cart_offset_,
                                                        ncart_total_);
    }
    JKOptions eff_opts = opts;
    eff_opts.compute_j = do_j;
    eff_opts.compute_k = do_k;
    detail::run_direct_jk_cuda_host(*cuda_cache_, dm, do_j ? vj_out : nullptr,
                                    do_k ? vk_out : nullptr, eff_opts);
    return;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  if (opts.backend != Backend::Host) {
    throw std::runtime_error("gpu4dft was built without HIP");
  }

  if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
  if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
  if ((!do_j && !do_k) || nao() == 0) return;

  const bool eff_sym_dm = opts.symmetrize_dm || !do_k;
  std::vector<double> d_cart, d_shell_max;
  double global_d_max = 0.0;
  prepare_cartesian_density(basis_, shell_cart_offset_, ncart_total_, dm, d_cart, d_shell_max,
                            global_d_max, eff_sym_dm);
  const double global_scale = (do_j ? 4.0 : 2.0) * global_d_max;
  if (global_scale == 0.0) return;

  // Build Schwarz-screened task tiles for Host OpenMP execution.
  std::vector<detail::JKTaskTile> tasks;
  for (int bp = 0; bp < static_cast<int>(pair_lists_.size()); ++bp) {
    const ShellPairData& bra = pair_lists_[bp];
    for (int P = 0; P < bra.npair(); ++P) {
      const double q_ab = bra.schwarz_bound[P];
      if (q_ab * q_ab * global_scale < opts.schwarz_cutoff * 1e-6) continue;
      for (int kp = 0; kp <= bp; ++kp) {
        const ShellPairData& ket = pair_lists_[kp];
        const int q_limit = (bp == kp) ? (P + 1) : ket.npair();
        for (int Q0 = 0; Q0 < q_limit; Q0 += kHostTileKetSize) {
          if (q_ab * ket.schwarz_bound[Q0] * global_scale < opts.schwarz_cutoff) {
            break;
          }
          const int Q1 = std::min(Q0 + kHostTileKetSize, q_limit);
          tasks.push_back(detail::JKTaskTile{bp, P, kp, Q0, Q1});
        }
      }
    }
  }
  if (tasks.empty()) return;

  std::vector<double> j_cart_raw, k_cart_raw;
  const std::size_t nc2 = static_cast<std::size_t>(ncart_total_) * ncart_total_;
  if (do_j) j_cart_raw.assign(nc2, 0.0);
  if (do_k) k_cart_raw.assign(nc2, 0.0);

  const int nsh = static_cast<int>(basis_.nshell());
  if (eff_sym_dm) {
    eval_host_direct_jk_tiles<true>(pair_lists_, tasks, shell_cart_offset_, d_cart, d_shell_max,
                                    global_scale, nsh, ncart_total_, do_j, do_k, opts, j_cart_raw,
                                    k_cart_raw);
  } else {
    eval_host_direct_jk_tiles<false>(pair_lists_, tasks, shell_cart_offset_, d_cart, d_shell_max,
                                     global_scale, nsh, ncart_total_, do_j, do_k, opts, j_cart_raw,
                                     k_cart_raw);
  }

  if (do_j) {
    finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, j_cart_raw, 1.0, vj_out,
                            /*symmetrize_out=*/true);
  }
  if (do_k) {
    finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, k_cart_raw, opts.k_scale,
                            vk_out, /*symmetrize_out=*/opts.symmetrize_dm);
  }
}

void DirectJKEngine::compute_incremental_jk(const double* dm_new, const double* dm_old,
                                            double* vj_accum, double* vk_accum,
                                            const JKOptions& opts) const {
  const std::size_t n2 = static_cast<std::size_t>(nao()) * nao();
  std::vector<double> delta_dm(n2, 0.0);
  for (std::size_t i = 0; i < n2; ++i) delta_dm[i] = dm_new[i] - dm_old[i];

  std::vector<double> dj(opts.compute_j && vj_accum ? n2 : 0, 0.0);
  std::vector<double> dk(opts.compute_k && vk_accum ? n2 : 0, 0.0);
  compute_jk(delta_dm.data(), dj.empty() ? nullptr : dj.data(), dk.empty() ? nullptr : dk.data(),
             opts);
  if (!dj.empty()) {
    for (std::size_t i = 0; i < n2; ++i) vj_accum[i] += dj[i];
  }
  if (!dk.empty()) {
    for (std::size_t i = 0; i < n2; ++i) vk_accum[i] += dk[i];
  }
}

RIJKEngine::RIJKEngine(const Basis& basis, const Basis& aux_basis, Backend backend)
    : basis_(basis), aux_basis_(aux_basis) {
  if (basis_.max_l() > kMaxSupportedL || aux_basis_.max_l() > kMaxSupportedL) {
    throw NotImplemented("RIJKEngine supports up to g shells (l <= 4)");
  }
  const Backend build_backend = backend_available(backend) ? backend : Backend::Host;
  const int n_ao = basis_.nao();
  const int n_aux = aux_basis_.nao();
  const std::size_t nao2 = static_cast<std::size_t>(n_ao) * n_ao;

  // 1. Auxiliary 1-shell and orbital 2-shell pair lists WITHOUT redundant 4-center
  //    CPU Schwarz evaluation/sorting (RIJKEngine never uses `schwarz_bound`!).
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kY00 = 0.28209479177387814347;

  std::vector<ShellPairData> aux_lists;
  for (int lP = 0; lP <= aux_basis_.max_l(); ++lP) {
    ShellPairData ap;
    ap.la = lP;
    ap.lb = 0;
    const int nsh_aux = static_cast<int>(aux_basis_.nshell());
    for (int i = 0; i < nsh_aux; ++i) {
      const Shell& s = aux_basis_.shell(i);
      if (s.l != lP) continue;
      const auto& A = s.center;
      for (std::size_t k = 0; k < s.exponents.size(); ++k) {
        const double a = s.exponents[k];
        ap.exponent.push_back(a);
        ap.Px.push_back(A[0]);
        ap.Py.push_back(A[1]);
        ap.Pz.push_back(A[2]);
        ap.prefactor.push_back(s.coefficients[k] * kY00 * std::pow(kPi / a, 1.5));
        ap.alpha_a.push_back(a);
        ap.alpha_b.push_back(0.0);
      }
      ap.ish.push_back(i);
      ap.jsh.push_back(i);
      ap.ao_i.push_back(aux_basis_.ao_offset(i));
      ap.ao_j.push_back(0);
      ap.Ax.push_back(A[0]);
      ap.Ay.push_back(A[1]);
      ap.Az.push_back(A[2]);
      ap.Bx.push_back(A[0]);
      ap.By.push_back(A[1]);
      ap.Bz.push_back(A[2]);
      ap.prim_offset.push_back(static_cast<int>(ap.exponent.size()));
    }
    ap.schwarz_bound.assign(static_cast<std::size_t>(ap.npair()), 0.0);
    if (ap.npair() > 0) aux_lists.push_back(std::move(ap));
  }

  std::vector<ShellPairData> orb_lists;
  const int nsh_orb = static_cast<int>(basis_.nshell());
  for (int la = 0; la <= basis_.max_l(); ++la) {
    for (int lb = 0; lb <= la; ++lb) {
      ShellPairData op;
      op.la = la;
      op.lb = lb;
      for (int i = 0; i < nsh_orb; ++i) {
        if (basis_.shell(i).l != la) continue;
        const Shell& si = basis_.shell(i);
        const auto& A = si.center;
        const int jend = (la == lb) ? (i + 1) : nsh_orb;
        for (int j = 0; j < jend; ++j) {
          if (basis_.shell(j).l != lb) continue;
          const Shell& sj = basis_.shell(j);
          const auto& B = sj.center;
          const double ABx = A[0] - B[0], ABy = A[1] - B[1], ABz = A[2] - B[2];
          const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;
          for (std::size_t ka = 0; ka < si.exponents.size(); ++ka) {
            const double a = si.exponents[ka];
            for (std::size_t kb = 0; kb < sj.exponents.size(); ++kb) {
              const double b = sj.exponents[kb];
              const double p = a + b;
              const double pref = si.coefficients[ka] * sj.coefficients[kb] * kY00 * kY00 *
                                  std::exp(-a * b / p * AB2) * std::pow(kPi / p, 1.5);
              op.exponent.push_back(p);
              op.Px.push_back((a * A[0] + b * B[0]) / p);
              op.Py.push_back((a * A[1] + b * B[1]) / p);
              op.Pz.push_back((a * A[2] + b * B[2]) / p);
              op.prefactor.push_back(pref);
              op.alpha_a.push_back(a);
              op.alpha_b.push_back(b);
            }
          }
          op.ish.push_back(i);
          op.jsh.push_back(j);
          op.ao_i.push_back(basis_.ao_offset(i));
          op.ao_j.push_back(basis_.ao_offset(j));
          op.Ax.push_back(A[0]);
          op.Ay.push_back(A[1]);
          op.Az.push_back(A[2]);
          op.Bx.push_back(B[0]);
          op.By.push_back(B[1]);
          op.Bz.push_back(B[2]);
          op.prim_offset.push_back(static_cast<int>(op.exponent.size()));
        }
      }
      op.schwarz_bound.assign(static_cast<std::size_t>(op.npair()), 0.0);
      if (op.npair() > 0) orb_lists.push_back(std::move(op));
    }
  }

#if defined(GPU4DFT_HAVE_CUDA)
  if (build_backend == Backend::Cuda) {
    // 100% GPU-native setup: 2c/3c integral kernels scatter directly into GPU `d_metric` & `d_B`,
    // factorize via `cusolverDnDpotrf` (or `cusolverDnDsyevd`), and solve in-place via `cublasDtrsm`
    // with ZERO host `eri_3c` / `b_perm_tensor_` allocation or D2H copy!
    cuda_cache_ = detail::build_rijk_cuda_cache_native(n_ao, n_aux, orb_lists, aux_lists);
    return;
  }
#endif

  // Host reference setup path (`Backend::Host`).
  std::vector<double> metric(static_cast<std::size_t>(n_aux) * n_aux, 0.0);
  for (std::size_t ip = 0; ip < aux_lists.size(); ++ip) {
    const ShellPairData& bra_aux = aux_lists[ip];
    const int nsP = n_sph(bra_aux.la);
    for (std::size_t iq = 0; iq <= ip; ++iq) {
      const ShellPairData& ket_aux = aux_lists[iq];
      const int nsQ = n_sph(ket_aux.la);
      const std::size_t nsph = static_cast<std::size_t>(nsP) * nsQ;
      const std::size_t nket = static_cast<std::size_t>(ket_aux.npair());
      const std::vector<double> blk =
          eri_pair_block(bra_aux, ket_aux, Backend::Host, Precision::FP64);

      for (int P = 0; P < bra_aux.npair(); ++P) {
        const int p0 = bra_aux.ao_i[P];
        for (int Q = 0; Q < ket_aux.npair(); ++Q) {
          const int q0 = ket_aux.ao_i[Q];
          const double* ptr = blk.data() + (static_cast<std::size_t>(P) * nket + Q) * nsph;
          for (int mp = 0; mp < nsP; ++mp) {
            for (int mq = 0; mq < nsQ; ++mq) {
              const double v = ptr[mp * nsQ + mq];
              metric[(p0 + mp) * n_aux + (q0 + mq)] = v;
              metric[(q0 + mq) * n_aux + (p0 + mp)] = v;
            }
          }
        }
      }
    }
  }

  std::vector<double> eri_3c(static_cast<std::size_t>(n_aux) * nao2, 0.0);
  for (const ShellPairData& orb : orb_lists) {
    const int nsa = n_sph(orb.la), nsb = n_sph(orb.lb);
    for (const ShellPairData& aux : aux_lists) {
      const int nsP = n_sph(aux.la);
      const std::size_t nsph = static_cast<std::size_t>(nsa) * nsb * nsP;
      const std::size_t nket = static_cast<std::size_t>(aux.npair());
      const std::vector<double> blk = eri_pair_block(orb, aux, Backend::Host, Precision::FP64);

      for (int ab = 0; ab < orb.npair(); ++ab) {
        const int u0 = orb.ao_i[ab];
        const int v0 = orb.ao_j[ab];
        for (int P = 0; P < aux.npair(); ++P) {
          const int p0 = aux.ao_i[P];
          const double* ptr = blk.data() + (static_cast<std::size_t>(ab) * nket + P) * nsph;
          for (int ma = 0; ma < nsa; ++ma) {
            const int u = u0 + ma;
            for (int mb = 0; mb < nsb; ++mb) {
              const int v = v0 + mb;
              for (int mp = 0; mp < nsP; ++mp) {
                const double val = ptr[(ma * nsb + mb) * nsP + mp];
                const std::size_t row = static_cast<std::size_t>(p0 + mp) * nao2;
                eri_3c[row + static_cast<std::size_t>(u) * n_ao + v] = val;
                eri_3c[row + static_cast<std::size_t>(v) * n_ao + u] = val;
              }
            }
          }
        }
      }
    }
  }

  solve_ri_metric(n_aux, nao2, metric, eri_3c, b_tensor_);
}

RIJKEngine::~RIJKEngine() {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_cache_ != nullptr) {
    detail::destroy_rijk_cuda_cache(cuda_cache_);
    cuda_cache_ = nullptr;
  }
#endif
}

RIJKEngine::RIJKEngine(RIJKEngine&& o) noexcept
    : basis_(std::move(o.basis_)),
      aux_basis_(std::move(o.aux_basis_)),
      b_tensor_(std::move(o.b_tensor_)),
      b_perm_tensor_(std::move(o.b_perm_tensor_)),
      cuda_cache_(o.cuda_cache_) {
  o.cuda_cache_ = nullptr;
}

RIJKEngine& RIJKEngine::operator=(RIJKEngine&& o) noexcept {
  if (this != &o) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (cuda_cache_ != nullptr) detail::destroy_rijk_cuda_cache(cuda_cache_);
#endif
    basis_ = std::move(o.basis_);
    aux_basis_ = std::move(o.aux_basis_);
    b_tensor_ = std::move(o.b_tensor_);
    b_perm_tensor_ = std::move(o.b_perm_tensor_);
    cuda_cache_ = o.cuda_cache_;
    o.cuda_cache_ = nullptr;
  }
  return *this;
}

bool RIJKEngine::has_device_b() const noexcept {
#if defined(GPU4DFT_HAVE_CUDA)
  return (cuda_cache_ != nullptr) && (detail::rijk_cuda_device_b_ptr(*cuda_cache_) != nullptr);
#else
  return false;
#endif
}

const double* RIJKEngine::device_b_ptr() const noexcept {
#if defined(GPU4DFT_HAVE_CUDA)
  return (cuda_cache_ != nullptr) ? detail::rijk_cuda_device_b_ptr(*cuda_cache_) : nullptr;
#else
  return nullptr;
#endif
}

const std::vector<double>& RIJKEngine::b_tensor() const {
#if defined(GPU4DFT_HAVE_CUDA)
  if (b_tensor_.empty() && cuda_cache_ != nullptr && nao() > 0 && naux() > 0) {
    detail::download_rijk_b_tensor(*cuda_cache_, b_tensor_);
  }
#endif
  return b_tensor_;
}

void RIJKEngine::compute_jk_device(const double* d_dm, double* d_vj, double* d_vk,
                                   const JKOptions& opts, void* stream) const {
  if (!backend_available(Backend::Cuda)) {
    throw std::runtime_error("RIJKEngine::compute_jk_device: backend 'cuda' is not available");
  }
  if (d_dm == nullptr) {
    throw std::invalid_argument("RIJKEngine::compute_jk_device: d_dm is null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  const int n_ao = nao();
  const int n_aux = naux();
  const bool do_j = opts.compute_j && (d_vj != nullptr);
  const bool do_k = opts.compute_k && (d_vk != nullptr);
  if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) return;

  if (!cuda_cache_) {
    cuda_cache_ = detail::create_rijk_cuda_cache_from_host(n_ao, n_aux, b_tensor_);
  }
  JKOptions eff_opts = opts;
  eff_opts.backend = Backend::Cuda;
  eff_opts.compute_j = do_j;
  eff_opts.compute_k = do_k;
  detail::run_rijk_cuda_device(*cuda_cache_, d_dm, do_j ? d_vj : nullptr, do_k ? d_vk : nullptr,
                               eff_opts, stream);
#else
  (void)d_vj;
  (void)d_vk;
  (void)opts;
  (void)stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void RIJKEngine::compute_jk(const double* dm, double* vj_out, double* vk_out,
                            const JKOptions& opts) const {
  if (!backend_available(opts.backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(opts.backend) +
                             "' is not available");
  }
  if (dm == nullptr) throw std::invalid_argument("RIJKEngine::compute_jk: dm is null");

  const int n_ao = nao();
  const int n_aux = naux();
  const std::size_t nao2 = static_cast<std::size_t>(n_ao) * n_ao;
  const bool do_j = opts.compute_j && (vj_out != nullptr);
  const bool do_k = opts.compute_k && (vk_out != nullptr);

  if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) {
      if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
      if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
      return;
    }
    if (!cuda_cache_) {
      cuda_cache_ = detail::create_rijk_cuda_cache_from_host(n_ao, n_aux, b_tensor_);
    }
    JKOptions eff_opts = opts;
    eff_opts.compute_j = do_j;
    eff_opts.compute_k = do_k;
    detail::run_rijk_cuda(*cuda_cache_, dm, do_j ? vj_out : nullptr, do_k ? vk_out : nullptr,
                          eff_opts);
    return;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  if (opts.backend != Backend::Host) {
    throw std::runtime_error("gpu4dft was built without HIP");
  }

  if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
  if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
  if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) return;

  const std::vector<double>& b_host = b_tensor();

  std::vector<double> d_eff(nao2);
  if (opts.symmetrize_dm) {
    for (int u = 0; u < n_ao; ++u)
      for (int v = 0; v < n_ao; ++v)
        d_eff[u * n_ao + v] = 0.5 * (dm[u * n_ao + v] + dm[v * n_ao + u]);
  } else {
    std::copy(dm, dm + nao2, d_eff.begin());
  }

  // Host OpenMP GEMV / chunked GEMM contraction directly on `b_host`.
  if (do_j) {
    std::vector<double> v_aux(static_cast<std::size_t>(n_aux), 0.0);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int p = 0; p < n_aux; ++p) {
      const double* bp = b_host.data() + static_cast<std::size_t>(p) * nao2;
      double acc = 0.0;
      for (std::size_t uv = 0; uv < nao2; ++uv) acc += bp[uv] * d_eff[uv];
      v_aux[static_cast<std::size_t>(p)] = acc;
    }

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t uv = 0; uv < static_cast<std::ptrdiff_t>(nao2); ++uv) {
      double acc = 0.0;
      for (int p = 0; p < n_aux; ++p) {
        acc += b_host[static_cast<std::size_t>(p) * nao2 + static_cast<std::size_t>(uv)] *
               v_aux[static_cast<std::size_t>(p)];
      }
      vj_out[uv] = acc;
    }
  }

  if (do_k && opts.k_scale != 0.0) {
    const int bc_max = std::min(n_aux, kHostRiAuxChunk);
    std::vector<double> t_chunk(static_cast<std::size_t>(bc_max) * nao2, 0.0);
    std::vector<double> bp_chunk(static_cast<std::size_t>(bc_max) * nao2, 0.0);
    std::vector<double> tp_chunk(static_cast<std::size_t>(bc_max) * nao2, 0.0);

    for (int p0 = 0; p0 < n_aux; p0 += kHostRiAuxChunk) {
      const int bc = std::min(kHostRiAuxChunk, n_aux - p0);
      const int m_rows = bc * n_ao;
      const std::size_t k_c = static_cast<std::size_t>(bc) * n_ao;
      const double* b_c = b_host.data() + static_cast<std::size_t>(p0) * nao2;

      // Step 1: T_chunk[pl * n_ao + u, s] = sum_l B_c[pl * n_ao + u, l] * D_eff[l, s]
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static)
      #endif
      for (int row = 0; row < m_rows; ++row) {
        const double* b_row = b_c + static_cast<std::size_t>(row) * n_ao;
        double* t_row = t_chunk.data() + static_cast<std::size_t>(row) * n_ao;
        std::fill(t_row, t_row + n_ao, 0.0);
        for (int l = 0; l < n_ao; ++l) {
          const double bv = b_row[l];
          const double* d_row = d_eff.data() + static_cast<std::size_t>(l) * n_ao;
          for (int s = 0; s < n_ao; ++s) {
            t_row[s] += bv * d_row[s];
          }
        }
      }

      // Permute `[bc, n_ao, n_ao] -> [n_ao, bc, n_ao]` for cache-contiguous dot products in Step 2
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static)
      #endif
      for (int u = 0; u < n_ao; ++u) {
        for (int pl = 0; pl < bc; ++pl) {
          const std::size_t src_off =
              (static_cast<std::size_t>(pl) * n_ao + static_cast<std::size_t>(u)) * n_ao;
          const std::size_t dst_off =
              (static_cast<std::size_t>(u) * static_cast<std::size_t>(bc) +
               static_cast<std::size_t>(pl)) *
              n_ao;
          std::copy(b_c + src_off, b_c + src_off + n_ao, bp_chunk.data() + dst_off);
          std::copy(t_chunk.data() + src_off, t_chunk.data() + src_off + n_ao,
                    tp_chunk.data() + dst_off);
        }
      }

      // Step 2: K[u, v] += k_scale * sum_{q=0..k_c-1} tp_chunk[u, q] * bp_chunk[v, q]
      #if defined(_OPENMP)
      #pragma omp parallel for schedule(static)
      #endif
      for (int u = 0; u < n_ao; ++u) {
        const double* tu = tp_chunk.data() + static_cast<std::size_t>(u) * k_c;
        const int v_end = opts.symmetrize_dm ? (u + 1) : n_ao;
        for (int v = 0; v < v_end; ++v) {
          const double* bv = bp_chunk.data() + static_cast<std::size_t>(v) * k_c;
          double acc = 0.0;
          for (std::size_t q = 0; q < k_c; ++q) acc += tu[q] * bv[q];
          vk_out[u * n_ao + v] += opts.k_scale * acc;
        }
      }
    }

    if (opts.symmetrize_dm) {
      for (int u = 0; u < n_ao; ++u) {
        for (int v = 0; v < u; ++v) {
          vk_out[v * n_ao + u] = vk_out[u * n_ao + v];
        }
      }
    }
  }
}

std::vector<double> jk_nuclear_grad(const Basis& basis, std::span<const Atom> atoms,
                                    const double* dm, double k_fraction, Backend backend,
                                    double schwarz_cutoff) {
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }
  if (basis.max_l() > kMaxSupportedL) {
    throw NotImplemented("jk_nuclear_grad supports up to g shells (l <= 4)");
  }
  std::vector<double> grad(atoms.size() * 3, 0.0);
  if (atoms.empty() || basis.nao() == 0) return grad;
  if (dm == nullptr) throw std::invalid_argument("jk_nuclear_grad: dm is null");

  const int nsh = static_cast<int>(basis.nshell());
  std::vector<int> shell_cart_off(nsh, 0);
  std::vector<int> shell_atom(nsh, -1);
  int ncart_tot = 0;
  for (int i = 0; i < nsh; ++i) {
    shell_cart_off[i] = ncart_tot;
    ncart_tot += n_cart(basis.shell(i).l);
    const auto& sc = basis.shell(i).center;
    for (std::size_t a = 0; a < atoms.size(); ++a) {
      const double dx = sc[0] - atoms[a].center[0];
      const double dy = sc[1] - atoms[a].center[1];
      const double dz = sc[2] - atoms[a].center[2];
      if (dx * dx + dy * dy + dz * dz < 1e-12) {
        shell_atom[i] = static_cast<int>(a);
        break;
      }
    }
  }

  std::vector<double> d_cart, d_shell_max;
  double global_d_max = 0.0;
  prepare_cartesian_density(basis, shell_cart_off, ncart_tot, dm, d_cart, d_shell_max,
                            global_d_max, /*symmetrize_dm=*/true);
  if (global_d_max == 0.0) return grad;

  std::vector<ShellPairData> pair_lists;
  for (int la = 0; la <= basis.max_l(); ++la) {
    for (int lb = 0; lb <= la; ++lb) {
      ShellPairData pd = build_shell_pairs(basis, la, lb, 1e-14);
      if (pd.npair() > 0) pair_lists.push_back(std::move(pd));
    }
  }

  const double global_2pdm = 4.0 * global_d_max * global_d_max;
  std::vector<detail::GradQuartetTask> tasks;
  for (int bp = 0; bp < static_cast<int>(pair_lists.size()); ++bp) {
    const ShellPairData& bra = pair_lists[bp];
    for (int P = 0; P < bra.npair(); ++P) {
      const double q_ab = bra.schwarz_bound[P];
      const int i = bra.ish[P], j = bra.jsh[P];
      const double deg_ij = (i == j) ? 1.0 : 2.0;
      for (int kp = 0; kp <= bp; ++kp) {
        const ShellPairData& ket = pair_lists[kp];
        const int q_lim = (bp == kp) ? (P + 1) : ket.npair();
        for (int Q = 0; Q < q_lim; ++Q) {
          const double q_cd = ket.schwarz_bound[Q];
          if (q_ab * q_cd * global_2pdm < schwarz_cutoff) break;
          const int k = ket.ish[Q], l = ket.jsh[Q];
          const double d2 = std::max({4.0 * d_shell_max[i * nsh + j] * d_shell_max[k * nsh + l],
                                      d_shell_max[i * nsh + k] * d_shell_max[j * nsh + l],
                                      d_shell_max[i * nsh + l] * d_shell_max[j * nsh + k]});
          if (q_ab * q_cd * d2 < schwarz_cutoff) continue;

          const bool same_PQ = (bp == kp) && (P == Q);
          const double deg_kl = (k == l) ? 1.0 : 2.0;
          const double s_PQ = same_PQ ? 1.0 : 2.0;
          tasks.push_back(detail::GradQuartetTask{bp, P, kp, Q, deg_ij * deg_kl * s_PQ});
        }
      }
    }
  }
  if (tasks.empty()) return grad;

  if (backend == Backend::Host) {
    #if defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
      std::vector<double> loc(grad.size(), 0.0);
      #if defined(_OPENMP)
      #pragma omp for schedule(dynamic, 8)
      #endif
      for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(tasks.size()); ++idx) {
        const auto& t = tasks[idx];
        const ShellPairData& bra = pair_lists[t.bp];
        const ShellPairData& ket = pair_lists[t.kp];
        const int i = bra.ish[t.P], j = bra.jsh[t.P];
        const int k = ket.ish[t.Q], l = ket.jsh[t.Q];
        double g12[12];
        generated::eri_quartet_grad_direct_dm<double>(
            bra.la, bra.lb, ket.la, ket.lb, bra.view(), t.P, ket.view(), t.Q, d_cart.data(),
            ncart_tot, shell_cart_off[i], shell_cart_off[j], shell_cart_off[k], shell_cart_off[l],
            t.w_sym, k_fraction, g12);
        const int atoms4[4] = {shell_atom[i], shell_atom[j], shell_atom[k], shell_atom[l]};
        for (int c = 0; c < 4; ++c) {
          const int at = atoms4[c];
          if (at >= 0) {
            loc[3 * at + 0] += g12[3 * c + 0];
            loc[3 * at + 1] += g12[3 * c + 1];
            loc[3 * at + 2] += g12[3 * c + 2];
          }
        }
      }
      #if defined(_OPENMP)
      #pragma omp critical
      #endif
      {
        for (std::size_t m = 0; m < grad.size(); ++m) grad[m] += loc[m];
      }
    }
    return grad;
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::run_jk_nuclear_grad_cuda(pair_lists, tasks, shell_cart_off, shell_atom, ncart_tot,
                                     static_cast<int>(atoms.size()), d_cart, k_fraction, grad);
    return grad;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  throw std::runtime_error("gpu4dft was built without HIP");
}

std::vector<double> jk_nuclear_grad(const Basis& basis, const double* dm, double k_fraction,
                                    Backend backend, double schwarz_cutoff) {
  const std::vector<Atom> atoms = infer_atoms_from_basis(basis);
  return jk_nuclear_grad(basis, std::span<const Atom>(atoms), dm, k_fraction, backend,
                         schwarz_cutoff);
}

}  // namespace gpu4dft

// Clean C-ABI entry points so standalone Python/ctypes tests can exercise all
// 1e, Schwarz, DirectJKEngine, RIJKEngine, device-resident APIs, non-symmetric
// transition-density contractions, and nuclear-gradient APIs.
#if defined(__GNUC__) || defined(__clang__)
#define GPU4DFT_C_EXPORT extern "C" __attribute__((visibility("default")))
#else
#define GPU4DFT_C_EXPORT extern "C"
#endif

namespace {

gpu4dft::Basis build_basis_from_flat(int nshell, const int* shell_l, const double* shell_centers,
                                     const int* prim_offsets, const double* exponents,
                                     const double* coefficients) {
  std::vector<gpu4dft::Shell> shells;
  shells.reserve(nshell);
  for (int i = 0; i < nshell; ++i) {
    const int b = prim_offsets[i];
    const int e = prim_offsets[i + 1];
    shells.push_back(gpu4dft::Shell{
        shell_l[i],
        {shell_centers[3 * i + 0], shell_centers[3 * i + 1], shell_centers[3 * i + 2]},
        std::vector<double>(exponents + b, exponents + e),
        std::vector<double>(coefficients + b, coefficients + e),
    });
  }
  return gpu4dft::Basis(std::move(shells));
}

std::vector<gpu4dft::Atom> build_atoms_from_flat(int natom, const int* Z, const double* coords) {
  std::vector<gpu4dft::Atom> atoms;
  atoms.reserve(natom);
  for (int i = 0; i < natom; ++i) {
    atoms.push_back(gpu4dft::Atom{Z[i], {coords[3 * i + 0], coords[3 * i + 1], coords[3 * i + 2]}});
  }
  return atoms;
}

gpu4dft::Backend c_backend(int b) {
  if (b == 1) return gpu4dft::Backend::Cuda;
  if (b == 2) return gpu4dft::Backend::Hip;
  return gpu4dft::Backend::Host;
}

gpu4dft::Precision c_precision(int p) {
  if (p == 1) return gpu4dft::Precision::FP32;
  if (p == 2) return gpu4dft::Precision::Mixed;
  return gpu4dft::Precision::FP64;
}

}  // namespace

GPU4DFT_C_EXPORT int gpu4dft_c_one_electron(int op, int nshell, const int* shell_l,
                                            const double* shell_centers, const int* prim_offsets,
                                            const double* exponents, const double* coefficients,
                                            int natom, const int* atom_Z,
                                            const double* atom_coords, int backend, double* out) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto be = c_backend(backend);
    std::vector<double> res;
    if (op == 0) {
      res = gpu4dft::overlap_matrix(basis, be);
    } else if (op == 1) {
      res = gpu4dft::kinetic_matrix(basis, be);
    } else {
      const auto atoms = build_atoms_from_flat(natom, atom_Z, atom_coords);
      res = gpu4dft::nuclear_attraction_matrix(basis, atoms, be);
    }
    std::copy(res.begin(), res.end(), out);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_nuc_repulsion(int natom, const int* atom_Z,
                                             const double* atom_coords, double* energy_out,
                                             double* grad_out) {
  try {
    const auto atoms = build_atoms_from_flat(natom, atom_Z, atom_coords);
    if (energy_out) *energy_out = gpu4dft::nuclear_repulsion_energy(atoms);
    if (grad_out) {
      const auto g = gpu4dft::nuclear_repulsion_grad(atoms);
      std::copy(g.begin(), g.end(), grad_out);
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_one_elec_grad(int nshell, const int* shell_l,
                                             const double* shell_centers, const int* prim_offsets,
                                             const double* exponents, const double* coefficients,
                                             int natom, const int* atom_Z,
                                             const double* atom_coords, const double* dm,
                                             const double* wdm, int backend, double* grad_out) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto atoms = build_atoms_from_flat(natom, atom_Z, atom_coords);
    const auto g = gpu4dft::one_electron_grad(basis, atoms, dm, wdm, c_backend(backend));
    std::copy(g.begin(), g.end(), grad_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_direct_jk(int nshell, const int* shell_l,
                                         const double* shell_centers, const int* prim_offsets,
                                         const double* exponents, const double* coefficients,
                                         const double* dm, int backend, int precision,
                                         double schwarz_cutoff, double fp32_threshold,
                                         int compute_j, int compute_k, double k_scale,
                                         double* vj_out, double* vk_out) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    gpu4dft::DirectJKEngine eng(basis);
    gpu4dft::JKOptions opts;
    opts.backend = c_backend(backend);
    opts.precision = c_precision(precision);
    opts.schwarz_cutoff = schwarz_cutoff;
    opts.fp32_threshold = fp32_threshold;
    opts.compute_j = (compute_j != 0);
    opts.compute_k = (compute_k != 0);
    opts.k_scale = k_scale;
    eng.compute_jk(dm, vj_out, vk_out, opts);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_direct_jk_ex(
    int nshell, const int* shell_l, const double* shell_centers, const int* prim_offsets,
    const double* exponents, const double* coefficients, const double* dm, int backend,
    int precision, double schwarz_cutoff, double fp32_threshold, int compute_j, int compute_k,
    double k_scale, int symmetrize_dm, int use_device_api, int bench_iters, double* setup_ms_out,
    double* per_call_ms_out, double* vj_out, double* vk_out) {
  try {
    using Clock = std::chrono::steady_clock;
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto be = c_backend(backend);
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == gpu4dft::Backend::Cuda) cudaDeviceSynchronize();
#endif
    const auto t0 = Clock::now();
    gpu4dft::DirectJKEngine eng(basis);
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == gpu4dft::Backend::Cuda) cudaDeviceSynchronize();
#endif
    const auto t1 = Clock::now();
    if (setup_ms_out) {
      *setup_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    gpu4dft::JKOptions opts;
    opts.backend = be;
    opts.precision = c_precision(precision);
    opts.schwarz_cutoff = schwarz_cutoff;
    opts.fp32_threshold = fp32_threshold;
    opts.compute_j = (compute_j != 0);
    opts.compute_k = (compute_k != 0);
    opts.k_scale = k_scale;
    opts.symmetrize_dm = (symmetrize_dm != 0);

    const int iters = std::max(1, bench_iters);
#if defined(GPU4DFT_HAVE_CUDA)
    if (use_device_api != 0 && be == gpu4dft::Backend::Cuda) {
      const std::size_t n2 = static_cast<std::size_t>(eng.nao()) * eng.nao();
      double *d_dm = nullptr, *d_vj = nullptr, *d_vk = nullptr;
      if (cudaMalloc(&d_dm, n2 * sizeof(double)) != cudaSuccess) return -2;
      if (opts.compute_j && cudaMalloc(&d_vj, n2 * sizeof(double)) != cudaSuccess) {
        cudaFree(d_dm);
        return -2;
      }
      if (opts.compute_k && cudaMalloc(&d_vk, n2 * sizeof(double)) != cudaSuccess) {
        if (d_vj) cudaFree(d_vj);
        cudaFree(d_dm);
        return -2;
      }
      cudaMemcpy(d_dm, dm, n2 * sizeof(double), cudaMemcpyHostToDevice);
      // Warmup if benchmarking multiple calls
      if (iters > 1) {
        eng.compute_jk_device(d_dm, d_vj, d_vk, opts, nullptr);
        cudaDeviceSynchronize();
      }
      const auto tc0 = Clock::now();
      for (int it = 0; it < iters; ++it) {
        eng.compute_jk_device(d_dm, d_vj, d_vk, opts, nullptr);
      }
      cudaDeviceSynchronize();
      const auto tc1 = Clock::now();
      if (per_call_ms_out) {
        *per_call_ms_out = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / iters;
      }
      if (opts.compute_j && vj_out) {
        cudaMemcpy(vj_out, d_vj, n2 * sizeof(double), cudaMemcpyDeviceToHost);
      }
      if (opts.compute_k && vk_out) {
        cudaMemcpy(vk_out, d_vk, n2 * sizeof(double), cudaMemcpyDeviceToHost);
      }
      if (d_vk) cudaFree(d_vk);
      if (d_vj) cudaFree(d_vj);
      cudaFree(d_dm);
      return 0;
    }
#else
    (void)use_device_api;
#endif
    if (iters > 1) {
      eng.compute_jk(dm, vj_out, vk_out, opts);
    }
    const auto tc0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
      eng.compute_jk(dm, vj_out, vk_out, opts);
    }
    const auto tc1 = Clock::now();
    if (per_call_ms_out) {
      *per_call_ms_out = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / iters;
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_incremental_jk(
    int nshell, const int* shell_l, const double* shell_centers, const int* prim_offsets,
    const double* exponents, const double* coefficients, const double* dm_new,
    const double* dm_old, int backend, int precision, double schwarz_cutoff,
    double fp32_threshold, double* vj_accum, double* vk_accum) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    gpu4dft::DirectJKEngine eng(basis);
    gpu4dft::JKOptions opts;
    opts.backend = c_backend(backend);
    opts.precision = c_precision(precision);
    opts.schwarz_cutoff = schwarz_cutoff;
    opts.fp32_threshold = fp32_threshold;
    eng.compute_incremental_jk(dm_new, dm_old, vj_accum, vk_accum, opts);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_ri_jk(
    int nshell, const int* shell_l, const double* shell_centers, const int* prim_offsets,
    const double* exponents, const double* coefficients, int aux_nshell, const int* aux_l,
    const double* aux_centers, const int* aux_prim_offsets, const double* aux_exponents,
    const double* aux_coefficients, const double* dm, int backend, int precision, int compute_j,
    int compute_k, double k_scale, double* vj_out, double* vk_out) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto aux_basis = build_basis_from_flat(aux_nshell, aux_l, aux_centers, aux_prim_offsets,
                                                 aux_exponents, aux_coefficients);
    const auto be = c_backend(backend);
    gpu4dft::RIJKEngine eng(basis, aux_basis, be);
    gpu4dft::JKOptions opts;
    opts.backend = be;
    opts.precision = c_precision(precision);
    opts.compute_j = (compute_j != 0);
    opts.compute_k = (compute_k != 0);
    opts.k_scale = k_scale;
    eng.compute_jk(dm, vj_out, vk_out, opts);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_ri_jk_ex(
    int nshell, const int* shell_l, const double* shell_centers, const int* prim_offsets,
    const double* exponents, const double* coefficients, int aux_nshell, const int* aux_l,
    const double* aux_centers, const int* aux_prim_offsets, const double* aux_exponents,
    const double* aux_coefficients, const double* dm, int backend, int precision, int compute_j,
    int compute_k, double k_scale, int symmetrize_dm, int use_device_api, int bench_iters,
    double* setup_ms_out, double* per_call_ms_out, double* vj_out, double* vk_out) {
  try {
    using Clock = std::chrono::steady_clock;
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto aux_basis = build_basis_from_flat(aux_nshell, aux_l, aux_centers, aux_prim_offsets,
                                                 aux_exponents, aux_coefficients);
    const auto be = c_backend(backend);
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == gpu4dft::Backend::Cuda) cudaDeviceSynchronize();
#endif
    const auto t0 = Clock::now();
    gpu4dft::RIJKEngine eng(basis, aux_basis, be);
#if defined(GPU4DFT_HAVE_CUDA)
    if (be == gpu4dft::Backend::Cuda) cudaDeviceSynchronize();
#endif
    const auto t1 = Clock::now();
    if (setup_ms_out) {
      *setup_ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    gpu4dft::JKOptions opts;
    opts.backend = be;
    opts.precision = c_precision(precision);
    opts.compute_j = (compute_j != 0);
    opts.compute_k = (compute_k != 0);
    opts.k_scale = k_scale;
    opts.symmetrize_dm = (symmetrize_dm != 0);

    const int iters = std::max(1, bench_iters);
#if defined(GPU4DFT_HAVE_CUDA)
    if (use_device_api != 0 && be == gpu4dft::Backend::Cuda) {
      if (!eng.has_device_b() || eng.device_b_ptr() == nullptr) return -3;
      const std::size_t n2 = static_cast<std::size_t>(eng.nao()) * eng.nao();
      double *d_dm = nullptr, *d_vj = nullptr, *d_vk = nullptr;
      if (cudaMalloc(&d_dm, n2 * sizeof(double)) != cudaSuccess) return -2;
      if (opts.compute_j && cudaMalloc(&d_vj, n2 * sizeof(double)) != cudaSuccess) {
        cudaFree(d_dm);
        return -2;
      }
      if (opts.compute_k && cudaMalloc(&d_vk, n2 * sizeof(double)) != cudaSuccess) {
        if (d_vj) cudaFree(d_vj);
        cudaFree(d_dm);
        return -2;
      }
      cudaMemcpy(d_dm, dm, n2 * sizeof(double), cudaMemcpyHostToDevice);
      if (iters > 1) {
        eng.compute_jk_device(d_dm, d_vj, d_vk, opts, nullptr);
        cudaDeviceSynchronize();
      }
      const auto tc0 = Clock::now();
      for (int it = 0; it < iters; ++it) {
        eng.compute_jk_device(d_dm, d_vj, d_vk, opts, nullptr);
      }
      cudaDeviceSynchronize();
      const auto tc1 = Clock::now();
      if (per_call_ms_out) {
        *per_call_ms_out = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / iters;
      }
      if (opts.compute_j && vj_out) {
        cudaMemcpy(vj_out, d_vj, n2 * sizeof(double), cudaMemcpyDeviceToHost);
      }
      if (opts.compute_k && vk_out) {
        cudaMemcpy(vk_out, d_vk, n2 * sizeof(double), cudaMemcpyDeviceToHost);
      }
      if (d_vk) cudaFree(d_vk);
      if (d_vj) cudaFree(d_vj);
      cudaFree(d_dm);
      return 0;
    }
#else
    (void)use_device_api;
#endif
    if (iters > 1) {
      eng.compute_jk(dm, vj_out, vk_out, opts);
    }
    const auto tc0 = Clock::now();
    for (int it = 0; it < iters; ++it) {
      eng.compute_jk(dm, vj_out, vk_out, opts);
    }
    const auto tc1 = Clock::now();
    if (per_call_ms_out) {
      *per_call_ms_out = std::chrono::duration<double, std::milli>(tc1 - tc0).count() / iters;
    }
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_jk_grad(int nshell, const int* shell_l, const double* shell_centers,
                                       const int* prim_offsets, const double* exponents,
                                       const double* coefficients, const double* dm,
                                       double k_fraction, int backend, double schwarz_cutoff,
                                       double* grad_out, int* natom_out) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto g = gpu4dft::jk_nuclear_grad(basis, dm, k_fraction, c_backend(backend),
                                            schwarz_cutoff);
    if (natom_out) *natom_out = static_cast<int>(g.size() / 3);
    std::copy(g.begin(), g.end(), grad_out);
    return 0;
  } catch (...) {
    return -1;
  }
}

GPU4DFT_C_EXPORT int gpu4dft_c_schwarz_check(int nshell, const int* shell_l,
                                             const double* shell_centers, const int* prim_offsets,
                                             const double* exponents, const double* coefficients,
                                             int la, int lb, int* npair_out, double* bounds_out,
                                             int* ish_out, int* jsh_out) {
  try {
    const auto basis = build_basis_from_flat(nshell, shell_l, shell_centers, prim_offsets,
                                             exponents, coefficients);
    const auto pd = gpu4dft::build_shell_pairs(basis, la, lb);
    *npair_out = pd.npair();
    if (bounds_out) std::copy(pd.schwarz_bound.begin(), pd.schwarz_bound.end(), bounds_out);
    if (ish_out) std::copy(pd.ish.begin(), pd.ish.end(), ish_out);
    if (jsh_out) std::copy(pd.jsh.begin(), pd.jsh.end(), jsh_out);
    return 0;
  } catch (...) {
    return -1;
  }
}
