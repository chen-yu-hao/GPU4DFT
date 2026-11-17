#include "gpu4dft/jk_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu4dft/eri.hpp"
#include "gpu4dft/generated/eri_deriv1.hpp"

namespace gpu4dft {
namespace {

constexpr int kMaxSupportedL = 4;
constexpr int kTileKetSize = 8;

// Symmetrizes spherical density [nao x nao] and transforms to Cartesian
// [ncart_total x ncart_total]: D_cart = C^T D_sym C. Also computes per-shell
// maximum |D_sym| norms D_sh[ish * nshell + jsh] and global_d_max.
void prepare_cartesian_density(const Basis& basis, const std::vector<int>& shell_cart_off,
                               int ncart_tot, const double* dm, std::vector<double>& d_cart,
                               std::vector<double>& d_shell_max, double& global_d_max) {
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
          const double d_uv = 0.5 * (dm[u * nao + v] + dm[v * nao + u]);
          const double ad = std::abs(d_uv);
          if (ad > blk_max) blk_max = ad;
          for (int cj = 0; cj < ncj; ++cj) {
            tmp[mi * ncj + cj] += d_uv * Cj[mj * ncj + cj];
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

// Symmetrizes M_cart = 0.5 * (M_raw + M_raw^T) and transforms M_sph = scale * C * M_cart * C^T.
void finalize_cartesian_fock(const Basis& basis, const std::vector<int>& shell_cart_off,
                             int ncart_tot, const std::vector<double>& m_cart_raw, double scale,
                             double* m_sph_out) {
  const int nao = basis.nao();
  const int nsh = static_cast<int>(basis.nshell());
  std::vector<double> m_sym(static_cast<std::size_t>(ncart_tot) * ncart_tot, 0.0);
  for (int r = 0; r < ncart_tot; ++r) {
    for (int c = 0; c < ncart_tot; ++c) {
      m_sym[r * ncart_tot + c] =
          0.5 * (m_cart_raw[r * ncart_tot + c] + m_cart_raw[c * ncart_tot + r]);
    }
  }

  for (int i = 0; i < nsh; ++i) {
    const int li = basis.shell(i).l;
    const int nci = n_cart(li), nsi = n_sph(li);
    const int aoi = basis.ao_offset(i);
    const int ci0 = shell_cart_off[i];
    for (int j = 0; j <= i; ++j) {
      const int lj = basis.shell(j).l;
      const int ncj = n_cart(lj), nsj = n_sph(lj);
      const int aoj = basis.ao_offset(j);
      const int cj0 = shell_cart_off[j];

      double blk_cart[225];
      for (int a = 0; a < nci; ++a)
        for (int b = 0; b < ncj; ++b)
          blk_cart[a * ncj + b] = m_sym[(ci0 + a) * ncart_tot + (cj0 + b)];

      double blk_sph[81];
      harmonics::cart_to_sph_2d<double>(li, lj, blk_cart, blk_sph);

      for (int mi = 0; mi < nsi; ++mi) {
        for (int mj = 0; mj < nsj; ++mj) {
          const double v = scale * blk_sph[mi * nsj + mj];
          m_sph_out[(aoi + mi) * nao + (aoj + mj)] = v;
          m_sph_out[(aoj + mj) * nao + (aoi + mi)] = v;
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
    m = std::max(m, d_sh[i * nsh + k]);
    m = std::max(m, d_sh[i * nsh + l]);
    m = std::max(m, d_sh[j * nsh + k]);
    m = std::max(m, d_sh[j * nsh + l]);
  }
  return m;
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
    // Forward substitution L * B = eri_3c across all nao^2 columns in parallel.
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
  if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
  if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
  if ((!do_j && !do_k) || nao() == 0) return;

  std::vector<double> d_cart, d_shell_max;
  double global_d_max = 0.0;
  prepare_cartesian_density(basis_, shell_cart_offset_, ncart_total_, dm, d_cart, d_shell_max,
                            global_d_max);
  const double global_scale = (do_j ? 4.0 : 1.0) * global_d_max;
  if (global_scale == 0.0) return;

  // Build coarse Schwarz-screened task tiles (bra P, ket Q_begin..Q_end).
  std::vector<detail::JKTaskTile> tasks;
  for (int bp = 0; bp < static_cast<int>(pair_lists_.size()); ++bp) {
    const ShellPairData& bra = pair_lists_[bp];
    for (int P = 0; P < bra.npair(); ++P) {
      const double q_ab = bra.schwarz_bound[P];
      if (q_ab * q_ab * global_scale < opts.schwarz_cutoff * 1e-6) continue;
      for (int kp = 0; kp <= bp; ++kp) {
        const ShellPairData& ket = pair_lists_[kp];
        const int q_limit = (bp == kp) ? (P + 1) : ket.npair();
        for (int Q0 = 0; Q0 < q_limit; Q0 += kTileKetSize) {
          // Pairs in `ket` are sorted in descending `schwarz_bound` order!
          if (q_ab * ket.schwarz_bound[Q0] * global_scale < opts.schwarz_cutoff) {
            break;
          }
          const int Q1 = std::min(Q0 + kTileKetSize, q_limit);
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

  if (opts.backend == Backend::Host) {
    #if defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
      std::vector<double> j_loc(do_j ? nc2 : 0, 0.0);
      std::vector<double> k_loc(do_k ? nc2 : 0, 0.0);
      auto accum_j = [&](int r, int c, double val) { j_loc[r * ncart_total_ + c] += val; };
      auto accum_k = [&](int r, int c, double val) { k_loc[r * ncart_total_ + c] += val; };

      #if defined(_OPENMP)
      #pragma omp for schedule(dynamic, 4)
      #endif
      for (std::ptrdiff_t tid = 0; tid < static_cast<std::ptrdiff_t>(tasks.size()); ++tid) {
        const auto& t = tasks[tid];
        const ShellPairData& bra = pair_lists_[t.bp];
        const ShellPairData& ket = pair_lists_[t.kp];
        const ShellPairView bv = bra.view();
        const ShellPairView kv = ket.view();
        const int P = t.P;
        const int i = bra.ish[P], j = bra.jsh[P];
        const double deg_ij = (i == j) ? 1.0 : 2.0;
        const double q_ab = bra.schwarz_bound[P];
        const int off_a = shell_cart_offset_[i], off_b = shell_cart_offset_[j];

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
          const double w_K = 0.25 * deg_ij * deg_kl * s_PQ;
          const int off_c = shell_cart_offset_[k], off_d = shell_cart_offset_[l];

          const bool use_fp64 =
              (opts.precision == Precision::FP64) ||
              (opts.precision == Precision::Mixed && est >= opts.fp32_threshold);
          if (use_fp64) {
            detail::eval_direct_jk_quartet<double>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_total_, off_a, off_b, off_c, off_d, accum_j, accum_k);
          } else {
            detail::eval_direct_jk_quartet<float>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_total_, off_a, off_b, off_c, off_d, accum_j, accum_k);
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
  } else if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (!cuda_cache_) {
      cuda_cache_ = detail::create_direct_jk_cuda_cache(basis_, pair_lists_, shell_cart_offset_,
                                                        ncart_total_);
    }
    JKOptions eff_opts = opts;
    eff_opts.compute_j = do_j;
    eff_opts.compute_k = do_k;
    detail::run_direct_jk_cuda(*cuda_cache_, tasks, d_cart, d_shell_max, global_scale, nsh,
                               ncart_total_, eff_opts, j_cart_raw, k_cart_raw);
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  } else {
    throw std::runtime_error("gpu4dft was built without HIP");
  }

  if (do_j) {
    finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, j_cart_raw, 1.0, vj_out);
  }
  if (do_k) {
    finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, k_cart_raw, opts.k_scale,
                            vk_out);
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

  // 1. Auxiliary 1-shell pair lists for 2-center (P|Q) and 3-center (ab|P).
  std::vector<ShellPairData> aux_lists;
  for (int lP = 0; lP <= aux_basis_.max_l(); ++lP) {
    ShellPairData ap = build_aux_shell_pairs(aux_basis_, lP);
    if (ap.npair() > 0) aux_lists.push_back(std::move(ap));
  }

  std::vector<ShellPairData> orb_lists;
  for (int la = 0; la <= basis_.max_l(); ++la) {
    for (int lb = 0; lb <= la; ++lb) {
      ShellPairData op = build_shell_pairs(basis_, la, lb);
      if (op.npair() > 0) orb_lists.push_back(std::move(op));
    }
  }

  // 2. Assemble 2-center Coulomb metric V_{PQ} = (P | Q) [n_aux x n_aux].
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
          eri_pair_block(bra_aux, ket_aux, build_backend, Precision::FP64);

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

  // 3. Assemble 3-center integrals I_{P, uv} = (uv | P) [n_aux x nao^2].
  std::vector<double> eri_3c(static_cast<std::size_t>(n_aux) * nao2, 0.0);
  for (const ShellPairData& orb : orb_lists) {
    const int nsa = n_sph(orb.la), nsb = n_sph(orb.lb);
    for (const ShellPairData& aux : aux_lists) {
      const int nsP = n_sph(aux.la);
      const std::size_t nsph = static_cast<std::size_t>(nsa) * nsb * nsP;
      const std::size_t nket = static_cast<std::size_t>(aux.npair());
      const std::vector<double> blk = eri_pair_block(orb, aux, build_backend, Precision::FP64);

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

  // 4. Solve Cholesky / symmetric inverse B_{P, uv} = V^{-1/2} (uv | P).
  solve_ri_metric(n_aux, nao2, metric, eri_3c, b_tensor_);

  // Pre-permute B_perm[u, P * nao + v] = B[P, u, v] for 2-GEMM RI-K contraction.
  b_perm_tensor_.assign(static_cast<std::size_t>(n_ao) * n_aux * n_ao, 0.0);
  for (int p = 0; p < n_aux; ++p) {
    for (int u = 0; u < n_ao; ++u) {
      for (int v = 0; v < n_ao; ++v) {
        b_perm_tensor_[(static_cast<std::size_t>(u) * n_aux + p) * n_ao + v] =
            b_tensor_[(static_cast<std::size_t>(p) * n_ao + u) * n_ao + v];
      }
    }
  }
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
  if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
  if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
  if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) return;

  std::vector<double> d_sym(nao2);
  for (int u = 0; u < n_ao; ++u)
    for (int v = 0; v < n_ao; ++v)
      d_sym[u * n_ao + v] = 0.5 * (dm[u * n_ao + v] + dm[v * n_ao + u]);

  if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if (!cuda_cache_) {
      cuda_cache_ = detail::create_rijk_cuda_cache(n_ao, n_aux, b_tensor_, b_perm_tensor_);
    }
    detail::run_rijk_cuda(*cuda_cache_, d_sym.data(), do_j ? vj_out : nullptr,
                          do_k ? vk_out : nullptr, opts);
    return;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  if (opts.backend != Backend::Host) {
    throw std::runtime_error("gpu4dft was built without HIP");
  }

  // Host OpenMP GEMV / GEMM contraction.
  if (do_j) {
    std::vector<double> v_aux(n_aux, 0.0);
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int p = 0; p < n_aux; ++p) {
      const double* bp = b_tensor_.data() + static_cast<std::size_t>(p) * nao2;
      double acc = 0.0;
      for (std::size_t uv = 0; uv < nao2; ++uv) acc += bp[uv] * d_sym[uv];
      v_aux[p] = acc;
    }

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (std::ptrdiff_t uv = 0; uv < static_cast<std::ptrdiff_t>(nao2); ++uv) {
      double acc = 0.0;
      for (int p = 0; p < n_aux; ++p) {
        acc += b_tensor_[static_cast<std::size_t>(p) * nao2 + uv] * v_aux[p];
      }
      vj_out[uv] = acc;
    }
  }

  if (do_k) {
    // GEMM 1: T_perm[u, P * nao + s] = sum_v B_perm[u, P * nao + v] * D[v, s].
    const std::size_t k_big = static_cast<std::size_t>(n_aux) * n_ao;
    std::vector<double> t_perm(static_cast<std::size_t>(n_ao) * k_big, 0.0);

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int up = 0; up < n_ao * n_aux; ++up) {
      const double* b_row = b_perm_tensor_.data() + static_cast<std::size_t>(up) * n_ao;
      double* t_row = t_perm.data() + static_cast<std::size_t>(up) * n_ao;
      for (int v = 0; v < n_ao; ++v) {
        const double b_val = b_row[v];
        const double* d_row = d_sym.data() + static_cast<std::size_t>(v) * n_ao;
        for (int s = 0; s < n_ao; ++s) {
          t_row[s] += b_val * d_row[s];
        }
      }
    }

    // GEMM 2: K[u, l] = k_scale * sum_{q=0}^{n_aux*nao-1} T_perm[u, q] * B_perm[l, q].
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int u = 0; u < n_ao; ++u) {
      const double* tu = t_perm.data() + static_cast<std::size_t>(u) * k_big;
      for (int l = 0; l <= u; ++l) {
        const double* bl = b_perm_tensor_.data() + static_cast<std::size_t>(l) * k_big;
        double acc = 0.0;
        for (std::size_t q = 0; q < k_big; ++q) acc += tu[q] * bl[q];
        const double val = opts.k_scale * acc;
        vk_out[u * n_ao + l] = val;
        vk_out[l * n_ao + u] = val;
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
                            global_d_max);
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
// 1e, Schwarz, DirectJKEngine, RIJKEngine, and nuclear-gradient APIs even prior
// to nanobind wrapper registration.
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
