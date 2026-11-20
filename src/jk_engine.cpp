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
#include "gpu4dft/generated/eri_general.hpp"
#include "gpu4dft/linalg.hpp"

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
                             double* m_sph_out, bool symmetrize_out = true,
                             bool accumulate = false) {
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

double quartet_d_max_uhf(const double* d_sh_tot, const double* d_sh_spin, int nsh, int i, int j,
                         int k, int l, bool do_j, bool do_k) {
  double m = 0.0;
  if (do_j) {
    m = std::max(m, 4.0 * d_sh_tot[i * nsh + j]);
    m = std::max(m, 4.0 * d_sh_tot[k * nsh + l]);
  }
  if (do_k) {
    m = std::max(m, 2.0 * d_sh_spin[i * nsh + k]);
    m = std::max(m, 2.0 * d_sh_spin[i * nsh + l]);
    m = std::max(m, 2.0 * d_sh_spin[j * nsh + k]);
    m = std::max(m, 2.0 * d_sh_spin[j * nsh + l]);
  }
  return m;
}

void eval_host_direct_jk_uhf_tiles(
    const std::vector<ShellPairData>& pair_lists, const std::vector<detail::JKTaskTile>& tasks,
    const std::vector<int>& shell_cart_off, const std::vector<double>& d_tot_cart,
    const std::vector<double>& d_a_cart, const std::vector<double>& d_b_cart,
    const std::vector<double>& d_sh_tot, const std::vector<double>& d_sh_spin, double global_scale,
    int nsh, int ncart_tot, bool do_j, bool do_ka, bool do_kb, const JKOptions& opts,
    std::vector<double>& j_cart_raw, std::vector<double>& ka_cart_raw,
    std::vector<double>& kb_cart_raw) {
  const std::size_t nc2 = static_cast<std::size_t>(ncart_tot) * ncart_tot;
  const bool do_k = do_ka || do_kb;
  #if defined(_OPENMP)
  #pragma omp parallel
  #endif
  {
    std::vector<double> j_loc(do_j ? nc2 : 0, 0.0);
    std::vector<double> ka_loc(do_ka ? nc2 : 0, 0.0);
    std::vector<double> kb_loc(do_kb ? nc2 : 0, 0.0);
    auto accum_j_ket = [&](int r, int c, double val) { j_loc[r * ncart_tot + c] += val; };
    auto accum_ka = [&](int r, int c, double val) { ka_loc[r * ncart_tot + c] += val; };
    auto accum_kb = [&](int r, int c, double val) { kb_loc[r * ncart_tot + c] += val; };

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
        const double dmax =
            quartet_d_max_uhf(d_sh_tot.data(), d_sh_spin.data(), nsh, i, j, k, l, do_j, do_k);
        const double est = q_ab * q_cd * dmax;
        if (est < opts.schwarz_cutoff) continue;

        const bool same_PQ = (t.bp == t.kp) && (P == Q);
        const double deg_kl = (k == l) ? 1.0 : 2.0;
        const double s_PQ = same_PQ ? 1.0 : 2.0;
        const double w_J = deg_ij * deg_kl;
        const double w_K = 0.25 * deg_ij * deg_kl * s_PQ;
        const int off_c = shell_cart_off[k], off_d = shell_cart_off[l];

        const bool use_fp64 =
            (opts.precision == Precision::FP64) ||
            (opts.precision == Precision::Mixed && est >= opts.fp32_threshold);
        if (max_l <= 1) {
          if (use_fp64) {
            detail::eval_rys_jk_quartet_tiered_uhf<1, double>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_ka, do_kb, w_J,
                w_K, d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(), ncart_tot, off_a, off_b,
                off_c, off_d, accum_j_bra, accum_j_ket, accum_ka, accum_kb, opts.omega);
          } else {
            detail::eval_rys_jk_quartet_tiered_uhf<1, float>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_ka, do_kb, w_J,
                w_K, d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(), ncart_tot, off_a, off_b,
                off_c, off_d, accum_j_bra, accum_j_ket, accum_ka, accum_kb, opts.omega);
          }
        } else if (max_l == 2) {
          if (use_fp64) {
            detail::eval_rys_jk_quartet_tiered_uhf<2, double>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_ka, do_kb, w_J,
                w_K, d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(), ncart_tot, off_a, off_b,
                off_c, off_d, accum_j_bra, accum_j_ket, accum_ka, accum_kb, opts.omega);
          } else {
            detail::eval_rys_jk_quartet_tiered_uhf<2, float>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_ka, do_kb, w_J,
                w_K, d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(), ncart_tot, off_a, off_b,
                off_c, off_d, accum_j_bra, accum_j_ket, accum_ka, accum_kb, opts.omega);
          }
        } else {
          if (use_fp64) {
            detail::eval_rys_jk_quartet_tiered_uhf<4, double>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_ka, do_kb, w_J,
                w_K, d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(), ncart_tot, off_a, off_b,
                off_c, off_d, accum_j_bra, accum_j_ket, accum_ka, accum_kb, opts.omega);
          } else {
            detail::eval_rys_jk_quartet_tiered_uhf<4, float>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_ka, do_kb, w_J,
                w_K, d_tot_cart.data(), d_a_cart.data(), d_b_cart.data(), ncart_tot, off_a, off_b,
                off_c, off_d, accum_j_bra, accum_j_ket, accum_ka, accum_kb, opts.omega);
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
      if (do_ka) {
        for (std::size_t idx = 0; idx < nc2; ++idx) ka_cart_raw[idx] += ka_loc[idx];
      }
      if (do_kb) {
        for (std::size_t idx = 0; idx < nc2; ++idx) kb_cart_raw[idx] += kb_loc[idx];
      }
    }
  }
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
                accum_k, opts.omega);
          } else {
            detail::eval_direct_jk_quartet_tiered<1, float, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k, opts.omega);
          }
        } else if (max_l == 2) {
          if (use_fp64) {
            detail::eval_direct_jk_quartet_tiered<2, double, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k, opts.omega);
          } else {
            detail::eval_direct_jk_quartet_tiered<2, float, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k, opts.omega);
          }
        } else {
          if (use_fp64) {
            detail::eval_direct_jk_quartet_tiered<4, double, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k, opts.omega);
          } else {
            detail::eval_direct_jk_quartet_tiered<4, float, SymmetrizeDM>(
                bra.la, bra.lb, ket.la, ket.lb, bv, P, kv, Q, same_PQ, do_j, do_k, w_J, w_K,
                d_cart.data(), ncart_tot, off_a, off_b, off_c, off_d, accum_j_bra, accum_j_ket,
                accum_k, opts.omega);
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

std::vector<RIAuxChunk> build_ri_aux_chunks(const Basis& aux_basis,
                                            const std::vector<ShellPairData>& aux_lists,
                                            const std::vector<int>& aux_shell_cart_off,
                                            int max_sph_width, int max_cart_width) {
  std::vector<RIAuxChunk> chunks;
  const int nsh_aux = static_cast<int>(aux_basis.nshell());
  if (nsh_aux == 0) return chunks;
  const int num_l = static_cast<int>(aux_lists.size());

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
      if (l < num_l && l_cursor[l] < aux_lists[l].npair() && aux_lists[l].ish[l_cursor[l]] == s) {
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

void factorize_host_ri_metric(int naux, const std::vector<ShellPairData>& aux_lists, double omega,
                              RIHostMetric& m_out) {
  m_out.built = true;
  if (naux == 0) {
    m_out.is_cholesky = true;
    m_out.factor.clear();
    return;
  }

  std::vector<double> metric(static_cast<std::size_t>(naux) * naux, 0.0);
  for (std::size_t ip = 0; ip < aux_lists.size(); ++ip) {
    const ShellPairData& bra_aux = aux_lists[ip];
    if (bra_aux.npair() == 0) continue;
    const int nsP = n_sph(bra_aux.la);
    for (std::size_t iq = 0; iq <= ip; ++iq) {
      const ShellPairData& ket_aux = aux_lists[iq];
      if (ket_aux.npair() == 0) continue;
      const int nsQ = n_sph(ket_aux.la);
      const std::size_t nsph = static_cast<std::size_t>(nsP) * nsQ;
      const std::size_t nket = static_cast<std::size_t>(ket_aux.npair());
      const std::vector<double> blk =
          eri_pair_block(bra_aux, ket_aux, Backend::Host, Precision::FP64, omega);
      for (int P = 0; P < bra_aux.npair(); ++P) {
        const int p0 = bra_aux.ao_i[P];
        for (int Q = 0; Q < ket_aux.npair(); ++Q) {
          const int q0 = ket_aux.ao_i[Q];
          const double* ptr = blk.data() + (static_cast<std::size_t>(P) * nket + Q) * nsph;
          for (int mp = 0; mp < nsP; ++mp) {
            for (int mq = 0; mq < nsQ; ++mq) {
              const double v = ptr[mp * nsQ + mq];
              metric[static_cast<std::size_t>(p0 + mp) * naux + (q0 + mq)] = v;
              metric[static_cast<std::size_t>(q0 + mq) * naux + (p0 + mp)] = v;
            }
          }
        }
      }
    }
  }

  std::vector<double> L = metric;
  bool chol_ok = true;
  for (int i = 0; i < naux; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = L[static_cast<std::size_t>(i) * naux + j];
      for (int k = 0; k < j; ++k) {
        sum -= L[static_cast<std::size_t>(i) * naux + k] *
               L[static_cast<std::size_t>(j) * naux + k];
      }
      if (i == j) {
        if (!(sum > 1e-11)) {
          chol_ok = false;
          break;
        }
        L[static_cast<std::size_t>(i) * naux + i] = std::sqrt(sum);
      } else {
        L[static_cast<std::size_t>(i) * naux + j] =
            sum / L[static_cast<std::size_t>(j) * naux + j];
      }
    }
    if (!chol_ok) break;
    for (int j = i + 1; j < naux; ++j) {
      L[static_cast<std::size_t>(i) * naux + j] = 0.0;
    }
  }

  if (chol_ok) {
    m_out.is_cholesky = true;
    m_out.factor = std::move(L);
    return;
  }

  std::vector<double> w(static_cast<std::size_t>(naux), 0.0);
  std::vector<double> U(static_cast<std::size_t>(naux) * naux, 0.0);
  syevd(naux, metric.data(), w.data(), U.data(), Backend::Host);

  std::vector<double> inv_sqrt(static_cast<std::size_t>(naux) * naux, 0.0);
  #if defined(_OPENMP)
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < naux; ++i) {
    for (int k = 0; k < naux; ++k) {
      const double ev = w[static_cast<std::size_t>(k)];
      if (ev > 1e-10) {
        const double sc = U[static_cast<std::size_t>(i) * naux + k] / std::sqrt(ev);
        for (int j = 0; j < naux; ++j) {
          inv_sqrt[static_cast<std::size_t>(i) * naux + j] +=
              sc * U[static_cast<std::size_t>(j) * naux + k];
        }
      }
    }
  }
  m_out.is_cholesky = false;
  m_out.factor = std::move(inv_sqrt);
}

// Solves L * X = Y in-place (if is_cholesky) for [naux x ncols] row-major Y.
void solve_host_L_forward_inplace(int naux, int ncols, const double* L, double* Y) {
  #if defined(_OPENMP)
  #pragma omp parallel for schedule(static) if (ncols >= 64)
  #endif
  for (int c = 0; c < ncols; ++c) {
    for (int i = 0; i < naux; ++i) {
      double sum = Y[static_cast<std::size_t>(i) * ncols + c];
      const double* L_row = L + static_cast<std::size_t>(i) * naux;
      for (int k = 0; k < i; ++k) {
        sum -= L_row[k] * Y[static_cast<std::size_t>(k) * ncols + c];
      }
      Y[static_cast<std::size_t>(i) * ncols + c] = sum / L_row[i];
    }
  }
}

// Solves V * X = Y in-place (V = L L^T or V^+ = V^{-1/2} V^{-1/2}) for [naux x ncols] row-major.
void solve_host_V_inplace(int naux, int ncols, const RIHostMetric& m, double* Y) {
  if (naux == 0 || ncols == 0) return;
  if (m.is_cholesky) {
    const double* L = m.factor.data();
    #if defined(_OPENMP)
    #pragma omp parallel for schedule(static) if (ncols >= 8)
    #endif
    for (int c = 0; c < ncols; ++c) {
      for (int i = 0; i < naux; ++i) {
        double sum = Y[static_cast<std::size_t>(i) * ncols + c];
        const double* L_row = L + static_cast<std::size_t>(i) * naux;
        for (int k = 0; k < i; ++k) {
          sum -= L_row[k] * Y[static_cast<std::size_t>(k) * ncols + c];
        }
        Y[static_cast<std::size_t>(i) * ncols + c] = sum / L_row[i];
      }
      for (int i = naux - 1; i >= 0; --i) {
        double sum = Y[static_cast<std::size_t>(i) * ncols + c];
        for (int k = i + 1; k < naux; ++k) {
          sum -= L[static_cast<std::size_t>(k) * naux + i] *
                 Y[static_cast<std::size_t>(k) * ncols + c];
        }
        Y[static_cast<std::size_t>(i) * ncols + c] =
            sum / L[static_cast<std::size_t>(i) * naux + i];
      }
    }
  } else {
    std::vector<double> tmp(static_cast<std::size_t>(naux) * ncols, 0.0);
    gemm(naux, ncols, naux, 1.0, m.factor.data(), naux, false, Y, ncols, false, 0.0, tmp.data(),
         ncols, Backend::Host);
    gemm(naux, ncols, naux, 1.0, m.factor.data(), naux, false, tmp.data(), ncols, false, 0.0, Y,
         ncols, Backend::Host);
  }
}

void eval_host_ri_3c_chunk(int nao, const std::vector<ShellPairData>& orb_lists,
                           const std::vector<ShellPairData>& aux_lists, const RIAuxChunk& chunk,
                           double omega, double* I_chunk) {
  const int bc = chunk.q1 - chunk.q0;
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  std::fill(I_chunk, I_chunk + static_cast<std::size_t>(bc) * nao2, 0.0);

  std::vector<std::pair<int, int>> orb_pairs;
  for (int io = 0; io < static_cast<int>(orb_lists.size()); ++io) {
    for (int ab = 0; ab < orb_lists[io].npair(); ++ab) {
      orb_pairs.emplace_back(io, ab);
    }
  }

  #if defined(_OPENMP)
  #pragma omp parallel for schedule(dynamic, 16)
  #endif
  for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(orb_pairs.size()); ++idx) {
    const int io = orb_pairs[idx].first;
    const int ab = orb_pairs[idx].second;
    const ShellPairData& orb = orb_lists[io];
    const ShellPairView ov = orb.view();
    const int nsa = n_sph(orb.la), nsb = n_sph(orb.lb);
    const int u0 = orb.ao_i[ab], v0 = orb.ao_j[ab];

    double sph[9 * 9 * 9];
    for (int lP = 0; lP < static_cast<int>(aux_lists.size()); ++lP) {
      const int pb = chunk.l_beg[lP];
      const int pe = chunk.l_end[lP];
      if (pb >= pe) continue;
      const ShellPairData& aux = aux_lists[lP];
      const ShellPairView av = aux.view();
      const int nsP = n_sph(lP);

      for (int P = pb; P < pe; ++P) {
        const int p_loc = aux.ao_i[P] - chunk.q0;
        generated::eri_general_eval<double>(orb.la, orb.lb, lP, 0, ov, ab, av, P, sph, omega);
        for (int mp = 0; mp < nsP; ++mp) {
          double* row = I_chunk + static_cast<std::size_t>(p_loc + mp) * nao2;
          for (int ma = 0; ma < nsa; ++ma) {
            const int u = u0 + ma;
            for (int mb = 0; mb < nsb; ++mb) {
              const int v = v0 + mb;
              const double val = sph[(ma * nsb + mb) * nsP + mp];
              row[static_cast<std::size_t>(u) * nao + v] = val;
              row[static_cast<std::size_t>(v) * nao + u] = val;
            }
          }
        }
      }
    }
  }
}

void build_host_b_from_metric(int nao, int naux, const Basis& aux_basis,
                              const std::vector<ShellPairData>& orb_lists,
                              const std::vector<ShellPairData>& aux_lists,
                              const std::vector<int>& aux_shell_cart_off, const RIHostMetric& m,
                              double omega, std::vector<double>& b_out) {
  const std::size_t nao2 = static_cast<std::size_t>(nao) * nao;
  b_out.assign(static_cast<std::size_t>(naux) * nao2, 0.0);
  if (nao == 0 || naux == 0) return;

  const auto chunks = build_ri_aux_chunks(aux_basis, aux_lists, aux_shell_cart_off, 64, 256);
  if (m.is_cholesky) {
    std::vector<double> eri_3c(static_cast<std::size_t>(naux) * nao2, 0.0);
    for (const auto& c : chunks) {
      eval_host_ri_3c_chunk(nao, orb_lists, aux_lists, c, omega,
                            eri_3c.data() + static_cast<std::size_t>(c.q0) * nao2);
    }
    b_out = std::move(eri_3c);
    solve_host_L_forward_inplace(naux, static_cast<int>(nao2), m.factor.data(), b_out.data());
  } else {
    std::vector<double> I_chunk;
    for (const auto& c : chunks) {
      const int bc = c.q1 - c.q0;
      I_chunk.resize(static_cast<std::size_t>(bc) * nao2);
      eval_host_ri_3c_chunk(nao, orb_lists, aux_lists, c, omega, I_chunk.data());
      gemm(naux, static_cast<int>(nao2), bc, 1.0, m.factor.data() + c.q0, naux, false,
           I_chunk.data(), static_cast<int>(nao2), false, 1.0, b_out.data(),
           static_cast<int>(nao2), Backend::Host);
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
  if (global_d_max == 0.0) return;

  auto build_tiles = [&](double gscale, double s_cutoff) {
    std::vector<detail::JKTaskTile> out;
    if (gscale <= 0.0) return out;
    for (int bp = 0; bp < static_cast<int>(pair_lists_.size()); ++bp) {
      const ShellPairData& bra = pair_lists_[bp];
      for (int P = 0; P < bra.npair(); ++P) {
        const double q_ab = bra.schwarz_bound[P];
        if (q_ab * q_ab * gscale < s_cutoff * 1e-6) continue;
        for (int kp = 0; kp <= bp; ++kp) {
          const ShellPairData& ket = pair_lists_[kp];
          const int q_limit = (bp == kp) ? (P + 1) : ket.npair();
          for (int Q0 = 0; Q0 < q_limit; Q0 += kHostTileKetSize) {
            if (q_ab * ket.schwarz_bound[Q0] * gscale < s_cutoff) break;
            const int Q1 = std::min(Q0 + kHostTileKetSize, q_limit);
            out.push_back(detail::JKTaskTile{bp, P, kp, Q0, Q1});
          }
        }
      }
    }
    return out;
  };

  const int nsh = static_cast<int>(basis_.nshell());
  const std::size_t nc2 = static_cast<std::size_t>(ncart_total_) * ncart_total_;
  std::vector<double> j_cart_raw;
  std::vector<double> k_cart_raw;

  auto run_pass = [&](bool pass_j, bool pass_k, double pass_omega, double k_coeff,
                      bool acc_k) {
    const double k_amp = pass_k ? std::max(1.0, std::abs(k_coeff)) : 1.0;
    JKOptions pass_opts = opts;
    pass_opts.omega = pass_omega;
    pass_opts.schwarz_cutoff = opts.schwarz_cutoff / k_amp;
    pass_opts.fp32_threshold = opts.fp32_threshold / k_amp;
    const double global_scale = (pass_j ? 4.0 : 2.0) * global_d_max;
    const auto tasks = build_tiles(global_scale, pass_opts.schwarz_cutoff);
    if (tasks.empty()) return;

    if (pass_j) j_cart_raw.assign(nc2, 0.0);
    if (pass_k) k_cart_raw.assign(nc2, 0.0);
    if (eff_sym_dm) {
      eval_host_direct_jk_tiles<true>(pair_lists_, tasks, shell_cart_offset_, d_cart, d_shell_max,
                                      global_scale, nsh, ncart_total_, pass_j, pass_k, pass_opts,
                                      j_cart_raw, k_cart_raw);
    } else {
      eval_host_direct_jk_tiles<false>(pair_lists_, tasks, shell_cart_offset_, d_cart, d_shell_max,
                                       global_scale, nsh, ncart_total_, pass_j, pass_k, pass_opts,
                                       j_cart_raw, k_cart_raw);
    }
    if (pass_j) {
      finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, j_cart_raw, 1.0, vj_out,
                              /*symmetrize_out=*/true, /*accumulate=*/false);
    }
    if (pass_k) {
      finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, k_cart_raw, k_coeff,
                              vk_out, /*symmetrize_out=*/opts.symmetrize_dm, /*accumulate=*/acc_k);
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

void DirectJKEngine::compute_jk_uhf_device(const double* d_dm_alpha, const double* d_dm_beta,
                                           double* d_vj_tot, double* d_vk_alpha,
                                           double* d_vk_beta, const JKOptions& opts,
                                           void* stream) const {
  if (!backend_available(Backend::Cuda)) {
    throw std::runtime_error(
        "DirectJKEngine::compute_jk_uhf_device: backend 'cuda' is not available");
  }
  if (d_dm_alpha == nullptr || d_dm_beta == nullptr) {
    throw std::invalid_argument("DirectJKEngine::compute_jk_uhf_device: spin DM pointer is null");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  const bool do_j = opts.compute_j && (d_vj_tot != nullptr);
  const bool do_ka = opts.compute_k && (d_vk_alpha != nullptr);
  const bool do_kb = opts.compute_k && (d_vk_beta != nullptr);
  if ((!do_j && !do_ka && !do_kb) || nao() == 0) return;

  if (!cuda_cache_) {
    cuda_cache_ = detail::create_direct_jk_cuda_cache(basis_, pair_lists_, shell_cart_offset_,
                                                      ncart_total_);
  }
  JKOptions eff_opts = opts;
  eff_opts.backend = Backend::Cuda;
  eff_opts.compute_j = do_j;
  eff_opts.compute_k = do_ka || do_kb;
  detail::run_direct_jk_uhf_cuda_device(*cuda_cache_, d_dm_alpha, d_dm_beta,
                                        do_j ? d_vj_tot : nullptr,
                                        do_ka ? d_vk_alpha : nullptr,
                                        do_kb ? d_vk_beta : nullptr, eff_opts, stream);
#else
  (void)d_vj_tot;
  (void)d_vk_alpha;
  (void)d_vk_beta;
  (void)opts;
  (void)stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void DirectJKEngine::compute_jk_uhf(const double* dm_alpha, const double* dm_beta,
                                    double* vj_tot_out, double* vk_alpha_out, double* vk_beta_out,
                                    const JKOptions& opts) const {
  if (!backend_available(opts.backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(opts.backend) +
                             "' is not available");
  }
  if (dm_alpha == nullptr || dm_beta == nullptr) {
    throw std::invalid_argument("DirectJKEngine::compute_jk_uhf: spin DM pointer is null");
  }

  const bool do_j = opts.compute_j && (vj_tot_out != nullptr);
  const bool do_ka = opts.compute_k && (vk_alpha_out != nullptr);
  const bool do_kb = opts.compute_k && (vk_beta_out != nullptr);
  const bool do_k = do_ka || do_kb;
  const std::size_t nao2 = static_cast<std::size_t>(nao()) * nao();

  if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if ((!do_j && !do_k) || nao() == 0) {
      if (do_j) std::fill(vj_tot_out, vj_tot_out + nao2, 0.0);
      if (do_ka) std::fill(vk_alpha_out, vk_alpha_out + nao2, 0.0);
      if (do_kb) std::fill(vk_beta_out, vk_beta_out + nao2, 0.0);
      return;
    }
    if (!cuda_cache_) {
      cuda_cache_ = detail::create_direct_jk_cuda_cache(basis_, pair_lists_, shell_cart_offset_,
                                                        ncart_total_);
    }
    JKOptions eff_opts = opts;
    eff_opts.compute_j = do_j;
    eff_opts.compute_k = do_k;
    detail::run_direct_jk_uhf_cuda_host(*cuda_cache_, dm_alpha, dm_beta,
                                        do_j ? vj_tot_out : nullptr,
                                        do_ka ? vk_alpha_out : nullptr,
                                        do_kb ? vk_beta_out : nullptr, eff_opts);
    return;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  if (opts.backend != Backend::Host) {
    throw std::runtime_error("gpu4dft was built without HIP");
  }

  if (do_j) std::fill(vj_tot_out, vj_tot_out + nao2, 0.0);
  if (do_ka) std::fill(vk_alpha_out, vk_alpha_out + nao2, 0.0);
  if (do_kb) std::fill(vk_beta_out, vk_beta_out + nao2, 0.0);
  if ((!do_j && !do_k) || nao() == 0) return;

  std::vector<double> dm_tot(nao2, 0.0);
  for (std::size_t i = 0; i < nao2; ++i) dm_tot[i] = dm_alpha[i] + dm_beta[i];

  // Pure-J path: delegate directly to Hermite J-engine!
  if (!do_k) {
    JKOptions j_opts = opts;
    j_opts.compute_j = true;
    j_opts.compute_k = false;
    if (opts.rsh_mode) {
      j_opts.rsh_mode = false;
      j_opts.omega = 0.0;
    }
    compute_jk(dm_tot.data(), vj_tot_out, nullptr, j_opts);
    return;
  }

  // Non-symmetric fallback if requested
  if (!opts.symmetrize_dm) {
    if (do_j) {
      JKOptions j_opts = opts;
      j_opts.compute_j = true;
      j_opts.compute_k = false;
      if (opts.rsh_mode) {
        j_opts.rsh_mode = false;
        j_opts.omega = 0.0;
      }
      compute_jk(dm_tot.data(), vj_tot_out, nullptr, j_opts);
    }
    JKOptions k_opts = opts;
    k_opts.compute_j = false;
    k_opts.compute_k = true;
    if (do_ka) compute_jk(dm_alpha, nullptr, vk_alpha_out, k_opts);
    if (do_kb) compute_jk(dm_beta, nullptr, vk_beta_out, k_opts);
    return;
  }

  std::vector<double> d_tot_cart, d_a_cart, d_b_cart;
  std::vector<double> d_sh_tot, d_sh_a, d_sh_b;
  double g_tot = 0.0, g_a = 0.0, g_b = 0.0;
  prepare_cartesian_density(basis_, shell_cart_offset_, ncart_total_, dm_tot.data(), d_tot_cart,
                            d_sh_tot, g_tot, /*symmetrize_dm=*/true);
  prepare_cartesian_density(basis_, shell_cart_offset_, ncart_total_, dm_alpha, d_a_cart, d_sh_a,
                            g_a, /*symmetrize_dm=*/true);
  prepare_cartesian_density(basis_, shell_cart_offset_, ncart_total_, dm_beta, d_b_cart, d_sh_b,
                            g_b, /*symmetrize_dm=*/true);

  const std::size_t nsh2 = d_sh_a.size();
  std::vector<double> d_sh_spin(nsh2, 0.0);
  double g_spin = 0.0;
  for (std::size_t i = 0; i < nsh2; ++i) {
    d_sh_spin[i] = std::max(d_sh_a[i], d_sh_b[i]);
    if (d_sh_spin[i] > g_spin) g_spin = d_sh_spin[i];
  }

  auto build_tiles = [&](double gscale, double s_cutoff) {
    std::vector<detail::JKTaskTile> out;
    if (gscale <= 0.0) return out;
    for (int bp = 0; bp < static_cast<int>(pair_lists_.size()); ++bp) {
      const ShellPairData& bra = pair_lists_[bp];
      for (int P = 0; P < bra.npair(); ++P) {
        const double q_ab = bra.schwarz_bound[P];
        if (q_ab * q_ab * gscale < s_cutoff * 1e-6) continue;
        for (int kp = 0; kp <= bp; ++kp) {
          const ShellPairData& ket = pair_lists_[kp];
          const int q_limit = (bp == kp) ? (P + 1) : ket.npair();
          for (int Q0 = 0; Q0 < q_limit; Q0 += kHostTileKetSize) {
            if (q_ab * ket.schwarz_bound[Q0] * gscale < s_cutoff) break;
            const int Q1 = std::min(Q0 + kHostTileKetSize, q_limit);
            out.push_back(detail::JKTaskTile{bp, P, kp, Q0, Q1});
          }
        }
      }
    }
    return out;
  };

  const int nsh = static_cast<int>(basis_.nshell());
  const std::size_t nc2 = static_cast<std::size_t>(ncart_total_) * ncart_total_;
  std::vector<double> j_cart_raw, ka_cart_raw, kb_cart_raw;

  auto run_uhf_pass = [&](bool pass_j, bool pass_ka, bool pass_kb, double pass_omega,
                          double k_coeff, bool acc_k) {
    const bool pass_k = pass_ka || pass_kb;
    const double k_amp = pass_k ? std::max(1.0, std::abs(k_coeff)) : 1.0;
    JKOptions pass_opts = opts;
    pass_opts.omega = pass_omega;
    pass_opts.schwarz_cutoff = opts.schwarz_cutoff / k_amp;
    pass_opts.fp32_threshold = opts.fp32_threshold / k_amp;
    const double global_scale =
        std::max(pass_j ? 4.0 * g_tot : 0.0, pass_k ? 2.0 * g_spin : 0.0);
    const auto tasks = build_tiles(global_scale, pass_opts.schwarz_cutoff);
    if (tasks.empty()) return;

    if (pass_j) j_cart_raw.assign(nc2, 0.0);
    if (pass_ka) ka_cart_raw.assign(nc2, 0.0);
    if (pass_kb) kb_cart_raw.assign(nc2, 0.0);

    if (pass_j && !pass_k) {
      eval_host_direct_jk_tiles<true>(pair_lists_, tasks, shell_cart_offset_, d_tot_cart,
                                      d_sh_tot, global_scale, nsh, ncart_total_, true, false,
                                      pass_opts, j_cart_raw, ka_cart_raw);
    } else {
      eval_host_direct_jk_uhf_tiles(pair_lists_, tasks, shell_cart_offset_, d_tot_cart, d_a_cart,
                                    d_b_cart, d_sh_tot, d_sh_spin, global_scale, nsh, ncart_total_,
                                    pass_j, pass_ka, pass_kb, pass_opts, j_cart_raw, ka_cart_raw,
                                    kb_cart_raw);
    }
    if (pass_j) {
      finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, j_cart_raw, 1.0, vj_tot_out,
                              /*symmetrize_out=*/true, /*accumulate=*/false);
    }
    if (pass_ka) {
      finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, ka_cart_raw, k_coeff,
                              vk_alpha_out, /*symmetrize_out=*/true, /*accumulate=*/acc_k);
    }
    if (pass_kb) {
      finalize_cartesian_fock(basis_, shell_cart_offset_, ncart_total_, kb_cart_raw, k_coeff,
                              vk_beta_out, /*symmetrize_out=*/true, /*accumulate=*/acc_k);
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
  }
}

RIJKEngine::RIJKEngine(const Basis& basis, const Basis& aux_basis, Backend backend, double rsh_omega)
    : basis_(basis), aux_basis_(aux_basis), rsh_omega_(rsh_omega > 0.0 ? rsh_omega : 0.0) {
  if (basis_.max_l() > kMaxSupportedL || aux_basis_.max_l() > kMaxSupportedL) {
    throw NotImplemented("RIJKEngine supports up to g shells (l <= 4)");
  }
  const Backend build_backend = backend_available(backend) ? backend : Backend::Host;

  const int nsh_orb = static_cast<int>(basis_.nshell());
  orb_shell_cart_off_.resize(nsh_orb);
  ncart_orb_ = 0;
  for (int i = 0; i < nsh_orb; ++i) {
    orb_shell_cart_off_[i] = ncart_orb_;
    ncart_orb_ += n_cart(basis_.shell(i).l);
  }

  const int nsh_aux = static_cast<int>(aux_basis_.nshell());
  aux_shell_cart_off_.resize(nsh_aux);
  ncart_aux_ = 0;
  for (int i = 0; i < nsh_aux; ++i) {
    aux_shell_cart_off_[i] = ncart_aux_;
    ncart_aux_ += n_cart(aux_basis_.shell(i).l);
  }

  constexpr double kPi = 3.14159265358979323846;
  constexpr double kY00 = 0.28209479177387814347;
  constexpr double kRiPrimCutoff = 1e-15;

  // Auxiliary 1-shell lists indexed by lP in 0..aux_basis_.max_l() in ascending shell/ao_i order.
  const int aux_max_l = std::max(0, aux_basis_.max_l());
  aux_lists_.resize(static_cast<std::size_t>(aux_max_l) + 1);
  for (int lP = 0; lP <= aux_max_l; ++lP) {
    aux_lists_[lP].la = lP;
    aux_lists_[lP].lb = 0;
  }
  for (int i = 0; i < nsh_aux; ++i) {
    const Shell& s = aux_basis_.shell(i);
    ShellPairData& ap = aux_lists_[s.l];
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
  for (auto& ap : aux_lists_) {
    ap.schwarz_bound.assign(static_cast<std::size_t>(ap.npair()), 0.0);
  }

  // Orbital 2-shell pair lists with radial primitive overlap screening (O(N_sh) for large molecules).
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
          const std::size_t start_prim = op.exponent.size();
          double max_pref = 0.0;
          for (std::size_t ka = 0; ka < si.exponents.size(); ++ka) {
            const double a = si.exponents[ka];
            for (std::size_t kb = 0; kb < sj.exponents.size(); ++kb) {
              const double b = sj.exponents[kb];
              const double p = a + b;
              const double pref = si.coefficients[ka] * sj.coefficients[kb] * kY00 * kY00 *
                                  std::exp(-a * b / p * AB2) * std::pow(kPi / p, 1.5);
              if (std::abs(pref) < kRiPrimCutoff) continue;
              max_pref = std::max(max_pref, std::abs(pref));
              op.exponent.push_back(p);
              op.Px.push_back((a * A[0] + b * B[0]) / p);
              op.Py.push_back((a * A[1] + b * B[1]) / p);
              op.Pz.push_back((a * A[2] + b * B[2]) / p);
              op.prefactor.push_back(pref);
              op.alpha_a.push_back(a);
              op.alpha_b.push_back(b);
            }
          }
          if (op.exponent.size() == start_prim) continue;
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
          op.schwarz_bound.push_back(max_pref);
        }
      }
      if (op.npair() > 0) orb_lists_.push_back(std::move(op));
    }
  }

#if defined(GPU4DFT_HAVE_CUDA)
  if (build_backend == Backend::Cuda) {
    ensure_cuda_cache();
    return;
  }
#else
  (void)build_backend;
#endif

  ensure_host_metric(0.0);
  if (rsh_omega_ > 0.0) {
    ensure_host_metric(rsh_omega_);
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
      rsh_omega_(o.rsh_omega_),
      orb_lists_(std::move(o.orb_lists_)),
      aux_lists_(std::move(o.aux_lists_)),
      orb_shell_cart_off_(std::move(o.orb_shell_cart_off_)),
      aux_shell_cart_off_(std::move(o.aux_shell_cart_off_)),
      ncart_orb_(o.ncart_orb_),
      ncart_aux_(o.ncart_aux_),
      host_metric_0_(std::move(o.host_metric_0_)),
      host_metric_w_(std::move(o.host_metric_w_)),
      b_tensor_(std::move(o.b_tensor_)),
      b_omega_tensor_(std::move(o.b_omega_tensor_)),
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
    rsh_omega_ = o.rsh_omega_;
    orb_lists_ = std::move(o.orb_lists_);
    aux_lists_ = std::move(o.aux_lists_);
    orb_shell_cart_off_ = std::move(o.orb_shell_cart_off_);
    aux_shell_cart_off_ = std::move(o.aux_shell_cart_off_);
    ncart_orb_ = o.ncart_orb_;
    ncart_aux_ = o.ncart_aux_;
    host_metric_0_ = std::move(o.host_metric_0_);
    host_metric_w_ = std::move(o.host_metric_w_);
    b_tensor_ = std::move(o.b_tensor_);
    b_omega_tensor_ = std::move(o.b_omega_tensor_);
    cuda_cache_ = o.cuda_cache_;
    o.cuda_cache_ = nullptr;
  }
  return *this;
}

void RIJKEngine::ensure_host_metric(double omega) const {
  RIHostMetric& target = (omega > 0.0) ? host_metric_w_ : host_metric_0_;
  if (target.built) return;
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_cache_ != nullptr &&
      detail::download_rijk_metric_factor(*cuda_cache_, omega, target.is_cholesky, target.factor)) {
    target.built = true;
    return;
  }
#endif
  factorize_host_ri_metric(naux(), aux_lists_, omega, target);
}

void RIJKEngine::ensure_cuda_cache() const {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_cache_ == nullptr) {
    cuda_cache_ = detail::build_rijk_cuda_cache_native(
        basis_, aux_basis_, orb_lists_, aux_lists_, orb_shell_cart_off_, aux_shell_cart_off_,
        ncart_orb_, ncart_aux_, rsh_omega_);
  }
#endif
}

void RIJKEngine::release_workspaces() const {
#if defined(GPU4DFT_HAVE_CUDA)
  if (cuda_cache_ != nullptr) {
    detail::release_rijk_workspaces(*cuda_cache_);
  }
#endif
}

bool RIJKEngine::has_device_b() const noexcept { return false; }
const double* RIJKEngine::device_b_ptr() const noexcept { return nullptr; }
bool RIJKEngine::has_device_b_omega() const noexcept { return false; }
const double* RIJKEngine::device_b_omega_ptr() const noexcept { return nullptr; }

const std::vector<double>& RIJKEngine::b_tensor() const {
  if (b_tensor_.empty() && nao() > 0 && naux() > 0) {
    ensure_host_metric(0.0);
    build_host_b_from_metric(nao(), naux(), aux_basis_, orb_lists_, aux_lists_,
                             aux_shell_cart_off_, host_metric_0_, 0.0, b_tensor_);
  }
  return b_tensor_;
}

const std::vector<double>& RIJKEngine::b_omega_tensor() const {
  if (b_omega_tensor_.empty() && rsh_omega_ > 0.0 && nao() > 0 && naux() > 0) {
    ensure_host_metric(rsh_omega_);
    build_host_b_from_metric(nao(), naux(), aux_basis_, orb_lists_, aux_lists_,
                             aux_shell_cart_off_, host_metric_w_, rsh_omega_, b_omega_tensor_);
  }
  return b_omega_tensor_;
}

void RIJKEngine::compute_jk_device(const double* d_dm, double* d_vj, double* d_vk,
                                   const JKOptions& opts, void* stream) const {
  compute_jk_mo_device(d_dm, nullptr, -1, 1.0, d_vj, d_vk, opts, stream);
}

void RIJKEngine::compute_jk(const double* dm, double* vj_out, double* vk_out,
                            const JKOptions& opts) const {
  compute_jk_mo(dm, nullptr, -1, 1.0, vj_out, vk_out, opts);
}

void RIJKEngine::compute_jk_mo_device(const double* d_dm, const double* d_c_occ, int n_occ,
                                      double occ_scale, double* d_vj, double* d_vk,
                                      const JKOptions& opts, void* stream) const {
  if (!backend_available(Backend::Cuda)) {
    throw std::runtime_error("RIJKEngine::compute_jk_mo_device: backend 'cuda' is not available");
  }
  if (d_dm == nullptr) {
    throw std::invalid_argument("RIJKEngine::compute_jk_mo_device: d_dm is null");
  }
  const bool do_j = opts.compute_j && (d_vj != nullptr);
  const bool do_k = opts.compute_k && (d_vk != nullptr);
  const bool needs_omega_k =
      (!opts.rsh_mode && opts.omega > 0.0 && (do_j || (do_k && opts.k_scale != 0.0))) ||
      (opts.rsh_mode && do_k && opts.omega > 0.0 && (opts.k_lr - opts.k_sr) != 0.0 &&
       opts.k_scale != 0.0);
  if (needs_omega_k && (rsh_omega_ <= 0.0 || std::abs(rsh_omega_ - opts.omega) > 1e-12)) {
    throw std::invalid_argument(
        "RIJKEngine::compute_jk_mo_device: requested omega does not match initialized rsh_omega");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  const int n_ao = nao();
  const int n_aux = naux();
  if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) return;

  ensure_cuda_cache();
  JKOptions eff_opts = opts;
  eff_opts.backend = Backend::Cuda;
  eff_opts.compute_j = do_j;
  eff_opts.compute_k = do_k;
  detail::run_rijk_mo_cuda_device(*cuda_cache_, d_dm, d_c_occ, n_occ, occ_scale,
                                  do_j ? d_vj : nullptr, do_k ? d_vk : nullptr, eff_opts, stream);
#else
  (void)d_c_occ;
  (void)n_occ;
  (void)occ_scale;
  (void)d_vj;
  (void)d_vk;
  (void)opts;
  (void)stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void RIJKEngine::compute_jk_uks_mo_device(const double* d_dm_alpha, const double* d_dm_beta,
                                          const double* d_c_occ_alpha, int n_alpha,
                                          const double* d_c_occ_beta, int n_beta, double* d_vj_tot,
                                          double* d_vk_alpha, double* d_vk_beta,
                                          const JKOptions& opts, void* stream) const {
  if (!backend_available(Backend::Cuda)) {
    throw std::runtime_error(
        "RIJKEngine::compute_jk_uks_mo_device: backend 'cuda' is not available");
  }
  if (d_dm_alpha == nullptr || d_dm_beta == nullptr) {
    throw std::invalid_argument("RIJKEngine::compute_jk_uks_mo_device: spin DM pointer is null");
  }
  const bool do_j = opts.compute_j && (d_vj_tot != nullptr);
  const bool do_ka = opts.compute_k && (d_vk_alpha != nullptr);
  const bool do_kb = opts.compute_k && (d_vk_beta != nullptr);
  const bool do_k = do_ka || do_kb;
  const bool needs_omega_k =
      (!opts.rsh_mode && opts.omega > 0.0 && (do_j || (do_k && opts.k_scale != 0.0))) ||
      (opts.rsh_mode && do_k && opts.omega > 0.0 && (opts.k_lr - opts.k_sr) != 0.0 &&
       opts.k_scale != 0.0);
  if (needs_omega_k && (rsh_omega_ <= 0.0 || std::abs(rsh_omega_ - opts.omega) > 1e-12)) {
    throw std::invalid_argument(
        "RIJKEngine::compute_jk_uks_mo_device: requested omega does not match initialized rsh_omega");
  }
#if defined(GPU4DFT_HAVE_CUDA)
  const int n_ao = nao();
  const int n_aux = naux();
  if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) return;

  ensure_cuda_cache();
  JKOptions eff_opts = opts;
  eff_opts.backend = Backend::Cuda;
  eff_opts.compute_j = do_j;
  eff_opts.compute_k = do_k;
  detail::run_rijk_uks_mo_cuda_device(*cuda_cache_, d_dm_alpha, d_dm_beta, d_c_occ_alpha, n_alpha,
                                      d_c_occ_beta, n_beta, do_j ? d_vj_tot : nullptr,
                                      do_ka ? d_vk_alpha : nullptr, do_kb ? d_vk_beta : nullptr,
                                      eff_opts, stream);
#else
  (void)d_c_occ_alpha;
  (void)n_alpha;
  (void)d_c_occ_beta;
  (void)n_beta;
  (void)d_vj_tot;
  (void)d_vk_alpha;
  (void)d_vk_beta;
  (void)opts;
  (void)stream;
  throw std::runtime_error("gpu4dft was built without CUDA");
#endif
}

void RIJKEngine::compute_jk_mo(const double* dm, const double* c_occ, int n_occ, double occ_scale,
                               double* vj_out, double* vk_out, const JKOptions& opts) const {
  if (!backend_available(opts.backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(opts.backend) +
                             "' is not available");
  }
  if (dm == nullptr) throw std::invalid_argument("RIJKEngine::compute_jk_mo: dm is null");

  const int n_ao = nao();
  const int n_aux = naux();
  const std::size_t nao2 = static_cast<std::size_t>(n_ao) * n_ao;
  const bool do_j = opts.compute_j && (vj_out != nullptr);
  const bool do_k = opts.compute_k && (vk_out != nullptr);
  const bool needs_omega_k =
      (!opts.rsh_mode && opts.omega > 0.0 && (do_j || (do_k && opts.k_scale != 0.0))) ||
      (opts.rsh_mode && do_k && opts.omega > 0.0 && (opts.k_lr - opts.k_sr) != 0.0 &&
       opts.k_scale != 0.0);
  if (needs_omega_k && (rsh_omega_ <= 0.0 || std::abs(rsh_omega_ - opts.omega) > 1e-12)) {
    throw std::invalid_argument(
        "RIJKEngine::compute_jk_mo: requested omega does not match initialized rsh_omega");
  }

  if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) {
      if (do_j) std::fill(vj_out, vj_out + nao2, 0.0);
      if (do_k) std::fill(vk_out, vk_out + nao2, 0.0);
      return;
    }
    ensure_cuda_cache();
    JKOptions eff_opts = opts;
    eff_opts.compute_j = do_j;
    eff_opts.compute_k = do_k;
    detail::run_rijk_mo_cuda(*cuda_cache_, dm, c_occ, n_occ, occ_scale, do_j ? vj_out : nullptr,
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
  if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) return;

  std::vector<double> d_eff(nao2);
  if (opts.symmetrize_dm) {
    for (int u = 0; u < n_ao; ++u)
      for (int v = 0; v < n_ao; ++v)
        d_eff[static_cast<std::size_t>(u) * n_ao + v] =
            0.5 * (dm[static_cast<std::size_t>(u) * n_ao + v] +
                   dm[static_cast<std::size_t>(v) * n_ao + u]);
  } else {
    std::copy(dm, dm + nao2, d_eff.begin());
  }

  const GemmMode k_gemm_mode =
      (opts.precision == Precision::FP32) ? Precision::FP32 == opts.precision ? GemmMode::FP32 : opts.gemm_mode : opts.gemm_mode;

  struct HostFactor {
    const double* A = nullptr;  // [nao x ncols]
    const double* B = nullptr;  // if nullptr, symmetric: B == A; else [nao x ncols]
    int ncols = 0;
    double weight = 1.0;
  };

  std::vector<double> dm_c_pos, dm_c_neg, dm_ident;
  std::vector<HostFactor> factors;

  const bool need_any_k =
      do_k && (opts.k_scale != 0.0) && (n_occ != 0) &&
      (!opts.rsh_mode || opts.k_sr != 0.0 || (opts.omega > 0.0 && opts.k_lr != opts.k_sr));

  if (need_any_k) {
    if (c_occ != nullptr && n_occ > 0) {
      factors.push_back(HostFactor{c_occ, nullptr, n_occ, occ_scale});
    } else if (opts.symmetrize_dm) {
      std::vector<double> w(n_ao, 0.0);
      std::vector<double> U(nao2, 0.0);
      syevd(n_ao, d_eff.data(), w.data(), U.data(), Backend::Host);
      std::vector<int> pos_idx, neg_idx;
      for (int k = 0; k < n_ao; ++k) {
        if (w[k] > 1e-14) pos_idx.push_back(k);
        else if (w[k] < -1e-14) neg_idx.push_back(k);
      }
      if (!pos_idx.empty()) {
        const int np = static_cast<int>(pos_idx.size());
        dm_c_pos.resize(static_cast<std::size_t>(n_ao) * np);
        for (int c = 0; c < np; ++c) {
          const int k = pos_idx[c];
          const double s = std::sqrt(w[k]);
          for (int r = 0; r < n_ao; ++r) {
            dm_c_pos[static_cast<std::size_t>(r) * np + c] =
                s * U[static_cast<std::size_t>(r) * n_ao + k];
          }
        }
        factors.push_back(HostFactor{dm_c_pos.data(), nullptr, np, 1.0});
      }
      if (!neg_idx.empty()) {
        const int nn = static_cast<int>(neg_idx.size());
        dm_c_neg.resize(static_cast<std::size_t>(n_ao) * nn);
        for (int c = 0; c < nn; ++c) {
          const int k = neg_idx[c];
          const double s = std::sqrt(-w[k]);
          for (int r = 0; r < n_ao; ++r) {
            dm_c_neg[static_cast<std::size_t>(r) * nn + c] =
                s * U[static_cast<std::size_t>(r) * n_ao + k];
          }
        }
        factors.push_back(HostFactor{dm_c_neg.data(), nullptr, nn, -1.0});
      }
    } else {
      dm_ident.assign(nao2, 0.0);
      for (int i = 0; i < n_ao; ++i) dm_ident[static_cast<std::size_t>(i) * n_ao + i] = 1.0;
      factors.push_back(HostFactor{d_eff.data(), dm_ident.data(), n_ao, 1.0});
    }
  }

  const auto chunks = build_ri_aux_chunks(aux_basis_, aux_lists_, aux_shell_cart_off_, 64, 256);

  auto run_host_op = [&](double pass_omega, bool pass_j, bool pass_k, double k_coeff,
                         bool acc_k) {
    ensure_host_metric(pass_omega);
    const RIHostMetric& m = (pass_omega > 0.0) ? host_metric_w_ : host_metric_0_;

    if (pass_j && (!pass_k || factors.empty() || k_coeff == 0.0)) {
      std::vector<double> v_aux(n_aux, 0.0);
      std::vector<double> I_chunk;
      for (const auto& c : chunks) {
        const int bc = c.q1 - c.q0;
        I_chunk.resize(static_cast<std::size_t>(bc) * nao2);
        eval_host_ri_3c_chunk(n_ao, orb_lists_, aux_lists_, c, pass_omega, I_chunk.data());
        gemm(bc, 1, static_cast<int>(nao2), 1.0, I_chunk.data(), static_cast<int>(nao2), false,
             d_eff.data(), 1, false, 0.0, v_aux.data() + c.q0, 1, Backend::Host);
      }
      solve_host_V_inplace(n_aux, 1, m, v_aux.data());
      for (const auto& c : chunks) {
        const int bc = c.q1 - c.q0;
        I_chunk.resize(static_cast<std::size_t>(bc) * nao2);
        eval_host_ri_3c_chunk(n_ao, orb_lists_, aux_lists_, c, pass_omega, I_chunk.data());
        gemm(1, static_cast<int>(nao2), bc, 1.0, v_aux.data() + c.q0, bc, false, I_chunk.data(),
             static_cast<int>(nao2), false, 1.0, vj_out, static_cast<int>(nao2), Backend::Host);
      }
      return;
    }

    if (!pass_k || k_coeff == 0.0 || factors.empty()) return;

    const std::size_t host_y_cap = 1536ULL << 20;  // 1.5 GiB cap on Host
    const std::size_t per_col_bytes = static_cast<std::size_t>(n_aux) * n_ao * sizeof(double);
    const int b_i_cap = std::max(
        1, static_cast<int>(std::min<std::size_t>(std::max(1, n_ao), host_y_cap / std::max<std::size_t>(1, per_col_bytes))));

    std::vector<double> v_aux(pass_j ? n_aux : 0, 0.0);
    bool j_pass1_done = false;
    bool j_pass2_done = !pass_j;

    // Count total tiles to decide if RI-J pass 2 can piggyback on tile 1
    int total_tiles = 0;
    for (const auto& f : factors) {
      const int step = (f.B != nullptr) ? std::max(1, b_i_cap / 2) : b_i_cap;
      total_tiles += (f.ncols + step - 1) / step;
    }

    std::vector<double> I_chunk;
    std::vector<double> C_tile;
    std::vector<double> Y_blk;
    std::vector<double> L_pchunk;
    std::vector<double> L_perm_a;
    std::vector<double> L_perm_b;

    bool first_k_write = !acc_k;
    int tile_counter = 0;

    for (const auto& f : factors) {
      const bool nonsym_pair = (f.B != nullptr);
      const int step = nonsym_pair ? std::max(1, b_i_cap / 2) : b_i_cap;

      for (int i0 = 0; i0 < f.ncols; i0 += step) {
        const int bi = std::min(step, f.ncols - i0);
        const int bi_tot = nonsym_pair ? (2 * bi) : bi;
        C_tile.resize(static_cast<std::size_t>(n_ao) * bi_tot);
        for (int r = 0; r < n_ao; ++r) {
          const double* sa = f.A + static_cast<std::size_t>(r) * f.ncols + i0;
          double* dt = C_tile.data() + static_cast<std::size_t>(r) * bi_tot;
          std::copy(sa, sa + bi, dt);
          if (nonsym_pair) {
            const double* sb = f.B + static_cast<std::size_t>(r) * f.ncols + i0;
            std::copy(sb, sb + bi, dt + bi);
          }
        }

        Y_blk.assign(static_cast<std::size_t>(n_aux) * n_ao * bi_tot, 0.0);
        const bool collect_j_now = pass_j && !j_pass1_done;
        const bool apply_j_now = pass_j && j_pass1_done && !j_pass2_done;

        for (const auto& c : chunks) {
          const int bc = c.q1 - c.q0;
          I_chunk.resize(static_cast<std::size_t>(bc) * nao2);
          eval_host_ri_3c_chunk(n_ao, orb_lists_, aux_lists_, c, pass_omega, I_chunk.data());

          if (collect_j_now) {
            gemm(bc, 1, static_cast<int>(nao2), 1.0, I_chunk.data(), static_cast<int>(nao2), false,
                 d_eff.data(), 1, false, 0.0, v_aux.data() + c.q0, 1, Backend::Host);
          }
          if (apply_j_now) {
            gemm(1, static_cast<int>(nao2), bc, 1.0, v_aux.data() + c.q0, bc, false,
                 I_chunk.data(), static_cast<int>(nao2), false, 1.0, vj_out,
                 static_cast<int>(nao2), Backend::Host);
          }

          gemm(bc * n_ao, bi_tot, n_ao, 1.0, I_chunk.data(), n_ao, false, C_tile.data(), bi_tot,
               false, 0.0, Y_blk.data() + static_cast<std::size_t>(c.q0) * n_ao * bi_tot, bi_tot,
               Backend::Host, k_gemm_mode, opts.ozaki_slices);
        }

        if (collect_j_now) {
          solve_host_V_inplace(n_aux, 1, m, v_aux.data());
          j_pass1_done = true;
          if (total_tiles <= 1) {
            for (const auto& c : chunks) {
              const int bc = c.q1 - c.q0;
              I_chunk.resize(static_cast<std::size_t>(bc) * nao2);
              eval_host_ri_3c_chunk(n_ao, orb_lists_, aux_lists_, c, pass_omega, I_chunk.data());
              gemm(1, static_cast<int>(nao2), bc, 1.0, v_aux.data() + c.q0, bc, false,
                   I_chunk.data(), static_cast<int>(nao2), false, 1.0, vj_out,
                   static_cast<int>(nao2), Backend::Host);
            }
            j_pass2_done = true;
          }
        } else if (apply_j_now) {
          j_pass2_done = true;
        }

        const int nao_bi_tot = n_ao * bi_tot;
        if (m.is_cholesky) {
          solve_host_L_forward_inplace(n_aux, nao_bi_tot, m.factor.data(), Y_blk.data());
        }

        const double w_eff = k_coeff * f.weight;
        constexpr int kHostPBlk = 64;
        for (int p0 = 0; p0 < n_aux; p0 += kHostPBlk) {
          const int bp = std::min(kHostPBlk, n_aux - p0);
          const double* L_src = nullptr;
          if (m.is_cholesky) {
            L_src = Y_blk.data() + static_cast<std::size_t>(p0) * nao_bi_tot;
          } else {
            L_pchunk.resize(static_cast<std::size_t>(bp) * nao_bi_tot);
            gemm(bp, nao_bi_tot, n_aux, 1.0,
                 m.factor.data() + static_cast<std::size_t>(p0) * n_aux, n_aux, false, Y_blk.data(),
                 nao_bi_tot, false, 0.0, L_pchunk.data(), nao_bi_tot, Backend::Host);
            L_src = L_pchunk.data();
          }

          L_perm_a.resize(static_cast<std::size_t>(n_ao) * bp * bi);
          if (nonsym_pair) L_perm_b.resize(static_cast<std::size_t>(n_ao) * bp * bi);

          #if defined(_OPENMP)
          #pragma omp parallel for schedule(static)
          #endif
          for (int u = 0; u < n_ao; ++u) {
            for (int pl = 0; pl < bp; ++pl) {
              const double* s_ptr =
                  L_src + (static_cast<std::size_t>(pl) * n_ao + u) * bi_tot;
              double* da_ptr =
                  L_perm_a.data() + (static_cast<std::size_t>(u) * bp + pl) * bi;
              std::copy(s_ptr, s_ptr + bi, da_ptr);
              if (nonsym_pair) {
                double* db_ptr =
                    L_perm_b.data() + (static_cast<std::size_t>(u) * bp + pl) * bi;
                std::copy(s_ptr + bi, s_ptr + 2 * bi, db_ptr);
              }
            }
          }

          const int k_dim = bp * bi;
          const double beta = first_k_write ? 0.0 : 1.0;
          gemm(n_ao, n_ao, k_dim, w_eff, L_perm_a.data(), k_dim, false,
               nonsym_pair ? L_perm_b.data() : L_perm_a.data(), k_dim, true, beta, vk_out, n_ao,
               Backend::Host, k_gemm_mode, opts.ozaki_slices);
          first_k_write = false;
        }
        ++tile_counter;
      }
    }

    if (opts.symmetrize_dm) {
      for (int u = 0; u < n_ao; ++u) {
        for (int v = u + 1; v < n_ao; ++v) {
          const double avg = 0.5 * (vk_out[static_cast<std::size_t>(u) * n_ao + v] +
                                    vk_out[static_cast<std::size_t>(v) * n_ao + u]);
          vk_out[static_cast<std::size_t>(u) * n_ao + v] = avg;
          vk_out[static_cast<std::size_t>(v) * n_ao + u] = avg;
        }
      }
    }
  };

  if (!opts.rsh_mode) {
    run_host_op(opts.omega, do_j, do_k && (opts.k_scale != 0.0), opts.k_scale, false);
  } else {
    const double k1 = opts.k_sr * opts.k_scale;
    const bool pass1_k = do_k && (k1 != 0.0);
    if (do_j || pass1_k) {
      run_host_op(0.0, do_j, pass1_k, k1, false);
    }
    const double k2 = (opts.k_lr - opts.k_sr) * opts.k_scale;
    const bool pass2_k = do_k && (opts.omega > 0.0) && (k2 != 0.0);
    if (pass2_k) {
      run_host_op(opts.omega, false, true, k2, pass1_k);
    }
  }
}

void RIJKEngine::compute_jk_uks_mo(const double* dm_alpha, const double* dm_beta,
                                   const double* c_occ_alpha, int n_alpha,
                                   const double* c_occ_beta, int n_beta, double* vj_tot_out,
                                   double* vk_alpha_out, double* vk_beta_out,
                                   const JKOptions& opts) const {
  if (!backend_available(opts.backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(opts.backend) +
                             "' is not available");
  }
  if (dm_alpha == nullptr || dm_beta == nullptr) {
    throw std::invalid_argument("RIJKEngine::compute_jk_uks_mo: spin DM pointer is null");
  }

  const int n_ao = nao();
  const int n_aux = naux();
  const std::size_t nao2 = static_cast<std::size_t>(n_ao) * n_ao;
  const bool do_j = opts.compute_j && (vj_tot_out != nullptr);
  const bool do_ka = opts.compute_k && (vk_alpha_out != nullptr);
  const bool do_kb = opts.compute_k && (vk_beta_out != nullptr);
  const bool do_k = do_ka || do_kb;
  const bool needs_omega_k =
      (!opts.rsh_mode && opts.omega > 0.0 && (do_j || (do_k && opts.k_scale != 0.0))) ||
      (opts.rsh_mode && do_k && opts.omega > 0.0 && (opts.k_lr - opts.k_sr) != 0.0 &&
       opts.k_scale != 0.0);
  if (needs_omega_k && (rsh_omega_ <= 0.0 || std::abs(rsh_omega_ - opts.omega) > 1e-12)) {
    throw std::invalid_argument(
        "RIJKEngine::compute_jk_uks_mo: requested omega does not match initialized rsh_omega");
  }

  if (opts.backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    if ((!do_j && !do_k) || n_ao == 0 || n_aux == 0) {
      if (do_j) std::fill(vj_tot_out, vj_tot_out + nao2, 0.0);
      if (do_ka) std::fill(vk_alpha_out, vk_alpha_out + nao2, 0.0);
      if (do_kb) std::fill(vk_beta_out, vk_beta_out + nao2, 0.0);
      return;
    }
    ensure_cuda_cache();
    JKOptions eff_opts = opts;
    eff_opts.compute_j = do_j;
    eff_opts.compute_k = do_k;
    detail::run_rijk_uks_mo_cuda(*cuda_cache_, dm_alpha, dm_beta, c_occ_alpha, n_alpha, c_occ_beta,
                                 n_beta, do_j ? vj_tot_out : nullptr,
                                 do_ka ? vk_alpha_out : nullptr, do_kb ? vk_beta_out : nullptr,
                                 eff_opts);
    return;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  if (do_j) {
    std::vector<double> dm_tot(nao2, 0.0);
    for (std::size_t i = 0; i < nao2; ++i) dm_tot[i] = dm_alpha[i] + dm_beta[i];
    JKOptions j_opts = opts;
    j_opts.compute_j = true;
    j_opts.compute_k = false;
    compute_jk_mo(dm_tot.data(), nullptr, 0, 1.0, vj_tot_out, nullptr, j_opts);
  }
  JKOptions k_opts = opts;
  k_opts.compute_j = false;
  k_opts.compute_k = true;
  if (do_ka) {
    compute_jk_mo(dm_alpha, c_occ_alpha, n_alpha, 1.0, nullptr, vk_alpha_out, k_opts);
  }
  if (do_kb) {
    compute_jk_mo(dm_beta, c_occ_beta, n_beta, 1.0, nullptr, vk_beta_out, k_opts);
  }
}

namespace {

std::vector<double> build_orb_sph_to_cart_mat(const Basis& basis,
                                              const std::vector<int>& shell_cart_off,
                                              int ncart_orb) {
  const int nao = basis.nao();
  std::vector<double> C(static_cast<std::size_t>(nao) * ncart_orb, 0.0);
  const int nsh = static_cast<int>(basis.nshell());
  for (int s = 0; s < nsh; ++s) {
    const int l = basis.shell(s).l;
    const int ns = n_sph(l);
    const int nc = n_cart(l);
    const int ao0 = basis.ao_offset(s);
    const int c0 = shell_cart_off[s];
    double unit[15] = {0.0};
    double col[9] = {0.0};
    for (int ic = 0; ic < nc; ++ic) {
      unit[ic] = 1.0;
      harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
      for (int m = 0; m < ns; ++m) {
        C[static_cast<std::size_t>(ao0 + m) * ncart_orb + (c0 + ic)] = col[m];
      }
      unit[ic] = 0.0;
    }
  }
  return C;
}

void transform_aux_rows_sph_to_cart(const Basis& aux_basis, int s0, int s1, int q0, int pc0,
                                    int ncols, const std::vector<int>& aux_shell_cart_off,
                                    const double* src_sph, double* dst_cart) {
  const int bc_cart = aux_shell_cart_off[s1 - 1] + n_cart(aux_basis.shell(s1 - 1).l) - pc0;
  std::fill(dst_cart, dst_cart + static_cast<std::size_t>(bc_cart) * ncols, 0.0);
  for (int s = s0; s < s1; ++s) {
    const int l = aux_basis.shell(s).l;
    const int ns = n_sph(l);
    const int nc = n_cart(l);
    const int p_loc = aux_basis.ao_offset(s) - q0;
    const int pc_loc = aux_shell_cart_off[s] - pc0;
    double unit[15] = {0.0};
    double col[9] = {0.0};
    for (int ic = 0; ic < nc; ++ic) {
      unit[ic] = 1.0;
      harmonics::cart_to_sph_axis<double>(l, unit, 1, col, 1);
      double* out_row = dst_cart + static_cast<std::size_t>(pc_loc + ic) * ncols;
      for (int m = 0; m < ns; ++m) {
        const double w = col[m];
        if (w == 0.0) continue;
        const double* in_row = src_sph + static_cast<std::size_t>(p_loc + m) * ncols;
        for (int c = 0; c < ncols; ++c) out_row[c] += w * in_row[c];
      }
      unit[ic] = 0.0;
    }
  }
}

}  // namespace

std::vector<double> RIJKEngine::compute_ri_jk_grad(
    std::span<const Atom> atoms, const double* dm_alpha, const double* c_occ_alpha, int n_alpha,
    const double* dm_beta, const double* c_occ_beta, int n_beta, bool unrestricted,
    double j_fraction, double k_sr, double k_lr, double omega, Backend backend) const {
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }
  const int n_ao = nao();
  const int n_aux = naux();
  std::vector<double> grad(atoms.size() * 3, 0.0);
  if (atoms.empty() || n_ao == 0 || n_aux == 0) return grad;
  if (dm_alpha == nullptr || (unrestricted && dm_beta == nullptr)) {
    throw std::invalid_argument("RIJKEngine::compute_ri_jk_grad: density matrix pointer is null");
  }
  const double k_lr_delta = (omega > 0.0) ? (k_lr - k_sr) : 0.0;
  if (omega > 0.0 && k_lr_delta != 0.0 && rsh_omega_ > 0.0 &&
      std::abs(rsh_omega_ - omega) > 1e-12) {
    throw std::invalid_argument(
        "RIJKEngine::compute_ri_jk_grad: requested omega does not match initialized rsh_omega");
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    ensure_cuda_cache();
    detail::run_ri_jk_grad_cuda(*cuda_cache_, atoms, dm_alpha, c_occ_alpha, n_alpha, dm_beta,
                                c_occ_beta, n_beta, unrestricted, j_fraction, k_sr, k_lr, omega,
                                grad);
    return grad;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  if (backend != Backend::Host) {
    throw std::runtime_error("gpu4dft was built without HIP");
  }

  const std::size_t nao2 = static_cast<std::size_t>(n_ao) * n_ao;
  const int nsh_aux = static_cast<int>(aux_basis_.nshell());
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
  const std::vector<int> orb_atom = map_shell_atoms(basis_);
  const std::vector<int> aux_atom = map_shell_atoms(aux_basis_);

  std::vector<double> dm_tot(nao2, 0.0);
  for (std::size_t i = 0; i < nao2; ++i) {
    const double da = dm_alpha[i];
    const double db = unrestricted ? dm_beta[i] : 0.0;
    dm_tot[i] = da + db;
  }
  // Symmetrize D_tot
  for (int u = 0; u < n_ao; ++u) {
    for (int v = u + 1; v < n_ao; ++v) {
      const double avg = 0.5 * (dm_tot[static_cast<std::size_t>(u) * n_ao + v] +
                                dm_tot[static_cast<std::size_t>(v) * n_ao + u]);
      dm_tot[static_cast<std::size_t>(u) * n_ao + v] = avg;
      dm_tot[static_cast<std::size_t>(v) * n_ao + u] = avg;
    }
  }

  // Transform D_tot to Cartesian once: D_tot_cart = C_orb^T * D_tot * C_orb
  const std::vector<double> C_orb = build_orb_sph_to_cart_mat(basis_, orb_shell_cart_off_, ncart_orb_);
  std::vector<double> D_tot_cart(static_cast<std::size_t>(ncart_orb_) * ncart_orb_, 0.0);
  {
    std::vector<double> tmp(static_cast<std::size_t>(n_ao) * ncart_orb_, 0.0);
    gemm(n_ao, ncart_orb_, n_ao, 1.0, dm_tot.data(), n_ao, false, C_orb.data(), ncart_orb_, false,
         0.0, tmp.data(), ncart_orb_, Backend::Host);
    gemm(ncart_orb_, ncart_orb_, n_ao, 1.0, C_orb.data(), ncart_orb_, true, tmp.data(), ncart_orb_,
         false, 0.0, D_tot_cart.data(), ncart_orb_, Backend::Host);
  }

  struct GradSpinChannel {
    std::vector<double> C_sph;   // [nao x ncols]
    std::vector<double> C_cart;  // [ncart_orb x ncols]
    std::vector<double> signs;   // [ncols] (+1 or -1)
    int ncols = 0;
    double occ_weight = 1.0;     // 2.0 for RHF C_occ, 1.0 for UHF or DM-factor
  };

  std::vector<GradSpinChannel> channels;
  const bool need_any_k = (k_sr != 0.0) || (omega > 0.0 && k_lr_delta != 0.0);
  if (need_any_k) {
    auto add_channel = [&](const double* dm_s, const double* c_occ_s, int n_occ_s, double occ_w) {
      if (n_occ_s == 0) return;
      GradSpinChannel ch;
      if (c_occ_s != nullptr && n_occ_s > 0) {
        ch.ncols = n_occ_s;
        ch.occ_weight = occ_w;
        ch.C_sph.assign(c_occ_s, c_occ_s + static_cast<std::size_t>(n_ao) * n_occ_s);
        ch.signs.assign(static_cast<std::size_t>(n_occ_s), 1.0);
      } else {
        std::vector<double> d_sym(nao2, 0.0);
        for (int u = 0; u < n_ao; ++u) {
          for (int v = 0; v < n_ao; ++v) {
            d_sym[static_cast<std::size_t>(u) * n_ao + v] =
                0.5 * (dm_s[static_cast<std::size_t>(u) * n_ao + v] +
                       dm_s[static_cast<std::size_t>(v) * n_ao + u]);
          }
        }
        std::vector<double> w(n_ao, 0.0), U(nao2, 0.0);
        syevd(n_ao, d_sym.data(), w.data(), U.data(), Backend::Host);
        std::vector<int> idx_keep;
        for (int k = 0; k < n_ao; ++k) {
          if (std::abs(w[k]) > 1e-14) idx_keep.push_back(k);
        }
        ch.ncols = static_cast<int>(idx_keep.size());
        if (ch.ncols == 0) return;
        // RHF DM-fallback D_tot = 1 * C_dm C_dm^T has E_K = -0.25 * Tr(D K(D)), so effective
        // factor relative to `0.5 * occ_weight` (2c) / `-occ_weight` (3c) is 0.5 (not 1.0)!
        ch.occ_weight = unrestricted ? 1.0 : 0.5;
        ch.C_sph.resize(static_cast<std::size_t>(n_ao) * ch.ncols);
        ch.signs.resize(static_cast<std::size_t>(ch.ncols));
        for (int c = 0; c < ch.ncols; ++c) {
          const int k = idx_keep[c];
          const double sc = std::sqrt(std::abs(w[k]));
          ch.signs[c] = (w[k] >= 0.0) ? 1.0 : -1.0;
          for (int r = 0; r < n_ao; ++r) {
            ch.C_sph[static_cast<std::size_t>(r) * ch.ncols + c] =
                sc * U[static_cast<std::size_t>(r) * n_ao + k];
          }
        }
      }
      ch.C_cart.assign(static_cast<std::size_t>(ncart_orb_) * ch.ncols, 0.0);
      gemm(ncart_orb_, ch.ncols, n_ao, 1.0, C_orb.data(), ncart_orb_, true, ch.C_sph.data(),
           ch.ncols, false, 0.0, ch.C_cart.data(), ch.ncols, Backend::Host);
      channels.push_back(std::move(ch));
    };

    if (!unrestricted) {
      add_channel(dm_alpha, c_occ_alpha, n_alpha, 2.0);
    } else {
      add_channel(dm_alpha, c_occ_alpha, n_alpha, 1.0);
      add_channel(dm_beta, c_occ_beta, n_beta, 1.0);
    }
  }

  std::vector<std::pair<int, int>> orb_pairs;
  for (int io = 0; io < static_cast<int>(orb_lists_.size()); ++io) {
    for (int ab = 0; ab < orb_lists_[io].npair(); ++ab) {
      orb_pairs.emplace_back(io, ab);
    }
  }

  const auto chunks = build_ri_aux_chunks(aux_basis_, aux_lists_, aux_shell_cart_off_, 32, 64);

  auto run_host_grad_pass = [&](double pass_omega, double j_frac, double k_frac) {
    if (j_frac == 0.0 && (k_frac == 0.0 || channels.empty())) return;
    RIHostMetric local_metric;
    const RIHostMetric* m_ptr = nullptr;
    if (pass_omega == 0.0 || (rsh_omega_ > 0.0 && std::abs(pass_omega - rsh_omega_) <= 1e-12)) {
      ensure_host_metric(pass_omega);
      m_ptr = (pass_omega > 0.0) ? &host_metric_w_ : &host_metric_0_;
    } else {
      factorize_host_ri_metric(n_aux, aux_lists_, pass_omega, local_metric);
      m_ptr = &local_metric;
    }

    const bool do_k = (k_frac != 0.0) && !channels.empty();
    std::vector<double> rho_sph(j_frac != 0.0 ? n_aux : 0, 0.0);
    std::vector<std::vector<double>> M_channels(do_k ? channels.size() : 0);
    if (do_k) {
      for (std::size_t s = 0; s < channels.size(); ++s) {
        const std::size_t nc2 =
            static_cast<std::size_t>(channels[s].ncols) * channels[s].ncols;
        M_channels[s].assign(static_cast<std::size_t>(n_aux) * nc2, 0.0);
      }
    }

    // Pass 1: Chunked 3c integrals -> t_P and I_MO_P -> solve V rho = t and V M = I_MO
    std::vector<double> I_chunk;
    std::vector<double> T_half;
    for (const auto& c : chunks) {
      const int bc = c.q1 - c.q0;
      I_chunk.resize(static_cast<std::size_t>(bc) * nao2);
      eval_host_ri_3c_chunk(n_ao, orb_lists_, aux_lists_, c, pass_omega, I_chunk.data());

      if (j_frac != 0.0) {
        gemm(bc, 1, static_cast<int>(nao2), 1.0, I_chunk.data(), static_cast<int>(nao2), false,
             dm_tot.data(), 1, false, 0.0, rho_sph.data() + c.q0, 1, Backend::Host);
      }
      if (do_k) {
        for (std::size_t s = 0; s < channels.size(); ++s) {
          const int nc = channels[s].ncols;
          const int nc2 = nc * nc;
          T_half.resize(static_cast<std::size_t>(bc) * n_ao * nc);
          gemm(bc * n_ao, nc, n_ao, 1.0, I_chunk.data(), n_ao, false, channels[s].C_sph.data(), nc,
               false, 0.0, T_half.data(), nc, Backend::Host);
          #if defined(_OPENMP)
          #pragma omp parallel for schedule(static)
          #endif
          for (int pl = 0; pl < bc; ++pl) {
            gemm(nc, nc, n_ao, 1.0, channels[s].C_sph.data(), nc, true,
                 T_half.data() + static_cast<std::size_t>(pl) * n_ao * nc, nc, false, 0.0,
                 M_channels[s].data() + static_cast<std::size_t>(c.q0 + pl) * nc2, nc,
                 Backend::Host);
          }
        }
      }
    }

    if (j_frac != 0.0) {
      solve_host_V_inplace(n_aux, 1, *m_ptr, rho_sph.data());
    }
    if (do_k) {
      for (std::size_t s = 0; s < channels.size(); ++s) {
        const int nc2 = channels[s].ncols * channels[s].ncols;
        solve_host_V_inplace(n_aux, nc2, *m_ptr, M_channels[s].data());
      }
    }

    // Pass 2: 2-center metric derivative
    {
      std::vector<double> Gamma_2c_sph(static_cast<std::size_t>(n_aux) * n_aux, 0.0);
      if (j_frac != 0.0) {
        for (int p = 0; p < n_aux; ++p) {
          for (int q = 0; q < n_aux; ++q) {
            Gamma_2c_sph[static_cast<std::size_t>(p) * n_aux + q] =
                -0.5 * j_frac * rho_sph[p] * rho_sph[q];
          }
        }
      }
      if (do_k) {
        for (std::size_t s = 0; s < channels.size(); ++s) {
          const int nc = channels[s].ncols;
          const int nc2 = nc * nc;
          const double w_2c = 0.5 * channels[s].occ_weight * k_frac;
          const auto& sg = channels[s].signs;
          bool has_neg = false;
          for (double sv : sg) {
            if (sv < 0.0) {
              has_neg = true;
              break;
            }
          }
          if (!has_neg) {
            gemm(n_aux, n_aux, nc2, w_2c, M_channels[s].data(), nc2, false, M_channels[s].data(),
                 nc2, true, 1.0, Gamma_2c_sph.data(), n_aux, Backend::Host);
          } else {
            // Apply sg[i]*sg[j] to ONLY ONE operand of Gamma_2c += w_2c * M_signed * M_unsigned^T
            // so cross-sign eigenpairs keep (-1) instead of squaring to (+1), then apply signs
            // in-place to M_channels[s] for Pass 3 (Gamma_3c).
            const int col_blk = std::clamp<int>(
                static_cast<int>((64ULL << 20) /
                                 std::max<std::size_t>(1, static_cast<std::size_t>(n_aux) *
                                                              sizeof(double))),
                1, nc2);
            std::vector<double> M_blk_signed(static_cast<std::size_t>(n_aux) * col_blk);
            double* M = M_channels[s].data();
            for (int c0 = 0; c0 < nc2; c0 += col_blk) {
              const int bw = std::min(col_blk, nc2 - c0);
              for (int p = 0; p < n_aux; ++p) {
                const double* src = M + static_cast<std::size_t>(p) * nc2 + c0;
                double* dst = M_blk_signed.data() + static_cast<std::size_t>(p) * bw;
                for (int k_loc = 0; k_loc < bw; ++k_loc) {
                  const int ij = c0 + k_loc;
                  const int i = ij / nc;
                  const int j = ij - i * nc;
                  dst[k_loc] = src[k_loc] * (sg[i] * sg[j]);
                }
              }
              gemm(n_aux, n_aux, bw, w_2c, M_blk_signed.data(), bw, false, M + c0, nc2, true, 1.0,
                   Gamma_2c_sph.data(), n_aux, Backend::Host);
            }
            for (int p = 0; p < n_aux; ++p) {
              double* mp = M + static_cast<std::size_t>(p) * nc2;
              for (int i = 0; i < nc; ++i) {
                for (int j = 0; j < nc; ++j) {
                  mp[i * nc + j] *= sg[i] * sg[j];
                }
              }
            }
          }
        }
      }

      const std::vector<double> C_aux =
          build_orb_sph_to_cart_mat(aux_basis_, aux_shell_cart_off_, ncart_aux_);
      std::vector<double> Gamma_2c_cart(static_cast<std::size_t>(ncart_aux_) * ncart_aux_, 0.0);
      {
        std::vector<double> tmp(static_cast<std::size_t>(n_aux) * ncart_aux_, 0.0);
        gemm(n_aux, ncart_aux_, n_aux, 1.0, Gamma_2c_sph.data(), n_aux, false, C_aux.data(),
             ncart_aux_, false, 0.0, tmp.data(), ncart_aux_, Backend::Host);
        gemm(ncart_aux_, ncart_aux_, n_aux, 1.0, C_aux.data(), ncart_aux_, true, tmp.data(),
             ncart_aux_, false, 0.0, Gamma_2c_cart.data(), ncart_aux_, Backend::Host);
      }

      #if defined(_OPENMP)
      #pragma omp parallel
      #endif
      {
        std::vector<double> loc(grad.size(), 0.0);
        for (std::size_t ip = 0; ip < aux_lists_.size(); ++ip) {
          const ShellPairData& bra_aux = aux_lists_[ip];
          if (bra_aux.npair() == 0) continue;
          const ShellPairView bv = bra_aux.view();
          for (std::size_t iq = 0; iq <= ip; ++iq) {
            const ShellPairData& ket_aux = aux_lists_[iq];
            if (ket_aux.npair() == 0) continue;
            const ShellPairView kv = ket_aux.view();
            const bool same_class = (ip == iq);
            const long long np_tot =
                static_cast<long long>(bra_aux.npair()) * ket_aux.npair();

            #if defined(_OPENMP)
            #pragma omp for schedule(dynamic, 16) nowait
            #endif
            for (long long pq = 0; pq < np_tot; ++pq) {
              const int P = static_cast<int>(pq / ket_aux.npair());
              const int Q = static_cast<int>(pq - static_cast<long long>(P) * ket_aux.npair());
              if (same_class && Q >= P) continue;
              const int sP = bra_aux.ish[P], sQ = ket_aux.ish[Q];
              const int atP = aux_atom[sP], atQ = aux_atom[sQ];
              if (atP == atQ) continue;
              const int pc0 = aux_shell_cart_off_[sP], qc0 = aux_shell_cart_off_[sQ];
              auto gamma_fn = [&](int ip_c, int iq_c) -> double {
                return 2.0 * Gamma_2c_cart[static_cast<std::size_t>(pc0 + ip_c) * ncart_aux_ +
                                           (qc0 + iq_c)];
              };
              double gP[3];
              generated::eri_2c_grad_contract_fn<double>(bra_aux.la, ket_aux.la, bv, P, kv, Q,
                                                         gamma_fn, gP, pass_omega);
              if (atP >= 0) {
                loc[3 * atP + 0] += gP[0];
                loc[3 * atP + 1] += gP[1];
                loc[3 * atP + 2] += gP[2];
              }
              if (atQ >= 0) {
                loc[3 * atQ + 0] -= gP[0];
                loc[3 * atQ + 1] -= gP[1];
                loc[3 * atQ + 2] -= gP[2];
              }
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
    }

    // Pass 3: Chunked Cartesian 3-center AO derivatives
    std::vector<double> rho_cart(j_frac != 0.0 ? ncart_aux_ : 0, 0.0);
    if (j_frac != 0.0) {
      transform_aux_rows_sph_to_cart(aux_basis_, 0, nsh_aux, 0, 0, 1, aux_shell_cart_off_,
                                     rho_sph.data(), rho_cart.data());
    }

    const std::size_t ncart2 = static_cast<std::size_t>(ncart_orb_) * ncart_orb_;
    std::vector<double> Gamma_chunk;
    std::vector<double> M_cart_chunk;
    std::vector<double> Y_cart_chunk;

    for (const auto& c : chunks) {
      const int bc_cart = c.pc1 - c.pc0;
      Gamma_chunk.resize(static_cast<std::size_t>(bc_cart) * ncart2);
      if (j_frac != 0.0) {
        #if defined(_OPENMP)
        #pragma omp parallel for schedule(static)
        #endif
        for (int p_loc = 0; p_loc < bc_cart; ++p_loc) {
          const double w = j_frac * rho_cart[c.pc0 + p_loc];
          double* g_ptr = Gamma_chunk.data() + static_cast<std::size_t>(p_loc) * ncart2;
          for (std::size_t ab = 0; ab < ncart2; ++ab) g_ptr[ab] = w * D_tot_cart[ab];
        }
      } else {
        std::fill(Gamma_chunk.begin(), Gamma_chunk.end(), 0.0);
      }

      if (do_k) {
        for (std::size_t s = 0; s < channels.size(); ++s) {
          const int nc = channels[s].ncols;
          const int nc2 = nc * nc;
          M_cart_chunk.resize(static_cast<std::size_t>(bc_cart) * nc2);
          transform_aux_rows_sph_to_cart(aux_basis_, c.s0, c.s1, c.q0, c.pc0, nc2,
                                         aux_shell_cart_off_,
                                         M_channels[s].data() + static_cast<std::size_t>(c.q0) * nc2,
                                         M_cart_chunk.data());

          Y_cart_chunk.resize(static_cast<std::size_t>(bc_cart) * nc * ncart_orb_);
          gemm(bc_cart * nc, ncart_orb_, nc, 1.0, M_cart_chunk.data(), nc, false,
               channels[s].C_cart.data(), nc, true, 0.0, Y_cart_chunk.data(), ncart_orb_,
               Backend::Host);

          const double w_3c = -channels[s].occ_weight * k_frac;
          #if defined(_OPENMP)
          #pragma omp parallel for schedule(static)
          #endif
          for (int p_loc = 0; p_loc < bc_cart; ++p_loc) {
            gemm(ncart_orb_, ncart_orb_, nc, w_3c, channels[s].C_cart.data(), nc, false,
                 Y_cart_chunk.data() + static_cast<std::size_t>(p_loc) * nc * ncart_orb_,
                 ncart_orb_, false, 1.0,
                 Gamma_chunk.data() + static_cast<std::size_t>(p_loc) * ncart2, ncart_orb_,
                 Backend::Host);
          }
        }
      }

      #if defined(_OPENMP)
      #pragma omp parallel
      #endif
      {
        std::vector<double> loc(grad.size(), 0.0);
        for (int lP = 0; lP < static_cast<int>(aux_lists_.size()); ++lP) {
          const int pb = c.l_beg[lP];
          const int pe = c.l_end[lP];
          if (pb >= pe) continue;
          const ShellPairData& aux = aux_lists_[lP];
          const ShellPairView av = aux.view();
          const int naux_p = pe - pb;
          const long long total = static_cast<long long>(orb_pairs.size()) * naux_p;

          #if defined(_OPENMP)
          #pragma omp for schedule(dynamic, 16) nowait
          #endif
          for (long long idx = 0; idx < total; ++idx) {
            const std::size_t op_idx = static_cast<std::size_t>(idx / naux_p);
            const int P = pb + static_cast<int>(idx - static_cast<long long>(op_idx) * naux_p);
            const int io = orb_pairs[op_idx].first;
            const int ab = orb_pairs[op_idx].second;
            const ShellPairData& orb = orb_lists_[io];
            const int i = orb.ish[ab], j = orb.jsh[ab], sP = aux.ish[P];
            const int atA = orb_atom[i], atB = orb_atom[j], atP = aux_atom[sP];
            if (atA == atB && atA == atP) continue;
            const double deg_ij = (i == j) ? 1.0 : 2.0;
            const int ca0 = orb_shell_cart_off_[i];
            const int cb0 = orb_shell_cart_off_[j];
            const int pc_loc = aux_shell_cart_off_[sP] - c.pc0;

            auto gamma_fn = [&](int ia, int ib_c, int iP_c) -> double {
              const double* g_slice =
                  Gamma_chunk.data() + static_cast<std::size_t>(pc_loc + iP_c) * ncart2;
              if (i == j) {
                return g_slice[static_cast<std::size_t>(ca0 + ia) * ncart_orb_ + (cb0 + ib_c)];
              }
              return g_slice[static_cast<std::size_t>(ca0 + ia) * ncart_orb_ + (cb0 + ib_c)] +
                     g_slice[static_cast<std::size_t>(cb0 + ib_c) * ncart_orb_ + (ca0 + ia)];
            };

            double g9[9];
            if (orb.la <= 2 && orb.lb <= 2) {
              generated::eri_3c_grad_contract_fn<2, double>(orb.la, orb.lb, lP, orb.view(), ab, av,
                                                            P, gamma_fn, g9, pass_omega);
            } else {
              generated::eri_3c_grad_contract_fn<4, double>(orb.la, orb.lb, lP, orb.view(), ab, av,
                                                            P, gamma_fn, g9, pass_omega);
            }
            (void)deg_ij;
            if (atA >= 0) {
              loc[3 * atA + 0] += g9[0];
              loc[3 * atA + 1] += g9[1];
              loc[3 * atA + 2] += g9[2];
            }
            if (atB >= 0) {
              loc[3 * atB + 0] += g9[3];
              loc[3 * atB + 1] += g9[4];
              loc[3 * atB + 2] += g9[5];
            }
            if (atP >= 0) {
              loc[3 * atP + 0] += g9[6];
              loc[3 * atP + 1] += g9[7];
              loc[3 * atP + 2] += g9[8];
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
    }
  };

  run_host_grad_pass(0.0, j_fraction, k_sr);
  if (omega > 0.0 && k_lr_delta != 0.0) {
    run_host_grad_pass(omega, 0.0, k_lr_delta);
  }
  return grad;
}

namespace {

std::vector<double> jk_nuclear_grad_impl(const Basis& basis, std::span<const Atom> atoms,
                                         const double* dm, double j_fraction, double k_sr,
                                         double k_lr, double omega, Backend backend,
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

  auto build_tasks = [&](double j_frac, double k_frac) {
    std::vector<detail::GradQuartetTask> out;
    const double abs_j = std::abs(j_frac);
    const double abs_k = std::abs(k_frac);
    const double global_2pdm =
        std::max(4.0 * abs_j * global_d_max * global_d_max, abs_k * global_d_max * global_d_max);
    if (global_2pdm == 0.0) return out;
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
            const double d2 = std::max({
                4.0 * abs_j * d_shell_max[i * nsh + j] * d_shell_max[k * nsh + l],
                abs_k * d_shell_max[i * nsh + k] * d_shell_max[j * nsh + l],
                abs_k * d_shell_max[i * nsh + l] * d_shell_max[j * nsh + k],
            });
            if (q_ab * q_cd * d2 < schwarz_cutoff) continue;

            const bool same_PQ = (bp == kp) && (P == Q);
            const double deg_kl = (k == l) ? 1.0 : 2.0;
            const double s_PQ = same_PQ ? 1.0 : 2.0;
            out.push_back(detail::GradQuartetTask{bp, P, kp, Q, deg_ij * deg_kl * s_PQ});
          }
        }
      }
    }
    return out;
  };

  const double k_lr_delta = (omega > 0.0) ? (k_lr - k_sr) : 0.0;
  const auto tasks_sr = build_tasks(j_fraction, k_sr);
  const auto tasks_lr = (omega > 0.0 && k_lr_delta != 0.0)
                            ? build_tasks(0.0, k_lr_delta)
                            : std::vector<detail::GradQuartetTask>{};
  if (tasks_sr.empty() && tasks_lr.empty()) return grad;

  if (backend == Backend::Host) {
    auto run_host_pass = [&](const std::vector<detail::GradQuartetTask>& tasks, double j_frac,
                             double k_frac, double pass_omega) {
      if (tasks.empty()) return;
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
              t.w_sym, k_frac, g12, pass_omega, j_frac);
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
    };

    run_host_pass(tasks_sr, j_fraction, k_sr, 0.0);
    run_host_pass(tasks_lr, 0.0, k_lr_delta, omega);
    return grad;
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::run_jk_nuclear_grad_cuda(pair_lists, tasks_sr, tasks_lr, shell_cart_off, shell_atom,
                                     ncart_tot, static_cast<int>(atoms.size()), d_cart, j_fraction,
                                     k_sr, k_lr_delta, omega, grad);
    return grad;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  throw std::runtime_error("gpu4dft was built without HIP");
}

std::vector<double> jk_nuclear_grad_uhf_impl(const Basis& basis, std::span<const Atom> atoms,
                                             const double* dm_alpha, const double* dm_beta,
                                             double j_fraction, double k_sr, double k_lr,
                                             double omega, Backend backend,
                                             double schwarz_cutoff) {
  if (!backend_available(backend)) {
    throw std::runtime_error(std::string("backend '") + to_string(backend) + "' is not available");
  }
  if (basis.max_l() > kMaxSupportedL) {
    throw NotImplemented("jk_nuclear_grad_uhf supports up to g shells (l <= 4)");
  }
  std::vector<double> grad(atoms.size() * 3, 0.0);
  if (atoms.empty() || basis.nao() == 0) return grad;
  if (dm_alpha == nullptr || dm_beta == nullptr) {
    throw std::invalid_argument("jk_nuclear_grad_uhf: dm_alpha and dm_beta must be non-null");
  }

  const std::size_t nao2 = static_cast<std::size_t>(basis.nao()) * basis.nao();
  std::vector<double> dm_tot(nao2, 0.0);
  for (std::size_t i = 0; i < nao2; ++i) dm_tot[i] = dm_alpha[i] + dm_beta[i];

  const double k_lr_delta = (omega > 0.0) ? (k_lr - k_sr) : 0.0;
  if (k_sr == 0.0 && k_lr_delta == 0.0) {
    return jk_nuclear_grad_impl(basis, atoms, dm_tot.data(), j_fraction, 0.0, 0.0, 0.0, backend,
                                schwarz_cutoff);
  }

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

  std::vector<double> d_tot_cart, d_a_cart, d_b_cart;
  std::vector<double> d_sh_tot, d_sh_a, d_sh_b;
  double g_tot = 0.0, g_a = 0.0, g_b = 0.0;
  prepare_cartesian_density(basis, shell_cart_off, ncart_tot, dm_tot.data(), d_tot_cart, d_sh_tot,
                            g_tot, /*symmetrize_dm=*/true);
  prepare_cartesian_density(basis, shell_cart_off, ncart_tot, dm_alpha, d_a_cart, d_sh_a, g_a,
                            /*symmetrize_dm=*/true);
  prepare_cartesian_density(basis, shell_cart_off, ncart_tot, dm_beta, d_b_cart, d_sh_b, g_b,
                            /*symmetrize_dm=*/true);
  if (g_tot == 0.0 && g_a == 0.0 && g_b == 0.0) return grad;

  std::vector<ShellPairData> pair_lists;
  for (int la = 0; la <= basis.max_l(); ++la) {
    for (int lb = 0; lb <= la; ++lb) {
      ShellPairData pd = build_shell_pairs(basis, la, lb, 1e-14);
      if (pd.npair() > 0) pair_lists.push_back(std::move(pd));
    }
  }

  auto build_tasks = [&](double j_frac, double k_frac) {
    std::vector<detail::GradQuartetTask> out;
    const double abs_j = std::abs(j_frac);
    const double abs_k = std::abs(k_frac);
    const double global_2pdm =
        std::max(4.0 * abs_j * g_tot * g_tot, 2.0 * abs_k * (g_a * g_a + g_b * g_b));
    if (global_2pdm == 0.0) return out;
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
            const double d2 = std::max({
                4.0 * abs_j * d_sh_tot[i * nsh + j] * d_sh_tot[k * nsh + l],
                2.0 * abs_k * (d_sh_a[i * nsh + k] * d_sh_a[j * nsh + l] +
                               d_sh_b[i * nsh + k] * d_sh_b[j * nsh + l]),
                2.0 * abs_k * (d_sh_a[i * nsh + l] * d_sh_a[j * nsh + k] +
                               d_sh_b[i * nsh + l] * d_sh_b[j * nsh + k]),
            });
            if (q_ab * q_cd * d2 < schwarz_cutoff) continue;

            const bool same_PQ = (bp == kp) && (P == Q);
            const double deg_kl = (k == l) ? 1.0 : 2.0;
            const double s_PQ = same_PQ ? 1.0 : 2.0;
            out.push_back(detail::GradQuartetTask{bp, P, kp, Q, deg_ij * deg_kl * s_PQ});
          }
        }
      }
    }
    return out;
  };

  const auto tasks_sr = build_tasks(j_fraction, k_sr);
  const auto tasks_lr = (omega > 0.0 && k_lr_delta != 0.0)
                            ? build_tasks(0.0, k_lr_delta)
                            : std::vector<detail::GradQuartetTask>{};
  if (tasks_sr.empty() && tasks_lr.empty()) return grad;

  if (backend == Backend::Host) {
    auto run_host_pass = [&](const std::vector<detail::GradQuartetTask>& tasks, double j_frac,
                             double k_frac, double pass_omega) {
      if (tasks.empty()) return;
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
          generated::eri_quartet_grad_direct_dm_uhf<double>(
              bra.la, bra.lb, ket.la, ket.lb, bra.view(), t.P, ket.view(), t.Q, d_tot_cart.data(),
              d_a_cart.data(), d_b_cart.data(), ncart_tot, shell_cart_off[i], shell_cart_off[j],
              shell_cart_off[k], shell_cart_off[l], t.w_sym, k_frac, g12, pass_omega, j_frac);
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
    };

    run_host_pass(tasks_sr, j_fraction, k_sr, 0.0);
    run_host_pass(tasks_lr, 0.0, k_lr_delta, omega);
    return grad;
  }

  if (backend == Backend::Cuda) {
#if defined(GPU4DFT_HAVE_CUDA)
    detail::run_jk_nuclear_grad_uhf_cuda(pair_lists, tasks_sr, tasks_lr, shell_cart_off, shell_atom,
                                         ncart_tot, static_cast<int>(atoms.size()), d_tot_cart,
                                         d_a_cart, d_b_cart, j_fraction, k_sr, k_lr_delta, omega,
                                         grad);
    return grad;
#else
    throw std::runtime_error("gpu4dft was built without CUDA");
#endif
  }

  throw std::runtime_error("gpu4dft was built without HIP");
}

}  // namespace

std::vector<double> jk_nuclear_grad(const Basis& basis, std::span<const Atom> atoms,
                                    const double* dm, double k_fraction, Backend backend,
                                    double schwarz_cutoff) {
  return jk_nuclear_grad_impl(basis, atoms, dm, 1.0, k_fraction, k_fraction, 0.0, backend,
                              schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad(const Basis& basis, const double* dm, double k_fraction,
                                    Backend backend, double schwarz_cutoff) {
  const std::vector<Atom> atoms = infer_atoms_from_basis(basis);
  return jk_nuclear_grad(basis, std::span<const Atom>(atoms), dm, k_fraction, backend,
                         schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad_rsh(const Basis& basis, std::span<const Atom> atoms,
                                        const double* dm, double k_sr, double k_lr, double omega,
                                        Backend backend, double schwarz_cutoff) {
  return jk_nuclear_grad_impl(basis, atoms, dm, 1.0, k_sr, k_lr, omega, backend, schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad_rsh(const Basis& basis, const double* dm, double k_sr,
                                        double k_lr, double omega, Backend backend,
                                        double schwarz_cutoff) {
  const std::vector<Atom> atoms = infer_atoms_from_basis(basis);
  return jk_nuclear_grad_rsh(basis, std::span<const Atom>(atoms), dm, k_sr, k_lr, omega, backend,
                             schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad_uhf(const Basis& basis, std::span<const Atom> atoms,
                                        const double* dm_alpha, const double* dm_beta,
                                        double k_fraction, Backend backend, double schwarz_cutoff) {
  return jk_nuclear_grad_uhf_impl(basis, atoms, dm_alpha, dm_beta, 1.0, k_fraction, k_fraction, 0.0,
                                  backend, schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad_uhf(const Basis& basis, const double* dm_alpha,
                                        const double* dm_beta, double k_fraction, Backend backend,
                                        double schwarz_cutoff) {
  const std::vector<Atom> atoms = infer_atoms_from_basis(basis);
  return jk_nuclear_grad_uhf(basis, std::span<const Atom>(atoms), dm_alpha, dm_beta, k_fraction,
                             backend, schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad_uhf_rsh(const Basis& basis, std::span<const Atom> atoms,
                                            const double* dm_alpha, const double* dm_beta,
                                            double k_sr, double k_lr, double omega,
                                            Backend backend, double schwarz_cutoff) {
  return jk_nuclear_grad_uhf_impl(basis, atoms, dm_alpha, dm_beta, 1.0, k_sr, k_lr, omega,
                                  backend, schwarz_cutoff);
}

std::vector<double> jk_nuclear_grad_uhf_rsh(const Basis& basis, const double* dm_alpha,
                                            const double* dm_beta, double k_sr, double k_lr,
                                            double omega, Backend backend, double schwarz_cutoff) {
  const std::vector<Atom> atoms = infer_atoms_from_basis(basis);
  return jk_nuclear_grad_uhf_rsh(basis, std::span<const Atom>(atoms), dm_alpha, dm_beta, k_sr,
                                 k_lr, omega, backend, schwarz_cutoff);
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
