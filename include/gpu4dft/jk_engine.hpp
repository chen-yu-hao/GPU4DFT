#pragma once

#include <memory>
#include <span>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/boys.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/math.hpp"
#include "gpu4dft/one_electron.hpp"
#include "gpu4dft/rys.hpp"
#include "gpu4dft/shell_pairs.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

struct JKOptions {
  Backend backend = Backend::Cuda;
  Precision precision = Precision::FP64;
  double schwarz_cutoff = 1e-12;
  double fp32_threshold = 1e-6;
  bool compute_j = true;
  bool compute_k = true;
  double k_scale = 1.0;
};

namespace detail {
struct DirectJKCudaCache;
struct RIJKCudaCache;
}  // namespace detail

// Direct on-the-fly Coulomb (J) and Exchange (K) engine without O(nao^4) ERI storage.
// Prebuilds and caches SoA shell pairs + Schwarz upper bounds on Host and GPU,
// supports Hermite J-engine contraction, Schwarz + density-weighted screening,
// incremental Fock updates, GPU atomic dynamic work-queue load balancing, and
// dynamic per-quartet FP64/FP32 mixed precision (Precision::Mixed).
class DirectJKEngine {
 public:
  explicit DirectJKEngine(const Basis& basis, double prim_cutoff = 1e-14);
  ~DirectJKEngine();

  DirectJKEngine(DirectJKEngine&&) noexcept;
  DirectJKEngine& operator=(DirectJKEngine&&) noexcept;
  DirectJKEngine(const DirectJKEngine&) = delete;
  DirectJKEngine& operator=(const DirectJKEngine&) = delete;

  int nao() const { return basis_.nao(); }
  const Basis& basis() const { return basis_; }
  const std::vector<ShellPairData>& shell_pair_lists() const { return pair_lists_; }

  // Evaluates J and/or K (scaled by `opts.k_scale`) into row-major [nao x nao]
  // buffers `vj_out` and `vk_out` (either pointer may be nullptr when disabled).
  void compute_jk(const double* dm, double* vj_out, double* vk_out,
                  const JKOptions& opts = {}) const;

  // Incremental Fock helper: evaluates Delta J(dm_new - dm_old) and
  // Delta K(dm_new - dm_old) using density-weighted Schwarz screening and
  // accumulates them into `vj_accum` and `vk_accum`.
  void compute_incremental_jk(const double* dm_new, const double* dm_old, double* vj_accum,
                              double* vk_accum, const JKOptions& opts = {}) const;

 private:
  Basis basis_;
  std::vector<ShellPairData> pair_lists_;
  std::vector<int> shell_cart_offset_;
  int ncart_total_ = 0;
  mutable detail::DirectJKCudaCache* cuda_cache_ = nullptr;
};

// Resolution-of-the-Identity (Density Fitting) J/K engine using 3-center (ab|P)
// and 2-center (P|Q) Coulomb integrals, Cholesky / symmetric inverse metric
// factorization B_{P, ab} = L^{-1} (ab|P), and dense cuBLAS / OpenMP GEMMs.
class RIJKEngine {
 public:
  RIJKEngine(const Basis& basis, const Basis& aux_basis, Backend backend = Backend::Cuda);
  ~RIJKEngine();

  RIJKEngine(RIJKEngine&&) noexcept;
  RIJKEngine& operator=(RIJKEngine&&) noexcept;
  RIJKEngine(const RIJKEngine&) = delete;
  RIJKEngine& operator=(const RIJKEngine&) = delete;

  int nao() const { return basis_.nao(); }
  int naux() const { return aux_basis_.nao(); }
  const std::vector<double>& b_tensor() const { return b_tensor_; }

  void compute_jk(const double* dm, double* vj_out, double* vk_out,
                  const JKOptions& opts = {}) const;

 private:
  Basis basis_;
  Basis aux_basis_;
  std::vector<double> b_tensor_;       // [naux * nao * nao]
  std::vector<double> b_perm_tensor_;  // [nao * naux * nao] for 2-GEMM RI-K
  mutable detail::RIJKCudaCache* cuda_cache_ = nullptr;
};

// Analytical 2-electron nuclear gradient [natom * 3] of:
//   E_2e = 0.5 * Tr(D * J(D)) - 0.25 * k_fraction * Tr(D * K(D)).
// Infers atom centers in order of first shell appearance in `basis`.
std::vector<double> jk_nuclear_grad(const Basis& basis, const double* dm, double k_fraction,
                                    Backend backend = Backend::Cuda,
                                    double schwarz_cutoff = 1e-12);

// Overload accepting an explicit atom list (handles basisless / ghost atoms).
std::vector<double> jk_nuclear_grad(const Basis& basis, std::span<const Atom> atoms,
                                    const double* dm, double k_fraction,
                                    Backend backend = Backend::Cuda,
                                    double schwarz_cutoff = 1e-12);

namespace detail {

inline constexpr double kJKTwoOverSqrtPi = 1.12837916709551257390;

struct JKTaskTile {
  int bp;       // bra pair-list index
  int P;        // bra pair index in pair_lists[bp]
  int kp;       // ket pair-list index
  int Q_begin;  // start ket pair index
  int Q_end;    // end ket pair index (exclusive)
};

struct GradQuartetTask {
  int bp;
  int P;
  int kp;
  int Q;
  double w_sym;
};

template <typename R>
GPU4DFT_HD inline void jk_hermite_1d(int max_a, int max_b, R inv_2p, R PA, R PB,
                                     R E[5][5][9], bool negate_odd) {
  const int max_t = max_a + max_b;
  for (int i = 0; i <= max_a; ++i)
    for (int j = 0; j <= max_b; ++j)
      for (int t = 0; t <= max_t; ++t) E[i][j][t] = R(0);
  E[0][0][0] = R(1);
  for (int i = 0; i < max_a; ++i) {
    for (int t = 0; t <= i + 1; ++t) {
      R val = PA * E[i][0][t];
      if (t > 0) val += inv_2p * E[i][0][t - 1];
      if (t + 1 <= i) val += R(t + 1) * E[i][0][t + 1];
      E[i + 1][0][t] = val;
    }
  }
  for (int i = 0; i <= max_a; ++i) {
    for (int j = 0; j < max_b; ++j) {
      const int deg = i + j;
      for (int t = 0; t <= deg + 1; ++t) {
        R val = PB * E[i][j][t];
        if (t > 0) val += inv_2p * E[i][j][t - 1];
        if (t + 1 <= deg) val += R(t + 1) * E[i][j][t + 1];
        E[i][j + 1][t] = val;
      }
    }
  }
  if (negate_odd) {
    for (int i = 0; i <= max_a; ++i)
      for (int j = 0; j <= max_b; ++j)
        for (int t = 1; t <= i + j; t += 2) E[i][j][t] = -E[i][j][t];
  }
}

// Hermite J-engine path for pure Coulomb (J only) when L = la+lb+lc+ld <= 4.
// Contracts D_cart -> Hermite density -> R_{t+tp,u+up,v+vp} -> Hermite potential -> J_cart
// without forming 4-center Cartesian integrals.
template <typename R, typename AccumFn>
GPU4DFT_HD inline void eval_hermite_j_quartet(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, double w_J, const double* D_cart, int ncart_tot, int off_a, int off_b,
    int off_c, int off_d, AccumFn accum_j) {
  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);
  const int lab = la + lb;
  const int lcd = lc + ld;
  const int L = lab + lcd;

  CartTriple ta[15], tb[15], tc[15], td[15];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib_c = 0; ib_c < ncb; ++ib_c) tb[ib_c] = cart_triple(lb, ib_c);
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  double j_ab[225] = {0.0};
  double j_cd[225] = {0.0};

  const R Ax = static_cast<R>(bra.Ax[ib]), Ay = static_cast<R>(bra.Ay[ib]), Az = static_cast<R>(bra.Az[ib]);
  const R Bx = static_cast<R>(bra.Bx[ib]), By = static_cast<R>(bra.By[ib]), Bz = static_cast<R>(bra.Bz[ib]);
  const R Cx = static_cast<R>(ket.Ax[ik]), Cy = static_cast<R>(ket.Ay[ik]), Cz = static_cast<R>(ket.Az[ik]);
  const R Dx = static_cast<R>(ket.Bx[ik]), Dy = static_cast<R>(ket.By[ik]), Dz = static_cast<R>(ket.Bz[ik]);

  const int bra_begin = bra.prim_offset[ib], bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik], ket_end = ket.prim_offset[ik + 1];

  for (int ip = bra_begin; ip < bra_end; ++ip) {
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    const R inv_2p = R(0.5) / p;

    R Ex_ab[5][5][9], Ey_ab[5][5][9], Ez_ab[5][5][9];
    jk_hermite_1d<R>(la, lb, inv_2p, Px - Ax, Px - Bx, Ex_ab, false);
    jk_hermite_1d<R>(la, lb, inv_2p, Py - Ay, Py - By, Ey_ab, false);
    jk_hermite_1d<R>(la, lb, inv_2p, Pz - Az, Pz - Bz, Ez_ab, false);

    R D_bra_h[5][5][5];
    R V_bra_h[5][5][5];
    for (int t = 0; t <= lab; ++t)
      for (int u = 0; u <= lab - t; ++u)
        for (int v = 0; v <= lab - t - u; ++v) {
          D_bra_h[t][u][v] = R(0);
          V_bra_h[t][u][v] = R(0);
        }

    if (!same_PQ) {
      for (int ia = 0; ia < nca; ++ia) {
        for (int ib_c = 0; ib_c < ncb; ++ib_c) {
          const R dab = static_cast<R>(D_cart[(off_a + ia) * ncart_tot + (off_b + ib_c)]);
          for (int t = 0; t <= ta[ia].lx + tb[ib_c].lx; ++t) {
            const R ex = dab * Ex_ab[ta[ia].lx][tb[ib_c].lx][t];
            for (int u = 0; u <= ta[ia].ly + tb[ib_c].ly; ++u) {
              const R exy = ex * Ey_ab[ta[ia].ly][tb[ib_c].ly][u];
              for (int v = 0; v <= ta[ia].lz + tb[ib_c].lz; ++v) {
                D_bra_h[t][u][v] += exy * Ez_ab[ta[ia].lz][tb[ib_c].lz][v];
              }
            }
          }
        }
      }
    }

    for (int kp = ket_begin; kp < ket_end; ++kp) {
      const R q = static_cast<R>(ket.exponent[kp]);
      const R Qx = static_cast<R>(ket.Px[kp]);
      const R Qy = static_cast<R>(ket.Py[kp]);
      const R Qz = static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R inv_2q = R(0.5) / q;

      R Ex_cd[5][5][9], Ey_cd[5][5][9], Ez_cd[5][5][9];
      jk_hermite_1d<R>(lc, ld, inv_2q, Qx - Cx, Qx - Dx, Ex_cd, true);
      jk_hermite_1d<R>(lc, ld, inv_2q, Qy - Cy, Qy - Dy, Ey_cd, true);
      jk_hermite_1d<R>(lc, ld, inv_2q, Qz - Cz, Qz - Dz, Ez_cd, true);

      R D_ket_h[5][5][5];
      for (int tp = 0; tp <= lcd; ++tp)
        for (int up = 0; up <= lcd - tp; ++up)
          for (int vp = 0; vp <= lcd - tp - up; ++vp) D_ket_h[tp][up][vp] = R(0);

      for (int ic = 0; ic < ncc; ++ic) {
        for (int id = 0; id < ncd; ++id) {
          const R dcd = static_cast<R>(D_cart[(off_c + ic) * ncart_tot + (off_d + id)]);
          for (int tp = 0; tp <= tc[ic].lx + td[id].lx; ++tp) {
            const R ex = dcd * Ex_cd[tc[ic].lx][td[id].lx][tp];
            for (int up = 0; up <= tc[ic].ly + td[id].ly; ++up) {
              const R exy = ex * Ey_cd[tc[ic].ly][td[id].ly][up];
              for (int vp = 0; vp <= tc[ic].lz + td[id].lz; ++vp) {
                D_ket_h[tp][up][vp] += exy * Ez_cd[tc[ic].lz][td[id].lz][vp];
              }
            }
          }
        }
      }

      const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const R rho = p * q / (p + q);
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      R F[5];
      boys_vec<R>(L, T, F);
      const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

      R R_cur[5][5][5], R_nxt[5][5][5];
      for (int t = 0; t <= L; ++t)
        for (int u = 0; u <= L - t; ++u)
          for (int v = 0; v <= L - t - u; ++v) R_cur[t][u][v] = R(0);

      R pow_m2rho[5];
      pow_m2rho[0] = scale;
      for (int m = 1; m <= L; ++m) pow_m2rho[m] = pow_m2rho[m - 1] * (R(-2) * rho);

      R_cur[0][0][0] = pow_m2rho[L] * F[L];
      for (int m = L - 1; m >= 0; --m) {
        const int max_deg = L - m;
        R_nxt[0][0][0] = pow_m2rho[m] * F[m];
        for (int deg = 1; deg <= max_deg; ++deg) {
          for (int t = 0; t <= deg; ++t) {
            for (int u = 0; u <= deg - t; ++u) {
              const int v = deg - t - u;
              if (t > 0) {
                R_nxt[t][u][v] =
                    PQx * R_cur[t - 1][u][v] + (t > 1 ? R(t - 1) * R_cur[t - 2][u][v] : R(0));
              } else if (u > 0) {
                R_nxt[0][u][v] =
                    PQy * R_cur[0][u - 1][v] + (u > 1 ? R(u - 1) * R_cur[0][u - 2][v] : R(0));
              } else {
                R_nxt[0][0][v] =
                    PQz * R_cur[0][0][v - 1] + (v > 1 ? R(v - 1) * R_cur[0][0][v - 2] : R(0));
              }
            }
          }
        }
        for (int deg = 0; deg <= max_deg; ++deg)
          for (int t = 0; t <= deg; ++t)
            for (int u = 0; u <= deg - t; ++u) {
              const int v = deg - t - u;
              R_cur[t][u][v] = R_nxt[t][u][v];
            }
      }

      R V_ket_h[5][5][5];
      if (!same_PQ) {
        for (int tp = 0; tp <= lcd; ++tp)
          for (int up = 0; up <= lcd - tp; ++up)
            for (int vp = 0; vp <= lcd - tp - up; ++vp) V_ket_h[tp][up][vp] = R(0);
      }

      for (int t = 0; t <= lab; ++t) {
        for (int u = 0; u <= lab - t; ++u) {
          for (int v = 0; v <= lab - t - u; ++v) {
            R sum_k = R(0);
            const R db = D_bra_h[t][u][v];
            for (int tp = 0; tp <= lcd; ++tp) {
              for (int up = 0; up <= lcd - tp; ++up) {
                for (int vp = 0; vp <= lcd - tp - up; ++vp) {
                  const R r_val = R_cur[t + tp][u + up][v + vp];
                  sum_k += r_val * D_ket_h[tp][up][vp];
                  if (!same_PQ) {
                    V_ket_h[tp][up][vp] += r_val * db;
                  }
                }
              }
            }
            V_bra_h[t][u][v] += sum_k;
          }
        }
      }

      if (!same_PQ) {
        for (int ic = 0; ic < ncc; ++ic) {
          for (int id = 0; id < ncd; ++id) {
            R sum = R(0);
            for (int tp = 0; tp <= tc[ic].lx + td[id].lx; ++tp) {
              const R ex = Ex_cd[tc[ic].lx][td[id].lx][tp];
              for (int up = 0; up <= tc[ic].ly + td[id].ly; ++up) {
                const R exy = ex * Ey_cd[tc[ic].ly][td[id].ly][up];
                for (int vp = 0; vp <= tc[ic].lz + td[id].lz; ++vp) {
                  sum += exy * Ez_cd[tc[ic].lz][td[id].lz][vp] * V_ket_h[tp][up][vp];
                }
              }
            }
            j_cd[ic * ncd + id] += static_cast<double>(sum);
          }
        }
      }
    }

    for (int ia = 0; ia < nca; ++ia) {
      for (int ib_c = 0; ib_c < ncb; ++ib_c) {
        R sum = R(0);
        for (int t = 0; t <= ta[ia].lx + tb[ib_c].lx; ++t) {
          const R ex = Ex_ab[ta[ia].lx][tb[ib_c].lx][t];
          for (int u = 0; u <= ta[ia].ly + tb[ib_c].ly; ++u) {
            const R exy = ex * Ey_ab[ta[ia].ly][tb[ib_c].ly][u];
            for (int v = 0; v <= ta[ia].lz + tb[ib_c].lz; ++v) {
              sum += exy * Ez_ab[ta[ia].lz][tb[ib_c].lz][v] * V_bra_h[t][u][v];
            }
          }
        }
        j_ab[ia * ncb + ib_c] += static_cast<double>(sum);
      }
    }
  }

  for (int ia = 0; ia < nca; ++ia) {
    for (int ib_c = 0; ib_c < ncb; ++ib_c) {
      accum_j(off_a + ia, off_b + ib_c, w_J * j_ab[ia * ncb + ib_c]);
    }
  }
  if (!same_PQ) {
    for (int ic = 0; ic < ncc; ++ic) {
      for (int id = 0; id < ncd; ++id) {
        accum_j(off_c + ic, off_d + id, w_J * j_cd[ic * ncd + id]);
      }
    }
  }
}

template <typename R>
GPU4DFT_HD inline void jk_rys_1d(int la, int lb, int lc, int ld, R C00, R D00, R B10, R B01,
                                 R B00, R AB, R CD, R I[5][5][5][5]) {
  const int lab = la + lb;
  const int lcd = lc + ld;
  R G[9][9];
  G[0][0] = R(1);
  for (int n = 0; n < lab; ++n) {
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? R(n) * B10 * G[n - 1][0] : R(0));
  }
  for (int m = 0; m < lcd; ++m) {
    for (int n = 0; n <= lab; ++n) {
      R val = D00 * G[n][m];
      if (m > 0) val += R(m) * B01 * G[n][m - 1];
      if (n > 0) val += R(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }
  }

  R H[5][5][9];
  R cur[9][9];
  for (int n = 0; n <= lab; ++n)
    for (int m = 0; m <= lcd; ++m) cur[n][m] = G[n][m];
  for (int j = 0; j <= lb; ++j) {
    for (int i = 0; i <= la; ++i)
      for (int m = 0; m <= lcd; ++m) H[i][j][m] = cur[i][m];
    if (j < lb) {
      for (int i = 0; i <= lab - j - 1; ++i)
        for (int m = 0; m <= lcd; ++m) cur[i][m] = cur[i + 1][m] + AB * cur[i][m];
    }
  }

  for (int i = 0; i <= la; ++i) {
    for (int j = 0; j <= lb; ++j) {
      R kcur[9];
      for (int m = 0; m <= lcd; ++m) kcur[m] = H[i][j][m];
      for (int l = 0; l <= ld; ++l) {
        for (int k = 0; k <= lc; ++k) I[i][j][k][l] = kcur[k];
        if (l < ld) {
          for (int k = 0; k <= lcd - l - 1; ++k) kcur[k] = kcur[k + 1] + CD * kcur[k];
        }
      }
    }
  }
}

// General Direct J + K quartet contraction for any 0 <= la, lb, lc, ld <= 4,
// parameterized on primitive evaluation precision `R` (`double` or `float`)
// with `double` accumulation and bounded stack memory.
template <typename R, typename AccumJFn, typename AccumKFn>
GPU4DFT_HD inline void eval_direct_jk_quartet(
    int la, int lb, int lc, int ld, const ShellPairView& bra, int ib, const ShellPairView& ket,
    int ik, bool same_PQ, bool do_j, bool do_k, double w_J, double w_K, const double* D_cart,
    int ncart_tot, int off_a, int off_b, int off_c, int off_d, AccumJFn accum_j,
    AccumKFn accum_k) {
  if (do_j && !do_k && (la + lb + lc + ld) <= 4) {
    eval_hermite_j_quartet<R>(la, lb, lc, ld, bra, ib, ket, ik, same_PQ, w_J, D_cart, ncart_tot,
                              off_a, off_b, off_c, off_d, accum_j);
    return;
  }

  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);
  double j_ab[225] = {0.0};
  double j_cd[225] = {0.0};
  double k_ac[225] = {0.0};
  double k_bc[225] = {0.0};
  double k_ad[225] = {0.0};
  double k_bd[225] = {0.0};

  const int bra_begin = bra.prim_offset[ib], bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik], ket_end = ket.prim_offset[ik + 1];

  if (la == 0 && lb == 0 && lc == 0 && ld == 0) {
    double val = 0.0;
    for (int ip = bra_begin; ip < bra_end; ++ip) {
      const R p = static_cast<R>(bra.exponent[ip]);
      const R Px = static_cast<R>(bra.Px[ip]);
      const R Py = static_cast<R>(bra.Py[ip]);
      const R Pz = static_cast<R>(bra.Pz[ip]);
      const R Cab = static_cast<R>(bra.prefactor[ip]);
      for (int kp = ket_begin; kp < ket_end; ++kp) {
        const R q = static_cast<R>(ket.exponent[kp]);
        const R PQx = Px - static_cast<R>(ket.Px[kp]);
        const R PQy = Py - static_cast<R>(ket.Py[kp]);
        const R PQz = Pz - static_cast<R>(ket.Pz[kp]);
        const R Ccd = static_cast<R>(ket.prefactor[kp]);
        const R rho = p * q / (p + q);
        const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
        val += static_cast<double>(Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho) * boys_f0(T));
      }
    }
    if (do_j) {
      accum_j(off_a, off_b, w_J * val * D_cart[off_c * ncart_tot + off_d]);
      if (!same_PQ) accum_j(off_c, off_d, w_J * val * D_cart[off_a * ncart_tot + off_b]);
    }
    if (do_k) {
      const double wk = w_K * val;
      accum_k(off_a, off_c, wk * D_cart[off_b * ncart_tot + off_d]);
      accum_k(off_b, off_c, wk * D_cart[off_a * ncart_tot + off_d]);
      accum_k(off_a, off_d, wk * D_cart[off_b * ncart_tot + off_c]);
      accum_k(off_b, off_d, wk * D_cart[off_a * ncart_tot + off_c]);
    }
    return;
  }

  const int n_roots = (la + lb + lc + ld) / 2 + 1;
  const double Ax = bra.Ax[ib], Ay = bra.Ay[ib], Az = bra.Az[ib];
  const double Bx = bra.Bx[ib], By = bra.By[ib], Bz = bra.Bz[ib];
  const double Cx = ket.Ax[ik], Cy = ket.Ay[ik], Cz = ket.Az[ik];
  const double Dx = ket.Bx[ik], Dy = ket.By[ik], Dz = ket.Bz[ik];
  const R ABx = static_cast<R>(Ax - Bx), ABy = static_cast<R>(Ay - By), ABz = static_cast<R>(Az - Bz);
  const R CDx = static_cast<R>(Cx - Dx), CDy = static_cast<R>(Cy - Dy), CDz = static_cast<R>(Cz - Dz);

  CartTriple ta[15], tb[15], tc[15], td[15];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib_c = 0; ib_c < ncb; ++ib_c) tb[ib_c] = cart_triple(lb, ib_c);
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  for (int ip = bra_begin; ip < bra_end; ++ip) {
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    const R PAx = Px - static_cast<R>(Ax);
    const R PAy = Py - static_cast<R>(Ay);
    const R PAz = Pz - static_cast<R>(Az);
    const R inv_2p = R(0.5) / p;

    for (int kp = ket_begin; kp < ket_end; ++kp) {
      const R q = static_cast<R>(ket.exponent[kp]);
      const R Qx = static_cast<R>(ket.Px[kp]);
      const R Qy = static_cast<R>(ket.Py[kp]);
      const R Qz = static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const R pq_sum = p + q;
      const R rho = p * q / pq_sum;
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      const R scale = Cab * Ccd * R(kJKTwoOverSqrtPi) * math::sqrt(rho);

      R u_roots[9];
      R w_weights[9];
      rys_roots_weights<R>(n_roots, T, u_roots, w_weights);

      const R inv_2q = R(0.5) / q;
      const R inv_2pq = R(0.5) / pq_sum;
      const R rho_over_p = rho / p;
      const R rho_over_q = rho / q;
      const R QCx = Qx - static_cast<R>(Cx);
      const R QCy = Qy - static_cast<R>(Cy);
      const R QCz = Qz - static_cast<R>(Cz);

      for (int r = 0; r < n_roots; ++r) {
        const R u = u_roots[r];
        const R Wr = scale * w_weights[r];
        const R B00 = inv_2pq * u;
        const R B10 = inv_2p * (R(1) - rho_over_p * u);
        const R B01 = inv_2q * (R(1) - rho_over_q * u);

        R Ix[5][5][5][5], Iy[5][5][5][5], Iz[5][5][5][5];
        jk_rys_1d<R>(la, lb, lc, ld, PAx - rho_over_p * u * PQx, QCx + rho_over_q * u * PQx, B10,
                     B01, B00, ABx, CDx, Ix);
        jk_rys_1d<R>(la, lb, lc, ld, PAy - rho_over_p * u * PQy, QCy + rho_over_q * u * PQy, B10,
                     B01, B00, ABy, CDy, Iy);
        jk_rys_1d<R>(la, lb, lc, ld, PAz - rho_over_p * u * PQz, QCz + rho_over_q * u * PQz, B10,
                     B01, B00, ABz, CDz, Iz);

        for (int ia = 0; ia < nca; ++ia) {
          for (int ib_c = 0; ib_c < ncb; ++ib_c) {
            const double dab = D_cart[(off_a + ia) * ncart_tot + (off_b + ib_c)];
            double sum_jab = 0.0;
            for (int ic = 0; ic < ncc; ++ic) {
              const double dac = D_cart[(off_a + ia) * ncart_tot + (off_c + ic)];
              const double dbc = D_cart[(off_b + ib_c) * ncart_tot + (off_c + ic)];
              for (int id = 0; id < ncd; ++id) {
                const double v = static_cast<double>(
                    Wr * Ix[ta[ia].lx][tb[ib_c].lx][tc[ic].lx][td[id].lx] *
                    Iy[ta[ia].ly][tb[ib_c].ly][tc[ic].ly][td[id].ly] *
                    Iz[ta[ia].lz][tb[ib_c].lz][tc[ic].lz][td[id].lz]);
                if (do_j) {
                  sum_jab += v * D_cart[(off_c + ic) * ncart_tot + (off_d + id)];
                  if (!same_PQ) j_cd[ic * ncd + id] += v * dab;
                }
                if (do_k) {
                  k_ac[ia * ncc + ic] += v * D_cart[(off_b + ib_c) * ncart_tot + (off_d + id)];
                  k_bc[ib_c * ncc + ic] += v * D_cart[(off_a + ia) * ncart_tot + (off_d + id)];
                  k_ad[ia * ncd + id] += v * dbc;
                  k_bd[ib_c * ncd + id] += v * dac;
                }
              }
            }
            if (do_j) j_ab[ia * ncb + ib_c] += sum_jab;
          }
        }
      }
    }
  }

  if (do_j) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ib_c = 0; ib_c < ncb; ++ib_c)
        accum_j(off_a + ia, off_b + ib_c, w_J * j_ab[ia * ncb + ib_c]);
    if (!same_PQ) {
      for (int ic = 0; ic < ncc; ++ic)
        for (int id = 0; id < ncd; ++id)
          accum_j(off_c + ic, off_d + id, w_J * j_cd[ic * ncd + id]);
    }
  }
  if (do_k) {
    for (int ia = 0; ia < nca; ++ia)
      for (int ic = 0; ic < ncc; ++ic)
        accum_k(off_a + ia, off_c + ic, w_K * k_ac[ia * ncc + ic]);
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int ic = 0; ic < ncc; ++ic)
        accum_k(off_b + ib_c, off_c + ic, w_K * k_bc[ib_c * ncc + ic]);
    for (int ia = 0; ia < nca; ++ia)
      for (int id = 0; id < ncd; ++id)
        accum_k(off_a + ia, off_d + id, w_K * k_ad[ia * ncd + id]);
    for (int ib_c = 0; ib_c < ncb; ++ib_c)
      for (int id = 0; id < ncd; ++id)
        accum_k(off_b + ib_c, off_d + id, w_K * k_bd[ib_c * ncd + id]);
  }
}

#if defined(GPU4DFT_HAVE_CUDA)
DirectJKCudaCache* create_direct_jk_cuda_cache(
    const Basis& basis, const std::vector<ShellPairData>& pair_lists,
    const std::vector<int>& shell_cart_offset, int ncart_total);
void destroy_direct_jk_cuda_cache(DirectJKCudaCache* p);

void run_direct_jk_cuda(DirectJKCudaCache& cache, const std::vector<JKTaskTile>& tasks,
                        const std::vector<double>& d_cart_host,
                        const std::vector<double>& d_shell_max_host, double global_d_max,
                        int nshell, int ncart_total, const JKOptions& opts,
                        std::vector<double>& j_cart_out, std::vector<double>& k_cart_out);

RIJKCudaCache* create_rijk_cuda_cache(int nao, int naux, const std::vector<double>& b_tensor,
                                      const std::vector<double>& b_perm_tensor);
void destroy_rijk_cuda_cache(RIJKCudaCache* p);
void run_rijk_cuda(RIJKCudaCache& cache, const double* dm, double* vj_out, double* vk_out,
                   const JKOptions& opts);

void run_jk_nuclear_grad_cuda(const std::vector<ShellPairData>& pair_lists,
                              const std::vector<GradQuartetTask>& tasks,
                              const std::vector<int>& shell_cart_offset,
                              const std::vector<int>& shell_atom, int ncart_total, int natom,
                              const std::vector<double>& d_cart_host, double k_fraction,
                              std::vector<double>& grad_out);
#endif

}  // namespace detail

}  // namespace gpu4dft
