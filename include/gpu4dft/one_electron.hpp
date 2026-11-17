#pragma once

#include <array>
#include <span>
#include <vector>

#include "gpu4dft/basis.hpp"
#include "gpu4dft/boys.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/math.hpp"
#include "gpu4dft/types.hpp"

namespace gpu4dft {

struct Atom {
  int atomic_number = 0;
  std::array<double, 3> center{};  // Bohr
};

// Overlap matrix S_{uv} = <u | v>, row-major [nao x nao].
std::vector<double> overlap_matrix(const Basis& basis, Backend backend = Backend::Host);

// Kinetic energy matrix T_{uv} = <u | -1/2 nabla^2 | v>, row-major [nao x nao].
std::vector<double> kinetic_matrix(const Basis& basis, Backend backend = Backend::Host);

// Nuclear attraction matrix V_{uv} = \sum_C <u | -Z_C / |r - R_C| | v>, row-major [nao x nao].
std::vector<double> nuclear_attraction_matrix(const Basis& basis, std::span<const Atom> atoms,
                                              Backend backend = Backend::Host);

// Classical nuclear-nuclear repulsion energy \sum_{A < B} Z_A Z_B / |R_A - R_B|.
double nuclear_repulsion_energy(std::span<const Atom> atoms);

// Analytical nuclear gradient of nuclear_repulsion_energy, row-major [natom * 3].
std::vector<double> nuclear_repulsion_grad(std::span<const Atom> atoms);

// Analytical 1-electron electronic nuclear gradient [natom * 3]:
//   d/dR_A Tr(D * (T + V_nuc)) - Tr(W * dS/dR_A),
// including both AO-center derivatives of (T, V_nuc, S) and the Hellmann-Feynman
// nuclear operator derivative of V_nuc.
std::vector<double> one_electron_grad(const Basis& basis, std::span<const Atom> atoms,
                                      const double* dm, const double* energy_weighted_dm,
                                      Backend backend = Backend::Host);

namespace detail {

// Trivially copyable SoA view of a Basis + Atom list usable on Host and CUDA.
struct BasisDeviceView {
  int nshell = 0;
  int nao = 0;
  int natom = 0;
  const int* shell_l = nullptr;          // [nshell]
  const int* shell_ao = nullptr;         // [nshell]
  const int* shell_atom = nullptr;       // [nshell], -1 if unattached
  const int* prim_offset = nullptr;      // [nshell + 1]
  const double* shell_center = nullptr;  // [nshell * 3]
  const double* prim_exp = nullptr;      // [nprim]
  const double* prim_coeff = nullptr;    // [nprim]
  const int* atom_Z = nullptr;           // [natom]
  const double* atom_center = nullptr;   // [natom * 3]
};

inline constexpr double kOneElecPi = 3.14159265358979323846;
inline constexpr double kOneElecTwoOverSqrtPi = 1.12837916709551257390;

GPU4DFT_HD inline void one_elec_overlap_1d(int max_a, int max_b, double inv_2p, double PA,
                                           double AB, double S[7][8]) {
  const int total = max_a + max_b;
  double col0[15];
  col0[0] = 1.0;
  for (int i = 0; i < total; ++i) {
    col0[i + 1] = PA * col0[i] + (i > 0 ? double(i) * inv_2p * col0[i - 1] : 0.0);
  }
  double cur[15];
  for (int i = 0; i <= total; ++i) cur[i] = col0[i];
  for (int j = 0; j <= max_b; ++j) {
    for (int i = 0; i <= max_a; ++i) S[i][j] = cur[i];
    if (j < max_b) {
      for (int i = 0; i <= total - j - 1; ++i) cur[i] = cur[i + 1] + AB * cur[i];
    }
  }
}

GPU4DFT_HD inline void one_elec_hermite_1d(int max_a, int max_b, double inv_2p, double PA,
                                           double PB, double E[6][5][11]) {
  const int max_t = max_a + max_b;
  for (int i = 0; i <= max_a; ++i)
    for (int j = 0; j <= max_b; ++j)
      for (int t = 0; t <= max_t; ++t) E[i][j][t] = 0.0;
  E[0][0][0] = 1.0;
  for (int i = 0; i < max_a; ++i) {
    for (int t = 0; t <= i + 1; ++t) {
      double val = PA * E[i][0][t];
      if (t > 0) val += inv_2p * E[i][0][t - 1];
      if (t + 1 <= i) val += double(t + 1) * E[i][0][t + 1];
      E[i + 1][0][t] = val;
    }
  }
  for (int i = 0; i <= max_a; ++i) {
    for (int j = 0; j < max_b; ++j) {
      const int deg = i + j;
      for (int t = 0; t <= deg + 1; ++t) {
        double val = PB * E[i][j][t];
        if (t > 0) val += inv_2p * E[i][j][t - 1];
        if (t + 1 <= deg) val += double(t + 1) * E[i][j][t + 1];
        E[i][j + 1][t] = val;
      }
    }
  }
}

GPU4DFT_HD inline void one_elec_coulomb_r(int L, double p, double scale, double PCx, double PCy,
                                          double PCz, const double* F,
                                          double R_out[11][11][11]) {
  double R_cur[11][11][11];
  double R_nxt[11][11][11];
  for (int t = 0; t <= L; ++t)
    for (int u = 0; u <= L - t; ++u)
      for (int v = 0; v <= L - t - u; ++v) R_cur[t][u][v] = 0.0;

  double pow_m2p[12];
  pow_m2p[0] = scale;
  const double m2p = -2.0 * p;
  for (int m = 1; m <= L; ++m) pow_m2p[m] = pow_m2p[m - 1] * m2p;

  R_cur[0][0][0] = pow_m2p[L] * F[L];
  for (int m = L - 1; m >= 0; --m) {
    const int max_deg = L - m;
    R_nxt[0][0][0] = pow_m2p[m] * F[m];
    for (int deg = 1; deg <= max_deg; ++deg) {
      for (int t = 0; t <= deg; ++t) {
        for (int u = 0; u <= deg - t; ++u) {
          const int v = deg - t - u;
          if (t > 0) {
            R_nxt[t][u][v] =
                PCx * R_cur[t - 1][u][v] + (t > 1 ? double(t - 1) * R_cur[t - 2][u][v] : 0.0);
          } else if (u > 0) {
            R_nxt[0][u][v] =
                PCy * R_cur[0][u - 1][v] + (u > 1 ? double(u - 1) * R_cur[0][u - 2][v] : 0.0);
          } else {
            R_nxt[0][0][v] =
                PCz * R_cur[0][0][v - 1] + (v > 1 ? double(v - 1) * R_cur[0][0][v - 2] : 0.0);
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
  for (int t = 0; t <= L; ++t)
    for (int u = 0; u <= L - t; ++u)
      for (int v = 0; v <= L - t - u; ++v) R_out[t][u][v] = R_cur[t][u][v];
}

GPU4DFT_HD inline void eval_one_elec_pair(const BasisDeviceView& b, int ish, int jsh, int op,
                                          double* mat) {
  const int la = b.shell_l[ish];
  const int lb = b.shell_l[jsh];
  const int nca = n_cart(la);
  const int ncb = n_cart(lb);
  const int nsa = n_sph(la);
  const int nsb = n_sph(lb);

  const double Ax = b.shell_center[3 * ish + 0];
  const double Ay = b.shell_center[3 * ish + 1];
  const double Az = b.shell_center[3 * ish + 2];
  const double Bx = b.shell_center[3 * jsh + 0];
  const double By = b.shell_center[3 * jsh + 1];
  const double Bz = b.shell_center[3 * jsh + 2];
  const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
  const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;

  CartTriple ta[15], tb[15];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib = 0; ib < ncb; ++ib) tb[ib] = cart_triple(lb, ib);

  double cart[15 * 15] = {0.0};
  const int pa_beg = b.prim_offset[ish], pa_end = b.prim_offset[ish + 1];
  const int pb_beg = b.prim_offset[jsh], pb_end = b.prim_offset[jsh + 1];

  for (int ka = pa_beg; ka < pa_end; ++ka) {
    const double a = b.prim_exp[ka];
    const double ca = b.prim_coeff[ka];
    for (int kb = pb_beg; kb < pb_end; ++kb) {
      const double beta = b.prim_exp[kb];
      const double cb = b.prim_coeff[kb];
      const double p = a + beta;
      const double inv_p = 1.0 / p;
      const double inv_2p = 0.5 * inv_p;
      const double Px = (a * Ax + beta * Bx) * inv_p;
      const double Py = (a * Ay + beta * By) * inv_p;
      const double Pz = (a * Az + beta * Bz) * inv_p;
      const double ratio = kOneElecPi * inv_p;
      const double pref = ca * cb * harmonics::kY00 * harmonics::kY00 *
                          math::exp(-a * beta * inv_p * AB2) * ratio * math::sqrt(ratio);

      if (op == 0 || op == 1) {
        const int extra_b = (op == 1) ? 2 : 0;
        double Sx[7][8], Sy[7][8], Sz[7][8];
        one_elec_overlap_1d(la, lb + extra_b, inv_2p, Px - Ax, ABx, Sx);
        one_elec_overlap_1d(la, lb + extra_b, inv_2p, Py - Ay, ABy, Sy);
        one_elec_overlap_1d(la, lb + extra_b, inv_2p, Pz - Az, ABz, Sz);

        if (op == 0) {
          for (int ia = 0; ia < nca; ++ia) {
            for (int ib = 0; ib < ncb; ++ib) {
              cart[ia * ncb + ib] += pref * Sx[ta[ia].lx][tb[ib].lx] *
                                     Sy[ta[ia].ly][tb[ib].ly] * Sz[ta[ia].lz][tb[ib].lz];
            }
          }
        } else {
          auto kin_1d = [&](const double S[7][8], int i, int j) {
            double v = beta * double(2 * j + 1) * S[i][j] - 2.0 * beta * beta * S[i][j + 2];
            if (j >= 2) v -= 0.5 * double(j * (j - 1)) * S[i][j - 2];
            return v;
          };
          for (int ia = 0; ia < nca; ++ia) {
            const int ix = ta[ia].lx, iy = ta[ia].ly, iz = ta[ia].lz;
            for (int ib = 0; ib < ncb; ++ib) {
              const int jx = tb[ib].lx, jy = tb[ib].ly, jz = tb[ib].lz;
              const double sx = Sx[ix][jx], sy = Sy[iy][jy], sz = Sz[iz][jz];
              const double tx = kin_1d(Sx, ix, jx);
              const double ty = kin_1d(Sy, iy, jy);
              const double tz = kin_1d(Sz, iz, jz);
              cart[ia * ncb + ib] += pref * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
            }
          }
        }
      } else {
        const int L = la + lb;
        double Ex[6][5][11], Ey[6][5][11], Ez[6][5][11];
        one_elec_hermite_1d(la, lb, inv_2p, Px - Ax, Px - Bx, Ex);
        one_elec_hermite_1d(la, lb, inv_2p, Py - Ay, Py - By, Ey);
        one_elec_hermite_1d(la, lb, inv_2p, Pz - Az, Pz - Bz, Ez);

        double V_herm[11][11][11];
        for (int t = 0; t <= L; ++t)
          for (int u = 0; u <= L - t; ++u)
            for (int v = 0; v <= L - t - u; ++v) V_herm[t][u][v] = 0.0;

        const double base_scale = pref * kOneElecTwoOverSqrtPi * math::sqrt(p);
        for (int c = 0; c < b.natom; ++c) {
          const int Z = b.atom_Z[c];
          if (Z == 0) continue;
          const double PCx = Px - b.atom_center[3 * c + 0];
          const double PCy = Py - b.atom_center[3 * c + 1];
          const double PCz = Pz - b.atom_center[3 * c + 2];
          const double T = p * (PCx * PCx + PCy * PCy + PCz * PCz);
          double F[12];
          boys_vec<double>(L, T, F);
          double Rc[11][11][11];
          one_elec_coulomb_r(L, p, -double(Z) * base_scale, PCx, PCy, PCz, F, Rc);
          for (int t = 0; t <= L; ++t)
            for (int u = 0; u <= L - t; ++u)
              for (int v = 0; v <= L - t - u; ++v) V_herm[t][u][v] += Rc[t][u][v];
        }

        for (int ia = 0; ia < nca; ++ia) {
          const int ix = ta[ia].lx, iy = ta[ia].ly, iz = ta[ia].lz;
          for (int ib = 0; ib < ncb; ++ib) {
            const int jx = tb[ib].lx, jy = tb[ib].ly, jz = tb[ib].lz;
            double sum = 0.0;
            for (int t = 0; t <= ix + jx; ++t) {
              const double ex = Ex[ix][jx][t];
              for (int u = 0; u <= iy + jy; ++u) {
                const double exy = ex * Ey[iy][jy][u];
                for (int v = 0; v <= iz + jz; ++v) {
                  sum += exy * Ez[iz][jz][v] * V_herm[t][u][v];
                }
              }
            }
            cart[ia * ncb + ib] += sum;
          }
        }
      }
    }
  }

  double sph[9 * 9];
  harmonics::cart_to_sph_2d<double>(la, lb, cart, sph);

  const int i0 = b.shell_ao[ish];
  const int j0 = b.shell_ao[jsh];
  const int nao = b.nao;
  for (int ma = 0; ma < nsa; ++ma) {
    for (int mb = 0; mb < nsb; ++mb) {
      const double v = sph[ma * nsb + mb];
      mat[(i0 + ma) * nao + (j0 + mb)] = v;
      mat[(j0 + mb) * nao + (i0 + ma)] = v;
    }
  }
}

template <typename AddFn>
GPU4DFT_HD inline void eval_one_elec_grad_ordered_pair(
    const BasisDeviceView& b, int ish, int jsh, const double* dm, const double* wdm,
    double bra_force[3], AddFn add_hf) {
  bra_force[0] = bra_force[1] = bra_force[2] = 0.0;

  const int la = b.shell_l[ish];
  const int lb = b.shell_l[jsh];
  const int nca = n_cart(la);
  const int ncb = n_cart(lb);
  const int nsa = n_sph(la);
  const int nsb = n_sph(lb);
  const int i0 = b.shell_ao[ish];
  const int j0 = b.shell_ao[jsh];
  const int nao = b.nao;

  double ca_mat[9][15], cb_mat[9][15];
  double unit[15] = {0.0};
  for (int ic = 0; ic < nca; ++ic) {
    unit[ic] = 1.0;
    double col[9] = {0.0};
    harmonics::cart_to_sph_axis<double>(la, unit, 1, col, 1);
    for (int ma = 0; ma < nsa; ++ma) ca_mat[ma][ic] = col[ma];
    unit[ic] = 0.0;
  }
  for (int ic = 0; ic < ncb; ++ic) {
    unit[ic] = 1.0;
    double col[9] = {0.0};
    harmonics::cart_to_sph_axis<double>(lb, unit, 1, col, 1);
    for (int mb = 0; mb < nsb; ++mb) cb_mat[mb][ic] = col[mb];
    unit[ic] = 0.0;
  }

  double D_cart[15 * 15] = {0.0};
  double W_cart[15 * 15] = {0.0};
  for (int ma = 0; ma < nsa; ++ma) {
    for (int mb = 0; mb < nsb; ++mb) {
      const int u = i0 + ma;
      const int v = j0 + mb;
      const double d_uv = 0.5 * (dm[u * nao + v] + dm[v * nao + u]);
      const double w_uv = 0.5 * (wdm[u * nao + v] + wdm[v * nao + u]);
      for (int ia = 0; ia < nca; ++ia) {
        const double ca = ca_mat[ma][ia];
        if (ca == 0.0) continue;
        for (int ib = 0; ib < ncb; ++ib) {
          const double cab = ca * cb_mat[mb][ib];
          D_cart[ia * ncb + ib] += cab * d_uv;
          W_cart[ia * ncb + ib] += cab * w_uv;
        }
      }
    }
  }

  const double Ax = b.shell_center[3 * ish + 0];
  const double Ay = b.shell_center[3 * ish + 1];
  const double Az = b.shell_center[3 * ish + 2];
  const double Bx = b.shell_center[3 * jsh + 0];
  const double By = b.shell_center[3 * jsh + 1];
  const double Bz = b.shell_center[3 * jsh + 2];
  const double ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
  const double AB2 = ABx * ABx + ABy * ABy + ABz * ABz;

  CartTriple ta[15], tb[15];
  for (int ia = 0; ia < nca; ++ia) ta[ia] = cart_triple(la, ia);
  for (int ib = 0; ib < ncb; ++ib) tb[ib] = cart_triple(lb, ib);

  const int pa_beg = b.prim_offset[ish], pa_end = b.prim_offset[ish + 1];
  const int pb_beg = b.prim_offset[jsh], pb_end = b.prim_offset[jsh + 1];

  double fAx = 0.0, fAy = 0.0, fAz = 0.0;

  for (int ka = pa_beg; ka < pa_end; ++ka) {
    const double a = b.prim_exp[ka];
    const double ca = b.prim_coeff[ka];
    const double two_a = 2.0 * a;
    for (int kb = pb_beg; kb < pb_end; ++kb) {
      const double beta = b.prim_exp[kb];
      const double cb = b.prim_coeff[kb];
      const double p = a + beta;
      const double inv_p = 1.0 / p;
      const double inv_2p = 0.5 * inv_p;
      const double Px = (a * Ax + beta * Bx) * inv_p;
      const double Py = (a * Ay + beta * By) * inv_p;
      const double Pz = (a * Az + beta * Bz) * inv_p;
      const double ratio = kOneElecPi * inv_p;
      const double pref = ca * cb * harmonics::kY00 * harmonics::kY00 *
                          math::exp(-a * beta * inv_p * AB2) * ratio * math::sqrt(ratio);

      double Sx[7][8], Sy[7][8], Sz[7][8];
      one_elec_overlap_1d(la + 1, lb + 2, inv_2p, Px - Ax, ABx, Sx);
      one_elec_overlap_1d(la + 1, lb + 2, inv_2p, Py - Ay, ABy, Sy);
      one_elec_overlap_1d(la + 1, lb + 2, inv_2p, Pz - Az, ABz, Sz);

      auto kin_1d = [&](const double S[7][8], int i, int j) {
        double v = beta * double(2 * j + 1) * S[i][j] - 2.0 * beta * beta * S[i][j + 2];
        if (j >= 2) v -= 0.5 * double(j * (j - 1)) * S[i][j - 2];
        return v;
      };

      for (int ia = 0; ia < nca; ++ia) {
        const int ix = ta[ia].lx, iy = ta[ia].ly, iz = ta[ia].lz;
        for (int ib = 0; ib < ncb; ++ib) {
          const int jx = tb[ib].lx, jy = tb[ib].ly, jz = tb[ib].lz;
          const double d_val = D_cart[ia * ncb + ib];
          const double w_val = W_cart[ia * ncb + ib];

          const double sx = Sx[ix][jx], sy = Sy[iy][jy], sz = Sz[iz][jz];
          const double dsx = two_a * Sx[ix + 1][jx] - (ix > 0 ? double(ix) * Sx[ix - 1][jx] : 0.0);
          const double dsy = two_a * Sy[iy + 1][jy] - (iy > 0 ? double(iy) * Sy[iy - 1][jy] : 0.0);
          const double dsz = two_a * Sz[iz + 1][jz] - (iz > 0 ? double(iz) * Sz[iz - 1][jz] : 0.0);

          fAx -= 2.0 * w_val * pref * (dsx * sy * sz);
          fAy -= 2.0 * w_val * pref * (sx * dsy * sz);
          fAz -= 2.0 * w_val * pref * (sx * sy * dsz);

          const double tx = kin_1d(Sx, ix, jx);
          const double ty = kin_1d(Sy, iy, jy);
          const double tz = kin_1d(Sz, iz, jz);
          const double dtx =
              two_a * kin_1d(Sx, ix + 1, jx) - (ix > 0 ? double(ix) * kin_1d(Sx, ix - 1, jx) : 0.0);
          const double dty =
              two_a * kin_1d(Sy, iy + 1, jy) - (iy > 0 ? double(iy) * kin_1d(Sy, iy - 1, jy) : 0.0);
          const double dtz =
              two_a * kin_1d(Sz, iz + 1, jz) - (iz > 0 ? double(iz) * kin_1d(Sz, iz - 1, jz) : 0.0);

          fAx += 2.0 * d_val * pref * (dtx * sy * sz + dsx * (ty * sz + sy * tz));
          fAy += 2.0 * d_val * pref * (tx * dsy * sz + sx * dty * sz + sx * dsy * tz);
          fAz += 2.0 * d_val * pref * ((tx * sy + sx * ty) * dsz + sx * sy * dtz);
        }
      }

      const int L = la + lb;
      const int L_hi = L + 1;
      double Ex[6][5][11], Ey[6][5][11], Ez[6][5][11];
      one_elec_hermite_1d(la + 1, lb, inv_2p, Px - Ax, Px - Bx, Ex);
      one_elec_hermite_1d(la + 1, lb, inv_2p, Py - Ay, Py - By, Ey);
      one_elec_hermite_1d(la + 1, lb, inv_2p, Pz - Az, Pz - Bz, Ez);

      double D_herm[10][10][10];
      double D_herm_dx[11][11][11], D_herm_dy[11][11][11], D_herm_dz[11][11][11];
      for (int t = 0; t <= L_hi; ++t)
        for (int u = 0; u <= L_hi - t; ++u)
          for (int v = 0; v <= L_hi - t - u; ++v) {
            if (t + u + v <= L) D_herm[t][u][v] = 0.0;
            D_herm_dx[t][u][v] = 0.0;
            D_herm_dy[t][u][v] = 0.0;
            D_herm_dz[t][u][v] = 0.0;
          }

      for (int ia = 0; ia < nca; ++ia) {
        const int ix = ta[ia].lx, iy = ta[ia].ly, iz = ta[ia].lz;
        for (int ib = 0; ib < ncb; ++ib) {
          const double d_val = D_cart[ia * ncb + ib];
          if (d_val == 0.0) continue;
          const int jx = tb[ib].lx, jy = tb[ib].ly, jz = tb[ib].lz;

          for (int t = 0; t <= ix + jx + 1; ++t) {
            const double ex0 = (t <= ix + jx) ? Ex[ix][jx][t] : 0.0;
            const double dex = two_a * Ex[ix + 1][jx][t] -
                               ((ix > 0 && t <= ix - 1 + jx) ? double(ix) * Ex[ix - 1][jx][t] : 0.0);
            for (int u = 0; u <= iy + jy + 1; ++u) {
              if (t + u > L_hi) break;
              const double ey0 = (u <= iy + jy) ? Ey[iy][jy][u] : 0.0;
              const double dey =
                  two_a * Ey[iy + 1][jy][u] -
                  ((iy > 0 && u <= iy - 1 + jy) ? double(iy) * Ey[iy - 1][jy][u] : 0.0);
              for (int v = 0; v <= iz + jz + 1; ++v) {
                if (t + u + v > L_hi) break;
                const double ez0 = (v <= iz + jz) ? Ez[iz][jz][v] : 0.0;
                const double dez =
                    two_a * Ez[iz + 1][jz][v] -
                    ((iz > 0 && v <= iz - 1 + jz) ? double(iz) * Ez[iz - 1][jz][v] : 0.0);

                if (t <= ix + jx && u <= iy + jy && v <= iz + jz) {
                  D_herm[t][u][v] += d_val * ex0 * ey0 * ez0;
                }
                if (u <= iy + jy && v <= iz + jz) {
                  D_herm_dx[t][u][v] += d_val * dex * ey0 * ez0;
                }
                if (t <= ix + jx && v <= iz + jz) {
                  D_herm_dy[t][u][v] += d_val * ex0 * dey * ez0;
                }
                if (t <= ix + jx && u <= iy + jy) {
                  D_herm_dz[t][u][v] += d_val * ex0 * ey0 * dez;
                }
              }
            }
          }
        }
      }

      const double base_scale = pref * kOneElecTwoOverSqrtPi * math::sqrt(p);
      for (int c = 0; c < b.natom; ++c) {
        const int Z = b.atom_Z[c];
        if (Z == 0) continue;
        const double PCx = Px - b.atom_center[3 * c + 0];
        const double PCy = Py - b.atom_center[3 * c + 1];
        const double PCz = Pz - b.atom_center[3 * c + 2];
        const double T = p * (PCx * PCx + PCy * PCy + PCz * PCz);
        double F[12];
        boys_vec<double>(L_hi, T, F);
        double R_pos[11][11][11];
        one_elec_coulomb_r(L_hi, p, double(Z) * base_scale, PCx, PCy, PCz, F, R_pos);

        double hf_x = 0.0, hf_y = 0.0, hf_z = 0.0;
        for (int t = 0; t <= L_hi; ++t) {
          for (int u = 0; u <= L_hi - t; ++u) {
            for (int v = 0; v <= L_hi - t - u; ++v) {
              const double r_val = R_pos[t][u][v];
              fAx -= 2.0 * D_herm_dx[t][u][v] * r_val;
              fAy -= 2.0 * D_herm_dy[t][u][v] * r_val;
              fAz -= 2.0 * D_herm_dz[t][u][v] * r_val;

              if (t + u + v <= L) {
                const double dh = D_herm[t][u][v];
                hf_x += dh * R_pos[t + 1][u][v];
                hf_y += dh * R_pos[t][u + 1][v];
                hf_z += dh * R_pos[t][u][v + 1];
              }
            }
          }
        }
        add_hf(c, hf_x, hf_y, hf_z);
      }
    }
  }

  bra_force[0] = fAx;
  bra_force[1] = fAy;
  bra_force[2] = fAz;
}

#if defined(GPU4DFT_HAVE_CUDA)
void run_one_electron_matrix_cuda(const BasisDeviceView& host_view, int op,
                                  std::vector<double>& out);
void run_one_electron_grad_cuda(const BasisDeviceView& host_view, const double* dm,
                                const double* wdm, std::vector<double>& grad_out);
#endif

}  // namespace detail

}  // namespace gpu4dft
