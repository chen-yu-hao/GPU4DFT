#pragma once

#include "gpu4dft/config.hpp"

namespace gpu4dft {

struct CartTriple {
  int lx;
  int ly;
  int lz;
};

GPU4DFT_HD constexpr int n_cart(int l) { return (l + 1) * (l + 2) / 2; }
GPU4DFT_HD constexpr int n_sph(int l) { return 2 * l + 1; }

// Canonical Cartesian ordering: lx = l..0, ly = (l-lx)..0, lz = l - lx - ly.
GPU4DFT_HD constexpr int cart_index(int l, int lx, int ly, int lz) {
  (void)ly;
  const int r = l - lx;
  return r * (r + 1) / 2 + lz;
}

GPU4DFT_HD constexpr CartTriple cart_triple(int l, int ic) {
  for (int r = 0; r <= l; ++r) {
    const int base = r * (r + 1) / 2;
    if (ic < base + r + 1) {
      const int lz = ic - base;
      return CartTriple{l - r, r - lz, lz};
    }
  }
  return CartTriple{0, 0, 0};
}

namespace harmonics {

inline constexpr double kY00 = 0.28209479177387814347;      // 1 / sqrt(4 pi)
inline constexpr double kSqrt3 = 1.73205080756887729353;    // sqrt(3)
inline constexpr double kSqrt5 = 2.23606797749978969641;    // sqrt(5)
inline constexpr double kSqrt7 = 2.64575131106459059050;    // sqrt(7)
inline constexpr double kSqrt10 = 3.16227766016837933200;   // sqrt(10)
inline constexpr double kSqrt15 = 3.87298334620741688518;   // sqrt(15)
inline constexpr double kSqrt35 = 5.91607978309961604257;   // sqrt(35)
inline constexpr double kSqrt42 = 6.48074069840786023097;   // sqrt(42)
inline constexpr double kSqrt70 = 8.36660026534075547978;   // sqrt(70)
inline constexpr double kSqrt105 = 10.24695076595959838322; // sqrt(105)

// Transforms one Cartesian axis of length n_cart(l) (with stride `in_stride`)
// into PySCF spherical ordering of length n_sph(l) (with stride `out_stride`),
// using reduced solid harmonics S_bar = sqrt(4*pi) * S (so S_bar_{0,0} = 1).
// Supports l = 0, 1, 2, 3, 4 (s, p, d, f, g).
template <typename R>
GPU4DFT_HD inline void cart_to_sph_axis(int l, const R* c, int cs, R* s, int ss) {
  switch (l) {
    case 0: {
      s[0] = c[0];
      break;
    }
    case 1: {
      // PySCF l=1 ordering: px, py, pz (NOT m = -1, 0, +1).
      const R a = R(kSqrt3);
      s[0 * ss] = a * c[0 * cs];
      s[1 * ss] = a * c[1 * cs];
      s[2 * ss] = a * c[2 * cs];
      break;
    }
    case 2: {
      // 0:xx, 1:xy, 2:xz, 3:yy, 4:yz, 5:zz -> m = -2, -1, 0, +1, +2
      const R s15 = R(kSqrt15);
      const R hs15 = R(0.5 * kSqrt15);
      const R s5 = R(kSqrt5);
      const R hs5 = R(0.5 * kSqrt5);
      s[0 * ss] = s15 * c[1 * cs];
      s[1 * ss] = s15 * c[4 * cs];
      s[2 * ss] = s5 * c[5 * cs] - hs5 * (c[0 * cs] + c[3 * cs]);
      s[3 * ss] = s15 * c[2 * cs];
      s[4 * ss] = hs15 * (c[0 * cs] - c[3 * cs]);
      break;
    }
    case 3: {
      // 0:xxx, 1:xxy, 2:xxz, 3:xyy, 4:xyz, 5:xzz, 6:yyy, 7:yyz, 8:yzz, 9:zzz
      const R q70 = R(0.25 * kSqrt70);
      const R s105 = R(kSqrt105);
      const R hs105 = R(0.5 * kSqrt105);
      const R q42 = R(0.25 * kSqrt42);
      const R s42 = R(kSqrt42);
      const R s7 = R(kSqrt7);
      const R h3s7 = R(1.5 * kSqrt7);
      s[0 * ss] = q70 * (R(3) * c[1 * cs] - c[6 * cs]);
      s[1 * ss] = s105 * c[4 * cs];
      s[2 * ss] = s42 * c[8 * cs] - q42 * (c[1 * cs] + c[6 * cs]);
      s[3 * ss] = s7 * c[9 * cs] - h3s7 * (c[2 * cs] + c[7 * cs]);
      s[4 * ss] = s42 * c[5 * cs] - q42 * (c[0 * cs] + c[3 * cs]);
      s[5 * ss] = hs105 * (c[2 * cs] - c[7 * cs]);
      s[6 * ss] = q70 * (c[0 * cs] - R(3) * c[3 * cs]);
      break;
    }
    case 4: {
      // 0:400, 1:310, 2:301, 3:220, 4:211, 5:202, 6:130, 7:121, 8:112, 9:103,
      // 10:040, 11:031, 12:022, 13:013, 14:004
      const R h3s35 = R(1.5 * kSqrt35);
      const R e3s35 = R(0.375 * kSqrt35);
      const R q3s70 = R(0.75 * kSqrt70);
      const R h3s5 = R(1.5 * kSqrt5);
      const R q3s5 = R(0.75 * kSqrt5);
      const R q3s10 = R(0.75 * kSqrt10);
      s[0 * ss] = h3s35 * (c[1 * cs] - c[6 * cs]);
      s[1 * ss] = q3s70 * (R(3) * c[4 * cs] - c[11 * cs]);
      s[2 * ss] = h3s5 * (R(6) * c[8 * cs] - c[1 * cs] - c[6 * cs]);
      s[3 * ss] = q3s10 * (R(4) * c[13 * cs] - R(3) * c[4 * cs] - R(3) * c[11 * cs]);
      s[4 * ss] = R(3) * c[14 * cs] - R(9) * (c[5 * cs] + c[12 * cs]) +
                  R(1.125) * (c[0 * cs] + R(2) * c[3 * cs] + c[10 * cs]);
      s[5 * ss] = q3s10 * (R(4) * c[9 * cs] - R(3) * c[2 * cs] - R(3) * c[7 * cs]);
      s[6 * ss] = q3s5 * (R(6) * (c[5 * cs] - c[12 * cs]) - c[0 * cs] + c[10 * cs]);
      s[7 * ss] = q3s70 * (c[2 * cs] - R(3) * c[7 * cs]);
      s[8 * ss] = e3s35 * (c[0 * cs] - R(6) * c[3 * cs] + c[10 * cs]);
      break;
    }
    default:
      break;
  }
}

// Returns reduced solid-harmonic coefficient C_bar^{(l)}_{m_idx, ic}.
GPU4DFT_HD inline double reduced_coeff(int l, int m_idx, int ic) {
  double unit_cart[15] = {0.0};
  double sph[9] = {0.0};
  unit_cart[ic] = 1.0;
  cart_to_sph_axis<double>(l, unit_cart, 1, sph, 1);
  return sph[m_idx];
}

// Returns full unit-sphere-normalized solid-harmonic coefficient C^{(l)}_{m_idx, ic}.
GPU4DFT_HD inline double full_coeff(int l, int m_idx, int ic) {
  return kY00 * reduced_coeff(l, m_idx, ic);
}

// 2-center [n_cart(la) x n_cart(lb)] -> [n_sph(la) x n_sph(lb)] transform.
template <typename R>
GPU4DFT_HD inline void cart_to_sph_2d(int la, int lb, const R* cart, R* sph) {
  const int nca = n_cart(la);
  const int ncb = n_cart(lb);
  const int nsb = n_sph(lb);
  R tmp[15 * 9];  // max l <= 4: 15 * 9 = 135
  for (int ia = 0; ia < nca; ++ia) {
    cart_to_sph_axis<R>(lb, cart + ia * ncb, 1, tmp + ia * nsb, 1);
  }
  for (int mb = 0; mb < nsb; ++mb) {
    cart_to_sph_axis<R>(la, tmp + mb, nsb, sph + mb, nsb);
  }
}

// 4-center [nca x ncb x ncc x ncd] -> [nsa x nsb x nsc x nsd] transform.
// Uses sequential single-index contractions to keep arithmetic O(N^5) instead
// of O(N^8). Max l <= 4.
template <typename R>
GPU4DFT_HD inline void cart_to_sph_4d(int la, int lb, int lc, int ld, const R* cart, R* sph) {
  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);
  const int nsa = n_sph(la), nsb = n_sph(lb), nsc = n_sph(lc), nsd = n_sph(ld);

  // Transform d then c for each (ia, ib), then transform b and a.
  // Workspace sizes for l <= 4:
  //   after c,d -> [nca * ncb * nsc * nsd] <= 15 * 15 * 9 * 9 = 18225 (on host);
  //   to avoid large GPU stacks, transform (c,d) per (ia,ib) into a small
  //   [nsc * nsd <= 81] buffer and accumulate into sph via (la, lb) coefficients!
  const int nscd = nsc * nsd;
  const int nsbsd = nsb * nscd;
  for (int i = 0; i < nsa * nsbsd; ++i) sph[i] = R(0);

  R cd_tmp[15 * 9] = {R(0)};
  R cd_sph[9 * 9] = {R(0)};
  R ca_col[9] = {R(0)};
  R cb_col[9] = {R(0)};
  R unit[15] = {R(0)};

  for (int ia = 0; ia < nca; ++ia) {
    unit[ia] = R(1);
    cart_to_sph_axis<R>(la, unit, 1, ca_col, 1);
    unit[ia] = R(0);

    for (int ib = 0; ib < ncb; ++ib) {
      unit[ib] = R(1);
      cart_to_sph_axis<R>(lb, unit, 1, cb_col, 1);
      unit[ib] = R(0);

      const R* blk = cart + (ia * ncb + ib) * (ncc * ncd);
      for (int ic = 0; ic < ncc; ++ic) {
        cart_to_sph_axis<R>(ld, blk + ic * ncd, 1, cd_tmp + ic * nsd, 1);
      }
      for (int md = 0; md < nsd; ++md) {
        cart_to_sph_axis<R>(lc, cd_tmp + md, nsd, cd_sph + md, nsd);
      }

      for (int ma = 0; ma < nsa; ++ma) {
        const R wa = ca_col[ma];
        if (wa == R(0)) continue;
        for (int mb = 0; mb < nsb; ++mb) {
          const R wab = wa * cb_col[mb];
          if (wab == R(0)) continue;
          R* dst = sph + ma * nsbsd + mb * nscd;
          for (int mcd = 0; mcd < nscd; ++mcd) {
            dst[mcd] += wab * cd_sph[mcd];
          }
        }
      }
    }
  }
}

}  // namespace harmonics
}  // namespace gpu4dft
