"""Integral math for each ERI class, emitted as backend-neutral C++.

Every generated class header ``include/gpu4dft/generated/eri_<name>.hpp`` defines:

* ``kEri_<name>_NCart`` and ``kEri_<name>_NSph``;
* ``eri_<name>_prim<R>(bra, ib, ip, ket, ik, kp, double* cart_acc)`` evaluating
  one primitive quartet in precision ``R`` (via McMurchie-Davidson for low
  angular momentum ``L <= 3`` or Rys quadrature + 2D VRR + 1D HRR for ``L >= 4``,
  configurable via ``CudaTuning.algorithm``) and accumulating into ``cart_acc``;
* ``eri_<name>_cart_to_sph(const double* cart, double* sph_out)``;
* ``eri_<name>_eval<R>(bra, ib, ket, ik, double* sph_out)`` evaluating the full
  contracted spherical shell quartet (plus ``eri_ssss_contracted<R>`` returning
  ``double`` for ``(ss|ss)``).

In addition, ``emit_general_header()`` emits ``include/gpu4dft/generated/eri_general.hpp``
handling arbitrary angular momentum ``0 <= la, lb, lc, ld <= 4`` (``s, p, d, f, g``)
on Host, CUDA, and HIP with bounded stack usage.
"""

from __future__ import annotations

import math

from .config import MAX_SUPPORTED_L, CudaTuning, EriClass
from .harmonics import cart_tuples, n_cart, n_sph, reduced_sph_sparse

TWO_OVER_SQRT_PI = repr(2.0 / math.sqrt(math.pi))


def contracted_function_name(cls: EriClass) -> str:
    return f"eri_{cls.name}_contracted"


def eval_function_name(cls: EriClass) -> str:
    return f"eri_{cls.name}_eval"


def prim_function_name(cls: EriClass) -> str:
    return f"eri_{cls.name}_prim"


def cart2sph_function_name(cls: EriClass) -> str:
    return f"eri_{cls.name}_cart_to_sph"


def header_relpath(cls: EriClass) -> str:
    return f"include/gpu4dft/generated/eri_{cls.name}.hpp"


def header_include(cls: EriClass) -> str:
    return f"gpu4dft/generated/eri_{cls.name}.hpp"


GENERAL_HEADER_RELPATH = "include/gpu4dft/generated/eri_general.hpp"
GENERAL_HEADER_INCLUDE = "gpu4dft/generated/eri_general.hpp"


def _unroll_pragma(prim_unroll: int) -> str:
    if prim_unroll <= 1:
        return ""
    return (
        "#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)\n"
        f"#pragma unroll {prim_unroll}\n"
        "#endif\n    "
    )


def _emit_cart_index_table(name: str, l: int) -> str:
    tuples = cart_tuples(l)
    xs = ", ".join(str(t[0]) for t in tuples)
    ys = ", ".join(str(t[1]) for t in tuples)
    zs = ", ".join(str(t[2]) for t in tuples)
    return (
        f"  constexpr int {name}x[{len(tuples)}] = {{{xs}}};\n"
        f"  constexpr int {name}y[{len(tuples)}] = {{{ys}}};\n"
        f"  constexpr int {name}z[{len(tuples)}] = {{{zs}}};"
    )


def _emit_cart_to_sph_fn(cls: EriClass) -> str:
    la, lb, lc, ld = cls.angular
    fn = cart2sph_function_name(cls)
    if cls.angular == (0, 0, 0, 0):
        return f"""\
GPU4DFT_HD inline void {fn}(const double* cart, double* out) {{
  out[0] = cart[0];
}}
"""
    if cls.total_l <= 2 and cls.n_cart_components <= 36:
        # Emit fully unrolled sparse linear combinations for ultra-low L classes!
        sa = reduced_sph_sparse(la)
        sb = reduced_sph_sparse(lb)
        sc = reduced_sph_sparse(lc)
        sd = reduced_sph_sparse(ld)
        ncb, ncc, ncd = n_cart(lb), n_cart(lc), n_cart(ld)
        nsb, nsc, nsd = n_sph(lb), n_sph(lc), n_sph(ld)
        lines: list[str] = []
        for ma, ta in enumerate(sa):
            for mb, tb in enumerate(sb):
                for mc, tc in enumerate(sc):
                    for md_i, td in enumerate(sd):
                        out_idx = ((ma * nsb + mb) * nsc + mc) * nsd + md_i
                        terms: list[str] = []
                        for ia, ca in ta:
                            for ib_c, cb in tb:
                                for ic, cc in tc:
                                    for id_c, cd in td:
                                        w = ca * cb * cc * cd
                                        c_idx = ((ia * ncb + ib_c) * ncc + ic) * ncd + id_c
                                        if abs(w - 1.0) < 1e-16:
                                            terms.append(f"cart[{c_idx}]")
                                        elif abs(w + 1.0) < 1e-16:
                                            terms.append(f"(-cart[{c_idx}])")
                                        else:
                                            terms.append(f"({repr(w)} * cart[{c_idx}])")
                        expr = " + ".join(terms) if terms else "0.0"
                        lines.append(f"  out[{out_idx}] = {expr};")
        body = "\n".join(lines)
        return f"""\
GPU4DFT_HD inline void {fn}(const double* cart, double* out) {{
{body}
}}
"""
    return f"""\
GPU4DFT_HD inline void {fn}(const double* cart, double* out) {{
  harmonics::cart_to_sph_4d<double>({la}, {lb}, {lc}, {ld}, cart, out);
}}
"""


def _emit_ssss_body(prim_unroll: int) -> str:
    unroll = _unroll_pragma(prim_unroll)
    return f"""\
inline constexpr int kEri_ssss_NCart = 1;
inline constexpr int kEri_ssss_NSph = 1;

template <typename R>
GPU4DFT_HD inline void eri_ssss_prim(const ShellPairView& bra, int /*ib*/, int ip,
                                     const ShellPairView& ket, int /*ik*/, int kp,
                                     double* cart_acc, double omega = 0.0) {{
  const R p = static_cast<R>(bra.exponent[ip]);
  const R q = static_cast<R>(ket.exponent[kp]);
  const R PQx = static_cast<R>(bra.Px[ip]) - static_cast<R>(ket.Px[kp]);
  const R PQy = static_cast<R>(bra.Py[ip]) - static_cast<R>(ket.Py[kp]);
  const R PQz = static_cast<R>(bra.Pz[ip]) - static_cast<R>(ket.Pz[kp]);
  const R Cab = static_cast<R>(bra.prefactor[ip]);
  const R Ccd = static_cast<R>(ket.prefactor[kp]);
  const R rho = p * q / (p + q);
  const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
  const R value = Cab * Ccd * R({TWO_OVER_SQRT_PI}) * math::sqrt(rho) *
                  boys_f0_erf(T, rho, static_cast<R>(omega));
  cart_acc[0] += static_cast<double>(value);
}}

GPU4DFT_HD inline void eri_ssss_cart_to_sph(const double* cart, double* out) {{
  out[0] = cart[0];
}}

template <typename R>
GPU4DFT_HD inline double eri_ssss_contracted(const ShellPairView& bra, int ib,
                                             const ShellPairView& ket, int ik,
                                             double omega = 0.0) {{
  double acc = 0.0;
  const int bra_begin = bra.prim_offset[ib];
  const int bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik];
  const int ket_end = ket.prim_offset[ik + 1];
  for (int ip = bra_begin; ip < bra_end; ++ip) {{
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    {unroll}for (int kp = ket_begin; kp < ket_end; ++kp) {{
      const R q = static_cast<R>(ket.exponent[kp]);
      const R PQx = Px - static_cast<R>(ket.Px[kp]);
      const R PQy = Py - static_cast<R>(ket.Py[kp]);
      const R PQz = Pz - static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R rho = p * q / (p + q);
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      const R value = Cab * Ccd * R({TWO_OVER_SQRT_PI}) * math::sqrt(rho) *
                      boys_f0_erf(T, rho, static_cast<R>(omega));
      acc += static_cast<double>(value);
    }}
  }}
  return acc;
}}

template <typename R>
GPU4DFT_HD inline void eri_ssss_eval(const ShellPairView& bra, int ib,
                                     const ShellPairView& ket, int ik, double* out,
                                     double omega = 0.0) {{
  out[0] = eri_ssss_contracted<R>(bra, ib, ket, ik, omega);
}}
"""


def _emit_md_body(cls: EriClass, prim_unroll: int) -> str:
    if cls.angular == (0, 0, 0, 0):
        return _emit_ssss_body(prim_unroll)

    la, lb, lc, ld = cls.angular
    lab = la + lb
    lcd = lc + ld
    L = lab + lcd
    nca, ncb, ncc, ncd = n_cart(la), n_cart(lb), n_cart(lc), n_cart(ld)
    nc_tot = cls.n_cart_components
    ns_tot = cls.n_sph_components
    unroll = _unroll_pragma(prim_unroll)

    tabs = "\n".join(
        [
            _emit_cart_index_table("a", la),
            _emit_cart_index_table("b", lb),
            _emit_cart_index_table("c", lc),
            _emit_cart_index_table("d", ld),
        ]
    )
    c2s_fn = _emit_cart_to_sph_fn(cls)

    return f"""\
inline constexpr int kEri_{cls.name}_NCart = {nc_tot};
inline constexpr int kEri_{cls.name}_NSph = {ns_tot};

namespace detail_{cls.name} {{

template <typename R, int MaxA, int MaxB>
GPU4DFT_HD inline void md_hermite_1d(R inv_2p, R PA, R PB,
                                     R E[MaxA + 1][MaxB + 1][MaxA + MaxB + 1],
                                     bool negate_odd = false) {{
  constexpr int MaxT = MaxA + MaxB;
  for (int i = 0; i <= MaxA; ++i) {{
    for (int j = 0; j <= MaxB; ++j) {{
      for (int t = 0; t <= MaxT; ++t) E[i][j][t] = R(0);
    }}
  }}
  E[0][0][0] = R(1);
  for (int i = 0; i < MaxA; ++i) {{
    for (int t = 0; t <= i + 1; ++t) {{
      R val = PA * E[i][0][t];
      if (t > 0) val += inv_2p * E[i][0][t - 1];
      if (t + 1 <= i) val += R(t + 1) * E[i][0][t + 1];
      E[i + 1][0][t] = val;
    }}
  }}
  for (int i = 0; i <= MaxA; ++i) {{
    for (int j = 0; j < MaxB; ++j) {{
      const int deg = i + j;
      for (int t = 0; t <= deg + 1; ++t) {{
        R val = PB * E[i][j][t];
        if (t > 0) val += inv_2p * E[i][j][t - 1];
        if (t + 1 <= deg) val += R(t + 1) * E[i][j][t + 1];
        E[i][j + 1][t] = val;
      }}
    }}
  }}
  if (negate_odd) {{
    for (int i = 0; i <= MaxA; ++i) {{
      for (int j = 0; j <= MaxB; ++j) {{
        for (int t = 1; t <= i + j; t += 2) E[i][j][t] = -E[i][j][t];
      }}
    }}
  }}
}}

template <typename R>
GPU4DFT_HD inline void md_coulomb_r(R rho, R scale, R PQx, R PQy, R PQz, const R* F,
                                    R R_out[{L + 1}][{L + 1}][{L + 1}]) {{
  constexpr int kL = {L};
  R R_cur[kL + 1][kL + 1][kL + 1];
  R R_nxt[kL + 1][kL + 1][kL + 1];
  for (int t = 0; t <= kL; ++t)
    for (int u = 0; u <= kL; ++u)
      for (int v = 0; v <= kL; ++v) R_cur[t][u][v] = R(0);

  R pow_m2rho[kL + 1];
  pow_m2rho[0] = scale;
  const R m2rho = R(-2) * rho;
  for (int m = 1; m <= kL; ++m) pow_m2rho[m] = pow_m2rho[m - 1] * m2rho;

  R_cur[0][0][0] = pow_m2rho[kL] * F[kL];
  for (int m = kL - 1; m >= 0; --m) {{
    const int max_deg = kL - m;
    R_nxt[0][0][0] = pow_m2rho[m] * F[m];
    for (int deg = 1; deg <= max_deg; ++deg) {{
      for (int t = 0; t <= deg; ++t) {{
        for (int u = 0; u <= deg - t; ++u) {{
          const int v = deg - t - u;
          if (t > 0) {{
            R_nxt[t][u][v] = PQx * R_cur[t - 1][u][v] +
                             (t > 1 ? R(t - 1) * R_cur[t - 2][u][v] : R(0));
          }} else if (u > 0) {{
            R_nxt[0][u][v] = PQy * R_cur[0][u - 1][v] +
                             (u > 1 ? R(u - 1) * R_cur[0][u - 2][v] : R(0));
          }} else {{
            R_nxt[0][0][v] = PQz * R_cur[0][0][v - 1] +
                             (v > 1 ? R(v - 1) * R_cur[0][0][v - 2] : R(0));
          }}
        }}
      }}
    }}
    for (int deg = 0; deg <= max_deg; ++deg) {{
      for (int t = 0; t <= deg; ++t) {{
        for (int u = 0; u <= deg - t; ++u) {{
          const int v = deg - t - u;
          R_cur[t][u][v] = R_nxt[t][u][v];
        }}
      }}
    }}
  }}
  for (int t = 0; t <= kL; ++t)
    for (int u = 0; u <= kL - t; ++u)
      for (int v = 0; v <= kL - t - u; ++v) R_out[t][u][v] = R_cur[t][u][v];
}}

}}  // namespace detail_{cls.name}

template <typename R>
GPU4DFT_HD inline void {prim_function_name(cls)}(const ShellPairView& bra, int ib, int ip,
                                                const ShellPairView& ket, int ik, int kp,
                                                double* cart_acc, double omega = 0.0) {{
{tabs}
  const R p = static_cast<R>(bra.exponent[ip]);
  const R Px = static_cast<R>(bra.Px[ip]);
  const R Py = static_cast<R>(bra.Py[ip]);
  const R Pz = static_cast<R>(bra.Pz[ip]);
  const R Cab = static_cast<R>(bra.prefactor[ip]);

  const R PAx = ({la} > 0) ? (Px - static_cast<R>(bra.Ax[ib])) : R(0);
  const R PAy = ({la} > 0) ? (Py - static_cast<R>(bra.Ay[ib])) : R(0);
  const R PAz = ({la} > 0) ? (Pz - static_cast<R>(bra.Az[ib])) : R(0);
  const R PBx = ({lb} > 0) ? (Px - static_cast<R>(bra.Bx[ib])) : R(0);
  const R PBy = ({lb} > 0) ? (Py - static_cast<R>(bra.By[ib])) : R(0);
  const R PBz = ({lb} > 0) ? (Pz - static_cast<R>(bra.Bz[ib])) : R(0);

  R Ex_ab[{la + 1}][{lb + 1}][{lab + 1}];
  R Ey_ab[{la + 1}][{lb + 1}][{lab + 1}];
  R Ez_ab[{la + 1}][{lb + 1}][{lab + 1}];
  const R inv_2p = R(0.5) / p;
  detail_{cls.name}::md_hermite_1d<R, {la}, {lb}>(inv_2p, PAx, PBx, Ex_ab, false);
  detail_{cls.name}::md_hermite_1d<R, {la}, {lb}>(inv_2p, PAy, PBy, Ey_ab, false);
  detail_{cls.name}::md_hermite_1d<R, {la}, {lb}>(inv_2p, PAz, PBz, Ez_ab, false);

  const R q = static_cast<R>(ket.exponent[kp]);
  const R Qx = static_cast<R>(ket.Px[kp]);
  const R Qy = static_cast<R>(ket.Py[kp]);
  const R Qz = static_cast<R>(ket.Pz[kp]);
  const R Ccd = static_cast<R>(ket.prefactor[kp]);

  const R QCx = ({lc} > 0) ? (Qx - static_cast<R>(ket.Ax[ik])) : R(0);
  const R QCy = ({lc} > 0) ? (Qy - static_cast<R>(ket.Ay[ik])) : R(0);
  const R QCz = ({lc} > 0) ? (Qz - static_cast<R>(ket.Az[ik])) : R(0);
  const R QDx = ({ld} > 0) ? (Qx - static_cast<R>(ket.Bx[ik])) : R(0);
  const R QDy = ({ld} > 0) ? (Qy - static_cast<R>(ket.By[ik])) : R(0);
  const R QDz = ({ld} > 0) ? (Qz - static_cast<R>(ket.Bz[ik])) : R(0);

  R Ex_cd[{lc + 1}][{ld + 1}][{lcd + 1}];
  R Ey_cd[{lc + 1}][{ld + 1}][{lcd + 1}];
  R Ez_cd[{lc + 1}][{ld + 1}][{lcd + 1}];
  const R inv_2q = R(0.5) / q;
  detail_{cls.name}::md_hermite_1d<R, {lc}, {ld}>(inv_2q, QCx, QDx, Ex_cd, true);
  detail_{cls.name}::md_hermite_1d<R, {lc}, {ld}>(inv_2q, QCy, QDy, Ey_cd, true);
  detail_{cls.name}::md_hermite_1d<R, {lc}, {ld}>(inv_2q, QCz, QDz, Ez_cd, true);

  const R PQx = Px - Qx;
  const R PQy = Py - Qy;
  const R PQz = Pz - Qz;
  const R rho0 = p * q / (p + q);
  const R theta = static_cast<R>(omega_theta<double>(static_cast<double>(rho0), omega));
  const R rho = rho0 * theta;
  const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
  R F[{L + 1}];
  boys_vec<R>({L}, T, F);
  const R scale = Cab * Ccd * R({TWO_OVER_SQRT_PI}) * math::sqrt(rho);

  R R_tuv[{L + 1}][{L + 1}][{L + 1}];
  detail_{cls.name}::md_coulomb_r<R>(rho, scale, PQx, PQy, PQz, F, R_tuv);

  // Contract ket (ic, id) -> Hermite bra potential V_bra[t][u][v][ic*ncd + id].
  R V_bra[{lab + 1}][{lab + 1}][{lab + 1}][{ncc * ncd}] = {{{{{{R(0)}}}}}};
  for (int t = 0; t <= {lab}; ++t) {{
    for (int u = 0; u <= {lab} - t; ++u) {{
      for (int v = 0; v <= {lab} - t - u; ++v) {{
        for (int ic = 0; ic < {ncc}; ++ic) {{
          for (int id = 0; id < {ncd}; ++id) {{
            const int tx_max = cx[ic] + dx[id];
            const int ty_max = cy[ic] + dy[id];
            const int tz_max = cz[ic] + dz[id];
            R sum_cd = R(0);
            for (int tp = 0; tp <= tx_max; ++tp) {{
              const R ex = Ex_cd[cx[ic]][dx[id]][tp];
              for (int up = 0; up <= ty_max; ++up) {{
                const R exy = ex * Ey_cd[cy[ic]][dy[id]][up];
                for (int vp = 0; vp <= tz_max; ++vp) {{
                  sum_cd += exy * Ez_cd[cz[ic]][dz[id]][vp] * R_tuv[t + tp][u + up][v + vp];
                }}
              }}
            }}
            V_bra[t][u][v][ic * {ncd} + id] = sum_cd;
          }}
        }}
      }}
    }}
  }}

  for (int ia = 0; ia < {nca}; ++ia) {{
    for (int ib_c = 0; ib_c < {ncb}; ++ib_c) {{
      const int tx_max = ax[ia] + bx[ib_c];
      const int ty_max = ay[ia] + by[ib_c];
      const int tz_max = az[ia] + bz[ib_c];
      double* dst = cart_acc + (ia * {ncb} + ib_c) * {ncc * ncd};
      for (int t = 0; t <= tx_max; ++t) {{
        const R ex = Ex_ab[ax[ia]][bx[ib_c]][t];
        for (int u = 0; u <= ty_max; ++u) {{
          const R exy = ex * Ey_ab[ay[ia]][by[ib_c]][u];
          for (int v = 0; v <= tz_max; ++v) {{
            const R w_ab = exy * Ez_ab[az[ia]][bz[ib_c]][v];
            for (int cd_idx = 0; cd_idx < {ncc * ncd}; ++cd_idx) {{
              dst[cd_idx] += static_cast<double>(w_ab * V_bra[t][u][v][cd_idx]);
            }}
          }}
        }}
      }}
    }}
  }}
}}

{c2s_fn}
template <typename R>
GPU4DFT_HD inline void {eval_function_name(cls)}(const ShellPairView& bra, int ib,
                                                const ShellPairView& ket, int ik, double* out,
                                                double omega = 0.0) {{
  double cart[{nc_tot}] = {{0.0}};
  const int bra_begin = bra.prim_offset[ib];
  const int bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik];
  const int ket_end = ket.prim_offset[ik + 1];
  for (int ip = bra_begin; ip < bra_end; ++ip) {{
    {unroll}for (int kp = ket_begin; kp < ket_end; ++kp) {{
      {prim_function_name(cls)}<R>(bra, ib, ip, ket, ik, kp, cart, omega);
    }}
  }}
  {cart2sph_function_name(cls)}(cart, out);
}}
"""


def _emit_rys_body(cls: EriClass, prim_unroll: int) -> str:
    la, lb, lc, ld = cls.angular
    lab = la + lb
    lcd = lc + ld
    n_roots = (la + lb + lc + ld) // 2 + 1
    nca, ncb, ncc, ncd = n_cart(la), n_cart(lb), n_cart(lc), n_cart(ld)
    nc_tot = cls.n_cart_components
    ns_tot = cls.n_sph_components
    unroll = _unroll_pragma(prim_unroll)

    tabs = "\n".join(
        [
            _emit_cart_index_table("a", la),
            _emit_cart_index_table("b", lb),
            _emit_cart_index_table("c", lc),
            _emit_cart_index_table("d", ld),
        ]
    )
    c2s_fn = _emit_cart_to_sph_fn(cls)
    extra_ssss = ""
    if cls.angular == (0, 0, 0, 0):
        extra_ssss = f"""
template <typename R>
GPU4DFT_HD inline double eri_ssss_contracted(const ShellPairView& bra, int ib,
                                             const ShellPairView& ket, int ik,
                                             double omega = 0.0) {{
  double v = 0.0;
  {eval_function_name(cls)}<R>(bra, ib, ket, ik, &v, omega);
  return v;
}}
"""

    return f"""\
inline constexpr int kEri_{cls.name}_NCart = {nc_tot};
inline constexpr int kEri_{cls.name}_NSph = {ns_tot};

namespace detail_{cls.name} {{

template <typename R>
GPU4DFT_HD inline void rys_1d_vrr_hrr(R C00, R D00, R B10, R B01, R B00, R AB, R CD,
                                      R I[{la + 1}][{lb + 1}][{lc + 1}][{ld + 1}]) {{
  constexpr int kLab = {lab};
  constexpr int kLcd = {lcd};
  R G[kLab + 1][kLcd + 1];
  G[0][0] = R(1);
  for (int n = 0; n < kLab; ++n) {{
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? R(n) * B10 * G[n - 1][0] : R(0));
  }}
  for (int m = 0; m < kLcd; ++m) {{
    for (int n = 0; n <= kLab; ++n) {{
      R val = D00 * G[n][m];
      if (m > 0) val += R(m) * B01 * G[n][m - 1];
      if (n > 0) val += R(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }}
  }}

  // 1D Bra HRR: (i, j+1 | m) = (i+1, j | m) + AB * (i, j | m)
  R H[{la + 1}][{lb + 1}][kLcd + 1];
  R cur[kLab + 1][kLcd + 1];
  for (int n = 0; n <= kLab; ++n)
    for (int m = 0; m <= kLcd; ++m) cur[n][m] = G[n][m];
  for (int j = 0; j <= {lb}; ++j) {{
    for (int i = 0; i <= {la}; ++i)
      for (int m = 0; m <= kLcd; ++m) H[i][j][m] = cur[i][m];
    if (j < {lb}) {{
      for (int i = 0; i <= kLab - j - 1; ++i)
        for (int m = 0; m <= kLcd; ++m) cur[i][m] = cur[i + 1][m] + AB * cur[i][m];
    }}
  }}

  // 1D Ket HRR: (i, j | k, l+1) = (i, j | k+1, l) + CD * (i, j | k, l)
  for (int i = 0; i <= {la}; ++i) {{
    for (int j = 0; j <= {lb}; ++j) {{
      R kcur[kLcd + 1];
      for (int m = 0; m <= kLcd; ++m) kcur[m] = H[i][j][m];
      for (int l = 0; l <= {ld}; ++l) {{
        for (int k = 0; k <= {lc}; ++k) I[i][j][k][l] = kcur[k];
        if (l < {ld}) {{
          for (int k = 0; k <= kLcd - l - 1; ++k) kcur[k] = kcur[k + 1] + CD * kcur[k];
        }}
      }}
    }}
  }}
}}

}}  // namespace detail_{cls.name}

template <typename R>
GPU4DFT_HD inline void {prim_function_name(cls)}(const ShellPairView& bra, int ib, int ip,
                                                const ShellPairView& ket, int ik, int kp,
                                                double* cart_acc, double omega = 0.0) {{
{tabs}
  (void)AB_unused_{cls.name};
  const R p = static_cast<R>(bra.exponent[ip]);
  const R q = static_cast<R>(ket.exponent[kp]);
  const R Px = static_cast<R>(bra.Px[ip]);
  const R Py = static_cast<R>(bra.Py[ip]);
  const R Pz = static_cast<R>(bra.Pz[ip]);
  const R Qx = static_cast<R>(ket.Px[kp]);
  const R Qy = static_cast<R>(ket.Py[kp]);
  const R Qz = static_cast<R>(ket.Pz[kp]);
  const R Cab = static_cast<R>(bra.prefactor[ip]);
  const R Ccd = static_cast<R>(ket.prefactor[kp]);

  const R Ax = ({lab} > 0) ? static_cast<R>(bra.Ax[ib]) : R(0);
  const R Ay = ({lab} > 0) ? static_cast<R>(bra.Ay[ib]) : R(0);
  const R Az = ({lab} > 0) ? static_cast<R>(bra.Az[ib]) : R(0);
  const R Bx = ({lb} > 0) ? static_cast<R>(bra.Bx[ib]) : R(0);
  const R By = ({lb} > 0) ? static_cast<R>(bra.By[ib]) : R(0);
  const R Bz = ({lb} > 0) ? static_cast<R>(bra.Bz[ib]) : R(0);
  const R Cx = ({lcd} > 0) ? static_cast<R>(ket.Ax[ik]) : R(0);
  const R Cy = ({lcd} > 0) ? static_cast<R>(ket.Ay[ik]) : R(0);
  const R Cz = ({lcd} > 0) ? static_cast<R>(ket.Az[ik]) : R(0);
  const R Dx = ({ld} > 0) ? static_cast<R>(ket.Bx[ik]) : R(0);
  const R Dy = ({ld} > 0) ? static_cast<R>(ket.By[ik]) : R(0);
  const R Dz = ({ld} > 0) ? static_cast<R>(ket.Bz[ik]) : R(0);

  const R PQx = Px - Qx;
  const R PQy = Py - Qy;
  const R PQz = Pz - Qz;
  const R pq_sum = p + q;
  const R rho = p * q / pq_sum;
  const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
  const R scale = Cab * Ccd * R({TWO_OVER_SQRT_PI}) * math::sqrt(rho);

  R u_roots[{n_roots}];
  R w_weights[{n_roots}];
  rys_roots_weights_erf<R>({n_roots}, T, rho, static_cast<R>(omega), u_roots, w_weights);

  const R inv_2p = R(0.5) / p;
  const R inv_2q = R(0.5) / q;
  const R inv_2pq = R(0.5) / pq_sum;
  const R rho_over_p = rho / p;
  const R rho_over_q = rho / q;
  const R PAx = Px - Ax, PAy = Py - Ay, PAz = Pz - Az;
  const R QCx = Qx - Cx, QCy = Qy - Cy, QCz = Qz - Cz;
  const R ABx = Ax - Bx, ABy = Ay - By, ABz = Az - Bz;
  const R CDx = Cx - Dx, CDy = Cy - Dy, CDz = Cz - Dz;

  for (int r = 0; r < {n_roots}; ++r) {{
    const R u = u_roots[r];
    const R Wr = scale * w_weights[r];
    const R B00 = inv_2pq * u;
    const R B10 = inv_2p * (R(1) - rho_over_p * u);
    const R B01 = inv_2q * (R(1) - rho_over_q * u);

    R Ix[{la + 1}][{lb + 1}][{lc + 1}][{ld + 1}];
    R Iy[{la + 1}][{lb + 1}][{lc + 1}][{ld + 1}];
    R Iz[{la + 1}][{lb + 1}][{lc + 1}][{ld + 1}];
    detail_{cls.name}::rys_1d_vrr_hrr<R>(PAx - rho_over_p * u * PQx, QCx + rho_over_q * u * PQx,
                                         B10, B01, B00, ABx, CDx, Ix);
    detail_{cls.name}::rys_1d_vrr_hrr<R>(PAy - rho_over_p * u * PQy, QCy + rho_over_q * u * PQy,
                                         B10, B01, B00, ABy, CDy, Iy);
    detail_{cls.name}::rys_1d_vrr_hrr<R>(PAz - rho_over_p * u * PQz, QCz + rho_over_q * u * PQz,
                                         B10, B01, B00, ABz, CDz, Iz);

    for (int ia = 0; ia < {nca}; ++ia) {{
      for (int ib_c = 0; ib_c < {ncb}; ++ib_c) {{
        for (int ic = 0; ic < {ncc}; ++ic) {{
          for (int id = 0; id < {ncd}; ++id) {{
            const R v = Wr * Ix[ax[ia]][bx[ib_c]][cx[ic]][dx[id]] *
                        Iy[ay[ia]][by[ib_c]][cy[ic]][dy[id]] *
                        Iz[az[ia]][bz[ib_c]][cz[ic]][dz[id]];
            cart_acc[((ia * {ncb} + ib_c) * {ncc} + ic) * {ncd} + id] += static_cast<double>(v);
          }}
        }}
      }}
    }}
  }}
}}

{c2s_fn}{extra_ssss}
template <typename R>
GPU4DFT_HD inline void {eval_function_name(cls)}(const ShellPairView& bra, int ib,
                                                const ShellPairView& ket, int ik, double* out,
                                                double omega = 0.0) {{
  double cart[{nc_tot}] = {{0.0}};
  const int bra_begin = bra.prim_offset[ib];
  const int bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik];
  const int ket_end = ket.prim_offset[ik + 1];
  for (int ip = bra_begin; ip < bra_end; ++ip) {{
    {unroll}for (int kp = ket_begin; kp < ket_end; ++kp) {{
      {prim_function_name(cls)}<R>(bra, ib, ip, ket, ik, kp, cart, omega);
    }}
  }}
  {cart2sph_function_name(cls)}(cart, out);
}}
"""


def emit_class_header(cls: EriClass, tuning: CudaTuning | None = None) -> str:
    """Return the backend-neutral header implementing ``cls`` (supports l <= 4)."""
    for l in cls.angular:
        if l < 0 or l > MAX_SUPPORTED_L:
            raise NotImplementedError(
                f"no integral algorithm for ({cls.name}) yet: angular momentum "
                f"l={l} exceeds supported max_l={MAX_SUPPORTED_L}"
            )
    cfg = tuning if tuning is not None else CudaTuning()
    alg = cfg.resolved_algorithm(cls)
    if alg == "md":
        body = _emit_md_body(cls, cfg.prim_unroll)
        includes = (
            '#include "gpu4dft/boys.hpp"\n'
            '#include "gpu4dft/config.hpp"\n'
            '#include "gpu4dft/harmonics.hpp"\n'
            '#include "gpu4dft/math.hpp"\n'
            '#include "gpu4dft/shell_pair_view.hpp"'
        )
    else:
        body = _emit_rys_body(cls, cfg.prim_unroll).replace(
            f"  (void)AB_unused_{cls.name};\n", ""
        )
        includes = (
            '#include "gpu4dft/boys.hpp"\n'
            '#include "gpu4dft/config.hpp"\n'
            '#include "gpu4dft/harmonics.hpp"\n'
            '#include "gpu4dft/math.hpp"\n'
            '#include "gpu4dft/rys.hpp"\n'
            '#include "gpu4dft/shell_pair_view.hpp"'
        )
    return f"""\
// Generated by gpu4dft_codegen. Do not edit.
#pragma once

{includes}

namespace gpu4dft::generated {{

{body}
}}  // namespace gpu4dft::generated
"""


def emit_general_header() -> str:
    """Emit backend-neutral general arbitrary-l (0 <= l <= 4: s,p,d,f,g) ERI header."""
    return f"""\
// Generated by gpu4dft_codegen. Do not edit.
#pragma once

#include "gpu4dft/boys.hpp"
#include "gpu4dft/config.hpp"
#include "gpu4dft/harmonics.hpp"
#include "gpu4dft/math.hpp"
#include "gpu4dft/rys.hpp"
#include "gpu4dft/shell_pair_view.hpp"

namespace gpu4dft::generated {{

namespace detail_general {{

template <typename R>
GPU4DFT_HD inline void rys_1d_general(int la, int lb, int lc, int ld, R C00, R D00, R B10, R B01,
                                      R B00, R AB, R CD, R I[5][5][5][5]) {{
  const int lab = la + lb;
  const int lcd = lc + ld;
  R G[9][9];
  G[0][0] = R(1);
  for (int n = 0; n < lab; ++n) {{
    G[n + 1][0] = C00 * G[n][0] + (n > 0 ? R(n) * B10 * G[n - 1][0] : R(0));
  }}
  for (int m = 0; m < lcd; ++m) {{
    for (int n = 0; n <= lab; ++n) {{
      R val = D00 * G[n][m];
      if (m > 0) val += R(m) * B01 * G[n][m - 1];
      if (n > 0) val += R(n) * B00 * G[n - 1][m];
      G[n][m + 1] = val;
    }}
  }}

  R H[5][5][9];
  R cur[9][9];
  for (int n = 0; n <= lab; ++n)
    for (int m = 0; m <= lcd; ++m) cur[n][m] = G[n][m];
  for (int j = 0; j <= lb; ++j) {{
    for (int i = 0; i <= la; ++i)
      for (int m = 0; m <= lcd; ++m) H[i][j][m] = cur[i][m];
    if (j < lb) {{
      for (int i = 0; i <= lab - j - 1; ++i)
        for (int m = 0; m <= lcd; ++m) cur[i][m] = cur[i + 1][m] + AB * cur[i][m];
    }}
  }}

  for (int i = 0; i <= la; ++i) {{
    for (int j = 0; j <= lb; ++j) {{
      R kcur[9];
      for (int m = 0; m <= lcd; ++m) kcur[m] = H[i][j][m];
      for (int l = 0; l <= ld; ++l) {{
        for (int k = 0; k <= lc; ++k) I[i][j][k][l] = kcur[k];
        if (l < ld) {{
          for (int k = 0; k <= lcd - l - 1; ++k) kcur[k] = kcur[k + 1] + CD * kcur[k];
        }}
      }}
    }}
  }}
}}

}}  // namespace detail_general

// Evaluates any contracted spherical quartet (la, lb | lc, ld) up to l = 4 (g shells)
// with bounded stack usage (< 4 KB even for (gg|gg)) by transforming ket (c, d)
// blocks per (ia, ib) pair directly into `sph_out`.
template <typename R>
GPU4DFT_HD inline void eri_general_eval(int la, int lb, int lc, int ld, const ShellPairView& bra,
                                        int ib, const ShellPairView& ket, int ik,
                                        double* sph_out, double omega = 0.0) {{
  const int nca = n_cart(la), ncb = n_cart(lb), ncc = n_cart(lc), ncd = n_cart(ld);
  const int nsa = n_sph(la), nsb = n_sph(lb), nsc = n_sph(lc), nsd = n_sph(ld);
  const int nscd = nsc * nsd;
  const int nsbsd = nsb * nscd;
  const int total_sph = nsa * nsbsd;
  for (int idx = 0; idx < total_sph; ++idx) sph_out[idx] = 0.0;

  const int n_roots = (la + lb + lc + ld) / 2 + 1;
  const double Ax = bra.Ax ? bra.Ax[ib] : 0.0;
  const double Ay = bra.Ay ? bra.Ay[ib] : 0.0;
  const double Az = bra.Az ? bra.Az[ib] : 0.0;
  const double Bx = bra.Bx ? bra.Bx[ib] : 0.0;
  const double By = bra.By ? bra.By[ib] : 0.0;
  const double Bz = bra.Bz ? bra.Bz[ib] : 0.0;
  const double Cx = ket.Ax ? ket.Ax[ik] : 0.0;
  const double Cy = ket.Ay ? ket.Ay[ik] : 0.0;
  const double Cz = ket.Az ? ket.Az[ik] : 0.0;
  const double Dx = ket.Bx ? ket.Bx[ik] : 0.0;
  const double Dy = ket.By ? ket.By[ik] : 0.0;
  const double Dz = ket.Bz ? ket.Bz[ik] : 0.0;
  const R ABx = static_cast<R>(Ax - Bx), ABy = static_cast<R>(Ay - By), ABz = static_cast<R>(Az - Bz);
  const R CDx = static_cast<R>(Cx - Dx), CDy = static_cast<R>(Cy - Dy), CDz = static_cast<R>(Cz - Dz);

  CartTriple tc[15], td[15];
  for (int ic = 0; ic < ncc; ++ic) tc[ic] = cart_triple(lc, ic);
  for (int id = 0; id < ncd; ++id) td[id] = cart_triple(ld, id);

  const int bra_begin = bra.prim_offset[ib];
  const int bra_end = bra.prim_offset[ib + 1];
  const int ket_begin = ket.prim_offset[ik];
  const int ket_end = ket.prim_offset[ik + 1];

  for (int ip = bra_begin; ip < bra_end; ++ip) {{
    const R p = static_cast<R>(bra.exponent[ip]);
    const R Px = static_cast<R>(bra.Px[ip]);
    const R Py = static_cast<R>(bra.Py[ip]);
    const R Pz = static_cast<R>(bra.Pz[ip]);
    const R Cab = static_cast<R>(bra.prefactor[ip]);
    const R PAx = Px - static_cast<R>(Ax);
    const R PAy = Py - static_cast<R>(Ay);
    const R PAz = Pz - static_cast<R>(Az);
    const R inv_2p = R(0.5) / p;

    for (int kp = ket_begin; kp < ket_end; ++kp) {{
      const R q = static_cast<R>(ket.exponent[kp]);
      const R Qx = static_cast<R>(ket.Px[kp]);
      const R Qy = static_cast<R>(ket.Py[kp]);
      const R Qz = static_cast<R>(ket.Pz[kp]);
      const R Ccd = static_cast<R>(ket.prefactor[kp]);
      const R PQx = Px - Qx, PQy = Py - Qy, PQz = Pz - Qz;
      const R pq_sum = p + q;
      const R rho = p * q / pq_sum;
      const R T = rho * (PQx * PQx + PQy * PQy + PQz * PQz);
      const R scale = Cab * Ccd * R({TWO_OVER_SQRT_PI}) * math::sqrt(rho);

      R u_roots[9];
      R w_weights[9];
      rys_roots_weights_erf<R>(n_roots, T, rho, static_cast<R>(omega), u_roots, w_weights);

      const R inv_2q = R(0.5) / q;
      const R inv_2pq = R(0.5) / pq_sum;
      const R rho_over_p = rho / p;
      const R rho_over_q = rho / q;
      const R QCx = Qx - static_cast<R>(Cx);
      const R QCy = Qy - static_cast<R>(Cy);
      const R QCz = Qz - static_cast<R>(Cz);

      R Ix_r[9][5][5][5][5];
      R Iy_r[9][5][5][5][5];
      R Iz_r[9][5][5][5][5];
      R W_r[9];
      for (int r = 0; r < n_roots; ++r) {{
        const R u = u_roots[r];
        W_r[r] = scale * w_weights[r];
        const R B00 = inv_2pq * u;
        const R B10 = inv_2p * (R(1) - rho_over_p * u);
        const R B01 = inv_2q * (R(1) - rho_over_q * u);
        detail_general::rys_1d_general<R>(la, lb, lc, ld, PAx - rho_over_p * u * PQx,
                                          QCx + rho_over_q * u * PQx, B10, B01, B00, ABx, CDx,
                                          Ix_r[r]);
        detail_general::rys_1d_general<R>(la, lb, lc, ld, PAy - rho_over_p * u * PQy,
                                          QCy + rho_over_q * u * PQy, B10, B01, B00, ABy, CDy,
                                          Iy_r[r]);
        detail_general::rys_1d_general<R>(la, lb, lc, ld, PAz - rho_over_p * u * PQz,
                                          QCz + rho_over_q * u * PQz, B10, B01, B00, ABz, CDz,
                                          Iz_r[r]);
      }}

      double unit[15] = {{0.0}};
      double ca_col[9] = {{0.0}}, cb_col[9] = {{0.0}};
      double cart_cd[225] = {{0.0}};
      double tmp_cd[135] = {{0.0}};
      double sph_cd[81] = {{0.0}};

      for (int ia = 0; ia < nca; ++ia) {{
        const CartTriple ta = cart_triple(la, ia);
        unit[ia] = 1.0;
        harmonics::cart_to_sph_axis<double>(la, unit, 1, ca_col, 1);
        unit[ia] = 0.0;

        for (int ib_c = 0; ib_c < ncb; ++ib_c) {{
          const CartTriple tb = cart_triple(lb, ib_c);
          unit[ib_c] = 1.0;
          harmonics::cart_to_sph_axis<double>(lb, unit, 1, cb_col, 1);
          unit[ib_c] = 0.0;

          for (int ic = 0; ic < ncc; ++ic) {{
            for (int id = 0; id < ncd; ++id) {{
              double acc = 0.0;
              for (int r = 0; r < n_roots; ++r) {{
                const R prod = W_r[r] * Ix_r[r][ta.lx][tb.lx][tc[ic].lx][td[id].lx] *
                               Iy_r[r][ta.ly][tb.ly][tc[ic].ly][td[id].ly] *
                               Iz_r[r][ta.lz][tb.lz][tc[ic].lz][td[id].lz];
                acc += static_cast<double>(prod);
              }}
              cart_cd[ic * ncd + id] = acc;
            }}
            harmonics::cart_to_sph_axis<double>(ld, cart_cd + ic * ncd, 1, tmp_cd + ic * nsd, 1);
          }}
          for (int md = 0; md < nsd; ++md) {{
            harmonics::cart_to_sph_axis<double>(lc, tmp_cd + md, nsd, sph_cd + md, nsd);
          }}

          for (int ma = 0; ma < nsa; ++ma) {{
            const double wa = ca_col[ma];
            if (wa == 0.0) continue;
            for (int mb = 0; mb < nsb; ++mb) {{
              const double wab = wa * cb_col[mb];
              if (wab == 0.0) continue;
              double* dst = sph_out + ma * nsbsd + mb * nscd;
              for (int mcd = 0; mcd < nscd; ++mcd) {{
                dst[mcd] += wab * sph_cd[mcd];
              }}
            }}
          }}
        }}
      }}
    }}
  }}
}}

}}  // namespace gpu4dft::generated
"""
