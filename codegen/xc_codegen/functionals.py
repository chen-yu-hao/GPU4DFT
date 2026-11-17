"""Exact spin-unpolarized RKS symbolic XC functionals matching Libxc / PySCF."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .sym_diff import (
    ONE,
    TWO,
    Expr,
    asinh,
    atan,
    c,
    cbrt,
    cube,
    diff,
    exp,
    expm1,
    log,
    log1p,
    sq,
    sqrt,
    sym_max,
    var,
    where,
)


@dataclass(frozen=True)
class BaseFunctionalSpec:
    name: str
    family: str  # "LDA", "GGA", "MGGA"
    zk_expr: Expr
    f_expr: Expr  # rho * zk_expr


@dataclass(frozen=True)
class NamedFunctionalSpec:
    name: str
    aliases: tuple[str, ...]
    family: str  # "LDA", "GGA", "MGGA"
    hyb_exx: float
    components: tuple[tuple[str, float], ...]


def _lda_x_zk(rho: Expr) -> Expr:
    cx = 0.75 * ((3.0 / math.pi) ** (1.0 / 3.0))
    return -c(cx) * cbrt(rho)


def _vwn_zk(rho: Expr, A: float, x0: float, b: float, c_param: float) -> Expr:
    rs_const = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    rs = c(rs_const) / cbrt(rho)
    x = sqrt(rs)
    X = sq(x) + c(b) * x + c(c_param)
    X0 = x0 * x0 + b * x0 + c_param
    Q = math.sqrt(4.0 * c_param - b * b)
    at = atan(c(Q) / (TWO * x + c(b)))
    term1 = log(sq(x) / X)
    term2 = c(2.0 * b / Q) * at
    term3 = c(b * x0 / X0) * (
        log(sq(x - c(x0)) / X) + c(2.0 * (b + 2.0 * x0) / Q) * at
    )
    return c(A) * (term1 + term2 - term3)


def _lda_c_pz_zk(rho: Expr) -> Expr:
    rs_const = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    rs = c(rs_const) / cbrt(rho)
    low_density = c(-0.1423) / (ONE + c(1.0529) * sqrt(rs) + c(0.3334) * rs)
    ln_rs = log(rs)
    high_density = c(0.0311) * ln_rs + c(-0.0480) + c(0.0020) * rs * ln_rs + c(-0.0116) * rs
    return where("ge", rs, ONE, low_density, high_density)


def _gga_x_b88_zk(rho: Expr, sigma: Expr) -> Expr:
    cx = 0.75 * ((3.0 / math.pi) ** (1.0 / 3.0))
    beta = 0.0042
    r13 = cbrt(rho)
    rho83 = sq(rho) * sq(r13)
    # u = x_alpha^2 = 2^(2/3) * sigma / rho^(8/3)
    u = c(2.0 ** (2.0 / 3.0)) * sigma / rho83
    # Smooth C^inf handling around u -> 0 (sigma -> 0) avoiding 1/sqrt(0) in derivatives:
    u_safe = where("ge", u, c(1e-10), u, ONE)
    xa = sqrt(u_safe)
    xa_asinh_large = xa * asinh(xa)
    u2 = sq(u)
    xa_asinh_small = u * (
        ONE + u * (c(-1.0 / 6.0) + u * (c(3.0 / 40.0) + u * (c(-5.0 / 112.0) + c(35.0 / 1152.0) * u)))
    )
    _ = u2  # keep clarity
    xa_asinh = where("ge", u, c(1e-10), xa_asinh_large, xa_asinh_small)
    g = u / (ONE + c(6.0 * beta) * xa_asinh)
    return (-c(cx) - c((2.0 ** (-1.0 / 3.0)) * beta) * g) * r13


def _gga_c_lyp_zk(rho: Expr, sigma: Expr) -> Expr:
    a_lyp, b_lyp, c_lyp, d_lyp = 0.04918, 0.132, 0.2533, 0.349
    cf = 0.3 * ((3.0 * (math.pi ** 2)) ** (2.0 / 3.0))
    r13 = cbrt(rho)
    inv_r13 = ONE / r13
    inv_denom = ONE / (ONE + c(d_lyp) * inv_r13)
    # Note: f_lyp / rho = -a * inv_denom - a * b * exp(-c * r^-1/3) * inv_denom * [CF - (3 + 7*delta)/72 * sigma / rho^(8/3)]
    # Writing zk directly cancels rho^(-11/3) * rho^(14/3) = rho and avoids tiny/huge intermediate powers!
    delta = (c(c_lyp) + c(d_lyp) * inv_denom) * inv_r13
    rho83 = sq(rho) * sq(r13)
    grad_term = (c(3.0) + c(7.0) * delta) * c(1.0 / 72.0) * (sigma / rho83)
    return -c(a_lyp) * inv_denom - c(a_lyp * b_lyp) * exp(-c(c_lyp) * inv_r13) * inv_denom * (
        c(cf) - grad_term
    )


def _gga_x_pbe_zk(rho: Expr, sigma: Expr) -> Expr:
    cx = 0.75 * ((3.0 / math.pi) ** (1.0 / 3.0))
    kappa = 0.804
    mu = 0.2195149727645171  # Libxc gga_x_pbe exact mu = beta_MB * pi^2 / 3
    c_s2 = 1.0 / (4.0 * ((3.0 * (math.pi ** 2)) ** (2.0 / 3.0)))
    r13 = cbrt(rho)
    rho83 = sq(rho) * sq(r13)
    s2 = c(c_s2) * sigma / rho83
    fx = c(1.0 + kappa) - c(kappa) / (ONE + c(mu / kappa) * s2)
    return -c(cx) * r13 * fx


def _pbe_c_spin_zk(rho_tot: Expr, sigma_tot: Expr, polarized: bool) -> Expr:
    """PW92_mod + PBE correlation per particle for zeta=0 (polarized=False) or zeta=1 (polarized=True)."""
    gamma = (1.0 - math.log(2.0)) / (math.pi ** 2)
    beta = 0.06672455060314922
    rs_const = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
    r13 = cbrt(rho_tot)
    rs = c(rs_const) / r13
    srs = sqrt(rs)

    if not polarized:
        A, a1, b1, b2, b3, b4 = 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
        phi = 1.0
    else:
        A, a1, b1, b2, b3, b4 = 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517
        phi = 2.0 ** (-1.0 / 3.0)

    denom = c(2.0 * A) * (
        c(b1) * srs + c(b2) * rs + c(b3) * rs * srs + c(b4) * sq(rs)
    )
    ec_pw = c(-2.0 * A) * (ONE + c(a1) * rs) * log1p(ONE / denom)

    gp3 = gamma * (phi ** 3)
    t2_const = math.pi / (16.0 * ((3.0 * (math.pi ** 2)) ** (1.0 / 3.0)) * (phi ** 2))
    rho73 = sq(rho_tot) * r13
    t2 = c(t2_const) * sigma_tot / rho73

    A_pbe = c(beta / gamma) / expm1(-ec_pw / c(gp3))
    At2 = A_pbe * t2
    H = c(gp3) * log1p(
        c(beta / gamma) * t2 * (ONE + At2) / (ONE + At2 + sq(At2))
    )
    return ec_pw + H


def _gga_c_pbe_zk(rho: Expr, sigma: Expr) -> Expr:
    return _pbe_c_spin_zk(rho, sigma, polarized=False)


def _mgga_x_tpss_zk(rho: Expr, sigma: Expr, tau: Expr) -> Expr:
    cx = 0.75 * ((3.0 / math.pi) ** (1.0 / 3.0))
    kappa = 0.804
    mu = 0.21951  # Exact TPSS 2003 / Libxc mgga_x_tpss mu
    b_tpss = 0.40
    c_tpss = 1.59096
    e_tpss = 1.537
    se = math.sqrt(e_tpss)

    r13 = cbrt(rho)
    r23 = sq(r13)
    rho53 = rho * r23
    rho83 = sq(rho) * r23

    c_p = 1.0 / (4.0 * ((3.0 * (math.pi ** 2)) ** (2.0 / 3.0)))
    c_tunif = 0.3 * ((3.0 * (math.pi ** 2)) ** (2.0 / 3.0))

    tau_safe = where("gt", tau, c(1e-30), tau, c(1e-30))
    tau_w = c(0.125) * sigma / rho
    tau_unif = c(c_tunif) * rho53

    p = c(c_p) * sigma / rho83
    z = tau_w / tau_safe
    z2 = sq(z)
    alpha = (tau_safe - tau_w) / tau_unif

    qb = c(9.0 / 20.0) * (alpha - ONE) / sqrt(
        ONE + c(b_tpss) * alpha * (alpha - ONE)
    ) + c(2.0 / 3.0) * p

    # Smooth factored radical: sqrt(0.5*(0.6*z)^2 + 0.5*p^2) = sigma * sqrt(...)
    # Strictly C^inf at sigma = 0 with zero 0/0 singularities!
    z_over_sigma = c(0.125) / (rho * tau_safe)
    p_over_sigma = c(c_p) / rho83
    rad = sigma * sqrt(
        c(0.5) * (sq(c(0.6) * z_over_sigma) + sq(p_over_sigma))
    )

    c10_81 = 10.0 / 81.0
    x_num = (
        (c(c10_81) + c(c_tpss) * z2 / sq(ONE + z2)) * p
        + c(146.0 / 2025.0) * sq(qb)
        - c(73.0 / 405.0) * qb * rad
        + c((c10_81 ** 2) / kappa) * sq(p)
        + c(2.0 * se * c10_81 * 0.36) * z2
        + c(e_tpss * mu) * cube(p)
    )
    x_tpss = x_num / sq(ONE + c(se) * p)
    fx = c(1.0 + kappa) - c(kappa) / (ONE + x_tpss / c(kappa))
    return -c(cx) * r13 * fx


def _mgga_c_tpss_zk(rho: Expr, sigma: Expr, tau: Expr) -> Expr:
    tau_safe = where("gt", tau, c(1e-30), tau, c(1e-30))
    tau_w = c(0.125) * sigma / rho
    z = tau_w / tau_safe
    z2 = sq(z)
    z3 = z2 * z

    ec_unpol = _pbe_c_spin_zk(rho, sigma, polarized=False)
    ec_pol = _pbe_c_spin_zk(c(0.5) * rho, c(0.25) * sigma, polarized=True)
    ec_tilde = sym_max(ec_pol, ec_unpol)

    ec_revpkzb = ec_unpol * (ONE + c(0.53) * z2) - c(1.53) * z2 * ec_tilde
    return ec_revpkzb * (ONE + c(2.8) * ec_revpkzb * z3)


def build_base_functionals() -> list[BaseFunctionalSpec]:
    rho = var("rho")
    sigma = var("sigma")
    tau = var("tau")

    defs: list[tuple[str, str, Expr]] = [
        ("lda_x", "LDA", _lda_x_zk(rho)),
        ("lda_c_vwn5", "LDA", _vwn_zk(rho, 0.0310907, -0.10498, 3.72744, 12.9352)),
        ("lda_c_vwn_rpa", "LDA", _vwn_zk(rho, 0.0310907, -0.409286, 13.0720, 42.7198)),
        ("lda_c_pz", "LDA", _lda_c_pz_zk(rho)),
        ("gga_x_b88", "GGA", _gga_x_b88_zk(rho, sigma)),
        ("gga_c_lyp", "GGA", _gga_c_lyp_zk(rho, sigma)),
        ("gga_x_pbe", "GGA", _gga_x_pbe_zk(rho, sigma)),
        ("gga_c_pbe", "GGA", _gga_c_pbe_zk(rho, sigma)),
        ("mgga_x_tpss", "MGGA", _mgga_x_tpss_zk(rho, sigma, tau)),
        ("mgga_c_tpss", "MGGA", _mgga_c_tpss_zk(rho, sigma, tau)),
    ]
    return [BaseFunctionalSpec(name, fam, zk, rho * zk) for name, fam, zk in defs]


NAMED_FUNCTIONALS: tuple[NamedFunctionalSpec, ...] = (
    # Pure individual components
    NamedFunctionalSpec("lda_x", ("slater",), "LDA", 0.0, (("lda_x", 1.0),)),
    NamedFunctionalSpec("lda_c_vwn5", ("lda_c_vwn", "vwn5", "vwn"), "LDA", 0.0, (("lda_c_vwn5", 1.0),)),
    NamedFunctionalSpec("lda_c_vwn_rpa", ("vwn_rpa",), "LDA", 0.0, (("lda_c_vwn_rpa", 1.0),)),
    NamedFunctionalSpec("lda_c_pz", ("pz81", "pz"), "LDA", 0.0, (("lda_c_pz", 1.0),)),
    NamedFunctionalSpec("gga_x_b88", ("b88",), "GGA", 0.0, (("gga_x_b88", 1.0),)),
    NamedFunctionalSpec("gga_c_lyp", ("lyp",), "GGA", 0.0, (("gga_c_lyp", 1.0),)),
    NamedFunctionalSpec("gga_x_pbe", ("pbe_x",), "GGA", 0.0, (("gga_x_pbe", 1.0),)),
    NamedFunctionalSpec("gga_c_pbe", ("pbe_c",), "GGA", 0.0, (("gga_c_pbe", 1.0),)),
    NamedFunctionalSpec("mgga_x_tpss", ("tpss_x",), "MGGA", 0.0, (("mgga_x_tpss", 1.0),)),
    NamedFunctionalSpec("mgga_c_tpss", ("tpss_c",), "MGGA", 0.0, (("mgga_c_tpss", 1.0),)),
    # Composite & Hybrid functionals
    NamedFunctionalSpec(
        "svwn",
        ("lda", "svwn5", "lda,vwn", "lda,vwn5"),
        "LDA",
        0.0,
        (("lda_x", 1.0), ("lda_c_vwn5", 1.0)),
    ),
    NamedFunctionalSpec(
        "svwn_rpa",
        ("lda,vwn_rpa",),
        "LDA",
        0.0,
        (("lda_x", 1.0), ("lda_c_vwn_rpa", 1.0)),
    ),
    NamedFunctionalSpec(
        "spz81",
        ("lda,pz", "lda,pz81"),
        "LDA",
        0.0,
        (("lda_x", 1.0), ("lda_c_pz", 1.0)),
    ),
    NamedFunctionalSpec(
        "pbe",
        ("pbe,pbe",),
        "GGA",
        0.0,
        (("gga_x_pbe", 1.0), ("gga_c_pbe", 1.0)),
    ),
    NamedFunctionalSpec(
        "blyp",
        ("b88,lyp",),
        "GGA",
        0.0,
        (("gga_x_b88", 1.0), ("gga_c_lyp", 1.0)),
    ),
    NamedFunctionalSpec(
        "b3lyp",
        ("b3lypg",),
        "GGA",
        0.20,
        (("lda_x", 0.08), ("gga_x_b88", 0.72), ("gga_c_lyp", 0.81), ("lda_c_vwn_rpa", 0.19)),
    ),
    NamedFunctionalSpec(
        "b3lyp5",
        (),
        "GGA",
        0.20,
        (("lda_x", 0.08), ("gga_x_b88", 0.72), ("gga_c_lyp", 0.81), ("lda_c_vwn5", 0.19)),
    ),
    NamedFunctionalSpec(
        "pbe0",
        ("pbeh",),
        "GGA",
        0.25,
        (("gga_x_pbe", 0.75), ("gga_c_pbe", 1.0)),
    ),
    NamedFunctionalSpec(
        "tpss",
        ("tpss,tpss",),
        "MGGA",
        0.0,
        (("mgga_x_tpss", 1.0), ("mgga_c_tpss", 1.0)),
    ),
    NamedFunctionalSpec(
        "hf",
        ("rhf",),
        "LDA",
        1.0,
        (),
    ),
)


def differentiate_functional(spec: BaseFunctionalSpec) -> dict[str, Expr]:
    """Compute zk, 1st derivatives (vrho, vsigma, vtau), and 2nd derivatives."""
    f = spec.f_expr
    vrho = diff(f, "rho")
    vsigma = diff(f, "sigma")
    vtau = diff(f, "tau")

    v2rho2 = diff(vrho, "rho")
    v2rhosigma = diff(vrho, "sigma")
    v2sigma2 = diff(vsigma, "sigma")
    v2rhotau = diff(vrho, "tau")
    v2sigmatau = diff(vsigma, "tau")
    v2tau2 = diff(vtau, "tau")

    return {
        "zk": spec.zk_expr,
        "vrho": vrho,
        "vsigma": vsigma,
        "vtau": vtau,
        "v2rho2": v2rho2,
        "v2rhosigma": v2rhosigma,
        "v2sigma2": v2sigma2,
        "v2rhotau": v2rhotau,
        "v2sigmatau": v2sigmatau,
        "v2tau2": v2tau2,
    }
