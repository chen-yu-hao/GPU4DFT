"""Exact real solid-harmonic transformations matching PySCF/libcint.

Conventions (see ``include/gpu4dft/basis.hpp`` and ``src/shell_pairs.cpp``):

* Cartesian monomials of shell angular momentum ``l`` are ordered with ``lx``
  descending from ``l`` to ``0``, ``ly`` descending from ``l - lx`` to ``0``,
  and ``lz = l - lx - ly`` (``(l + 1) * (l + 2) // 2`` components).
* Spherical AOs of shell angular momentum ``l`` follow PySCF's ``mol.cart=False``
  ordering (``2 * l + 1`` components):
    - ``l == 0``: ``[0]`` (``s``)
    - ``l == 1``: ``[+1, -1, 0]`` (``px, py, pz``)
    - ``l >= 2``: ``[-l, -l + 1, ..., 0, ..., +l]``
* Because ``ShellPairData::prefactor`` already folds in ``Y_00^2 = 1 / (4 pi)``
  per shell pair, Cartesian-to-spherical transforms on integrals multiplied by
  ``prefactor`` use the *reduced* solid harmonics
  ``S_bar_{l,m}(r) = sqrt(4 pi) S_{l,m}(r)`` (so ``S_bar_{0,0} == 1``).
"""

from __future__ import annotations

import math
from fractions import Fraction

MAX_HARMONIC_L = 6


def n_cart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def n_sph(l: int) -> int:
    return 2 * l + 1


def cart_tuples(l: int) -> list[tuple[int, int, int]]:
    """Canonical Cartesian exponent triples (lx, ly, lz) for angular momentum l."""
    out: list[tuple[int, int, int]] = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            out.append((lx, ly, l - lx - ly))
    return out


def pyscf_m_order(l: int) -> list[int]:
    """PySCF/libcint spherical m ordering for angular momentum l."""
    if l < 0:
        raise ValueError(f"angular momentum must be non-negative, got {l}")
    if l == 0:
        return [0]
    if l == 1:
        return [1, -1, 0]  # px, py, pz
    return list(range(-l, l + 1))


def _exact_reduced_poly(l: int, m: int) -> dict[tuple[int, int, int], tuple[int, Fraction]]:
    """Return {(lx, ly, lz): (sign, rad_fraction)} where coeff = sign * sqrt(rad_fraction)."""
    a = abs(m)
    xy_terms: dict[tuple[int, int], Fraction] = {}
    if m == 0:
        xy_terms[(0, 0)] = Fraction(1)
    elif m > 0:
        for p in range(0, a + 1, 2):
            sign = -1 if (p // 2) % 2 == 1 else 1
            xy_terms[(a - p, p)] = Fraction(sign * math.comb(a, p))
    else:
        for p in range(1, a + 1, 2):
            sign = -1 if ((p - 1) // 2) % 2 == 1 else 1
            xy_terms[(a - p, p)] = Fraction(sign * math.comb(a, p))

    z_terms: dict[tuple[int, int, int], Fraction] = {}
    for k in range((l - a) // 2 + 1):
        sign = -1 if (k % 2 == 1) else 1
        pref = Fraction(
            sign * math.factorial(2 * l - 2 * k),
            (2**l) * math.factorial(k) * math.factorial(l - k) * math.factorial(l - a - 2 * k),
        )
        z_pow = l - a - 2 * k
        for u in range(k + 1):
            for v in range(k - u + 1):
                w = k - u - v
                multinom = Fraction(
                    math.factorial(k),
                    math.factorial(u) * math.factorial(v) * math.factorial(w),
                )
                key = (2 * u, 2 * v, 2 * w + z_pow)
                z_terms[key] = z_terms.get(key, Fraction(0)) + pref * multinom

    poly: dict[tuple[int, int, int], Fraction] = {}
    for (ax, ay), c_xy in xy_terms.items():
        for (zx, zy, zz), c_z in z_terms.items():
            key = (ax + zx, ay + zy, zz)
            poly[key] = poly.get(key, Fraction(0)) + c_xy * c_z

    norm2 = Fraction(
        (2 * l + 1) * (1 if m == 0 else 2) * math.factorial(l - a),
        math.factorial(l + a),
    )
    out: dict[tuple[int, int, int], tuple[int, Fraction]] = {}
    for key, q in poly.items():
        if q == 0:
            continue
        sgn = 1 if q > 0 else -1
        out[key] = (sgn, q * q * norm2)
    return out


def reduced_sph_matrix(l: int) -> list[list[float]]:
    """Return [2*l+1][n_cart(l)] reduced solid-harmonic matrix S_bar = sqrt(4*pi) * S."""
    carts = cart_tuples(l)
    cart_idx = {t: i for i, t in enumerate(carts)}
    mat: list[list[float]] = []
    for m in pyscf_m_order(l):
        row = [0.0] * len(carts)
        for key, (sgn, rad) in _exact_reduced_poly(l, m).items():
            row[cart_idx[key]] = sgn * math.sqrt(float(rad.numerator) / float(rad.denominator))
        mat.append(row)
    return mat


def reduced_sph_sparse(l: int) -> list[list[tuple[int, float]]]:
    """Return sparse [(cart_index, coeff), ...] rows for each spherical m of l."""
    mat = reduced_sph_matrix(l)
    out: list[list[tuple[int, float]]] = []
    for row in mat:
        out.append([(ic, val) for ic, val in enumerate(row) if val != 0.0])
    return out


def full_sph_matrix(l: int) -> list[list[float]]:
    """Return [2*l+1][n_cart(l)] unit-sphere-normalized solid-harmonic matrix."""
    y00 = 1.0 / math.sqrt(4.0 * math.pi)
    return [[val * y00 for val in row] for row in reduced_sph_matrix(l)]
