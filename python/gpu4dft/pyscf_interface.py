"""Conversion from PySCF objects. PySCF is only needed when these are called."""

from __future__ import annotations

from ._core import Basis, Shell


def basis_from_pyscf(mol) -> Basis:
    """Build a :class:`Basis` from a ``pyscf.gto.Mole``.

    Generally contracted PySCF shells are split into one shell per contraction,
    which preserves PySCF's spherical AO ordering.
    """
    if mol.cart and any(mol.bas_angular(ib) >= 2 for ib in range(mol.nbas)):
        raise ValueError("Cartesian d and higher functions are not supported; use mol.cart = False")
    shells = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        center = [float(x) for x in mol.bas_coord(ib)]
        exponents = [float(x) for x in mol.bas_exp(ib)]
        coefficients = mol.bas_ctr_coeff(ib)  # (nprim, nctr), for normalised primitives
        for ic in range(coefficients.shape[1]):
            shells.append(Shell(l, center, exponents, [float(c) for c in coefficients[:, ic]]))
    return Basis(shells)
