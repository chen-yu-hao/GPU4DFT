"""Conversion helpers for PySCF ``Mole`` objects. PySCF is imported lazily."""

from __future__ import annotations

import numpy as np

from ._core import Atom, Basis, Shell


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


def atoms_from_pyscf(mol) -> list[Atom]:
    """Extract :class:`Atom` list (atomic numbers and Bohr coordinates) from a ``pyscf.gto.Mole``."""
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    return [
        Atom(int(mol.atom_charge(i)), [float(coords_bohr[i, 0]), float(coords_bohr[i, 1]), float(coords_bohr[i, 2])])
        for i in range(mol.natm)
    ]


def aux_basis_from_pyscf(mol, auxbasis: str | dict = "def2-universal-jkfit") -> Basis:
    """Build an auxiliary density-fitting :class:`Basis` for ``mol`` using ``pyscf.df``."""
    from pyscf import df

    auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    return basis_from_pyscf(auxmol)


def minao_init_guess(mol) -> np.ndarray:
    """Compute PySCF's atomic ``minao`` initial density guess as a contiguous float64 array."""
    from pyscf.scf import hf

    dm0 = hf.init_guess_by_minao(mol)
    return np.ascontiguousarray(dm0, dtype=np.float64)
