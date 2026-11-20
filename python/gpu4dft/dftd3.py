"""Grimme DFT-D3(BJ) empirical dispersion energy and analytical nuclear gradients."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import _core
from ._core import D3BJParams, D3BJResult

_SYMBOLS_1_TO_103 = (
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
)
_SYMBOL_TO_Z: dict[str, int] = {sym.lower(): idx + 1 for idx, sym in enumerate(_SYMBOLS_1_TO_103)}


def _parse_atoms_and_coords(
    atomic_numbers_or_symbols: Sequence[int | str] | np.ndarray | Any,
    coords_bohr: np.ndarray | Sequence[Sequence[float]] | None,
) -> tuple[list[int], np.ndarray]:
    if coords_bohr is None and hasattr(atomic_numbers_or_symbols, "atom_charges") and hasattr(
        atomic_numbers_or_symbols, "atom_coords"
    ):
        mol = atomic_numbers_or_symbols
        z_list = [int(z) for z in mol.atom_charges()]
        coords = np.ascontiguousarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
        return z_list, coords

    if coords_bohr is None:
        raise ValueError("coords_bohr must be provided when first argument is not a Mole object")

    z_list: list[int] = []
    for item in atomic_numbers_or_symbols:
        if isinstance(item, (int, np.integer)):
            z_list.append(int(item))
        elif isinstance(item, str):
            key = item.strip().lower()
            if key.isdigit():
                z_list.append(int(key))
            elif key in _SYMBOL_TO_Z:
                z_list.append(_SYMBOL_TO_Z[key])
            else:
                raise ValueError(f"Unknown element symbol {item!r} for DFT-D3(BJ)")
        elif hasattr(item, "atomic_number"):
            z_list.append(int(item.atomic_number))
        else:
            raise TypeError(f"Unsupported atom entry type: {type(item).__name__}")

    coords = np.ascontiguousarray(coords_bohr, dtype=np.float64)
    if coords.ndim != 2 or coords.shape != (len(z_list), 3):
        raise ValueError(
            f"coords_bohr must have shape ({len(z_list)}, 3), got {coords.shape}"
        )
    return z_list, coords


def dftd3_params(xc: str = "b3lyp") -> D3BJParams:
    """Return Grimme DFT-D3(BJ) damping parameters (s6, a1, s8, a2) for ``xc``."""
    return _core.dftd3_params(str(xc))


def supported_dftd3_functionals() -> list[str]:
    """Return list of supported functional keys for DFT-D3(BJ)."""
    return list(_core.supported_dftd3_functionals())


def dftd3_energy_grad(
    atomic_numbers_or_symbols: Sequence[int | str] | np.ndarray | Any,
    coords_bohr: np.ndarray | Sequence[Sequence[float]] | None = None,
    xc: str = "b3lyp",
) -> tuple[float, np.ndarray]:
    """Compute DFT-D3(BJ) dispersion energy (Hartree) and nuclear gradient (Hartree/Bohr)."""
    z_list, coords = _parse_atoms_and_coords(atomic_numbers_or_symbols, coords_bohr)
    res = _core.compute_d3bj(z_list, coords, str(xc), True)
    return float(res.energy), np.asarray(res.gradient, dtype=np.float64)


def dftd3_energy(
    atomic_numbers_or_symbols: Sequence[int | str] | np.ndarray | Any,
    coords_bohr: np.ndarray | Sequence[Sequence[float]] | None = None,
    xc: str = "b3lyp",
) -> float:
    """Compute DFT-D3(BJ) dispersion energy (Hartree)."""
    z_list, coords = _parse_atoms_and_coords(atomic_numbers_or_symbols, coords_bohr)
    res = _core.compute_d3bj(z_list, coords, str(xc), False)
    return float(res.energy)


__all__ = [
    "D3BJParams",
    "D3BJResult",
    "dftd3_energy",
    "dftd3_energy_grad",
    "dftd3_params",
    "supported_dftd3_functionals",
]
