"""Authoritative S22 weak-interaction dataset loader wrapping ase.data.s22 + S22A/S22B refs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# Sign convention: interaction energy E_int = E_dimer - E_monomerA - E_monomerB (kcal/mol, negative = bound).
# S22A: Takatani et al., J. Chem. Phys. 2010, 132, 144104 (CCSD(T)/CBS).
S22_A_REFS_KCAL: dict[str, float] = {
    "Ammonia_dimer": -3.17,
    "Water_dimer": -5.02,
    "Formic_acid_dimer": -18.80,
    "Formamide_dimer": -16.12,
    "Uracil_dimer_h-bonded": -20.69,
    "2-pyridoxine_2-aminopyridine_complex": -17.00,
    "Adenine-thymine_Watson-Crick_complex": -16.74,
    "Methane_dimer": -0.53,
    "Ethene_dimer": -1.50,
    "Benzene-methane_complex": -1.45,
    "Benzene_dimer_parallel_displaced": -2.62,
    "Pyrazine_dimer": -4.20,
    "Uracil_dimer_stack": -9.74,
    "Indole-benzene_complex_stack": -4.59,
    "Adenine-thymine_complex_stack": -11.66,
    "Ethene-ethyne_complex": -1.51,
    "Benzene-water_complex": -3.29,
    "Benzene-ammonia_complex": -2.32,
    "Benzene-HCN_complex": -4.55,
    "Benzene_dimer_T-shaped": -2.71,
    "Indole-benzene_T-shape_complex": -5.62,
    "Phenol_dimer": -7.09,
}

# S22B: Marshall, Burns, Sherrill, J. Chem. Phys. 2011, 135, 194102 (also GMTKN55 S22/.res).
S22_B_REFS_KCAL: dict[str, float] = {
    "Ammonia_dimer": -3.133,
    "Water_dimer": -4.989,
    "Formic_acid_dimer": -18.753,
    "Formamide_dimer": -16.062,
    "Uracil_dimer_h-bonded": -20.641,
    "2-pyridoxine_2-aminopyridine_complex": -16.934,
    "Adenine-thymine_Watson-Crick_complex": -16.660,
    "Methane_dimer": -0.527,
    "Ethene_dimer": -1.472,
    "Benzene-methane_complex": -1.448,
    "Benzene_dimer_parallel_displaced": -2.654,
    "Pyrazine_dimer": -4.255,
    "Uracil_dimer_stack": -9.805,
    "Indole-benzene_complex_stack": -4.524,
    "Adenine-thymine_complex_stack": -11.730,
    "Ethene-ethyne_complex": -1.496,
    "Benzene-water_complex": -3.275,
    "Benzene-ammonia_complex": -2.312,
    "Benzene-HCN_complex": -4.541,
    "Benzene_dimer_T-shaped": -2.717,
    "Indole-benzene_T-shape_complex": -5.627,
    "Phenol_dimer": -7.097,
}

S22_CATEGORIES: dict[str, str] = {
    "Ammonia_dimer": "hydrogen_bonded",
    "Water_dimer": "hydrogen_bonded",
    "Formic_acid_dimer": "hydrogen_bonded",
    "Formamide_dimer": "hydrogen_bonded",
    "Uracil_dimer_h-bonded": "hydrogen_bonded",
    "2-pyridoxine_2-aminopyridine_complex": "hydrogen_bonded",
    "Adenine-thymine_Watson-Crick_complex": "hydrogen_bonded",
    "Methane_dimer": "dispersion_dominated",
    "Ethene_dimer": "dispersion_dominated",
    "Benzene-methane_complex": "dispersion_dominated",
    "Benzene_dimer_parallel_displaced": "dispersion_dominated",
    "Pyrazine_dimer": "dispersion_dominated",
    "Uracil_dimer_stack": "dispersion_dominated",
    "Indole-benzene_complex_stack": "dispersion_dominated",
    "Adenine-thymine_complex_stack": "dispersion_dominated",
    "Ethene-ethyne_complex": "mixed",
    "Benzene-water_complex": "mixed",
    "Benzene-ammonia_complex": "mixed",
    "Benzene-HCN_complex": "mixed",
    "Benzene_dimer_T-shaped": "mixed",
    "Indole-benzene_T-shape_complex": "mixed",
    "Phenol_dimer": "mixed",
}


@dataclass(frozen=True)
class S22Complex:
    """Single S22 dimer complex with monomer splits and CCSD(T)/CBS references."""

    index: int
    name: str
    category: str
    dimer: Any
    monomer_a: Any
    monomer_b: Any
    n_atoms_a: int
    n_atoms_b: int
    ref_ase_kcal: float
    ref_s22a_kcal: float
    ref_s22b_kcal: float


def _min_distance(atoms: Any) -> float:
    pos = np.asarray(atoms.get_positions(), dtype=np.float64)
    if len(pos) < 2:
        return float("inf")
    diffs = pos[:, None, :] - pos[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(len(pos), k=1)
    return float(np.min(dists[iu]))


def load_s22_complexes() -> list[S22Complex]:
    """Load and validate all 22 S22 complexes from ase.collections.s22 & ase.data.s22."""
    import ase.collections
    import ase.data.s22 as ase_s22
    import ase.units

    ev_to_kcal = 1.0 / (ase.units.kcal / ase.units.mol)
    names = list(ase_s22.s22)
    if len(names) != 22:
        raise RuntimeError(f"Expected 22 S22 complexes from ase.data.s22, got {len(names)}")

    complexes: list[S22Complex] = []
    for idx, name in enumerate(names, start=1):
        dimer = ase.collections.s22[name].copy()
        n_a, n_b = [int(x) for x in ase_s22.get_number_of_dimer_atoms(name)]
        if len(dimer) != n_a + n_b:
            raise ValueError(f"S22 {name}: total atoms {len(dimer)} != {n_a} + {n_b}")
        rmin = _min_distance(dimer)
        if rmin <= 0.7:
            raise ValueError(f"S22 {name}: collapsed geometry R_min={rmin:.4f} A <= 0.7 A")

        mon_a = dimer[:n_a].copy()
        mon_b = dimer[n_a:].copy()

        ref_ase_kcal = float(ase_s22.get_interaction_energy_s22(name)) * ev_to_kcal
        ref_s22a = S22_A_REFS_KCAL[name]
        ref_s22b = S22_B_REFS_KCAL[name]

        # Verify ase.data.s22 matches published S22A Takatani 2010 within rounding tolerance
        if abs(ref_ase_kcal - ref_s22a) > 0.01:
            raise ValueError(
                f"S22 {name}: ASE ref {ref_ase_kcal:.4f} kcal/mol deviates from S22A {ref_s22a:.4f}"
            )

        complexes.append(
            S22Complex(
                index=idx,
                name=name,
                category=S22_CATEGORIES[name],
                dimer=dimer,
                monomer_a=mon_a,
                monomer_b=mon_b,
                n_atoms_a=n_a,
                n_atoms_b=n_b,
                ref_ase_kcal=ref_ase_kcal,
                ref_s22a_kcal=ref_s22a,
                ref_s22b_kcal=ref_s22b,
            )
        )
    return complexes


def get_s22_complex(name: str) -> S22Complex:
    """Return a single validated S22Complex by exact name."""
    for item in load_s22_complexes():
        if item.name == name:
            return item
    raise KeyError(f"Unknown S22 complex {name!r}")
