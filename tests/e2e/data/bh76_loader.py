"""Authoritative GMTKN55 BH76 open-shell barrier-height dataset loader with conservation guards."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

BH76_JSON_PATH = Path(__file__).resolve().parent / "bh76_openshell.json"

_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
}


@dataclass(frozen=True)
class BH76Species:
    """Validated BH76 molecular/atomic species from GMTKN55."""

    name: str
    charge: int
    spin: int  # 2S = n_alpha - n_beta (GMTKN55 .UHF)
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def pyscf_atom(self) -> list[tuple[str, tuple[float, float, float]]]:
        return list(self.atoms)

    @property
    def coords_angstrom(self) -> np.ndarray:
        return np.array([xyz for _, xyz in self.atoms], dtype=np.float64)

    def to_pyscf_mol(self, basis: str = "def2-svp", unit: str = "Angstrom") -> Any:
        from pyscf import gto

        return gto.M(
            atom=self.pyscf_atom,
            basis=basis,
            charge=self.charge,
            spin=self.spin,
            unit=unit,
            cart=False,
            verbose=0,
        )


@dataclass(frozen=True)
class BH76Reaction:
    """Single BH76 open-shell barrier height reaction (E_barrier = E_TS - sum E_reactants)."""

    id: str
    ts_name: str
    reactant_names: tuple[str, ...]
    species_names: tuple[str, ...]
    coeffs: tuple[int, ...]
    ref_barrier_kcal: float
    direction: str
    tags: tuple[str, ...]


def _validate_species(sp: BH76Species) -> None:
    if sp.n_atoms == 0:
        raise ValueError(f"BH76 species {sp.name!r} has no atoms")
    z_sum = sum(_ATOMIC_NUMBERS[sym] for sym, _ in sp.atoms)
    n_elec = z_sum - sp.charge
    if n_elec <= 0:
        raise ValueError(f"BH76 species {sp.name!r} has non-positive electron count {n_elec}")
    if sp.spin < 0 or sp.spin > n_elec:
        raise ValueError(f"BH76 species {sp.name!r} has invalid spin {sp.spin} for {n_elec} electrons")
    if (n_elec - sp.spin) % 2 != 0:
        raise ValueError(
            f"BH76 species {sp.name!r} fails electron-spin parity: Z={z_sum}, q={sp.charge}, spin={sp.spin}"
        )
    if sp.n_atoms >= 2:
        coords = sp.coords_angstrom
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(sp.n_atoms, k=1)
        rmin = float(np.min(dists[iu]))
        if rmin <= 0.6:
            raise ValueError(f"BH76 species {sp.name!r} has collapsed contact R_min={rmin:.4f} A <= 0.6 A")


def _validate_reaction(rxn: BH76Reaction, species: dict[str, BH76Species]) -> None:
    if len(rxn.species_names) != len(rxn.coeffs):
        raise ValueError(f"BH76 reaction {rxn.id!r}: species/coeffs length mismatch")
    if not any(species[name].spin > 0 for name in rxn.species_names):
        raise ValueError(f"BH76 reaction {rxn.id!r} has no open-shell species (spin > 0)")

    atom_balance: collections.Counter[str] = collections.Counter()
    charge_balance = 0
    for name, coeff in zip(rxn.species_names, rxn.coeffs):
        sp = species[name]
        for sym, _ in sp.atoms:
            atom_balance[sym] += coeff
        charge_balance += coeff * sp.charge

    bad_atoms = {sym: cnt for sym, cnt in atom_balance.items() if cnt != 0}
    if bad_atoms:
        raise ValueError(f"BH76 reaction {rxn.id!r} violates atom conservation: {bad_atoms}")
    if charge_balance != 0:
        raise ValueError(f"BH76 reaction {rxn.id!r} violates charge conservation: delta_q={charge_balance}")


_CACHE: tuple[dict[str, BH76Species], list[BH76Reaction], dict[str, Any]] | None = None


def load_bh76_openshell_data() -> tuple[dict[str, BH76Species], list[BH76Reaction], dict[str, Any]]:
    """Load and validate GMTKN55 BH76 open-shell species and reactions."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    raw = json.loads(BH76_JSON_PATH.read_text(encoding="utf-8"))
    species: dict[str, BH76Species] = {}
    for name, sdata in raw["species"].items():
        atoms = tuple((str(a[0]), (float(a[1]), float(a[2]), float(a[3]))) for a in sdata["atoms"])
        sp = BH76Species(
            name=name,
            charge=int(sdata["charge"]),
            spin=int(sdata["spin"]),
            atoms=atoms,
        )
        _validate_species(sp)
        species[name] = sp

    reactions: list[BH76Reaction] = []
    for rdata in raw["reactions"]:
        rxn = BH76Reaction(
            id=str(rdata["id"]),
            ts_name=str(rdata["ts"]),
            reactant_names=tuple(str(x) for x in rdata["reactants"]),
            species_names=tuple(str(x) for x in rdata["species"]),
            coeffs=tuple(int(c) for c in rdata["coeffs"]),
            ref_barrier_kcal=float(rdata["ref_barrier_kcal"]),
            direction=str(rdata["direction"]),
            tags=tuple(str(t) for t in rdata["tags"]),
        )
        _validate_reaction(rxn, species)
        reactions.append(rxn)

    _CACHE = (species, reactions, raw.get("metadata", {}))
    return _CACHE


def get_bh76_species() -> dict[str, BH76Species]:
    """Return validated dict of all BH76 open-shell species."""
    species, _, _ = load_bh76_openshell_data()
    return dict(species)


def get_bh76_openshell_reactions(subset: str = "full_openshell") -> list[BH76Reaction]:
    """Return validated BH76 open-shell reactions filtered by tag ('core' or 'full_openshell')."""
    _, reactions, _ = load_bh76_openshell_data()
    tag = subset.strip().lower()
    if tag in ("all", "full", "full_openshell"):
        return [r for r in reactions if "full_openshell" in r.tags]
    if tag == "core":
        return [r for r in reactions if "core" in r.tags]
    raise ValueError(f"Unknown BH76 subset {subset!r}; choose 'core' or 'full_openshell'")
