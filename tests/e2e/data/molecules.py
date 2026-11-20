"""Authoritative molecular geometry builders and large-molecule loaders for GPU4DFT E2E tests."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

GEOMETRIES_DIR = Path(__file__).resolve().parent / "geometries"

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
class MoleculeSpecData:
    """Validated molecular geometry specification in Angstrom."""

    name: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    charge: int = 0
    spin: int = 0
    formula: str = ""
    source: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def symbols(self) -> list[str]:
        return [sym for sym, _ in self.atoms]

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.array([_ATOMIC_NUMBERS[sym] for sym, _ in self.atoms], dtype=np.int32)

    @property
    def coords_angstrom(self) -> np.ndarray:
        return np.array([xyz for _, xyz in self.atoms], dtype=np.float64)

    @property
    def coords_bohr(self) -> np.ndarray:
        return self.coords_angstrom / 0.529177210903

    @property
    def pyscf_atom(self) -> list[tuple[str, tuple[float, float, float]]]:
        return list(self.atoms)

    @property
    def atom_string(self) -> str:
        return "; ".join(f"{sym} {x:.8f} {y:.8f} {z:.8f}" for sym, (x, y, z) in self.atoms)

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

    def min_distance_angstrom(self) -> float:
        coords = self.coords_angstrom
        if len(coords) < 2:
            return float("inf")
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.triu_indices(len(coords), k=1)
        return float(np.min(dists[iu]))


def _hill_formula(atoms: list[tuple[str, tuple[float, float, float]]]) -> str:
    counts = collections.Counter(sym for sym, _ in atoms)
    parts: list[str] = []
    if "C" in counts:
        n = counts.pop("C")
        parts.append("C" if n == 1 else f"C{n}")
        if "H" in counts:
            nh = counts.pop("H")
            parts.append("H" if nh == 1 else f"H{nh}")
    for sym in sorted(counts):
        n = counts[sym]
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts)


def _validate_molecule(spec: MoleculeSpecData, min_r_ang: float = 0.65) -> MoleculeSpecData:
    if spec.n_atoms == 0:
        raise ValueError(f"Molecule {spec.name!r} has zero atoms")
    z_sum = int(np.sum(spec.atomic_numbers))
    n_elec = z_sum - spec.charge
    if n_elec <= 0:
        raise ValueError(f"Molecule {spec.name!r} has non-positive electron count {n_elec}")
    if spec.spin < 0 or spec.spin > n_elec:
        raise ValueError(f"Molecule {spec.name!r} has invalid spin {spec.spin} for {n_elec} electrons")
    if (n_elec - spec.spin) % 2 != 0:
        raise ValueError(
            f"Molecule {spec.name!r} violates electron-spin parity: n_elec={n_elec}, spin={spec.spin}"
        )
    if spec.formula:
        actual = _hill_formula(list(spec.atoms))
        if actual != spec.formula:
            raise ValueError(
                f"Molecule {spec.name!r} formula mismatch: expected {spec.formula}, got {actual}"
            )
    if spec.n_atoms >= 2:
        rmin = spec.min_distance_angstrom()
        if rmin < min_r_ang:
            raise ValueError(
                f"Molecule {spec.name!r} has steric clash R_min={rmin:.4f} A < {min_r_ang:.4f} A"
            )
    return spec


def parse_xyz_file(
    xyz_path: Path,
    *,
    name: str | None = None,
    charge: int = 0,
    spin: int = 0,
    expected_formula: str = "",
    expected_atoms: int | None = None,
    min_r_ang: float = 0.85,
) -> MoleculeSpecData:
    """Load and validate an .xyz geometry file in Angstrom."""
    lines = xyz_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Invalid XYZ file {xyz_path}: fewer than 3 lines")
    header_n = int(lines[0].strip())
    comment = lines[1].strip()
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for raw in lines[2:]:
        parts = raw.split()
        if not parts:
            continue
        sym = parts[0].capitalize()
        if sym not in _ATOMIC_NUMBERS:
            raise ValueError(f"Unsupported element {sym!r} in {xyz_path}")
        atoms.append((sym, (float(parts[1]), float(parts[2]), float(parts[3]))))
    if len(atoms) != header_n:
        raise ValueError(f"{xyz_path}: header atom count {header_n} != parsed {len(atoms)}")
    if expected_atoms is not None and len(atoms) != expected_atoms:
        raise ValueError(f"{xyz_path}: expected {expected_atoms} atoms, got {len(atoms)}")
    spec = MoleculeSpecData(
        name=name or xyz_path.stem,
        atoms=tuple(atoms),
        charge=charge,
        spin=spin,
        formula=expected_formula or _hill_formula(atoms),
        source=comment or str(xyz_path),
    )
    return _validate_molecule(spec, min_r_ang=min_r_ang)


def get_water_cluster_coords(nx: int, ny: int, nz: int, spacing: float = 2.90) -> list[tuple[str, tuple[float, float, float]]]:
    """Deterministic 3D hydrogen-bonded (H2O)_N cluster builder (O-H=0.96 A, HOH=104.52 deg)."""
    r_oh = 0.96
    half_ang = math.radians(104.52 / 2.0)
    h1_local = np.array([0.0, r_oh * math.sin(half_ang), r_oh * math.cos(half_ang)], dtype=np.float64)
    h2_local = np.array([0.0, -r_oh * math.sin(half_ang), r_oh * math.cos(half_ang)], dtype=np.float64)

    atoms: list[tuple[str, tuple[float, float, float]]] = []
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                origin = np.array(
                    [
                        ix * spacing + 0.12 * ((iy + iz) % 2),
                        iy * spacing + 0.10 * ((ix + iz) % 2),
                        iz * spacing + 0.08 * ((ix + iy) % 2),
                    ],
                    dtype=np.float64,
                )
                theta = 0.65 * idx
                phi = 1.15 * idx
                ct, st = math.cos(theta), math.sin(theta)
                cp_a, sp_a = math.cos(phi), math.sin(phi)
                rot = np.array(
                    [
                        [ct * cp_a, -sp_a, st * cp_a],
                        [ct * sp_a, cp_a, st * sp_a],
                        [-st, 0.0, ct],
                    ],
                    dtype=np.float64,
                )
                p_o = origin
                p_h1 = origin + rot @ h1_local
                p_h2 = origin + rot @ h2_local
                atoms.append(("O", (float(p_o[0]), float(p_o[1]), float(p_o[2]))))
                atoms.append(("H", (float(p_h1[0]), float(p_h1[1]), float(p_h1[2]))))
                atoms.append(("H", (float(p_h2[0]), float(p_h2[1]), float(p_h2[2]))))
                idx += 1
    return atoms


def _build_c60_atoms(bond_length_ang: float = 1.425) -> list[tuple[str, tuple[float, float, float]]]:
    """Deterministic I_h truncated icosahedron C60 coordinates."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    pts: set[tuple[float, float, float]] = set()

    def _add_even_perms(base: tuple[float, float, float]) -> None:
        for p in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            pts.add((round(base[p[0]], 12), round(base[p[1]], 12), round(base[p[2]], 12)))

    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            _add_even_perms((0.0, s1 * 1.0, s2 * 3.0 * phi))
            for s3 in (-1.0, 1.0):
                _add_even_perms((s1 * 1.0, s2 * (2.0 + phi), s3 * 2.0 * phi))
                _add_even_perms((s1 * phi, s2 * 2.0, s3 * (2.0 * phi + 1.0)))

    scale = bond_length_ang / 2.0
    return [("C", (x * scale, y * scale, z * scale)) for x, y, z in sorted(pts)]


def _build_benzene_atoms() -> list[tuple[str, tuple[float, float, float]]]:
    """Planar C6H6 in the xy-plane (z=0)."""
    r_cc = 1.397
    r_ch = 1.087
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for k in range(6):
        ang = k * math.pi / 3.0
        atoms.append(("C", (r_cc * math.cos(ang), r_cc * math.sin(ang), 0.0)))
        atoms.append(("H", ((r_cc + r_ch) * math.cos(ang), (r_cc + r_ch) * math.sin(ang), 0.0)))
    return atoms


def _build_small_medium_catalog() -> dict[str, MoleculeSpecData]:
    raw: dict[str, MoleculeSpecData] = {
        "h2o": MoleculeSpecData(
            name="h2o",
            atoms=(
                ("O", (0.0000, 0.0000, 0.1173)),
                ("H", (0.0000, 0.7572, -0.4692)),
                ("H", (0.0000, -0.7572, -0.4692)),
            ),
            formula="H2O",
            source="standard_small_test",
        ),
        "ch2o": MoleculeSpecData(
            name="ch2o",
            atoms=(
                ("C", (0.0000, 0.0000, -0.5288)),
                ("O", (0.0000, 0.0000, 0.6780)),
                ("H", (0.0000, 0.9355, -1.1160)),
                ("H", (0.0000, -0.9355, -1.1160)),
            ),
            formula="CH2O",
            source="standard_small_test",
        ),
        "ch4": MoleculeSpecData(
            name="ch4",
            atoms=(
                ("C", (0.0000, 0.0000, 0.0000)),
                ("H", (0.6291, 0.6291, 0.6291)),
                ("H", (-0.6291, -0.6291, 0.6291)),
                ("H", (-0.6291, 0.6291, -0.6291)),
                ("H", (0.6291, -0.6291, -0.6291)),
            ),
            formula="CH4",
            source="standard_small_test",
        ),
        "co2": MoleculeSpecData(
            name="co2",
            atoms=(
                ("C", (0.0000, 0.0000, 0.0000)),
                ("O", (0.0000, 0.0000, 1.1620)),
                ("O", (0.0000, 0.0000, -1.1620)),
            ),
            formula="CO2",
            source="linear_z_axis",
        ),
        "c6h6": MoleculeSpecData(
            name="c6h6",
            atoms=tuple(_build_benzene_atoms()),
            formula="C6H6",
            source="planar_xy_plane",
        ),
        "h2o_dimer": MoleculeSpecData(
            name="h2o_dimer",
            atoms=tuple(get_water_cluster_coords(2, 1, 1)),
            formula="H4O2",
            source="water_cluster_2x1x1",
        ),
        "oh_radical": MoleculeSpecData(
            name="oh_radical",
            atoms=(
                ("O", (0.0000, 0.0000, 0.0000)),
                ("H", (0.0000, 0.0000, 0.9700)),
            ),
            charge=0,
            spin=1,
            formula="HO",
            source="standard_radical",
        ),
        "ch3_radical": MoleculeSpecData(
            name="ch3_radical",
            atoms=(
                ("C", (0.0000, 0.0000, 0.0000)),
                ("H", (1.0790, 0.0000, 0.0000)),
                ("H", (-0.5395, 0.9344, 0.0000)),
                ("H", (-0.5395, -0.9344, 0.0000)),
            ),
            charge=0,
            spin=1,
            formula="CH3",
            source="standard_radical",
        ),
        "ho2_radical": MoleculeSpecData(
            name="ho2_radical",
            atoms=(
                ("O", (0.0000, 0.0000, 0.0000)),
                ("O", (1.3300, 0.0000, 0.0000)),
                ("H", (-0.2300, 0.9450, 0.0000)),
            ),
            charge=0,
            spin=1,
            formula="HO2",
            source="standard_radical",
        ),
        "o2_triplet": MoleculeSpecData(
            name="o2_triplet",
            atoms=(
                ("O", (0.0000, 0.0000, -0.6040)),
                ("O", (0.0000, 0.0000, 0.6040)),
            ),
            charge=0,
            spin=2,
            formula="O2",
            source="standard_triplet",
        ),
        "no_radical": MoleculeSpecData(
            name="no_radical",
            atoms=(
                ("N", (0.0000, 0.0000, -0.5750)),
                ("O", (0.0000, 0.0000, 0.5750)),
            ),
            charge=0,
            spin=1,
            formula="NO",
            source="standard_radical",
        ),
        "h2_cation": MoleculeSpecData(
            name="h2_cation",
            atoms=(
                ("H", (0.0000, 0.0000, -0.5300)),
                ("H", (0.0000, 0.0000, 0.5300)),
            ),
            charge=1,
            spin=1,
            formula="H2",
            source="1_electron_cation",
        ),
        "h2o_cation": MoleculeSpecData(
            name="h2o_cation",
            atoms=(
                ("O", (0.0000, 0.0000, 0.1173)),
                ("H", (0.0000, 0.7800, -0.4700)),
                ("H", (0.0000, -0.7800, -0.4700)),
            ),
            charge=1,
            spin=1,
            formula="H2O",
            source="charged_radical",
        ),
        "o2_anion": MoleculeSpecData(
            name="o2_anion",
            atoms=(
                ("O", (0.0000, 0.0000, -0.6750)),
                ("O", (0.0000, 0.0000, 0.6750)),
            ),
            charge=-1,
            spin=1,
            formula="O2",
            source="charged_radical_anion",
        ),
        "ch3cl_h2s": MoleculeSpecData(
            name="ch3cl_h2s",
            atoms=(
                ("C", (-1.5000, 0.0000, 0.0000)),
                ("Cl", (0.2800, 0.0000, 0.0000)),
                ("H", (-1.8600, 1.0300, 0.0000)),
                ("H", (-1.8600, -0.5150, 0.8920)),
                ("H", (-1.8600, -0.5150, -0.8920)),
                ("S", (3.6000, 0.0000, 0.0000)),
                ("H", (4.5400, 0.9500, 0.0000)),
                ("H", (4.5400, -0.9500, 0.0000)),
            ),
            formula="CH5ClS",
            source="heteroatom_d3_complex",
        ),
        "formamide_dimer": MoleculeSpecData(
            name="formamide_dimer",
            atoms=(
                ("C", (-1.8680, 0.0290, 0.0000)),
                ("O", (-1.3370, 1.1480, 0.0000)),
                ("N", (-1.2090, -1.1420, 0.0000)),
                ("H", (-2.9630, -0.0560, 0.0000)),
                ("H", (-0.1800, -1.1410, 0.0000)),
                ("H", (-1.7160, -2.0120, 0.0000)),
                ("C", (1.8680, -0.0290, 0.0000)),
                ("O", (1.3370, -1.1480, 0.0000)),
                ("N", (1.2090, 1.1420, 0.0000)),
                ("H", (2.9630, 0.0560, 0.0000)),
                ("H", (0.1800, 1.1410, 0.0000)),
                ("H", (1.7160, 2.0120, 0.0000)),
            ),
            formula="C2H6N2O2",
            source="c2h_hbonded_dimer",
        ),
        "nh3_dimer_cation": MoleculeSpecData(
            name="nh3_dimer_cation",
            atoms=(
                ("N", (-1.1000, 0.0000, 0.0000)),
                ("H", (-1.4500, 0.9500, 0.0000)),
                ("H", (-1.4500, -0.4750, 0.8227)),
                ("H", (-1.4500, -0.4750, -0.8227)),
                ("N", (1.1000, 0.0000, 0.0000)),
                ("H", (1.4500, -0.9500, 0.0000)),
                ("H", (1.4500, 0.4750, 0.8227)),
                ("H", (1.4500, 0.4750, -0.8227)),
            ),
            charge=1,
            spin=1,
            formula="H6N2",
            source="charged_radical_dimer",
        ),
        "stretched_h2": MoleculeSpecData(
            name="stretched_h2",
            atoms=(
                ("H", (0.0000, 0.0000, -5.0000)),
                ("H", (0.0000, 0.0000, 5.0000)),
            ),
            charge=0,
            spin=0,
            formula="H2",
            source="low_density_10A",
        ),
        "stretched_lih": MoleculeSpecData(
            name="stretched_lih",
            atoms=(
                ("Li", (0.0000, 0.0000, 0.0000)),
                ("H", (0.0000, 0.0000, 4.0000)),
            ),
            charge=0,
            spin=0,
            formula="HLi",
            source="dissociation_4A",
        ),
        "he2_cation": MoleculeSpecData(
            name="he2_cation",
            atoms=(
                ("He", (0.0000, 0.0000, -1.5000)),
                ("He", (0.0000, 0.0000, 1.5000)),
            ),
            charge=1,
            spin=1,
            formula="He2",
            source="stretched_3A_radical_cation",
        ),
    }
    return {k: _validate_molecule(v, min_r_ang=0.65) for k, v in raw.items()}


def _build_large_catalog() -> dict[str, MoleculeSpecData]:
    h2o_32 = _validate_molecule(
        MoleculeSpecData(
            name="h2o_32",
            atoms=tuple(get_water_cluster_coords(4, 2, 4)),
            charge=0,
            spin=0,
            formula="H64O32",
            source="deterministic_4x2x4_water_cluster",
        ),
        min_r_ang=0.85,
    )
    c60 = _validate_molecule(
        MoleculeSpecData(
            name="c60",
            atoms=tuple(_build_c60_atoms()),
            charge=0,
            spin=0,
            formula="C60",
            source="deterministic_Ih_truncated_icosahedron",
        ),
        min_r_ang=0.85,
    )
    taxol = parse_xyz_file(
        GEOMETRIES_DIR / "taxol.xyz",
        name="taxol",
        expected_formula="C47H51NO14",
        expected_atoms=113,
        min_r_ang=0.85,
    )
    valinomycin = parse_xyz_file(
        GEOMETRIES_DIR / "valinomycin.xyz",
        name="valinomycin",
        expected_formula="C54H90N6O18",
        expected_atoms=168,
        min_r_ang=0.85,
    )
    return {
        "h2o_32": h2o_32,
        "c60": c60,
        "taxol": taxol,
        "valinomycin": valinomycin,
    }


SMALL_MEDIUM_MOLECULE_NAMES: tuple[str, ...] = (
    "h2o",
    "ch2o",
    "ch4",
    "co2",
    "c6h6",
    "h2o_dimer",
    "oh_radical",
    "ch3_radical",
    "ho2_radical",
    "o2_triplet",
    "no_radical",
    "h2_cation",
    "h2o_cation",
    "o2_anion",
    "ch3cl_h2s",
    "formamide_dimer",
    "nh3_dimer_cation",
    "stretched_h2",
    "stretched_lih",
    "he2_cation",
)

LARGE_MOLECULE_NAMES: tuple[str, ...] = ("h2o_32", "c60", "taxol", "valinomycin")

_CATALOG_CACHE: dict[str, MoleculeSpecData] | None = None


def _get_catalog() -> dict[str, MoleculeSpecData]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        cat = {}
        cat.update(_build_small_medium_catalog())
        cat.update(_build_large_catalog())
        _CATALOG_CACHE = cat
    return _CATALOG_CACHE


def list_molecules() -> list[str]:
    """Return sorted list of available molecule keys."""
    return list(_get_catalog().keys())


def get_molecule(name: str) -> MoleculeSpecData:
    """Return validated MoleculeSpecData for the requested molecule key."""
    key = name.strip().lower().replace("(", "").replace(")", "")
    aliases = {
        "water_32": "h2o_32",
        "h2o32": "h2o_32",
    }
    key = aliases.get(key, key)
    cat = _get_catalog()
    if key not in cat:
        raise KeyError(f"Unknown molecule {name!r}; available: {sorted(cat)}")
    return cat[key]
