"""Shared pytest configuration, fixtures, and geometry/VRAM helpers for GPU4DFT E2E tests."""

from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Iterator

import numpy as np
import pytest

TESTS_E2E_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_E2E_DIR.parent.parent
CODEGEN_DIR = REPO_ROOT / "codegen"

for _p in (str(TESTS_E2E_DIR), str(CODEGEN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_KCAL_MOL = 627.5094740631


def pytest_configure(config: pytest.Config) -> None:
    """Register E2E markers when invoked without -c tests/e2e/pytest.ini."""
    markers = [
        "tier1: Tier 1 isolated happy-path feature tests",
        "tier2: Tier 2 boundary, edge, invariant, and negative tests",
        "tier3: Tier 3 cross-feature pairwise and multi-feature combination tests",
        "tier4: Tier 4 realistic end-to-end application workflow tests",
        "f1: Feature F1 (libxc CMake linkage, spin-polarized XcEngine, grid_response)",
        "f2: Feature F2 (DFT-D3(BJ) dispersion energy and analytical gradients)",
        "f3: Feature F3 (Open-shell UHF/UKS SCF and direct analytical gradients)",
        "f4: Feature F4 (Range-separated hybrids CAM-B3LYP and wB97X)",
        "f5: Feature F5 (Memory-bounded occupied-space RI-K)",
        "f6: Feature F6 (RI-JK analytical nuclear gradients with auxbasis_response)",
        "f7: Feature F7 (Fair 5-repeat benchmark protocol and large molecules)",
        "f8: Feature F8 (Geometry optimizer integration and S22/BH76 workflows)",
        "gpu: Requires CUDA GPU execution or GPU VRAM telemetry",
        "slow: Long-running or large-molecule tests",
        "existing_core: Baseline smoke tests runnable on existing closed-shell core stack",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def require_gpu4dft():
    return pytest.importorskip("gpu4dft")


@pytest.fixture(scope="session")
def require_pyscf():
    return pytest.importorskip("pyscf")


@pytest.fixture(scope="session")
def require_dftd3():
    return pytest.importorskip("dftd3")


@pytest.fixture(scope="session")
def require_ase():
    return pytest.importorskip("ase")


@pytest.fixture(scope="session")
def gpu4dft_backend(require_gpu4dft) -> str:
    """Return concrete GPU4DFT backend ('cuda' if available else 'host'; never 'auto')."""
    return "cuda" if require_gpu4dft.backend_available("cuda") else "host"


@pytest.fixture(scope="session")
def make_pyscf_mol(require_pyscf) -> Callable[..., Any]:
    """Factory creating spherical PySCF Mole objects from str, atom list, or ASE Atoms."""
    from pyscf import gto

    def _make(
        atom_spec: Any,
        basis: str = "def2-svp",
        charge: int = 0,
        spin: int = 0,
        unit: str = "Angstrom",
    ):
        if hasattr(atom_spec, "get_chemical_symbols") and hasattr(atom_spec, "get_positions"):
            syms = atom_spec.get_chemical_symbols()
            pos = np.asarray(atom_spec.get_positions(), dtype=np.float64)
            spec: Any = [
                (str(sym), (float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2])))
                for i, sym in enumerate(syms)
            ]
        else:
            spec = atom_spec
        return gto.M(
            atom=spec,
            basis=basis,
            charge=int(charge),
            spin=int(spin),
            unit=unit,
            cart=False,
            verbose=0,
        )

    return _make


@pytest.fixture(scope="session")
def match_grids() -> Callable[[Any, Any], Any]:
    """Copy PySCF integration grid onto a GPU4DFT SCF object for accuracy tests ONLY.

    WARNING: NEVER use match_grids / shared_grids in F7 benchmark tests!
    """

    def _match(gpu4dft_mf: Any, pyscf_mf: Any) -> Any:
        grids = getattr(pyscf_mf, "grids", None)
        if grids is None:
            raise ValueError("pyscf_mf has no .grids attribute")
        if getattr(grids, "coords", None) is None or getattr(grids, "weights", None) is None:
            grids.build()
        gpu4dft_mf.grids.coords = np.ascontiguousarray(grids.coords, dtype=np.float64)
        gpu4dft_mf.grids.weights = np.ascontiguousarray(grids.weights, dtype=np.float64)
        gpu4dft_mf._engine = None
        return gpu4dft_mf

    return _match


def get_gpu_vram_usage_mib() -> tuple[float, float]:
    """Return (used_mib, total_mib) via nvidia-smi, or (0.0, 0.0) if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        if out:
            first_line = out.splitlines()[0]
            used_s, total_s = [x.strip() for x in first_line.split(",")[:2]]
            return float(used_s), float(total_s)
    except Exception:
        pass
    return 0.0, 0.0


@contextmanager
def track_peak_vram_mib() -> Iterator[dict[str, float]]:
    """Measure process/device GPU VRAM delta and peak around an execution block."""
    used_before, total_mib = get_gpu_vram_usage_mib()
    info: dict[str, float] = {
        "used_before_mib": used_before,
        "used_after_mib": used_before,
        "delta_mib": 0.0,
        "total_mib": total_mib,
    }
    try:
        yield info
    finally:
        used_after, total_after = get_gpu_vram_usage_mib()
        info["used_after_mib"] = used_after
        info["delta_mib"] = max(0.0, used_after - used_before)
        if total_after > 0.0:
            info["total_mib"] = total_after


def rotation_matrix(axis: tuple[float, float, float], angle_rad: float) -> np.ndarray:
    """Rodrigues 3x3 rotation matrix for a given axis and angle in radians."""
    ax = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(ax))
    if norm == 0.0:
        raise ValueError("rotation axis must be non-zero")
    k = ax / norm
    kx = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle_rad) * kx + (1.0 - math.cos(angle_rad)) * (kx @ kx)


def translate_coords(coords: np.ndarray, shift: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    """Return translated (natm, 3) coordinate copy."""
    arr = np.asarray(coords, dtype=np.float64)
    return arr + np.asarray(shift, dtype=np.float64).reshape(1, 3)


def rotate_coords(
    coords: np.ndarray,
    axis: tuple[float, float, float] = (1.0, 2.0, 3.0),
    angle_deg: float = 37.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rotated_coords, R_3x3) where rotated_coords = coords @ R.T."""
    rot = rotation_matrix(axis, math.radians(angle_deg))
    arr = np.asarray(coords, dtype=np.float64)
    return arr @ rot.T, rot


def min_interatomic_distance(coords: np.ndarray) -> float:
    """Compute minimum pairwise Euclidean distance for (natm, 3) coordinates."""
    arr = np.asarray(coords, dtype=np.float64)
    n = len(arr)
    if n < 2:
        return float("inf")
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.min(dists[iu]))
