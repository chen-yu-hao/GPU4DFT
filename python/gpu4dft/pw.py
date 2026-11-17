"""Plane-Wave Kohn-Sham & Hybrid ACE DFT Python interface (`gpu4dft.pw`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core
from ._core import Backend
from ._core import pw as _pw
from ._core.pw import (
    ExxDiv,
    GthNonLocalChannel,
    GthPseudopotential,
    KPointBasis,
    PwAtom,
    PwBasis,
    PwCell,
    PwEigenResult,
    PwKohnShamSolver,
    PwScfOptions,
    PwScfResult,
    PwSolverType,
    cuda_pw_available,
    element_atomic_number,
    element_symbol,
    lookup_gth_pseudopotential,
    next_fft_smooth_size,
)

BOHR_PER_ANGSTROM = 1.8897259885789233

__all__ = [
    "ExxDiv",
    "GthNonLocalChannel",
    "GthPseudopotential",
    "KRKS",
    "KPointBasis",
    "PwAtom",
    "PwBasis",
    "PwCell",
    "PwEigenResult",
    "PwKohnShamSolver",
    "PwScfOptions",
    "PwScfResult",
    "PwSolverType",
    "RHF",
    "RKS",
    "cell_from_pyscf",
    "cuda_pw_available",
    "element_atomic_number",
    "element_symbol",
    "ewald_energy",
    "fft3d_batch",
    "lookup_gth_pseudopotential",
    "madelung_constant",
    "next_fft_smooth_size",
    "run_scf",
    "solve_pw_eigenproblem",
]


def _as_backend(val: Backend | str | None) -> Backend:
    if val is None:
        return Backend.cuda if cuda_pw_available() else Backend.host
    if isinstance(val, Backend):
        return val
    if isinstance(val, str):
        key = val.strip().lower()
        if key in ("cuda", "gpu"):
            return Backend.cuda
        if key in ("host", "cpu"):
            return Backend.host
        if key == "hip":
            return Backend.hip
    raise ValueError(f"unknown Backend {val!r}; choose from 'cuda', 'host', 'hip'")


def _as_solver(val: PwSolverType | str) -> PwSolverType:
    if isinstance(val, PwSolverType):
        return val
    if isinstance(val, str):
        key = val.strip().lower().replace("-", "_")
        if key in ("davidson", "block_davidson"):
            return PwSolverType.davidson
        if key == "lobpcg":
            return PwSolverType.lobpcg
    raise ValueError(f"unknown PwSolverType {val!r}; choose from 'davidson', 'lobpcg'")


def _as_exxdiv(val: ExxDiv | str | None) -> ExxDiv:
    if val is None:
        return ExxDiv.none
    if isinstance(val, ExxDiv):
        return val
    if isinstance(val, str):
        key = val.strip().lower()
        if key in ("none", ""):
            return ExxDiv.none
        if key == "ewald":
            return ExxDiv.ewald
    raise ValueError(f"unknown ExxDiv {val!r}; choose from None, 'none', or 'ewald'")


def ewald_energy(
    a_bohr: np.ndarray,
    atomic_numbers: list[int] | np.ndarray,
    coords_bohr: np.ndarray,
    pseudo_family: str = "gth-pade",
    precision: float = 1e-13,
) -> float:
    """Exact 3D periodic Ewald ion-ion interaction energy (Hartree)."""
    a_c = np.ascontiguousarray(a_bohr, dtype=np.float64)
    z_list = [int(z) for z in atomic_numbers]
    pos_c = np.ascontiguousarray(coords_bohr, dtype=np.float64)
    return float(_pw.ewald_energy(a_c, z_list, pos_c, str(pseudo_family), float(precision)))


def madelung_constant(
    a_bohr: np.ndarray,
    atomic_numbers: list[int] | np.ndarray,
    coords_bohr: np.ndarray,
    pseudo_family: str = "gth-pade",
    kmesh: tuple[int, int, int] = (1, 1, 1),
    precision: float = 1e-13,
) -> float:
    """Exact periodic Wigner-Seitz supercell Madelung constant."""
    a_c = np.ascontiguousarray(a_bohr, dtype=np.float64)
    z_list = [int(z) for z in atomic_numbers]
    pos_c = np.ascontiguousarray(coords_bohr, dtype=np.float64)
    km = (int(kmesh[0]), int(kmesh[1]), int(kmesh[2]))
    return float(
        _pw.madelung_constant(a_c, z_list, pos_c, str(pseudo_family), km, float(precision))
    )


def fft3d_batch(
    data: np.ndarray,
    direction: int = -1,
    backend: Backend | str | None = None,
) -> np.ndarray:
    """Batched 3D complex-to-complex FFT on CUDA (`cuFFT`) or Host (Stockham/Bluestein).

    Accepts shape ``(n1, n2, n3)`` or ``(nbatch, n1, n2, n3)``.
    ``direction = -1`` is unnormalized forward FFT; ``+1`` is unnormalized inverse FFT.
    """
    arr = np.ascontiguousarray(data, dtype=np.complex128)
    squeeze = False
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
        squeeze = True
    elif arr.ndim != 4:
        raise ValueError("fft3d_batch expects a 3D or 4D complex array")
    out = np.asarray(_pw.fft3d_batch(arr, int(direction), _as_backend(backend)))
    return out[0] if squeeze else out


def solve_pw_eigenproblem(
    basis: PwBasis,
    ik: int = 0,
    vloc_r: np.ndarray | None = None,
    nband: int = 4,
    solver: PwSolverType | str = "davidson",
    backend: Backend | str | None = None,
    residual_tol: float = 1e-10,
    max_iter: int = 80,
    include_nonlocal_pp: bool = False,
) -> PwEigenResult:
    """Solve lowest ``nband`` plane-wave eigenpairs at k-point ``ik``."""
    v_arr = None if vloc_r is None else np.ascontiguousarray(vloc_r, dtype=np.float64).ravel()
    return _pw.solve_pw_eigenproblem(
        basis,
        int(ik),
        v_arr,
        int(nband),
        _as_solver(solver),
        _as_backend(backend),
        float(residual_tol),
        int(max_iter),
        bool(include_nonlocal_pp),
    )


def cell_from_pyscf(pbc_cell: Any, pseudo_family: str | None = None) -> PwCell:
    """Convert a ``pyscf.pbc.gto.Cell`` into a :class:`PwCell`."""
    if getattr(pbc_cell, "_built", True) is False and hasattr(pbc_cell, "build"):
        pbc_cell.build()
    a_bohr = np.ascontiguousarray(pbc_cell.lattice_vectors(), dtype=np.float64)
    coords_bohr = np.ascontiguousarray(pbc_cell.atom_coords(unit="Bohr"), dtype=np.float64)
    z_list = [
        element_atomic_number(pbc_cell.atom_pure_symbol(i)) for i in range(pbc_cell.natm)
    ]
    if pseudo_family is None:
        raw_pp = getattr(pbc_cell, "pseudo", None)
        pseudo_family = raw_pp if isinstance(raw_pp, str) and raw_pp else "gth-pade"
    return PwCell(a_bohr, z_list, coords_bohr, str(pseudo_family))


def _coerce_pw_cell(
    cell_or_a: PwCell | Any,
    atoms: list[PwAtom] | list[tuple[str | int, Any]] | None = None,
    pseudo_family: str = "gth-pade",
    unit: str = "Bohr",
) -> PwCell:
    if isinstance(cell_or_a, PwCell):
        return cell_or_a
    if hasattr(cell_or_a, "lattice_vectors") and hasattr(cell_or_a, "atom_coords"):
        return cell_from_pyscf(cell_or_a, pseudo_family=pseudo_family)

    scale = BOHR_PER_ANGSTROM if unit.strip().lower().startswith("ang") else 1.0
    a_bohr = np.ascontiguousarray(cell_or_a, dtype=np.float64) * scale
    pw_atoms: list[PwAtom] = []
    if atoms is not None:
        for item in atoms:
            if isinstance(item, PwAtom):
                pw_atoms.append(item)
            else:
                sym_or_z, pos = item
                pos_b = tuple(float(x) * scale for x in pos)
                if isinstance(sym_or_z, (int, np.integer)):
                    pw_atoms.append(PwAtom(int(sym_or_z), pos_b))
                else:
                    pw_atoms.append(PwAtom(str(sym_or_z), pos_b))
    return PwCell(a_bohr, pw_atoms, str(pseudo_family))


def run_scf(
    cell: PwCell | Any,
    atoms: list[PwAtom] | list[tuple[str | int, Any]] | None = None,
    *,
    xc: str = "pbe",
    ecut_wfc: float = 15.0,
    ecut_rho: float = 0.0,
    pseudo_family: str = "gth-pbe",
    unit: str = "Bohr",
    kpts: np.ndarray | None = None,
    kweights: np.ndarray | None = None,
    kmesh: tuple[int, int, int] = (1, 1, 1),
    fft_grid: tuple[int, int, int] = (0, 0, 0),
    solver: PwSolverType | str = "davidson",
    backend: Backend | str | None = None,
    exxdiv: ExxDiv | str | None = "ewald",
    use_ace: bool = True,
    charge: int = 0,
    nbands: int = 0,
    max_scf_iter: int = 80,
    conv_tol: float = 1e-9,
) -> PwScfResult:
    """High-level single-call Plane-Wave Kohn-Sham / ACE Hybrid DFT entry point."""
    pw_cell = _coerce_pw_cell(cell, atoms=atoms, pseudo_family=pseudo_family, unit=unit)
    km = (int(kmesh[0]), int(kmesh[1]), int(kmesh[2]))

    k_arr = None
    kw_arr = None
    if kpts is not None:
        k_arr = np.ascontiguousarray(kpts, dtype=np.float64)
        if kweights is not None:
            kw_arr = np.ascontiguousarray(kweights, dtype=np.float64)
    elif km != (1, 1, 1):
        k_arr, kw_arr = pw_cell.monkhorst_pack_kpoints(km)

    fg = (int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2]))
    basis = PwBasis(pw_cell, float(ecut_wfc), float(ecut_rho), k_arr, kw_arr, fg)

    opts = PwScfOptions()
    opts.backend = _as_backend(backend)
    opts.eigensolver = _as_solver(solver)
    opts.xc = str(xc)
    opts.exxdiv = _as_exxdiv(exxdiv)
    opts.kmesh = km
    opts.use_ace = bool(use_ace)
    opts.charge = int(charge)
    opts.nbands = int(nbands)
    opts.max_scf_iter = int(max_scf_iter)
    opts.conv_tol = float(conv_tol)

    ks = PwKohnShamSolver(basis, opts)
    return ks.run()


class _PWSCFBase:
    """PySCF-style periodic Plane-Wave Kohn-Sham / ACE Hybrid DFT driver."""

    def __init__(
        self,
        cell: PwCell | Any,
        *,
        xc: str = "pbe",
        ecut_wfc: float = 15.0,
        ecut_rho: float = 0.0,
        pseudo_family: str | None = None,
        kpts: np.ndarray | None = None,
        kweights: np.ndarray | None = None,
        kmesh: tuple[int, int, int] = (1, 1, 1),
        fft_grid: tuple[int, int, int] = (0, 0, 0),
        solver: PwSolverType | str = "davidson",
        backend: Backend | str | None = None,
        exxdiv: ExxDiv | str | None = "ewald",
        use_ace: bool = True,
        charge: int = 0,
        nbands: int = 0,
        max_cycle: int = 80,
        conv_tol: float = 1e-9,
    ) -> None:
        if isinstance(cell, PwCell):
            self.cell = cell
        else:
            default_pp = pseudo_family or (
                "gth-pade" if xc.lower() in ("lda", "svwn", "spz81", "hf") else "gth-pbe"
            )
            self.cell = _coerce_pw_cell(cell, pseudo_family=default_pp)
        self.xc = xc
        self.ecut_wfc = float(ecut_wfc)
        self.ecut_rho = float(ecut_rho)
        self.kpts = kpts
        self.kweights = kweights
        self.kmesh = kmesh
        self.fft_grid = fft_grid
        self.solver = solver
        self.backend = backend
        self.exxdiv = exxdiv
        self.use_ace = bool(use_ace)
        self.charge = int(charge)
        self.nbands = int(nbands)
        self.max_cycle = int(max_cycle)
        self.conv_tol = float(conv_tol)

        self.result: PwScfResult | None = None
        self.converged: bool = False
        self.e_tot: float = 0.0
        self.mo_energy: list[np.ndarray] | np.ndarray | None = None

    def kernel(self) -> float:
        res = run_scf(
            self.cell,
            xc=self.xc,
            ecut_wfc=self.ecut_wfc,
            ecut_rho=self.ecut_rho,
            kpts=self.kpts,
            kweights=self.kweights,
            kmesh=self.kmesh,
            fft_grid=self.fft_grid,
            solver=self.solver,
            backend=self.backend,
            exxdiv=self.exxdiv,
            use_ace=self.use_ace,
            charge=self.charge,
            nbands=self.nbands,
            max_scf_iter=self.max_cycle,
            conv_tol=self.conv_tol,
        )
        self.result = res
        self.converged = bool(res.converged)
        self.e_tot = float(res.total_energy)
        eigs = [np.asarray(ev, dtype=np.float64) for ev in res.eigenvalues]
        self.mo_energy = eigs[0] if len(eigs) == 1 else eigs
        return self.e_tot

    def run(self) -> _PWSCFBase:
        self.kernel()
        return self


class RKS(_PWSCFBase):
    """Gamma-point Plane-Wave Restricted Kohn-Sham / ACE Hybrid DFT driver."""

    def __init__(self, cell: PwCell | Any, xc: str = "pbe", **kwargs: Any) -> None:
        super().__init__(cell, xc=xc, **kwargs)


class KRKS(_PWSCFBase):
    """Brillouin-zone k-point Plane-Wave Restricted Kohn-Sham / ACE Hybrid DFT driver."""

    def __init__(
        self,
        cell: PwCell | Any,
        kpts: np.ndarray | None = None,
        xc: str = "pbe",
        **kwargs: Any,
    ) -> None:
        super().__init__(cell, xc=xc, kpts=kpts, **kwargs)


class RHF(_PWSCFBase):
    """Plane-Wave Restricted Hartree-Fock driver with Lin's ACE compression."""

    def __init__(self, cell: PwCell | Any, **kwargs: Any) -> None:
        kwargs.pop("xc", None)
        super().__init__(cell, xc="hf", **kwargs)
