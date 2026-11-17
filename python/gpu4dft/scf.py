"""PySCF-style Python frontend wrapping the zero-Python-hot-path C++20/CUDA SCF workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core
from ._core import (
    Atom,
    Backend,
    Basis,
    DensitySolver,
    GemmMode,
    GradientResult,
    JKMode,
    MolecularScf,
    PartitionScheme,
    Precision,
    WorkflowOptions,
    WorkflowResult,
)
from .pyscf_interface import (
    atoms_from_pyscf,
    aux_basis_from_pyscf,
    basis_from_pyscf,
    minao_init_guess,
)


def _coerce_enum(enum_cls, val):
    if isinstance(val, enum_cls):
        return val
    if isinstance(val, str):
        key = val.strip().lower().replace("-", "_")
        try:
            return enum_cls[key]
        except KeyError:
            pass
    names = ", ".join(member.name for member in enum_cls)
    raise ValueError(f"unknown {enum_cls.__name__} {val!r}; choose from {names}")


def _default_backend() -> Backend:
    return Backend.cuda if _core.backend_available(Backend.cuda) else Backend.host


def _is_pyscf_mole(mol: Any) -> bool:
    return hasattr(mol, "nbas") and hasattr(mol, "bas_angular") and hasattr(mol, "atom_coords")


@dataclass
class MoleculeSpec:
    """Standalone molecule specification when not using a ``pyscf.gto.Mole``."""

    basis: Basis
    atoms: list[Atom]
    nelectron: int
    aux_basis: Basis | None = None


class GridsConfig:
    """Lightweight PySCF-like grids configuration proxy (`mf.grids.level = 3`, etc.)."""

    def __init__(
        self,
        level: int = 3,
        coords: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        n_rad: int = 0,
        n_ang: int = 0,
        prune: bool = True,
        partition: PartitionScheme | str = "stratmann",
        max_batch_points: int = 256,
        ao_cutoff: float = 1e-12,
    ) -> None:
        self.level = int(level)
        self.coords = coords
        self.weights = weights
        self.n_rad = int(n_rad)
        self.n_ang = int(n_ang)
        self.prune = bool(prune)
        self.partition = partition
        self.max_batch_points = int(max_batch_points)
        self.ao_cutoff = float(ao_cutoff)


class NuclearGradients:
    """PySCF-like analytical nuclear gradient driver (`mf.nuc_grad_method().kernel()`)."""

    def __init__(self, mf: _SCFBase) -> None:
        self.base = mf
        self.de: np.ndarray | None = None
        self.result: GradientResult | None = None

    def kernel(self, dm: np.ndarray | None = None) -> np.ndarray:
        if self.base.result is None:
            self.base.kernel(dm0=dm)
        assert self.base._engine is not None
        gres = self.base._engine.compute_gradient()
        self.result = gres
        self.de = np.asarray(gres.total_grad, dtype=np.float64)
        return self.de

    def grad(self, dm: np.ndarray | None = None) -> np.ndarray:
        return self.kernel(dm=dm)


class _SCFBase:
    """Shared base class for :class:`RHF` and :class:`RKS`."""

    def __init__(
        self,
        mol: Any,
        *,
        xc: str = "hf",
        backend: Backend | str | None = None,
        precision: Precision | str = "fp64",
        dynamic_precision: bool | None = None,
        gemm_mode: GemmMode | str = "fp64",
        early_gemm_mode: GemmMode | str | None = None,
        solver: DensitySolver | str = "diag",
        incremental: bool = True,
        use_cuda_graph: bool = True,
        grids_level: int = 3,
        grids: tuple[np.ndarray, np.ndarray] | Any | None = None,
        conv_tol: float = 1e-9,
        diis_tol: float = 1e-6,
        max_cycle: int = 100,
        diis_space: int = 8,
        damping: float = 0.0,
        level_shift: float = 0.0,
        lindep_tol: float = 1e-7,
        schwarz_cutoff: float = 1e-13,
        fp32_threshold: float = 1e-6,
        ozaki_slices: int = 8,
        init_guess: str = "minao",
    ) -> None:
        self.mol = mol
        self.xc = xc
        self.backend = _default_backend() if backend is None else _coerce_enum(Backend, backend)
        self.precision = _coerce_enum(Precision, precision)
        self.dynamic_precision = dynamic_precision
        self.gemm_mode = _coerce_enum(GemmMode, gemm_mode)
        self.early_gemm_mode = (
            None if early_gemm_mode is None else _coerce_enum(GemmMode, early_gemm_mode)
        )
        self.solver = _coerce_enum(DensitySolver, solver)
        self.incremental = bool(incremental)
        self.use_cuda_graph = bool(use_cuda_graph)

        self.conv_tol = float(conv_tol)
        self.diis_tol = float(diis_tol)
        self.max_cycle = int(max_cycle)
        self.diis_space = int(diis_space)
        self.damping = float(damping)
        self.level_shift = float(level_shift)
        self.lindep_tol = float(lindep_tol)
        self.schwarz_cutoff = float(schwarz_cutoff)
        self.fp32_threshold = float(fp32_threshold)
        self.ozaki_slices = int(ozaki_slices)
        self.init_guess = str(init_guess)

        self.grids = GridsConfig(level=grids_level)
        if grids is not None:
            self._apply_grids_arg(grids)

        # J/K mode flags (modulated via .density_fit() and .snk())
        self._with_df: bool = False
        self.auxbasis: str | dict | Basis = "def2-universal-jkfit"
        self._with_snk: bool = False
        self.snk_overlap_fitting: bool = True
        self.snk_pair_cutoff: float = 1e-13
        self.snk_lindep_tol: float = 1e-8
        self.snk_precision: Precision | str | None = None

        # Populated on .setup() / .kernel()
        self._engine: MolecularScf | None = None
        self._setup_sig: tuple | None = None
        self.result: WorkflowResult | None = None
        self.converged: bool = False
        self.e_tot: float = 0.0
        self.iterations: int = 0
        self.mo_energy: np.ndarray | None = None
        self.mo_coeff: np.ndarray | None = None
        self.scf_summary: dict[str, float] = {}

    def _apply_grids_arg(self, grids: Any) -> None:
        if isinstance(grids, GridsConfig):
            self.grids = grids
        elif isinstance(grids, tuple) and len(grids) == 2:
            self.grids.coords = np.ascontiguousarray(grids[0], dtype=np.float64)
            self.grids.weights = np.ascontiguousarray(grids[1], dtype=np.float64)
        elif hasattr(grids, "coords") and hasattr(grids, "weights"):
            if getattr(grids, "coords", None) is None and hasattr(grids, "build"):
                grids.build()
            self.grids.coords = np.ascontiguousarray(grids.coords, dtype=np.float64)
            self.grids.weights = np.ascontiguousarray(grids.weights, dtype=np.float64)
        else:
            raise TypeError(
                "grids must be None, (coords, weights), GridsConfig, or a PySCF Grids object"
            )

    def density_fit(self, auxbasis: str | dict | Basis = "def2-universal-jkfit") -> _SCFBase:
        """Switch Coulomb/Exchange evaluation to 3-center Density Fitting (`RIJKEngine`)."""
        self._with_df = True
        self.auxbasis = auxbasis
        self._engine = None
        return self

    def snk(
        self,
        *,
        overlap_fitting: bool = True,
        pair_cutoff: float = 1e-13,
        lindep_tol: float = 1e-8,
        precision: Precision | str | None = None,
    ) -> _SCFBase:
        """Switch exact exchange K evaluation to Semi-Numerical Exchange (`sn-K` / COSX)."""
        self._with_snk = True
        self.snk_overlap_fitting = bool(overlap_fitting)
        self.snk_pair_cutoff = float(pair_cutoff)
        self.snk_lindep_tol = float(lindep_tol)
        self.snk_precision = precision
        self._engine = None
        return self

    def _extract_molecule(self) -> tuple[Basis, list[Atom], int, Basis | None]:
        if isinstance(self.mol, MoleculeSpec):
            aux_b = self.mol.aux_basis
            if self._with_df and isinstance(self.auxbasis, Basis):
                aux_b = self.auxbasis
            return self.mol.basis, self.mol.atoms, int(self.mol.nelectron), aux_b

        if _is_pyscf_mole(self.mol):
            if getattr(self.mol, "_built", True) is False and hasattr(self.mol, "build"):
                self.mol.build()
            basis = basis_from_pyscf(self.mol)
            atoms = atoms_from_pyscf(self.mol)
            nelec = int(self.mol.nelectron)
            aux_b: Basis | None = None
            if self._with_df:
                if isinstance(self.auxbasis, Basis):
                    aux_b = self.auxbasis
                else:
                    aux_b = aux_basis_from_pyscf(self.mol, auxbasis=self.auxbasis)
            return basis, atoms, nelec, aux_b

        raise TypeError("mol must be a pyscf.gto.Mole or gpu4dft.MoleculeSpec")

    def _build_workflow_options(self) -> WorkflowOptions:
        be = _coerce_enum(Backend, self.backend)
        prec = _coerce_enum(Precision, self.precision)
        gmode = _coerce_enum(GemmMode, self.gemm_mode)
        solv = _coerce_enum(DensitySolver, self.solver)

        if self.dynamic_precision is None:
            dyn_prec = prec != Precision.fp64
        else:
            dyn_prec = bool(self.dynamic_precision)

        if dyn_prec:
            early_prec = Precision.mixed if prec == Precision.fp64 else prec
            early_gemm = (
                _coerce_enum(GemmMode, self.early_gemm_mode)
                if self.early_gemm_mode is not None
                else (GemmMode.ozaki_int8 if gmode == GemmMode.fp64 else gmode)
            )
        else:
            early_prec = prec
            early_gemm = gmode

        if self._with_df and self._with_snk:
            jk_mode = JKMode.ri_snk
        elif self._with_df:
            jk_mode = JKMode.ri
        elif self._with_snk:
            jk_mode = JKMode.snk
        else:
            jk_mode = JKMode.direct

        wopts = WorkflowOptions()
        wopts.xc = self.xc
        wopts.jk_mode = jk_mode
        wopts.schwarz_cutoff = float(self.schwarz_cutoff)
        wopts.fp32_threshold = float(self.fp32_threshold)

        sopts = wopts.scf
        sopts.backend = be
        sopts.solver = solv
        sopts.gemm_mode = gmode
        sopts.early_gemm_mode = early_gemm
        sopts.early_integral_precision = early_prec
        sopts.max_iter = int(self.max_cycle)
        sopts.conv_tol = float(self.conv_tol)
        sopts.diis_tol = float(self.diis_tol)
        sopts.diis_max_vec = int(self.diis_space)
        sopts.damping = float(self.damping)
        sopts.level_shift = float(self.level_shift)
        sopts.lindep_tol = float(self.lindep_tol)
        sopts.incremental_fock = bool(self.incremental)
        sopts.dynamic_precision = dyn_prec
        sopts.use_cuda_graphs = bool(self.use_cuda_graph)
        sopts.ozaki_slices = int(self.ozaki_slices)
        wopts.scf = sopts

        gopts = wopts.grid
        gopts.level = int(self.grids.level)
        gopts.n_rad = int(self.grids.n_rad)
        gopts.n_ang = int(self.grids.n_ang)
        gopts.prune = bool(self.grids.prune)
        gopts.partition = _coerce_enum(PartitionScheme, self.grids.partition)
        gopts.max_batch_points = int(self.grids.max_batch_points)
        gopts.ao_cutoff = float(self.grids.ao_cutoff)
        gopts.backend = be
        wopts.grid = gopts

        xopts = wopts.xc_options
        xopts.backend = be
        xopts.precision = Precision.fp32 if prec == Precision.fp32 else Precision.fp64
        wopts.xc_options = xopts

        snk_opts = wopts.snk_options
        snk_opts.backend = be
        snk_opts.overlap_fitting = bool(self.snk_overlap_fitting)
        snk_opts.pair_cutoff = float(self.snk_pair_cutoff)
        snk_opts.lindep_tol = float(self.snk_lindep_tol)
        if self.snk_precision is not None:
            p_snk = _coerce_enum(Precision, self.snk_precision)
            snk_opts.precision = Precision.fp32 if p_snk == Precision.fp32 else Precision.fp64
        else:
            snk_opts.precision = Precision.fp32 if prec == Precision.fp32 else Precision.fp64
        wopts.snk_options = snk_opts

        return wopts

    def _config_signature(self) -> tuple:
        g_id = (
            None
            if self.grids.coords is None
            else (self.grids.coords.shape, int(self.grids.coords.__array_interface__["data"][0]))
        )
        return (
            self.xc,
            str(self.backend),
            str(self.precision),
            self.dynamic_precision,
            str(self.gemm_mode),
            str(self.early_gemm_mode),
            str(self.solver),
            self.incremental,
            self.use_cuda_graph,
            self.grids.level,
            self.grids.n_rad,
            self.grids.n_ang,
            self.grids.prune,
            str(self.grids.partition),
            g_id,
            self.conv_tol,
            self.diis_tol,
            self.max_cycle,
            self.diis_space,
            self.damping,
            self.level_shift,
            self.lindep_tol,
            self.schwarz_cutoff,
            self.fp32_threshold,
            self.ozaki_slices,
            self._with_df,
            id(self.auxbasis) if isinstance(self.auxbasis, Basis) else str(self.auxbasis),
            self._with_snk,
            self.snk_overlap_fitting,
            self.snk_pair_cutoff,
            self.snk_lindep_tol,
            str(self.snk_precision),
        )

    def setup(self) -> _SCFBase:
        """Explicitly construct persistent C++20/CUDA integral, JK, grid, and XC engines once."""
        sig = self._config_signature()
        if self._engine is not None and self._setup_sig == sig:
            return self

        basis, atoms, nelec, aux_b = self._extract_molecule()
        wopts = self._build_workflow_options()

        g_coords = None
        g_weights = None
        if self.grids.coords is not None or self.grids.weights is not None:
            if self.grids.coords is None or self.grids.weights is None:
                raise ValueError("Both grids.coords and grids.weights must be set")
            g_coords = np.ascontiguousarray(self.grids.coords, dtype=np.float64)
            g_weights = np.ascontiguousarray(self.grids.weights, dtype=np.float64)

        self._engine = MolecularScf(
            basis,
            atoms,
            nelec,
            opts=wopts,
            aux_basis=aux_b,
            grid_coords=g_coords,
            grid_weights=g_weights,
        )
        self._engine.setup()
        self._setup_sig = sig
        return self

    def get_init_guess(self, key: str | None = None) -> np.ndarray | None:
        method = (key or self.init_guess).strip().lower()
        if method in ("1e", "hcore"):
            return None
        if method == "minao" and _is_pyscf_mole(self.mol):
            return minao_init_guess(self.mol)
        return None

    def kernel(self, dm0: np.ndarray | None = None) -> float:
        """Execute the entire SCF loop in C++20/CUDA and return converged total energy (Hartree)."""
        self.setup()
        assert self._engine is not None

        if dm0 is None:
            init_dm = self.get_init_guess()
        else:
            init_dm = np.ascontiguousarray(dm0, dtype=np.float64)

        wres = self._engine.run(dm0=init_dm)
        self.result = wres
        sres = wres.scf
        self.converged = bool(sres.converged)
        self.e_tot = float(sres.total_energy)
        self.iterations = int(sres.iterations)
        self.mo_energy = np.asarray(sres.mo_energies, dtype=np.float64)
        self.mo_coeff = np.asarray(sres.mo_coefficients, dtype=np.float64)
        self.scf_summary = {
            "e1": float(sres.one_electron_energy),
            "coul_exch": float(sres.two_electron_energy),
            "exc": float(sres.xc_energy),
            "nuc": float(sres.nuclear_repulsion),
            "e_tot": float(sres.total_energy),
            "setup_time_s": float(wres.setup_time_s),
            "scf_time_s": float(wres.scf_time_s),
        }
        return self.e_tot

    def run(self, dm0: np.ndarray | None = None) -> _SCFBase:
        """PySCF-compatible convenience wrapper executing `.kernel()` and returning `self`."""
        self.kernel(dm0=dm0)
        return self

    def make_rdm1(self) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("SCF has not been run yet; call .kernel() first")
        return np.asarray(self.result.scf.density_matrix, dtype=np.float64)

    def make_rdm1e(self) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("SCF has not been run yet; call .kernel() first")
        return np.asarray(self.result.scf.energy_weighted_dm, dtype=np.float64)

    def get_ovlp(self) -> np.ndarray:
        if self.result is not None:
            return np.asarray(self.result.overlap_matrix, dtype=np.float64)
        basis, _, _, _ = self._extract_molecule()
        return np.asarray(_core.overlap_matrix(basis, _coerce_enum(Backend, self.backend)))

    def get_hcore(self) -> np.ndarray:
        if self.result is not None:
            return np.asarray(self.result.hcore_matrix, dtype=np.float64)
        basis, atoms, _, _ = self._extract_molecule()
        be = _coerce_enum(Backend, self.backend)
        return np.asarray(_core.kinetic_matrix(basis, be)) + np.asarray(
            _core.nuclear_attraction_matrix(basis, atoms, be)
        )

    def get_fock(self) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("SCF has not been run yet; call .kernel() first")
        return np.asarray(self.result.scf.fock_matrix, dtype=np.float64)

    def nuc_grad_method(self) -> NuclearGradients:
        return NuclearGradients(self)

    Gradients = nuc_grad_method


class RHF(_SCFBase):
    """Restricted Hartree-Fock driver executing end-to-end inside C++20/CUDA."""

    def __init__(self, mol: Any, **kwargs: Any) -> None:
        kwargs.pop("xc", None)
        super().__init__(mol, xc="hf", **kwargs)


class RKS(_SCFBase):
    """Restricted Kohn-Sham / Hybrid DFT driver executing end-to-end inside C++20/CUDA."""

    def __init__(self, mol: Any, xc: str = "b3lyp", **kwargs: Any) -> None:
        super().__init__(mol, xc=xc, **kwargs)
