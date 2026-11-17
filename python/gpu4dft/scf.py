"""PySCF-style Python frontend wrapping the zero-Python-hot-path C++20/CUDA SCF,
TDDFT/TDA/CIS Linear Response, Analytical Nuclear Gradients, and Nuclear Hessian workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    HessianOptions,
    JKMode,
    MolecularScf,
    PartitionScheme,
    Precision,
    ResponseMethod,
    ResponseOptions,
    ResponseResult,
    VibrationalResult,
    WorkflowOptions,
    WorkflowResult,
)
from .pyscf_interface import (
    atoms_from_pyscf,
    aux_basis_from_pyscf,
    basis_from_pyscf,
    minao_init_guess,
)

AU_TO_DEBYE = 2.541746473


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
        max_batch_points: int = 2048,
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


class LinearResponseSCF:
    """PySCF-like CIS / TDA / TDHF / TDDFT driver (`mf.TDA().kernel()`, `mf.TDDFT().kernel()`).

    Executes the entire Block-Davidson / Casida / SSF-RPA Krylov solver in C++20/CUDA
    reusing GPU-resident `DirectJKEngine` / `RIJKEngine` (supporting `.density_fit()` RI-TDDFT!)
    and GPU cached-AO 2nd-derivative `XcEngine::eval_fxc_device`.
    """

    def __init__(
        self,
        mf: _SCFBase,
        *,
        method: ResponseMethod | str = "tddft",
        nstates: int | None = None,
        n_states: int | None = None,
        conv_tol: float = 1e-7,
        max_cycle: int | None = None,
        max_iter: int | None = None,
        max_space: int | None = None,
        max_subspace: int | None = None,
        backend: Backend | str | None = None,
        gemm_mode: GemmMode | str | None = None,
        dynamic_precision: bool = False,
        early_gemm_mode: GemmMode | str = "fp32",
        fp64_switch_residual: float = 1e-3,
        ozaki_slices: int | None = None,
        denom_floor: float = 1e-8,
    ) -> None:
        self.base = mf
        self.method = _coerce_enum(ResponseMethod, method)
        self.nstates = int(nstates if nstates is not None else (n_states if n_states is not None else 5))
        self.conv_tol = float(conv_tol)
        self.max_cycle = int(
            max_cycle if max_cycle is not None else (max_iter if max_iter is not None else 60)
        )
        self.max_space = int(
            max_space if max_space is not None else (max_subspace if max_subspace is not None else 100)
        )
        self.backend = mf.backend if backend is None else _coerce_enum(Backend, backend)
        self.gemm_mode = mf.gemm_mode if gemm_mode is None else _coerce_enum(GemmMode, gemm_mode)
        self.dynamic_precision = bool(dynamic_precision)
        self.early_gemm_mode = _coerce_enum(GemmMode, early_gemm_mode)
        self.fp64_switch_residual = float(fp64_switch_residual)
        self.ozaki_slices = int(mf.ozaki_slices if ozaki_slices is None else ozaki_slices)
        self.denom_floor = float(denom_floor)

        self.result: ResponseResult | None = None
        self.converged: bool = False
        self.iterations: int = 0
        self.e: np.ndarray | None = None
        self.e_ev: np.ndarray | None = None
        self.f_osc: np.ndarray | None = None
        self.transition_dipoles: np.ndarray | None = None
        self.x_vectors: np.ndarray | None = None
        self.y_vectors: np.ndarray | None = None
        self.xy: list[tuple[np.ndarray, np.ndarray]] | None = None

    def kernel(
        self, nstates: int | None = None
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        if nstates is not None:
            self.nstates = int(nstates)
        if self.base.result is None:
            self.base.kernel()
        assert self.base._engine is not None

        opts = ResponseOptions()
        opts.backend = _coerce_enum(Backend, self.backend)
        opts.method = _coerce_enum(ResponseMethod, self.method)
        opts.gemm_mode = _coerce_enum(GemmMode, self.gemm_mode)
        opts.dynamic_precision = bool(self.dynamic_precision)
        opts.early_gemm_mode = _coerce_enum(GemmMode, self.early_gemm_mode)
        opts.fp64_switch_residual = float(self.fp64_switch_residual)
        opts.ozaki_slices = int(self.ozaki_slices)
        opts.n_states = int(self.nstates)
        opts.max_iter = int(self.max_cycle)
        opts.max_subspace = int(self.max_space)
        opts.conv_tol = float(self.conv_tol)
        opts.denom_floor = float(self.denom_floor)

        res = self.base._engine.run_response(opts)
        self.result = res
        self.converged = bool(res.converged)
        self.iterations = int(res.iterations)
        self.e = np.asarray(res.excitation_energies, dtype=np.float64)
        self.e_ev = np.asarray(res.excitation_energies_ev, dtype=np.float64)
        self.f_osc = np.asarray(res.oscillator_strengths, dtype=np.float64)
        self.transition_dipoles = np.asarray(res.transition_dipoles, dtype=np.float64)
        self.x_vectors = np.asarray(res.x_vectors, dtype=np.float64)
        self.y_vectors = np.asarray(res.y_vectors, dtype=np.float64)

        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        is_tda = opts.method == ResponseMethod.tda
        self.xy = [
            (
                self.x_vectors[k] * inv_sqrt2,
                0.0 if is_tda else (self.y_vectors[k] * inv_sqrt2),
            )
            for k in range(int(res.n_states))
        ]
        return self.e, self.xy

    def run(self, nstates: int | None = None) -> LinearResponseSCF:
        self.kernel(nstates=nstates)
        return self

    def oscillator_strength(self) -> np.ndarray:
        if self.f_osc is None:
            self.kernel()
        assert self.f_osc is not None
        return self.f_osc

    def transition_dipole(self) -> np.ndarray:
        if self.transition_dipoles is None:
            self.kernel()
        assert self.transition_dipoles is not None
        return self.transition_dipoles


class NuclearHessian:
    """PySCF-like molecular nuclear Hessian & harmonic vibrational frequency driver (`mf.Hessian()`)."""

    def __init__(
        self,
        mf: _SCFBase,
        *,
        step_bohr: float = 1e-3,
        project_trans_rot: bool = True,
        fixed_xc_grid: bool = True,
        masses: np.ndarray | list[float] | None = None,
        backend: Backend | str | None = None,
    ) -> None:
        self.base = mf
        self.step_bohr = float(step_bohr)
        self.project_trans_rot = bool(project_trans_rot)
        self.fixed_xc_grid = bool(fixed_xc_grid)
        self.masses = masses
        self.backend = mf.backend if backend is None else _coerce_enum(Backend, backend)

        self.result: VibrationalResult | None = None
        self.de: np.ndarray | None = None
        self.hessian_2d: np.ndarray | None = None
        self.frequencies_cm1: np.ndarray | None = None
        self.reduced_masses_amu: np.ndarray | None = None
        self.force_constants_au: np.ndarray | None = None
        self.normal_modes: np.ndarray | None = None
        self.zpe_hartree: float = 0.0

    @property
    def grid_response(self) -> bool:
        return not self.fixed_xc_grid

    @grid_response.setter
    def grid_response(self, val: bool) -> None:
        self.fixed_xc_grid = not bool(val)

    def kernel(self) -> np.ndarray:
        if self.base.result is None:
            self.base.kernel()
        assert self.base._engine is not None

        opts = HessianOptions()
        opts.backend = _coerce_enum(Backend, self.backend)
        opts.step_bohr = float(self.step_bohr)
        opts.project_trans_rot = bool(self.project_trans_rot)
        opts.fixed_xc_grid = bool(self.fixed_xc_grid)

        m_arr = (
            None
            if self.masses is None
            else np.ascontiguousarray(self.masses, dtype=np.float64)
        )
        vres = self.base._engine.run_hessian(opts=opts, masses=m_arr)
        self.result = vres
        n = int(vres.natom)
        h2 = np.asarray(vres.hessian, dtype=np.float64)
        self.hessian_2d = h2
        # PySCF Hessian convention: shape (natom, natom, 3, 3)
        self.de = np.ascontiguousarray(h2.reshape(n, 3, n, 3).transpose(0, 2, 1, 3))
        self.frequencies_cm1 = np.asarray(vres.frequencies_cm1, dtype=np.float64)
        self.reduced_masses_amu = np.asarray(vres.reduced_masses_amu, dtype=np.float64)
        self.force_constants_au = np.asarray(vres.force_constants_au, dtype=np.float64)
        self.normal_modes = np.asarray(vres.normal_modes, dtype=np.float64)
        self.zpe_hartree = float(vres.zero_point_energy_hartree)
        return self.de

    def run(self) -> NuclearHessian:
        self.kernel()
        return self

    def harmonic_analysis(self) -> dict[str, Any]:
        if self.result is None:
            self.kernel()
        return {
            "freq_wavenumber": self.frequencies_cm1,
            "reduced_mass": self.reduced_masses_amu,
            "force_const_au": self.force_constants_au,
            "norm_mode": self.normal_modes,
            "zpe_hartree": self.zpe_hartree,
        }


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
            self.grids.max_batch_points,
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

    def dipole_moment(
        self,
        unit: str = "Debye",
        origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Compute permanent molecular dipole moment in `'Debye'` (default) or `'AU'` (`e*Bohr`)."""
        if self.result is None:
            self.kernel()
        assert self._engine is not None
        orig = (float(origin[0]), float(origin[1]), float(origin[2]))
        mu_au = np.asarray(self._engine.compute_dipole(orig), dtype=np.float64)
        u = unit.strip().lower()
        if u in ("debye", "d"):
            return mu_au * AU_TO_DEBYE
        if u in ("au", "a.u.", "atomic", "bohr", "e*bohr"):
            return mu_au
        raise ValueError(f"unknown dipole unit {unit!r}; choose 'Debye' or 'AU'")

    dip_moment = dipole_moment

    def CIS(self, **kwargs: Any) -> LinearResponseSCF:
        """Configuration Interaction Singles / Tamm-Dancoff excited-state driver."""
        return LinearResponseSCF(self, method=ResponseMethod.tda, **kwargs)

    def TDA(self, **kwargs: Any) -> LinearResponseSCF:
        """Tamm-Dancoff Approximation singlet excited-state driver (Direct or RI-TDA)."""
        return LinearResponseSCF(self, method=ResponseMethod.tda, **kwargs)

    def TDHF(self, **kwargs: Any) -> LinearResponseSCF:
        """Time-Dependent Hartree-Fock / RPA singlet excited-state driver."""
        return LinearResponseSCF(self, method=ResponseMethod.tddft, **kwargs)

    def TDDFT(self, **kwargs: Any) -> LinearResponseSCF:
        """Full Casida / SSF-RPA Time-Dependent DFT excited-state driver (Direct or RI-TDDFT)."""
        return LinearResponseSCF(self, method=ResponseMethod.tddft, **kwargs)

    def Hessian(self, **kwargs: Any) -> NuclearHessian:
        """Seminumerical molecular nuclear Hessian & harmonic vibrational analysis driver."""
        return NuclearHessian(self, **kwargs)

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
