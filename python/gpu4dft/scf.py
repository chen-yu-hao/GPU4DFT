"""PySCF-style Python frontend wrapping the zero-Python-hot-path C++20/CUDA SCF,
TDDFT/TDA/CIS Linear Response, Analytical Nuclear Gradients, and Nuclear Hessian workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import sys
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
    Shell,
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
_BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


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
    spin: int = 0


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
        becke_radii_adjust: bool = True,
        weight_cutoff: float = 1e-14,
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
        self.becke_radii_adjust = bool(becke_radii_adjust)
        self.weight_cutoff = float(weight_cutoff)


class NuclearGradients:
    """PySCF-like analytical nuclear gradient driver (`mf.nuc_grad_method().kernel()`)."""

    def __init__(self, mf: _SCFBase) -> None:
        self.base = mf
        self.grid_response: bool = True
        self.auxbasis_response: bool = True
        self.de: np.ndarray | None = None
        self.result: GradientResult | None = None

    def kernel(self, dm: np.ndarray | None = None) -> np.ndarray:
        if getattr(self.base, "_with_df", False) and not bool(self.auxbasis_response):
            raise NotImplementedError(
                "NuclearGradients with auxbasis_response=False is not supported for density-fitted "
                "SCF; set auxbasis_response=True (default)"
            )
        if self.base.result is None or self.base._engine is None:
            dm_reuse = dm
            if dm_reuse is None and self.base.result is not None and self.base.converged:
                try:
                    dm_reuse = self.base.make_rdm1()
                except Exception:
                    dm_reuse = None
            self.base.kernel(dm0=dm_reuse)
            if dm_reuse is not None and dm is None and not self.base.converged:
                self.base.kernel(dm0=None)
        assert self.base._engine is not None
        self.base._engine.set_xc_grid_response(bool(self.grid_response))
        gres = self.base._engine.compute_gradient()
        self.result = gres
        self.de = np.asarray(gres.total_grad, dtype=np.float64)
        return self.de

    def grad(self, dm: np.ndarray | None = None) -> np.ndarray:
        return self.kernel(dm=dm)

    def as_scanner(self) -> GradScanner:
        return GradScanner(self)


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
        if self.base.result is None or self.base._engine is None:
            dm_reuse = (
                self.base.make_rdm1()
                if (self.base.result is not None and self.base.converged)
                else None
            )
            self.base.kernel(dm0=dm_reuse)
            if dm_reuse is not None and not self.base.converged:
                self.base.kernel(dm0=None)
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
        if self.base.result is None or self.base._engine is None:
            dm_reuse = (
                self.base.make_rdm1()
                if (self.base.result is not None and self.base.converged)
                else None
            )
            self.base.kernel(dm0=dm_reuse)
            if dm_reuse is not None and not self.base.converged:
                self.base.kernel(dm0=None)
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
    """Shared base class for :class:`RHF`, :class:`RKS`, :class:`UHF`, and :class:`UKS`."""

    def __init__(
        self,
        mol: Any,
        *,
        xc: str = "hf",
        unrestricted: bool = False,
        spin: int | None = None,
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
        dispersion: str | None = None,
        stdout: Any = None,
        verbose: int | None = None,
    ) -> None:
        self.mol = mol
        self.xc = xc
        self.stdout = (getattr(mol, "stdout", None) or sys.stdout) if stdout is None else stdout
        raw_verbose = getattr(mol, "verbose", None) if verbose is None else verbose
        self.verbose = 0 if raw_verbose is None else int(raw_verbose)
        self.unrestricted = bool(unrestricted)
        self.spin = None if spin is None else int(spin)
        if backend is None or (isinstance(backend, str) and backend.strip().lower() == "auto"):
            self.backend = _default_backend()
        else:
            self.backend = _coerce_enum(Backend, backend)
        self.precision = _coerce_enum(Precision, precision)
        self.dispersion = None if dispersion is None else str(dispersion)
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
        self.e_elec: float = 0.0
        self.e_disp: float = 0.0
        self.iterations: int = 0
        self.mo_energy: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
        self.mo_coeff: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
        self.mo_occ: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None
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
        self.result = None
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
        self.result = None
        return self

    def _resolve_spin(self) -> int:
        if self.spin is not None:
            return int(self.spin)
        return int(getattr(self.mol, "spin", 0))

    def _validate_occupations(self, nelec: int, spin: int) -> None:
        if self.unrestricted:
            if nelec < 0 or spin < 0 or spin > nelec or (nelec - spin) % 2 != 0:
                raise ValueError(
                    f"Invalid (nelectron={nelec}, spin={spin}) for unrestricted SCF: "
                    "require 0 <= spin <= nelectron and (nelectron - spin) even"
                )
        else:
            if spin != 0:
                raise ValueError(f"Restricted SCF requires spin=0, got spin={spin}")
            if nelec < 0 or nelec % 2 != 0:
                raise ValueError(
                    f"Restricted SCF requires non-negative even nelectron, got {nelec}"
                )

    def _extract_molecule(self) -> tuple[Basis, list[Atom], int, Basis | None]:
        spin = self._resolve_spin()
        if isinstance(self.mol, MoleculeSpec):
            nelec = int(self.mol.nelectron)
            self._validate_occupations(nelec, spin)
            aux_b = self.mol.aux_basis
            if self._with_df and isinstance(self.auxbasis, Basis):
                aux_b = self.auxbasis
            return self.mol.basis, self.mol.atoms, nelec, aux_b

        if _is_pyscf_mole(self.mol):
            if getattr(self.mol, "_built", True) is False and hasattr(self.mol, "build"):
                self.mol.build()
            nelec = int(self.mol.nelectron)
            self._validate_occupations(nelec, spin)
            basis = basis_from_pyscf(self.mol)
            atoms = atoms_from_pyscf(self.mol)
            aux_b: Basis | None = None
            if self._with_df:
                if self.auxbasis is not None and not isinstance(self.auxbasis, (str, dict, Basis)):
                    raise TypeError(
                        "auxbasis must be None, str, dict, or gpu4dft.Basis when density_fit is enabled"
                    )
                if isinstance(self.auxbasis, str) and not self.auxbasis.strip():
                    raise ValueError("auxbasis must not be an empty string")
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
        spin = self._resolve_spin()
        wopts.xc = self.xc
        wopts.dispersion = "" if self.dispersion is None else str(self.dispersion)
        wopts.jk_mode = jk_mode
        wopts.unrestricted = bool(self.unrestricted)
        wopts.spin = int(spin)
        wopts.schwarz_cutoff = float(self.schwarz_cutoff)
        wopts.fp32_threshold = float(self.fp32_threshold)

        sopts = wopts.scf
        sopts.backend = be
        sopts.solver = solv
        sopts.unrestricted = bool(self.unrestricted)
        sopts.spin = int(spin)
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
        gopts.becke_radii_adjust = bool(self.grids.becke_radii_adjust)
        gopts.weight_cutoff = float(self.grids.weight_cutoff)
        gopts.backend = be
        wopts.grid = gopts

        xopts = wopts.xc_options
        xopts.backend = be
        xopts.spin_polarized = bool(self.unrestricted)
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

    @staticmethod
    def _mol_signature(mol: Any) -> tuple:
        if isinstance(mol, MoleculeSpec):
            return (
                "MoleculeSpec",
                id(mol),
                id(mol.basis),
                int(mol.basis.nao),
                int(mol.basis.nshell),
                id(mol.aux_basis),
                int(mol.nelectron),
                int(mol.spin),
                tuple(int(a.atomic_number) for a in mol.atoms),
                np.asarray([a.center for a in mol.atoms], dtype=np.float64).tobytes(),
            )
        if _is_pyscf_mole(mol):
            if getattr(mol, "_built", True) is False and hasattr(mol, "build"):
                mol.build()
            return (
                "Mole",
                id(mol),
                int(mol.nbas),
                int(mol.natm),
                int(mol.nelectron),
                int(getattr(mol, "charge", 0)),
                int(getattr(mol, "spin", 0)),
                bool(getattr(mol, "cart", False)),
                tuple(int(mol.atom_charge(i)) for i in range(mol.natm)),
                np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64).tobytes(),
                repr(getattr(mol, "basis", None)),
            )
        return ("Other", id(mol), type(mol).__name__)

    @staticmethod
    def _arr_signature(arr: Any) -> Any:
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return (arr.shape, int(arr.__array_interface__["data"][0]))
        return ("obj", id(arr))

    def _config_signature(self) -> tuple:
        g_id = (
            self._arr_signature(self.grids.coords),
            self._arr_signature(self.grids.weights),
        )
        if isinstance(self.auxbasis, Basis):
            aux_sig: Any = ("Basis", id(self.auxbasis))
        elif isinstance(self.auxbasis, dict):
            aux_sig = ("dict", tuple(sorted((str(k), repr(v)) for k, v in self.auxbasis.items())))
        else:
            aux_sig = repr(self.auxbasis)
        return (
            self._mol_signature(self.mol),
            self.xc,
            self.unrestricted,
            self._resolve_spin(),
            self.dispersion,
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
            float(self.grids.ao_cutoff),
            bool(self.grids.becke_radii_adjust),
            float(self.grids.weight_cutoff),
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
            aux_sig,
            self._with_snk,
            self.snk_overlap_fitting,
            self.snk_pair_cutoff,
            self.snk_lindep_tol,
            str(self.snk_precision),
        )

    def release(self) -> None:
        """Release persistent C++/GPU engines while retaining host SCF results."""
        self._engine = None
        self._setup_sig = None

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
            return minao_init_guess(self.mol, unrestricted=self.unrestricted)
        return None

    def kernel(self, dm0: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None) -> float:
        """Execute the entire SCF loop in C++20/CUDA and return converged total energy (Hartree)."""
        self.setup()
        assert self._engine is not None

        nao = int(self._engine.nao)
        if dm0 is None:
            init_dm = self.get_init_guess()
        elif self.unrestricted:
            if isinstance(dm0, (tuple, list)):
                if len(dm0) != 2:
                    raise ValueError(
                        f"UHF/UKS dm0 tuple/list must have length 2, got {len(dm0)}"
                    )
                dm_a = np.asarray(dm0[0], dtype=np.float64)
                dm_b = np.asarray(dm0[1], dtype=np.float64)
                if dm_a.shape != (nao, nao) or dm_b.shape != (nao, nao):
                    raise ValueError(
                        f"UHF/UKS dm0 spin matrices must have shape ({nao}, {nao}), "
                        f"got {dm_a.shape} and {dm_b.shape}"
                    )
                init_dm = np.ascontiguousarray(np.stack([dm_a, dm_b], axis=0), dtype=np.float64)
            else:
                init_dm = np.ascontiguousarray(dm0, dtype=np.float64)
                if init_dm.ndim != 3 or init_dm.shape != (2, nao, nao):
                    raise ValueError(
                        f"UHF/UKS dm0 must have shape (2, {nao}, {nao}), got {init_dm.shape}"
                    )
        else:
            init_dm = np.ascontiguousarray(dm0, dtype=np.float64)
            if init_dm.ndim != 2 or init_dm.shape != (nao, nao):
                raise ValueError(
                    f"RHF/RKS dm0 must have shape ({nao}, {nao}), got {init_dm.shape}"
                )

        wres = self._engine.run(dm0=init_dm)
        self.result = wres
        sres = wres.scf
        self.converged = bool(sres.converged)
        self.e_tot = float(sres.total_energy)
        self.e_elec = float(sres.electronic_energy)
        self.e_disp = float(sres.dispersion_energy)
        self.iterations = int(sres.iterations)
        if self.unrestricted:
            self.mo_energy = (
                np.asarray(sres.mo_energies, dtype=np.float64),
                np.asarray(sres.mo_energies_beta, dtype=np.float64),
            )
            self.mo_coeff = (
                np.asarray(sres.mo_coefficients, dtype=np.float64),
                np.asarray(sres.mo_coefficients_beta, dtype=np.float64),
            )
            self.mo_occ = (
                np.asarray(sres.mo_occ_alpha, dtype=np.float64),
                np.asarray(sres.mo_occ_beta, dtype=np.float64),
            )
        else:
            self.mo_energy = np.asarray(sres.mo_energies, dtype=np.float64)
            self.mo_coeff = np.asarray(sres.mo_coefficients, dtype=np.float64)
            nmo = int(sres.nmo)
            occ = np.zeros(nmo, dtype=np.float64)
            occ[: int(self._engine.n_elec) // 2] = 2.0
            self.mo_occ = occ
        self.scf_summary = {
            "e1": float(sres.one_electron_energy),
            "coul_exch": float(sres.two_electron_energy),
            "exc": float(sres.xc_energy),
            "e_elec": float(sres.electronic_energy),
            "e_disp": float(sres.dispersion_energy),
            "nuc": float(sres.nuclear_repulsion),
            "e_tot": float(sres.total_energy),
            "setup_time_s": float(wres.setup_time_s),
            "scf_time_s": float(wres.scf_time_s),
        }
        if self.unrestricted:
            self.scf_summary["s2"] = float(sres.s2)
            self.scf_summary["multiplicity"] = float(sres.multiplicity)
        return self.e_tot

    def run(self, dm0: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None) -> _SCFBase:
        """PySCF-compatible convenience wrapper executing `.kernel()` and returning `self`."""
        self.kernel(dm0=dm0)
        return self

    def make_rdm1(self, mo_coeff: Any = None, mo_occ: Any = None) -> np.ndarray:
        if mo_coeff is not None or mo_occ is not None:
            c = self.mo_coeff if mo_coeff is None else mo_coeff
            o = self.mo_occ if mo_occ is None else mo_occ
            if c is None or o is None:
                raise RuntimeError("MO coefficients/occupations unavailable; call .kernel() first")
            if self.unrestricted:
                ca = np.asarray(c[0], dtype=np.float64)
                cb = np.asarray(c[1], dtype=np.float64)
                oa = np.asarray(o[0], dtype=np.float64)
                ob = np.asarray(o[1], dtype=np.float64)
                return np.stack([(ca * oa) @ ca.T, (cb * ob) @ cb.T], axis=0)
            c_arr = np.asarray(c, dtype=np.float64)
            o_arr = np.asarray(o, dtype=np.float64)
            return (c_arr * o_arr) @ c_arr.T

        if self.result is None:
            raise RuntimeError("SCF has not been run yet; call .kernel() first")
        sres = self.result.scf
        if self.unrestricted:
            return np.stack(
                [
                    np.asarray(sres.density_matrix_alpha, dtype=np.float64),
                    np.asarray(sres.density_matrix_beta, dtype=np.float64),
                ],
                axis=0,
            )
        return np.asarray(sres.density_matrix, dtype=np.float64)

    def make_rdm1e(self) -> np.ndarray:
        if self.result is None:
            raise RuntimeError("SCF has not been run yet; call .kernel() first")
        sres = self.result.scf
        if self.unrestricted:
            return np.stack(
                [
                    np.asarray(sres.energy_weighted_dm_alpha, dtype=np.float64),
                    np.asarray(sres.energy_weighted_dm_beta, dtype=np.float64),
                ],
                axis=0,
            )
        return np.asarray(sres.energy_weighted_dm, dtype=np.float64)

    def spin_square(
        self, mo_coeff: tuple[np.ndarray, np.ndarray] | None = None, s: np.ndarray | None = None
    ) -> tuple[float, float]:
        """Return ``(<S^2>, 2S+1)`` matching PySCF ``uhf.spin_square()``."""
        if not self.unrestricted:
            return (0.0, 1.0)
        if mo_coeff is None:
            if self.result is None:
                raise RuntimeError("SCF has not been run yet; call .kernel() first")
            sres = self.result.scf
            return (float(sres.s2), float(sres.multiplicity))
        ovlp = self.get_ovlp() if s is None else np.asarray(s, dtype=np.float64)
        assert self.mo_occ is not None
        ca = np.asarray(mo_coeff[0], dtype=np.float64)[:, np.asarray(self.mo_occ[0]) > 0]
        cb = np.asarray(mo_coeff[1], dtype=np.float64)[:, np.asarray(self.mo_occ[1]) > 0]
        na = ca.shape[1]
        nb = cb.shape[1]
        sz = 0.5 * (na - nb)
        if na > 0 and nb > 0:
            s_ab = ca.T @ ovlp @ cb
            s2 = sz * (sz + 1.0) + nb - float(np.sum(s_ab * s_ab))
        else:
            s2 = sz * (sz + 1.0) + nb
        mult = math.sqrt(1.0 + 4.0 * max(0.0, s2))
        return (float(s2), float(mult))

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
        sres = self.result.scf
        if self.unrestricted:
            return np.stack(
                [
                    np.asarray(sres.fock_matrix_alpha, dtype=np.float64),
                    np.asarray(sres.fock_matrix_beta, dtype=np.float64),
                ],
                axis=0,
            )
        return np.asarray(sres.fock_matrix, dtype=np.float64)

    def dipole_moment(
        self,
        unit: str = "Debye",
        origin: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Compute permanent molecular dipole moment in `'Debye'` (default) or `'AU'` (`e*Bohr`)."""
        if self.result is None:
            self.kernel()
        orig = (float(origin[0]), float(origin[1]), float(origin[2]))
        if self._engine is not None:
            mu_au = np.asarray(self._engine.compute_dipole(orig), dtype=np.float64)
        else:
            assert self.result is not None
            basis, atoms, _, _ = self._extract_molecule()
            dm_tot = np.ascontiguousarray(self.result.scf.density_matrix, dtype=np.float64)
            mu_au = np.asarray(
                _core.molecular_dipole_moment(
                    basis, atoms, dm_tot, orig, _coerce_enum(Backend, self.backend)
                ),
                dtype=np.float64,
            )
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

    def as_scanner(self) -> SCFScanner:
        return SCFScanner(self)


_PYSCF_SCF_SCANNER_CACHE: dict[type, type] = {}
_PYSCF_GRAD_SCANNER_CACHE: dict[type, type] = {}


def _resolve_pyscf_scanner_cls(
    base_cls: type, attr_name: str, cache: dict[type, type]
) -> type:
    try:
        from pyscf import lib

        pyscf_mixin = getattr(lib, attr_name, None)
    except ImportError:
        return base_cls
    if pyscf_mixin is None or issubclass(base_cls, pyscf_mixin):
        return base_cls
    cached = cache.get(pyscf_mixin)
    if cached is not None:
        return cached

    class _PySCFBackedScanner(base_cls, pyscf_mixin):
        pass

    _PySCFBackedScanner.__name__ = base_cls.__name__
    _PySCFBackedScanner.__qualname__ = base_cls.__qualname__
    _PySCFBackedScanner.__module__ = base_cls.__module__
    cache[pyscf_mixin] = _PySCFBackedScanner
    return _PySCFBackedScanner


def _clone_mol(mol: Any) -> Any:
    if isinstance(mol, MoleculeSpec):
        return MoleculeSpec(
            basis=mol.basis,
            atoms=[Atom(int(a.atomic_number), list(a.center)) for a in mol.atoms],
            nelectron=int(mol.nelectron),
            aux_basis=mol.aux_basis,
            spin=int(mol.spin),
        )
    if _is_pyscf_mole(mol):
        if getattr(mol, "_built", True) is False and hasattr(mol, "build"):
            mol.build()
        return mol.copy()
    return mol


def _shift_basis_centers(
    basis: Basis,
    old_atoms: list[Atom],
    new_centers_bohr: np.ndarray,
) -> Basis:
    old_centers = np.asarray([a.center for a in old_atoms], dtype=np.float64)
    new_shells: list[Shell] = []
    for sh in basis.shells:
        sc = np.asarray(sh.center, dtype=np.float64)
        dists = np.linalg.norm(old_centers - sc, axis=1)
        idx = int(np.argmin(dists))
        if float(dists[idx]) > 1e-6:
            raise ValueError("Cannot map shell center to an atom in MoleculeSpec")
        nc = [
            float(new_centers_bohr[idx, 0]),
            float(new_centers_bohr[idx, 1]),
            float(new_centers_bohr[idx, 2]),
        ]
        raw_coeffs: list[float] = []
        for k, exp_k in enumerate(sh.exponents):
            unit_basis = Basis([Shell(int(sh.l), [0.0, 0.0, 0.0], [float(exp_k)], [1.0])])
            norm_k = float(unit_basis.shells[0].coefficients[0])
            raw_coeffs.append(float(sh.coefficients[k]) / norm_k)
        new_shells.append(Shell(int(sh.l), nc, list(sh.exponents), raw_coeffs))
    return Basis(new_shells)


def _shift_molecule_spec(
    spec: MoleculeSpec,
    new_centers_bohr: np.ndarray,
    *,
    charge_override: int | None = None,
    spin_override: int | None = None,
) -> MoleculeSpec:
    new_atoms = [
        Atom(
            int(a.atomic_number),
            [
                float(new_centers_bohr[i, 0]),
                float(new_centers_bohr[i, 1]),
                float(new_centers_bohr[i, 2]),
            ],
        )
        for i, a in enumerate(spec.atoms)
    ]
    new_basis = _shift_basis_centers(spec.basis, spec.atoms, new_centers_bohr)
    new_aux = (
        None
        if spec.aux_basis is None
        else _shift_basis_centers(spec.aux_basis, spec.atoms, new_centers_bohr)
    )
    nelec = int(spec.nelectron)
    if charge_override is not None:
        nelec = int(sum(int(a.atomic_number) for a in spec.atoms) - int(charge_override))
    spin = int(spec.spin) if spin_override is None else int(spin_override)
    return MoleculeSpec(
        basis=new_basis,
        atoms=new_atoms,
        nelectron=nelec,
        aux_basis=new_aux,
        spin=spin,
    )


class SCFScanner(_SCFBase):
    """PySCF-compatible single-point SCF scanner (`mf.as_scanner()`)."""

    def __new__(cls, *args: Any, **kwargs: Any) -> SCFScanner:
        if cls is SCFScanner:
            resolved = _resolve_pyscf_scanner_cls(
                SCFScanner, "SinglePointScanner", _PYSCF_SCF_SCANNER_CACHE
            )
            return super().__new__(resolved)
        return super().__new__(cls)

    def __init__(self, base: _SCFBase) -> None:
        if not isinstance(base, _SCFBase):
            raise TypeError(f"SCFScanner expects an _SCFBase instance, got {type(base).__name__}")
        mol_copy = _clone_mol(base.mol)
        grids_copy = GridsConfig(
            level=base.grids.level,
            coords=None,
            weights=None,
            n_rad=base.grids.n_rad,
            n_ang=base.grids.n_ang,
            prune=base.grids.prune,
            partition=base.grids.partition,
            max_batch_points=base.grids.max_batch_points,
            ao_cutoff=base.grids.ao_cutoff,
            becke_radii_adjust=base.grids.becke_radii_adjust,
            weight_cutoff=base.grids.weight_cutoff,
        )
        super().__init__(
            mol_copy,
            xc=base.xc,
            unrestricted=base.unrestricted,
            spin=base.spin,
            backend=base.backend,
            precision=base.precision,
            dynamic_precision=base.dynamic_precision,
            gemm_mode=base.gemm_mode,
            early_gemm_mode=base.early_gemm_mode,
            solver=base.solver,
            incremental=base.incremental,
            use_cuda_graph=base.use_cuda_graph,
            grids=grids_copy,
            conv_tol=base.conv_tol,
            diis_tol=base.diis_tol,
            max_cycle=base.max_cycle,
            diis_space=base.diis_space,
            damping=base.damping,
            level_shift=base.level_shift,
            lindep_tol=base.lindep_tol,
            schwarz_cutoff=base.schwarz_cutoff,
            fp32_threshold=base.fp32_threshold,
            ozaki_slices=base.ozaki_slices,
            init_guess=base.init_guess,
            dispersion=base.dispersion,
            stdout=base.stdout,
            verbose=base.verbose,
        )
        self._with_df = bool(base._with_df)
        self.auxbasis = dict(base.auxbasis) if isinstance(base.auxbasis, dict) else base.auxbasis
        self._with_snk = bool(base._with_snk)
        self.snk_overlap_fitting = bool(base.snk_overlap_fitting)
        self.snk_pair_cutoff = float(base.snk_pair_cutoff)
        self.snk_lindep_tol = float(base.snk_lindep_tol)
        self.snk_precision = base.snk_precision

        self._prev_dm: np.ndarray | None = None
        self._prev_key: tuple | None = None
        self._prev_converged: bool = False
        self._last_nao: int | None = None

    def _build_workflow_options(self) -> WorkflowOptions:
        wopts = super()._build_workflow_options()
        if self.conv_tol <= 1e-10:
            sopts = wopts.scf
            sopts.diis_tol = min(float(self.diis_tol), max(1e-10, float(self.conv_tol) * 10.0))
            wopts.scf = sopts
        return wopts

    def _mol_natm_and_z(self) -> tuple[int, tuple[int, ...]]:
        if isinstance(self.mol, MoleculeSpec):
            zs = tuple(int(a.atomic_number) for a in self.mol.atoms)
            return len(zs), zs
        if _is_pyscf_mole(self.mol):
            if getattr(self.mol, "_built", True) is False and hasattr(self.mol, "build"):
                self.mol.build()
            zs = tuple(int(self.mol.atom_charge(i)) for i in range(self.mol.natm))
            return int(self.mol.natm), zs
        raise TypeError("Scanner base mol must be a pyscf.gto.Mole or gpu4dft.MoleculeSpec")

    def _apply_coords_to_mol(
        self,
        coords_or_atom_spec: Any,
        *,
        is_angstrom: bool = True,
        charge_override: int | None = None,
        spin_override: int | None = None,
    ) -> Any:
        if isinstance(self.mol, MoleculeSpec):
            arr = np.asarray(coords_or_atom_spec, dtype=np.float64)
            centers_bohr = arr * _BOHR_PER_ANGSTROM if is_angstrom else arr
            return _shift_molecule_spec(
                self.mol,
                centers_bohr,
                charge_override=charge_override,
                spin_override=spin_override,
            )
        m = self.mol.copy()
        if charge_override is not None:
            m.charge = int(charge_override)
        if spin_override is not None:
            m.spin = int(spin_override)
        try:
            m.set_geom_(coords_or_atom_spec, inplace=True)
        except (IndexError, KeyError, RuntimeError) as exc:
            raise ValueError(f"Invalid geometry specification: {exc}") from exc
        return m

    def _coerce_mol(self, mol_or_geom: Any) -> Any:
        if mol_or_geom is None:
            raise TypeError("mol_or_geom must not be None")
        if isinstance(mol_or_geom, (bool, int, float, bytes, bytearray, dict)):
            raise TypeError(f"Unsupported mol_or_geom type: {type(mol_or_geom).__name__}")
        if _is_pyscf_mole(mol_or_geom):
            if getattr(mol_or_geom, "_built", True) is False and hasattr(mol_or_geom, "build"):
                mol_or_geom.build()
            return mol_or_geom
        if isinstance(mol_or_geom, MoleculeSpec):
            return mol_or_geom

        natm, expected_z = self._mol_natm_and_z()

        if hasattr(mol_or_geom, "get_atomic_numbers") and hasattr(mol_or_geom, "get_positions"):
            z_arr = np.asarray(mol_or_geom.get_atomic_numbers(), dtype=int)
            pos_ang = np.asarray(mol_or_geom.get_positions(), dtype=np.float64)
            if z_arr.shape != (natm,) or tuple(int(z) for z in z_arr) != expected_z:
                raise ValueError(
                    f"Atomic numbers mismatch: expected {expected_z}, got {tuple(int(z) for z in z_arr)}"
                )
            if pos_ang.ndim != 2 or pos_ang.shape != (natm, 3) or not np.all(np.isfinite(pos_ang)):
                raise ValueError(
                    f"Atoms positions must be finite with shape ({natm}, 3), got {pos_ang.shape}"
                )
            info = getattr(mol_or_geom, "info", {}) or {}
            q_ov = None if info.get("charge") is None else int(info["charge"])
            s_ov = None if info.get("spin") is None else int(info["spin"])
            if isinstance(self.mol, MoleculeSpec):
                return self._apply_coords_to_mol(
                    pos_ang, is_angstrom=True, charge_override=q_ov, spin_override=s_ov
                )
            m = self.mol.copy()
            if q_ov is not None:
                m.charge = q_ov
            if s_ov is not None:
                m.spin = s_ov
            try:
                m.set_geom_(pos_ang, unit="Angstrom", inplace=True)
            except (IndexError, KeyError, RuntimeError) as exc:
                raise ValueError(f"Invalid Atoms geometry: {exc}") from exc
            return m

        if isinstance(mol_or_geom, str):
            text = mol_or_geom.strip()
            if not text:
                raise ValueError("Geometry string must not be empty")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            if len(lines) == 1 and ";" in lines[0]:
                lines = [ln.strip() for ln in lines[0].split(";") if ln.strip()]
            if (
                len(lines) >= 2
                and lines[0].isdigit()
                and int(lines[0]) == natm
                and len(lines) == natm + 2
            ):
                lines = lines[2:]
            if len(lines) != natm:
                raise ValueError(
                    f"Expected {natm} atom lines in geometry string, got {len(lines)}"
                )
            coords_parsed: list[list[float]] = []
            for ln in lines:
                parts = ln.split()
                if len(parts) != 4:
                    raise ValueError(f"Invalid atom line in geometry string: {ln!r}")
                try:
                    xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                except ValueError as exc:
                    raise ValueError(f"Non-numeric coordinates in atom line: {ln!r}") from exc
                if not all(math.isfinite(v) for v in xyz):
                    raise ValueError(f"Non-finite coordinates in atom line: {ln!r}")
                coords_parsed.append(xyz)
            if isinstance(self.mol, MoleculeSpec):
                return self._apply_coords_to_mol(
                    np.asarray(coords_parsed, dtype=np.float64), is_angstrom=True
                )
            return self._apply_coords_to_mol("\n".join(lines))

        if isinstance(mol_or_geom, (np.ndarray, list, tuple)):
            try:
                arr = np.asarray(mol_or_geom, dtype=np.float64)
            except (TypeError, ValueError):
                arr = None
            if arr is not None:
                if arr.ndim != 2 or arr.shape != (natm, 3) or not np.all(np.isfinite(arr)):
                    raise ValueError(
                        f"Coordinate array must be finite with shape ({natm}, 3), got {arr.shape}"
                    )
                return self._apply_coords_to_mol(arr, is_angstrom=False)
            if isinstance(mol_or_geom, (list, tuple)) and len(mol_or_geom) == natm:
                coords_list: list[list[float]] = []
                for entry in mol_or_geom:
                    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                        raise ValueError(f"Invalid atom entry in geometry list: {entry!r}")
                    xyz_seq = entry[1]
                    if not isinstance(xyz_seq, (list, tuple, np.ndarray)) or len(xyz_seq) != 3:
                        raise ValueError(f"Invalid coordinates in atom entry: {entry!r}")
                    xyz_vals = [float(v) for v in xyz_seq]
                    if not all(math.isfinite(v) for v in xyz_vals):
                        raise ValueError(f"Non-finite coordinates in atom entry: {entry!r}")
                    coords_list.append(xyz_vals)
                if isinstance(self.mol, MoleculeSpec):
                    return self._apply_coords_to_mol(
                        np.asarray(coords_list, dtype=np.float64), is_angstrom=False
                    )
                return self._apply_coords_to_mol(mol_or_geom)
            raise ValueError("Invalid geometry list/tuple/array")

        raise TypeError(f"Unsupported mol_or_geom type: {type(mol_or_geom).__name__}")

    def _dm_key_for_mol(self, mol: Any) -> tuple[int, tuple]:
        if isinstance(mol, MoleculeSpec):
            nao = int(mol.basis.nao)
            nelec = int(mol.nelectron)
            spin = int(self.spin) if self.spin is not None else int(mol.spin)
            atom_z = tuple(int(a.atomic_number) for a in mol.atoms)
            basis_sig: Any = ("MoleculeSpec", nao, int(mol.basis.nshell), bool(mol.aux_basis is not None))
        else:
            nao = int(mol.nao_nr())
            nelec = int(mol.nelectron)
            spin = int(self.spin) if self.spin is not None else int(getattr(mol, "spin", 0))
            atom_z = tuple(int(mol.atom_charge(i)) for i in range(mol.natm))
            basis_sig = ("Mole", repr(getattr(mol, "basis", None)), bool(getattr(mol, "cart", False)))
        if isinstance(self.auxbasis, Basis):
            aux_sig: Any = ("Basis", id(self.auxbasis))
        elif isinstance(self.auxbasis, dict):
            aux_sig = ("dict", tuple(sorted((str(k), repr(v)) for k, v in self.auxbasis.items())))
        else:
            aux_sig = repr(self.auxbasis)
        key = (
            nao,
            bool(self.unrestricted),
            nelec,
            spin,
            atom_z,
            basis_sig,
            self.xc,
            bool(self._with_df),
            aux_sig,
        )
        return nao, key

    def __call__(self, mol_or_geom: Any) -> float:
        mol = self._coerce_mol(mol_or_geom)
        cur_nao, cur_key = self._dm_key_for_mol(mol)
        dm0 = (
            self._prev_dm
            if (self._prev_converged and self._prev_dm is not None and self._prev_key == cur_key)
            else None
        )
        prev_mol = self.mol
        self.release()
        if self._last_nao is not None and self._last_nao != cur_nao:
            trim_fn = getattr(_core, "trim_device_pool", None)
            if trim_fn is not None:
                trim_fn()
        self.mol = mol
        try:
            e_tot = self.kernel(dm0=dm0)
            if dm0 is not None and not self.converged:
                e_tot = self.kernel(dm0=None)
        except Exception:
            self.mol = prev_mol
            self.release()
            raise
        self._last_nao = cur_nao
        if self.converged:
            self._prev_dm = self.make_rdm1()
            self._prev_key = cur_key
            self._prev_converged = True
        else:
            self._prev_dm = None
            self._prev_key = None
            self._prev_converged = False
        return float(e_tot)


class GradScanner:
    """PySCF-compatible analytical nuclear gradient scanner (`mf.nuc_grad_method().as_scanner()`)."""

    def __new__(cls, *args: Any, **kwargs: Any) -> GradScanner:
        if cls is GradScanner:
            resolved = _resolve_pyscf_scanner_cls(
                GradScanner, "GradScanner", _PYSCF_GRAD_SCANNER_CACHE
            )
            return super().__new__(resolved)
        return super().__new__(cls)

    def __init__(self, grad_method: NuclearGradients) -> None:
        if not isinstance(grad_method, NuclearGradients):
            raise TypeError(
                f"GradScanner expects a NuclearGradients instance, got {type(grad_method).__name__}"
            )
        self.base = grad_method.base.as_scanner()
        self.grid_response: bool = bool(grad_method.grid_response)
        self.auxbasis_response: bool = bool(grad_method.auxbasis_response)
        self.e_tot: float = 0.0
        self.de: np.ndarray | None = None
        self.result: GradientResult | None = None
        self._converged: bool = False

    @property
    def converged(self) -> bool:
        return bool(self._converged)

    @converged.setter
    def converged(self, value: bool) -> None:
        self._converged = bool(value)

    @property
    def mol(self) -> Any:
        return self.base.mol

    @mol.setter
    def mol(self, value: Any) -> None:
        self.base.mol = value

    @property
    def stdout(self) -> Any:
        return self.base.stdout

    @stdout.setter
    def stdout(self, value: Any) -> None:
        self.base.stdout = value

    @property
    def verbose(self) -> int:
        return self.base.verbose

    @verbose.setter
    def verbose(self, value: int) -> None:
        self.base.verbose = int(value)

    def __call__(self, mol_or_geom: Any) -> tuple[float, np.ndarray]:
        e_tot = self.base(mol_or_geom)
        grad_drv = self.base.nuc_grad_method()
        grad_drv.grid_response = bool(self.grid_response)
        grad_drv.auxbasis_response = bool(self.auxbasis_response)
        try:
            de = grad_drv.kernel()
        finally:
            self.base.release()
        self.result = grad_drv.result
        self.e_tot = float(e_tot)
        self.de = np.asarray(de, dtype=np.float64)
        self.converged = bool(self.base.converged)
        return self.e_tot, self.de


class RHF(_SCFBase):
    """Restricted Hartree-Fock driver executing end-to-end inside C++20/CUDA."""

    def __init__(self, mol: Any, **kwargs: Any) -> None:
        kwargs.pop("xc", None)
        kwargs.pop("unrestricted", None)
        super().__init__(mol, xc="hf", unrestricted=False, **kwargs)


class RKS(_SCFBase):
    """Restricted Kohn-Sham / Hybrid DFT driver executing end-to-end inside C++20/CUDA."""

    def __init__(self, mol: Any, xc: str = "b3lyp", **kwargs: Any) -> None:
        kwargs.pop("unrestricted", None)
        super().__init__(mol, xc=xc, unrestricted=False, **kwargs)


class UHF(_SCFBase):
    """Unrestricted Hartree-Fock driver executing end-to-end inside C++20/CUDA."""

    def __init__(self, mol: Any, **kwargs: Any) -> None:
        kwargs.pop("xc", None)
        kwargs.pop("unrestricted", None)
        super().__init__(mol, xc="hf", unrestricted=True, **kwargs)


class UKS(_SCFBase):
    """Unrestricted Kohn-Sham / Hybrid DFT driver executing end-to-end inside C++20/CUDA."""

    def __init__(self, mol: Any, xc: str = "b3lyp", **kwargs: Any) -> None:
        kwargs.pop("unrestricted", None)
        super().__init__(mol, xc=xc, unrestricted=True, **kwargs)
