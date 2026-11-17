"""GPU4DFT: GPU-native quantum chemistry and density functional theory.

Unimplemented features raise a subclass of the builtin ``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np

from . import _core
from ._core import (
    Atom,
    Backend,
    Basis,
    DensitySolver,
    DirectJKEngine,
    GemmBenchmarkResult,
    GemmMode,
    GradientResult,
    GridOptions,
    JKMode,
    JKOptions,
    MolecularGrid,
    MolecularScf,
    PartitionScheme,
    Precision,
    RIJKEngine,
    ScfIterationLog,
    ScfOptions,
    ScfResult,
    SemiNumericalK,
    Shell,
    SnkOptions,
    Sp2Options,
    WorkflowOptions,
    WorkflowResult,
    XcEngine,
    XcOptions,
    XcResult,
    kernel_names,
    nuclear_repulsion_energy,
    nuclear_repulsion_grad,
    run_grad,
    run_scf,
    xc_exact_exchange_fraction,
)
from .ase_calculator import GPU4DFTCalculator, SimpleAtoms
from .pyscf_interface import (
    atoms_from_pyscf,
    aux_basis_from_pyscf,
    basis_from_pyscf,
    minao_init_guess,
)
from .scf import GridsConfig, MoleculeSpec, NuclearGradients, RHF, RKS

__all__ = [
    "Atom",
    "Backend",
    "Basis",
    "DensitySolver",
    "DirectJKEngine",
    "GPU4DFTCalculator",
    "GemmBenchmarkResult",
    "GemmMode",
    "GradientResult",
    "GridOptions",
    "GridsConfig",
    "JKMode",
    "JKOptions",
    "MolecularGrid",
    "MolecularScf",
    "MoleculeSpec",
    "NuclearGradients",
    "PartitionScheme",
    "Precision",
    "RHF",
    "RIJKEngine",
    "RKS",
    "ScfIterationLog",
    "ScfOptions",
    "ScfResult",
    "SemiNumericalK",
    "Shell",
    "SimpleAtoms",
    "SnkOptions",
    "Sp2Options",
    "WorkflowOptions",
    "WorkflowResult",
    "XcEngine",
    "XcOptions",
    "XcResult",
    "atoms_from_pyscf",
    "aux_basis_from_pyscf",
    "backend_available",
    "basis_from_pyscf",
    "benchmark_gemm",
    "eri_tensor",
    "eval_ao_grid",
    "gemm",
    "jk_nuclear_grad",
    "kernel_names",
    "kinetic_matrix",
    "minao_init_guess",
    "nuclear_attraction_matrix",
    "nuclear_repulsion_energy",
    "nuclear_repulsion_grad",
    "one_electron_grad",
    "overlap_matrix",
    "run_grad",
    "run_scf",
    "xc_exact_exchange_fraction",
]


def _as_enum(enum_type, value):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        key = value.strip().lower().replace("-", "_")
        try:
            return enum_type[key]
        except KeyError:
            pass
    try:
        return enum_type[value]
    except KeyError:
        names = ", ".join(member.name for member in enum_type)
        raise ValueError(f"unknown {enum_type.__name__} {value!r}; choose from {names}") from None


def backend_available(backend: Backend | str) -> bool:
    """True if ``backend`` is compiled in and has a usable device."""
    return _core.backend_available(_as_enum(Backend, backend))


def eri_tensor(
    basis: Basis, backend: Backend | str = "host", precision: Precision | str = "fp64"
) -> np.ndarray:
    """Full (ij|kl) ERI tensor as a (nao, nao, nao, nao) NumPy array.

    Intended for validation on small systems; production code paths contract
    integrals on the fly instead of storing them.
    """
    return _core.eri_tensor(basis, _as_enum(Backend, backend), _as_enum(Precision, precision))


def overlap_matrix(basis: Basis, backend: Backend | str = "host") -> np.ndarray:
    """Analytical 1e overlap matrix S_{uv} = <u | v>, shape (nao, nao)."""
    return _core.overlap_matrix(basis, _as_enum(Backend, backend))


def kinetic_matrix(basis: Basis, backend: Backend | str = "host") -> np.ndarray:
    """Analytical 1e kinetic matrix T_{uv} = <u | -1/2 nabla^2 | v>, shape (nao, nao)."""
    return _core.kinetic_matrix(basis, _as_enum(Backend, backend))


def nuclear_attraction_matrix(
    basis: Basis, atoms: list[Atom], backend: Backend | str = "host"
) -> np.ndarray:
    """Analytical 1e nuclear attraction matrix V_{uv}, shape (nao, nao)."""
    return _core.nuclear_attraction_matrix(basis, atoms, _as_enum(Backend, backend))


def one_electron_grad(
    basis: Basis,
    atoms: list[Atom],
    dm: np.ndarray,
    wdm: np.ndarray,
    backend: Backend | str = "cuda",
) -> np.ndarray:
    """Analytical 1e nuclear gradient (including Pulay -Tr(W dS/dR)), shape (natom, 3)."""
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    wdm_c = np.ascontiguousarray(wdm, dtype=np.float64)
    return _core.one_electron_grad(basis, atoms, dm_c, wdm_c, _as_enum(Backend, backend))


def jk_nuclear_grad(
    basis: Basis,
    dm: np.ndarray,
    k_fraction: float = 1.0,
    backend: Backend | str = "cuda",
    schwarz_cutoff: float = 1e-13,
    atoms: list[Atom] | None = None,
) -> np.ndarray:
    """Analytical 2e Coulomb/Exchange nuclear gradient, shape (natom, 3) in Hartree/Bohr."""
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    return _core.jk_nuclear_grad(
        basis, dm_c, float(k_fraction), _as_enum(Backend, backend), float(schwarz_cutoff), atoms
    )


def eval_ao_grid(
    basis: Basis,
    coords: np.ndarray,
    deriv: int = 0,
    backend: Backend | str = "host",
) -> np.ndarray:
    """Evaluate spherical AOs (and derivatives up to ``deriv`` in {0, 1, 2}) on ``coords`` [N, 3]."""
    c_arr = np.ascontiguousarray(coords, dtype=np.float64)
    return _core.eval_ao_grid(basis, c_arr, int(deriv), _as_enum(Backend, backend))


def benchmark_gemm(
    n: int,
    warmup_iters: int = 2,
    bench_iters: int = 5,
    backend: Backend | str = "cuda",
    ozaki_slices: int = 8,
) -> GemmBenchmarkResult:
    """Benchmark FP64 vs Ozaki INT8 Tensor Core vs FP32 GEMM TFLOPS and error."""
    return _core.benchmark_gemm(
        int(n), int(warmup_iters), int(bench_iters), _as_enum(Backend, backend), int(ozaki_slices)
    )


def gemm(
    A: np.ndarray,
    B: np.ndarray,
    transA: bool = False,
    transB: bool = False,
    alpha: float = 1.0,
    beta: float = 0.0,
    backend: Backend | str = "cuda",
    mode: GemmMode | str = "fp64",
    ozaki_slices: int = 8,
) -> np.ndarray:
    """Row-major GEMM supporting FP64, FP32 Tensor Core, and Ozaki INT8 Tensor Core modes."""
    a_c = np.ascontiguousarray(A, dtype=np.float64)
    b_c = np.ascontiguousarray(B, dtype=np.float64)
    return _core.gemm(
        a_c,
        b_c,
        bool(transA),
        bool(transB),
        float(alpha),
        float(beta),
        _as_enum(Backend, backend),
        _as_enum(GemmMode, mode),
        int(ozaki_slices),
    )
