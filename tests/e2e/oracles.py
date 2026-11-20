"""Dynamic runtime reference oracles for GPU4DFT E2E verification (zero hardcoded results)."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np


_SUPPORTED_DFTD3_BJ = {
    "b3lyp": "b3lyp",
    "b3lypg": "b3lyp",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "pbeh": "pbe0",
    "cam-b3lyp": "cam-b3lyp",
    "camb3lyp": "cam-b3lyp",
    "wb97x": "wb97x",
}


def run_pyscf_scf(
    mol: Any,
    method: str,
    xc: str | None = None,
    density_fit: bool = False,
    auxbasis: str = "def2-universal-jkfit",
    grids_level: int = 3,
    conv_tol: float = 1e-10,
) -> dict[str, Any]:
    """Run a PySCF RHF/RKS/UHF/UKS reference SCF calculation dynamically."""
    from pyscf import dft, scf

    m = method.strip().lower()
    if m == "rhf":
        mf = scf.RHF(mol)
    elif m == "uhf":
        mf = scf.UHF(mol)
    elif m == "rks":
        if not xc:
            raise ValueError("xc must be provided for RKS")
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = int(grids_level)
        mf.grids.build()
    elif m == "uks":
        if not xc:
            raise ValueError("xc must be provided for UKS")
        mf = dft.UKS(mol, xc=xc)
        mf.grids.level = int(grids_level)
        mf.grids.build()
    else:
        raise ValueError(f"Unsupported PySCF method {method!r}; expected rhf/rks/uhf/uks")

    if density_fit:
        mf = mf.density_fit(auxbasis=auxbasis)

    mf.conv_tol = float(conv_tol)
    mf.max_cycle = max(int(getattr(mf, "max_cycle", 50)), 120)
    e_tot = float(mf.kernel())
    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)

    spin_sq: tuple[float, float] | None = None
    if m in ("uhf", "uks"):
        ss, mult = mf.spin_square()
        spin_sq = (float(ss), float(mult))

    return {
        "e_tot": e_tot,
        "converged": bool(mf.converged),
        "spin_square": spin_sq,
        "dm": dm,
        "mf": mf,
    }


def run_pyscf_grad(
    mf: Any,
    grid_response: bool = True,
    auxbasis_response: bool = True,
) -> np.ndarray:
    """Compute PySCF analytical nuclear gradients (natm, 3) in Eh/Bohr."""
    grad_driver = mf.nuc_grad_method()
    if hasattr(grad_driver, "grid_response"):
        grad_driver.grid_response = bool(grid_response)
    if hasattr(grad_driver, "auxbasis_response"):
        grad_driver.auxbasis_response = bool(auxbasis_response)
    grad = grad_driver.kernel()
    return np.asarray(grad, dtype=np.float64)


def run_dftd3_bj(
    atomic_numbers: Sequence[int] | np.ndarray,
    coords_bohr: np.ndarray,
    xc: str,
) -> tuple[float, np.ndarray]:
    """Compute Grimme DFT-D3(BJ) energy (Eh) and gradient (Eh/Bohr) via dftd3==1.7.0."""
    from dftd3.interface import DispersionModel, RationalDampingParam

    key = xc.strip().lower().replace("_", "-")
    if key not in _SUPPORTED_DFTD3_BJ:
        raise ValueError(
            f"Unsupported DFT-D3(BJ) functional {xc!r}; choose from {sorted(set(_SUPPORTED_DFTD3_BJ.values()))}"
        )
    norm_xc = _SUPPORTED_DFTD3_BJ[key]

    z_arr = np.asarray(atomic_numbers, dtype=np.int32)
    pos_arr = np.asarray(coords_bohr, dtype=np.float64)
    if pos_arr.ndim != 2 or pos_arr.shape != (len(z_arr), 3):
        raise ValueError(f"coords_bohr must have shape ({len(z_arr)}, 3), got {pos_arr.shape}")

    model = DispersionModel(z_arr, pos_arr)
    param = RationalDampingParam(method=norm_xc)
    res = model.get_dispersion(param, grad=True)
    energy = float(res["energy"])
    grad = np.asarray(res["gradient"], dtype=np.float64)
    return energy, grad


def central_fd_grad(
    energy_fn: Callable[[np.ndarray], float],
    coords_bohr: np.ndarray,
    step: float = 1e-4,
) -> np.ndarray:
    """Compute 2-point central finite-difference gradient (natm, 3) in Eh/Bohr."""
    base = np.asarray(coords_bohr, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError(f"coords_bohr must have shape (natm, 3), got {base.shape}")
    h = float(step)
    if h <= 0.0:
        raise ValueError("step must be positive")

    natm = base.shape[0]
    grad = np.zeros((natm, 3), dtype=np.float64)
    for a in range(natm):
        for axis in range(3):
            c_plus = base.copy()
            c_minus = base.copy()
            c_plus[a, axis] += h
            c_minus[a, axis] -= h
            e_plus = float(energy_fn(c_plus))
            e_minus = float(energy_fn(c_minus))
            grad[a, axis] = (e_plus - e_minus) / (2.0 * h)
    return grad


def translational_invariance_norm(grad: np.ndarray) -> float:
    """Return max_axis |sum_A g_A| in Eh/Bohr."""
    g = np.asarray(grad, dtype=np.float64)
    return float(np.max(np.abs(np.sum(g, axis=0))))


def rotational_invariance_norm(coords_bohr: np.ndarray, grad: np.ndarray) -> float:
    """Return max_axis |sum_A r_A x g_A| in Eh."""
    r = np.asarray(coords_bohr, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    torque = np.sum(np.cross(r, g), axis=0)
    return float(np.max(np.abs(torque)))


def calc_mae_rmse(vals: Sequence[float] | np.ndarray, refs: Sequence[float] | np.ndarray) -> tuple[float, float]:
    """Return (MAE, RMSE) between predicted values and reference values."""
    v = np.asarray(vals, dtype=np.float64)
    r = np.asarray(refs, dtype=np.float64)
    if v.shape != r.shape:
        raise ValueError(f"Shape mismatch: vals {v.shape} vs refs {r.shape}")
    if v.size == 0:
        raise ValueError("vals and refs must be non-empty")
    diff = v - r
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    return mae, rmse
