"""Tier 1 E2E tests for Feature F1: libxc linkage, spin-polarized XC, and grid_response."""

from __future__ import annotations

import numpy as np
import pytest

from data.molecules import get_molecule
from oracles import (
    central_fd_grad,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


def _distorted_atoms(name: str, shifts_ang: list[tuple[float, float, float]]) -> list[tuple[str, tuple[float, float, float]]]:
    mol_data = get_molecule(name)
    assert len(shifts_ang) == mol_data.n_atoms
    out: list[tuple[str, tuple[float, float, float]]] = []
    for (sym, (x, y, z)), (dx, dy, dz) in zip(mol_data.atoms, shifts_ang):
        out.append((sym, (x + dx, y + dy, z + dz)))
    return out


@pytest.mark.tier1
@pytest.mark.f1
@pytest.mark.gpu
def test_f1_libxc_rks_functionals(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F1-01: RKS across GGA/hybrid/meta-GGA functionals matches PySCF <= 1e-6 Eh (matched grid)."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")

    for xc in ("pbe", "b3lyp", "pbe0", "blyp", "tpss"):
        ref = run_pyscf_scf(mol, "rks", xc=xc, grids_level=3, conv_tol=1e-11)
        assert ref["converged"], f"PySCF RKS({xc}) did not converge"

        mf = RKS(mol, xc=xc, backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf, ref["mf"])
        e_got = mf.kernel()

        assert mf.converged, f"GPU4DFT RKS({xc}) did not converge"
        diff = abs(e_got - ref["e_tot"])
        assert diff <= 1e-6, f"RKS({xc}) energy diff {diff:.3e} > 1e-6 Eh"


@pytest.mark.tier1
@pytest.mark.f1
@pytest.mark.gpu
def test_f1_libxc_uks_spin_polarized(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F1-02: Spin-polarized libxc UKS on OH radical matches PySCF <= 1e-6 Eh & <S^2> <= 1e-5."""
    from gpu4dft import UKS

    oh = get_molecule("oh_radical")
    mol = make_pyscf_mol(oh.pyscf_atom, basis="def2-svp", charge=oh.charge, spin=oh.spin)

    for xc in ("pbe", "b3lyp", "pbe0"):
        ref = run_pyscf_scf(mol, "uks", xc=xc, grids_level=3, conv_tol=1e-11)
        assert ref["converged"], f"PySCF UKS({xc}) did not converge"

        mf = UKS(mol, xc=xc, backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf, ref["mf"])
        e_got = mf.kernel()

        assert mf.converged, f"GPU4DFT UKS({xc}) did not converge"
        e_diff = abs(e_got - ref["e_tot"])
        assert e_diff <= 1e-6, f"UKS({xc}) energy diff {e_diff:.3e} > 1e-6 Eh"

        ss, _ = mf.spin_square()
        ss_ref = ref["spin_square"][0]
        ss_diff = abs(ss - ss_ref)
        assert ss_diff <= 1e-5, f"UKS({xc}) <S^2> diff {ss_diff:.3e} > 1e-5"


@pytest.mark.tier1
@pytest.mark.f1
@pytest.mark.gpu
def test_f1_grid_response_translational_invariance(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F1-03: Native MolecularGrid gradients satisfy |sum_A g_A|_inf <= 1e-6 Eh/a0 via grid_response."""
    from gpu4dft import RKS

    distorted_cases = (
        (
            "h2o",
            _distorted_atoms(
                "h2o",
                [(0.02, -0.01, 0.03), (0.04, 0.05, -0.02), (-0.03, 0.02, -0.04)],
            ),
        ),
        (
            "ch4",
            _distorted_atoms(
                "ch4",
                [
                    (0.01, -0.02, 0.01),
                    (0.05, 0.02, -0.01),
                    (-0.02, 0.04, 0.02),
                    (0.01, -0.03, 0.04),
                    (-0.03, -0.01, -0.03),
                ],
            ),
        ),
    )

    for name, atom_spec in distorted_cases:
        mol = make_pyscf_mol(atom_spec, basis="def2-svp")
        for xc in ("pbe", "b3lyp"):
            mf = RKS(
                mol,
                xc=xc,
                backend=gpu4dft_backend,
                grids_level=3,
                conv_tol=1e-11,
                diis_tol=1e-8,
            )
            mf.kernel()
            assert mf.converged, f"{name} RKS({xc}) did not converge"

            grad_drv = mf.nuc_grad_method()
            assert grad_drv.grid_response is True
            grad = grad_drv.kernel()
            assert grad.shape == (mol.natm, 3)
            assert np.max(np.abs(grad)) > 1e-3, "Expected non-zero gradient on distorted geometry"

            net_force = translational_invariance_norm(grad)
            assert net_force <= 1e-6, (
                f"{name} RKS({xc}) net force {net_force:.3e} > 1e-6 Eh/a0"
            )


@pytest.mark.tier1
@pytest.mark.f1
@pytest.mark.gpu
def test_f1_grid_response_vs_pyscf_and_fd(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F1-04: Analytical RKS gradient with grid_response=True matches PySCF and moving-grid FD <= 1e-5."""
    from gpu4dft import RKS

    atom_spec = _distorted_atoms(
        "h2o",
        [(0.02, -0.01, 0.02), (0.04, 0.05, -0.02), (-0.03, 0.02, -0.03)],
    )
    mol = make_pyscf_mol(atom_spec, basis="6-31g*")
    symbols = [sym for sym, _ in atom_spec]
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    for xc in ("pbe", "b3lyp"):
        ref = run_pyscf_scf(mol, "rks", xc=xc, grids_level=5, conv_tol=1e-11)
        assert ref["converged"]
        grad_pyscf = run_pyscf_grad(ref["mf"], grid_response=True)

        mf = RKS(
            mol,
            xc=xc,
            backend=gpu4dft_backend,
            grids_level=5,
            conv_tol=1e-11,
            diis_tol=1e-8,
        )
        mf.kernel()
        assert mf.converged
        grad_ana = mf.nuc_grad_method().kernel()

        diff_pyscf = float(np.max(np.abs(grad_ana - grad_pyscf)))
        assert diff_pyscf <= 1e-5, f"RKS({xc}) grad diff vs PySCF {diff_pyscf:.3e} > 1e-5"

        def _moving_grid_energy(c_bohr: np.ndarray, _xc: str = xc) -> float:
            mol_d = make_pyscf_mol(
                [(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                basis="6-31g*",
                unit="Bohr",
            )
            mf_d = RKS(
                mol_d,
                xc=_xc,
                backend=gpu4dft_backend,
                grids_level=5,
                conv_tol=1e-11,
                diis_tol=1e-8,
            )
            e_d = mf_d.kernel()
            assert mf_d.converged
            return float(e_d)

        grad_fd = central_fd_grad(_moving_grid_energy, coords_bohr, step=1e-4)
        diff_fd = float(np.max(np.abs(grad_ana - grad_fd)))
        assert diff_fd <= 1e-5, f"RKS({xc}) grad diff vs moving-grid FD {diff_fd:.3e} > 1e-5"
        assert translational_invariance_norm(grad_ana) <= 1e-6


@pytest.mark.tier1
@pytest.mark.f1
@pytest.mark.gpu
def test_f1_libxc_native_grid_level_convergence(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F1-05: Native GPU4DFT grid level-4 RKS(B3LYP) without PySCF grid matches PySCF <= 1.5e-4 Eh."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")

    ref = run_pyscf_scf(mol, "rks", xc="b3lyp", grids_level=4, conv_tol=1e-11)
    assert ref["converged"]

    mf = RKS(mol, xc="b3lyp", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10)
    e_got = mf.kernel()

    assert mf.converged
    diff = abs(e_got - ref["e_tot"])
    assert diff <= 1.5e-4, f"Native level-4 RKS(B3LYP) diff {diff:.3e} > 1.5e-4 Eh"
