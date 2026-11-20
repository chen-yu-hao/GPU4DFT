"""Tier 1 E2E tests for Feature F4: Range-separated hybrids (CAM-B3LYP & wB97X)."""

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


@pytest.mark.tier1
@pytest.mark.f4
@pytest.mark.gpu
def test_f4_rks_camb3lyp_basis_ladder(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F4-01: RKS(CAM-B3LYP) across 6-31g*, def2-svp, def2-tzvp (f-shells) matches PySCF <= 1e-6 Eh."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    for basis in ("6-31g*", "def2-svp", "def2-tzvp"):
        mol = make_pyscf_mol(h2o.pyscf_atom, basis=basis)
        ref = run_pyscf_scf(mol, "rks", xc="cam-b3lyp", grids_level=3, conv_tol=1e-11)
        assert ref["converged"]

        mf = RKS(mol, xc="cam-b3lyp", backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf, ref["mf"])
        e_got = mf.kernel()

        assert mf.converged
        diff = abs(e_got - ref["e_tot"])
        assert diff <= 1e-6, f"CAM-B3LYP/{basis} diff {diff:.3e} > 1e-6 Eh"


@pytest.mark.tier1
@pytest.mark.f4
@pytest.mark.gpu
def test_f4_rks_wb97x_basis_ladder(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F4-02: RKS(wB97X) across 6-31g*, def2-svp, def2-tzvp matches PySCF wb97x (distinct from wb97x-d)."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    for basis in ("6-31g*", "def2-svp", "def2-tzvp"):
        mol = make_pyscf_mol(h2o.pyscf_atom, basis=basis)
        ref = run_pyscf_scf(mol, "rks", xc="wb97x", grids_level=3, conv_tol=1e-11)
        assert ref["converged"]

        mf = RKS(mol, xc="wb97x", backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf, ref["mf"])
        e_got = mf.kernel()

        assert mf.converged
        diff = abs(e_got - ref["e_tot"])
        assert diff <= 1e-6, f"wB97X/{basis} diff {diff:.3e} > 1e-6 Eh"

        if basis == "def2-svp":
            ref_xd = run_pyscf_scf(mol, "rks", xc="wb97x-d", grids_level=3, conv_tol=1e-11)
            assert abs(e_got - ref_xd["e_tot"]) > 1e-2


@pytest.mark.tier1
@pytest.mark.f4
@pytest.mark.gpu
def test_f4_uks_rsh_doublet_and_triplet(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F4-03: UKS(CAM-B3LYP & wB97X) on doublet CH3 and triplet O2 matches PySCF <= 1e-6 Eh & <S^2> <= 1e-5."""
    from gpu4dft import UKS

    for name in ("ch3_radical", "o2_triplet"):
        mol_data = get_molecule(name)
        mol = make_pyscf_mol(
            mol_data.pyscf_atom,
            basis="def2-svp",
            charge=mol_data.charge,
            spin=mol_data.spin,
        )
        for xc in ("cam-b3lyp", "wb97x"):
            ref = run_pyscf_scf(mol, "uks", xc=xc, grids_level=3, conv_tol=1e-11)
            assert ref["converged"]

            mf = UKS(mol, xc=xc, backend=gpu4dft_backend, conv_tol=1e-10)
            match_grids(mf, ref["mf"])
            e_got = mf.kernel()

            assert mf.converged
            assert abs(e_got - ref["e_tot"]) <= 1e-6
            ss, _ = mf.spin_square()
            assert abs(ss - ref["spin_square"][0]) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f4
@pytest.mark.gpu
def test_f4_rks_rsh_direct_gradients(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F4-04: Direct RKS(CAM-B3LYP & wB97X) gradients on distorted CH2O match PySCF and FD <= 1e-5."""
    from gpu4dft import RKS

    ch2o = get_molecule("ch2o")
    shifts = [(0.02, -0.01, 0.03), (-0.01, 0.02, -0.02), (0.03, 0.02, -0.01), (-0.02, -0.03, 0.01)]
    atom_spec = [
        (sym, (x + dx, y + dy, z + dz))
        for (sym, (x, y, z)), (dx, dy, dz) in zip(ch2o.atoms, shifts)
    ]
    mol = make_pyscf_mol(atom_spec, basis="def2-svp")
    symbols = [sym for sym, _ in atom_spec]
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    for xc in ("cam-b3lyp", "wb97x"):
        ref = run_pyscf_scf(mol, "rks", xc=xc, grids_level=5, conv_tol=1e-11)
        assert ref["converged"]
        g_pyscf = run_pyscf_grad(ref["mf"], grid_response=True)

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
        g_ana = mf.nuc_grad_method().kernel()

        assert np.max(np.abs(g_ana - g_pyscf)) <= 1e-5
        assert translational_invariance_norm(g_ana) <= 1e-6

        def _energy_fn(c_bohr: np.ndarray, _xc: str = xc) -> float:
            mol_d = make_pyscf_mol(
                [(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                basis="def2-svp",
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

        g_fd = central_fd_grad(_energy_fn, coords_bohr, step=1e-4)
        assert np.max(np.abs(g_ana - g_fd)) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f4
@pytest.mark.gpu
def test_f4_uks_rsh_direct_gradients(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F4-05: Direct UKS(CAM-B3LYP & wB97X) gradients on distorted HO2 match PySCF and FD <= 1e-5."""
    from gpu4dft import UKS

    atom_spec = [
        ("O", (0.0000, 0.0000, 0.0000)),
        ("O", (1.3800, 0.0300, 0.0000)),
        ("H", (-0.2100, 0.9900, 0.0400)),
    ]
    mol = make_pyscf_mol(atom_spec, basis="def2-svp", charge=0, spin=1)
    symbols = [sym for sym, _ in atom_spec]
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    for xc in ("cam-b3lyp", "wb97x"):
        ref = run_pyscf_scf(mol, "uks", xc=xc, grids_level=5, conv_tol=1e-11)
        assert ref["converged"]
        g_pyscf = run_pyscf_grad(ref["mf"], grid_response=True)

        mf = UKS(
            mol,
            xc=xc,
            backend=gpu4dft_backend,
            grids_level=5,
            conv_tol=1e-11,
            diis_tol=1e-8,
        )
        mf.kernel()
        assert mf.converged
        g_ana = mf.nuc_grad_method().kernel()

        assert np.max(np.abs(g_ana - g_pyscf)) <= 1e-5
        assert translational_invariance_norm(g_ana) <= 1e-6

        def _energy_fn(c_bohr: np.ndarray, _xc: str = xc) -> float:
            mol_d = make_pyscf_mol(
                [(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                basis="def2-svp",
                charge=0,
                spin=1,
                unit="Bohr",
            )
            mf_d = UKS(
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

        g_fd = central_fd_grad(_energy_fn, coords_bohr, step=1e-4)
        assert np.max(np.abs(g_ana - g_fd)) <= 1e-5
