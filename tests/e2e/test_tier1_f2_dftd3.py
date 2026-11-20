"""Tier 1 E2E tests for Feature F2: DFT-D3(BJ) dispersion energy, gradients, and SCF/ASE hooks."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import BOHR_TO_ANGSTROM
from data.molecules import get_molecule
from oracles import (
    central_fd_grad,
    run_dftd3_bj,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)

D3BJ_FUNCTIONALS = ("b3lyp", "pbe", "pbe0", "cam-b3lyp", "wb97x")


def _ch4_h2o_complex() -> tuple[np.ndarray, np.ndarray]:
    ch4 = get_molecule("ch4")
    h2o = get_molecule("h2o")
    z = np.concatenate([ch4.atomic_numbers, h2o.atomic_numbers])
    coords_h2o = h2o.coords_bohr + np.array([[0.0, 0.0, 6.2]], dtype=np.float64)
    coords = np.vstack([ch4.coords_bohr, coords_h2o])
    return z, coords


def _oh_plus_h2_atoms() -> list[tuple[str, tuple[float, float, float]]]:
    return [
        ("O", (0.0000, 0.0000, 0.0000)),
        ("H", (0.0000, 0.0000, 0.9800)),
        ("H", (0.2000, 0.0000, 2.5500)),
        ("H", (0.3500, 0.1000, 3.2800)),
    ]


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_d3bj_energy_all_five_functionals(
    require_gpu4dft,
    require_dftd3,
) -> None:
    """T1-F2-01: DFT-D3(BJ) energy matches s-dftd3 <= 1e-8 Eh across all 5 required functionals."""
    from gpu4dft import dftd3_energy_grad

    h2o_dimer = get_molecule("h2o_dimer")
    z_ch4_h2o, c_ch4_h2o = _ch4_h2o_complex()

    systems = (
        ("h2o_dimer", h2o_dimer.atomic_numbers, h2o_dimer.coords_bohr),
        ("ch4_h2o", z_ch4_h2o, c_ch4_h2o),
    )

    for sys_name, z, coords_bohr in systems:
        for xc in D3BJ_FUNCTIONALS:
            e_ref, _ = run_dftd3_bj(z, coords_bohr, xc=xc)
            e_got, g_got = dftd3_energy_grad(z, coords_bohr, xc=xc)
            assert e_got < 0.0
            assert g_got.shape == (len(z), 3)
            diff = abs(e_got - e_ref)
            assert diff <= 1e-8, f"{sys_name} D3(BJ) {xc} energy diff {diff:.3e} > 1e-8 Eh"


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_d3bj_grad_vs_sdftd3_and_fd(
    require_gpu4dft,
    require_dftd3,
) -> None:
    """T1-F2-02: Analytical DFT-D3(BJ) gradients match s-dftd3 and central FD <= 1e-7 Eh/a0."""
    from gpu4dft import dftd3_energy_grad

    dimer = get_molecule("h2o_dimer")
    z = dimer.atomic_numbers
    coords_bohr = dimer.coords_bohr.copy()
    # Asymmetric distortion so all gradient components are non-trivial
    coords_bohr[3:, 0] += 0.18
    coords_bohr[3:, 1] -= 0.11

    for xc in D3BJ_FUNCTIONALS:
        _, g_ref = run_dftd3_bj(z, coords_bohr, xc=xc)
        _, g_got = dftd3_energy_grad(z, coords_bohr, xc=xc)
        g_fd = central_fd_grad(
            lambda c, _xc=xc: dftd3_energy_grad(z, c, xc=_xc)[0],
            coords_bohr,
            step=1e-4,
        )

        diff_ref = float(np.max(np.abs(g_got - g_ref)))
        diff_fd = float(np.max(np.abs(g_got - g_fd)))
        assert diff_ref <= 1e-7, f"D3(BJ) {xc} grad vs s-dftd3 diff {diff_ref:.3e} > 1e-7"
        assert diff_fd <= 1e-7, f"D3(BJ) {xc} grad vs FD diff {diff_fd:.3e} > 1e-7"
        assert translational_invariance_norm(g_got) <= 1e-12


@pytest.mark.tier1
@pytest.mark.f2
def test_f2_d3bj_heteroatom_cn_coverage(
    require_gpu4dft,
    require_dftd3,
) -> None:
    """T1-F2-03: DFT-D3(BJ) coordination numbers & C6/C8 parameters cover C,H,N,O,S,Cl vs s-dftd3."""
    from gpu4dft import dftd3_energy_grad

    for name in ("ch3cl_h2s", "formamide_dimer"):
        mol_data = get_molecule(name)
        z = mol_data.atomic_numbers
        coords_bohr = mol_data.coords_bohr
        for xc in ("b3lyp", "wb97x"):
            e_ref, g_ref = run_dftd3_bj(z, coords_bohr, xc=xc)
            e_got, g_got = dftd3_energy_grad(z, coords_bohr, xc=xc)

            e_diff = abs(e_got - e_ref)
            g_diff = float(np.max(np.abs(g_got - g_ref)))
            assert e_diff <= 1e-8, f"{name} {xc} energy diff {e_diff:.3e} > 1e-8"
            assert g_diff <= 1e-7, f"{name} {xc} grad diff {g_diff:.3e} > 1e-7"


@pytest.mark.tier1
@pytest.mark.f2
@pytest.mark.gpu
def test_f2_rks_dispersion_hook_additivity(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F2-04: RKS(dispersion='d3bj') adds exact D3(BJ) energy and nuclear gradient."""
    from gpu4dft import RKS, dftd3_energy_grad

    dimer = get_molecule("h2o_dimer")
    mol = make_pyscf_mol(dimer.pyscf_atom, basis="def2-svp")
    z = np.asarray(mol.atom_charges(), dtype=np.int32)
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    e_d3, g_d3 = dftd3_energy_grad(z, coords_bohr, xc="b3lyp")

    mf_nod3 = RKS(mol, xc="b3lyp", backend=gpu4dft_backend, grids_level=3, conv_tol=1e-11)
    e_nod3 = mf_nod3.kernel()
    assert mf_nod3.converged
    g_nod3 = mf_nod3.nuc_grad_method().kernel()

    mf_d3 = RKS(
        mol,
        xc="b3lyp",
        dispersion="d3bj",
        backend=gpu4dft_backend,
        grids_level=3,
        conv_tol=1e-11,
    )
    e_tot = mf_d3.kernel()
    assert mf_d3.converged
    g_tot = mf_d3.nuc_grad_method().kernel()

    assert abs((e_tot - e_nod3) - e_d3) <= 1e-12
    assert abs(mf_d3.e_disp - e_d3) <= 1e-12
    assert float(np.max(np.abs((g_tot - g_nod3) - g_d3))) <= 1e-11


@pytest.mark.tier1
@pytest.mark.f2
@pytest.mark.gpu
def test_f2_uks_and_ase_d3bj_hook(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F2-05: UKS(dispersion='d3bj') and GPU4DFTCalculator(dispersion='d3bj') match PySCF+s-dftd3."""
    import ase
    import ase.units
    from gpu4dft import GPU4DFTCalculator, UKS

    atom_spec = _oh_plus_h2_atoms()
    mol = make_pyscf_mol(atom_spec, basis="def2-svp", charge=0, spin=1)
    z = np.asarray(mol.atom_charges(), dtype=np.int32)
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    e_d3_ref, g_d3_ref = run_dftd3_bj(z, coords_bohr, xc="b3lyp")

    # 1. Direct UKS with matched grid vs PySCF + s-dftd3
    ref_matched = run_pyscf_scf(mol, "uks", xc="b3lyp", grids_level=3, conv_tol=1e-11)
    assert ref_matched["converged"]
    mf_uks = UKS(mol, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, conv_tol=1e-10)
    match_grids(mf_uks, ref_matched["mf"])
    e_uks = mf_uks.kernel()
    assert mf_uks.converged
    assert abs(e_uks - (ref_matched["e_tot"] + e_d3_ref)) <= 1e-6

    # 2. ASE GPU4DFTCalculator with method='uks', spin=1, dispersion='d3bj' vs PySCF+s-dftd3 (fine grid)
    ref_fine = run_pyscf_scf(mol, "uks", xc="b3lyp", grids_level=5, conv_tol=1e-11)
    assert ref_fine["converged"]
    g_fine_ref = run_pyscf_grad(ref_fine["mf"], grid_response=True) + g_d3_ref

    symbols = [sym for sym, _ in atom_spec]
    positions_ang = np.array([xyz for _, xyz in atom_spec], dtype=np.float64)
    atoms = ase.Atoms(symbols=symbols, positions=positions_ang)
    atoms.calc = GPU4DFTCalculator(
        method="uks",
        xc="b3lyp",
        basis="def2-svp",
        charge=0,
        spin=1,
        dispersion="d3bj",
        backend=gpu4dft_backend,
        grids_level=5,
        conv_tol=1e-11,
    )
    e_ev = atoms.get_potential_energy()
    forces_ev_ang = atoms.get_forces()

    e_eh = float(e_ev) / ase.units.Hartree
    grad_eh_bohr = -np.asarray(forces_ev_ang, dtype=np.float64) * (
        BOHR_TO_ANGSTROM / ase.units.Hartree
    )

    assert abs(e_eh - (ref_fine["e_tot"] + e_d3_ref)) <= 1.5e-4
    assert float(np.max(np.abs(grad_eh_bohr - g_fine_ref))) <= 1e-4

    atoms_fd = ase.Atoms(symbols=symbols, positions=positions_ang)
    atoms_fd.calc = GPU4DFTCalculator(
        method="uks",
        xc="b3lyp",
        basis="def2-svp",
        charge=0,
        spin=1,
        dispersion="d3bj",
        backend=gpu4dft_backend,
        grids_level=5,
        conv_tol=1e-11,
    )

    def _calc_energy_eh(c_bohr: np.ndarray) -> float:
        atoms_fd.set_positions(np.asarray(c_bohr, dtype=np.float64) * BOHR_TO_ANGSTROM)
        return float(atoms_fd.get_potential_energy()) / ase.units.Hartree

    g_fd = central_fd_grad(_calc_energy_eh, coords_bohr, step=1e-4)
    assert float(np.max(np.abs(grad_eh_bohr - g_fd))) <= 1e-5
