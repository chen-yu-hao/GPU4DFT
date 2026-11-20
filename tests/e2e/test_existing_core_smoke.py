"""Baseline E2E smoke tests runnable immediately on the existing closed-shell core stack."""

from __future__ import annotations

import numpy as np
import pytest

from data.bh76_loader import get_bh76_openshell_reactions, get_bh76_species
from data.molecules import LARGE_MOLECULE_NAMES, SMALL_MEDIUM_MOLECULE_NAMES, get_molecule
from data.s22_loader import load_s22_complexes
from oracles import (
    calc_mae_rmse,
    central_fd_grad,
    rotational_invariance_norm,
    run_dftd3_bj,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


@pytest.mark.tier1
@pytest.mark.existing_core
@pytest.mark.gpu
def test_existing_core_rhf_rks_ri_grad_ase(
    require_gpu4dft,
    require_pyscf,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """Exercise existing closed-shell RHF, RKS(B3LYP), RI-JK, Direct grad, and ASE Calculator."""
    import ase
    from gpu4dft import GPU4DFTCalculator, RHF, RKS, SimpleAtoms

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")

    # 1. RHF Direct SCF + Analytical Gradient vs PySCF
    ref_rhf = run_pyscf_scf(mol, "rhf", conv_tol=1e-11)
    ref_rhf_grad = run_pyscf_grad(ref_rhf["mf"])

    mf_rhf = RHF(mol, backend=gpu4dft_backend, conv_tol=1e-10)
    e_rhf = mf_rhf.kernel()
    assert mf_rhf.converged
    assert abs(e_rhf - ref_rhf["e_tot"]) <= 1e-8
    grad_rhf = mf_rhf.nuc_grad_method().kernel()
    assert np.max(np.abs(grad_rhf - ref_rhf_grad)) <= 1e-6
    assert translational_invariance_norm(grad_rhf) <= 1e-7

    # 2. RKS(B3LYP) Direct SCF (matched grid) + Analytical Gradient vs PySCF
    ref_rks = run_pyscf_scf(mol, "rks", xc="b3lyp", grids_level=3, conv_tol=1e-11)
    ref_rks_grad = run_pyscf_grad(ref_rks["mf"], grid_response=True)

    mf_rks = RKS(mol, xc="b3lyp", backend=gpu4dft_backend, conv_tol=1e-10)
    match_grids(mf_rks, ref_rks["mf"])
    e_rks = mf_rks.kernel()
    assert mf_rks.converged
    assert abs(e_rks - ref_rks["e_tot"]) <= 1e-8
    grad_rks = mf_rks.nuc_grad_method().kernel()
    # Allow ~1e-4 before M1 grid_response lands, or <=1e-5 once grid_response=True is active
    assert np.max(np.abs(grad_rks - ref_rks_grad)) <= 2e-4

    # 3. Closed-shell RI-JK SCF vs PySCF DF
    ref_ri = run_pyscf_scf(
        mol,
        "rks",
        xc="b3lyp",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        grids_level=3,
        conv_tol=1e-11,
    )
    mf_ri = RKS(mol, xc="b3lyp", backend=gpu4dft_backend, conv_tol=1e-10).density_fit(
        auxbasis="def2-universal-jkfit"
    )
    match_grids(mf_ri, ref_ri["mf"])
    e_ri = mf_ri.kernel()
    assert mf_ri.converged
    assert abs(e_ri - ref_ri["e_tot"]) <= 1e-8

    # 4. GPU4DFTCalculator on both SimpleAtoms and ase.Atoms
    coords_ang = h2o.coords_angstrom
    z_nums = h2o.atomic_numbers.tolist()
    calc_simple = GPU4DFTCalculator(
        xc="b3lyp",
        basis="def2-svp",
        backend=gpu4dft_backend,
        grids_level=3,
        conv_tol=1e-10,
    )
    simple_atoms = SimpleAtoms(z_nums, coords_ang.tolist())
    simple_atoms.calc = calc_simple
    e_simple_ev = simple_atoms.get_potential_energy()
    f_simple = simple_atoms.get_forces()

    ase_atoms = ase.Atoms(symbols=h2o.symbols, positions=coords_ang)
    ase_atoms.calc = GPU4DFTCalculator(
        xc="b3lyp",
        basis="def2-svp",
        backend=gpu4dft_backend,
        grids_level=3,
        conv_tol=1e-10,
    )
    e_ase_ev = ase_atoms.get_potential_energy()
    f_ase = ase_atoms.get_forces()

    assert abs(e_simple_ev - e_ase_ev) <= 1e-8
    assert f_simple.shape == (3, 3) and f_ase.shape == (3, 3)
    assert np.max(np.abs(f_simple - f_ase)) <= 1e-7


@pytest.mark.tier1
@pytest.mark.existing_core
def test_existing_core_oracle_loaders_integrity(
    require_pyscf,
    require_dftd3,
    require_ase,
    make_pyscf_mol,
) -> None:
    """Verify S22, BH76 open-shell, large-molecule NAO ranges, dftd3, and PySCF/FD oracles."""
    # 1. S22 dataset (22 complexes)
    s22 = load_s22_complexes()
    assert len(s22) == 22
    for c in s22:
        assert len(c.dimer) == c.n_atoms_a + c.n_atoms_b
        assert len(c.monomer_a) == c.n_atoms_a
        assert len(c.monomer_b) == c.n_atoms_b
        assert c.ref_s22b_kcal < 0.0

    # 2. BH76 open-shell dataset
    species = get_bh76_species()
    full_rxns = get_bh76_openshell_reactions("full_openshell")
    core_rxns = get_bh76_openshell_reactions("core")
    assert len(species) >= 50
    assert len(full_rxns) >= 50
    assert len(core_rxns) >= 10

    # 3. Small/medium molecules + 4 large R1 molecules with PySCF NAO check
    for name in SMALL_MEDIUM_MOLECULE_NAMES:
        m = get_molecule(name)
        assert m.n_atoms >= 2

    expected_large = {
        "h2o_32": (96, "H64O32", "def2-svp", 768),
        "c60": (60, "C60", "def2-svp", 840),
        "taxol": (113, "C47H51NO14", "6-31g*", 970),
        "valinomycin": (168, "C54H90N6O18", "6-31g*", 1272),
    }
    assert set(LARGE_MOLECULE_NAMES) == set(expected_large)
    for name, (natoms, formula, basis, expected_nao) in expected_large.items():
        m = get_molecule(name)
        assert m.n_atoms == natoms
        assert m.formula == formula
        assert m.min_distance_angstrom() > 0.85
        mol = make_pyscf_mol(m.pyscf_atom, basis=basis)
        assert mol.nao_nr() == expected_nao
        assert 500 <= mol.nao_nr() <= 2000

    # 4. dftd3==1.7.0 across all 5 required functionals + FD check
    h2o_dimer = get_molecule("h2o_dimer")
    z = h2o_dimer.atomic_numbers
    coords_bohr = h2o_dimer.coords_bohr
    for xc in ("b3lyp", "pbe", "pbe0", "cam-b3lyp", "wb97x"):
        e_d3, g_d3 = run_dftd3_bj(z, coords_bohr, xc=xc)
        assert e_d3 < 0.0
        assert g_d3.shape == (h2o_dimer.n_atoms, 3)
        assert translational_invariance_norm(g_d3) <= 1e-12
        assert rotational_invariance_norm(coords_bohr, g_d3) <= 1e-12

    g_fd = central_fd_grad(lambda c: run_dftd3_bj(z, c, xc="b3lyp")[0], coords_bohr, step=1e-4)
    _, g_an = run_dftd3_bj(z, coords_bohr, xc="b3lyp")
    assert np.max(np.abs(g_an - g_fd)) <= 1e-8

    # 5. PySCF SCF + grad + MAE/RMSE utility sanity check
    oh = get_molecule("oh_radical")
    mol_oh = make_pyscf_mol(oh.pyscf_atom, basis="def2-svp", charge=oh.charge, spin=oh.spin)
    res_oh = run_pyscf_scf(mol_oh, "uks", xc="cam-b3lyp", density_fit=True, grids_level=3)
    assert res_oh["converged"]
    assert res_oh["spin_square"] is not None
    assert abs(res_oh["spin_square"][0] - 0.75) < 0.05
    grad_oh = run_pyscf_grad(res_oh["mf"], grid_response=True, auxbasis_response=True)
    assert grad_oh.shape == (2, 3)

    mae, rmse = calc_mae_rmse([1.0, 2.0, 3.0], [1.1, 1.8, 3.3])
    assert np.isclose(mae, 0.2)
    assert np.isclose(rmse, np.sqrt((0.01 + 0.04 + 0.09) / 3.0))
