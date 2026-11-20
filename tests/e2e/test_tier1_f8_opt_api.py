"""Tier 1 E2E tests for Feature F8: Geometry Optimizer API, S22 & BH76 (T1-F8-01..05)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_MOL
from data.bh76_loader import get_bh76_openshell_reactions, get_bh76_species
from data.molecules import get_molecule
from data.s22_loader import get_s22_complex
from oracles import calc_mae_rmse, central_fd_grad, run_dftd3_bj, run_pyscf_grad, run_pyscf_scf


def _run_geometric_opt(
    mol_init: Any,
    grad_scanner: Any,
    work_dir: Path,
    *,
    transition: bool = False,
    maxiter: int = 100,
) -> tuple[Any, float, np.ndarray]:
    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from pyscf import gto

    symbols = [mol_init.atom_symbol(i) for i in range(mol_init.natm)]
    coords_ang = np.asarray(mol_init.atom_coords(unit="Angstrom"), dtype=np.float64)

    gmol = geometric.molecule.Molecule()
    gmol.elem = symbols
    gmol.xyzs = [coords_ang]

    class _ScannerEngine(geometric.engine.Engine):
        def calc_new(self, coords: np.ndarray, dirname: str) -> dict[str, Any]:
            c_ang = np.asarray(coords, dtype=np.float64).reshape(-1, 3) * BOHR_TO_ANGSTROM
            mol_step = gto.M(
                atom=[(symbols[i], tuple(c_ang[i])) for i in range(len(symbols))],
                basis=mol_init.basis,
                charge=mol_init.charge,
                spin=mol_init.spin,
                unit="Angstrom",
                cart=False,
                verbose=0,
            )
            e_tot, grad = grad_scanner(mol_step)
            return {
                "energy": float(e_tot),
                "gradient": np.asarray(grad, dtype=np.float64).ravel(),
            }

    work_dir.mkdir(parents=True, exist_ok=True)
    input_xyz = work_dir / "opt_init.xyz"
    lines = [str(len(symbols)), "geometric opt init"]
    for sym, xyz in zip(symbols, coords_ang):
        lines.append(f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    input_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")

    engine = _ScannerEngine(gmol)
    m_opt = geometric.optimize.run_optimizer(
        customengine=engine,
        input=str(input_xyz),
        transition=bool(transition),
        maxiter=int(maxiter),
        prefix=str(work_dir / "geom_opt"),
    )
    final_ang = np.asarray(m_opt.xyzs[-1], dtype=np.float64)
    mol_final = gto.M(
        atom=[(symbols[i], tuple(final_ang[i])) for i in range(len(symbols))],
        basis=mol_init.basis,
        charge=mol_init.charge,
        spin=mol_init.spin,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    e_final, g_final = grad_scanner(mol_final)
    return mol_final, float(e_final), np.asarray(g_final, dtype=np.float64)


@pytest.mark.tier1
@pytest.mark.f8
@pytest.mark.gpu
def test_f8_as_scanner_scf_and_grad(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F8-01: SCF .as_scanner() and grad .as_scanner() match fresh single-point runs <= 1e-9."""
    from gpu4dft import RKS, UKS

    for mol_name, method, xc in (("h2o", "rks", "b3lyp"), ("ho2_radical", "uks", "cam-b3lyp")):
        spec = get_molecule(mol_name)
        mol1 = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
        coords2 = spec.coords_angstrom.copy()
        coords2[0, 0] += 0.05
        atoms2 = [(spec.symbols[i], tuple(coords2[i])) for i in range(spec.n_atoms)]
        mol2 = make_pyscf_mol(atoms2, basis="def2-svp", charge=spec.charge, spin=spec.spin)

        cls = RKS if method == "rks" else UKS
        base = cls(
            mol1, xc=xc, backend=gpu4dft_backend, precision="fp64", grids_level=3, conv_tol=1e-11
        ).density_fit(auxbasis="def2-universal-jkfit")

        scf_scanner = base.as_scanner()
        grad_scanner = base.nuc_grad_method().as_scanner()

        for m_cur in (mol1, mol2):
            e_scan = float(scf_scanner(m_cur))
            e_gscan, g_scan = grad_scanner(m_cur)

            mf_fresh = cls(
                m_cur, xc=xc, backend=gpu4dft_backend, precision="fp64", grids_level=3, conv_tol=1e-11
            ).density_fit(auxbasis="def2-universal-jkfit")
            e_fresh = float(mf_fresh.kernel())
            g_fresh = np.asarray(mf_fresh.nuc_grad_method().kernel(), dtype=np.float64)

            assert abs(e_scan - e_fresh) <= 1e-9
            assert abs(float(e_gscan) - e_fresh) <= 1e-9
            assert np.max(np.abs(np.asarray(g_scan, dtype=np.float64) - g_fresh)) <= 1e-9


@pytest.mark.tier1
@pytest.mark.f8
@pytest.mark.gpu
def test_f8_ase_calculator_uhf_uks_d3bj(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F8-02: GPU4DFTCalculator(method='uks', xc='wb97x', spin=1, d3bj, DF) matches PySCF+D3 & FD."""
    import ase
    from ase.units import Bohr as ASE_BOHR
    from ase.units import Hartree as ASE_HARTREE
    from gpu4dft import GPU4DFTCalculator, SimpleAtoms

    spec = get_molecule("ho2_radical")
    ase_atoms = ase.Atoms(symbols=spec.symbols, positions=spec.coords_angstrom)
    simple_atoms = SimpleAtoms(spec.atomic_numbers.tolist(), spec.coords_angstrom.tolist(), charge=0)

    # 1. Grid-free UHF check via GPU4DFTCalculator (<= 1e-6 Eh exact)
    mol_ref = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=0, spin=1)
    ref_uhf = run_pyscf_scf(mol_ref, "uhf", density_fit=True, conv_tol=1e-11)
    ref_uhf_grad = run_pyscf_grad(ref_uhf["mf"], auxbasis_response=True)

    ase_atoms.calc = GPU4DFTCalculator(
        method="uhf",
        xc="hf",
        basis="def2-svp",
        charge=0,
        spin=1,
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=gpu4dft_backend,
        conv_tol=1e-10,
    )
    e_uhf_eh = ase_atoms.get_potential_energy() / ASE_HARTREE
    g_uhf_eh_bohr = -ase_atoms.get_forces() * (ASE_BOHR / ASE_HARTREE)
    assert abs(e_uhf_eh - ref_uhf["e_tot"]) <= 1e-6
    assert np.max(np.abs(g_uhf_eh_bohr - ref_uhf_grad)) <= 1e-5

    # 2. UKS(wB97X-D3(BJ)) + RI-JK check via GPU4DFTCalculator on ASE & SimpleAtoms
    ref_uks = run_pyscf_scf(
        mol_ref, "uks", xc="wb97x", density_fit=True, grids_level=5, conv_tol=1e-11
    )
    ref_uks_grad = run_pyscf_grad(ref_uks["mf"], grid_response=True, auxbasis_response=True)
    e_d3, g_d3 = run_dftd3_bj(spec.atomic_numbers, spec.coords_bohr, xc="wb97x")
    e_oracle_eh = ref_uks["e_tot"] + e_d3
    g_oracle_eh_bohr = ref_uks_grad + g_d3

    calc_uks = GPU4DFTCalculator(
        method="uks",
        xc="wb97x",
        basis="def2-svp",
        charge=0,
        spin=1,
        dispersion="d3bj",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=gpu4dft_backend,
        grids_level=5,
        conv_tol=1e-10,
    )
    ase_atoms.calc = calc_uks
    e_uks_eh = ase_atoms.get_potential_energy() / ASE_HARTREE
    g_uks_eh_bohr = -ase_atoms.get_forces() * (ASE_BOHR / ASE_HARTREE)

    simple_atoms.calc = GPU4DFTCalculator(
        method="uks",
        xc="wb97x",
        basis="def2-svp",
        charge=0,
        spin=1,
        dispersion="d3bj",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=gpu4dft_backend,
        grids_level=5,
        conv_tol=1e-10,
    )
    e_simple_eh = simple_atoms.get_potential_energy() / ASE_HARTREE
    assert abs(e_uks_eh - e_simple_eh) <= 1e-8
    assert np.max(np.abs(g_uks_eh_bohr - g_oracle_eh_bohr)) <= 1e-5
    assert abs(e_uks_eh - e_oracle_eh) <= 1.5e-4

    # Self-consistent central FD forces on GPU4DFTCalculator itself
    def _calc_energy_bohr(c_bohr: np.ndarray) -> float:
        a = ase.Atoms(symbols=spec.symbols, positions=c_bohr * BOHR_TO_ANGSTROM)
        a.calc = calc_uks
        return float(a.get_potential_energy() / ASE_HARTREE)

    g_fd = central_fd_grad(_calc_energy_bohr, spec.coords_bohr, step=1e-4)
    assert np.max(np.abs(g_uks_eh_bohr - g_fd)) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f8
@pytest.mark.gpu
def test_f8_s22_single_complex_opt_convergence(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T1-F8-03: S22 Water_dimer B3LYP-D3(BJ)/def2-svp RI-JK optimization converges <= 4.5e-4 Eh/a0."""
    from gpu4dft import RKS

    cpx = get_s22_complex("Water_dimer")
    mol_d = make_pyscf_mol(cpx.dimer, basis="def2-svp")
    mol_a = make_pyscf_mol(cpx.monomer_a, basis="def2-svp")
    mol_b = make_pyscf_mol(cpx.monomer_b, basis="def2-svp")
    auxbasis = "def2-universal-jkfit"

    mf_d = RKS(
        mol_d, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mol_d_opt, e_d_opt, g_d_opt = _run_geometric_opt(
        mol_d, mf_d.nuc_grad_method().as_scanner(), tmp_path / "water_dimer", maxiter=100
    )
    assert np.max(np.abs(g_d_opt)) <= 4.5e-4

    mf_a = RKS(
        mol_a, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mol_a_opt, e_a_opt, g_a_opt = _run_geometric_opt(
        mol_a, mf_a.nuc_grad_method().as_scanner(), tmp_path / "mon_a", maxiter=80
    )
    assert np.max(np.abs(g_a_opt)) <= 4.5e-4

    mf_b = RKS(
        mol_b, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mol_b_opt, e_b_opt, g_b_opt = _run_geometric_opt(
        mol_b, mf_b.nuc_grad_method().as_scanner(), tmp_path / "mon_b", maxiter=80
    )
    assert np.max(np.abs(g_b_opt)) <= 4.5e-4

    e_int_g4d = (e_d_opt - e_a_opt - e_b_opt) * HARTREE_TO_KCAL_MOL

    def _pyscf_b3lyp_d3(mol_x) -> float:
        res = run_pyscf_scf(mol_x, "rks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=4)
        e_d3, _ = run_dftd3_bj(
            mol_x.atom_charges(), np.asarray(mol_x.atom_coords(unit="Bohr")), xc="b3lyp"
        )
        return float(res["e_tot"] + e_d3)

    e_int_pyscf = (_pyscf_b3lyp_d3(mol_d_opt) - _pyscf_b3lyp_d3(mol_a_opt) - _pyscf_b3lyp_d3(mol_b_opt)) * HARTREE_TO_KCAL_MOL
    assert abs(e_int_g4d - e_int_pyscf) <= 0.05


@pytest.mark.tier1
@pytest.mark.f8
@pytest.mark.gpu
def test_f8_bh76_single_reaction_barrier_and_opt(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T1-F8-04: BH76 open-shell reaction opt + barrier height matches PySCF DF within 0.1 kcal/mol."""
    from gpu4dft import UKS

    core_rxns = get_bh76_openshell_reactions("core")
    species = get_bh76_species()
    rxn = next(
        (r for r in core_rxns if r.direction == "fwd" and "H2" in r.reactant_names),
        core_rxns[0],
    )
    auxbasis = "def2-universal-jkfit"

    opt_mols: dict[str, Any] = {}
    opt_energies: dict[str, float] = {}

    for sp_name in set(rxn.species_names):
        sp = species[sp_name]
        mol = make_pyscf_mol(sp.pyscf_atom, basis="def2-svp", charge=sp.charge, spin=sp.spin)
        mf = UKS(
            mol, xc="cam-b3lyp", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
        ).density_fit(auxbasis=auxbasis)

        if sp.n_atoms == 1:
            e_sp = float(mf.kernel())
            assert mf.converged
            opt_mols[sp_name] = mol
            opt_energies[sp_name] = e_sp
        else:
            is_ts = sp_name == rxn.ts_name
            mol_opt, e_opt, g_opt = _run_geometric_opt(
                mol,
                mf.nuc_grad_method().as_scanner(),
                tmp_path / f"bh76_{sp_name}",
                transition=is_ts,
                maxiter=100,
            )
            assert np.max(np.abs(g_opt)) <= 4.5e-4
            opt_mols[sp_name] = mol_opt
            opt_energies[sp_name] = e_opt

    barrier_g4d = (
        sum(c * opt_energies[s] for s, c in zip(rxn.species_names, rxn.coeffs))
        * HARTREE_TO_KCAL_MOL
    )

    pyscf_energies: dict[str, float] = {}
    for sp_name, mol_opt in opt_mols.items():
        res = run_pyscf_scf(
            mol_opt, "uks", xc="cam-b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=4
        )
        assert res["converged"]
        pyscf_energies[sp_name] = float(res["e_tot"])

    barrier_pyscf = (
        sum(c * pyscf_energies[s] for s, c in zip(rxn.species_names, rxn.coeffs))
        * HARTREE_TO_KCAL_MOL
    )
    assert abs(barrier_g4d - barrier_pyscf) <= 0.1


@pytest.mark.tier1
@pytest.mark.f8
@pytest.mark.gpu
def test_f8_s22_representative_sp_mae(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F8-05: 3 representative S22 complexes SP B3LYP-D3(BJ)/def2-svp DF MAE <= 0.05 kcal/mol vs PySCF."""
    from gpu4dft import RKS

    names = ("Water_dimer", "Methane_dimer", "Ethene-ethyne_complex")
    auxbasis = "def2-universal-jkfit"

    g4d_eints: list[float] = []
    pyscf_eints: list[float] = []

    for name in names:
        cpx = get_s22_complex(name)

        def _eval_frag(atoms_obj) -> tuple[float, float]:
            mol = make_pyscf_mol(atoms_obj, basis="def2-svp")
            ref = run_pyscf_scf(
                mol, "rks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=3, conv_tol=1e-10
            )
            mf = RKS(
                mol, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf, ref["mf"])
            e_g4d = float(mf.kernel())
            assert mf.converged
            e_d3, _ = run_dftd3_bj(
                mol.atom_charges(), np.asarray(mol.atom_coords(unit="Bohr")), xc="b3lyp"
            )
            return e_g4d, float(ref["e_tot"] + e_d3)

        ed_g, ed_p = _eval_frag(cpx.dimer)
        ea_g, ea_p = _eval_frag(cpx.monomer_a)
        eb_g, eb_p = _eval_frag(cpx.monomer_b)

        eint_g = (ed_g - ea_g - eb_g) * HARTREE_TO_KCAL_MOL
        eint_p = (ed_p - ea_p - eb_p) * HARTREE_TO_KCAL_MOL
        assert abs(eint_g - eint_p) <= 0.05
        g4d_eints.append(eint_g)
        pyscf_eints.append(eint_p)

    mae, _ = calc_mae_rmse(g4d_eints, pyscf_eints)
    assert mae <= 0.05
