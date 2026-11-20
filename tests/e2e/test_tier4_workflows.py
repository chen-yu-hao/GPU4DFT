"""Tier 4 realistic end-to-end application workflow tests (T4-01..06)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from conftest import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_MOL, REPO_ROOT, track_peak_vram_mib
from data.bh76_loader import get_bh76_openshell_reactions, get_bh76_species
from data.molecules import get_molecule
from data.s22_loader import load_s22_complexes
from oracles import (
    calc_mae_rmse,
    rotational_invariance_norm,
    run_dftd3_bj,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


RUN_BENCHMARKS_PATH = REPO_ROOT / "benchmarks" / "run_benchmarks.py"


def _run_geometric_opt(
    mol_init: Any,
    grad_scanner: Any,
    work_dir: Path,
    *,
    transition: bool = False,
    maxiter: int = 120,
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


def _iter_dict_nodes(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dict_nodes(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dict_nodes(item)


def _has_stat_token(keys_lower: set[str], token: str) -> bool:
    return any(
        k == token or k.startswith(token + "_") or k.endswith("_" + token)
        for k in keys_lower
    )


def _has_stat_quad(row: dict[str, Any]) -> bool:
    keys_lower = {str(k).lower() for d in _iter_dict_nodes(row) for k in d.keys()}
    return all(_has_stat_token(keys_lower, m) for m in ("mean", "std", "min", "max"))


def _extract_repeats_count(row: dict[str, Any]) -> int | None:
    for d in _iter_dict_nodes(row):
        for k, v in d.items():
            kl = str(k).lower()
            if kl in ("n_repeats", "repeats", "num_repeats") and isinstance(v, int):
                return v
            if kl in (
                "repeat_times_s",
                "repeats_s",
                "timings_s",
                "samples_s",
                "e2e_times_s",
            ) and isinstance(v, list):
                return len(v)
    return None


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f5
@pytest.mark.f8
@pytest.mark.gpu
@pytest.mark.slow
def test_t4_s22_all22_single_point_benchmark(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T4-01: All 22 S22 complexes SP B3LYP-D3(BJ)/def2-svp + RI-JK (MAE <= 0.05 kcal/mol vs PySCF)."""
    from gpu4dft import RKS

    complexes = load_s22_complexes()
    assert len(complexes) == 22
    auxbasis = "def2-universal-jkfit"

    g4d_eints: list[float] = []
    pyscf_eints: list[float] = []
    s22a_refs: list[float] = []
    s22b_refs: list[float] = []

    for cpx in complexes:
        def _eval_sp(atoms_obj) -> tuple[float, float]:
            mol = make_pyscf_mol(atoms_obj, basis="def2-svp")
            ref = run_pyscf_scf(
                mol, "rks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=3, conv_tol=1e-10
            )
            assert ref["converged"]
            mf = RKS(
                mol, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf, ref["mf"])
            e_g4d = float(mf.kernel())
            assert mf.converged
            e_d3, _ = run_dftd3_bj(
                mol.atom_charges(), np.asarray(mol.atom_coords(unit="Bohr")), xc="b3lyp"
            )
            return e_g4d, float(ref["e_tot"] + e_d3)

        ed_g, ed_p = _eval_sp(cpx.dimer)
        ea_g, ea_p = _eval_sp(cpx.monomer_a)
        eb_g, eb_p = _eval_sp(cpx.monomer_b)

        eint_g = (ed_g - ea_g - eb_g) * HARTREE_TO_KCAL_MOL
        eint_p = (ed_p - ea_p - eb_p) * HARTREE_TO_KCAL_MOL
        g4d_eints.append(eint_g)
        pyscf_eints.append(eint_p)
        s22a_refs.append(cpx.ref_s22a_kcal)
        s22b_refs.append(cpx.ref_s22b_kcal)

    mae_pyscf, rmse_pyscf = calc_mae_rmse(g4d_eints, pyscf_eints)
    mae_s22a, rmse_s22a = calc_mae_rmse(g4d_eints, s22a_refs)
    mae_s22b, rmse_s22b = calc_mae_rmse(g4d_eints, s22b_refs)

    print(
        f"\n[S22 All-22 SP B3LYP-D3(BJ)/def2-svp] "
        f"vs PySCF MAE={mae_pyscf:.6f} RMSE={rmse_pyscf:.6f} kcal/mol | "
        f"vs S22A MAE={mae_s22a:.3f} RMSE={rmse_s22a:.3f} | "
        f"vs S22B MAE={mae_s22b:.3f} RMSE={rmse_s22b:.3f}"
    )
    assert mae_pyscf <= 0.05


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f8
@pytest.mark.gpu
@pytest.mark.slow
def test_t4_s22_all22_geometry_optimizations(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T4-02: Geometry optimization across all 22 S22 complexes (max|g|<=4.5e-4, MAE<=0.05 kcal/mol vs PySCF)."""
    from gpu4dft import RKS

    complexes = load_s22_complexes()
    assert len(complexes) == 22
    auxbasis = "def2-universal-jkfit"

    g4d_eints_opt: list[float] = []
    pyscf_eints_opt: list[float] = []
    s22a_refs: list[float] = []
    s22b_refs: list[float] = []

    for cpx in complexes:
        mol_d = make_pyscf_mol(cpx.dimer, basis="def2-svp")
        mol_a = make_pyscf_mol(cpx.monomer_a, basis="def2-svp")
        mol_b = make_pyscf_mol(cpx.monomer_b, basis="def2-svp")

        mf_d = RKS(
            mol_d, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-9
        ).density_fit(auxbasis=auxbasis)
        mol_d_opt, e_d_opt, g_d_opt = _run_geometric_opt(
            mol_d, mf_d.nuc_grad_method().as_scanner(), tmp_path / f"s22_{cpx.index}_d", maxiter=120
        )
        assert np.max(np.abs(g_d_opt)) <= 4.5e-4, f"{cpx.name} dimer opt did not converge"

        mf_a = RKS(
            mol_a, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-9
        ).density_fit(auxbasis=auxbasis)
        mol_a_opt, e_a_opt, g_a_opt = _run_geometric_opt(
            mol_a, mf_a.nuc_grad_method().as_scanner(), tmp_path / f"s22_{cpx.index}_a", maxiter=100
        )
        assert np.max(np.abs(g_a_opt)) <= 4.5e-4, f"{cpx.name} monomer A opt did not converge"

        mf_b = RKS(
            mol_b, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-9
        ).density_fit(auxbasis=auxbasis)
        mol_b_opt, e_b_opt, g_b_opt = _run_geometric_opt(
            mol_b, mf_b.nuc_grad_method().as_scanner(), tmp_path / f"s22_{cpx.index}_b", maxiter=100
        )
        assert np.max(np.abs(g_b_opt)) <= 4.5e-4, f"{cpx.name} monomer B opt did not converge"

        eint_g4d = (e_d_opt - e_a_opt - e_b_opt) * HARTREE_TO_KCAL_MOL

        def _pyscf_sp_d3(mol_x) -> float:
            res = run_pyscf_scf(
                mol_x, "rks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=4, conv_tol=1e-10
            )
            e_d3, _ = run_dftd3_bj(
                mol_x.atom_charges(), np.asarray(mol_x.atom_coords(unit="Bohr")), xc="b3lyp"
            )
            return float(res["e_tot"] + e_d3)

        eint_pyscf = (_pyscf_sp_d3(mol_d_opt) - _pyscf_sp_d3(mol_a_opt) - _pyscf_sp_d3(mol_b_opt)) * HARTREE_TO_KCAL_MOL

        g4d_eints_opt.append(eint_g4d)
        pyscf_eints_opt.append(eint_pyscf)
        s22a_refs.append(cpx.ref_s22a_kcal)
        s22b_refs.append(cpx.ref_s22b_kcal)

    mae_pyscf, rmse_pyscf = calc_mae_rmse(g4d_eints_opt, pyscf_eints_opt)
    mae_s22a, rmse_s22a = calc_mae_rmse(g4d_eints_opt, s22a_refs)
    mae_s22b, rmse_s22b = calc_mae_rmse(g4d_eints_opt, s22b_refs)

    print(
        f"\n[S22 All-22 Opt B3LYP-D3(BJ)/def2-svp] "
        f"vs PySCF MAE={mae_pyscf:.6f} RMSE={rmse_pyscf:.6f} kcal/mol | "
        f"vs S22A MAE={mae_s22a:.3f} RMSE={rmse_s22a:.3f} | "
        f"vs S22B MAE={mae_s22b:.3f} RMSE={rmse_s22b:.3f}"
    )
    assert mae_pyscf <= 0.05


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f8
@pytest.mark.gpu
@pytest.mark.slow
def test_t4_bh76_openshell_barrier_heights(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T4-03: All 56 open-shell BH76 barrier heights UKS(CAM-B3LYP)/def2-svp RI-JK (MAE <= 0.1 vs PySCF)."""
    from gpu4dft import UKS

    reactions = get_bh76_openshell_reactions("full_openshell")
    species = get_bh76_species()
    auxbasis = "def2-universal-jkfit"

    needed_species = sorted({s for r in reactions for s in r.species_names})
    g4d_energies: dict[str, float] = {}
    pyscf_energies: dict[str, float] = {}
    s2_diffs: list[float] = []

    for sp_name in needed_species:
        sp = species[sp_name]
        mol = make_pyscf_mol(sp.pyscf_atom, basis="def2-svp", charge=sp.charge, spin=sp.spin)
        ref = run_pyscf_scf(
            mol, "uks", xc="cam-b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=3, conv_tol=1e-10
        )
        assert ref["converged"], f"PySCF failed on BH76 species {sp_name}"

        mf = UKS(
            mol, xc="cam-b3lyp", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
        ).density_fit(auxbasis=auxbasis)
        match_grids(mf, ref["mf"])
        e_g4d = float(mf.kernel())
        assert mf.converged, f"GPU4DFT failed on BH76 species {sp_name}"

        ss_got, _ = mf.spin_square()
        if ref["spin_square"] is not None:
            s2_diffs.append(abs(ss_got - ref["spin_square"][0]))

        g4d_energies[sp_name] = e_g4d
        pyscf_energies[sp_name] = float(ref["e_tot"])

    barriers_g4d: list[float] = []
    barriers_pyscf: list[float] = []
    refs_w2f12: list[float] = []
    rxn_ids: list[str] = []

    for rxn in reactions:
        b_g4d = sum(c * g4d_energies[s] for s, c in zip(rxn.species_names, rxn.coeffs)) * HARTREE_TO_KCAL_MOL
        b_pyscf = sum(c * pyscf_energies[s] for s, c in zip(rxn.species_names, rxn.coeffs)) * HARTREE_TO_KCAL_MOL
        barriers_g4d.append(b_g4d)
        barriers_pyscf.append(b_pyscf)
        refs_w2f12.append(rxn.ref_barrier_kcal)
        rxn_ids.append(rxn.id)

    mae_pyscf, rmse_pyscf = calc_mae_rmse(barriers_g4d, barriers_pyscf)
    mae_lit, rmse_lit = calc_mae_rmse(barriers_g4d, refs_w2f12)
    max_s2_diff = max(s2_diffs) if s2_diffs else 0.0

    print(
        f"\n[BH76 Open-Shell SP ({len(reactions)} reactions): {', '.join(rxn_ids)}] "
        f"vs PySCF MAE={mae_pyscf:.6f} RMSE={rmse_pyscf:.6f} kcal/mol | "
        f"max|d<S^2>|={max_s2_diff:.2e} | "
        f"vs W2-F12 MAE={mae_lit:.3f} RMSE={rmse_lit:.3f} kcal/mol"
    )
    assert mae_pyscf <= 0.1
    assert max_s2_diff <= 1e-5


@pytest.mark.tier4
@pytest.mark.f1
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f8
@pytest.mark.gpu
@pytest.mark.slow
def test_t4_bh76_openshell_geometry_optimizations(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T4-04: Reactant/product/TS geometry opts on documented core BH76 subset (MAE <= 0.1 vs PySCF)."""
    from gpu4dft import UKS

    core_rxns = get_bh76_openshell_reactions("core")
    species = get_bh76_species()
    auxbasis = "def2-universal-jkfit"

    ts_names = {r.ts_name for r in core_rxns}
    needed_species = sorted({s for r in core_rxns for s in r.species_names})

    opt_mols: dict[str, Any] = {}
    opt_e_g4d: dict[str, float] = {}
    opt_e_pyscf: dict[str, float] = {}

    for sp_name in needed_species:
        sp = species[sp_name]
        mol = make_pyscf_mol(sp.pyscf_atom, basis="def2-svp", charge=sp.charge, spin=sp.spin)
        mf = UKS(
            mol, xc="cam-b3lyp", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
        ).density_fit(auxbasis=auxbasis)

        if sp.n_atoms == 1:
            e_sp = float(mf.kernel())
            assert mf.converged
            mol_opt = mol
            e_opt = e_sp
        else:
            is_ts = sp_name in ts_names
            mol_opt, e_opt, g_opt = _run_geometric_opt(
                mol,
                mf.nuc_grad_method().as_scanner(),
                tmp_path / f"bh76_core_{sp_name}",
                transition=is_ts,
                maxiter=120,
            )
            assert np.max(np.abs(g_opt)) <= 4.5e-4, f"BH76 {sp_name} opt failed to converge"

        opt_mols[sp_name] = mol_opt
        opt_e_g4d[sp_name] = e_opt

        res = run_pyscf_scf(
            mol_opt, "uks", xc="cam-b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=4, conv_tol=1e-10
        )
        assert res["converged"]
        opt_e_pyscf[sp_name] = float(res["e_tot"])

    barriers_g4d: list[float] = []
    barriers_pyscf: list[float] = []
    refs_w2f12: list[float] = []
    rxn_ids: list[str] = []

    for rxn in core_rxns:
        b_g4d = sum(c * opt_e_g4d[s] for s, c in zip(rxn.species_names, rxn.coeffs)) * HARTREE_TO_KCAL_MOL
        b_pyscf = sum(c * opt_e_pyscf[s] for s, c in zip(rxn.species_names, rxn.coeffs)) * HARTREE_TO_KCAL_MOL
        barriers_g4d.append(b_g4d)
        barriers_pyscf.append(b_pyscf)
        refs_w2f12.append(rxn.ref_barrier_kcal)
        rxn_ids.append(rxn.id)

    mae_pyscf, rmse_pyscf = calc_mae_rmse(barriers_g4d, barriers_pyscf)
    mae_lit, rmse_lit = calc_mae_rmse(barriers_g4d, refs_w2f12)

    print(
        f"\n[BH76 Core Opt ({len(core_rxns)} reactions): {', '.join(rxn_ids)}] "
        f"vs PySCF MAE={mae_pyscf:.6f} RMSE={rmse_pyscf:.6f} kcal/mol | "
        f"vs W2-F12 MAE={mae_lit:.3f} RMSE={rmse_lit:.3f} kcal/mol"
    )
    assert mae_pyscf <= 0.1


@pytest.mark.tier4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_t4_large_molecules_rijk_scf_and_grad(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T4-05: Large molecules (H2O)_32, C60, Taxol, Valinomycin RI-JK SCF + grad end-to-end."""
    from gpu4dft import RKS

    auxbasis = "def2-universal-jkfit"
    targets = (
        ("h2o_32", "def2-svp", 768),
        ("c60", "def2-svp", 840),
        ("taxol", "6-31g*", 970),
        ("valinomycin", "6-31g*", 1272),
    )

    for mol_name, basis, expected_nao in targets:
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis=basis)
        assert mol.nao_nr() == expected_nao
        assert 500 <= expected_nao <= 2000

        # Prefer GPU4PySCF DF reference when available (fast for Taxol/Valinomycin), fallback to CPU PySCF DF
        e_ref: float
        g_ref: np.ndarray
        ref_grids_tuple: tuple[np.ndarray, np.ndarray] | None = None
        try:
            import cupy as cp  # type: ignore
            from gpu4pyscf import dft as g4p_dft  # type: ignore

            mf_ref = g4p_dft.RKS(mol, xc="b3lyp").density_fit(auxbasis=auxbasis)
            mf_ref.grids.level = 5
            mf_ref.conv_tol = 1e-9
            e_ref = float(mf_ref.kernel())
            assert mf_ref.converged
            gd = mf_ref.nuc_grad_method()
            gd.grid_response = True
            gd.auxbasis_response = True
            g_ref = np.asarray(gd.kernel(), dtype=np.float64)
            coords = mf_ref.grids.coords
            weights = mf_ref.grids.weights
            ref_grids_tuple = (
                np.ascontiguousarray(
                    cp.asnumpy(coords) if hasattr(coords, "get") else coords, dtype=np.float64
                ),
                np.ascontiguousarray(
                    cp.asnumpy(weights) if hasattr(weights, "get") else weights, dtype=np.float64
                ),
            )
        except Exception:
            ref = run_pyscf_scf(
                mol,
                "rks",
                xc="b3lyp",
                density_fit=True,
                auxbasis=auxbasis,
                grids_level=5,
                conv_tol=1e-9,
            )
            assert ref["converged"]
            e_ref = float(ref["e_tot"])
            g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)
            ref_grids_tuple = (
                np.ascontiguousarray(ref["mf"].grids.coords, dtype=np.float64),
                np.ascontiguousarray(ref["mf"].grids.weights, dtype=np.float64),
            )

        mf_scf = RKS(
            mol,
            xc="b3lyp",
            backend=gpu4dft_backend,
            precision="fp64",
            grids=ref_grids_tuple,
            conv_tol=1e-9,
        ).density_fit(auxbasis=auxbasis)
        e_got = float(mf_scf.kernel())
        assert mf_scf.converged
        assert abs(e_got - e_ref) <= 1e-6

        with track_peak_vram_mib() as vram:
            mf_grad = RKS(
                mol,
                xc="b3lyp",
                backend=gpu4dft_backend,
                precision="fp64",
                grids_level=5,
                conv_tol=1e-9,
            ).density_fit(auxbasis=auxbasis)
            mf_grad.kernel()
            assert mf_grad.converged
            grad = np.asarray(mf_grad.nuc_grad_method().kernel(), dtype=np.float64)

        assert grad.shape == (spec.n_atoms, 3)
        assert np.max(np.abs(grad - g_ref)) <= 1e-4
        assert translational_invariance_norm(grad) <= 1e-6
        assert rotational_invariance_norm(spec.coords_bohr, grad) <= 1e-5
        if gpu4dft_backend == "cuda" and vram["total_mib"] > 0.0:
            assert vram["used_after_mib"] < vram["total_mib"]
    _ = match_grids


@pytest.mark.tier4
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_t4_fair_5repeat_benchmark_pipeline(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    tmp_path: Path,
) -> None:
    """T4-06: End-to-end fair 5-repeat benchmark pipeline with large molecules, JSON & MD reports."""
    json_out = tmp_path / "full_5repeat_bench.json"
    md_out = tmp_path / "full_5repeat_bench.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_BENCHMARKS_PATH),
            "--preset",
            "standard",
            "--suites",
            "4,5",
            "--backend",
            gpu4dft_backend,
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=str(tmp_path),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert proc.returncode == 0, f"run_benchmarks failed:\n{proc.stderr}\n{proc.stdout}"
    assert json_out.exists() and json_out.stat().st_size > 0
    assert md_out.exists() and md_out.stat().st_size > 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    meta = payload.get("metadata", {})
    assert "shared_grids" in meta and meta["shared_grids"] is False
    assert "git_revision" in meta and bool(meta["git_revision"])
    if (REPO_ROOT / ".git").exists():
        assert meta.get("git_revision") != "unknown"

    bench_rows = payload.get("benchmarks", [])
    grad_rows = payload.get("gradient_benchmarks", [])
    assert len(bench_rows) >= 1 and len(grad_rows) >= 1

    # Verify all 4 target large systems appear in the standard benchmark report
    seen_systems = {
        str(r.get("system", "")).lower().replace("(", "").replace(")", "").replace("_", "")
        for r in bench_rows + grad_rows
    }
    for target in ("taxol", "valinomycin", "c60", "h2o32"):
        assert target in seen_systems, f"Missing {target} in standard benchmark report: {seen_systems}"

    for row in bench_rows + grad_rows:
        assert _extract_repeats_count(row) == 5, f"Expected 5 repeats in {row}"
        assert _has_stat_quad(row), f"Missing mean/std/min/max stats in {row}"
        keys_flat = {str(k).lower(): v for d in _iter_dict_nodes(row) for k, v in d.items()}
        assert any("vram" in k for k in keys_flat)
        assert any("speedup" in k for k in keys_flat)

    md_text = md_out.read_text(encoding="utf-8")
    assert "Taxol" in md_text or "taxol" in md_text.lower()
    assert "Valinomycin" in md_text or "valinomycin" in md_text.lower()
    assert "C60" in md_text
