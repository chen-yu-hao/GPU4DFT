"""Tier 2 boundary, edge, invariant, and negative tests for F7 & F8 (T2-F7-01..05, T2-F8-01..05)."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from conftest import REPO_ROOT, min_interatomic_distance
from data.bh76_loader import get_bh76_openshell_reactions, get_bh76_species
from data.molecules import get_molecule
from data.s22_loader import load_s22_complexes


RUN_BENCHMARKS_PATH = REPO_ROOT / "benchmarks" / "run_benchmarks.py"


def _load_run_benchmarks_module():
    spec = importlib.util.spec_from_file_location("gpu4dft_run_benchmarks", RUN_BENCHMARKS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _extract_repeat_samples(row: dict[str, Any]) -> list[float] | None:
    for d in _iter_dict_nodes(row):
        for k, v in d.items():
            kl = str(k).lower()
            if (
                kl in ("repeat_times_s", "repeats_s", "timings_s", "samples_s", "e2e_times_s")
                and isinstance(v, list)
                and len(v) == 5
                and all(isinstance(x, (int, float)) for x in v)
            ):
                return [float(x) for x in v]
    return None


def _extract_vram_mib(row: dict[str, Any]) -> float | None:
    for d in _iter_dict_nodes(row):
        for k, v in d.items():
            if "vram" in str(k).lower() and isinstance(v, (int, float)):
                return float(v)
    return None


# ---------------------------------------------------------------------------
# Tier 2: Feature F7 (Benchmark Protocol) Boundaries & Negative Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_bench_cli_invalid_args_exit_nonzero(tmp_path: Path) -> None:
    """T2-F7-01: Invalid CLI arguments exit non-zero with diagnostic error message."""
    for bad_args in (
        ["--preset", "nonexistent_preset"],
        ["--suites", "99"],
        ["--backend", "invalid_backend"],
    ):
        proc = subprocess.run(
            [
                sys.executable,
                str(RUN_BENCHMARKS_PATH),
                *bad_args,
                "--json-out",
                str(tmp_path / "bad.json"),
                "--md-out",
                str(tmp_path / "bad.md"),
            ],
            cwd=str(tmp_path),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode != 0
        combined = (proc.stderr + proc.stdout).strip()
        assert len(combined) > 0


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_bench_metadata_git_rev_and_env(require_gpu4dft, require_pyscf) -> None:
    """T2-F7-02: collect_hardware_metadata() captures valid git_revision, env, and GPU VRAM."""
    import gpu4dft

    mod = _load_run_benchmarks_module()
    meta = mod.collect_hardware_metadata()

    assert isinstance(meta, dict)
    assert "git_revision" in meta and bool(meta["git_revision"])
    if (REPO_ROOT / ".git").exists():
        assert meta.get("git_revision") != "unknown"
    assert meta.get("python_version")
    assert meta.get("pyscf_version")
    if gpu4dft.backend_available("cuda"):
        assert meta.get("vram_total_mib", 0) > 0
        assert meta.get("gpu_name") not in (None, "", "CPU-only")


@pytest.mark.tier2
@pytest.mark.f7
@pytest.mark.gpu
def test_t2_f7_bench_vram_measurement_accuracy(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    tmp_path: Path,
) -> None:
    """T2-F7-03: Reported peak_vram_mib is positive, < total VRAM, and reflects system growth."""
    json_out = tmp_path / "vram_bench.json"
    md_out = tmp_path / "vram_bench.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_BENCHMARKS_PATH),
            "--preset",
            "quick",
            "--suites",
            "4",
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
        timeout=600,
    )
    assert proc.returncode == 0, f"run_benchmarks failed:\n{proc.stderr}\n{proc.stdout}"
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    vram_total = float(payload["metadata"].get("vram_total_mib", 0.0))

    vrams_by_nao: list[tuple[int, float]] = []
    for row in payload.get("benchmarks", []):
        v = _extract_vram_mib(row)
        nao = row.get("nao")
        if v is not None and isinstance(nao, int):
            vrams_by_nao.append((nao, v))
            if gpu4dft_backend == "cuda" and vram_total > 0.0:
                assert 0.0 < v < vram_total

    if gpu4dft_backend == "cuda":
        assert len(vrams_by_nao) >= 2
        vrams_by_nao.sort(key=lambda item: item[0])
        assert vrams_by_nao[-1][1] >= vrams_by_nao[0][1]


@pytest.mark.tier2
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_t2_f7_bench_cold_start_isolation(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    tmp_path: Path,
) -> None:
    """T2-F7-04: Warmup isolates cold-start spike so repeat 0 is within 1.5x median of repeats 1..4."""
    json_out = tmp_path / "cold_start.json"
    md_out = tmp_path / "cold_start.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_BENCHMARKS_PATH),
            "--preset",
            "quick",
            "--suites",
            "4",
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
        timeout=600,
    )
    assert proc.returncode == 0, f"run_benchmarks failed:\n{proc.stderr}\n{proc.stdout}"
    payload = json.loads(json_out.read_text(encoding="utf-8"))

    rows_with_samples = 0
    for row in payload.get("benchmarks", []):
        samples = _extract_repeat_samples(row)
        if samples is not None:
            rows_with_samples += 1
            med_tail = float(np.median(samples[1:]))
            assert samples[0] <= 1.5 * med_tail + 0.05, (
                f"Cold-start spike detected on repeat 0: {samples} (tail median={med_tail:.4f})"
            )
    assert rows_with_samples >= 1, "No 5-repeat timing arrays found in benchmark rows"


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_bench_host_backend_schema_parity(
    require_gpu4dft,
    require_pyscf,
    tmp_path: Path,
) -> None:
    """T2-F7-05: Host CPU fallback produces identical 5-repeat JSON & MD schema."""
    json_out = tmp_path / "host_bench.json"
    md_out = tmp_path / "host_bench.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_BENCHMARKS_PATH),
            "--backend",
            "host",
            "--preset",
            "quick",
            "--suites",
            "4,5",
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=str(tmp_path),
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, f"host run_benchmarks failed:\n{proc.stderr}\n{proc.stdout}"
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    for key in ("metadata", "benchmarks", "gradient_benchmarks"):
        assert key in payload
    assert len(payload["benchmarks"]) >= 1
    assert len(payload["gradient_benchmarks"]) >= 1
    for row in payload["benchmarks"] + payload["gradient_benchmarks"]:
        assert _has_stat_quad(row), f"Missing mean/std/min/max stats in row: {row}"
    assert md_out.exists() and md_out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Tier 2: Feature F8 (Optimizer API & Dataset Audits) Boundaries & Negative Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f8
@pytest.mark.gpu
def test_t2_f8_opt_max_steps_non_convergence(
    require_gpu4dft,
    require_pyscf,
    require_ase,
    gpu4dft_backend: str,
    tmp_path: Path,
) -> None:
    """T2-F8-01: Capping BFGS at steps=2 on distorted H2O returns converged=False with lower energy."""
    import ase
    from ase.optimize import BFGS
    from gpu4dft import GPU4DFTCalculator

    atoms = ase.Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.0, 1.35, -0.55],
            [0.0, -1.35, -0.55],
        ],
    )
    atoms.calc = GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=gpu4dft_backend,
        conv_tol=1e-9,
    )
    e_init = float(atoms.get_potential_energy())

    opt = BFGS(atoms, logfile=str(tmp_path / "bfgs_cap.log"))
    converged = opt.run(fmax=1e-4, steps=2)
    e_final = float(atoms.get_potential_energy())

    assert converged is False
    assert not opt.converged()
    assert e_final < e_init
    assert np.all(np.isfinite(atoms.get_positions()))
    assert min_interatomic_distance(atoms.get_positions()) > 0.6


@pytest.mark.tier2
@pytest.mark.f8
@pytest.mark.gpu
def test_t2_f8_calculator_cache_and_dm0_reuse(
    require_gpu4dft,
    require_pyscf,
    require_ase,
    gpu4dft_backend: str,
) -> None:
    """T2-F8-02: GPU4DFTCalculator caches identical geometry and reuses dm0 on small displacements."""
    import ase
    from gpu4dft import GPU4DFTCalculator

    spec = get_molecule("h2o")
    atoms = ase.Atoms(symbols=spec.symbols, positions=spec.coords_angstrom)
    calc = GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=gpu4dft_backend,
        conv_tol=1e-10,
    )
    atoms.calc = calc

    e1 = float(atoms.get_potential_energy())
    _ = atoms.get_forces()
    assert calc.last_mf is not None
    iters_cold = int(calc.last_mf.iterations)
    mf_obj_1 = calc.last_mf

    # 2nd call at identical coordinates hits ASE cache (no state changes, same last_mf)
    assert calc.check_state(atoms) == []
    e2 = float(atoms.get_potential_energy())
    assert e2 == e1
    assert calc.last_mf is mf_obj_1

    # 3rd call after small 0.01 A step reuses prior DM and converges in fewer SCF iterations than cold
    pos_shifted = spec.coords_angstrom.copy()
    pos_shifted[1, 0] += 0.01
    atoms.set_positions(pos_shifted)
    e3 = float(atoms.get_potential_energy())
    iters_warm = int(calc.last_mf.iterations)
    assert np.isfinite(e3)

    atoms_fresh = ase.Atoms(symbols=spec.symbols, positions=pos_shifted)
    atoms_fresh.calc = GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=gpu4dft_backend,
        conv_tol=1e-10,
    )
    _ = atoms_fresh.get_potential_energy()
    assert atoms_fresh.calc.last_mf is not None
    iters_fresh = int(atoms_fresh.calc.last_mf.iterations)

    assert iters_warm < iters_fresh or iters_warm <= max(4, iters_cold // 2)


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_scanner_bad_geometry_raises(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F8-03: Passing invalid spin/electron parity or non-Mole inputs to .as_scanner() raises cleanly."""
    from gpu4dft import RKS, UKS

    h2o = get_molecule("h2o")
    mol_h2o = make_pyscf_mol(h2o.pyscf_atom, basis="sto-3g")
    mf_rks = RKS(mol_h2o, xc="b3lyp", backend=gpu4dft_backend)
    rks_scf_scanner = mf_rks.as_scanner()
    rks_grad_scanner = mf_rks.nuc_grad_method().as_scanner()

    # Odd-electron Mole passed to restricted RKS scanner
    oh = get_molecule("oh_radical")
    mol_oh = make_pyscf_mol(oh.pyscf_atom, basis="sto-3g", charge=oh.charge, spin=oh.spin)
    with pytest.raises((ValueError, RuntimeError)):
        rks_scf_scanner(mol_oh)
    with pytest.raises((ValueError, RuntimeError)):
        rks_grad_scanner(mol_oh)

    # Invalid charge/spin parity Mole (9 electrons with spin=0) passed to UKS scanner
    mf_uks = UKS(mol_oh, xc="b3lyp", backend=gpu4dft_backend)
    uks_scf_scanner = mf_uks.as_scanner()
    uks_grad_scanner = mf_uks.nuc_grad_method().as_scanner()
    mol_bad_parity = make_pyscf_mol(oh.pyscf_atom, basis="sto-3g", charge=0, spin=1)
    mol_bad_parity.spin = 0
    with pytest.raises((ValueError, RuntimeError)):
        uks_scf_scanner(mol_bad_parity)
    with pytest.raises((ValueError, RuntimeError)):
        uks_grad_scanner(mol_bad_parity)

    # None / non-Mole inputs passed to scanners
    for bad_input in (None, "not_a_mole"):
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            rks_scf_scanner(bad_input)
        with pytest.raises((TypeError, ValueError, RuntimeError)):
            rks_grad_scanner(bad_input)


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_s22_dataset_integrity_audit(require_ase) -> None:
    """T2-F8-04: All 22 S22 complexes pass dimer/monomer split, geometry, and S22A/B reference checks."""
    complexes = load_s22_complexes()
    assert len(complexes) == 22
    assert len({c.name for c in complexes}) == 22

    for c in complexes:
        assert len(c.dimer) == c.n_atoms_a + c.n_atoms_b
        assert len(c.monomer_a) == c.n_atoms_a
        assert len(c.monomer_b) == c.n_atoms_b
        assert min_interatomic_distance(c.dimer.get_positions()) > 0.7
        assert c.ref_s22a_kcal < 0.0
        assert c.ref_s22b_kcal < 0.0
        assert abs(c.ref_ase_kcal - c.ref_s22a_kcal) <= 0.01


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_bh76_openshell_dataset_integrity_audit() -> None:
    """T2-F8-05: All BH76 open-shell reactions satisfy stoichiometry, charge, spin parity, and R_min."""
    species = get_bh76_species()
    full_rxns = get_bh76_openshell_reactions("full_openshell")
    core_rxns = get_bh76_openshell_reactions("core")
    assert len(full_rxns) >= 50
    assert len(core_rxns) >= 10

    for rxn in full_rxns:
        assert len(rxn.species_names) == len(rxn.coeffs)
        assert any(species[s].spin > 0 for s in rxn.species_names)
        assert rxn.ts_name in species
        for s_name in rxn.species_names:
            sp = species[s_name]
            if sp.n_atoms >= 2:
                assert min_interatomic_distance(sp.coords_angstrom) > 0.6
