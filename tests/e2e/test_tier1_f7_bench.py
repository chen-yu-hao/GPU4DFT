"""Tier 1 E2E tests for Feature F7: Fair 5-Repeat Benchmark Protocol & Large Molecules (T1-F7-01..05)."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from conftest import REPO_ROOT, track_peak_vram_mib
from data.molecules import get_molecule
from oracles import (
    rotational_invariance_norm,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


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


@pytest.mark.tier1
@pytest.mark.f7
def test_f7_large_molecule_catalog_nao_range(require_gpu4dft, require_pyscf) -> None:
    """T1-F7-01: Taxol, Valinomycin, C60, (H2O)_32 exist in run_benchmarks catalog with 500<=NAO<=2000."""
    mod = _load_run_benchmarks_module()
    catalog = mod.MOLECULE_CATALOG
    canon_map = {str(k).lower().replace("(", "").replace(")", "").replace("_", ""): k for k in catalog}

    expected_targets = {
        "taxol": ("6-31g*", 113),
        "valinomycin": ("6-31g*", 168),
        "c60": ("def2-svp", 60),
        "h2o32": ("def2-svp", 96),
    }

    for target_norm, (basis, expected_natm) in expected_targets.items():
        assert target_norm in canon_map, f"Missing {target_norm} in MOLECULE_CATALOG: {list(catalog)}"
        key = canon_map[target_norm]
        mol = mod.build_mol(key, basis)
        assert mol.natm == expected_natm
        assert 500 <= mol.nao <= 2000


@pytest.mark.tier1
@pytest.mark.f7
@pytest.mark.gpu
def test_f7_no_shared_grids_and_symmetric_warmup(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    tmp_path: Path,
) -> None:
    """T1-F7-02: AST/CLI audit proves zero shared_grids injection and symmetric warmup."""
    source = RUN_BENCHMARKS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn_name = ""
            if isinstance(node.func, ast.Attribute):
                fn_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                fn_name = node.func.id
            if fn_name in ("RKS", "UKS"):
                kw_names = {kw.arg for kw in node.keywords if kw.arg is not None}
                assert "grids" not in kw_names, f"Found forbidden shared grids= injection in {fn_name}"

        assign_targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            assign_targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            assign_targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            assign_targets = [node.target]
        for tgt in assign_targets:
            for sub in ast.walk(tgt):
                if (
                    isinstance(sub, ast.Attribute)
                    and sub.attr in ("coords", "weights")
                    and isinstance(sub.value, ast.Attribute)
                    and sub.value.attr == "grids"
                ):
                    raise AssertionError(
                        f"Found forbidden shared .grids.{sub.attr} assignment in run_benchmarks.py"
                    )

    warmup_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "warmup_contexts"),
        None,
    )
    assert warmup_fn is not None, "warmup_contexts function not found"
    warmup_src = ast.get_source_segment(source, warmup_fn) or ""
    assert "density_fit" in warmup_src, "warmup_contexts must warm GPU4DFT RI-JK path"
    assert "nuc_grad_method" in warmup_src, "warmup_contexts must warm gradient path"
    assert ("dft.RKS" in warmup_src or "pyscf" in warmup_src.lower()), (
        "warmup_contexts must also warm CPU PySCF"
    )

    json_out = tmp_path / "bench_meta.json"
    md_out = tmp_path / "bench_meta.md"
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
    data = json.loads(json_out.read_text(encoding="utf-8"))
    meta = data.get("metadata", {})
    assert "shared_grids" in meta and meta["shared_grids"] is False


@pytest.mark.tier1
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_f7_five_repeat_json_and_md_schema(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    tmp_path: Path,
) -> None:
    """T1-F7-03: 5-repeat benchmark produces valid JSON + MD with mean, std, min, max, VRAM, diff, speedup."""
    json_out = tmp_path / "bench_quick.json"
    md_out = tmp_path / "bench_quick.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(RUN_BENCHMARKS_PATH),
            "--preset",
            "quick",
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
        timeout=900,
    )
    assert proc.returncode == 0, f"run_benchmarks failed:\n{proc.stderr}\n{proc.stdout}"
    assert json_out.exists() and json_out.stat().st_size > 0
    assert md_out.exists() and md_out.stat().st_size > 0

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert "metadata" in payload
    assert len(payload.get("benchmarks", [])) >= 1
    assert len(payload.get("gradient_benchmarks", [])) >= 1

    for section in ("benchmarks", "gradient_benchmarks"):
        for row in payload[section]:
            n_rep = _extract_repeats_count(row)
            assert n_rep == 5, f"Expected 5 repeats in {section} row, got {n_rep}: {row}"
            assert _has_stat_quad(row), f"Missing mean/std/min/max stats in {section} row: {row}"

            keys_flat = {str(k).lower(): v for d in _iter_dict_nodes(row) for k, v in d.items()}
            assert any("vram" in k for k in keys_flat), f"Missing VRAM metric in {row}"
            assert any("speedup" in k for k in keys_flat), f"Missing speedup metric in {row}"
            if section == "benchmarks":
                assert any("diff" in k for k in keys_flat), f"Missing abs_diff metric in {row}"

    md_text = md_out.read_text(encoding="utf-8").lower()
    assert "mean" in md_text and "std" in md_text


@pytest.mark.tier1
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_f7_large_mol_h2o32_c60_ri_scf_grad(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F7-04: (H2O)_32 and C60 RI-JK matched-grid SCF (<=1e-6) + native lvl5 grad (<=1e-4, inv)."""
    from gpu4dft import RKS

    auxbasis = "def2-universal-jkfit"
    for mol_name, expected_nao in (("h2o_32", 768), ("c60", 840)):
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp")
        assert mol.nao_nr() == expected_nao

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
        g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

        mf_scf = RKS(
            mol,
            xc="b3lyp",
            backend=gpu4dft_backend,
            precision="fp64",
            conv_tol=1e-9,
        ).density_fit(auxbasis=auxbasis)
        match_grids(mf_scf, ref["mf"])

        e_got = mf_scf.kernel()
        assert mf_scf.converged
        assert abs(e_got - ref["e_tot"]) <= 1e-6

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
        assert grad.shape == (mol.natm, 3)
        assert np.max(np.abs(grad - g_ref)) <= 1e-4
        assert translational_invariance_norm(grad) <= 1e-6
        assert rotational_invariance_norm(spec.coords_bohr, grad) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_f7_large_mol_taxol_valinomycin_ri_scf_grad(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F7-05: Taxol and Valinomycin (6-31g*) RI-JK SCF + grad complete within VRAM and invariants."""
    from gpu4dft import RKS

    auxbasis = "def2-universal-jkfit"
    for mol_name, expected_nao in (("taxol", 970), ("valinomycin", 1272)):
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis="6-31g*")
        assert mol.nao_nr() == expected_nao

        with track_peak_vram_mib() as vram:
            mf = RKS(
                mol,
                xc="b3lyp",
                backend=gpu4dft_backend,
                precision="fp64",
                grids_level=3,
                conv_tol=1e-9,
            ).density_fit(auxbasis=auxbasis)
            e_got = mf.kernel()
            assert mf.converged
            grad = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)

        assert np.isfinite(e_got)
        assert grad.shape == (mol.natm, 3)
        assert np.all(np.isfinite(grad))
        assert translational_invariance_norm(grad) <= 1e-6
        assert rotational_invariance_norm(spec.coords_bohr, grad) <= 1e-6
        if gpu4dft_backend == "cuda" and vram["total_mib"] > 0.0:
            assert vram["used_after_mib"] < vram["total_mib"]
