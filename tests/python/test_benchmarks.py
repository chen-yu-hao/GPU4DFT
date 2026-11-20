"""Fast deterministic protocol, AST fairness, VRAM monitor, and schema tests for M5 benchmarks."""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_BENCHMARKS_PATH = REPO_ROOT / "benchmarks" / "run_benchmarks.py"


def _load_run_benchmarks() -> Any:
    spec = importlib.util.spec_from_file_location("gpu4dft_run_benchmarks_test", RUN_BENCHMARKS_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_large_molecule_catalog_nao_range() -> None:
    mod = _load_run_benchmarks()
    expected = {
        "(H2O)_32": ("def2-svp", 96, 768),
        "C60": ("def2-svp", 60, 840),
        "Taxol": ("6-31g*", 113, 970),
        "Valinomycin": ("6-31g*", 168, 1272),
    }
    for name, (basis, natm, nao) in expected.items():
        assert name in mod.MOLECULE_CATALOG, f"Missing {name} in MOLECULE_CATALOG"
        mol = mod.build_mol(name, basis)
        assert mol.natm == natm
        assert mol.nao == nao
        assert 500 <= mol.nao <= 2000

    assert (REPO_ROOT / "benchmarks" / "geometries" / "taxol.xyz").is_file()
    assert (REPO_ROOT / "benchmarks" / "geometries" / "valinomycin.xyz").is_file()


def test_no_shared_grids_and_symmetric_warmup_ast() -> None:
    source = RUN_BENCHMARKS_PATH.read_text(encoding="utf-8")
    assert "tests.e2e" not in source and "tests/e2e" not in source
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
                assert "grids" not in kw_names, f"Forbidden shared grids= kwarg in {fn_name}"

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
                    raise AssertionError(f"Forbidden .grids.{sub.attr} assignment in run_benchmarks.py")

    warmup_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "warmup_contexts"),
        None,
    )
    assert warmup_fn is not None
    warmup_src = ast.get_source_segment(source, warmup_fn) or ""
    for token in (
        "gpu4dft.RHF",
        "gpu4dft.RKS",
        "gpu4pyscf_dft.RKS",
        "dft.RKS",
        "scf.RHF",
        "density_fit",
        "nuc_grad_method",
        "grid_response",
        "auxbasis_response",
        '"fp64"',
        '"ozaki_int8"',
    ):
        assert token in warmup_src, f"Missing symmetric warmup token {token!r} in warmup_contexts"

    eri_fn = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_eri_throughput_suite"),
        None,
    )
    assert eri_fn is not None
    eri_src = ast.get_source_segment(source, eri_fn) or ""
    assert 'precision="fp64"' in eri_src and 'precision="fp32"' in eri_src


def test_summarize_samples_and_vram_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_run_benchmarks()

    stats = mod._summarize_samples([2.0, 4.0, 4.0, 4.0, 6.0])
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["std"] == pytest.approx(math.sqrt(2.0))
    assert stats["min"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(6.0)

    single = mod._summarize_samples([3.5])
    assert single == {"mean": 3.5, "std": 0.0, "min": 3.5, "max": 3.5}

    empty = mod._summarize_samples([])
    assert empty == {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    seq = iter([100.0, 250.0, 420.0, 180.0])
    last_val = [180.0]

    def _fake_query() -> float:
        try:
            v = next(seq)
            last_val[0] = v
            return v
        except StopIteration:
            return last_val[0]

    monkeypatch.setattr(mod, "_query_vram_used_mib", _fake_query)
    with mod.VramPeakMonitor(poll_gpu=True, poll_interval_s=0.002) as vmon:
        for _ in range(4):
            vmon._sample()
    assert vmon.baseline_mib == pytest.approx(100.0)
    assert vmon.peak_vram_mib == pytest.approx(420.0)
    assert vmon.peak_vram_delta_mib == pytest.approx(320.0)

    with mod.VramPeakMonitor(poll_gpu=False) as vmon_off:
        pass
    assert vmon_off.peak_vram_mib is None
    assert vmon_off.peak_vram_delta_mib is None


def test_five_repeat_metadata_schema_and_markdown_report() -> None:
    mod = _load_run_benchmarks()
    meta = mod.collect_hardware_metadata(
        n_repeats=5,
        grids_level=3,
        auxbasis="def2-universal-jkfit",
        warmup_info={
            "warmup_performed": True,
            "warmup_backends": ["GPU4DFT (cuda)", "GPU4PySCF", "PySCF (CPU)"],
            "warmup_paths": ["gpu4dft_rijk_fp64_scf_grad", "gpu4pyscf_df_scf_grad", "pyscf_cpu_df_scf_grad"],
        },
    )
    assert meta["shared_grids"] is False
    assert meta["n_repeats"] == 5
    assert meta["warmup_performed"] is True
    assert meta["grids_level"] == 3
    assert meta["auxbasis"] == "def2-universal-jkfit"
    assert meta["precision"] == "fp64"
    assert meta["conv_tol_scf"] == pytest.approx(1e-9)
    assert meta["conv_tol_grad"] == pytest.approx(1e-10)

    bench_rows = []
    grad_rows = []
    for sys_name, basis_name, nao in (
        ("(H2O)_32", "def2-svp", 768),
        ("C60", "def2-svp", 840),
        ("Taxol", "6-31g*", 970),
        ("Valinomycin", "6-31g*", 1272),
    ):
        bench_rows.append(
            {
                "system": sys_name,
                "basis": basis_name,
                "nao": nao,
                "method": "B3LYP",
                "jk_mode": "RI_JK_FP64",
                "iterations": 12,
                "n_repeats": 5,
                "repeat_times_s": [1.0, 1.02, 0.99, 1.01, 1.03],
                "abs_diff_vs_pyscf_eh": 1.2e-5,
                "time_setup_s": 0.12,
                "time_scf_s": 0.89,
                "time_e2e_s": 1.01,
                "time_e2e_mean_s": 1.01,
                "time_e2e_std_s": 0.0158,
                "time_e2e_min_s": 0.99,
                "time_e2e_max_s": 1.03,
                "peak_vram_mib": 2400.0,
                "pyscf_cpu_time_s": 40.0,
                "gpu4pyscf_time_s": 1.8,
                "speedup_vs_pyscf_cpu": 39.6,
                "speedup_e2e_vs_gpu4pyscf": 1.78,
                "speedup_scf_loop_vs_gpu4pyscf": 2.02,
            }
        )
        grad_rows.append(
            {
                "system": sys_name,
                "basis": basis_name,
                "nao": nao,
                "method": "B3LYP",
                "jk_mode": "RI_JK_FP64",
                "n_repeats": 5,
                "repeat_times_s": [0.80, 0.81, 0.79, 0.80, 0.82],
                "time_grad_s": 0.804,
                "time_grad_mean_s": 0.804,
                "time_grad_std_s": 0.0114,
                "time_grad_min_s": 0.79,
                "time_grad_max_s": 0.82,
                "grad_to_scf_ratio": 0.80,
                "peak_vram_mib": 2600.0,
                "max_grad_diff_eh_bohr": 2.0e-5,
                "net_translation_force_eh_bohr": 1.5e-9,
                "pyscf_cpu_grad_s": 50.0,
                "gpu4pyscf_grad_s": 1.40,
                "speedup_vs_pyscf_cpu": 62.19,
                "speedup_vs_gpu4pyscf": 1.74,
            }
        )

    payload = {
        "metadata": meta,
        "gemm_benchmarks": [],
        "eri_throughput": [],
        "solver_ablation": [],
        "benchmarks": bench_rows,
        "gradient_benchmarks": grad_rows,
    }
    md = mod.render_markdown_report(payload)
    for token in (
        "shared_grids=False",
        "Mean ± Std (s)",
        "Min / Max (s)",
        "Peak VRAM (MiB)",
        "Speedup vs GPU4PySCF",
        "(H2O)_32",
        "C60",
        "Taxol",
        "Valinomycin",
    ):
        assert token in md


def test_cli_invalid_args_fail_fast(tmp_path: Path) -> None:
    for bad in (
        ["--preset", "nonexistent"],
        ["--suites", "99"],
        ["--backend", "invalid_backend"],
        ["--repeats", "0"],
    ):
        proc = subprocess.run(
            [
                sys.executable,
                str(RUN_BENCHMARKS_PATH),
                *bad,
                "--json-out",
                str(tmp_path / "out.json"),
                "--md-out",
                str(tmp_path / "out.md"),
            ],
            cwd=str(tmp_path),
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode != 0
        assert (proc.stderr + proc.stdout).strip()
