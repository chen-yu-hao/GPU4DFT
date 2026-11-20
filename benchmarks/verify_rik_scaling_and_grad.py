#!/usr/bin/env python3
"""M5 Occupied-Space RI-K Scaling, SCF Energy Parity, and RI-JK Analytical Gradient Verification.

Generates:
  - benchmarks/results/scaling_and_accuracy.json
  - benchmarks/results/SCALING_AND_ACCURACY.md

Verifies:
  1. Matched-grid FP64 RI-K SCF energy parity <= 1e-6 Eh vs PySCF DF across
     RHF, RKS (B3LYP, PBE0, CAM-B3LYP, wB97X), UHF, UKS (B3LYP, CAM-B3LYP, wB97X),
     plus (H2O)_32 and C60.
  2. Empirical O(N_aux * N_AO^2 * N_occ) scaling across (H2O)_n (n = 4..80, nao up to 1920)
     and fixed-N_occ (H2O)_4 basis ladder (sto-3g, 6-31g*, def2-svp, def2-tzvp), measuring
     isolated RIJKEngine.compute_jk_mo (K-only) and per-iteration SCF time.
  3. RI-JK analytical nuclear gradients:
     - Translational invariance <= 1e-6 Eh/a0 on small systems + Taxol, Valinomycin, C60, (H2O)_32
     - Matched-grid grid_response=False parity <= 1e-5 Eh/a0 vs PySCF DF (RHF, UHF, RKS, UKS, RSH)
     - Native-grid grid_response=True central FD parity <= 1e-5 Eh/a0
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))

import gpu4dft  # noqa: E402
from pyscf import dft, gto, scf  # noqa: E402
from run_benchmarks import (  # noqa: E402
    VramPeakMonitor,
    _water_cluster_geom,
    build_mol,
    collect_hardware_metadata,
    release_gpu_memory,
    sync_gpu,
)

BOHR_TO_ANG = 0.529177210903
AUXBASIS = "def2-universal-jkfit"

SMALL_GEOMS: dict[str, tuple[str, int, int]] = {
    "H2O": ("O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692", 0, 0),
    "CH2O": ("C 0.0 0.0 -0.5288; O 0.0 0.0 0.6780; H 0.0 0.9355 -1.1160; H 0.0 -0.9355 -1.1160", 0, 0),
    "HO2": ("O 0.0 0.0 0.0; O 0.0 0.0 1.33; H 0.92 0.0 -0.30", 0, 1),
    "O2": ("O 0.0 0.0 -0.605; O 0.0 0.0 0.605", 0, 2),
}


def _make_mol(atom: str, basis: str, charge: int = 0, spin: int = 0) -> gto.Mole:
    mol = gto.M(atom=atom, basis=basis, charge=charge, spin=spin, unit="Angstrom", cart=False, verbose=0)
    mol.max_memory = 64000
    return mol


def _make_pyscf_scf(mol: gto.Mole, method: str, xc: str | None, grids_level: int = 4, conv_tol: float = 1e-10) -> Any:
    m = method.lower()
    if m == "rhf":
        mf = scf.RHF(mol)
    elif m == "uhf":
        mf = scf.UHF(mol)
    elif m == "rks":
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = grids_level
    elif m == "uks":
        mf = dft.UKS(mol, xc=xc)
        mf.grids.level = grids_level
    else:
        raise ValueError(f"Unsupported method: {method}")
    mf = mf.density_fit(auxbasis=AUXBASIS)
    mf.conv_tol = conv_tol
    mf.diis_space = 8
    mf.direct_scf_tol = 1e-13
    return mf


def _make_gpu4dft_scf(
    mol: gto.Mole,
    method: str,
    xc: str | None,
    backend: str,
    grids: tuple[np.ndarray, np.ndarray] | None = None,
    grids_level: int = 4,
    conv_tol: float = 1e-10,
    diis_tol: float | None = None,
) -> Any:
    m = method.lower()
    kw: dict[str, Any] = dict(
        backend=backend,
        precision="fp64",
        dynamic_precision=False,
        gemm_mode="fp64",
        solver="diag",
        conv_tol=conv_tol,
        diis_space=8,
        schwarz_cutoff=1e-13,
        init_guess="minao",
    )
    if diis_tol is not None:
        kw["diis_tol"] = diis_tol
    if m == "rhf":
        mf = gpu4dft.RHF(mol, **kw)
    elif m == "uhf":
        mf = gpu4dft.UHF(mol, **kw)
    elif m in ("rks", "uks"):
        if grids is not None:
            kw["grids"] = grids
        else:
            kw["grids_level"] = grids_level
        cls = gpu4dft.RKS if m == "rks" else gpu4dft.UKS
        mf = cls(mol, xc=xc, **kw)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return mf.density_fit(auxbasis=AUXBASIS)


def _log_log_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    lx = np.log(np.asarray(xs, dtype=np.float64))
    ly = np.log(np.asarray(ys, dtype=np.float64))
    slope, intercept = np.polyfit(lx, ly, 1)
    y_pred = slope * lx + intercept
    ss_res = float(np.sum((ly - y_pred) ** 2))
    ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return float(slope), float(r2)


# ---------------------------------------------------------------------------
# 1. Matched-Grid FP64 RI-K SCF Energy Parity (<= 1e-6 Eh)
# ---------------------------------------------------------------------------


def verify_energy_parity(backend: str, checkpoint_cb: Any = None) -> list[dict[str, Any]]:
    print("\n=== [1/3] FP64 RI-K Matched-Grid SCF Energy Parity (<= 1e-6 Eh) ===")
    cases: list[tuple[str, str, str, str | None, int]] = [
        ("H2O", "def2-svp", "rhf", None, 4),
        ("C6H6", "def2-svp", "rhf", None, 4),
        ("H2O", "def2-svp", "rks", "b3lyp", 4),
        ("H2O", "def2-svp", "rks", "pbe0", 4),
        ("CH2O", "def2-svp", "rks", "cam-b3lyp", 4),
        ("CH2O", "def2-svp", "rks", "wb97x", 4),
        ("HO2", "def2-svp", "uhf", None, 4),
        ("O2", "def2-svp", "uhf", None, 4),
        ("HO2", "def2-svp", "uks", "b3lyp", 4),
        ("HO2", "def2-svp", "uks", "cam-b3lyp", 4),
        ("HO2", "def2-svp", "uks", "wb97x", 4),
        ("(H2O)_32", "def2-svp", "rks", "b3lyp", 3),
        ("C60", "def2-svp", "rks", "b3lyp", 3),
    ]

    rows: list[dict[str, Any]] = []
    for sys_name, basis_name, method, xc_name, g_lvl in cases:
        if sys_name in SMALL_GEOMS:
            atom, charge, spin = SMALL_GEOMS[sys_name]
            mol = _make_mol(atom, basis_name, charge=charge, spin=spin)
        else:
            mol = build_mol(sys_name, basis_name)
        unrestricted = method.lower() in ("uhf", "uks")
        dm0 = gpu4dft.minao_init_guess(mol, unrestricted=unrestricted)

        mf_ref = _make_pyscf_scf(mol, method, xc_name, grids_level=g_lvl, conv_tol=1e-9)
        if method.lower() in ("rks", "uks"):
            mf_ref.grids.build()
        e_ref = float(mf_ref.kernel(dm0=dm0))
        s2_ref = float(mf_ref.spin_square()[0]) if unrestricted else 0.0

        matched_grids = None
        if method.lower() in ("rks", "uks"):
            matched_grids = (
                np.ascontiguousarray(mf_ref.grids.coords, dtype=np.float64),
                np.ascontiguousarray(mf_ref.grids.weights, dtype=np.float64),
            )
        del mf_ref
        gc.collect()

        release_gpu_memory()
        mf_g4d = _make_gpu4dft_scf(
            mol,
            method,
            xc_name,
            backend=backend,
            grids=matched_grids,
            grids_level=g_lvl,
            conv_tol=1e-9,
        )
        e_g4d = float(mf_g4d.kernel(dm0=dm0))
        s2_g4d = float(mf_g4d.spin_square()[0]) if unrestricted else 0.0
        converged = bool(mf_g4d.converged)
        release_gpu_memory(mf_g4d)

        abs_de = abs(e_g4d - e_ref)
        abs_ds2 = abs(s2_g4d - s2_ref)
        passed = converged and (abs_de <= 1e-6) and (not unrestricted or abs_ds2 <= 1e-5)

        rec = {
            "system": sys_name,
            "basis": basis_name,
            "nao": int(mol.nao),
            "method": method.upper(),
            "xc": xc_name.upper() if xc_name else "HF",
            "converged": converged,
            "energy_gpu4dft_eh": e_g4d,
            "energy_pyscf_df_eh": e_ref,
            "abs_diff_eh": abs_de,
            "s2_gpu4dft": s2_g4d if unrestricted else None,
            "s2_pyscf_df": s2_ref if unrestricted else None,
            "abs_s2_diff": abs_ds2 if unrestricted else None,
            "passed": bool(passed),
        }
        rows.append(rec)
        if checkpoint_cb is not None:
            checkpoint_cb(rows)
        s2_str = f", |dS2|={abs_ds2:.2e}" if unrestricted else ""
        print(
            f"  [{'PASS' if passed else 'FAIL'}] {sys_name:10s} {method.upper()}/{(xc_name or 'HF').upper():9s} "
            f"(nao={mol.nao:4d}): |dE|={abs_de:.2e} Eh{s2_str}"
        )
    return rows


# ---------------------------------------------------------------------------
# 2. Empirical O(N_aux * N_AO^2 * N_occ) Scaling
# ---------------------------------------------------------------------------


def _measure_rik_point(
    mol: gto.Mole,
    label: str,
    backend: str,
    run_scf: bool = True,
) -> dict[str, Any]:
    release_gpu_memory()
    basis = gpu4dft.basis_from_pyscf(mol)
    aux_basis = gpu4dft.aux_basis_from_pyscf(mol, AUXBASIS)
    be_enum = gpu4dft._core.Backend.cuda if backend == "cuda" else gpu4dft._core.Backend.host

    nao = int(basis.nao)
    naux = int(aux_basis.nao)
    nocc = int(mol.nelectron // 2)

    rng = np.random.default_rng(42)
    q_mat, _ = np.linalg.qr(rng.standard_normal((nao, nao)))
    c_occ = np.ascontiguousarray(q_mat[:, :nocc], dtype=np.float64)
    dm = np.ascontiguousarray(2.0 * (c_occ @ c_occ.T), dtype=np.float64)

    opts = gpu4dft._core.JKOptions()
    opts.backend = be_enum
    opts.precision = gpu4dft._core.Precision.fp64
    opts.gemm_mode = gpu4dft._core.GemmMode.fp64
    opts.compute_j = False
    opts.compute_k = True
    opts.schwarz_cutoff = 1e-13

    with VramPeakMonitor(poll_gpu=(backend == "cuda")) as vmon_k:
        eng = gpu4dft._core.RIJKEngine(basis, aux_basis, be_enum, 0.0)
        # Warmup once for this size
        _ = eng.compute_jk_mo(dm, c_occ, 2.0, opts)
        eng.release_workspaces()
        sync_gpu()

        k_times: list[float] = []
        for _ in range(3):
            sync_gpu()
            t0 = time.perf_counter()
            _ = eng.compute_jk_mo(dm, c_occ, 2.0, opts)
            sync_gpu()
            k_times.append(time.perf_counter() - t0)
            eng.release_workspaces()
        del eng
        gc.collect()
    release_gpu_memory()

    scf_per_iter_ms: float | None = None
    scf_iters: int | None = None
    scf_peak_vram_mib: float | None = None
    if run_scf:
        dm0 = gpu4dft.minao_init_guess(mol, unrestricted=False)
        with VramPeakMonitor(poll_gpu=(backend == "cuda")) as vmon_s:
            mf = gpu4dft.RHF(
                mol,
                backend=backend,
                precision="fp64",
                dynamic_precision=False,
                gemm_mode="fp64",
                conv_tol=1e-8,
                max_cycle=40,
            ).density_fit(auxbasis=AUXBASIS)
            mf.kernel(dm0=dm0)
            sync_gpu()
            scf_iters = max(1, int(mf.iterations))
            scf_per_iter_ms = float(mf.result.scf_time_s) * 1e3 / scf_iters
            release_gpu_memory(mf)
        scf_peak_vram_mib = vmon_s.peak_vram_mib

    k_mean_ms = float(statistics.mean(k_times)) * 1e3
    flops_proxy = float(naux) * float(nao) * float(nao) * float(nocc)
    print(
        f"  {label:16s} nao={nao:4d}, naux={naux:4d}, nocc={nocc:3d}: "
        f"RI-K(MO)={k_mean_ms:8.2f} ms (VRAM={vmon_k.peak_vram_mib}), "
        f"SCF/iter={scf_per_iter_ms:.2f} ms ({scf_iters} it)"
    )
    return {
        "label": label,
        "nao": nao,
        "naux": naux,
        "nocc": nocc,
        "naux_nao2_nocc": flops_proxy,
        "naux_nao2": float(naux) * float(nao) * float(nao),
        "rik_mo_times_ms": [round(x * 1e3, 4) for x in k_times],
        "rik_mo_mean_ms": k_mean_ms,
        "rik_mo_peak_vram_mib": vmon_k.peak_vram_mib,
        "scf_iterations": scf_iters,
        "scf_per_iter_ms": scf_per_iter_ms,
        "scf_peak_vram_mib": scf_peak_vram_mib,
    }


def verify_rik_scaling(backend: str) -> dict[str, Any]:
    print("\n=== [2/3] Occupied-Space RI-K Scaling O(N_aux * N_AO^2 * N_occ) ===")
    # Global warmup on (H2O)_2
    mol_w = _make_mol(_water_cluster_geom(2, 1, 1), "def2-svp")
    _ = _measure_rik_point(mol_w, "warmup_(H2O)_2", backend, run_scf=True)

    # Water cluster ladder: n = 4, 8, 16, 32, 64, 80 (up to nao=1920)
    water_configs = [
        ("(H2O)_4", (2, 1, 2)),
        ("(H2O)_8", (2, 2, 2)),
        ("(H2O)_16", (2, 2, 4)),
        ("(H2O)_32", (4, 2, 4)),
        ("(H2O)_64", (4, 4, 4)),
        ("(H2O)_80", (5, 4, 4)),
    ]
    water_rows: list[dict[str, Any]] = []
    for label, grid_dims in water_configs:
        mol = _make_mol(_water_cluster_geom(*grid_dims), "def2-svp")
        water_rows.append(_measure_rik_point(mol, label, backend, run_scf=True))

    # Fit log-log exponent vs N_aux * N_AO^2 * N_occ
    xs_full = [r["naux_nao2_nocc"] for r in water_rows]
    ys_k_full = [r["rik_mo_mean_ms"] for r in water_rows]
    ys_s_full = [r["scf_per_iter_ms"] for r in water_rows if r["scf_per_iter_ms"] is not None]
    exp_k_full, r2_k_full = _log_log_fit(xs_full, ys_k_full)
    exp_s_full, r2_s_full = _log_log_fit(xs_full, ys_s_full)

    asym_rows = [r for r in water_rows if r["nao"] >= 384]
    xs_asym = [r["naux_nao2_nocc"] for r in asym_rows]
    ys_k_asym = [r["rik_mo_mean_ms"] for r in asym_rows]
    ys_s_asym = [r["scf_per_iter_ms"] for r in asym_rows if r["scf_per_iter_ms"] is not None]
    exp_k_asym, r2_k_asym = _log_log_fit(xs_asym, ys_k_asym)
    exp_s_asym, r2_s_asym = _log_log_fit(xs_asym, ys_s_asym)

    # Fixed-N_occ (H2O)_4 basis ladder
    basis_rows: list[dict[str, Any]] = []
    h2o4_geom = _water_cluster_geom(2, 1, 2)
    for basis_name in ("sto-3g", "6-31g*", "def2-svp", "def2-tzvp"):
        mol = _make_mol(h2o4_geom, basis_name)
        basis_rows.append(_measure_rik_point(mol, f"(H2O)_4/{basis_name}", backend, run_scf=True))

    b_nao = [float(r["nao"]) for r in basis_rows]
    b_work = [r["naux_nao2"] for r in basis_rows]
    b_k_ms = [r["rik_mo_mean_ms"] for r in basis_rows]
    b_s_ms = [r["scf_per_iter_ms"] for r in basis_rows if r["scf_per_iter_ms"] is not None]
    exp_basis_nao_k, _ = _log_log_fit(b_nao, b_k_ms)
    exp_basis_nao_s, _ = _log_log_fit(b_nao, b_s_ms)
    exp_basis_work_k, r2_basis_work_k = _log_log_fit(b_work, b_k_ms)

    print(
        f"  Water ladder exponent vs (N_aux*N_AO^2*N_occ): "
        f"RI-K(MO) full={exp_k_full:.3f} (R2={r2_k_full:.4f}), asym(nao>=384)={exp_k_asym:.3f} (R2={r2_k_asym:.4f}) | "
        f"SCF/iter asym={exp_s_asym:.3f}"
    )
    print(
        f"  Fixed-N_occ basis ladder exponent vs N_AO: RI-K(MO)={exp_basis_nao_k:.3f}, SCF/iter={exp_basis_nao_s:.3f}"
    )

    return {
        "water_ladder": water_rows,
        "basis_ladder": basis_rows,
        "fits": {
            "water_rik_mo_full_exp_vs_naux_nao2_nocc": exp_k_full,
            "water_rik_mo_full_r2": r2_k_full,
            "water_rik_mo_asym_exp_vs_naux_nao2_nocc": exp_k_asym,
            "water_rik_mo_asym_r2": r2_k_asym,
            "water_scf_iter_full_exp_vs_naux_nao2_nocc": exp_s_full,
            "water_scf_iter_full_r2": r2_s_full,
            "water_scf_iter_asym_exp_vs_naux_nao2_nocc": exp_s_asym,
            "water_scf_iter_asym_r2": r2_s_asym,
            "basis_rik_mo_exp_vs_nao": exp_basis_nao_k,
            "basis_scf_iter_exp_vs_nao": exp_basis_nao_s,
            "basis_rik_mo_exp_vs_naux_nao2": exp_basis_work_k,
            "basis_rik_mo_r2_vs_naux_nao2": r2_basis_work_k,
        },
    }


# ---------------------------------------------------------------------------
# 3. RI-JK Analytical Nuclear Gradient Verification
# ---------------------------------------------------------------------------


def verify_ri_jk_gradients(backend: str) -> dict[str, Any]:
    print("\n=== [3/3] RI-JK Analytical Nuclear Gradients Verification ===")

    # 3A. Translational Invariance (<= 1e-6 Eh/a0) across small & large systems
    trans_cases: list[tuple[str, str, str, str | None, int]] = [
        ("H2O", "def2-svp", "rhf", None, 4),
        ("H2O", "def2-svp", "rks", "b3lyp", 4),
        ("H2O", "def2-svp", "rks", "cam-b3lyp", 4),
        ("H2O", "def2-svp", "rks", "wb97x", 4),
        ("HO2", "def2-svp", "uhf", None, 4),
        ("HO2", "def2-svp", "uks", "b3lyp", 4),
        ("HO2", "def2-svp", "uks", "cam-b3lyp", 4),
        ("HO2", "def2-svp", "uks", "wb97x", 4),
        ("(H2O)_32", "def2-svp", "rks", "b3lyp", 3),
        ("C60", "def2-svp", "rks", "b3lyp", 3),
        ("Taxol", "6-31g*", "rks", "b3lyp", 3),
        ("Valinomycin", "6-31g*", "rks", "b3lyp", 3),
    ]
    trans_rows: list[dict[str, Any]] = []
    for sys_name, basis_name, method, xc_name, g_lvl in trans_cases:
        if sys_name in SMALL_GEOMS:
            atom, charge, spin = SMALL_GEOMS[sys_name]
            mol = _make_mol(atom, basis_name, charge=charge, spin=spin)
        else:
            mol = build_mol(sys_name, basis_name)
        unrestricted = method.lower() in ("uhf", "uks")
        dm0 = gpu4dft.minao_init_guess(mol, unrestricted=unrestricted)

        release_gpu_memory()
        with VramPeakMonitor(poll_gpu=(backend == "cuda")) as vmon:
            mf = _make_gpu4dft_scf(
                mol,
                method,
                xc_name,
                backend=backend,
                grids_level=g_lvl,
                conv_tol=1e-10,
            )
            mf.kernel(dm0=dm0)
            gm = mf.nuc_grad_method()
            grad = np.asarray(gm.kernel(), dtype=np.float64)
            converged = bool(mf.converged)
            release_gpu_memory(mf, gm)

        net_force = float(np.linalg.norm(np.sum(grad, axis=0)))
        max_abs_net = float(np.max(np.abs(np.sum(grad, axis=0))))
        passed = converged and (net_force <= 1e-6)
        trans_rows.append(
            {
                "system": sys_name,
                "basis": basis_name,
                "nao": int(mol.nao),
                "method": method.upper(),
                "xc": xc_name.upper() if xc_name else "HF",
                "converged": converged,
                "net_force_norm_eh_bohr": net_force,
                "net_force_max_component_eh_bohr": max_abs_net,
                "peak_vram_mib": vmon.peak_vram_mib,
                "passed": bool(passed),
            }
        )
        print(
            f"  [TransInv {'PASS' if passed else 'FAIL'}] {sys_name:11s} {method.upper()}/{(xc_name or 'HF').upper():9s} "
            f"(nao={mol.nao:4d}): |sum g|={net_force:.2e} Eh/a0"
        )

    # 3B. Matched-grid parity vs PySCF DF (grid_response=False for DFT) <= 1e-5 Eh/a0
    pyscf_grad_cases: list[tuple[str, str, str, str | None]] = [
        ("H2O", "def2-svp", "rhf", None),
        ("HO2", "def2-svp", "uhf", None),
        ("H2O", "def2-svp", "rks", "b3lyp"),
        ("H2O", "def2-svp", "rks", "pbe0"),
        ("H2O", "def2-svp", "rks", "cam-b3lyp"),
        ("H2O", "def2-svp", "rks", "wb97x"),
        ("HO2", "def2-svp", "uks", "b3lyp"),
        ("HO2", "def2-svp", "uks", "cam-b3lyp"),
        ("HO2", "def2-svp", "uks", "wb97x"),
    ]
    pyscf_rows: list[dict[str, Any]] = []
    for sys_name, basis_name, method, xc_name in pyscf_grad_cases:
        atom, charge, spin = SMALL_GEOMS[sys_name]
        mol = _make_mol(atom, basis_name, charge=charge, spin=spin)
        unrestricted = method.lower() in ("uhf", "uks")
        dm0 = gpu4dft.minao_init_guess(mol, unrestricted=unrestricted)

        mf_ref = _make_pyscf_scf(mol, method, xc_name, grids_level=4, conv_tol=1e-11)
        if method.lower() in ("rks", "uks"):
            mf_ref.grids.build()
        mf_ref.kernel(dm0=dm0)
        gm_ref = mf_ref.nuc_grad_method()
        if hasattr(gm_ref, "auxbasis_response"):
            gm_ref.auxbasis_response = True
        if method.lower() in ("rks", "uks") and hasattr(gm_ref, "grid_response"):
            gm_ref.grid_response = False
        g_ref = np.asarray(gm_ref.kernel(), dtype=np.float64)

        matched_grids = None
        if method.lower() in ("rks", "uks"):
            matched_grids = (
                np.ascontiguousarray(mf_ref.grids.coords, dtype=np.float64),
                np.ascontiguousarray(mf_ref.grids.weights, dtype=np.float64),
            )
        del gm_ref, mf_ref
        gc.collect()

        release_gpu_memory()
        mf_g4d = _make_gpu4dft_scf(
            mol,
            method,
            xc_name,
            backend=backend,
            grids=matched_grids,
            grids_level=4,
            conv_tol=1e-11,
            diis_tol=1e-9,
        )
        mf_g4d.kernel(dm0=dm0)
        gm_g4d = mf_g4d.nuc_grad_method()
        gm_g4d.grid_response = False
        g_g4d = np.asarray(gm_g4d.kernel(), dtype=np.float64)
        release_gpu_memory(mf_g4d, gm_g4d)

        max_dg = float(np.max(np.abs(g_g4d - g_ref)))
        passed = max_dg <= 1e-5
        pyscf_rows.append(
            {
                "system": sys_name,
                "basis": basis_name,
                "method": method.upper(),
                "xc": xc_name.upper() if xc_name else "HF",
                "max_abs_grad_diff_eh_bohr": max_dg,
                "passed": bool(passed),
            }
        )
        print(
            f"  [PySCF Parity {'PASS' if passed else 'FAIL'}] {sys_name:5s} {method.upper()}/{(xc_name or 'HF').upper():9s}: "
            f"max|dg|={max_dg:.2e} Eh/a0"
        )

    # 3C. Native-grid grid_response=True central FD parity (<= 1e-5 Eh/a0)
    fd_cases: list[tuple[str, str, str, str | None]] = [
        ("H2O", "def2-svp", "rhf", None),
        ("H2O", "def2-svp", "rks", "b3lyp"),
        ("H2O", "def2-svp", "rks", "wb97x"),
        ("HO2", "def2-svp", "uhf", None),
        ("HO2", "def2-svp", "uks", "cam-b3lyp"),
        ("HO2", "def2-svp", "uks", "wb97x"),
    ]
    h_bohr = 1e-4
    h_ang = h_bohr * BOHR_TO_ANG
    fd_rows: list[dict[str, Any]] = []
    for sys_name, basis_name, method, xc_name in fd_cases:
        atom_str, charge, spin = SMALL_GEOMS[sys_name]
        mol0 = _make_mol(atom_str, basis_name, charge=charge, spin=spin)
        coords0_ang = mol0.atom_coords(unit="Angstrom")
        syms = [mol0.atom_symbol(i) for i in range(mol0.natm)]
        unrestricted = method.lower() in ("uhf", "uks")
        dm0 = gpu4dft.minao_init_guess(mol0, unrestricted=unrestricted)

        release_gpu_memory()
        mf0 = _make_gpu4dft_scf(
            mol0, method, xc_name, backend=backend, grids_level=4, conv_tol=1e-11, diis_tol=1e-9
        )
        mf0.kernel(dm0=dm0)
        gm0 = mf0.nuc_grad_method()
        gm0.grid_response = True
        g_ana = np.asarray(gm0.kernel(), dtype=np.float64)
        release_gpu_memory(mf0, gm0)

        g_fd = np.zeros_like(g_ana)
        for ia in range(mol0.natm):
            for ic in range(3):
                e_pm: list[float] = []
                for sgn in (1.0, -1.0):
                    c_pert = coords0_ang.copy()
                    c_pert[ia, ic] += sgn * h_ang
                    atom_pert = [(syms[k], tuple(c_pert[k])) for k in range(mol0.natm)]
                    mol_p = _make_mol(atom_pert, basis_name, charge=charge, spin=spin)
                    mf_p = _make_gpu4dft_scf(
                        mol_p, method, xc_name, backend=backend, grids_level=4, conv_tol=1e-11, diis_tol=1e-9
                    )
                    e_p = float(mf_p.kernel(dm0=dm0))
                    release_gpu_memory(mf_p)
                    e_pm.append(e_p)
                g_fd[ia, ic] = (e_pm[0] - e_pm[1]) / (2.0 * h_bohr)

        max_dg_fd = float(np.max(np.abs(g_ana - g_fd)))
        passed = max_dg_fd <= 1e-5
        fd_rows.append(
            {
                "system": sys_name,
                "basis": basis_name,
                "method": method.upper(),
                "xc": xc_name.upper() if xc_name else "HF",
                "h_bohr": h_bohr,
                "max_abs_fd_diff_eh_bohr": max_dg_fd,
                "passed": bool(passed),
            }
        )
        print(
            f"  [FD Parity {'PASS' if passed else 'FAIL'}] {sys_name:5s} {method.upper()}/{(xc_name or 'HF').upper():9s}: "
            f"max|g_ana - g_FD|={max_dg_fd:.2e} Eh/a0"
        )

    return {
        "translational_invariance": trans_rows,
        "matched_grid_pyscf_parity": pyscf_rows,
        "native_grid_fd_parity": fd_rows,
    }


def render_scaling_markdown(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    lines = [
        "# GPU4DFT M5 Occupied-Space RI-K Scaling & Accuracy Report",
        "",
        f"- **GPU**: `{meta.get('gpu_name')}` ({meta.get('vram_total_mib')} MiB)",
        f"- **Commit**: `{meta.get('git_revision')}`",
        f"- **Timestamp (UTC)**: `{meta.get('timestamp_utc')}`",
        "",
        "## 1. Matched-Grid FP64 RI-K SCF Energy Parity (Threshold: <= 1.0e-6 Eh)",
        "",
        "| System | Basis | $N_{\\text{AO}}$ | Method | XC | $|\\Delta E|$ ($E_\\text{h}$) | $|\\Delta \\langle S^2 \\rangle|$ | Status |",
        "| :--- | :--- | ---: | :--- | :--- | ---: | ---: | :---: |",
    ]
    for r in payload.get("energy_parity", []):
        ds2 = f"`{r['abs_s2_diff']:.2e}`" if r.get("abs_s2_diff") is not None else "—"
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | `{r['method']}` | `{r['xc']}` | "
            f"`{r['abs_diff_eh']:.2e}` | {ds2} | {'PASS' if r['passed'] else 'FAIL'} |"
        )

    scaling = payload.get("scaling", {})
    fits = scaling.get("fits", {})
    lines.extend(
        [
            "",
            "## 2. Empirical $O(N_{\\text{aux}} N_{\\text{AO}}^2 N_{\\text{occ}})$ Scaling",
            "",
            f"- **Water ladder RI-K(MO) exponent vs $N_{{\\text{{aux}}}} N_{{\\text{{AO}}}}^2 N_{{\\text{{occ}}}}$**: "
            f"`{fits.get('water_rik_mo_full_exp_vs_naux_nao2_nocc', 0.0):.3f}` (full, $R^2={fits.get('water_rik_mo_full_r2', 0.0):.4f}$), "
            f"`{fits.get('water_rik_mo_asym_exp_vs_naux_nao2_nocc', 0.0):.3f}` ($N_{{\\text{{AO}}}} \\ge 384$, $R^2={fits.get('water_rik_mo_asym_r2', 0.0):.4f}$)",
            f"- **Water ladder SCF/iter exponent ($N_{{\\text{{AO}}}} \\ge 384$)**: `{fits.get('water_scf_iter_asym_exp_vs_naux_nao2_nocc', 0.0):.3f}`",
            f"- **Fixed-$N_{{\\text{{occ}}}}$ basis ladder exponent vs $N_{{\\text{{AO}}}}$**: "
            f"RI-K(MO) `{fits.get('basis_rik_mo_exp_vs_nao', 0.0):.3f}`, SCF/iter `{fits.get('basis_scf_iter_exp_vs_nao', 0.0):.3f}`",
            "",
            "### 2A. `(H2O)_n` Cluster Ladder (`def2-svp` / `def2-universal-jkfit`)",
            "",
            "| Cluster | $N_{\\text{AO}}$ | $N_{\\text{aux}}$ | $N_{\\text{occ}}$ | RI-K(MO) Mean (ms) | RI-K Peak VRAM (MiB) | SCF/iter (ms) | SCF Peak VRAM (MiB) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in scaling.get("water_ladder", []):
        kv = f"{r['rik_mo_peak_vram_mib']:.1f}" if r.get("rik_mo_peak_vram_mib") is not None else "—"
        sv = f"{r['scf_peak_vram_mib']:.1f}" if r.get("scf_peak_vram_mib") is not None else "—"
        si = f"{r['scf_per_iter_ms']:.2f}" if r.get("scf_per_iter_ms") is not None else "—"
        lines.append(
            f"| `{r['label']}` | {r['nao']} | {r['naux']} | {r['nocc']} | "
            f"{r['rik_mo_mean_ms']:.2f} | {kv} | {si} | {sv} |"
        )

    lines.extend(
        [
            "",
            "### 2B. Fixed-$N_{\\text{occ}}$ Basis Ladder (`(H2O)_4`, $N_{\\text{occ}}=20$)",
            "",
            "| Basis Case | $N_{\\text{AO}}$ | $N_{\\text{aux}}$ | RI-K(MO) Mean (ms) | SCF/iter (ms) |",
            "| :--- | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in scaling.get("basis_ladder", []):
        si = f"{r['scf_per_iter_ms']:.2f}" if r.get("scf_per_iter_ms") is not None else "—"
        lines.append(
            f"| `{r['label']}` | {r['nao']} | {r['naux']} | {r['rik_mo_mean_ms']:.2f} | {si} |"
        )

    grad = payload.get("gradients", {})
    lines.extend(
        [
            "",
            "## 3. RI-JK Analytical Nuclear Gradients",
            "",
            "### 3A. Translational Invariance (Threshold: <= 1.0e-6 Eh/a0)",
            "",
            "| System | Basis | $N_{\\text{AO}}$ | Method | XC | $|\\sum \\mathbf{g}_A|$ ($E_\\text{h}/a_0$) | Peak VRAM (MiB) | Status |",
            "| :--- | :--- | ---: | :--- | :--- | ---: | ---: | :---: |",
        ]
    )
    for r in grad.get("translational_invariance", []):
        v = f"{r['peak_vram_mib']:.1f}" if r.get("peak_vram_mib") is not None else "—"
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | `{r['method']}` | `{r['xc']}` | "
            f"`{r['net_force_norm_eh_bohr']:.2e}` | {v} | {'PASS' if r['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "### 3B. Matched-Grid Parity vs PySCF DF (`grid_response=False`, Threshold: <= 1.0e-5 Eh/a0)",
            "",
            "| System | Basis | Method | XC | $\\max|\\Delta \\mathbf{g}|$ ($E_\\text{h}/a_0$) | Status |",
            "| :--- | :--- | :--- | :--- | ---: | :---: |",
        ]
    )
    for r in grad.get("matched_grid_pyscf_parity", []):
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | `{r['method']}` | `{r['xc']}` | "
            f"`{r['max_abs_grad_diff_eh_bohr']:.2e}` | {'PASS' if r['passed'] else 'FAIL'} |"
        )

    lines.extend(
        [
            "",
            "### 3C. Native-Grid `grid_response=True` Central Finite-Difference Parity (Threshold: <= 1.0e-5 Eh/a0)",
            "",
            "| System | Basis | Method | XC | $\\max|\\mathbf{g}_{\\text{ana}} - \\mathbf{g}_{\\text{FD}}|$ ($E_\\text{h}/a_0$) | Status |",
            "| :--- | :--- | :--- | :--- | ---: | :---: |",
        ]
    )
    for r in grad.get("native_grid_fd_parity", []):
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | `{r['method']}` | `{r['xc']}` | "
            f"`{r['max_abs_fd_diff_eh_bohr']:.2e}` | {'PASS' if r['passed'] else 'FAIL'} |"
        )

    lines.append("")
    return "\n".join(lines)


def _save(payload: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2))
    md_out.write_text(render_scaling_markdown(payload))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Verify M5 occupied-space RI-K scaling and RI-JK gradients.")
    parser.add_argument(
        "--backend",
        choices=("cuda", "host"),
        default="cuda" if gpu4dft.backend_available("cuda") else "host",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "scaling_and_accuracy.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "SCALING_AND_ACCURACY.md",
    )
    parser.add_argument(
        "--sections",
        type=str,
        default="energy,scaling,grad",
        help="comma-separated sections to run (energy,scaling,grad)",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=None,
        help="optional existing JSON file to resume/merge",
    )
    args = parser.parse_args(argv)
    sections = {s.strip().lower() for s in args.sections.split(",") if s.strip()}

    meta = collect_hardware_metadata(n_repeats=1, grids_level=4, auxbasis=AUXBASIS)
    payload: dict[str, Any] = {
        "metadata": meta,
        "energy_parity": [],
        "scaling": {},
        "gradients": {},
    }
    if args.resume_json is not None and args.resume_json.exists():
        prev = json.loads(args.resume_json.read_text(encoding="utf-8"))
        for k in ("energy_parity", "scaling", "gradients"):
            if k in prev:
                payload[k] = prev[k]
        payload["metadata"] = meta

    def _ckpt_energy(rows: list[dict[str, Any]]) -> None:
        payload["energy_parity"] = rows
        _save(payload, args.json_out, args.md_out)

    if "energy" in sections:
        payload["energy_parity"] = verify_energy_parity(args.backend, checkpoint_cb=_ckpt_energy)
        _save(payload, args.json_out, args.md_out)

    if "scaling" in sections:
        payload["scaling"] = verify_rik_scaling(args.backend)
        _save(payload, args.json_out, args.md_out)

    if "grad" in sections:
        payload["gradients"] = verify_ri_jk_gradients(args.backend)
        _save(payload, args.json_out, args.md_out)

    energy_ok = (not payload["energy_parity"]) or all(r["passed"] for r in payload["energy_parity"])
    grad_d = payload.get("gradients", {})
    grad_ok = (
        all(r["passed"] for r in grad_d.get("translational_invariance", []))
        and all(r["passed"] for r in grad_d.get("matched_grid_pyscf_parity", []))
        and all(r["passed"] for r in grad_d.get("native_grid_fd_parity", []))
    )
    fits = payload.get("scaling", {}).get("fits", {})
    scale_ok = True if not fits else (fits.get("basis_rik_mo_exp_vs_nao", 99.0) < 2.6)
    all_passed = energy_ok and grad_ok and scale_ok
    payload["all_passed"] = bool(all_passed)
    _save(payload, args.json_out, args.md_out)

    print(f"\nSaved scaling & accuracy JSON -> {args.json_out}")
    print(f"Saved scaling & accuracy MD   -> {args.md_out}")
    print(f"Overall M5 scaling/accuracy status: {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
