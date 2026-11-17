#!/usr/bin/env python3
"""Comprehensive, reproducible GPU4DFT vs GPU4PySCF vs CPU PySCF benchmark suite.

Evaluates:
  1. Ozaki INT8 Tensor Core vs FP32 Tensor Core vs Native FP64 GEMM TFLOPS & error
  2. Generated ERI kernel throughput (MQuartets/s) across angular-momentum classes
  3. Density solver & CUDA Graph ablation (cuSOLVER Dsyevd vs Niklasson SP2 + OzakiInt8)
  4. End-to-end RHF & Hybrid/GGA DFT (Direct FP64, Direct Dynamic Mixed + OzakiInt8,
     RI-JK FP64, RI-JK OzakiInt8 Tensor Core, sn-K / COSX, and RIJCOSX) head-to-head
     against GPU4PySCF and multi-core CPU PySCF across molecules:
       (H2O)_2, C6H6, C8H18, (H2O)_8, aspirin (C9H8O4), C12H26, (H2O)_16
     and basis sets:
       6-31g*, def2-svp, def2-tzvp
  5. Analytical Nuclear Gradients (RHF, PBE, B3LYP): accuracy vs PySCF, translational
     invariance, T_grad / T_SCF ratio, and speedup vs GPU4PySCF & CPU PySCF
  6. GPU TDDFT & RI-TDDFT Excited States (TDA & Full Casida/SSF-RPA across C6H6, (H2O)_8,
     and aspirin; Direct, RI-JK FP64, and RI-JK OzakiInt8 vs PySCF & GPU4PySCF)
  7. Plane-Wave Kohn-Sham & Lin's ACE Hybrid DFT (Diamond Si2, Si4, Si8 across PBE & PBE0
     vs PySCF-PBC, GPU Uncompressed EXX, and multi-core OpenMP Host)

Outputs structured JSON + publication-ready Markdown summary tables.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CODEGEN_DIR = REPO_ROOT / "codegen"
if str(CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(CODEGEN_DIR))

import gpu4dft  # noqa: E402
from pyscf import df, dft, gto, scf, tdscf  # noqa: E402
from pyscf.grad import rhf as pyscf_grad_rhf  # noqa: E402
from pyscf.grad import rks as pyscf_grad_rks  # noqa: E402
from pyscf.pbc import dft as pbc_dft  # noqa: E402
from pyscf.pbc import gto as pbc_gto  # noqa: E402
from pyscf.pbc.gto.pseudo import pp as pbc_pp  # noqa: E402

try:
    import cupy as cp  # type: ignore

    HAVE_CUPY = True
except Exception:
    cp = None
    HAVE_CUPY = False

try:
    from gpu4pyscf import dft as gpu4pyscf_dft  # type: ignore
    from gpu4pyscf import scf as gpu4pyscf_scf  # type: ignore

    HAVE_GPU4PYSCF = True
except Exception:
    gpu4pyscf_dft = None
    gpu4pyscf_scf = None
    HAVE_GPU4PYSCF = False


# ---------------------------------------------------------------------------
# Synchronized Timing & Hardware Metadata Helpers
# ---------------------------------------------------------------------------


def sync_gpu() -> None:
    if HAVE_CUPY and cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass


def get_peak_vram_mib() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip()
        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return vals[0] if vals else None
    except Exception:
        return None


def collect_hardware_metadata() -> dict[str, Any]:
    gpu_name = "CPU-only"
    vram_total_mib = 0
    driver_ver = "N/A"
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        if out:
            parts = [p.strip() for p in out.splitlines()[0].split(",")]
            if len(parts) >= 3:
                gpu_name = parts[0]
                vram_total_mib = int(float(parts[1]))
                driver_ver = parts[2]
    except Exception:
        pass

    cuda_ver = "N/A"
    try:
        nvcc_out = subprocess.check_output(["nvcc", "--version"], text=True, timeout=5)
        for line in nvcc_out.splitlines():
            if "release" in line:
                cuda_ver = line.strip()
                break
    except Exception:
        pass

    git_rev = "unknown"
    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        pass

    gpu4pyscf_ver = None
    if HAVE_GPU4PYSCF:
        import gpu4pyscf as _g4p  # type: ignore

        gpu4pyscf_ver = getattr(_g4p, "__version__", "installed")

    import pyscf as _pyscf

    return {
        "git_revision": git_rev,
        "gpu_name": gpu_name,
        "vram_total_mib": vram_total_mib,
        "nvidia_driver": driver_ver,
        "cuda_compiler": cuda_ver,
        "cpu_count": os.cpu_count() or 1,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pyscf_version": getattr(_pyscf, "__version__", "unknown"),
        "gpu4pyscf_available": HAVE_GPU4PYSCF,
        "gpu4pyscf_version": gpu4pyscf_ver,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Reproducible Chemically Realistic Molecular Geometry Builders (Angstrom)
# ---------------------------------------------------------------------------


def _water_cluster_geom(nx: int, ny: int, nz: int, spacing: float = 2.90) -> str:
    """Build a hydrogen-bonded 3D water cluster with realistic O-H = 0.96 A, H-O-H = 104.5 deg."""
    r_oh = 0.96
    half_ang = math.radians(104.52 / 2.0)
    h1_local = np.array([0.0, r_oh * math.sin(half_ang), r_oh * math.cos(half_ang)])
    h2_local = np.array([0.0, -r_oh * math.sin(half_ang), r_oh * math.cos(half_ang)])

    lines: list[str] = []
    idx = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                origin = np.array(
                    [
                        ix * spacing + 0.12 * ((iy + iz) % 2),
                        iy * spacing + 0.10 * ((ix + iz) % 2),
                        iz * spacing + 0.08 * ((ix + iy) % 2),
                    ]
                )
                # Alternate molecular orientation across lattice sites to form H-bond network
                theta = 0.65 * idx
                phi = 1.15 * idx
                ct, st = math.cos(theta), math.sin(theta)
                cp_a, sp_a = math.cos(phi), math.sin(phi)
                rot = np.array(
                    [
                        [ct * cp_a, -sp_a, st * cp_a],
                        [ct * sp_a, cp_a, st * sp_a],
                        [-st, 0.0, ct],
                    ]
                )
                p_o = origin
                p_h1 = origin + rot @ h1_local
                p_h2 = origin + rot @ h2_local
                lines.append(f"O {p_o[0]:.6f} {p_o[1]:.6f} {p_o[2]:.6f}")
                lines.append(f"H {p_h1[0]:.6f} {p_h1[1]:.6f} {p_h1[2]:.6f}")
                lines.append(f"H {p_h2[0]:.6f} {p_h2[1]:.6f} {p_h2[2]:.6f}")
                idx += 1
    return "; ".join(lines)


def _benzene_geom() -> str:
    r_cc = 1.397
    r_ch = 1.087
    lines: list[str] = []
    for k in range(6):
        ang = k * math.pi / 3.0
        # Slight 0.005 A distortion breaks exact D6h degeneracy so gradient components are non-zero
        dr = 0.005 if k == 0 else 0.0
        cx = (r_cc + dr) * math.cos(ang)
        cy = (r_cc + dr) * math.sin(ang)
        hx = (r_cc + r_ch + dr) * math.cos(ang)
        hy = (r_cc + r_ch + dr) * math.sin(ang)
        lines.append(f"C {cx:.6f} {cy:.6f} 0.000000")
        lines.append(f"H {hx:.6f} {hy:.6f} 0.000000")
    return "; ".join(lines)


def _alkane_chain_geom(n_carbons: int) -> str:
    """All-trans linear alkane C_n H_{2n+2} (R_CC = 1.53 A, <CCC = 112 deg, R_CH = 1.09 A)."""
    r_cc = 1.53
    r_ch = 1.09
    half_ccc = math.radians(112.0 / 2.0)
    dx = r_cc * math.sin(half_ccc)
    dy = r_cc * math.cos(half_ccc)

    c_pos = [np.array([i * dx, (i % 2) * dy, 0.0]) for i in range(n_carbons)]
    lines = [f"C {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in c_pos]

    h_ang = math.radians(109.47 / 2.0)
    for i in range(n_carbons):
        sign_y = 1.0 if (i % 2 == 1) else -1.0
        # Two methylene CH2 hydrogens above/below the zig-zag plane
        hz = r_ch * math.sin(h_ang)
        hy = sign_y * r_ch * math.cos(h_ang)
        lines.append(f"H {c_pos[i][0]:.6f} {c_pos[i][1] + hy:.6f} {hz:.6f}")
        lines.append(f"H {c_pos[i][0]:.6f} {c_pos[i][1] + hy:.6f} {-hz:.6f}")

    # Two terminal methyl hydrogens in the zig-zag plane
    v0 = c_pos[0] - c_pos[1]
    v0 = v0 / np.linalg.norm(v0)
    p_end0 = c_pos[0] + r_ch * v0
    lines.append(f"H {p_end0[0]:.6f} {p_end0[1]:.6f} 0.000000")

    vn = c_pos[-1] - c_pos[-2]
    vn = vn / np.linalg.norm(vn)
    p_endn = c_pos[-1] + r_ch * vn
    lines.append(f"H {p_endn[0]:.6f} {p_endn[1]:.6f} 0.000000")
    return "; ".join(lines)


def _aspirin_geom() -> str:
    """3D acetylsalicylic acid (C9H8O4, 21 atoms, 94 electrons)."""
    r_ring = 1.395
    r_ch_ar = 1.085
    atoms: list[tuple[str, np.ndarray]] = []

    # 6 aromatic ring carbons C1..C6
    for k in range(6):
        ang = k * math.pi / 3.0
        atoms.append(("C", np.array([r_ring * math.cos(ang), r_ring * math.sin(ang), 0.0])))

    # 4 aromatic hydrogens on C3, C4, C5, C6 (k = 2, 3, 4, 5)
    for k in (2, 3, 4, 5):
        ang = k * math.pi / 3.0
        atoms.append(
            (
                "H",
                np.array([(r_ring + r_ch_ar) * math.cos(ang), (r_ring + r_ch_ar) * math.sin(ang), 0.0]),
            )
        )

    # Carboxylic acid -C7(=O1)-O2-H on C1 (k = 0)
    atoms.append(("C", np.array([2.885, 0.000, 0.000])))
    atoms.append(("O", np.array([3.500, 1.050, 0.150])))
    atoms.append(("O", np.array([3.550, -1.180, -0.150])))
    atoms.append(("H", np.array([4.510, -1.080, -0.150])))

    # Ester -O3-C8(=O4)-C9H3 on C2 (k = 1)
    c8 = np.array([1.350, 3.250, 1.150])
    c9 = np.array([2.250, 4.440, 0.980])
    atoms.append(("O", np.array([1.390, 2.410, 0.050])))
    atoms.append(("C", c8))
    atoms.append(("O", np.array([0.650, 3.020, 2.120])))
    atoms.append(("C", c9))

    # 3 tetrahedral methyl hydrogens around C9
    axis = c9 - c8
    axis = axis / np.linalg.norm(axis)
    perp1 = np.cross(axis, np.array([0.0, 0.0, 1.0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    r_ch = 1.09
    cos_t = math.cos(math.radians(70.53))
    sin_t = math.sin(math.radians(70.53))
    for m in range(3):
        phi = m * 2.0 * math.pi / 3.0
        h_vec = r_ch * (cos_t * axis + sin_t * (math.cos(phi) * perp1 + math.sin(phi) * perp2))
        atoms.append(("H", c9 + h_vec))

    return "; ".join(f"{sym} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for sym, p in atoms)


MOLECULE_CATALOG: dict[str, str] = {
    "(H2O)_2": _water_cluster_geom(2, 1, 1),
    "C6H6": _benzene_geom(),
    "C8H18": _alkane_chain_geom(8),
    "(H2O)_8": _water_cluster_geom(2, 2, 2),
    "aspirin": _aspirin_geom(),
    "C12H26": _alkane_chain_geom(12),
    "(H2O)_16": _water_cluster_geom(2, 2, 4),
}


def build_mol(system_name: str, basis: str) -> gto.Mole:
    mol = gto.M(
        atom=MOLECULE_CATALOG[system_name],
        basis=basis,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    mol.max_memory = 64000
    return mol


# ---------------------------------------------------------------------------
# Suite 1: Ozaki INT8 Tensor Core vs FP32 vs FP64 GEMM TFLOPS
# ---------------------------------------------------------------------------


def run_gemm_suite(backend: str) -> list[dict[str, Any]]:
    print("\n=== [1/7] Ozaki INT8 Tensor Core GEMM Benchmark ===")
    sizes = [256, 512, 1003, 1024, 2048, 4096]
    rows: list[dict[str, Any]] = []
    for n in sizes:
        bench_iters = 5 if n <= 2048 else 3
        res = gpu4dft.benchmark_gemm(n, warmup_iters=2, bench_iters=bench_iters, backend=backend)
        speedup = res.ozaki_tflops / max(res.fp64_tflops, 1e-12)
        row = {
            "n": n,
            "fp64_ms": res.fp64_ms,
            "fp64_tflops": res.fp64_tflops,
            "ozaki_int8_ms": res.ozaki_ms,
            "ozaki_int8_tflops": res.ozaki_tflops,
            "ozaki_speedup_vs_fp64": speedup,
            "fp32_tflops": res.fp32_tflops,
            "ozaki_rel_error": res.ozaki_rel_error,
            "ozaki_max_abs_error": res.ozaki_max_abs_error,
        }
        rows.append(row)
        print(
            f"  N={n:4d}: FP64={res.fp64_tflops:6.2f} TFLOPS | "
            f"Ozaki-INT8={res.ozaki_tflops:6.2f} TFLOPS ({speedup:4.2f}x, rel_err={res.ozaki_rel_error:.2e}) | "
            f"FP32-TC={res.fp32_tflops:7.2f} TFLOPS"
        )
    return rows


# ---------------------------------------------------------------------------
# Suite 2: Generated ERI Kernel Quartet Throughput
# ---------------------------------------------------------------------------


def run_eri_throughput_suite(backend: str) -> list[dict[str, Any]]:
    print("\n=== [2/7] ERI Quartet Kernel Throughput ===")
    cases = [
        ("(H2O)_2", "sto-3g", "s/p deep contraction"),
        ("(H2O)_2", "6-31g*", "s/p/d mixed shells"),
        ("(H2O)_2", "def2-tzvp", "up to f (l=3) shells"),
    ]
    rows: list[dict[str, Any]] = []
    for sys_name, basis_name, label in cases:
        mol = build_mol(sys_name, basis_name)
        basis = gpu4dft.basis_from_pyscf(mol)
        # Warmup
        _ = gpu4dft.eri_tensor(basis, backend=backend, precision="fp64")
        sync_gpu()

        times_fp64 = []
        times_fp32 = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = gpu4dft.eri_tensor(basis, backend=backend, precision="fp64")
            sync_gpu()
            times_fp64.append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            _ = gpu4dft.eri_tensor(basis, backend=backend, precision="fp32")
            sync_gpu()
            times_fp32.append((time.perf_counter() - t0) * 1e3)

        nsh = basis.nshell
        n_quartets = nsh**4
        med_64 = statistics.median(times_fp64)
        med_32 = statistics.median(times_fp32)
        mq_s_64 = (n_quartets / 1e6) / (med_64 / 1e3)
        mq_s_32 = (n_quartets / 1e6) / (med_32 / 1e3)
        row = {
            "system": sys_name,
            "basis": basis_name,
            "description": label,
            "nao": basis.nao,
            "nshell": nsh,
            "fp64_ms": med_64,
            "fp32_ms": med_32,
            "fp64_mquartets_per_s": mq_s_64,
            "fp32_mquartets_per_s": mq_s_32,
            "fp32_speedup": med_64 / max(med_32, 1e-9),
        }
        rows.append(row)
        print(
            f"  {sys_name}/{basis_name} (nao={basis.nao}, nsh={nsh}): "
            f"FP64={med_64:6.2f} ms ({mq_s_64:6.2f} Mq/s) | "
            f"FP32={med_32:6.2f} ms ({mq_s_32:6.2f} Mq/s, {row['fp32_speedup']:.2f}x)"
        )
    return rows


# ---------------------------------------------------------------------------
# Suite 3: Solver & CUDA Graph Ablation (cuSOLVER Diag vs SP2 + OzakiInt8)
# ---------------------------------------------------------------------------


def run_solver_ablation_suite(backend: str) -> list[dict[str, Any]]:
    print("\n=== [3/7] Solver & CUDA Graph Ablation (Diag vs SP2 + OzakiInt8) ===")
    targets = [
        ("C8H18", "def2-svp"),
        ("(H2O)_8", "def2-svp"),
        ("C12H26", "def2-svp"),
    ]
    rows: list[dict[str, Any]] = []
    for sys_name, basis_name in targets:
        mol = build_mol(sys_name, basis_name)
        dm0 = gpu4dft.minao_init_guess(mol)
        configs = [
            ("RI-RHF Diag (FP64)", "diag", "fp64", True),
            ("RI-RHF Diag (OzakiInt8 + Graph)", "diag", "ozaki_int8", True),
            ("RI-RHF SP2 (OzakiInt8, no Graph)", "sp2", "ozaki_int8", False),
            ("RI-RHF SP2 (OzakiInt8 + Graph)", "sp2", "ozaki_int8", True),
        ]
        e_ref_diag = None
        for label, solver, gmode, use_graph in configs:
            mf = gpu4dft.RHF(
                mol,
                backend=backend,
                solver=solver,
                gemm_mode=gmode,
                use_cuda_graph=use_graph,
                conv_tol=1e-9,
            ).density_fit(auxbasis="def2-universal-jkfit")
            mf.setup()
            sync_gpu()
            t0 = time.perf_counter()
            e_tot = mf.kernel(dm0=dm0)
            sync_gpu()
            scf_wall = time.perf_counter() - t0
            if e_ref_diag is None:
                e_ref_diag = e_tot
            diff = abs(e_tot - e_ref_diag)
            row = {
                "system": sys_name,
                "basis": basis_name,
                "nao": mol.nao,
                "config": label,
                "solver": solver,
                "gemm_mode": gmode,
                "use_cuda_graph": use_graph,
                "converged": mf.converged,
                "iterations": mf.iterations,
                "energy_hartree": e_tot,
                "diff_vs_diag_fp64_eh": diff,
                "scf_time_s": scf_wall,
                "per_iter_ms": (scf_wall * 1e3) / max(1, mf.iterations),
                "cuda_graph_captures": int(mf.result.scf.cuda_graph_captures),
                "cuda_graph_replays": int(mf.result.scf.cuda_graph_replays),
                "steady_state_allocs": int(mf.result.scf.steady_state_allocations),
            }
            rows.append(row)
            print(
                f"  {sys_name}/{basis_name} [{label}]: "
                f"T_scf={scf_wall*1e3:7.2f} ms ({mf.iterations} it, |dE|={diff:.2e} Eh, "
                f"graphs={row['cuda_graph_replays']}, allocs={row['steady_state_allocs']})"
            )
            del mf
            gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Suite 4: Head-to-Head SCF Benchmarks (GPU4DFT vs GPU4PySCF vs CPU PySCF)
# ---------------------------------------------------------------------------


def _run_pyscf_cpu_scf(
    mol: gto.Mole, method: str, dm0: np.ndarray, density_fit: bool, grids_level: int = 3
) -> dict[str, Any]:
    is_hf = method.lower() == "rhf"
    if is_hf:
        mf = scf.RHF(mol)
    else:
        mf = dft.RKS(mol, xc=method)
        mf.grids.level = grids_level
    if density_fit:
        mf = mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.conv_tol = 1e-9
    mf.diis_space = 8
    mf.direct_scf_tol = 1e-13

    t0 = time.perf_counter()
    if not is_hf:
        mf.grids.build()
    e_tot = float(mf.kernel(dm0=dm0))
    t_e2e = time.perf_counter() - t0
    return {
        "engine": "PySCF (CPU)" + (" [DF]" if density_fit else " [Direct]"),
        "converged": bool(mf.converged),
        "energy_hartree": e_tot,
        "time_e2e_s": t_e2e,
        "mf_obj": mf,
    }


def _run_gpu4pyscf_scf(
    mol: gto.Mole, method: str, dm0: np.ndarray, density_fit: bool, grids_level: int = 3
) -> dict[str, Any] | None:
    if not HAVE_GPU4PYSCF:
        return None
    try:
        is_hf = method.lower() == "rhf"
        if is_hf:
            mf = gpu4pyscf_scf.RHF(mol)
        else:
            mf = gpu4pyscf_dft.RKS(mol, xc=method)
            mf.grids.level = grids_level
        if density_fit:
            mf = mf.density_fit(auxbasis="def2-universal-jkfit")
        mf.conv_tol = 1e-9
        mf.diis_space = 8
        mf.direct_scf_tol = 1e-13

        sync_gpu()
        t0 = time.perf_counter()
        e_tot = float(mf.kernel(dm0=dm0))
        sync_gpu()
        t_e2e = time.perf_counter() - t0
        return {
            "engine": "GPU4PySCF" + (" [DF]" if density_fit else " [Direct]"),
            "converged": bool(mf.converged),
            "energy_hartree": e_tot,
            "time_e2e_s": t_e2e,
            "mf_obj": mf,
        }
    except Exception as exc:
        return {
            "engine": "GPU4PySCF" + (" [DF]" if density_fit else " [Direct]"),
            "error": str(exc),
        }


def run_end_to_end_scf_suite(
    backend: str, preset: str, include_slow_cpu: bool
) -> list[dict[str, Any]]:
    print("\n=== [4/7] Head-to-Head SCF Benchmarks (GPU4DFT vs GPU4PySCF vs CPU PySCF) ===")

    # Matrix of (system, basis, method) covering all requested molecules & basis sets
    if preset == "quick":
        cases = [
            ("(H2O)_2", "6-31g*", "b3lyp"),
            ("C6H6", "def2-svp", "b3lyp"),
            ("(H2O)_8", "def2-svp", "b3lyp"),
            ("aspirin", "def2-svp", "b3lyp"),
            ("(H2O)_2", "def2-tzvp", "b3lyp"),
        ]
    else:
        cases = [
            ("(H2O)_2", "6-31g*", "b3lyp"),
            ("(H2O)_2", "def2-svp", "b3lyp"),
            ("(H2O)_2", "def2-tzvp", "b3lyp"),
            ("C6H6", "6-31g*", "b3lyp"),
            ("C6H6", "def2-svp", "b3lyp"),
            ("C6H6", "def2-tzvp", "b3lyp"),
            ("C8H18", "def2-svp", "b3lyp"),
            ("(H2O)_8", "6-31g*", "b3lyp"),
            ("(H2O)_8", "def2-svp", "b3lyp"),
            ("(H2O)_8", "def2-tzvp", "pbe"),
            ("aspirin", "6-31g*", "b3lyp"),
            ("aspirin", "def2-svp", "b3lyp"),
            ("C12H26", "def2-svp", "b3lyp"),
            ("(H2O)_16", "6-31g*", "b3lyp"),
            ("(H2O)_16", "def2-svp", "b3lyp"),
        ]
        if preset == "full":
            cases.extend(
                [
                    ("aspirin", "def2-tzvp", "b3lyp"),
                    ("(H2O)_16", "def2-tzvp", "pbe"),
                ]
            )

    rows: list[dict[str, Any]] = []

    for sys_name, basis_name, method in cases:
        mol = build_mol(sys_name, basis_name)
        nao = mol.nao
        dm0 = gpu4dft.minao_init_guess(mol)
        is_hf = method.lower() == "rhf"

        # Prebuild PySCF level-3 quadrature grid once for exact 1:1 grid apple-to-apple accuracy check
        shared_grids = None
        if not is_hf:
            g_ref = dft.Grids(mol)
            g_ref.level = 3
            g_ref.build()
            shared_grids = (
                np.ascontiguousarray(g_ref.coords, dtype=np.float64),
                np.ascontiguousarray(g_ref.weights, dtype=np.float64),
            )

        # Run CPU PySCF Direct on moderate systems (nao <= 260 unless --include-slow-cpu)
        cpu_direct = None
        if nao <= 260 or include_slow_cpu:
            cpu_direct = _run_pyscf_cpu_scf(mol, method, dm0, density_fit=False)
        cpu_df = _run_pyscf_cpu_scf(mol, method, dm0, density_fit=True)

        # Run GPU4PySCF Direct & DF
        g4p_direct = _run_gpu4pyscf_scf(mol, method, dm0, density_fit=False)
        g4p_df = _run_gpu4pyscf_scf(mol, method, dm0, density_fit=True)

        ref_exact_e = (
            cpu_direct["energy_hartree"]
            if cpu_direct is not None
            else (
                g4p_direct["energy_hartree"]
                if g4p_direct and "energy_hartree" in g4p_direct
                else None
            )
        )
        ref_df_e = cpu_df["energy_hartree"]

        cpu_dir_t = cpu_direct["time_e2e_s"] if cpu_direct else None
        cpu_df_t = cpu_df["time_e2e_s"]
        g4p_dir_t = g4p_direct["time_e2e_s"] if g4p_direct and "time_e2e_s" in g4p_direct else None
        g4p_df_t = g4p_df["time_e2e_s"] if g4p_df and "time_e2e_s" in g4p_df else None

        # GPU4DFT configurations
        g4d_modes: list[tuple[str, dict[str, Any], bool, bool]] = []
        if nao <= 400:
            g4d_modes.append(
                (
                    "Direct_FP64",
                    dict(precision="fp64", dynamic_precision=False, gemm_mode="fp64"),
                    False,
                    False,
                )
            )
            g4d_modes.append(
                (
                    "Direct_Mixed_OzakiInt8",
                    dict(precision="mixed", dynamic_precision=True, gemm_mode="ozaki_int8"),
                    False,
                    False,
                )
            )
        g4d_modes.extend(
            [
                (
                    "RI_JK_FP64",
                    dict(precision="fp64", dynamic_precision=False, gemm_mode="fp64"),
                    True,
                    False,
                ),
                (
                    "RI_JK_OzakiInt8",
                    dict(precision="mixed", dynamic_precision=True, gemm_mode="ozaki_int8"),
                    True,
                    False,
                ),
            ]
        )
        if method.lower() in ("rhf", "b3lyp", "pbe0") and nao <= 300:
            g4d_modes.append(
                (
                    "RIJCOSX_snK",
                    dict(precision="fp64", dynamic_precision=False, gemm_mode="ozaki_int8"),
                    True,
                    True,
                )
            )

        print(f"\n  --- {sys_name} / {basis_name} ({method.upper()}, nao={nao}, nelec={mol.nelectron}) ---")
        if cpu_dir_t is not None:
            print(f"    PySCF CPU [Direct] : {cpu_dir_t:7.3f} s  (E = {ref_exact_e:.10f} Eh)")
        print(f"    PySCF CPU [DF]     : {cpu_df_t:7.3f} s  (E = {ref_df_e:.10f} Eh)")
        if g4p_dir_t is not None:
            print(f"    GPU4PySCF [Direct] : {g4p_dir_t:7.3f} s  (E = {g4p_direct['energy_hartree']:.10f} Eh)")
        if g4p_df_t is not None:
            print(f"    GPU4PySCF [DF]     : {g4p_df_t:7.3f} s  (E = {g4p_df['energy_hartree']:.10f} Eh)")

        for mode_label, kw, use_df, use_snk in g4d_modes:
            mf = (
                gpu4dft.RHF(mol, backend=backend, conv_tol=1e-9, **kw)
                if is_hf
                else gpu4dft.RKS(
                    mol, xc=method, backend=backend, grids=shared_grids, conv_tol=1e-9, **kw
                )
            )
            if use_df:
                mf.density_fit(auxbasis="def2-universal-jkfit")
            if use_snk:
                mf.snk()

            sync_gpu()
            t0 = time.perf_counter()
            e_got = mf.kernel(dm0=dm0)
            sync_gpu()
            t_e2e = time.perf_counter() - t0

            wres = mf.result
            ref_cmp = ref_df_e if (use_df and not use_snk) else (ref_exact_e or ref_df_e)
            abs_diff = abs(e_got - ref_cmp) if ref_cmp is not None else None

            base_cpu_t = cpu_df_t if use_df else cpu_dir_t
            base_g4p_t = g4p_df_t if use_df else g4p_dir_t
            sp_cpu = (base_cpu_t / t_e2e) if (base_cpu_t and t_e2e > 0) else None
            sp_g4p = (base_g4p_t / t_e2e) if (base_g4p_t and t_e2e > 0) else None
            sp_scf_vs_g4p = (
                (base_g4p_t / wres.scf_time_s) if (base_g4p_t and wres.scf_time_s > 0) else None
            )

            rec = {
                "system": sys_name,
                "basis": basis_name,
                "nao": nao,
                "nelec": mol.nelectron,
                "method": method.upper(),
                "jk_mode": mode_label,
                "gemm_mode": kw["gemm_mode"],
                "solver": "Diag_CudaGraph",
                "converged": mf.converged,
                "iterations": mf.iterations,
                "energy_hartree": e_got,
                "abs_diff_vs_pyscf_eh": abs_diff,
                "time_setup_s": wres.setup_time_s,
                "time_scf_s": wres.scf_time_s,
                "time_e2e_s": t_e2e,
                "time_per_iter_ms": (wres.scf_time_s * 1e3) / max(1, mf.iterations),
                "active_ao_ratio": wres.active_ao_ratio,
                "steady_state_allocations": int(wres.scf.steady_state_allocations),
                "peak_vram_mib": get_peak_vram_mib(),
                "pyscf_cpu_time_s": base_cpu_t,
                "gpu4pyscf_time_s": base_g4p_t,
                "speedup_vs_pyscf_cpu": sp_cpu,
                "speedup_e2e_vs_gpu4pyscf": sp_g4p,
                "speedup_scf_loop_vs_gpu4pyscf": sp_scf_vs_g4p,
            }
            rows.append(rec)
            diff_str = f"{abs_diff:.2e}" if abs_diff is not None else "N/A"
            cpu_str = f"{sp_cpu:5.2f}x" if sp_cpu is not None else "  N/A "
            g4p_str = f"{sp_g4p:5.2f}x" if sp_g4p is not None else "  N/A "
            print(
                f"    GPU4DFT [{mode_label:22s}]: E2E={t_e2e:7.3f} s "
                f"(setup={wres.setup_time_s:6.3f}s, scf={wres.scf_time_s:6.3f}s, {mf.iterations:2d} it) | "
                f"|dE|={diff_str} Eh | vs CPU={cpu_str} | vs GPU4PySCF={g4p_str}"
            )
            del mf
            gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Suite 5: Analytical Nuclear Gradients Benchmark
# ---------------------------------------------------------------------------


def run_gradient_suite(backend: str, preset: str) -> list[dict[str, Any]]:
    print("\n=== [5/7] Analytical Nuclear Gradients Benchmark ===")
    cases = [
        ("C6H6", "def2-svp", "rhf"),
        ("C6H6", "def2-svp", "b3lyp"),
        ("(H2O)_8", "6-31g*", "b3lyp"),
        ("aspirin", "6-31g*", "b3lyp"),
    ]
    if preset != "quick":
        cases.append(("C8H18", "def2-svp", "pbe"))

    rows: list[dict[str, Any]] = []
    for sys_name, basis_name, method in cases:
        mol = build_mol(sys_name, basis_name)
        dm0 = gpu4dft.minao_init_guess(mol)
        is_hf = method.lower() == "rhf"

        # Reference PySCF CPU SCF + Gradients
        if is_hf:
            mf_cpu = scf.RHF(mol)
        else:
            mf_cpu = dft.RKS(mol, xc=method)
            mf_cpu.grids.level = 3
            mf_cpu.grids.build()
        mf_cpu.conv_tol = 1e-10
        mf_cpu.kernel(dm0=dm0)

        t0 = time.perf_counter()
        g_cpu = (
            pyscf_grad_rhf.Gradients(mf_cpu).kernel()
            if is_hf
            else pyscf_grad_rks.Gradients(mf_cpu).kernel()
        )
        t_grad_cpu = time.perf_counter() - t0

        # Optional GPU4PySCF Gradients
        t_grad_g4p = None
        if HAVE_GPU4PYSCF:
            try:
                mf_g = (
                    gpu4pyscf_scf.RHF(mol)
                    if is_hf
                    else gpu4pyscf_dft.RKS(mol, xc=method)
                )
                if not is_hf:
                    mf_g.grids.level = 3
                mf_g.conv_tol = 1e-10
                mf_g.kernel(dm0=dm0)
                sync_gpu()
                t0 = time.perf_counter()
                _ = mf_g.nuc_grad_method().kernel()
                sync_gpu()
                t_grad_g4p = time.perf_counter() - t0
            except Exception:
                t_grad_g4p = None

        # GPU4DFT SCF + Analytical Nuclear Gradients
        mf = (
            gpu4dft.RHF(mol, backend=backend, precision="mixed", gemm_mode="ozaki_int8", conv_tol=1e-10)
            if is_hf
            else gpu4dft.RKS(
                mol,
                xc=method,
                backend=backend,
                precision="mixed",
                gemm_mode="ozaki_int8",
                grids=(mf_cpu.grids.coords, mf_cpu.grids.weights),
                conv_tol=1e-10,
            )
        )
        sync_gpu()
        t0 = time.perf_counter()
        mf.kernel(dm0=dm0)
        sync_gpu()
        t_scf_g4d = time.perf_counter() - t0

        g_driver = mf.nuc_grad_method()
        sync_gpu()
        t0 = time.perf_counter()
        g_got = g_driver.kernel()
        sync_gpu()
        t_grad_g4d = time.perf_counter() - t0

        max_grad_diff = float(np.max(np.abs(g_got - g_cpu)))
        net_force = float(np.max(np.abs(np.sum(g_got, axis=0))))
        ratio_grad_scf = t_grad_g4d / max(t_scf_g4d, 1e-12)
        sp_cpu = t_grad_cpu / max(t_grad_g4d, 1e-12)
        sp_g4p = (t_grad_g4p / max(t_grad_g4d, 1e-12)) if t_grad_g4p is not None else None

        rec = {
            "system": sys_name,
            "basis": basis_name,
            "nao": mol.nao,
            "natom": mol.natm,
            "method": method.upper(),
            "time_scf_s": t_scf_g4d,
            "time_grad_s": t_grad_g4d,
            "time_grad_1e_s": g_driver.result.one_electron_time_s,
            "time_grad_2e_s": g_driver.result.two_electron_time_s,
            "time_grad_xc_s": g_driver.result.xc_time_s,
            "grad_to_scf_ratio": ratio_grad_scf,
            "max_grad_diff_eh_bohr": max_grad_diff,
            "net_translation_force_eh_bohr": net_force,
            "pyscf_cpu_grad_s": t_grad_cpu,
            "gpu4pyscf_grad_s": t_grad_g4p,
            "speedup_vs_pyscf_cpu": sp_cpu,
            "speedup_vs_gpu4pyscf": sp_g4p,
        }
        rows.append(rec)
        g4p_str = f"{t_grad_g4p:6.3f}s ({sp_g4p:4.2f}x)" if t_grad_g4p is not None else "N/A"
        print(
            f"  {sys_name}/{basis_name} ({method.upper()}, nao={mol.nao}): "
            f"T_grad={t_grad_g4d:6.3f}s (T_grad/T_scf={ratio_grad_scf:.2f}) | "
            f"max|dg|={max_grad_diff:.2e} Eh/Bohr | CPU={t_grad_cpu:6.3f}s ({sp_cpu:5.2f}x) | GPU4PySCF={g4p_str}"
        )
    return rows


# ---------------------------------------------------------------------------
# Suite 6: GPU TDDFT & RI-TDDFT Excited States vs PySCF & GPU4PySCF
# ---------------------------------------------------------------------------


def _run_pyscf_cpu_td(
    mf_cpu: Any, resp_kind: str, nstates: int = 5, conv_tol: float = 1e-6
) -> dict[str, Any]:
    td = tdscf.TDA(mf_cpu) if resp_kind == "tda" else tdscf.TDDFT(mf_cpu)
    td.nstates = nstates
    td.conv_tol = conv_tol
    t0 = time.perf_counter()
    td.kernel()
    f_osc = np.asarray(td.oscillator_strength(), dtype=np.float64)
    t_td = time.perf_counter() - t0
    return {
        "converged": bool(np.all(td.converged)) if isinstance(td.converged, (list, tuple, np.ndarray)) else bool(td.converged),
        "e_eh": np.asarray(td.e, dtype=np.float64),
        "f_osc": f_osc,
        "time_td_s": t_td,
    }


def _run_gpu4pyscf_td(
    mol: gto.Mole,
    xc: str,
    dm0: np.ndarray,
    density_fit: bool,
    resp_kind: str,
    nstates: int = 5,
    conv_tol: float = 1e-6,
) -> dict[str, Any] | None:
    if not HAVE_GPU4PYSCF:
        return None
    try:
        is_hf = xc.lower() == "rhf"
        mf_g = gpu4pyscf_scf.RHF(mol) if is_hf else gpu4pyscf_dft.RKS(mol, xc=xc)
        if not is_hf:
            mf_g.grids.level = 3
        if density_fit:
            mf_g = mf_g.density_fit(auxbasis="def2-universal-jkfit")
        mf_g.conv_tol = 1e-10
        sync_gpu()
        t_scf0 = time.perf_counter()
        mf_g.kernel(dm0=dm0)
        sync_gpu()
        t_scf = time.perf_counter() - t_scf0

        td_g = mf_g.TDA() if resp_kind == "tda" else mf_g.TDDFT()
        td_g.nstates = nstates
        td_g.conv_tol = conv_tol
        sync_gpu()
        t0 = time.perf_counter()
        td_g.kernel()
        sync_gpu()
        t_td = time.perf_counter() - t0
        return {
            "time_scf_s": t_scf,
            "time_td_s": t_td,
            "time_e2e_s": t_scf + t_td,
        }
    except Exception:
        return None


def run_tddft_suite(backend: str, preset: str) -> list[dict[str, Any]]:
    print("\n=== [6/7] GPU TDDFT & RI-TDDFT Excited States Benchmark ===")
    nstates = 5
    if preset == "quick":
        groups = [
            ("C6H6", "def2-svp", "b3lyp", [("tddft", False, "fp64"), ("tddft", True, "ozaki_int8")]),
            ("(H2O)_8", "def2-svp", "pbe", [("tddft", True, "ozaki_int8")]),
        ]
    else:
        groups = [
            (
                "C6H6",
                "def2-svp",
                "b3lyp",
                [
                    ("tddft", False, "fp64"),
                    ("tddft", True, "fp64"),
                    ("tddft", True, "ozaki_int8"),
                    ("tda", True, "ozaki_int8"),
                ],
            ),
            (
                "(H2O)_8",
                "def2-svp",
                "pbe",
                [
                    ("tddft", False, "fp64"),
                    ("tddft", True, "fp64"),
                    ("tddft", True, "ozaki_int8"),
                ],
            ),
            (
                "(H2O)_8",
                "def2-svp",
                "b3lyp",
                [
                    ("tddft", True, "fp64"),
                    ("tddft", True, "ozaki_int8"),
                ],
            ),
            (
                "aspirin",
                "def2-svp",
                "b3lyp",
                [
                    ("tda", True, "ozaki_int8"),
                    ("tddft", True, "fp64"),
                    ("tddft", True, "ozaki_int8"),
                ],
            ),
        ]

    rows: list[dict[str, Any]] = []
    for sys_name, basis_name, xc_name, specs in groups:
        mol = build_mol(sys_name, basis_name)
        dm0 = gpu4dft.minao_init_guess(mol)
        g_ref = dft.Grids(mol)
        g_ref.level = 3
        g_ref.build()
        shared_grids = (
            np.ascontiguousarray(g_ref.coords, dtype=np.float64),
            np.ascontiguousarray(g_ref.weights, dtype=np.float64),
        )

        # Cache CPU PySCF SCF + TD references per (use_df, resp_kind)
        cpu_scf_cache: dict[bool, dict[str, Any]] = {}
        cpu_td_cache: dict[tuple[bool, str], dict[str, Any]] = {}
        g4p_td_cache: dict[tuple[bool, str], dict[str, Any] | None] = {}

        for resp_kind, use_df, gmode in specs:
            if use_df not in cpu_scf_cache:
                mf_cpu = dft.RKS(mol, xc=xc_name)
                mf_cpu.grids.level = 3
                if use_df:
                    mf_cpu = mf_cpu.density_fit(auxbasis="def2-universal-jkfit")
                mf_cpu.conv_tol = 1e-10
                t_c0 = time.perf_counter()
                mf_cpu.grids.build()
                mf_cpu.kernel(dm0=dm0)
                t_cpu_scf = time.perf_counter() - t_c0
                cpu_scf_cache[use_df] = {"mf": mf_cpu, "time_scf_s": t_cpu_scf}

            if (use_df, resp_kind) not in cpu_td_cache:
                cpu_td_cache[(use_df, resp_kind)] = _run_pyscf_cpu_td(
                    cpu_scf_cache[use_df]["mf"], resp_kind, nstates=nstates, conv_tol=1e-6
                )
            if (use_df, resp_kind) not in g4p_td_cache:
                g4p_td_cache[(use_df, resp_kind)] = _run_gpu4pyscf_td(
                    mol, xc_name, dm0, use_df, resp_kind, nstates=nstates, conv_tol=1e-6
                )

            ref_td = cpu_td_cache[(use_df, resp_kind)]
            t_cpu_scf = cpu_scf_cache[use_df]["time_scf_s"]
            t_cpu_td = ref_td["time_td_s"]
            t_cpu_e2e = t_cpu_scf + t_cpu_td

            g4p_td = g4p_td_cache[(use_df, resp_kind)]
            t_g4p_td = g4p_td["time_td_s"] if g4p_td is not None else None

            mf = gpu4dft.RKS(
                mol,
                xc=xc_name,
                backend=backend,
                precision="mixed" if gmode == "ozaki_int8" else "fp64",
                dynamic_precision=(gmode == "ozaki_int8"),
                gemm_mode=gmode,
                grids=shared_grids,
                conv_tol=1e-10,
            )
            if use_df:
                mf.density_fit(auxbasis="def2-universal-jkfit")

            sync_gpu()
            t_s0 = time.perf_counter()
            mf.kernel(dm0=dm0)
            sync_gpu()
            t_scf_g4d = time.perf_counter() - t_s0

            td = (
                mf.TDA(nstates=nstates, gemm_mode=gmode, conv_tol=1e-6)
                if resp_kind == "tda"
                else mf.TDDFT(nstates=nstates, gemm_mode=gmode, conv_tol=1e-6)
            )
            sync_gpu()
            t_r0 = time.perf_counter()
            e_got, _ = td.kernel()
            sync_gpu()
            t_td_g4d = time.perf_counter() - t_r0
            t_e2e_g4d = t_scf_g4d + t_td_g4d

            f_got = td.oscillator_strength()
            max_de_eh = float(np.max(np.abs(e_got - ref_td["e_eh"])))
            max_de_ev = max_de_eh * 27.21138602
            max_df = float(np.max(np.abs(f_got - ref_td["f_osc"])))
            sp_td_cpu = t_cpu_td / max(t_td_g4d, 1e-12)
            sp_e2e_cpu = t_cpu_e2e / max(t_e2e_g4d, 1e-12)
            sp_td_g4p = (t_g4p_td / max(t_td_g4d, 1e-12)) if t_g4p_td is not None else None

            mode_tag = ("RI_" if use_df else "Direct_") + resp_kind.upper() + f"_{gmode}"
            rec = {
                "system": sys_name,
                "basis": basis_name,
                "nao": mol.nao,
                "n_occ": int(td.result.n_occ),
                "n_vir": int(td.result.n_vir),
                "xc": xc_name.upper(),
                "response_method": resp_kind.upper(),
                "mode": mode_tag,
                "use_df": use_df,
                "gemm_mode": gmode,
                "nstates": nstates,
                "converged": td.converged,
                "iterations": td.iterations,
                "matvec_count": int(td.result.matvec_count),
                "lowest_excitation_ev": float(td.e_ev[0]),
                "max_oscillator_strength": float(np.max(f_got)),
                "max_abs_omega_diff_eh": max_de_eh,
                "max_abs_omega_diff_ev": max_de_ev,
                "max_abs_fosc_diff": max_df,
                "time_scf_s": t_scf_g4d,
                "time_td_s": t_td_g4d,
                "time_e2e_s": t_e2e_g4d,
                "pyscf_cpu_td_s": t_cpu_td,
                "pyscf_cpu_e2e_s": t_cpu_e2e,
                "gpu4pyscf_td_s": t_g4p_td,
                "speedup_td_vs_pyscf_cpu": sp_td_cpu,
                "speedup_e2e_vs_pyscf_cpu": sp_e2e_cpu,
                "speedup_td_vs_gpu4pyscf": sp_td_g4p,
            }
            rows.append(rec)
            g4p_s = f"{t_g4p_td:6.3f}s ({sp_td_g4p:4.2f}x)" if t_g4p_td is not None else "N/A"
            print(
                f"  {sys_name}/{basis_name} [{xc_name.upper()} {mode_tag:22s}]: "
                f"T_td={t_td_g4d:6.3f}s ({td.iterations:2d} it, {rec['matvec_count']:2d} mv) | "
                f"E1={rec['lowest_excitation_ev']:.3f} eV | max|dw|={max_de_eh:.2e} Eh | "
                f"CPU_td={t_cpu_td:6.3f}s ({sp_td_cpu:5.2f}x) | GPU4PySCF={g4p_s}"
            )
            del td, mf
            gc.collect()
    return rows


# ---------------------------------------------------------------------------
# Suite 7: Plane-Wave Kohn-Sham & Lin's ACE Hybrid DFT Benchmark
# ---------------------------------------------------------------------------


class _BenchPyscfExactPWCell(pbc_gto.Cell):
    """Exact spherical plane-wave basis adapter for PySCF-PBC FFTDF head-to-head comparison."""

    def __init__(self, ecut_wfc: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ecut_wfc = float(ecut_wfc)
        self._pw_cache: dict[str, np.ndarray] = {}

    def build(self, *args: Any, **kwargs: Any) -> Any:
        res = super().build(*args, **kwargs)
        gv = self.get_Gv(self.mesh)
        self._max_npw = int(np.sum(0.5 * np.einsum("gx,gx->g", gv, gv) <= self.ecut_wfc + 1e-12))
        return res

    def _active_gv(self) -> tuple[np.ndarray, np.ndarray]:
        gv = self.get_Gv(self.mesh)
        kin = 0.5 * np.einsum("gx,gx->g", gv, gv)
        idx = np.where(kin <= self.ecut_wfc + 1e-12)[0]
        return gv[idx], kin[idx]

    def nao_nr(self, cart: Any = None) -> int:
        return self._max_npw

    def _pos_half_g(self, gv: np.ndarray) -> np.ndarray:
        c0 = gv[:, 0] > 1e-12
        c1 = (np.abs(gv[:, 0]) <= 1e-12) & (gv[:, 1] > 1e-12)
        c2 = (np.abs(gv[:, 0]) <= 1e-12) & (np.abs(gv[:, 1]) <= 1e-12) & (gv[:, 2] > 1e-12)
        return gv[c0 | c1 | c2]

    def _full_ao(self, want_deriv1: bool) -> np.ndarray:
        key = str(int(want_deriv1))
        if key in self._pw_cache:
            return self._pw_cache[key]
        coords = self.get_uniform_grids()
        gv, _ = self._active_gv()
        pos = self._pos_half_g(gv)
        npos = len(pos)
        nao = self._max_npw
        ao0 = np.zeros((len(coords), nao), dtype=np.float64)
        ao0[:, 0] = 1.0 / math.sqrt(self.vol)
        phase = coords @ pos.T
        cp_m = math.sqrt(2.0 / self.vol) * np.cos(phase)
        sp_m = math.sqrt(2.0 / self.vol) * np.sin(phase)
        ao0[:, 1 : 1 + npos] = cp_m
        ao0[:, 1 + npos : 1 + 2 * npos] = sp_m
        if want_deriv1:
            ao1 = np.zeros((4, len(coords), nao), dtype=np.float64)
            ao1[0] = ao0
            for d in range(3):
                ao1[1 + d, :, 1 : 1 + npos] = -sp_m * pos[None, :, d]
                ao1[1 + d, :, 1 + npos : 1 + 2 * npos] = cp_m * pos[None, :, d]
            res = ao1
        else:
            res = ao0
        self._pw_cache[key] = res
        return res

    def pbc_eval_gto(
        self,
        eval_name: str,
        coords: np.ndarray,
        comp: Any = None,
        kpts: Any = None,
        kpt: Any = None,
        **kwargs: Any,
    ) -> Any:
        _ = (comp, kpt, kwargs)
        want_d1 = ("deriv1" in eval_name) or ("ip" in eval_name)
        ugrids = self.get_uniform_grids()
        if len(coords) == len(ugrids):
            sl = slice(None)
        else:
            p0 = int(np.argmin(np.sum(np.abs(ugrids - coords[0]), axis=1)))
            sl = slice(p0, p0 + len(coords))
        full = self._full_ao(want_d1)
        out = full[:, sl, :] if want_d1 else full[sl, :]
        if kpts is not None and np.ndim(kpts) == 2:
            return [out for _ in range(len(kpts))]
        return out

    def pbc_intor(
        self,
        intor: str,
        hermi: int = 1,
        kpts: Any = None,
        kpt: Any = None,
        **kwargs: Any,
    ) -> Any:
        _ = (hermi, kpt, kwargs)
        nao = self._max_npw
        if "ovlp" in intor:
            mat = np.eye(nao, dtype=np.float64)
        else:
            gv, _ = self._active_gv()
            pos = self._pos_half_g(gv)
            npos = len(pos)
            kin_pos = 0.5 * np.einsum("gx,gx->g", pos, pos)
            d = np.zeros(nao, dtype=np.float64)
            d[1 : 1 + npos] = kin_pos
            d[1 + npos : 1 + 2 * npos] = kin_pos
            mat = np.diag(d)
        if kpts is not None and np.ndim(kpts) == 2:
            return np.asarray([mat for _ in range(len(kpts))])
        return mat


def _diamond_si_supercell(nx: int, ny: int, nz: int, a_ang: float = 5.43) -> tuple[np.ndarray, list[int], np.ndarray]:
    a_b = a_ang * 1.8897259885789233
    a_prim = np.array(
        [[0.0, 0.5 * a_b, 0.5 * a_b], [0.5 * a_b, 0.0, 0.5 * a_b], [0.5 * a_b, 0.5 * a_b, 0.0]],
        dtype=np.float64,
    )
    prim_pos = np.array([[0.0, 0.0, 0.0], [0.25 * a_b, 0.25 * a_b, 0.25 * a_b]], dtype=np.float64)
    a_super = np.array([nx * a_prim[0], ny * a_prim[1], nz * a_prim[2]], dtype=np.float64)
    coords_list = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = ix * a_prim[0] + iy * a_prim[1] + iz * a_prim[2]
                coords_list.append(prim_pos + shift)
    coords = np.vstack(coords_list)
    return a_super, [14] * len(coords), coords


def _run_pyscf_pbc_pw(
    a_bohr: np.ndarray, coords_bohr: np.ndarray, xc: str, ecut: float, mesh: tuple[int, int, int]
) -> tuple[float, float]:
    atoms_pyscf = [("Si", p.tolist()) for p in coords_bohr]
    t0 = time.perf_counter()
    pcell = _BenchPyscfExactPWCell(
        ecut_wfc=ecut,
        a=a_bohr,
        atom=atoms_pyscf,
        unit="B",
        pseudo="gth-pbe",
        mesh=list(mesh),
        verbose=0,
    )
    pcell.build()
    hcore = pcell.pbc_intor("int1e_kin") + pbc_pp.get_pp(pcell)
    mf = pbc_dft.RKS(pcell, xc=xc)
    mf.exxdiv = "ewald"
    mf.chkfile = None
    mf.get_hcore = lambda *a, **kw: hcore
    mf.conv_tol = 1e-9
    e0, u0 = mf.eig(hcore, np.eye(pcell.nao_nr()))
    dm0 = mf.make_rdm1(u0, mf.get_occ(e0, u0))
    etot = float(mf.kernel(dm0=dm0))
    elapsed_ms = (time.perf_counter() - t0) * 1e3
    return etot, elapsed_ms


def run_pw_dft_suite(backend: str, preset: str) -> list[dict[str, Any]]:
    print("\n=== [7/7] Plane-Wave Kohn-Sham & ACE Hybrid DFT Benchmark ===")
    has_cuda_pw = (backend == "cuda") and gpu4dft.pw.cuda_pw_available()

    if preset == "quick":
        cases = [
            ("Diamond Si2", (1, 1, 1), "pbe0", 10.0, True),
            ("Diamond Si4 (2x1x1)", (2, 1, 1), "pbe0", 16.0, False),
        ]
    else:
        cases = [
            ("Diamond Si2", (1, 1, 1), "pbe", 10.0, True),
            ("Diamond Si2", (1, 1, 1), "pbe0", 10.0, True),
            ("Diamond Si2", (1, 1, 1), "pbe0", 15.0, False),
            ("Diamond Si4 (2x1x1)", (2, 1, 1), "pbe0", 18.0, False),
            ("Diamond Si8 (2x2x1)", (2, 2, 1), "pbe", 20.0, False),
            ("Diamond Si8 (2x2x1)", (2, 2, 1), "pbe0", 16.0, False),
        ]

    rows: list[dict[str, Any]] = []
    for label, sc, xc_name, ecut, run_pyscf_ref in cases:
        a_bohr, z_list, coords_bohr = _diamond_si_supercell(*sc)
        cell = gpu4dft.pw.PwCell(a_bohr, z_list, coords_bohr, "gth-pbe")
        basis = gpu4dft.pw.PwBasis(cell, ecut_wfc=ecut)
        mesh = tuple(int(x) for x in basis.fft_grid)
        npw = int(basis.kpoint(0).npw)
        is_hybrid = xc_name.lower() in ("pbe0", "b3lyp", "hf")

        # Warmup GPU cuFFT plans on this grid
        if has_cuda_pw:
            _ = gpu4dft.pw.run_scf(
                cell, xc=xc_name, ecut_wfc=ecut, backend="cuda", use_ace=True, conv_tol=1e-3
            )
            sync_gpu()

        # 1. GPU4DFT CUDA (ACE for hybrids)
        gpu_res = None
        gpu_ms = None
        if has_cuda_pw:
            t_samples = []
            for _ in range(2):
                r = gpu4dft.pw.run_scf(
                    cell, xc=xc_name, ecut_wfc=ecut, backend="cuda", use_ace=True, conv_tol=1e-9
                )
                t_samples.append(float(r.elapsed_ms))
                gpu_res = r
            gpu_ms = statistics.median(t_samples)

        # 2. GPU4DFT CUDA Uncompressed EXX (for hybrid ACE speedup & FFT reduction ablation)
        uncomp_ms = None
        uncomp_ffts = None
        if has_cuda_pw and is_hybrid:
            r_unc = gpu4dft.pw.run_scf(
                cell, xc=xc_name, ecut_wfc=ecut, backend="cuda", use_ace=False, conv_tol=1e-9
            )
            uncomp_ms = float(r_unc.elapsed_ms)
            uncomp_ffts = int(r_unc.total_poisson_ffts)

        # 3. Multi-core Host OpenMP reference
        host_res = gpu4dft.pw.run_scf(
            cell, xc=xc_name, ecut_wfc=ecut, backend="host", use_ace=True, conv_tol=1e-9
        )
        host_ms = float(host_res.elapsed_ms)

        # 4. PySCF-PBC exact plane-wave reference (when enabled)
        pyscf_e = None
        pyscf_ms = None
        if run_pyscf_ref:
            pyscf_e, pyscf_ms = _run_pyscf_pbc_pw(a_bohr, coords_bohr, xc_name, ecut, mesh)

        primary_res = gpu_res if gpu_res is not None else host_res
        e_got = float(primary_res.total_energy)
        ref_e = pyscf_e if pyscf_e is not None else float(host_res.total_energy)
        abs_de = abs(e_got - ref_e)

        sp_host = (host_ms / gpu_ms) if (gpu_ms is not None and gpu_ms > 0) else None
        sp_ace = (uncomp_ms / gpu_ms) if (uncomp_ms is not None and gpu_ms is not None and gpu_ms > 0) else None
        fft_red = (
            (uncomp_ffts / max(1, int(primary_res.total_poisson_ffts)))
            if uncomp_ffts is not None
            else None
        )
        sp_pyscf = (pyscf_ms / gpu_ms) if (pyscf_ms is not None and gpu_ms is not None and gpu_ms > 0) else None

        rec = {
            "system": label,
            "natom": len(z_list),
            "xc": xc_name.upper(),
            "ecut_ha": ecut,
            "fft_grid": f"{mesh[0]}x{mesh[1]}x{mesh[2]}",
            "npw": npw,
            "converged": bool(primary_res.converged),
            "scf_iterations": int(primary_res.scf_iterations),
            "poisson_ffts": int(primary_res.total_poisson_ffts),
            "uncompressed_poisson_ffts": uncomp_ffts,
            "fft_reduction_ace": fft_red,
            "energy_hartree": e_got,
            "abs_diff_vs_ref_eh": abs_de,
            "gpu_cuda_ms": gpu_ms,
            "uncompressed_gpu_ms": uncomp_ms,
            "host_omp_ms": host_ms,
            "pyscf_pbc_ms": pyscf_ms,
            "speedup_vs_host_omp": sp_host,
            "speedup_ace_vs_uncompressed": sp_ace,
            "speedup_vs_pyscf_pbc": sp_pyscf,
        }
        rows.append(rec)
        gpu_s = f"{gpu_ms:7.2f} ms" if gpu_ms is not None else "   N/A   "
        host_sp_s = f"{sp_host:5.2f}x" if sp_host is not None else " N/A "
        ace_s = f"{sp_ace:4.2f}x ({fft_red:4.1f}x fewer FFTs)" if sp_ace is not None and fft_red is not None else "—"
        pbc_s = f"{pyscf_ms:7.1f} ms ({sp_pyscf:5.1f}x)" if pyscf_ms is not None and sp_pyscf is not None else "—"
        print(
            f"  {label:20s} [{xc_name.upper():4s}, Ecut={ecut:4.1f} Ha, {rec['fft_grid']}, NPW={npw:4d}]: "
            f"GPU={gpu_s} | Host={host_ms:7.1f} ms ({host_sp_s}) | ACE={ace_s} | PySCF-PBC={pbc_s} | |dE|={abs_de:.2e} Eh"
        )
    return rows


# ---------------------------------------------------------------------------
# Markdown Report Formatter
# ---------------------------------------------------------------------------


def render_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload["metadata"]
    lines = [
        "# GPU4DFT Benchmark Summary Report",
        "",
        "## Hardware & Software Environment",
        f"- **GPU**: `{meta['gpu_name']}` ({meta['vram_total_mib']} MiB VRAM, Driver `{meta['nvidia_driver']}`)",
        f"- **CUDA**: `{meta['cuda_compiler']}`",
        f"- **Host**: `{meta['cpu_count']}` logical CPUs (`{meta['platform']}`)",
        f"- **Software**: Python `{meta['python_version']}`, PySCF `{meta['pyscf_version']}`, "
        f"GPU4PySCF `{meta['gpu4pyscf_version']}` (commit `{meta['git_revision']}`)",
        f"- **Timestamp (UTC)**: `{meta['timestamp_utc']}`",
        "",
        "## 1. Ozaki INT8 Tensor Core vs Native FP64 GEMM",
        "",
        "| Matrix N | FP64 (TFLOPS) | Ozaki INT8 (TFLOPS) | Speedup vs FP64 | FP32 TC (TFLOPS) | Ozaki Rel. Frobenius Error |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in payload.get("gemm_benchmarks", []):
        lines.append(
            f"| {r['n']} | {r['fp64_tflops']:.2f} | **{r['ozaki_int8_tflops']:.2f}** | "
            f"**{r['ozaki_speedup_vs_fp64']:.2f}x** | {r['fp32_tflops']:.2f} | `{r['ozaki_rel_error']:.2e}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Generated ERI Kernel Throughput",
            "",
            "| System | Basis | $N_{\\text{AO}}$ | FP64 (ms) | FP64 (Mq/s) | FP32 (ms) | FP32 (Mq/s) | FP32 Speedup |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in payload.get("eri_throughput", []):
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | {r['fp64_ms']:.2f} | "
            f"{r['fp64_mquartets_per_s']:.2f} | {r['fp32_ms']:.2f} | {r['fp32_mquartets_per_s']:.2f} | "
            f"{r['fp32_speedup']:.2f}x |"
        )

    lines.extend(
        [
            "",
            "## 3. Density Solver & CUDA Graph Ablation",
            "",
            "| System | Basis | $N_{\\text{AO}}$ | Configuration | Iters | $T_{\\text{SCF}}$ (ms) | ms/iter | $|\\Delta E|$ vs Diag FP64 ($E_\\text{h}$) | Graph Replays | Steady-State Allocs |",
            "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in payload.get("solver_ablation", []):
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | {r['config']} | {r['iterations']} | "
            f"{r['scf_time_s']*1e3:.2f} | {r['per_iter_ms']:.2f} | `{r['diff_vs_diag_fp64_eh']:.2e}` | "
            f"{r['cuda_graph_replays']} | {r['steady_state_allocs']} |"
        )

    lines.extend(
        [
            "",
            "## 4. End-to-End SCF Head-to-Head (GPU4DFT vs GPU4PySCF vs CPU PySCF)",
            "",
            "| System | Basis | $N_{\\text{AO}}$ | Method | GPU4DFT Mode | Iters | $|\\Delta E|$ vs PySCF ($E_\\text{h}$) | $T_{\\text{setup}}$ (s) | $T_{\\text{SCF}}$ (s) | $T_{\\text{E2E}}$ (s) | PySCF CPU (s) | GPU4PySCF (s) | Speedup vs CPU | Speedup vs GPU4PySCF (SCF / E2E) |",
            "| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
    )
    for r in payload.get("benchmarks", []):
        de = f"`{r['abs_diff_vs_pyscf_eh']:.2e}`" if r["abs_diff_vs_pyscf_eh"] is not None else "N/A"
        t_cpu = f"{r['pyscf_cpu_time_s']:.3f}" if r["pyscf_cpu_time_s"] is not None else "—"
        t_g4p = f"{r['gpu4pyscf_time_s']:.3f}" if r["gpu4pyscf_time_s"] is not None else "—"
        sp_cpu = f"**{r['speedup_vs_pyscf_cpu']:.1f}x**" if r["speedup_vs_pyscf_cpu"] is not None else "—"
        if r["speedup_e2e_vs_gpu4pyscf"] is not None and r["speedup_scf_loop_vs_gpu4pyscf"] is not None:
            sp_g4p = f"{r['speedup_scf_loop_vs_gpu4pyscf']:.2f}x / {r['speedup_e2e_vs_gpu4pyscf']:.2f}x"
        else:
            sp_g4p = "—"
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | `{r['method']}` | `{r['jk_mode']}` | "
            f"{r['iterations']} | {de} | {r['time_setup_s']:.3f} | {r['time_scf_s']:.3f} | "
            f"{r['time_e2e_s']:.3f} | {t_cpu} | {t_g4p} | {sp_cpu} | {sp_g4p} |"
        )

    lines.extend(
        [
            "",
            "## 5. Analytical Nuclear Gradients",
            "",
            "| System | Basis | $N_{\\text{AO}}$ | Method | $T_{\\text{grad}}$ (s) | $T_{\\text{grad}} / T_{\\text{SCF}}$ | $\\max|\\Delta \\mathbf{g}|$ vs PySCF ($E_\\text{h}/a_0$) | Net Force ($E_\\text{h}/a_0$) | PySCF CPU (s) | GPU4PySCF (s) | Speedup vs CPU |",
            "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in payload.get("gradient_benchmarks", []):
        t_g4p = f"{r['gpu4pyscf_grad_s']:.3f}" if r["gpu4pyscf_grad_s"] is not None else "—"
        lines.append(
            f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | `{r['method']}` | "
            f"{r['time_grad_s']:.3f} | {r['grad_to_scf_ratio']:.2f} | `{r['max_grad_diff_eh_bohr']:.2e}` | "
            f"`{r['net_translation_force_eh_bohr']:.2e}` | {r['pyscf_cpu_grad_s']:.3f} | {t_g4p} | "
            f"**{r['speedup_vs_pyscf_cpu']:.1f}x** |"
        )

    if payload.get("tddft_benchmarks"):
        lines.extend(
            [
                "",
                "## 6. GPU TDDFT & RI-TDDFT Excited States (Lowest 5 Singlets)",
                "",
                "| System | Basis | $N_{\\text{AO}}$ | XC | Mode | Iters (Matvecs) | $E_1$ (eV) | $\\max|\\Delta\\omega|$ ($E_\\text{h}$) | $\\max|\\Delta f_{\\text{osc}}|$ | $T_{\\text{TD}}$ (s) | PySCF CPU $T_{\\text{TD}}$ (s) | GPU4PySCF $T_{\\text{TD}}$ (s) | Speedup vs CPU (TD / E2E) |",
                "| :--- | :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in payload["tddft_benchmarks"]:
            t_g4p = f"{r['gpu4pyscf_td_s']:.3f}" if r["gpu4pyscf_td_s"] is not None else "—"
            lines.append(
                f"| `{r['system']}` | `{r['basis']}` | {r['nao']} | `{r['xc']}` | `{r['mode']}` | "
                f"{r['iterations']} ({r['matvec_count']}) | {r['lowest_excitation_ev']:.3f} | "
                f"`{r['max_abs_omega_diff_eh']:.2e}` | `{r['max_abs_fosc_diff']:.2e}` | "
                f"{r['time_td_s']:.3f} | {r['pyscf_cpu_td_s']:.3f} | {t_g4p} | "
                f"**{r['speedup_td_vs_pyscf_cpu']:.1f}x** / **{r['speedup_e2e_vs_pyscf_cpu']:.1f}x** |"
            )

    if payload.get("pw_dft_benchmarks"):
        lines.extend(
            [
                "",
                "## 7. Plane-Wave Kohn-Sham & Lin's ACE Hybrid DFT",
                "",
                "| System | XC | $E_{\\text{cut}}$ (Ha) | FFT Grid | $N_{\\text{PW}}$ | $|\\Delta E|$ ($E_\\text{h}$) | GPU CUDA (ms) | Host OpenMP (ms) | Uncomp. GPU (ms) | PySCF-PBC (ms) | Speedup vs Host | ACE Speedup (FFT Red.) | Speedup vs PySCF-PBC |",
                "| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for r in payload["pw_dft_benchmarks"]:
            gpu_s = f"**{r['gpu_cuda_ms']:.2f}**" if r["gpu_cuda_ms"] is not None else "—"
            unc_s = f"{r['uncompressed_gpu_ms']:.2f}" if r["uncompressed_gpu_ms"] is not None else "—"
            pbc_s = f"{r['pyscf_pbc_ms']:.1f}" if r["pyscf_pbc_ms"] is not None else "—"
            sp_h = f"**{r['speedup_vs_host_omp']:.2f}x**" if r["speedup_vs_host_omp"] is not None else "—"
            sp_ace = (
                f"**{r['speedup_ace_vs_uncompressed']:.2f}x** ({r['fft_reduction_ace']:.1f}x)"
                if r["speedup_ace_vs_uncompressed"] is not None and r["fft_reduction_ace"] is not None
                else "—"
            )
            sp_pbc = f"**{r['speedup_vs_pyscf_pbc']:.1f}x**" if r["speedup_vs_pyscf_pbc"] is not None else "—"
            lines.append(
                f"| `{r['system']}` | `{r['xc']}` | {r['ecut_ha']:.1f} | `{r['fft_grid']}` | {r['npw']} | "
                f"`{r['abs_diff_vs_ref_eh']:.2e}` | {gpu_s} | {r['host_omp_ms']:.1f} | {unc_s} | {pbc_s} | "
                f"{sp_h} | {sp_ace} | {sp_pbc} |"
            )

    lines.append("")
    return "\n".join(lines)


def warmup_contexts(backend: str) -> None:
    mol_w = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="6-31g*",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    mf_w = gpu4dft.RKS(
        mol_w, xc="b3lyp", backend=backend, precision="mixed", gemm_mode="ozaki_int8"
    )
    mf_w.kernel()
    mf_w.nuc_grad_method().kernel()
    if HAVE_GPU4PYSCF:
        try:
            g4p_w = gpu4pyscf_dft.RKS(mol_w, xc="b3lyp")
            g4p_w.kernel()
        except Exception:
            pass
    sync_gpu()


def _write_benchmark_outputs(payload: dict[str, Any], json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2))
    md_out.write_text(render_markdown_report(payload))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Run GPU4DFT comprehensive benchmark suite.")
    parser.add_argument(
        "--preset",
        choices=("quick", "standard", "full"),
        default="standard",
        help="benchmark matrix size preset (default: standard)",
    )
    parser.add_argument(
        "--backend",
        choices=("cuda", "host"),
        default="cuda" if gpu4dft.backend_available("cuda") else "host",
        help="GPU4DFT execution backend",
    )
    parser.add_argument(
        "--include-slow-cpu",
        action="store_true",
        help="also run Direct 4-center CPU PySCF on large nao > 260 systems",
    )
    parser.add_argument(
        "--suites",
        type=str,
        default="1,2,3,4,5,6,7",
        help="comma-separated list of suite IDs (1..7) to run (default: 1,2,3,4,5,6,7)",
    )
    parser.add_argument(
        "--resume-json",
        type=Path,
        default=None,
        help="optional existing JSON report to preload before running selected --suites",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "benchmark_results.json",
        help="output path for JSON benchmark report",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "BENCHMARK_REPORT.md",
        help="output path for Markdown benchmark summary",
    )
    args = parser.parse_args(argv)

    selected_suites = {int(s.strip()) for s in args.suites.split(",") if s.strip()}

    meta = collect_hardware_metadata()
    payload: dict[str, Any] = {
        "metadata": meta,
        "gemm_benchmarks": [],
        "eri_throughput": [],
        "solver_ablation": [],
        "benchmarks": [],
        "gradient_benchmarks": [],
        "tddft_benchmarks": [],
        "pw_dft_benchmarks": [],
    }
    if args.resume_json is not None and args.resume_json.exists():
        prev = json.loads(args.resume_json.read_text())
        for k in (
            "gemm_benchmarks",
            "eri_throughput",
            "solver_ablation",
            "benchmarks",
            "gradient_benchmarks",
            "tddft_benchmarks",
            "pw_dft_benchmarks",
        ):
            if k in prev:
                payload[k] = prev[k]
        payload["metadata"] = meta

    print(
        f"GPU4DFT Benchmark Suite | Preset={args.preset} | Backend={args.backend} | Suites={sorted(selected_suites)}"
    )
    print(f"Hardware: {meta['gpu_name']} ({meta['vram_total_mib']} MiB) | CPUs={meta['cpu_count']}")
    print("Running CUDA / cuBLAS / cuSOLVER / PTX JIT warmup...")
    warmup_contexts(args.backend)

    if 1 in selected_suites:
        payload["gemm_benchmarks"] = run_gemm_suite(args.backend)
        _write_benchmark_outputs(payload, args.json_out, args.md_out)
    if 2 in selected_suites:
        payload["eri_throughput"] = run_eri_throughput_suite(args.backend)
        _write_benchmark_outputs(payload, args.json_out, args.md_out)
    if 3 in selected_suites:
        payload["solver_ablation"] = run_solver_ablation_suite(args.backend)
        _write_benchmark_outputs(payload, args.json_out, args.md_out)
    if 4 in selected_suites:
        payload["benchmarks"] = run_end_to_end_scf_suite(
            args.backend, args.preset, args.include_slow_cpu
        )
        _write_benchmark_outputs(payload, args.json_out, args.md_out)
    if 5 in selected_suites:
        payload["gradient_benchmarks"] = run_gradient_suite(args.backend, args.preset)
        _write_benchmark_outputs(payload, args.json_out, args.md_out)
    if 6 in selected_suites:
        payload["tddft_benchmarks"] = run_tddft_suite(args.backend, args.preset)
        _write_benchmark_outputs(payload, args.json_out, args.md_out)
    if 7 in selected_suites:
        payload["pw_dft_benchmarks"] = run_pw_dft_suite(args.backend, args.preset)
        _write_benchmark_outputs(payload, args.json_out, args.md_out)

    _write_benchmark_outputs(payload, args.json_out, args.md_out)

    print(f"\nSaved JSON report     -> {args.json_out}")
    print(f"Saved Markdown report -> {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
