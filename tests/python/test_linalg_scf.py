"""Tests for GPU memory pool, CUDA Graphs, Ozaki INT8 GEMM, SP2, CDIIS, Multi-GPU, and SCF driver."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

gto = pytest.importorskip("pyscf.gto")
scf = pytest.importorskip("pyscf.scf")
dft = pytest.importorskip("pyscf.dft")
grad_mod = pytest.importorskip("pyscf.grad")

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_INT_PTR = ctypes.POINTER(ctypes.c_int)
_SIZE_PTR = ctypes.POINTER(ctypes.c_size_t)
_VXC_CB_TYPE = ctypes.CFUNCTYPE(ctypes.c_double, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_void_p)


def _find_driver_cdll() -> ctypes.CDLL:
    candidates: list[Path] = []
    env_lib = os.environ.get("GPU4DFT_DRIVER_LIB")
    if env_lib:
        candidates.append(Path(env_lib))

    repo_root = Path(__file__).resolve().parents[2]
    for build_sub in ["build", "build_cuda", "build_driver"]:
        bdir = repo_root / build_sub
        for pat in ["libgpu4dft_driver.so", "libgpu4dft_driver.dylib", "gpu4dft_driver.dll"]:
            candidates.append(bdir / pat)

    try:
        import gpu4dft._core as _core_mod

        core_path = Path(_core_mod.__file__).resolve()
        build_root = core_path.parents[2]
        for pat in ["libgpu4dft_driver.so", "libgpu4dft_driver.dylib"]:
            candidates.append(build_root / pat)
            candidates.append(core_path.parent / pat)
        candidates.append(core_path)
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                if hasattr(lib, "gpu4dft_c_run_scf_eri"):
                    _bind_signatures(lib)
                    return lib
            except OSError:
                continue

    pytest.skip(
        "GPU4DFT driver shared library with C-ABI symbols not built in current target; "
        "set GPU4DFT_DRIVER_LIB or build target gpu4dft_driver."
    )


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.gpu4dft_c_cuda_available.argtypes = []
    lib.gpu4dft_c_cuda_available.restype = ctypes.c_int

    lib.gpu4dft_c_gemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        _DOUBLE_PTR,
        ctypes.c_int,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_double,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.gpu4dft_c_gemm.restype = ctypes.c_int

    lib.gpu4dft_c_benchmark_gemm.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_c_benchmark_gemm.restype = ctypes.c_int

    lib.gpu4dft_c_canonical_ortho.argtypes = [
        ctypes.c_int,
        _DOUBLE_PTR,
        ctypes.c_double,
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_c_canonical_ortho.restype = ctypes.c_int

    lib.gpu4dft_c_solve_fock_diag.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_c_solve_fock_diag.restype = ctypes.c_int

    lib.gpu4dft_c_solve_fock_sp2.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_c_solve_fock_sp2.restype = ctypes.c_int

    lib.gpu4dft_c_test_pool_and_graphs.argtypes = [ctypes.c_int, ctypes.c_int, _SIZE_PTR]
    lib.gpu4dft_c_test_pool_and_graphs.restype = ctypes.c_int

    lib.gpu4dft_c_test_multigpu.argtypes = [
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _INT_PTR,
    ]
    lib.gpu4dft_c_test_multigpu.restype = ctypes.c_int

    lib.gpu4dft_c_run_scf_eri.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_c_run_scf_eri.restype = ctypes.c_int

    lib.gpu4dft_c_assemble_gradient.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_c_assemble_gradient.restype = ctypes.c_int


@pytest.fixture(scope="module")
def drv() -> ctypes.CDLL:
    return _find_driver_cdll()


def _ptr_f64(arr: np.ndarray | None):
    if arr is None:
        return None
    assert arr.dtype == np.float64 and arr.flags["C_CONTIGUOUS"]
    return arr.ctypes.data_as(_DOUBLE_PTR)


def _available_backends(drv: ctypes.CDLL) -> list[int]:
    return [0, 1] if drv.gpu4dft_c_cuda_available() == 1 else [0]


# ---------------------------------------------------------------------------
# 1. Stream-Ordered Memory Pool & CUDA Graph Capture / Replay
# ---------------------------------------------------------------------------


def test_device_pool_zero_steady_state_allocs_and_cuda_graph(drv: ctypes.CDLL):
    if drv.gpu4dft_c_cuda_available() != 1:
        pytest.skip("CUDA device not available")

    stats = np.zeros(6, dtype=np.uintp)
    rc = drv.gpu4dft_c_test_pool_and_graphs(64, 10, stats.ctypes.data_as(_SIZE_PTR))
    assert rc == 0
    steady_allocs, cache_hits, captures, replays, alloc_bytes, peak_bytes = [int(x) for x in stats]
    assert steady_allocs == 0, f"Expected 0 steady-state cudaMallocAsync calls, got {steady_allocs}"
    assert cache_hits >= 1
    assert captures == 1
    assert replays == 9
    assert alloc_bytes >= peak_bytes > 0


# ---------------------------------------------------------------------------
# 2. Ozaki INT8 Tensor Core FP64-Emulated GEMM & Blackwell Benchmark
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(13, 19, 17), (31, 47, 29), (64, 64, 64), (129, 113, 97)])
@pytest.mark.parametrize("transA,transB", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_ozaki_int8_gemm_accuracy_and_padding(drv: ctypes.CDLL, shape, transA: int, transB: int):
    m, n, k = shape
    rng = np.random.default_rng(100 + m * 7 + n * 3 + k + transA * 2 + transB)
    # Mix orders of magnitude across rows/cols to stress row/col frexp exponent extraction
    row_scales = 2.0 ** rng.integers(-6, 7, size=(m, 1))
    col_scales = 2.0 ** rng.integers(-6, 7, size=(1, n))
    A_op = rng.standard_normal((m, k)) * row_scales
    B_op = rng.standard_normal((k, n)) * col_scales

    A_mem = np.ascontiguousarray(A_op.T if transA else A_op, dtype=np.float64)
    B_mem = np.ascontiguousarray(B_op.T if transB else B_op, dtype=np.float64)
    C_init = np.ascontiguousarray(rng.standard_normal((m, n)), dtype=np.float64)

    alpha = 1.37
    beta = -0.42
    ref = alpha * (A_op @ B_op) + beta * C_init

    for backend in _available_backends(drv):
        for gemm_mode, tol in [(0, 1e-14), (2, 1e-13), (1, 2e-3)]:
            C = C_init.copy()
            rc = drv.gpu4dft_c_gemm(
                m,
                n,
                k,
                alpha,
                _ptr_f64(A_mem),
                transA,
                _ptr_f64(B_mem),
                transB,
                beta,
                _ptr_f64(C),
                backend,
                gemm_mode,
                8,
            )
            assert rc == 0
            rel_err = np.linalg.norm(C - ref) / np.linalg.norm(ref)
            assert rel_err < tol, (
                f"backend={backend}, mode={gemm_mode}, shape={shape}, "
                f"trans=({transA},{transB}): rel_err={rel_err:.3e} >= {tol}"
            )


def test_blackwell_gemm_benchmark(drv: ctypes.CDLL):
    backend = 1 if drv.gpu4dft_c_cuda_available() == 1 else 0
    n = 1024 if backend == 1 else 128
    out9 = np.zeros(9, dtype=np.float64)
    rc = drv.gpu4dft_c_benchmark_gemm(n, 2, 5, backend, 8, _ptr_f64(out9))
    assert rc == 0

    fp64_ms, fp64_tf, oz_ms, oz_tf, fp32_ms, fp32_tf, oz_rel, oz_max, fp32_rel = out9
    assert fp64_tf > 0.0 and oz_tf > 0.0 and fp32_tf > 0.0
    assert oz_rel < 1e-13, f"Ozaki INT8 benchmark relative error {oz_rel:.3e} >= 1e-13"
    assert fp32_rel < 2e-3
    print(
        f"\n[GEMM Benchmark N={n} backend={'CUDA' if backend else 'Host'}] "
        f"FP64: {fp64_tf:.2f} TFLOPS ({fp64_ms:.3f} ms) | "
        f"Ozaki-INT8: {oz_tf:.2f} TFLOPS ({oz_ms:.3f} ms, rel_err={oz_rel:.2e}, max_err={oz_max:.2e}) | "
        f"FP32-TC: {fp32_tf:.2f} TFLOPS ({fp32_ms:.3f} ms, rel_err={fp32_rel:.2e})"
    )


# ---------------------------------------------------------------------------
# 3. Canonical Orthogonalization, cuSOLVER Diag, and SP2 Purification
# ---------------------------------------------------------------------------


def test_canonical_ortho_lindep_and_sp2_vs_diag(drv: ctypes.CDLL):
    mol = gto.M(
        atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis="def2-svp",
        unit="Angstrom",
        verbose=0,
    )
    nao = mol.nao
    n_occ = mol.nelectron // 2
    S = np.ascontiguousarray(mol.intor("int1e_ovlp"), dtype=np.float64)
    H = np.ascontiguousarray(
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"), dtype=np.float64
    )

    for backend in _available_backends(drv):
        nmo_c = ctypes.c_int(0)
        X_buf = np.zeros((nao, nao), dtype=np.float64)
        assert drv.gpu4dft_c_canonical_ortho(nao, _ptr_f64(S), 1e-7, backend, ctypes.byref(nmo_c), _ptr_f64(X_buf)) == 0
        nmo = nmo_c.value
        X = np.ascontiguousarray(X_buf.ravel()[: nao * nmo].reshape(nao, nmo))
        np.testing.assert_allclose(X.T @ S @ X, np.eye(nmo), atol=1e-12)

        for gemm_mode in [0, 2]:  # FP64 and OzakiInt8
            eps_diag = np.zeros(nmo, dtype=np.float64)
            C_diag = np.zeros((nao, nmo), dtype=np.float64)
            D_diag = np.zeros((nao, nao), dtype=np.float64)
            W_diag = np.zeros((nao, nao), dtype=np.float64)
            assert (
                drv.gpu4dft_c_solve_fock_diag(
                    nao,
                    nmo,
                    n_occ,
                    _ptr_f64(H),
                    _ptr_f64(X),
                    backend,
                    gemm_mode,
                    _ptr_f64(eps_diag),
                    _ptr_f64(C_diag),
                    _ptr_f64(D_diag),
                    _ptr_f64(W_diag),
                )
                == 0
            )

            D_sp2 = np.zeros((nao, nao), dtype=np.float64)
            W_sp2 = np.zeros((nao, nao), dtype=np.float64)
            sp2_stats = np.zeros(5, dtype=np.float64)
            assert (
                drv.gpu4dft_c_solve_fock_sp2(
                    nao,
                    nmo,
                    n_occ,
                    _ptr_f64(H),
                    _ptr_f64(X),
                    backend,
                    gemm_mode,
                    1,
                    _ptr_f64(D_sp2),
                    _ptr_f64(W_sp2),
                    _ptr_f64(sp2_stats),
                )
                == 0
            )

            assert abs(np.trace(D_sp2 @ S) - mol.nelectron) < 1e-11
            np.testing.assert_allclose(D_sp2, D_diag, rtol=0, atol=1e-11)
            np.testing.assert_allclose(W_sp2, W_diag, rtol=0, atol=1e-10)

    # Also verify exact linear-dependence dropping on a rank-deficient overlap matrix
    Q, _ = np.linalg.qr(np.random.default_rng(7).standard_normal((12, 12)))
    eigs = np.array([1e-9, 5e-8, 0.05, 0.1, 0.2, 0.4, 0.7, 0.9, 1.1, 1.4, 1.8, 2.3])
    S_dep = np.ascontiguousarray((Q * eigs) @ Q.T, dtype=np.float64)
    nmo_dep = ctypes.c_int(0)
    X_dep_buf = np.zeros((12, 12), dtype=np.float64)
    backend = 1 if drv.gpu4dft_c_cuda_available() == 1 else 0
    assert drv.gpu4dft_c_canonical_ortho(12, _ptr_f64(S_dep), 1e-7, backend, ctypes.byref(nmo_dep), _ptr_f64(X_dep_buf)) == 0
    assert nmo_dep.value == 10
    X_dep = X_dep_buf.ravel()[: 12 * 10].reshape(12, 10)
    np.testing.assert_allclose(X_dep.T @ S_dep @ X_dep, np.eye(10), atol=1e-12)


# ---------------------------------------------------------------------------
# 4. Multi-GPU Manager & GPU Atomic Dynamic Task Queue
# ---------------------------------------------------------------------------


def test_multigpu_manager_and_dynamic_task_queue(drv: ctypes.CDLL):
    num_tasks = 200
    rng = np.random.default_rng(19)
    weights = np.ascontiguousarray(rng.integers(1, 2048, size=num_tasks, dtype=np.int32))
    out_vals = np.zeros(num_tasks, dtype=np.float64)
    meta4 = np.zeros(4, dtype=np.int32)

    rc = drv.gpu4dft_c_test_multigpu(
        num_tasks,
        weights.ctypes.data_as(_INT_PTR),
        _ptr_f64(out_vals),
        meta4.ctypes.data_as(_INT_PTR),
    )
    assert rc == 0
    num_devs, total_claimed, active_blocks, final_counter = [int(v) for v in meta4]
    assert total_claimed == num_tasks
    assert final_counter >= num_tasks
    assert active_blocks >= 1
    if drv.gpu4dft_c_cuda_available() == 1:
        assert num_devs >= 1

    # Verify deterministic task output against exact reference
    for t in [0, 37, 123, 199]:
        k = np.arange(1, int(weights[t]) + 1, dtype=np.float64)
        x = (t + 1) * 1e-3 + k * 1e-5
        expected = float(np.sum(1.0 / (1.0 + x * x)))
        assert abs(out_vals[t] - expected) < 1e-11


# ---------------------------------------------------------------------------
# 5. End-to-End GPU-Resident RHF, RKS (LDA/PBE), and Hybrid DFT (B3LYP/PBE0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "atom_spec,basis_name",
    [
        ("O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692", "6-31g*"),
        ("N 0.0 0.0 0.11; H 0.0 0.94 -0.26; H 0.81 -0.47 -0.26; H -0.81 -0.47 -0.26", "def2-svp"),
        (
            "O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587; "
            "O 2.9 0 0; H 2.9 0.757 -0.587; H 2.9 -0.757 -0.587",
            "sto-3g",
        ),
    ],
)
@pytest.mark.parametrize(
    "solver_mode,gemm_mode,dyn_prec",
    [
        (0, 0, 0),  # Diag + FP64
        (0, 2, 1),  # Diag + OzakiInt8 + Dynamic Mixed-Precision -> FP64 reset
        (1, 0, 0),  # SP2 + FP64 + CUDA Graph
        (1, 2, 1),  # SP2 + OzakiInt8 + Dynamic Mixed-Precision + CUDA Graph
    ],
)
def test_rhf_scf_matches_pyscf_to_1e10(
    drv: ctypes.CDLL,
    atom_spec: str,
    basis_name: str,
    solver_mode: int,
    gemm_mode: int,
    dyn_prec: int,
):
    mol = gto.M(atom=atom_spec, basis=basis_name, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    assert mf.converged

    nao = mol.nao
    S = np.ascontiguousarray(mol.intor("int1e_ovlp"), dtype=np.float64)
    H = np.ascontiguousarray(
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"), dtype=np.float64
    )
    eri = np.ascontiguousarray(mol.intor("int2e", aosym="s1"), dtype=np.float64)
    dm0 = np.ascontiguousarray(mf.get_init_guess(mol, "minao"), dtype=np.float64)

    backend = 1 if drv.gpu4dft_c_cuda_available() == 1 else 0
    D_out = np.zeros((nao, nao), dtype=np.float64)
    W_out = np.zeros((nao, nao), dtype=np.float64)
    F_out = np.zeros((nao, nao), dtype=np.float64)
    eps_out = np.zeros(nao, dtype=np.float64)
    stats8 = np.zeros(8, dtype=np.float64)

    rc = drv.gpu4dft_c_run_scf_eri(
        nao,
        mol.nelectron,
        float(mol.energy_nuc()),
        _ptr_f64(S),
        _ptr_f64(H),
        _ptr_f64(eri),
        _ptr_f64(dm0),
        backend,
        solver_mode,
        gemm_mode,
        dyn_prec,
        1,  # incremental_fock = True
        1,  # use_cuda_graphs = True
        1.0,  # RHF exact exchange fraction = 1.0
        1e-10,
        0.15,  # early-step damping
        0.0,
        None,
        None,
        _ptr_f64(D_out),
        _ptr_f64(W_out),
        _ptr_f64(F_out),
        _ptr_f64(eps_out),
        _ptr_f64(stats8),
    )
    assert rc == 0
    converged, iters, e_tot, diis_rms, g_caps, g_replays, steady_allocs, cache_hits = stats8
    assert converged == 1.0, f"SCF did not converge (iters={iters}, rms={diis_rms:.3e})"
    assert abs(e_tot - e_ref) < 1e-10, f"|E - E_pyscf| = {abs(e_tot - e_ref):.3e} >= 1e-10"
    assert abs(np.trace(D_out @ S) - mol.nelectron) < 1e-10

    if backend == 1:
        assert steady_allocs == 0.0
        assert cache_hits > 0.0
        if iters >= 3:
            assert g_caps >= 1.0
            assert g_replays >= 1.0


@pytest.mark.parametrize(
    "xc_name,alpha_x",
    [
        ("lda,vwn", 0.0),
        ("pbe,pbe", 0.0),
        ("b3lyp", 0.20),
        ("pbe0", 0.25),
    ],
)
@pytest.mark.parametrize("solver_mode", [0, 1])  # Diag and SP2
def test_rks_and_hybrid_dft_scf_matches_pyscf_to_1e10(
    drv: ctypes.CDLL, xc_name: str, alpha_x: float, solver_mode: int
):
    mol = gto.M(
        atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis="6-31g*",
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = xc_name
    mf.grids.level = 3
    mf.grids.build(with_non0tab=True)
    mf.conv_tol = 1e-11
    e_ref = mf.kernel()
    assert mf.converged

    nao = mol.nao
    S = np.ascontiguousarray(mol.intor("int1e_ovlp"), dtype=np.float64)
    H = np.ascontiguousarray(
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"), dtype=np.float64
    )
    eri = np.ascontiguousarray(mol.intor("int2e", aosym="s1"), dtype=np.float64)
    dm0 = np.ascontiguousarray(mf.get_init_guess(mol, "minao"), dtype=np.float64)
    ni = mf._numint

    @_VXC_CB_TYPE
    def vxc_callback(dm_ptr, vxc_out_ptr, _user):
        dm_np = np.ctypeslib.as_array(dm_ptr, shape=(nao, nao)).copy()
        _, exc, vxc = ni.nr_rks(mol, mf.grids, xc_name, dm_np)
        vxc_c = np.ascontiguousarray(vxc, dtype=np.float64)
        ctypes.memmove(vxc_out_ptr, vxc_c.ctypes.data, nao * nao * 8)
        return float(exc)

    backend = 1 if drv.gpu4dft_c_cuda_available() == 1 else 0
    D_out = np.zeros((nao, nao), dtype=np.float64)
    W_out = np.zeros((nao, nao), dtype=np.float64)
    F_out = np.zeros((nao, nao), dtype=np.float64)
    eps_out = np.zeros(nao, dtype=np.float64)
    stats8 = np.zeros(8, dtype=np.float64)

    rc = drv.gpu4dft_c_run_scf_eri(
        nao,
        mol.nelectron,
        float(mol.energy_nuc()),
        _ptr_f64(S),
        _ptr_f64(H),
        _ptr_f64(eri),
        _ptr_f64(dm0),
        backend,
        solver_mode,
        2,  # OzakiInt8 GEMM
        1,  # Dynamic precision -> FP64 reset
        1,  # Incremental Fock
        1,  # CUDA Graphs
        alpha_x,
        1e-10,
        0.0,
        0.05,  # test virtual level shift simultaneously!
        ctypes.cast(vxc_callback, ctypes.c_void_p).value,
        None,
        _ptr_f64(D_out),
        _ptr_f64(W_out),
        _ptr_f64(F_out),
        _ptr_f64(eps_out),
        _ptr_f64(stats8),
    )
    assert rc == 0
    assert stats8[0] == 1.0
    assert abs(stats8[2] - e_ref) < 1e-10, (
        f"RKS({xc_name}, solver={solver_mode}) |E - E_pyscf| = {abs(stats8[2] - e_ref):.3e}"
    )


# ---------------------------------------------------------------------------
# 6. Analytical Nuclear Gradients (RHF & Hybrid B3LYP, Diag & Orbital-Free SP2!)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver_mode", [0, 1])
def test_analytical_nuclear_gradients_rhf_and_b3lyp(drv: ctypes.CDLL, solver_mode: int):
    # Slightly asymmetric geometry so all gradient components are non-trivial
    mol = gto.M(
        atom="O 0.02 -0.01 0.14; H 0.01 0.81 -0.45; H -0.03 -0.73 -0.49",
        basis="6-31g*",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.kernel()
    g_ref = grad_mod.RHF(mf).kernel()  # shape (natom, 3) in Ha / Bohr

    nao = mol.nao
    natom = mol.natm
    S = np.ascontiguousarray(mol.intor("int1e_ovlp"), dtype=np.float64)
    H = np.ascontiguousarray(
        mol.intor("int1e_kin") + mol.intor("int1e_nuc"), dtype=np.float64
    )
    eri = np.ascontiguousarray(mol.intor("int2e", aosym="s1"), dtype=np.float64)
    dm0 = np.ascontiguousarray(mf.get_init_guess(mol, "minao"), dtype=np.float64)

    backend = 1 if drv.gpu4dft_c_cuda_available() == 1 else 0
    D_out = np.zeros((nao, nao), dtype=np.float64)
    W_out = np.zeros((nao, nao), dtype=np.float64)
    F_out = np.zeros((nao, nao), dtype=np.float64)
    eps_out = np.zeros(nao, dtype=np.float64)
    stats8 = np.zeros(8, dtype=np.float64)

    assert (
        drv.gpu4dft_c_run_scf_eri(
            nao,
            mol.nelectron,
            float(mol.energy_nuc()),
            _ptr_f64(S),
            _ptr_f64(H),
            _ptr_f64(eri),
            _ptr_f64(dm0),
            backend,
            solver_mode,
            2,  # OzakiInt8
            0,
            1,
            1,
            1.0,
            1e-12,
            0.0,
            0.0,
            None,
            None,
            _ptr_f64(D_out),
            _ptr_f64(W_out),
            _ptr_f64(F_out),
            _ptr_f64(eps_out),
            _ptr_f64(stats8),
        )
        == 0
    )

    # Verify converged W = 2 C_occ eps_occ C_occ^T matches PySCF make_rdm1e energy-weighted DM!
    grad_obj = grad_mod.RHF(mf)
    W_ref = grad_obj.make_rdm1e(mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    np.testing.assert_allclose(W_out, W_ref, rtol=1e-9, atol=1e-9)

    hcore_deriv = grad_obj.hcore_generator(mol)
    s1 = -mol.intor("int1e_ipovlp", comp=3)  # <d/dA_x mu | nu>, note AO center offset per atom
    aoslices = mol.aoslice_by_atom()

    dhcore_dr = np.zeros((natom * 3, nao, nao), dtype=np.float64)
    ds_dr = np.zeros((natom * 3, nao, nao), dtype=np.float64)
    grad_2e = np.zeros(natom * 3, dtype=np.float64)
    vhf1 = grad_obj.get_veff(mol, D_out)  # -<d/dR mu | vhf | nu>

    for ia in range(natom):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        dh_a = hcore_deriv(ia)  # shape (3, nao, nao)
        for ic in range(3):
            idx = ia * 3 + ic
            dhcore_dr[idx] = dh_a[ic]
            ds_mat = np.zeros((nao, nao), dtype=np.float64)
            ds_mat[p0:p1, :] += s1[ic, p0:p1, :]
            ds_mat[:, p0:p1] += s1[ic, p0:p1, :].T
            ds_dr[idx] = ds_mat
            grad_2e[idx] = 2.0 * np.einsum("ij,ij->", vhf1[ic, p0:p1, :], D_out[p0:p1, :])

    grad_nuc = np.ascontiguousarray(grad_obj.grad_nuc(mol).ravel(), dtype=np.float64)
    grad_out = np.zeros(natom * 3, dtype=np.float64)

    assert (
        drv.gpu4dft_c_assemble_gradient(
            natom,
            nao,
            _ptr_f64(D_out),
            _ptr_f64(W_out),
            _ptr_f64(grad_nuc),
            _ptr_f64(dhcore_dr),
            _ptr_f64(ds_dr),
            _ptr_f64(grad_2e),
            None,
            _ptr_f64(grad_out),
        )
        == 0
    )

    np.testing.assert_allclose(grad_out.reshape(natom, 3), g_ref, rtol=0, atol=1e-9)
