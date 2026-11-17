"""End-to-end integration tests for GPU4DFT's C++20/CUDA RHF & RKS workflows,
nanobind Python API, RI-JK, sn-K (COSX), dynamic mixed precision, Ozaki INT8
Tensor Core GEMM, SP2 purification, analytical nuclear gradients (vs PySCF and
central finite differences), ASE calculator, and Host/CUDA/HIP codegen.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpu4dft
from gpu4dft_codegen.__main__ import main as eri_codegen_main
from xc_codegen.__main__ import main as xc_codegen_main

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, scf  # noqa: E402
from pyscf.grad import rhf as pyscf_grad_rhf  # noqa: E402
from pyscf.grad import rks as pyscf_grad_rks  # noqa: E402


def _available_backends() -> list[str]:
    out = ["host"]
    if gpu4dft.backend_available("cuda"):
        out.append("cuda")
    return out


BACKENDS = _available_backends()


def _h2o_mol(basis: str = "def2-svp", distorted: bool = False):
    if distorted:
        atom = "O 0.02 -0.01 0.14; H 0.05 0.81 -0.44; H -0.04 -0.72 -0.51"
    else:
        atom = "O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692"
    return gto.M(atom=atom, basis=basis, unit="Angstrom", cart=False, verbose=0)


# ---------------------------------------------------------------------------
# 1. End-to-End RHF (Direct, RI-JK, sn-K, RIJCOSX, minao vs hcore)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("basis_name", ["6-31g*", "def2-tzvp"])
def test_rhf_direct_matches_pyscf(backend: str, basis_name: str):
    mol = _h2o_mol(basis=basis_name)
    mf_ref = scf.RHF(mol)
    mf_ref.conv_tol = 1e-12
    e_ref = mf_ref.kernel()

    mf = gpu4dft.RHF(mol, backend=backend, conv_tol=1e-10)
    e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - e_ref) <= 1e-10
    if backend == "cuda":
        assert mf.result.scf.steady_state_allocations == 0


def test_rhf_minao_guess_reduces_iterations():
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = gto.M(
        atom=(
            "O 0.000 0.000 0.000; H 0.000 -0.757 0.587; H 0.000 0.757 0.587; "
            "O 2.900 0.000 0.000; H 2.900 0.757 -0.587; H 2.900 -0.757 -0.587"
        ),
        basis="def2-svp",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    mf_ref = scf.RHF(mol)
    mf_ref.conv_tol = 1e-12
    e_ref = mf_ref.kernel()

    mf_hcore = gpu4dft.RHF(mol, backend=backend, init_guess="hcore", conv_tol=1e-10).run()
    mf_minao = gpu4dft.RHF(mol, backend=backend, init_guess="minao", conv_tol=1e-10).run()

    assert mf_hcore.converged and mf_minao.converged
    assert abs(mf_hcore.e_tot - e_ref) <= 1e-10
    assert abs(mf_minao.e_tot - e_ref) <= 1e-10
    assert mf_minao.iterations < mf_hcore.iterations


@pytest.mark.parametrize("backend", BACKENDS)
def test_rhf_ri_jk_matches_pyscf_df(backend: str):
    mol = _h2o_mol(basis="def2-svp")
    mf_ref = scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit")
    mf_ref.conv_tol = 1e-12
    e_ref = mf_ref.kernel()

    mf = gpu4dft.RHF(mol, backend=backend, conv_tol=1e-10).density_fit(
        auxbasis="def2-universal-jkfit"
    )
    e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - e_ref) <= 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("with_ri_j", [False, True])
def test_rhf_snk_and_rijcosx(backend: str, with_ri_j: bool):
    mol = _h2o_mol(basis="def2-svp")
    mf_ref = scf.RHF(mol)
    mf_ref.conv_tol = 1e-12
    e_ref = mf_ref.kernel()

    mf = gpu4dft.RHF(mol, backend=backend, grids_level=4, conv_tol=1e-9)
    if with_ri_j:
        mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.snk()
    e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - e_ref) <= 2e-4


# ---------------------------------------------------------------------------
# 2. End-to-End RKS & Hybrid DFT across LDA / GGA / Hybrid / Meta-GGA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("xc_name", ["svwn", "pbe", "blyp", "b3lyp", "pbe0", "tpss"])
def test_rks_all_functionals_exact_grid_match_pyscf(backend: str, xc_name: str):
    mol = _h2o_mol(basis="def2-svp")
    mf_ref = dft.RKS(mol, xc=xc_name)
    mf_ref.grids.level = 3
    mf_ref.grids.build()
    mf_ref.conv_tol = 1e-12
    e_ref = mf_ref.kernel()

    mf = gpu4dft.RKS(
        mol,
        xc=xc_name,
        backend=backend,
        grids=(mf_ref.grids.coords, mf_ref.grids.weights),
        conv_tol=1e-10,
    )
    e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - e_ref) <= 1e-10


@pytest.mark.parametrize("xc_name", ["svwn", "pbe", "b3lyp", "tpss"])
def test_rks_native_gpu_grid_and_octree_screening(xc_name: str):
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _h2o_mol(basis="def2-svp")

    mf_ref = dft.RKS(mol, xc=xc_name)
    mf_ref.grids.level = 4
    mf_ref.conv_tol = 1e-11
    e_ref = mf_ref.kernel()

    # Uses GPU4DFT's native M4 + Lebedev + GPU Stratmann SSF + 3D octree grid (no PySCF grid!)
    mf = gpu4dft.RKS(mol, xc=xc_name, backend=backend, grids_level=4, conv_tol=1e-10)
    e_got = mf.kernel()

    assert mf.converged
    assert mf.result.grid_npoints > 1000
    assert mf.result.grid_nbatches >= 1
    assert 0.0 < mf.result.active_ao_ratio <= 1.0
    assert abs(e_got - e_ref) <= 1.5e-4


@pytest.mark.parametrize("xc_name", ["pbe", "b3lyp", "pbe0", "tpss"])
def test_rks_ri_jk_and_snk(xc_name: str):
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _h2o_mol(basis="def2-svp")

    mf_df_ref = dft.RKS(mol, xc=xc_name).density_fit(auxbasis="def2-universal-jkfit")
    mf_df_ref.grids.level = 3
    mf_df_ref.grids.build()
    mf_df_ref.conv_tol = 1e-12
    e_df_ref = mf_df_ref.kernel()

    mf_df = gpu4dft.RKS(
        mol,
        xc=xc_name,
        backend=backend,
        grids=mf_df_ref.grids,
        conv_tol=1e-10,
    ).density_fit(auxbasis="def2-universal-jkfit")
    e_df_got = mf_df.kernel()

    assert mf_df.converged
    assert abs(e_df_got - e_df_ref) <= 1e-10

    if xc_name in ("b3lyp", "pbe0"):
        mf_snk = (
            gpu4dft.RKS(mol, xc=xc_name, backend=backend, grids=mf_df_ref.grids, conv_tol=1e-9)
            .snk()
            .run()
        )
        assert mf_snk.converged
        assert abs(mf_snk.e_tot - e_df_ref) <= 5e-4


# ---------------------------------------------------------------------------
# 3. Dynamic Mixed Precision + Ozaki INT8 Tensor Core GEMM + SP2 Purification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("solver", ["diag", "sp2"])
@pytest.mark.parametrize("with_df", [False, True])
def test_mixed_precision_ozaki_int8_and_sp2_accuracy(backend: str, solver: str, with_df: bool):
    mol = _h2o_mol(basis="def2-svp")
    if with_df:
        mf_ref = dft.RKS(mol, xc="b3lyp").density_fit(auxbasis="def2-universal-jkfit")
    else:
        mf_ref = dft.RKS(mol, xc="b3lyp")
    mf_ref.grids.level = 3
    mf_ref.grids.build()
    mf_ref.conv_tol = 1e-12
    e_ref = mf_ref.kernel()

    mf = gpu4dft.RKS(
        mol,
        xc="b3lyp",
        backend=backend,
        precision="mixed",
        dynamic_precision=True,
        gemm_mode="ozaki_int8",
        solver=solver,
        incremental=True,
        use_cuda_graph=True,
        grids=mf_ref.grids,
        conv_tol=1e-10,
    )
    if with_df:
        mf.density_fit(auxbasis="def2-universal-jkfit")
    e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - e_ref) <= 1e-10


def test_gemm_and_benchmark_gemm_bindings():
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    rng = np.random.default_rng(42)
    a = rng.standard_normal((73, 59))
    b = rng.standard_normal((59, 67))
    c_ref = a @ b
    c_oz = gpu4dft.gemm(a, b, backend=backend, mode="ozaki_int8")
    np.testing.assert_allclose(c_oz, c_ref, rtol=1e-13, atol=1e-13)

    bench = gpu4dft.benchmark_gemm(128, warmup_iters=1, bench_iters=2, backend=backend)
    assert bench.n == 128
    assert bench.ozaki_rel_error < 1e-13
    assert bench.fp64_tflops > 0.0
    assert bench.ozaki_tflops > 0.0


# ---------------------------------------------------------------------------
# 4. Analytical Nuclear Gradients vs PySCF & Central Finite Differences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["diag", "sp2"])
@pytest.mark.parametrize("method", ["rhf", "svwn", "pbe", "blyp", "b3lyp", "pbe0", "tpss"])
def test_analytical_nuclear_gradients_vs_pyscf(method: str, solver: str):
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _h2o_mol(basis="def2-svp", distorted=True)

    if method == "rhf":
        mf_ref = scf.RHF(mol)
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        g_ref = pyscf_grad_rhf.Gradients(mf_ref).kernel()

        mf = gpu4dft.RHF(mol, backend=backend, solver=solver, conv_tol=1e-12, diis_tol=1e-9).run()
    else:
        mf_ref = dft.RKS(mol, xc=method)
        mf_ref.grids.level = 3
        mf_ref.grids.build()
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        g_ref = pyscf_grad_rks.Gradients(mf_ref).kernel()

        mf = gpu4dft.RKS(
            mol,
            xc=method,
            backend=backend,
            solver=solver,
            grids=mf_ref.grids,
            conv_tol=1e-12,
            diis_tol=1e-9,
        ).run()

    g_got = mf.nuc_grad_method().kernel()
    assert g_got.shape == (mol.natm, 3)
    np.testing.assert_allclose(g_got, g_ref, rtol=0.0, atol=1e-8)
    # Exact translational invariance (< 1e-11) holds for grid-free RHF; for finite-grid RKS
    # (grid_response=False in both PySCF and GPU4DFT), net force matches PySCF to < 1e-9 and is O(1e-5).
    if method == "rhf":
        np.testing.assert_allclose(np.sum(g_got, axis=0), np.zeros(3), rtol=0.0, atol=1e-11)
    else:
        np.testing.assert_allclose(np.sum(g_got, axis=0), np.sum(g_ref, axis=0), rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(np.sum(g_got, axis=0), np.zeros(3), rtol=0.0, atol=5e-5)


def test_nuclear_gradients_match_central_finite_differences():
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol0 = _h2o_mol(basis="6-31g*", distorted=True)
    coords0 = np.asarray(mol0.atom_coords(unit="Bohr"), dtype=np.float64)
    charges = [int(mol0.atom_charge(i)) for i in range(mol0.natm)]
    h_bohr = 1e-4

    # 1. RHF analytical gradient vs moving-atom central finite differences
    mf_rhf = gpu4dft.RHF(mol0, backend=backend, conv_tol=1e-12).run()
    g_rhf_ana = mf_rhf.nuc_grad_method().kernel()

    g_rhf_fd = np.zeros_like(g_rhf_ana)
    for ia in range(mol0.natm):
        for ix in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, ix] += h_bohr
            cm[ia, ix] -= h_bohr
            mol_p = gto.M(
                atom=[(charges[k], tuple(cp[k])) for k in range(mol0.natm)],
                basis="6-31g*",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            mol_m = gto.M(
                atom=[(charges[k], tuple(cm[k])) for k in range(mol0.natm)],
                basis="6-31g*",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            ep = gpu4dft.RHF(mol_p, backend=backend, conv_tol=1e-12).kernel()
            em = gpu4dft.RHF(mol_m, backend=backend, conv_tol=1e-12).kernel()
            g_rhf_fd[ia, ix] = (ep - em) / (2.0 * h_bohr)

    np.testing.assert_allclose(g_rhf_ana, g_rhf_fd, rtol=0.0, atol=1e-7)

    # 2. Hybrid RKS (B3LYP) analytical gradient vs fixed-quadrature central finite differences
    #    (eliminates Becke grid-weight nuclear response, matching PySCF grid_response=False to < 1e-7)
    ref_grids = dft.Grids(mol0)
    ref_grids.level = 3
    ref_grids.build()
    fixed_grids = (ref_grids.coords.copy(), ref_grids.weights.copy())

    mf_ks = gpu4dft.RKS(mol0, xc="b3lyp", backend=backend, grids=fixed_grids, conv_tol=1e-12).run()
    g_ks_ana = mf_ks.nuc_grad_method().kernel()

    g_ks_fd = np.zeros_like(g_ks_ana)
    for ia in range(mol0.natm):
        for ix in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, ix] += h_bohr
            cm[ia, ix] -= h_bohr
            mol_p = gto.M(
                atom=[(charges[k], tuple(cp[k])) for k in range(mol0.natm)],
                basis="6-31g*",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            mol_m = gto.M(
                atom=[(charges[k], tuple(cm[k])) for k in range(mol0.natm)],
                basis="6-31g*",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            ep = gpu4dft.RKS(
                mol_p, xc="b3lyp", backend=backend, grids=fixed_grids, conv_tol=1e-12
            ).kernel()
            em = gpu4dft.RKS(
                mol_m, xc="b3lyp", backend=backend, grids=fixed_grids, conv_tol=1e-12
            ).kernel()
            g_ks_fd[ia, ix] = (ep - em) / (2.0 * h_bohr)

    np.testing.assert_allclose(g_ks_ana, g_ks_fd, rtol=0.0, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. ASE Calculator Interface (with and without `ase` installed)
# ---------------------------------------------------------------------------


def test_ase_calculator_standalone_and_forces():
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    pos0 = np.array(
        [[0.02, -0.01, 0.14], [0.05, 0.81, -0.44], [-0.04, -0.72, -0.51]], dtype=np.float64
    )
    atoms = gpu4dft.SimpleAtoms([8, 1, 1], pos0)
    calc = gpu4dft.GPU4DFTCalculator(xc="hf", basis="sto-3g", backend=backend, conv_tol=1e-11)
    atoms.calc = calc

    e_ev = atoms.get_potential_energy()
    forces = atoms.get_forces()
    assert forces.shape == (3, 3)

    # Verify F_z on oxygen against central finite difference in eV / Angstrom
    delta_ang = 1e-4
    pos_p = pos0.copy()
    pos_m = pos0.copy()
    pos_p[0, 2] += delta_ang
    pos_m[0, 2] -= delta_ang
    ep = calc.get_potential_energy(gpu4dft.SimpleAtoms([8, 1, 1], pos_p))
    em = calc.get_potential_energy(gpu4dft.SimpleAtoms([8, 1, 1], pos_m))
    fz_fd = -(ep - em) / (2.0 * delta_ang)
    assert abs(forces[0, 2] - fz_fd) <= 1e-4
    assert e_ev < -2000.0  # ~ -74.96 Eh * 27.211 eV/Eh

    # Also test real ase.Atoms if ASE is installed in the environment
    try:
        import ase
    except ImportError:
        return
    ase_atoms = ase.Atoms("OH2", positions=pos0)
    ase_atoms.calc = gpu4dft.GPU4DFTCalculator(
        xc="b3lyp", basis="def2-svp", backend=backend, conv_tol=1e-10
    )
    e_ase = ase_atoms.get_potential_energy()
    f_ase = ase_atoms.get_forces()
    assert np.isfinite(e_ase) and f_ase.shape == (3, 3)


# ---------------------------------------------------------------------------
# 6. Multi-Backend Code Generation (`--backends host,cuda,hip`)
# ---------------------------------------------------------------------------


def test_host_cuda_hip_codegen_cli(tmp_path):
    assert eri_codegen_main(["--backends", "host,cuda,hip", "--out-dir", str(tmp_path)]) == 0
    assert xc_codegen_main(["--backends", "host,cuda,hip", "--out-dir", str(tmp_path)]) == 0

    assert (tmp_path / "include/gpu4dft/generated/eri_launchers.hpp").is_file()
    assert (tmp_path / "include/gpu4dft/generated/eri_deriv1.hpp").is_file()
    assert (tmp_path / "include/gpu4dft/generated/xc_kernels.hpp").is_file()
    assert (tmp_path / "src/eri_ssss_host.cpp").is_file()
    assert (tmp_path / "src/eri_ssss_cuda.cu").is_file()
    assert (tmp_path / "src/eri_ssss_hip.hip").is_file()
    assert (tmp_path / "src/eri_general_hip.hip").is_file()


# ---------------------------------------------------------------------------
# 7. Permanent Dipole Moment & End-to-End CIS / TDA / TDHF / TDDFT (Direct & RI)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_dipole_moment_and_dipole_matrices_vs_pyscf(backend: str):
    mol = _h2o_mol(basis="def2-svp")
    mf_ref = dft.RKS(mol, xc="b3lyp")
    mf_ref.grids.level = 3
    mf_ref.grids.build()
    mf_ref.conv_tol = 1e-12
    mf_ref.kernel()
    dip_debye_ref = mf_ref.dip_moment(unit="Debye", verbose=0)
    dip_au_ref = mf_ref.dip_moment(unit="AU", verbose=0)

    basis = gpu4dft.basis_from_pyscf(mol)
    dip_mats = gpu4dft.dipole_matrices(basis, origin=(0.0, 0.0, 0.0), backend=backend)
    with mol.with_common_orig((0.0, 0.0, 0.0)):
        np.testing.assert_allclose(dip_mats, -mol.intor("int1e_r"), rtol=0.0, atol=1e-12)

    mf = gpu4dft.RKS(
        mol, xc="b3lyp", backend=backend, grids=mf_ref.grids, conv_tol=1e-11, diis_tol=1e-8
    ).run()
    np.testing.assert_allclose(mf.dipole_moment(unit="Debye"), dip_debye_ref, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(mf.dipole_moment(unit="AU"), dip_au_ref, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("with_df", [False, True])
@pytest.mark.parametrize("resp_kind", ["tda", "tddft"])
@pytest.mark.parametrize("xc_name", ["rhf", "pbe", "b3lyp"])
def test_end_to_end_tda_and_tddft_direct_and_ri_vs_pyscf(
    xc_name: str, resp_kind: str, with_df: bool
):
    from pyscf import tdscf

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = gto.M(
        atom="C 0.0 0.0 0.0; O 0.0 0.0 1.203; H 0.0 0.942 -0.587; H 0.0 -0.942 -0.587",
        basis="def2-svp",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )

    if xc_name == "rhf":
        mf_ref = scf.RHF(mol)
        if with_df:
            mf_ref = mf_ref.density_fit(auxbasis="def2-universal-jkfit")
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        mf = gpu4dft.RHF(mol, backend=backend, conv_tol=1e-11, diis_tol=1e-8)
    else:
        mf_ref = dft.RKS(mol, xc=xc_name)
        mf_ref.grids.level = 3
        mf_ref.grids.build()
        if with_df:
            mf_ref = mf_ref.density_fit(auxbasis="def2-universal-jkfit")
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        mf = gpu4dft.RKS(
            mol, xc=xc_name, backend=backend, grids=mf_ref.grids, conv_tol=1e-11, diis_tol=1e-8
        )

    if with_df:
        mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.run()
    assert abs(mf.e_tot - mf_ref.e_tot) <= 1e-10

    td_ref = tdscf.TDA(mf_ref) if resp_kind == "tda" else tdscf.TDDFT(mf_ref)
    td_ref.nstates = 5
    td_ref.conv_tol = 1e-10
    td_ref.kernel()
    f_ref = td_ref.oscillator_strength()

    if xc_name == "rhf":
        td = mf.CIS(nstates=5, conv_tol=1e-8) if resp_kind == "tda" else mf.TDHF(nstates=5, conv_tol=1e-8)
    else:
        td = mf.TDA(nstates=5, conv_tol=1e-8) if resp_kind == "tda" else mf.TDDFT(nstates=5, conv_tol=1e-8)
    e_got, xy_got = td.kernel()

    assert td.converged
    assert len(xy_got) == 5
    np.testing.assert_allclose(e_got, td_ref.e, rtol=0.0, atol=5e-7)
    np.testing.assert_allclose(td.oscillator_strength(), f_ref, rtol=0.0, atol=5e-6)


@pytest.mark.parametrize("with_df", [False, True])
def test_end_to_end_tddft_ozaki_int8_and_sp2_scf(with_df: bool):
    from pyscf import tdscf

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _h2o_mol(basis="def2-svp")

    mf_ref = dft.RKS(mol, xc="b3lyp")
    mf_ref.grids.level = 3
    mf_ref.grids.build()
    if with_df:
        mf_ref = mf_ref.density_fit(auxbasis="def2-universal-jkfit")
    mf_ref.conv_tol = 1e-12
    mf_ref.kernel()
    td_ref = tdscf.TDDFT(mf_ref)
    td_ref.nstates = 4
    td_ref.conv_tol = 1e-10
    td_ref.kernel()

    mf = gpu4dft.RKS(
        mol,
        xc="b3lyp",
        backend=backend,
        solver="sp2",
        gemm_mode="ozaki_int8",
        grids=mf_ref.grids,
        conv_tol=1e-11,
        diis_tol=1e-8,
    )
    if with_df:
        mf.density_fit(auxbasis="def2-universal-jkfit")
    mf.run()

    td = mf.TDDFT(nstates=4, gemm_mode="ozaki_int8", conv_tol=1e-8).run()
    assert td.converged
    np.testing.assert_allclose(td.e, td_ref.e, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(td.oscillator_strength(), td_ref.oscillator_strength(), rtol=0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 8. End-to-End Molecular Nuclear Hessian & Harmonic Vibrational Frequencies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["rhf", "b3lyp"])
def test_end_to_end_hessian_and_frequencies_vs_pyscf(method: str):
    from pyscf.hessian import thermo as pyscf_thermo

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _h2o_mol(basis="6-31g*")

    if method == "rhf":
        mf_ref = scf.RHF(mol)
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        h_ref = mf_ref.Hessian().kernel()
        mf = gpu4dft.RHF(mol, backend=backend, conv_tol=1e-12, diis_tol=1e-9).run()
    else:
        mf_ref = dft.RKS(mol, xc=method)
        mf_ref.grids.level = 4
        mf_ref.grids.build()
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        hobj_ref = mf_ref.Hessian()
        hobj_ref.grid_response = False
        h_ref = hobj_ref.kernel()
        mf = gpu4dft.RKS(
            mol, xc=method, backend=backend, grids=mf_ref.grids, conv_tol=1e-12, diis_tol=1e-9
        ).run()

    harm_ref = pyscf_thermo.harmonic_analysis(mol, h_ref)
    freqs_ref = np.asarray(np.real(harm_ref["freq_wavenumber"]), dtype=np.float64)

    hess_drv = mf.Hessian()
    h_got = hess_drv.kernel()
    harm_got = hess_drv.harmonic_analysis()

    assert h_got.shape == (mol.natm, mol.natm, 3, 3)
    np.testing.assert_allclose(h_got, h_ref, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(harm_got["freq_wavenumber"], freqs_ref, rtol=0.0, atol=0.5)
    assert harm_got["zpe_hartree"] > 0.01


# ---------------------------------------------------------------------------
# 9. End-to-End Plane-Wave Kohn-Sham & ACE Hybrid DFT (`gpu4dft.pw`)
# ---------------------------------------------------------------------------


def test_end_to_end_pw_submodule_rks_krks_and_ace_pbe0():
    from pyscf.pbc import gto as pbc_gto

    backend = "cuda" if gpu4dft.pw.cuda_pw_available() else "host"
    a_b = 5.43 * 1.8897259885789233
    a_bohr = np.array(
        [[0.0, 0.5 * a_b, 0.5 * a_b], [0.5 * a_b, 0.0, 0.5 * a_b], [0.5 * a_b, 0.5 * a_b, 0.0]],
        dtype=np.float64,
    )
    coords_bohr = np.array(
        [[0.0, 0.0, 0.0], [0.25 * a_b, 0.25 * a_b, 0.25 * a_b]], dtype=np.float64
    )

    pcell = pbc_gto.M(
        a=a_bohr,
        atom=[("Si", coords_bohr[0].tolist()), ("Si", coords_bohr[1].tolist())],
        unit="B",
        pseudo="gth-pbe",
        verbose=0,
    )
    cell = gpu4dft.pw.cell_from_pyscf(pcell)
    assert abs(cell.ewald_energy() - float(pcell.ewald())) < 1e-11

    # Batched 3D FFT check via gpu4dft.pw.fft3d_batch
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((3, 12, 12, 12)) + 1j * rng.standard_normal((3, 12, 12, 12))
    fwd = gpu4dft.pw.fft3d_batch(arr, direction=-1, backend=backend)
    np.testing.assert_allclose(fwd, np.fft.fftn(arr, axes=(1, 2, 3)), rtol=0.0, atol=1e-12)

    # Gamma-point PBE0 ACE vs Uncompressed EXX via gpu4dft.pw.RKS
    mf_ace = gpu4dft.pw.RKS(
        cell, xc="pbe0", ecut_wfc=8.0, backend=backend, use_ace=True, conv_tol=1e-9
    ).run()
    mf_exx = gpu4dft.pw.RKS(
        cell, xc="pbe0", ecut_wfc=8.0, backend=backend, use_ace=False, conv_tol=1e-9
    ).run()
    assert mf_ace.converged and mf_exx.converged
    assert abs(mf_ace.e_tot - mf_exx.e_tot) < 1e-8
    assert mf_ace.result.total_poisson_ffts * 4 < mf_exx.result.total_poisson_ffts

    # 2x2x2 Monkhorst-Pack k-point PBE via gpu4dft.pw.KRKS
    kmf = gpu4dft.pw.KRKS(cell, xc="pbe", ecut_wfc=8.0, kmesh=(2, 2, 2), backend=backend).run()
    assert kmf.converged
    assert len(kmf.mo_energy) == 8

