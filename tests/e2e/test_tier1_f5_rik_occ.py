"""Tier 1 E2E tests for Feature F5: Memory-Bounded Occupied-Space RI-K (T1-F5-01..05)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import track_peak_vram_mib
from data.molecules import get_molecule, get_water_cluster_coords
from oracles import run_pyscf_scf


@pytest.mark.tier1
@pytest.mark.f5
@pytest.mark.gpu
def test_f5_ri_k_closed_shell_accuracy(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F5-01: Closed-shell RHF/RKS occupied-space RI-K SCF matches PySCF DF within 1e-6 Eh."""
    from gpu4dft import RHF, RKS

    auxbasis = "def2-universal-jkfit"
    cases = [
        ("c6h6", "def2-svp", "rhf", None),
        ("c6h6", "def2-svp", "rks", "b3lyp"),
        ("h2o_dimer", "def2-tzvp", "rks", "pbe0"),
    ]
    for mol_name, basis, method, xc in cases:
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis=basis, charge=spec.charge, spin=spec.spin)
        ref = run_pyscf_scf(
            mol,
            method,
            xc=xc,
            density_fit=True,
            auxbasis=auxbasis,
            grids_level=3,
            conv_tol=1e-11,
        )
        assert ref["converged"]

        if method == "rhf":
            mf = RHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
                auxbasis=auxbasis
            )
        else:
            mf = RKS(
                mol, xc=xc or "b3lyp", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf, ref["mf"])

        e_got = mf.kernel()
        assert mf.converged
        assert abs(e_got - ref["e_tot"]) <= 1e-6


@pytest.mark.tier1
@pytest.mark.f5
@pytest.mark.gpu
def test_f5_ri_k_open_shell_accuracy(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F5-02: Open-shell UHF/UKS RI-K matches PySCF DF energy (<=1e-6 Eh) and <S^2> (<=1e-5)."""
    from gpu4dft import UHF, UKS

    auxbasis = "def2-universal-jkfit"
    cases = [
        ("ch3_radical", "uhf", None),
        ("o2_triplet", "uhf", None),
        ("ch3_radical", "uks", "b3lyp"),
        ("o2_triplet", "uks", "b3lyp"),
    ]
    for mol_name, method, xc in cases:
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
        ref = run_pyscf_scf(
            mol,
            method,
            xc=xc,
            density_fit=True,
            auxbasis=auxbasis,
            grids_level=3,
            conv_tol=1e-11,
        )
        assert ref["converged"]
        assert ref["spin_square"] is not None

        if method == "uhf":
            mf = UHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
                auxbasis=auxbasis
            )
        else:
            mf = UKS(
                mol, xc=xc or "b3lyp", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf, ref["mf"])

        e_got = mf.kernel()
        assert mf.converged
        assert abs(e_got - ref["e_tot"]) <= 1e-6

        ss_got, _ = mf.spin_square()
        assert abs(ss_got - ref["spin_square"][0]) <= 1e-5
        dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
        assert dm.shape == (2, mol.nao_nr(), mol.nao_nr())


@pytest.mark.tier1
@pytest.mark.f5
@pytest.mark.gpu
def test_f5_ri_k_rsh_accuracy(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F5-03: RSH (CAM-B3LYP, wB97X) RKS/UKS RI-K matches PySCF DF within 1e-6 Eh."""
    from gpu4dft import RKS, UKS

    auxbasis = "def2-universal-jkfit"
    for xc in ("cam-b3lyp", "wb97x"):
        for mol_name, method in (("h2o", "rks"), ("ho2_radical", "uks")):
            spec = get_molecule(mol_name)
            mol = make_pyscf_mol(
                spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin
            )
            ref = run_pyscf_scf(
                mol,
                method,
                xc=xc,
                density_fit=True,
                auxbasis=auxbasis,
                grids_level=3,
                conv_tol=1e-11,
            )
            assert ref["converged"]

            cls = RKS if method == "rks" else UKS
            mf = cls(
                mol, xc=xc, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf, ref["mf"])

            e_got = mf.kernel()
            assert mf.converged
            assert abs(e_got - ref["e_tot"]) <= 1e-6
            if method == "uks":
                ss_got, _ = mf.spin_square()
                assert ref["spin_square"] is not None
                assert abs(ss_got - ref["spin_square"][0]) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f5
@pytest.mark.gpu
@pytest.mark.slow
def test_f5_ri_k_memory_bounded_large_nao(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F5-04: Large-NAO RI-K (C60/def2-svp, NAO=840) runs memory-bounded without OOM."""
    from pyscf import df
    from gpu4dft import RHF

    spec = get_molecule("c60")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp")
    nao = mol.nao_nr()
    assert nao == 840

    auxbasis = "def2-universal-jkfit"
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    naux = auxmol.nao_nr()
    dense_b_mib = (naux * nao * nao * 8) / (1024.0 * 1024.0)

    ref = run_pyscf_scf(
        mol,
        "rhf",
        density_fit=True,
        auxbasis=auxbasis,
        conv_tol=1e-10,
    )
    assert ref["converged"]

    with track_peak_vram_mib() as vram:
        mf = RHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-9).density_fit(
            auxbasis=auxbasis
        )
        e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - ref["e_tot"]) <= 1e-6
    if gpu4dft_backend == "cuda" and vram["total_mib"] > 0.0:
        assert vram["used_after_mib"] < vram["total_mib"]
        assert vram["delta_mib"] < dense_b_mib


@pytest.mark.tier1
@pytest.mark.f5
@pytest.mark.gpu
@pytest.mark.slow
def test_f5_ri_k_occupied_space_scaling(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F5-05: Occupied-space RI-K exhibits O(N_aux * N_AO^2 * N_occ) scaling across (H2O)_N."""
    from pyscf import df
    from gpu4dft import RHF

    auxbasis = "def2-universal-jkfit"

    # Warmup on (H2O)_2 so JIT / CUDA context initialization does not skew scaling timings
    mol_warm = make_pyscf_mol(get_water_cluster_coords(2, 1, 1), basis="def2-svp")
    RHF(mol_warm, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-9).density_fit(
        auxbasis=auxbasis
    ).kernel()

    dims = [(2, 1, 2), (2, 2, 2), (2, 2, 4), (4, 2, 4)]
    ops_counts: list[float] = []
    per_iter_times: list[float] = []

    for nx, ny, nz in dims:
        mol = make_pyscf_mol(get_water_cluster_coords(nx, ny, nz), basis="def2-svp")
        auxmol = df.addons.make_auxmol(mol, auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        nocc = mol.nelectron // 2

        mf = RHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-9).density_fit(
            auxbasis=auxbasis
        )
        mf.kernel()
        assert mf.converged
        assert mf.iterations >= 1

        t_per_iter = float(mf.result.scf_time_s) / max(1, int(mf.iterations))
        ops_counts.append(float(naux) * float(nao) * float(nao) * float(nocc))
        per_iter_times.append(max(t_per_iter, 1e-6))

    slope, _ = np.polyfit(np.log(ops_counts), np.log(per_iter_times), 1)
    assert 0.35 <= float(slope) <= 1.25

    # Fixed-N_occ basis-ladder check on (H2O)_4 verifying sub-cubic N_AO scaling
    # consistent with O(N_aux * N_AO^2 * N_occ) rather than O(N_aux * N_AO^3)
    basis_naos: list[float] = []
    basis_times: list[float] = []
    h2o4_atoms = get_water_cluster_coords(2, 2, 1)
    for basis in ("sto-3g", "6-31g*", "def2-tzvp"):
        mol_b = make_pyscf_mol(h2o4_atoms, basis=basis)
        assert mol_b.nelectron // 2 == 20
        mf_b = RHF(mol_b, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-9).density_fit(
            auxbasis=auxbasis
        )
        mf_b.kernel()
        assert mf_b.converged
        assert mf_b.iterations >= 1
        t_b = float(mf_b.result.scf_time_s) / max(1, int(mf_b.iterations))
        basis_naos.append(float(mol_b.nao_nr()))
        basis_times.append(max(t_b, 1e-6))

    basis_slope, _ = np.polyfit(np.log(basis_naos), np.log(basis_times), 1)
    assert float(basis_slope) < 2.6, (
        f"Fixed-N_occ basis ladder scaling slope {float(basis_slope):.3f} >= 2.6"
    )
