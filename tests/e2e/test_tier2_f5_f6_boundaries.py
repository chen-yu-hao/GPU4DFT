"""Tier 2 boundary, edge, invariant, and negative tests for F5 & F6 (T2-F5-01..05, T2-F6-01..05)."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import BOHR_TO_ANGSTROM, rotate_coords
from data.molecules import get_molecule, get_water_cluster_coords
from oracles import (
    central_fd_grad,
    rotational_invariance_norm,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


# ---------------------------------------------------------------------------
# Tier 2: Feature F5 (Occupied-Space RI-K) Boundaries & Negative Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f5
@pytest.mark.gpu
def test_t2_f5_ri_k_zero_beta_occ_h2_cation(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F5-01: H2+ (N_occ_alpha=1, N_occ_beta=0) RI-K handles zero beta occupancy cleanly."""
    from gpu4dft import UHF

    spec = get_molecule("h2_cation")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(mol, "uhf", density_fit=True, auxbasis=auxbasis, conv_tol=1e-11)
    assert ref["converged"]
    assert ref["spin_square"] is not None

    mf = UHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
        auxbasis=auxbasis
    )
    e_got = mf.kernel()
    assert mf.converged
    assert abs(e_got - ref["e_tot"]) <= 1e-6
    ss_got, _ = mf.spin_square()
    assert abs(ss_got - ref["spin_square"][0]) <= 1e-5
    assert abs(ss_got - 0.75) <= 1e-5


@pytest.mark.tier2
@pytest.mark.f5
@pytest.mark.gpu
def test_t2_f5_ri_k_non_aligned_prime_dims(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F5-02: Non-warp-aligned / prime NAO & N_occ dimensions (CH3/def2-svp) in FP64 & Ozaki INT8."""
    from gpu4dft import UKS

    spec = get_molecule("ch3_radical")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
    # CH3/def2-svp: nao = 14 + 3*5 = 29 (prime), n_alpha = 5 (prime)
    assert mol.nao_nr() == 29
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(
        mol, "uks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=3, conv_tol=1e-11
    )
    assert ref["converged"]
    assert ref["spin_square"] is not None

    for precision, gemm_mode in (("fp64", "fp64"), ("mixed", "ozaki_int8")):
        mf = UKS(
            mol,
            xc="b3lyp",
            backend=gpu4dft_backend,
            precision=precision,
            dynamic_precision=(precision == "mixed"),
            gemm_mode=gemm_mode,
            conv_tol=1e-10,
        ).density_fit(auxbasis=auxbasis)
        match_grids(mf, ref["mf"])

        e_got = mf.kernel()
        assert mf.converged
        assert abs(e_got - ref["e_tot"]) <= 1e-6
        ss_got, _ = mf.spin_square()
        assert abs(ss_got - ref["spin_square"][0]) <= 1e-5


@pytest.mark.tier2
@pytest.mark.f5
@pytest.mark.gpu
def test_t2_f5_ri_k_linear_dependence_removal(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F5-03: Diffuse/tight dimer triggering lindep_tol (N_MO < N_AO) works cleanly in RI-K."""
    from gpu4dft import RHF

    tight_dimer = get_water_cluster_coords(2, 1, 1, spacing=1.95)
    mol = make_pyscf_mol(tight_dimer, basis="aug-cc-pvdz")
    auxbasis = "def2-universal-jkfit"

    # 1. Exact DF parity at standard lindep_tol=1e-7
    ref = run_pyscf_scf(mol, "rhf", density_fit=True, auxbasis=auxbasis, conv_tol=1e-10)
    assert ref["converged"]
    mf = RHF(mol, backend=gpu4dft_backend, lindep_tol=1e-7, conv_tol=1e-10).density_fit(
        auxbasis=auxbasis
    )
    e_got = mf.kernel()
    assert mf.converged
    assert abs(e_got - ref["e_tot"]) <= 1e-6

    # 2. Force canonical orthogonalization truncation (N_MO < N_AO) to verify rectangular C_occ
    mf_trunc = RHF(mol, backend=gpu4dft_backend, lindep_tol=1e-2, conv_tol=1e-9).density_fit(
        auxbasis=auxbasis
    )
    e_trunc = mf_trunc.kernel()
    assert mf_trunc.converged
    assert np.isfinite(e_trunc)
    assert mf_trunc.mo_coeff is not None
    assert mf_trunc.mo_coeff.shape[0] == mol.nao_nr()
    assert mf_trunc.mo_coeff.shape[1] < mol.nao_nr()


@pytest.mark.tier2
@pytest.mark.f5
@pytest.mark.gpu
def test_t2_f5_ri_k_host_vs_cuda_consistency(
    require_gpu4dft,
    require_pyscf,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F5-04: Occupied-space RI-K UKS(CAM-B3LYP) agrees between host and cuda <= 1e-10 Eh."""
    import gpu4dft
    from gpu4dft import UKS

    assert gpu4dft.backend_available("cuda"), "CUDA backend required for host-vs-cuda consistency"

    spec = get_molecule("ho2_radical")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(
        mol, "uks", xc="cam-b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=3, conv_tol=1e-11
    )

    mf_host = UKS(
        mol, xc="cam-b3lyp", backend="host", precision="fp64", conv_tol=1e-11
    ).density_fit(auxbasis=auxbasis)
    match_grids(mf_host, ref["mf"])
    e_host = mf_host.kernel()
    assert mf_host.converged
    ss_host, _ = mf_host.spin_square()

    mf_cuda = UKS(
        mol, xc="cam-b3lyp", backend="cuda", precision="fp64", conv_tol=1e-11
    ).density_fit(auxbasis=auxbasis)
    match_grids(mf_cuda, ref["mf"])
    e_cuda = mf_cuda.kernel()
    assert mf_cuda.converged
    ss_cuda, _ = mf_cuda.spin_square()

    assert abs(e_host - e_cuda) <= 1e-10
    assert abs(ss_host - ss_cuda) <= 1e-10


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_ri_k_invalid_auxbasis_raises(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F5-05: Unknown auxbasis or MoleculeSpec without aux_basis raises cleanly."""
    from gpu4dft import Atom, Basis, MoleculeSpec, RHF, Shell

    spec = get_molecule("h2o")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="sto-3g")

    with pytest.raises((ValueError, RuntimeError, KeyError)):
        RHF(mol, backend=gpu4dft_backend).density_fit(auxbasis="nonexistent-jkfit-999").kernel()

    # Standalone MoleculeSpec with aux_basis=None must also fail cleanly on .density_fit()
    shells = [Shell(0, [0.0, 0.0, 0.0], [3.42525091, 0.62391373], [0.15432897, 0.53532814])]
    atoms = [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    mspec = MoleculeSpec(
        basis=Basis(
            shells
            + [Shell(0, [0.0, 0.0, 1.4], [3.42525091, 0.62391373], [0.15432897, 0.53532814])]
        ),
        atoms=atoms,
        nelectron=2,
        aux_basis=None,
    )
    with pytest.raises((ValueError, RuntimeError)):
        RHF(mspec, backend=gpu4dft_backend).density_fit().kernel()


# ---------------------------------------------------------------------------
# Tier 2: Feature F6 (RI-JK Analytical Gradients) Boundaries & Negative Cases
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f6
@pytest.mark.gpu
def test_t2_f6_ri_grad_zero_beta_occ_h2_cation(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F6-01: H2+ (N_occ_beta=0) RI-JK analytical gradient matches PySCF DF and FD <= 1e-5."""
    from gpu4dft import UHF

    spec = get_molecule("h2_cation")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(mol, "uhf", density_fit=True, auxbasis=auxbasis, conv_tol=1e-11)
    g_ref = run_pyscf_grad(ref["mf"], auxbasis_response=True)

    mf = UHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-11).density_fit(
        auxbasis=auxbasis
    )
    mf.kernel()
    assert mf.converged
    g_got = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)

    assert np.max(np.abs(g_got - g_ref)) <= 1e-5
    assert translational_invariance_norm(g_got) <= 1e-6

    def _e_fn(c_bohr: np.ndarray) -> float:
        m_s = make_pyscf_mol(
            [("H", tuple(c_bohr[0])), ("H", tuple(c_bohr[1]))],
            basis="def2-svp",
            charge=1,
            spin=1,
            unit="Bohr",
        )
        return float(
            UHF(m_s, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-11)
            .density_fit(auxbasis=auxbasis)
            .kernel()
        )

    g_fd = central_fd_grad(_e_fn, spec.coords_bohr, step=1e-4)
    assert np.max(np.abs(g_got - g_fd)) <= 1e-5


@pytest.mark.tier2
@pytest.mark.f6
@pytest.mark.gpu
def test_t2_f6_ri_grad_g_shell_auxbasis(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F6-02: def2-tzvp + def2-universal-jkfit (l=4 g-aux shells) RI-JK SCF & native grad vs FD."""
    from pyscf import df
    from gpu4dft import RKS

    spec = get_molecule("h2o")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-tzvp")
    auxbasis = "def2-universal-jkfit"
    auxmol = df.addons.make_auxmol(mol, auxbasis)
    assert any(auxmol.bas_angular(i) == 4 for i in range(auxmol.nbas))
    syms = spec.symbols
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    for xc in ("b3lyp", "wb97x"):
        ref = run_pyscf_scf(
            mol, "rks", xc=xc, density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
        )
        g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

        mf_scf = RKS(mol, xc=xc, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
            auxbasis=auxbasis
        )
        match_grids(mf_scf, ref["mf"])
        e_got = float(mf_scf.kernel())
        assert mf_scf.converged
        assert abs(e_got - ref["e_tot"]) <= 1e-6

        def _build_native(c_bohr: np.ndarray, _xc=xc):
            m_s = make_pyscf_mol(
                [(syms[i], tuple(c_bohr[i])) for i in range(len(syms))],
                basis="def2-tzvp",
                unit="Bohr",
            )
            return RKS(
                m_s,
                xc=_xc,
                backend=gpu4dft_backend,
                precision="fp64",
                grids_level=5,
                conv_tol=1e-11,
                diis_tol=1e-8,
            ).density_fit(auxbasis=auxbasis)

        mf_grad = _build_native(coords_bohr)
        mf_grad.kernel()
        assert mf_grad.converged
        g_got = np.asarray(mf_grad.nuc_grad_method().kernel(), dtype=np.float64)

        assert translational_invariance_norm(g_got) <= 1e-6
        assert np.max(np.abs(g_got - g_ref)) <= 1e-4

        def _e_fn(c_bohr: np.ndarray, _b=_build_native) -> float:
            m_d = _b(c_bohr)
            e_d = float(m_d.kernel())
            assert m_d.converged
            return e_d

        g_fd = central_fd_grad(_e_fn, coords_bohr, step=1e-4)
        assert np.max(np.abs(g_got - g_fd)) <= 1e-5


@pytest.mark.tier2
@pytest.mark.f6
@pytest.mark.gpu
def test_t2_f6_ri_grad_symmetry_zero_components(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F6-03: Symmetry-forbidden out-of-axis (CO2) and out-of-plane (C6H6) RI-JK grads are zero."""
    from gpu4dft import RKS

    co2 = get_molecule("co2")
    mol_co2 = make_pyscf_mol(co2.pyscf_atom, basis="def2-svp")
    mf_co2 = RKS(
        mol_co2, xc="b3lyp", backend=gpu4dft_backend, grids_level=3, conv_tol=1e-10
    ).density_fit(auxbasis="def2-universal-jkfit")
    mf_co2.kernel()
    assert mf_co2.converged
    g_co2 = np.asarray(mf_co2.nuc_grad_method().kernel(), dtype=np.float64)
    assert np.max(np.abs(g_co2[:, :2])) <= 1e-10

    c6h6 = get_molecule("c6h6")
    mol_c6h6 = make_pyscf_mol(c6h6.pyscf_atom, basis="def2-svp")
    mf_c6h6 = RKS(
        mol_c6h6, xc="b3lyp", backend=gpu4dft_backend, grids_level=3, conv_tol=1e-10
    ).density_fit(auxbasis="def2-universal-jkfit")
    mf_c6h6.kernel()
    assert mf_c6h6.converged
    g_c6h6 = np.asarray(mf_c6h6.nuc_grad_method().kernel(), dtype=np.float64)
    assert np.max(np.abs(g_c6h6[:, 2])) <= 1e-10


@pytest.mark.tier2
@pytest.mark.f6
@pytest.mark.gpu
def test_t2_f6_ri_grad_rotational_invariance(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F6-04: RI-JK gradients satisfy rotational torque invariance (<=1e-6 Eh) and SO(3) equivariance."""
    from gpu4dft import RKS, UKS

    auxbasis = "def2-universal-jkfit"

    # 1. RKS(CAM-B3LYP) H2O
    h2o = get_molecule("h2o")
    mol_h2o = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")
    mf_h2o = RKS(
        mol_h2o, xc="cam-b3lyp", backend=gpu4dft_backend, grids_level=3, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mf_h2o.kernel()
    assert mf_h2o.converged
    g_h2o = np.asarray(mf_h2o.nuc_grad_method().kernel(), dtype=np.float64)
    assert translational_invariance_norm(g_h2o) <= 1e-6
    assert rotational_invariance_norm(h2o.coords_bohr, g_h2o) <= 1e-6

    # 2. UKS(B3LYP) HO2 + rigid rotation equivariance
    ho2 = get_molecule("ho2_radical")
    mol_ho2 = make_pyscf_mol(ho2.pyscf_atom, basis="def2-svp", charge=ho2.charge, spin=ho2.spin)
    mf_ho2 = UKS(
        mol_ho2, xc="b3lyp", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    e0 = mf_ho2.kernel()
    assert mf_ho2.converged
    g0 = np.asarray(mf_ho2.nuc_grad_method().kernel(), dtype=np.float64)
    assert translational_invariance_norm(g0) <= 1e-6
    assert rotational_invariance_norm(ho2.coords_bohr, g0) <= 1e-6

    coords_rot_ang, rot = rotate_coords(ho2.coords_angstrom, axis=(1.0, 2.0, 3.0), angle_deg=37.0)
    atoms_rot = [(ho2.symbols[i], tuple(coords_rot_ang[i])) for i in range(ho2.n_atoms)]
    mol_rot = make_pyscf_mol(atoms_rot, basis="def2-svp", charge=ho2.charge, spin=ho2.spin)
    mf_rot = UKS(
        mol_rot, xc="b3lyp", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    e_rot = mf_rot.kernel()
    assert mf_rot.converged
    g_rot = np.asarray(mf_rot.nuc_grad_method().kernel(), dtype=np.float64)

    assert abs(e_rot - e0) <= 1e-6
    assert np.max(np.abs(g_rot - (g0 @ rot.T))) <= 1e-5
    _ = BOHR_TO_ANGSTROM


@pytest.mark.tier2
@pytest.mark.f6
@pytest.mark.gpu
def test_t2_f6_ri_grad_no_silent_4c_fallback(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F6-05: Mismatched/minimal auxbasis proves RI-JK grad uses true DF, NOT silent 4c Direct fallback."""
    from gpu4dft import RHF

    spec = get_molecule("ch2o")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp")
    small_aux = "sto-3g"

    ref_4c = run_pyscf_scf(mol, "rhf", density_fit=False, conv_tol=1e-11)
    g_4c = run_pyscf_grad(ref_4c["mf"])

    ref_df = run_pyscf_scf(mol, "rhf", density_fit=True, auxbasis=small_aux, conv_tol=1e-11)
    g_df = run_pyscf_grad(ref_df["mf"], auxbasis_response=True)

    # Confirm the small auxbasis creates a >1e-3 Eh/a0 gap between 4c Direct and DF gradients
    assert np.max(np.abs(g_4c - g_df)) > 1e-3

    mf = RHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
        auxbasis=small_aux
    )
    e_got = mf.kernel()
    assert mf.converged
    assert abs(e_got - ref_df["e_tot"]) <= 1e-6

    g_got = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)
    assert np.max(np.abs(g_got - g_df)) <= 1e-5
    assert np.max(np.abs(g_got - g_4c)) > 1e-3
