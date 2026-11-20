"""Tier 2 boundary, invariant, and negative E2E tests for Features F3 and F4."""

from __future__ import annotations

import numpy as np
import pytest

from data.molecules import get_molecule
from oracles import run_pyscf_grad, run_pyscf_scf, translational_invariance_norm


# ---------------------------------------------------------------------------
# Feature F3 Boundaries & Negative Tests (T2-F3-01..05)
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_spin_electron_parity_mismatch_raises(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F3-01: Odd-electron RHF/RKS and UHF/UKS spin/electron parity/bounds mismatches raise cleanly."""
    from gpu4dft import RHF, RKS, UHF, UKS

    h2o = get_molecule("h2o")
    oh = get_molecule("oh_radical")

    # Odd-electron system (9 electrons, spin=1) must be rejected by restricted RHF and RKS
    mol_oh = make_pyscf_mol(oh.pyscf_atom, basis="sto-3g", charge=oh.charge, spin=oh.spin)
    with pytest.raises((ValueError, RuntimeError)):
        RHF(mol_oh, backend=gpu4dft_backend).kernel()
    with pytest.raises((ValueError, RuntimeError)):
        RKS(mol_oh, xc="b3lyp", backend=gpu4dft_backend).kernel()

    # 10 electrons with spin=1 (parity mismatch)
    mol_even_odd = make_pyscf_mol(h2o.pyscf_atom, basis="sto-3g", charge=0, spin=0)
    mol_even_odd.spin = 1
    with pytest.raises(ValueError):
        UHF(mol_even_odd, backend=gpu4dft_backend).kernel()

    # 9 electrons with spin=0 (parity mismatch)
    mol_odd_even = make_pyscf_mol(oh.pyscf_atom, basis="sto-3g", charge=0, spin=1)
    mol_odd_even.spin = 0
    with pytest.raises(ValueError):
        UKS(mol_odd_even, xc="b3lyp", backend=gpu4dft_backend).kernel()

    # Negative spin
    mol_neg = make_pyscf_mol(h2o.pyscf_atom, basis="sto-3g", charge=0, spin=0)
    mol_neg.spin = -1
    with pytest.raises(ValueError):
        UHF(mol_neg, backend=gpu4dft_backend).kernel()

    # spin > n_elec (12 > 10)
    mol_over = make_pyscf_mol(h2o.pyscf_atom, basis="sto-3g", charge=0, spin=0)
    mol_over.spin = 12
    with pytest.raises(ValueError):
        UHF(mol_over, backend=gpu4dft_backend).kernel()


@pytest.mark.tier2
@pytest.mark.f3
@pytest.mark.gpu
def test_t2_f3_single_electron_h_atom_zero_beta(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F3-02: Single-electron H atom (N_alpha=1, N_beta=0) yields exact 1e UHF energy & <S^2>=0.75."""
    from gpu4dft import UHF

    mol = make_pyscf_mol([("H", (0.0, 0.0, 0.0))], basis="def2-tzvp", charge=0, spin=1)
    nao = mol.nao_nr()

    ref = run_pyscf_scf(mol, "uhf", conv_tol=1e-12)
    assert ref["converged"]

    mf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-11)
    e_got = mf.kernel()
    assert mf.converged
    assert abs(e_got - ref["e_tot"]) <= 1e-10

    ss, mult = mf.spin_square()
    assert abs(ss - 0.75) <= 1e-7
    assert abs(mult - 2.0) <= 1e-7

    dm = mf.make_rdm1()
    assert dm.shape == (2, nao, nao)
    assert float(np.max(np.abs(dm[1]))) <= 1e-12


@pytest.mark.tier2
@pytest.mark.f3
@pytest.mark.gpu
def test_t2_f3_uks_spin0_matches_rks(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F3-03: Closed-shell H2O via UHF/UKS(spin=0) matches RHF/RKS <= 1e-10 Eh & <S^2> <= 1e-10."""
    from gpu4dft import RHF, RKS, UHF, UKS

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp", charge=0, spin=0)

    mf_rhf = RHF(mol, backend=gpu4dft_backend, conv_tol=1e-11, diis_tol=1e-8)
    e_rhf = mf_rhf.kernel()
    mf_uhf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-11, diis_tol=1e-8)
    e_uhf = mf_uhf.kernel()
    assert mf_rhf.converged and mf_uhf.converged
    assert abs(e_uhf - e_rhf) <= 1e-10
    assert mf_uhf.spin_square()[0] <= 1e-10

    mf_rks = RKS(mol, xc="b3lyp", backend=gpu4dft_backend, grids_level=3, conv_tol=1e-11, diis_tol=1e-8)
    e_rks = mf_rks.kernel()
    mf_uks = UKS(mol, xc="b3lyp", backend=gpu4dft_backend, grids_level=3, conv_tol=1e-11, diis_tol=1e-8)
    e_uks = mf_uks.kernel()
    assert mf_rks.converged and mf_uks.converged
    assert abs(e_uks - e_rks) <= 1e-10
    assert mf_uks.spin_square()[0] <= 1e-10


@pytest.mark.tier2
@pytest.mark.f3
@pytest.mark.gpu
def test_t2_f3_high_spin_quartet_n_atom_and_no2(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F3-04: High-spin quartet (spin=3) N atom & O2+ cation match PySCF <= 1e-6 Eh & <S^2> <= 1e-5."""
    from gpu4dft import UHF, UKS

    cases = (
        ("n_quartet", [("N", (0.0, 0.0, 0.0))], 0, 3),
        ("o2_cation_quartet", [("O", (0.0, 0.0, -0.575)), ("O", (0.0, 0.0, 0.575))], 1, 3),
    )

    for name, atom_spec, charge, spin in cases:
        mol = make_pyscf_mol(atom_spec, basis="def2-svp", charge=charge, spin=spin)

        ref_uhf = run_pyscf_scf(mol, "uhf", conv_tol=1e-11)
        assert ref_uhf["converged"], f"PySCF UHF failed for {name}"
        mf_uhf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-10)
        e_uhf = mf_uhf.kernel()
        assert mf_uhf.converged
        assert abs(e_uhf - ref_uhf["e_tot"]) <= 1e-6
        assert abs(mf_uhf.spin_square()[0] - ref_uhf["spin_square"][0]) <= 1e-5

        ref_uks = run_pyscf_scf(mol, "uks", xc="b3lyp", grids_level=3, conv_tol=1e-11)
        assert ref_uks["converged"], f"PySCF UKS failed for {name}"
        mf_uks = UKS(mol, xc="b3lyp", backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf_uks, ref_uks["mf"])
        e_uks = mf_uks.kernel()
        assert mf_uks.converged
        assert abs(e_uks - ref_uks["e_tot"]) <= 1e-6
        assert abs(mf_uks.spin_square()[0] - ref_uks["spin_square"][0]) <= 1e-5


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_invalid_dm0_shape_raises(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F3-05: Passing (nao, nao) or (3, nao, nao) dm0 to UHF/UKS raises ValueError."""
    from gpu4dft import UHF, UKS

    oh = get_molecule("oh_radical")
    mol = make_pyscf_mol(oh.pyscf_atom, basis="sto-3g", charge=oh.charge, spin=oh.spin)
    nao = mol.nao_nr()

    for bad_shape in ((nao, nao), (3, nao, nao)):
        bad_dm = np.zeros(bad_shape, dtype=np.float64)
        with pytest.raises(ValueError):
            UHF(mol, backend=gpu4dft_backend).kernel(dm0=bad_dm)
        with pytest.raises(ValueError):
            UKS(mol, xc="b3lyp", backend=gpu4dft_backend).kernel(dm0=bad_dm)


# ---------------------------------------------------------------------------
# Feature F4 Boundaries & Negative Tests (T2-F4-01..05)
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f4
@pytest.mark.gpu
def test_t2_f4_rsh_alias_normalization(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F4-01: Case/hyphen variants of CAM-B3LYP and wB97X produce identical SCF energies <= 1e-12 Eh."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")

    cam_energies = []
    for alias in ("CAM-B3LYP", "cam-b3lyp", "camb3lyp"):
        mf = RKS(mol, xc=alias, backend=gpu4dft_backend, grids_level=3, conv_tol=1e-11)
        e = mf.kernel()
        assert mf.converged
        cam_energies.append(float(e))
    assert max(cam_energies) - min(cam_energies) <= 1e-12

    wb_energies = []
    for alias in ("wB97X", "wb97x", "WB97X"):
        mf = RKS(mol, xc=alias, backend=gpu4dft_backend, grids_level=3, conv_tol=1e-11)
        e = mf.kernel()
        assert mf.converged
        wb_energies.append(float(e))
    assert max(wb_energies) - min(wb_energies) <= 1e-12


@pytest.mark.tier2
@pytest.mark.f4
@pytest.mark.gpu
def test_t2_f4_wb97x_distinct_from_wb97xd(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F4-02: xc='wb97x' matches PySCF 'wb97x' <= 1e-6 Eh and differs from 'wb97x-d' by > 1e-2 Eh."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")

    ref_wb97x = run_pyscf_scf(mol, "rks", xc="wb97x", grids_level=3, conv_tol=1e-11)
    ref_wb97xd = run_pyscf_scf(mol, "rks", xc="wb97x-d", grids_level=3, conv_tol=1e-11)
    assert ref_wb97x["converged"] and ref_wb97xd["converged"]

    mf = RKS(mol, xc="wb97x", backend=gpu4dft_backend, conv_tol=1e-10)
    match_grids(mf, ref_wb97x["mf"])
    e_got = mf.kernel()

    assert mf.converged
    assert abs(e_got - ref_wb97x["e_tot"]) <= 1e-6
    assert abs(e_got - ref_wb97xd["e_tot"]) > 1e-2


@pytest.mark.tier2
@pytest.mark.f4
@pytest.mark.gpu
def test_t2_f4_rsh_diffuse_and_high_l_basis(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F4-03: wB97X RKS/UKS with diffuse aug-cc-pvdz & f-shell def2-tzvp matches PySCF SCF & grad."""
    from gpu4dft import RKS, UKS

    h2o_dist = [
        ("O", (0.0200, -0.0100, 0.1400)),
        ("H", (0.0500, 0.8100, -0.4400)),
        ("H", (-0.0400, -0.7200, -0.5100)),
    ]
    oh_dist = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.05))]

    for basis in ("aug-cc-pvdz", "def2-tzvp"):
        mol_r = make_pyscf_mol(h2o_dist, basis=basis)
        ref_r_scf = run_pyscf_scf(mol_r, "rks", xc="wb97x", grids_level=3, conv_tol=1e-11)
        mf_r_matched = RKS(mol_r, xc="wb97x", backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf_r_matched, ref_r_scf["mf"])
        e_r = mf_r_matched.kernel()
        assert mf_r_matched.converged
        assert abs(e_r - ref_r_scf["e_tot"]) <= 1e-6

        ref_r_fine = run_pyscf_scf(mol_r, "rks", xc="wb97x", grids_level=5, conv_tol=1e-11)
        g_r_ref = run_pyscf_grad(ref_r_fine["mf"], grid_response=True)
        mf_r_fine = RKS(
            mol_r, xc="wb97x", backend=gpu4dft_backend, grids_level=5, conv_tol=1e-11, diis_tol=1e-8
        )
        mf_r_fine.kernel()
        assert mf_r_fine.converged
        g_r_got = mf_r_fine.nuc_grad_method().kernel()
        assert float(np.max(np.abs(g_r_got - g_r_ref))) <= 1e-5
        assert translational_invariance_norm(g_r_got) <= 1e-6

        mol_u = make_pyscf_mol(oh_dist, basis=basis, charge=0, spin=1)
        ref_u_scf = run_pyscf_scf(mol_u, "uks", xc="wb97x", grids_level=3, conv_tol=1e-11)
        mf_u_matched = UKS(mol_u, xc="wb97x", backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf_u_matched, ref_u_scf["mf"])
        e_u = mf_u_matched.kernel()
        assert mf_u_matched.converged
        assert abs(e_u - ref_u_scf["e_tot"]) <= 1e-6

        ref_u_fine = run_pyscf_scf(mol_u, "uks", xc="wb97x", grids_level=5, conv_tol=1e-11)
        g_u_ref = run_pyscf_grad(ref_u_fine["mf"], grid_response=True)
        mf_u_fine = UKS(
            mol_u, xc="wb97x", backend=gpu4dft_backend, grids_level=5, conv_tol=1e-11, diis_tol=1e-8
        )
        mf_u_fine.kernel()
        assert mf_u_fine.converged
        g_u_got = mf_u_fine.nuc_grad_method().kernel()
        assert float(np.max(np.abs(g_u_got - g_u_ref))) <= 1e-5
        assert translational_invariance_norm(g_u_got) <= 1e-6


@pytest.mark.tier2
@pytest.mark.f4
@pytest.mark.gpu
def test_t2_f4_rsh_stretched_dissociation_uhf_uks(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F4-04: Stretched LiH (R=4.0 A) & He2+ (R=3.0 A, spin=1) converge cleanly under RSH UKS."""
    from gpu4dft import UKS

    lih = get_molecule("stretched_lih")
    mol_lih = make_pyscf_mol(lih.pyscf_atom, basis="def2-svp", charge=lih.charge, spin=lih.spin)
    ref_lih = run_pyscf_scf(mol_lih, "uks", xc="wb97x", grids_level=3, conv_tol=1e-11)
    assert ref_lih["converged"]
    mf_lih = UKS(mol_lih, xc="wb97x", backend=gpu4dft_backend, conv_tol=1e-10, max_cycle=120)
    match_grids(mf_lih, ref_lih["mf"])
    e_lih = mf_lih.kernel()
    assert mf_lih.converged
    assert abs(e_lih - ref_lih["e_tot"]) <= 1e-6
    assert abs(mf_lih.spin_square()[0] - ref_lih["spin_square"][0]) <= 1e-5

    he2_p = get_molecule("he2_cation")
    mol_he2 = make_pyscf_mol(he2_p.pyscf_atom, basis="def2-svp", charge=he2_p.charge, spin=he2_p.spin)
    for xc in ("wb97x", "cam-b3lyp"):
        ref_he2 = run_pyscf_scf(mol_he2, "uks", xc=xc, grids_level=3, conv_tol=1e-11)
        assert ref_he2["converged"]
        mf_he2 = UKS(mol_he2, xc=xc, backend=gpu4dft_backend, conv_tol=1e-10, max_cycle=120)
        match_grids(mf_he2, ref_he2["mf"])
        e_he2 = mf_he2.kernel()
        assert mf_he2.converged
        assert abs(e_he2 - ref_he2["e_tot"]) <= 1e-6
        assert abs(mf_he2.spin_square()[0] - ref_he2["spin_square"][0]) <= 1e-5


@pytest.mark.tier2
@pytest.mark.f4
@pytest.mark.gpu
def test_t2_f4_rsh_snk_unsupported_or_exact(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F4-05: RKS(CAM-B3LYP).snk() either raises NotImplementedError or matches PySCF <= 5e-4 Eh."""
    from gpu4dft import RKS

    h2o = get_molecule("h2o")
    mol = make_pyscf_mol(h2o.pyscf_atom, basis="def2-svp")
    ref = run_pyscf_scf(mol, "rks", xc="cam-b3lyp", grids_level=4, conv_tol=1e-11)
    assert ref["converged"]

    mf = RKS(mol, xc="cam-b3lyp", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-9).snk()
    try:
        e_got = mf.kernel()
    except NotImplementedError:
        return

    assert mf.converged
    assert abs(e_got - ref["e_tot"]) <= 5e-4
