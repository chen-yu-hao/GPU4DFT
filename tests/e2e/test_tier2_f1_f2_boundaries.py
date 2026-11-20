"""Tier 2 boundary, invariant, and negative E2E tests for Features F1 and F2."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import rotate_coords, translate_coords
from data.molecules import get_molecule
from oracles import (
    rotational_invariance_norm,
    run_dftd3_bj,
    run_pyscf_scf,
    translational_invariance_norm,
)


# ---------------------------------------------------------------------------
# Feature F1 Boundaries & Negative Tests (T2-F1-01..05)
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f1
def test_t2_f1_unknown_xc_raises(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F1-01: Unknown XC functional raises ValueError or RuntimeError in RKS and UKS."""
    from gpu4dft import RKS, UKS

    h2o = get_molecule("h2o")
    mol_r = make_pyscf_mol(h2o.pyscf_atom, basis="sto-3g")
    with pytest.raises((ValueError, RuntimeError)):
        RKS(mol_r, xc="nonexistent_xc_999", backend=gpu4dft_backend).kernel()

    oh = get_molecule("oh_radical")
    mol_u = make_pyscf_mol(oh.pyscf_atom, basis="sto-3g", charge=oh.charge, spin=oh.spin)
    with pytest.raises((ValueError, RuntimeError)):
        UKS(mol_u, xc="nonexistent_xc_999", backend=gpu4dft_backend).kernel()


@pytest.mark.tier2
@pytest.mark.f1
@pytest.mark.gpu
def test_t2_f1_coarse_grid_translational_invariance(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F1-02: Even on coarse grids_level=1, grid_response=True preserves |sum_A g_A|_inf <= 1e-6."""
    from gpu4dft import RKS

    atom_spec = [
        ("O", (0.0200, -0.0100, 0.1400)),
        ("H", (0.0500, 0.8100, -0.4400)),
        ("H", (-0.0400, -0.7200, -0.5100)),
    ]
    mol = make_pyscf_mol(atom_spec, basis="def2-svp")

    for xc in ("pbe", "b3lyp"):
        mf = RKS(
            mol,
            xc=xc,
            backend=gpu4dft_backend,
            grids_level=1,
            conv_tol=1e-11,
            diis_tol=1e-8,
        )
        mf.kernel()
        assert mf.converged
        grad = mf.nuc_grad_method().kernel()
        assert np.max(np.abs(grad)) > 1e-3
        net_force = translational_invariance_norm(grad)
        assert net_force <= 1e-6, f"Level-1 {xc} net force {net_force:.3e} > 1e-6"


@pytest.mark.tier2
@pytest.mark.f1
@pytest.mark.gpu
def test_t2_f1_single_atom_zero_grad_and_nelec(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T2-F1-03: Single atom Ne (RKS) and O (UKS spin=2) have zero gradient and exact electron count."""
    from gpu4dft import RKS, UKS

    mol_ne = make_pyscf_mol([("Ne", (0.0, 0.0, 0.0))], basis="def2-svp", charge=0, spin=0)
    ref_ne = run_pyscf_scf(mol_ne, "rks", xc="b3lyp", grids_level=3, conv_tol=1e-11)
    assert ref_ne["converged"]
    mf_ne = RKS(mol_ne, xc="b3lyp", backend=gpu4dft_backend, conv_tol=1e-11)
    match_grids(mf_ne, ref_ne["mf"])
    e_ne = mf_ne.kernel()
    assert mf_ne.converged
    assert abs(e_ne - ref_ne["e_tot"]) <= 1e-6
    g_ne = mf_ne.nuc_grad_method().kernel()
    assert g_ne.shape == (1, 3)
    assert float(np.max(np.abs(g_ne))) <= 1e-12
    nelec_ne = float(np.trace(mf_ne.make_rdm1() @ mf_ne.get_ovlp()))
    assert abs(nelec_ne - 10.0) <= 1e-8

    mol_o = make_pyscf_mol([("O", (0.0, 0.0, 0.0))], basis="def2-svp", charge=0, spin=2)
    ref_o = run_pyscf_scf(mol_o, "uks", xc="pbe", grids_level=3, conv_tol=1e-11)
    assert ref_o["converged"]
    mf_o = UKS(mol_o, xc="pbe", backend=gpu4dft_backend, conv_tol=1e-11)
    match_grids(mf_o, ref_o["mf"])
    e_o = mf_o.kernel()
    assert mf_o.converged
    assert abs(e_o - ref_o["e_tot"]) <= 1e-6
    g_o = mf_o.nuc_grad_method().kernel()
    assert g_o.shape == (1, 3)
    assert float(np.max(np.abs(g_o))) <= 1e-12
    dm_o = mf_o.make_rdm1()
    nelec_o = float(np.trace((dm_o[0] + dm_o[1]) @ mf_o.get_ovlp()))
    assert abs(nelec_o - 8.0) <= 1e-8


@pytest.mark.tier2
@pytest.mark.f1
@pytest.mark.gpu
def test_t2_f1_low_density_stretched_bond_no_nan(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F1-04: Stretched H2 (R=10.0 A) in low-density rho->0 regime yields finite SCF & grad."""
    from gpu4dft import RKS, UKS

    h2_far = get_molecule("stretched_h2")
    mol = make_pyscf_mol(h2_far.pyscf_atom, basis="def2-svp", charge=0, spin=0)

    for cls, xc in ((UKS, "pbe"), (RKS, "tpss")):
        mf = cls(mol, xc=xc, backend=gpu4dft_backend, grids_level=3, conv_tol=1e-9, max_cycle=120)
        e_tot = mf.kernel()
        assert np.isfinite(e_tot), f"Non-finite energy for stretched H2 {xc}"
        grad = mf.nuc_grad_method().kernel()
        assert grad.shape == (2, 3)
        assert np.all(np.isfinite(grad)), f"Non-finite gradient for stretched H2 {xc}"
        assert translational_invariance_norm(grad) <= 1e-6


@pytest.mark.tier2
@pytest.mark.f1
@pytest.mark.gpu
def test_t2_f1_rigid_translation_and_rotation_invariance(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T2-F1-05: Rigid translation (<=1e-8 Eh, <=1e-6 Eh/a0) and rotation (<=1e-6 Eh, <=1e-5 Eh/a0) invariance."""
    from gpu4dft import RKS

    atom0 = [
        ("O", (0.0200, -0.0100, 0.1400)),
        ("H", (0.0500, 0.8100, -0.4400)),
        ("H", (-0.0400, -0.7200, -0.5100)),
    ]
    mol0 = make_pyscf_mol(atom0, basis="def2-svp", unit="Angstrom")
    symbols = [sym for sym, _ in atom0]
    coords0_bohr = np.asarray(mol0.atom_coords(unit="Bohr"), dtype=np.float64)

    def _run_native(c_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        mol_c = make_pyscf_mol(
            [(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
            basis="def2-svp",
            unit="Bohr",
        )
        mf = RKS(
            mol_c,
            xc="b3lyp",
            backend=gpu4dft_backend,
            grids_level=5,
            conv_tol=1e-11,
            diis_tol=1e-8,
        )
        mf.grids.prune = False
        e = mf.kernel()
        assert mf.converged
        g = mf.nuc_grad_method().kernel()
        return float(e), np.asarray(g, dtype=np.float64)

    e0, g0 = _run_native(coords0_bohr)

    coords_trans = translate_coords(coords0_bohr, (10.5, -3.2, 7.1))
    e_trans, g_trans = _run_native(coords_trans)
    assert abs(e_trans - e0) <= 1e-8
    assert float(np.max(np.abs(g_trans - g0))) <= 1e-6

    coords_rot, rot = rotate_coords(coords0_bohr, axis=(1.0, 2.0, 3.0), angle_deg=37.0)
    e_rot, g_rot = _run_native(coords_rot)
    assert abs(e_rot - e0) <= 1e-6
    assert float(np.max(np.abs(g_rot - (g0 @ rot.T)))) <= 1e-5


# ---------------------------------------------------------------------------
# Feature F2 Boundaries & Negative Tests (T2-F2-01..05)
# ---------------------------------------------------------------------------


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_unsupported_functional_raises(require_gpu4dft) -> None:
    """T2-F2-01: Unsupported DFT-D3(BJ) functional raises ValueError (no silent zero fallback)."""
    from gpu4dft import dftd3_energy_grad

    z = [8, 1, 1]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.4, -1.0], [0.0, -1.4, -1.0]], dtype=np.float64)
    for bad_xc in ("unknown_func", "nonexistent_d3_xc", ""):
        with pytest.raises(ValueError):
            dftd3_energy_grad(z, coords, xc=bad_xc)


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_single_atom_exact_zero(require_gpu4dft) -> None:
    """T2-F2-02: Single atom DFT-D3(BJ) yields exact zero energy and gradient."""
    from gpu4dft import dftd3_energy_grad

    for z_list in ([6], [1]):
        coords = np.zeros((1, 3), dtype=np.float64)
        e, g = dftd3_energy_grad(z_list, coords, xc="b3lyp")
        assert abs(e) <= 1e-15
        assert g.shape == (1, 3)
        assert float(np.max(np.abs(g))) <= 1e-15


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_asymptotic_cutoff_and_compression(
    require_gpu4dft,
    require_dftd3,
) -> None:
    """T2-F2-03: Ne2 at compressed R=0.5 a0 and asymptotic R=65.0 a0 matches s-dftd3 & decays cleanly."""
    from gpu4dft import dftd3_energy_grad

    z = np.array([10, 10], dtype=np.int32)
    coords_comp = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=np.float64)
    coords_far = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 65.0]], dtype=np.float64)

    for xc in ("b3lyp", "wb97x"):
        e_ref_c, g_ref_c = run_dftd3_bj(z, coords_comp, xc=xc)
        e_got_c, g_got_c = dftd3_energy_grad(z, coords_comp, xc=xc)
        assert np.isfinite(e_got_c) and np.all(np.isfinite(g_got_c))
        assert abs(e_got_c - e_ref_c) <= 1e-8
        assert float(np.max(np.abs(g_got_c - g_ref_c))) <= 1e-7
        assert translational_invariance_norm(g_got_c) <= 1e-12

        e_ref_f, g_ref_f = run_dftd3_bj(z, coords_far, xc=xc)
        e_got_f, g_got_f = dftd3_energy_grad(z, coords_far, xc=xc)
        assert abs(e_got_f) <= 1e-7
        assert abs(e_got_f - e_ref_f) <= 1e-8
        assert float(np.max(np.abs(g_got_f - g_ref_f))) <= 1e-7
        assert translational_invariance_norm(g_got_f) <= 1e-12


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_invalid_input_shapes_and_z_raises(require_gpu4dft) -> None:
    """T2-F2-04: Mismatched atom/coord lengths, wrong coord rank/shape, and Z <= 0 raise ValueError."""
    from gpu4dft import dftd3_energy_grad

    with pytest.raises(ValueError):
        dftd3_energy_grad([6, 8], np.zeros((1, 3), dtype=np.float64), xc="b3lyp")

    with pytest.raises(ValueError):
        dftd3_energy_grad([6, 8], np.zeros(6, dtype=np.float64), xc="b3lyp")

    with pytest.raises(ValueError):
        dftd3_energy_grad([6, 8], np.zeros((2, 2), dtype=np.float64), xc="b3lyp")

    for bad_z in (0, -1):
        with pytest.raises(ValueError):
            dftd3_energy_grad(
                [bad_z, 8],
                np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], dtype=np.float64),
                xc="b3lyp",
            )


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_rigid_invariance_arbitrary_cluster(require_gpu4dft) -> None:
    """T2-F2-05: Arbitrary 6-atom cluster obeys exact rigid translation/rotation invariance & zero net torque."""
    from gpu4dft import dftd3_energy_grad

    z = np.array([6, 7, 8, 1, 1, 1], dtype=np.int32)
    coords0 = np.array(
        [
            [0.00, 0.10, -0.20],
            [2.45, -0.35, 0.60],
            [-1.90, 1.85, 0.95],
            [0.75, 1.95, -1.10],
            [3.80, 0.90, 1.40],
            [-2.95, 0.45, 2.15],
        ],
        dtype=np.float64,
    )

    coords_trans = translate_coords(coords0, (12.3, -4.5, 8.9))
    coords_rot, rot = rotate_coords(coords0, axis=(1.0, 2.0, 3.0), angle_deg=37.0)

    for xc in ("b3lyp", "pbe0", "wb97x"):
        e0, g0 = dftd3_energy_grad(z, coords0, xc=xc)
        e_trans, g_trans = dftd3_energy_grad(z, coords_trans, xc=xc)
        e_rot, g_rot = dftd3_energy_grad(z, coords_rot, xc=xc)

        assert abs(e_trans - e0) <= 1e-14
        assert abs(e_rot - e0) <= 1e-14
        assert translational_invariance_norm(g0) <= 1e-13
        assert rotational_invariance_norm(coords0, g0) <= 1e-13
        assert float(np.max(np.abs(g_trans - g0))) <= 1e-13
        assert float(np.max(np.abs(g_rot - (g0 @ rot.T)))) <= 1e-13
