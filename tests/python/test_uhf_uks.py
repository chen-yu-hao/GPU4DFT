"""End-to-end and unit tests for Milestone M2: Open-Shell UHF/UKS SCF,
DirectJKEngine UHF J/K and 2e nuclear gradients, SP2 open-shell solver,
<S^2> expectation values, non-PSD DIIS guard, D3(BJ), and ASE calculator.
"""

from __future__ import annotations

import numpy as np
import pytest

import gpu4dft

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, scf  # noqa: E402
from pyscf.grad import uhf as pyscf_grad_uhf  # noqa: E402
from pyscf.grad import uks as pyscf_grad_uks  # noqa: E402


def _available_backends() -> list[str]:
    out = ["host"]
    if gpu4dft.backend_available("cuda"):
        out.append("cuda")
    return out


BACKENDS = _available_backends()


def _mol_oh(basis: str = "6-31g"):
    return gto.M(
        atom="O 0.0 0.0 0.0; H 0.0 0.0 0.97",
        basis=basis,
        charge=0,
        spin=1,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_o2(basis: str = "6-31g"):
    return gto.M(
        atom="O 0.0 0.0 0.0; O 0.0 0.0 1.21",
        basis=basis,
        charge=0,
        spin=2,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_ch3(basis: str = "6-31g"):
    return gto.M(
        atom="C 0.0 0.0 0.0; H 1.08 0.0 0.0; H -0.54 0.935 0.0; H -0.54 -0.935 0.0",
        basis=basis,
        charge=0,
        spin=1,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_cn(basis: str = "6-31g"):
    return gto.M(
        atom="C 0.0 0.0 0.0; N 0.0 0.0 1.17",
        basis=basis,
        charge=0,
        spin=1,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_h2o_cation(basis: str = "def2-svp"):
    return gto.M(
        atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.0 -0.7572 -0.4692",
        basis=basis,
        charge=1,
        spin=1,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_ch2_triplet(basis: str = "def2-svp"):
    return gto.M(
        atom="C 0.0 0.0 0.0; H 0.0 0.99 0.43; H 0.0 -0.99 0.43",
        basis=basis,
        charge=0,
        spin=2,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_h_atom(basis: str = "def2-svp"):
    return gto.M(
        atom="H 0.0 0.0 0.0",
        basis=basis,
        charge=0,
        spin=1,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


# ---------------------------------------------------------------------------
# 1. DirectJKEngine UHF J/K & Open-Shell 2e Nuclear Gradients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("basis_name", ["6-31g*", "def2-tzvp"])
def test_direct_jk_uhf_matches_pyscf_and_spin_flip(backend: str, basis_name: str):
    mol = _mol_oh(basis=basis_name)
    mf_ref = scf.UHF(mol)
    mf_ref.conv_tol = 1e-10
    mf_ref.kernel()
    da, db = mf_ref.make_rdm1()

    basis = gpu4dft.basis_from_pyscf(mol)
    engine = gpu4dft.DirectJKEngine(basis, 1e-14)
    opts = gpu4dft.JKOptions()
    opts.backend = gpu4dft.Backend.cuda if backend == "cuda" else gpu4dft.Backend.host
    opts.precision = gpu4dft.Precision.fp64
    opts.schwarz_cutoff = 1e-13
    opts.compute_k = True

    vj_tot, vk_a, vk_b = engine.compute_jk_uhf(da, db, opts)

    vj_ref, vk_ref = mf_ref.get_jk(mol, np.stack([da, db], axis=0), hermi=1)
    vj_tot_ref = vj_ref[0] + vj_ref[1]
    np.testing.assert_allclose(vj_tot, vj_tot_ref, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(vk_a, vk_ref[0], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(vk_b, vk_ref[1], rtol=0.0, atol=1e-10)

    # Incremental spin-flip test: swap D_alpha and D_beta on the same engine instance
    # to verify split Schwarz bounds (d_sh_tot, d_sh_spin) remain valid when D_a != D_b flips.
    vj_flip, vk_a_flip, vk_b_flip = engine.compute_jk_uhf(db, da, opts)
    np.testing.assert_allclose(vj_flip, vj_tot_ref, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(vk_a_flip, vk_ref[1], rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(vk_b_flip, vk_ref[0], rtol=0.0, atol=1e-10)

    # Pure Coulomb mode (compute_k=False) delegates to Hermite J(D_tot)
    opts_j = gpu4dft.JKOptions()
    opts_j.backend = opts.backend
    opts_j.precision = gpu4dft.Precision.fp64
    opts_j.schwarz_cutoff = 1e-13
    opts_j.compute_k = False
    vj_only, vka_zero, vkb_zero = engine.compute_jk_uhf(da, db, opts_j)
    np.testing.assert_allclose(vj_only, vj_tot_ref, rtol=0.0, atol=1e-10)
    assert np.max(np.abs(vka_zero)) == 0.0
    assert np.max(np.abs(vkb_zero)) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k_fraction", [1.0, 0.2, 0.0])
def test_jk_nuclear_grad_uhf_matches_finite_difference(backend: str, k_fraction: float):
    mol0 = _mol_oh(basis="6-31g*")
    mf_ref = scf.UHF(mol0)
    mf_ref.conv_tol = 1e-11
    mf_ref.kernel()
    da, db = mf_ref.make_rdm1()

    basis0 = gpu4dft.basis_from_pyscf(mol0)
    atoms0 = gpu4dft.atoms_from_pyscf(mol0)
    be = gpu4dft.Backend.cuda if backend == "cuda" else gpu4dft.Backend.host
    g_ana = np.asarray(
        gpu4dft.jk_nuclear_grad_uhf(basis0, da, db, k_fraction, be, 1e-13, atoms0),
        dtype=np.float64,
    )
    assert g_ana.shape == (mol0.natm, 3)
    np.testing.assert_allclose(np.sum(g_ana, axis=0), np.zeros(3), rtol=0.0, atol=1e-11)

    # 2e energy at fixed AO density matrices for finite difference verification
    def e2e_fixed_dm(coords_bohr: np.ndarray) -> float:
        m = gto.M(
            atom=[(int(mol0.atom_charge(i)), tuple(coords_bohr[i])) for i in range(mol0.natm)],
            basis="6-31g*",
            charge=0,
            spin=1,
            unit="Bohr",
            cart=False,
            verbose=0,
        )
        b = gpu4dft.basis_from_pyscf(m)
        eng = gpu4dft.DirectJKEngine(b, 1e-14)
        jopts = gpu4dft.JKOptions()
        jopts.backend = be
        jopts.precision = gpu4dft.Precision.fp64
        jopts.schwarz_cutoff = 1e-13
        jopts.compute_k = abs(k_fraction) > 0.0
        vj, vka, vkb = eng.compute_jk_uhf(da, db, jopts)
        dtot = da + db
        return 0.5 * float(np.sum(dtot * vj)) - 0.5 * k_fraction * float(
            np.sum(da * vka) + np.sum(db * vkb)
        )

    coords0 = np.asarray(mol0.atom_coords(unit="Bohr"), dtype=np.float64)
    h = 2e-5
    g_fd = np.zeros_like(g_ana)
    for ia in range(mol0.natm):
        for ix in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, ix] += h
            cm[ia, ix] -= h
            g_fd[ia, ix] = (e2e_fixed_dm(cp) - e2e_fixed_dm(cm)) / (2.0 * h)

    np.testing.assert_allclose(g_ana, g_fd, rtol=0.0, atol=5e-9)


# ---------------------------------------------------------------------------
# 2. End-to-End UHF / UKS SCF Energy, <S^2>, Orbitals & Host/CUDA Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("xc_name", ["hf", "pbe", "b3lyp", "pbe0", "tpss"])
@pytest.mark.parametrize(
    "mol_fn",
    [_mol_oh, _mol_o2, _mol_ch3, _mol_cn, _mol_h2o_cation, _mol_ch2_triplet, _mol_h_atom],
    ids=["OH", "O2", "CH3", "CN", "H2O+", "CH2_triplet", "H_atom_Nbeta0"],
)
def test_uhf_uks_scf_and_spin_square_vs_pyscf(backend: str, xc_name: str, mol_fn):
    mol = mol_fn()
    if xc_name == "hf":
        mf_ref = scf.UHF(mol)
        mf_ref.conv_tol = 1e-11
        mf_ref.conv_tol_grad = 1e-8
        e_ref = mf_ref.kernel()
        if not mf_ref.converged:
            pytest.skip("PySCF reference UHF did not converge for this molecule/basis")
        mf = gpu4dft.UHF(mol, backend=backend, conv_tol=1e-10, diis_tol=1e-7).run()
    else:
        mf_ref = dft.UKS(mol, xc=xc_name)
        mf_ref.grids.level = 3
        mf_ref.grids.build()
        mf_ref.conv_tol = 1e-11
        mf_ref.conv_tol_grad = 1e-8
        e_ref = mf_ref.kernel()
        damp = 0.3 if mol_fn is _mol_oh else 0.0
        ls = 0.1 if mol_fn is _mol_oh else 0.0
        mf = gpu4dft.UKS(
            mol,
            xc=xc_name,
            backend=backend,
            grids=mf_ref.grids,
            damping=damp,
            level_shift=ls,
            conv_tol=1e-10,
            diis_tol=1e-7,
        ).run()

    assert mf.converged
    assert abs(mf.e_tot - e_ref) <= 1e-6

    s2_ref, mult_ref = mf_ref.spin_square()
    s2_got, mult_got = mf.spin_square()
    assert abs(s2_got - s2_ref) <= 1e-5
    assert abs(mult_got - mult_ref) <= 1e-5
    assert abs(mf.scf_summary["s2"] - s2_got) <= 1e-12

    # Open-shell attribute shapes: 2-tuples for MOs, (2, nao, nao) for DMs & Fock
    nao = mol.nao_nr()
    assert isinstance(mf.mo_energy, tuple) and len(mf.mo_energy) == 2
    assert isinstance(mf.mo_coeff, tuple) and len(mf.mo_coeff) == 2
    assert isinstance(mf.mo_occ, tuple) and len(mf.mo_occ) == 2
    assert mf.make_rdm1().shape == (2, nao, nao)
    assert mf.make_rdm1e().shape == (2, nao, nao)
    assert mf.get_fock().shape == (2, nao, nao)

    if mol_fn is _mol_h_atom:
        # N_beta == 0 edge case: exact spin-contamination-free <S^2> = S(S+1) = 0.75
        assert mf.result.scf.n_alpha == 1
        assert mf.result.scf.n_beta == 0
        assert abs(s2_got - 0.75) <= 1e-12

    if backend == "cuda":
        assert mf.result.scf.steady_state_allocations == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_uks_sp2_and_density_fit_and_d3bj(backend: str):
    mol = _mol_o2(basis="def2-svp")

    # 1. SP2 open-shell solver matches diag solver (including post-convergence MOs)
    mf_diag = gpu4dft.UKS(
        mol, xc="b3lyp", backend=backend, solver="diag", conv_tol=1e-10, diis_tol=1e-7
    ).run()
    mf_sp2 = gpu4dft.UKS(
        mol, xc="b3lyp", backend=backend, solver="sp2", conv_tol=1e-10, diis_tol=1e-7
    ).run()
    assert mf_diag.converged and mf_sp2.converged
    assert abs(mf_sp2.e_tot - mf_diag.e_tot) <= 1e-8
    nao = mol.nao_nr()
    nmo = int(mf_sp2.result.scf.nmo)
    assert mf_sp2.mo_energy[0].shape == (nmo,)
    assert mf_sp2.mo_energy[1].shape == (nmo,)
    assert mf_sp2.mo_coeff[0].shape == (nao, nmo)
    assert mf_sp2.mo_coeff[1].shape == (nao, nmo)
    np.testing.assert_allclose(mf_sp2.mo_energy[0], mf_diag.mo_energy[0], rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(mf_sp2.mo_energy[1], mf_diag.mo_energy[1], rtol=0.0, atol=1e-6)
    s2_diag, _ = mf_diag.spin_square()
    s2_sp2, _ = mf_sp2.spin_square()
    s2_sp2_mo, mult_sp2_mo = mf_sp2.spin_square(mo_coeff=mf_sp2.mo_coeff)
    assert abs(s2_sp2 - s2_diag) <= 1e-6
    assert abs(s2_sp2_mo - s2_diag) <= 1e-6
    assert mult_sp2_mo > 1.0
    np.testing.assert_allclose(mf_sp2.make_rdm1(), mf_diag.make_rdm1(), rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(
        mf_sp2.make_rdm1(mf_sp2.mo_coeff, mf_sp2.mo_occ),
        mf_sp2.make_rdm1(),
        rtol=0.0,
        atol=1e-6,
    )
    if backend == "cuda":
        assert mf_sp2.result.scf.steady_state_allocations == 0

    # 1b. SP2 with N_beta == 0 (H atom) populates valid MOs cleanly
    mol_h = _mol_h_atom(basis="def2-svp")
    mf_h_sp2 = gpu4dft.UHF(mol_h, backend=backend, solver="sp2", conv_tol=1e-10, diis_tol=1e-7).run()
    assert mf_h_sp2.converged
    assert mf_h_sp2.mo_energy[0].shape == (mol_h.nao_nr(),)
    assert mf_h_sp2.mo_energy[1].shape == (mol_h.nao_nr(),)
    assert mf_h_sp2.mo_coeff[0].shape == (mol_h.nao_nr(), mol_h.nao_nr())
    assert mf_h_sp2.mo_coeff[1].shape == (mol_h.nao_nr(), mol_h.nao_nr())
    s2_h_mo, _ = mf_h_sp2.spin_square(mo_coeff=mf_h_sp2.mo_coeff)
    assert abs(s2_h_mo - 0.75) <= 1e-12
    np.testing.assert_allclose(
        mf_h_sp2.make_rdm1(mf_h_sp2.mo_coeff, mf_h_sp2.mo_occ),
        mf_h_sp2.make_rdm1(),
        rtol=0.0,
        atol=1e-10,
    )

    # 2. Density-fitting RI-UKS parity vs PySCF RI-UKS
    mf_ri_ref = dft.UKS(mol, xc="pbe0").density_fit(auxbasis="def2-universal-jkfit")
    mf_ri_ref.grids.level = 3
    mf_ri_ref.grids.build()
    mf_ri_ref.conv_tol = 1e-11
    e_ri_ref = mf_ri_ref.kernel()
    s2_ri_ref, _ = mf_ri_ref.spin_square()

    mf_ri = gpu4dft.UKS(
        mol, xc="pbe0", backend=backend, grids=mf_ri_ref.grids, conv_tol=1e-10, diis_tol=1e-7
    ).density_fit(auxbasis="def2-universal-jkfit").run()
    assert mf_ri.converged
    assert abs(mf_ri.e_tot - e_ri_ref) <= 1e-6
    assert abs(mf_ri.spin_square()[0] - s2_ri_ref) <= 1e-5

    # 3. UKS + DFT-D3(BJ) energy additivity
    mf_d3 = gpu4dft.UKS(
        mol, xc="b3lyp", backend=backend, dispersion="d3bj", conv_tol=1e-10, diis_tol=1e-7
    ).run()
    assert mf_d3.converged
    assert mf_d3.e_disp < 0.0
    assert abs((mf_d3.e_tot - mf_d3.e_disp) - mf_diag.e_tot) <= 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
def test_uks_dipole_and_minao_guess_and_custom_dm0(backend: str):
    mol = _mol_h2o_cation(basis="def2-svp")
    dm0 = gpu4dft.minao_init_guess(mol, unrestricted=True)
    assert dm0.shape == (2, mol.nao_nr(), mol.nao_nr())

    mf_ref = dft.UKS(mol, xc="b3lyp")
    mf_ref.grids.level = 3
    mf_ref.grids.build()
    mf_ref.conv_tol = 1e-11
    mf_ref.kernel()
    dip_ref = mf_ref.dip_moment(unit="Debye", verbose=0)

    mf = gpu4dft.UKS(
        mol, xc="b3lyp", backend=backend, grids=mf_ref.grids, conv_tol=1e-10, diis_tol=1e-7
    ).run(dm0=dm0)
    assert mf.converged
    assert abs(mf.e_tot - mf_ref.e_tot) <= 1e-9
    np.testing.assert_allclose(mf.dipole_moment(unit="Debye"), dip_ref, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Open-Shell Analytical Nuclear Gradients vs PySCF & Moving-Grid FD
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "case",
    [
        ("hf", _mol_oh),
        ("pbe", _mol_o2),
        ("b3lyp", _mol_ch2_triplet),
        ("pbe0", _mol_h2o_cation),
        ("tpss", _mol_cn),
    ],
    ids=["OH_UHF", "O2_PBE", "CH2_B3LYP", "H2O+_PBE0", "CN_TPSS"],
)
def test_open_shell_analytical_gradients_vs_pyscf(backend: str, case):
    xc_name, mol_fn = case
    mol = mol_fn()

    if xc_name == "hf":
        mf_ref = scf.UHF(mol)
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        g_ref = pyscf_grad_uhf.Gradients(mf_ref).kernel()
        mf = gpu4dft.UHF(mol, backend=backend, conv_tol=1e-11, diis_tol=1e-8).run()
        grad_drv = mf.nuc_grad_method()
    else:
        mf_ref = dft.UKS(mol, xc=xc_name)
        mf_ref.grids.level = 3
        mf_ref.grids.build()
        mf_ref.conv_tol = 1e-12
        mf_ref.conv_tol_grad = 1e-9
        mf_ref.kernel()
        g_ref_drv = pyscf_grad_uks.Gradients(mf_ref)
        g_ref_drv.grid_response = False
        g_ref = g_ref_drv.kernel()

        mf = gpu4dft.UKS(
            mol,
            xc=xc_name,
            backend=backend,
            grids=mf_ref.grids,
            conv_tol=1e-11,
            diis_tol=1e-8,
        ).run()
        grad_drv = mf.nuc_grad_method()
        grad_drv.grid_response = False

    g_got = grad_drv.kernel()
    assert g_got.shape == (mol.natm, 3)
    np.testing.assert_allclose(g_got, g_ref, rtol=0.0, atol=1e-6)
    if xc_name == "hf":
        np.testing.assert_allclose(np.sum(g_got, axis=0), np.zeros(3), rtol=0.0, atol=1e-10)
    else:
        # Fixed-grid net force matches PySCF fixed-grid net force to < 1e-8
        np.testing.assert_allclose(
            np.sum(g_got, axis=0), np.sum(g_ref, axis=0), rtol=0.0, atol=1e-8
        )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("xc_name", ["hf", "b3lyp", "tpss"])
def test_open_shell_gradients_match_moving_grid_finite_differences(backend: str, xc_name: str):
    mol0 = _mol_h2o_cation(basis="6-31g")
    coords0 = np.asarray(mol0.atom_coords(unit="Bohr"), dtype=np.float64)
    charges = [int(mol0.atom_charge(i)) for i in range(mol0.natm)]
    h_bohr = 2e-4

    def make_mf(m):
        if xc_name == "hf":
            return gpu4dft.UHF(m, backend=backend, conv_tol=1e-11, diis_tol=1e-8).run()
        return gpu4dft.UKS(
            m,
            xc=xc_name,
            backend=backend,
            dispersion="d3bj",
            grids_level=3,
            conv_tol=1e-11,
            diis_tol=1e-8,
        ).run()

    mf0 = make_mf(mol0)
    grad_drv = mf0.nuc_grad_method()
    grad_drv.grid_response = True
    g_ana = grad_drv.kernel()
    np.testing.assert_allclose(np.sum(g_ana, axis=0), np.zeros(3), rtol=0.0, atol=1e-6)

    # Verify moving-grid analytical gradient across all 3 Cartesian directions of H1
    g_fd_h1 = np.zeros(3, dtype=np.float64)
    for ix in range(3):
        cp = coords0.copy()
        cm = coords0.copy()
        cp[1, ix] += h_bohr
        cm[1, ix] -= h_bohr
        mol_p = gto.M(
            atom=[(charges[k], tuple(cp[k])) for k in range(mol0.natm)],
            basis="6-31g",
            charge=1,
            spin=1,
            unit="Bohr",
            cart=False,
            verbose=0,
        )
        mol_m = gto.M(
            atom=[(charges[k], tuple(cm[k])) for k in range(mol0.natm)],
            basis="6-31g",
            charge=1,
            spin=1,
            unit="Bohr",
            cart=False,
            verbose=0,
        )
        e_p = make_mf(mol_p).e_tot
        e_m = make_mf(mol_m).e_tot
        g_fd_h1[ix] = (e_p - e_m) / (2.0 * h_bohr)

    np.testing.assert_allclose(g_ana[1], g_fd_h1, rtol=0.0, atol=2e-5)


# ---------------------------------------------------------------------------
# 4. Validation Errors & Response Guard
# ---------------------------------------------------------------------------


def test_open_shell_validation_and_response_guard():
    mol_oh = _mol_oh()
    nao = mol_oh.nao_nr()

    with pytest.raises(ValueError):
        gpu4dft.UHF(mol_oh, spin=-1).kernel()
    with pytest.raises(ValueError):
        gpu4dft.UHF(mol_oh, spin=2).kernel()  # OH has odd nelec=9 -> spin=2 impossible
    with pytest.raises(ValueError):
        gpu4dft.UHF(mol_oh, spin=99).kernel()
    with pytest.raises(ValueError):
        gpu4dft.UHF(mol_oh).kernel(dm0=np.zeros((3, nao, nao)))
    with pytest.raises(ValueError):
        gpu4dft.UHF(mol_oh).kernel(dm0=np.zeros((2, nao + 1, nao + 1)))

    # Unrestricted TDDFT/TDA is cleanly guarded as not implemented in M2
    mf = gpu4dft.UHF(mol_oh, backend="host").run()
    with pytest.raises((NotImplementedError, RuntimeError)):
        mf.TDA().kernel()


# ---------------------------------------------------------------------------
# 5. ASE GPU4DFTCalculator Open-Shell + D3(BJ) + Trajectory DM Reuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_ase_calculator_open_shell_and_dm_reuse(backend: str):
    pos0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]], dtype=np.float64)
    atoms0 = gpu4dft.SimpleAtoms(numbers=[8, 8], positions=pos0, charge=0, spin=2)
    calc = gpu4dft.GPU4DFTCalculator(
        method="uks",
        xc="b3lyp",
        basis="6-31g",
        spin=2,
        dispersion="d3bj",
        backend=backend,
        grids_level=3,
        conv_tol=1e-10,
    )
    atoms0.calc = calc
    e0 = atoms0.get_potential_energy()
    f0 = atoms0.get_forces()

    mol0 = _mol_o2(basis="6-31g")
    mf_ref = gpu4dft.UKS(
        mol0,
        xc="b3lyp",
        backend=backend,
        dispersion="d3bj",
        grids_level=3,
        conv_tol=1e-10,
    ).run()
    g_ref = mf_ref.nuc_grad_method()
    g_ref.grid_response = True
    de_ref = g_ref.kernel()

    from gpu4dft.ase_calculator import _BOHR_IN_ANGSTROM, _HARTREE_IN_EV

    bohr_per_ang = 1.0 / _BOHR_IN_ANGSTROM
    ha_to_ev = _HARTREE_IN_EV
    assert abs(e0 - mf_ref.e_tot * ha_to_ev) <= 1e-8
    np.testing.assert_allclose(f0, -de_ref * ha_to_ev * bohr_per_ang, rtol=0.0, atol=1e-7)
    assert calc.last_mf is not None
    assert calc.last_mf.make_rdm1().shape == (2, mol0.nao_nr(), mol0.nao_nr())
    assert calc._last_dm_key is not None and calc._last_dm_key[2] == 2

    # Displaced geometry reuses (2, nao, nao) last_dm without errors
    pos1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.215]], dtype=np.float64)
    atoms1 = gpu4dft.SimpleAtoms(numbers=[8, 8], positions=pos1, charge=0, spin=2)
    assert calc.check_state(atoms1) == ["positions"]
    e1 = calc.get_potential_energy(atoms1)
    f1 = calc.get_forces(atoms1)
    assert np.isfinite(e1) and f1.shape == (2, 3)

    # Optional ase.Atoms compatibility if ASE is installed
    try:
        import ase
    except ImportError:
        return
    ase_atoms = ase.Atoms("O2", positions=pos0)
    ase_atoms.calc = gpu4dft.GPU4DFTCalculator(
        method="uks",
        xc="b3lyp",
        basis="6-31g",
        spin=2,
        dispersion="d3bj",
        backend=backend,
        conv_tol=1e-10,
    )
    assert abs(ase_atoms.get_potential_energy() - e0) <= 1e-8
    np.testing.assert_allclose(ase_atoms.get_forces(), f0, rtol=0.0, atol=1e-7)

    # Mutating ase.Atoms.info["spin"] triggers recalculation via check_state
    ase_atoms.info["spin"] = 2
    assert "initial_magmoms" in ase_atoms.calc.check_state(ase_atoms)
    e_with_info = ase_atoms.get_potential_energy()
    assert abs(e_with_info - e0) <= 1e-8
    ase_atoms.info["spin"] = 0
    assert "initial_magmoms" in ase_atoms.calc.check_state(ase_atoms)
    e_singlet = ase_atoms.get_potential_energy()
    assert abs(e_singlet - e0) > 1e-2

    # Negative initial_magnetic_moments raises ValueError cleanly
    ase_neg = ase.Atoms("O2", positions=pos0, magmoms=[-1.0, -1.0])
    ase_neg.calc = gpu4dft.GPU4DFTCalculator(
        method="uks", xc="b3lyp", basis="6-31g", backend=backend, conv_tol=1e-10
    )
    with pytest.raises(ValueError, match="non-negative spin"):
        ase_neg.get_potential_energy()


@pytest.mark.parametrize("backend", BACKENDS)
def test_uks_tpss_non_psd_diis_guard(backend: str):
    mol = _mol_h2o_cation(basis="def2-svp")
    mf_ref = dft.UKS(mol, xc="tpss")
    mf_ref.grids.level = 3
    mf_ref.grids.build()
    mf_ref.conv_tol = 1e-11
    mf_ref.kernel()
    da_valid, db_valid = mf_ref.make_rdm1()

    basis = gpu4dft.basis_from_pyscf(mol)
    grid = gpu4dft.MolecularGrid.from_points(
        np.ascontiguousarray(mf_ref.grids.coords, dtype=np.float64),
        np.ascontiguousarray(mf_ref.grids.weights, dtype=np.float64),
        basis,
    )
    xopts = gpu4dft.XcOptions()
    xopts.backend = gpu4dft.Backend.cuda if backend == "cuda" else gpu4dft.Backend.host
    xopts.spin_polarized = True
    xopts.use_libxc = True
    xc_eng = gpu4dft.XcEngine(basis, grid, "tpss", xopts)

    # 1. Valid PSD density preserves <= 1e-12 parity vs PySCF
    _, exc_ref, vxc_ref = mf_ref._numint.nr_uks(mol, mf_ref.grids, "tpss", (da_valid, db_valid))
    res_valid = xc_eng.eval_vxc_uks(da_valid, db_valid)
    assert abs(res_valid.exc - float(exc_ref)) <= 1e-12
    np.testing.assert_allclose(res_valid.vxc_alpha, vxc_ref[0], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(res_valid.vxc_beta, vxc_ref[1], rtol=0.0, atol=1e-12)

    # 2. Aggressive non-PSD intermediate DIIS DM (top two eigenvalues flipped to -2.0, -1.5)
    evals, evecs = np.linalg.eigh(da_valid)
    evals[-1] = -2.0
    evals[-2] = -1.5
    da_nonpsd = np.ascontiguousarray((evecs * evals) @ evecs.T, dtype=np.float64)
    res_nonpsd = xc_eng.eval_vxc_uks(da_nonpsd, db_valid)
    assert np.isfinite(res_nonpsd.exc)
    assert np.all(np.isfinite(res_nonpsd.vxc_alpha))
    assert np.all(np.isfinite(res_nonpsd.vxc_beta))
    assert np.max(np.abs(res_nonpsd.vxc_alpha)) < 1e6
    assert np.max(np.abs(res_nonpsd.vxc_beta)) < 1e6
