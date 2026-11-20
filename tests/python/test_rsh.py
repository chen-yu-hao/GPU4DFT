import numpy as np
import pytest

import gpu4dft

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, df


def _backends():
    bes = ["host"]
    if gpu4dft.backend_available("cuda"):
        bes.append("cuda")
    return bes


def _build_h2o_mol(basis="cc-pvdz", spin=0, atoms=None):
    if atoms is None:
        atoms = "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586"
    mol = pyscf.gto.M(atom=atoms, basis=basis, spin=spin, unit="Angstrom", verbose=0)
    return mol


@pytest.mark.parametrize("backend", _backends())
def test_eri_omega_vs_pyscf_up_to_g(backend):
    # Minimal custom basis with s, p, d, f, g shells (spherical, matching PySCF default)
    mol = pyscf.gto.M(
        atom="O 0 0 0; H 0.9 0.4 -0.3; H -0.7 0.6 0.2",
        basis={
            "O": pyscf.gto.parse(
                """
                O S
                  1.50  1.0
                O P
                  0.85  1.0
                O D
                  0.65  1.0
                O F
                  0.55  1.0
                O G
                  0.45  1.0
                """
            ),
            "H": pyscf.gto.parse(
                """
                H S
                  0.70  1.0
                """
            ),
        },
        unit="Bohr",
        verbose=0,
    )
    basis = gpu4dft.basis_from_pyscf(mol)
    omega = 0.33
    got = gpu4dft.eri_tensor(basis, backend=backend, precision="fp64", omega=omega)
    with mol.with_range_coulomb(omega):
        ref = mol.intor("int2e", aosym="s1")
    np.testing.assert_allclose(got, ref, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize("backend", _backends())
def test_direct_jk_and_ri_jk_rsh_vs_pyscf(backend):
    # Include aug-cc-pvdz + def2-universal-jkfit to verify diffuse RI metric conditioning
    mol = _build_h2o_mol(basis="aug-cc-pvdz")
    basis = gpu4dft.basis_from_pyscf(mol)
    aux_basis = gpu4dft.aux_basis_from_pyscf(mol, "def2-universal-jkfit")

    rng = np.random.default_rng(42)
    nao = basis.nao
    a = rng.normal(size=(nao, nao))
    dm = 0.05 * (a + a.T)

    omega = 0.33
    k_sr = 0.19
    k_lr = 0.65

    # PySCF reference: J_{1/r}, K_{1/r}, K_omega
    j_ref, k_0_ref = pyscf.scf.hf.get_jk(mol, dm)
    with mol.with_range_coulomb(omega):
        _, k_w_ref = pyscf.scf.hf.get_jk(mol, dm)
    k_rsh_ref = k_sr * k_0_ref + (k_lr - k_sr) * k_w_ref

    djk = gpu4dft.DirectJKEngine(basis)
    opts = gpu4dft.JKOptions()
    opts.backend = gpu4dft._as_enum(gpu4dft.Backend, backend)
    opts.precision = gpu4dft.Precision.fp64
    opts.schwarz_cutoff = 1e-14
    opts.omega = omega
    opts.rsh_mode = True
    opts.k_sr = k_sr
    opts.k_lr = k_lr
    vj_d, vk_d = djk.compute_jk(dm, opts)
    np.testing.assert_allclose(vj_d, j_ref, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(vk_d, k_rsh_ref, atol=1e-10, rtol=1e-10)

    # Pure omega DirectJKEngine
    opts_w = gpu4dft.JKOptions()
    opts_w.backend = opts.backend
    opts_w.precision = gpu4dft.Precision.fp64
    opts_w.schwarz_cutoff = 1e-14
    opts_w.omega = omega
    opts_w.rsh_mode = False
    _, vk_w_d = djk.compute_jk(dm, opts_w)
    np.testing.assert_allclose(vk_w_d, k_w_ref, atol=1e-10, rtol=1e-10)

    # RIJKEngine RSH vs PySCF Density Fitting
    rijk = gpu4dft.RIJKEngine(basis, aux_basis, opts.backend, rsh_omega=omega)
    assert abs(rijk.rsh_omega - omega) < 1e-15
    vj_ri, vk_ri = rijk.compute_jk(dm, opts)

    with_df0 = df.DF(mol, auxbasis="def2-universal-jkfit")
    j_df_ref, k0_df_ref = with_df0.get_jk(dm, hermi=1)
    with df.DF(mol, auxbasis="def2-universal-jkfit").range_coulomb(omega) as with_dfw:
        _, kw_df_ref = with_dfw.get_jk(dm, hermi=1)
    k_rsh_df_ref = k_sr * k0_df_ref + (k_lr - k_sr) * kw_df_ref

    np.testing.assert_allclose(vj_ri, j_df_ref, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(vk_ri, k_rsh_df_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(
    "mol_spec",
    [
        ("h2o_631g", "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586", "6-31g"),
        ("h2o_hf_tzvp", "O 0 0 0; H 0 0.757 0.586; H 0 -0.757 0.586; H 2.5 0 0; F 2.5 0 0.92", "def2-tzvp"),
    ],
)
def test_direct_jk_nonsym_rsh_pass1_pure_j_vs_pyscf(backend, mol_spec):
    _, atom_str, basis_name = mol_spec
    mol = pyscf.gto.M(atom=atom_str, basis=basis_name, unit="Angstrom", verbose=0)
    basis = gpu4dft.basis_from_pyscf(mol)
    nao = basis.nao

    rng = np.random.default_rng(12345)
    dm_asym = rng.normal(size=(nao, nao)) * 0.05 + 0.3 * np.triu(rng.normal(size=(nao, nao)), k=1)
    da_asym = rng.normal(size=(nao, nao)) * 0.04 + 0.25 * np.triu(rng.normal(size=(nao, nao)), k=1)
    db_asym = rng.normal(size=(nao, nao)) * 0.03 - 0.20 * np.triu(rng.normal(size=(nao, nao)), k=1)
    assert np.max(np.abs(dm_asym - dm_asym.T)) > 0.1

    j_ref, _ = pyscf.scf.hf.get_jk(mol, dm_asym, hermi=0)
    j_uhf_ref, _ = pyscf.scf.hf.get_jk(mol, da_asym + db_asym, hermi=0)

    djk = gpu4dft.DirectJKEngine(basis)
    be_enum = gpu4dft._as_enum(gpu4dft.Backend, backend)
    prec_list = [("fp64", gpu4dft.Precision.fp64, 1e-10)]
    if backend == "cuda":
        prec_list.extend(
            [
                ("mixed", gpu4dft.Precision.mixed, 1e-7),
                ("fp32", gpu4dft.Precision.fp32, 1e-4),
            ]
        )

    cases = [
        ("pure_lr", 0.4, 0.0, 1.0, 1.0),
        ("k_scale_zero", 0.33, 0.19, 0.65, 0.0),
    ]
    for _, omega, k_sr, k_lr, k_scale in cases:
        with mol.with_range_coulomb(omega):
            _, kw_ref = pyscf.scf.hf.get_jk(mol, dm_asym, hermi=0)
            _, kwa_ref = pyscf.scf.hf.get_jk(mol, da_asym, hermi=0)
            _, kwb_ref = pyscf.scf.hf.get_jk(mol, db_asym, hermi=0)

        k_ref = k_scale * (k_lr - k_sr) * kw_ref if k_sr == 0.0 and k_scale != 0.0 else np.zeros_like(dm_asym)
        ka_ref = k_scale * (k_lr - k_sr) * kwa_ref if k_sr == 0.0 and k_scale != 0.0 else np.zeros_like(da_asym)
        kb_ref = k_scale * (k_lr - k_sr) * kwb_ref if k_sr == 0.0 and k_scale != 0.0 else np.zeros_like(db_asym)

        for _, prec_enum, tol in prec_list:
            opts = gpu4dft.JKOptions()
            opts.backend = be_enum
            opts.precision = prec_enum
            opts.schwarz_cutoff = 1e-14
            opts.omega = omega
            opts.rsh_mode = True
            opts.k_sr = k_sr
            opts.k_lr = k_lr
            opts.k_scale = k_scale
            opts.symmetrize_dm = False
            opts.compute_j = True
            opts.compute_k = True

            vj, vk = djk.compute_jk(dm_asym, opts)
            np.testing.assert_allclose(vj, j_ref, atol=tol, rtol=tol)
            np.testing.assert_allclose(vk, k_ref, atol=tol, rtol=tol)

            uvj, uvka, uvkb = djk.compute_jk_uhf(da_asym, db_asym, opts)
            np.testing.assert_allclose(uvj, j_uhf_ref, atol=tol, rtol=tol)
            np.testing.assert_allclose(uvka, ka_ref, atol=tol, rtol=tol)
            np.testing.assert_allclose(uvkb, kb_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("backend", _backends())
def test_jk_nuclear_grad_rsh_vs_fd(backend):
    coords_0 = np.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]], dtype=np.float64)
    syms = ["O", "H", "H"]

    def make_mol(c):
        atom_spec = [(s, tuple(xyz)) for s, xyz in zip(syms, c)]
        return pyscf.gto.M(atom=atom_spec, basis="6-31g", unit="Bohr", verbose=0)

    mol0 = make_mol(coords_0)
    basis0 = gpu4dft.basis_from_pyscf(mol0)
    atoms0 = gpu4dft.atoms_from_pyscf(mol0)
    nao = basis0.nao
    rng = np.random.default_rng(7)
    a = rng.normal(size=(nao, nao))
    b = rng.normal(size=(nao, nao))
    da = 0.04 * (a + a.T)
    db = 0.03 * (b + b.T)
    dtot = da + db

    k_sr, k_lr, omega = 0.19, 0.65, 0.33
    g_rhf = gpu4dft.jk_nuclear_grad_rsh(
        basis0, dtot, k_sr, k_lr, omega, backend=backend, schwarz_cutoff=1e-14, atoms=atoms0
    )
    g_uhf = gpu4dft.jk_nuclear_grad_uhf_rsh(
        basis0, da, db, k_sr, k_lr, omega, backend=backend, schwarz_cutoff=1e-14, atoms=atoms0
    )

    np.testing.assert_allclose(g_rhf.sum(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(g_uhf.sum(axis=0), 0.0, atol=1e-10)

    def eval_e2e_pyscf(c):
        m = make_mol(c)
        j0, k0 = pyscf.scf.hf.get_jk(m, dtot)
        with m.with_range_coulomb(omega):
            _, kw = pyscf.scf.hf.get_jk(m, dtot)
        k_rsh = k_sr * k0 + (k_lr - k_sr) * kw
        return 0.5 * np.einsum("ij,ij->", dtot, j0) - 0.25 * np.einsum("ij,ij->", dtot, k_rsh)

    def eval_e2e_uhf_pyscf(c):
        m = make_mol(c)
        j0, k0a = pyscf.scf.hf.get_jk(m, da)
        j0b, k0b = pyscf.scf.hf.get_jk(m, db)
        with m.with_range_coulomb(omega):
            _, kwa = pyscf.scf.hf.get_jk(m, da)
            _, kwb = pyscf.scf.hf.get_jk(m, db)
        j_tot = j0 + j0b
        ka_rsh = k_sr * k0a + (k_lr - k_sr) * kwa
        kb_rsh = k_sr * k0b + (k_lr - k_sr) * kwb
        return 0.5 * np.einsum("ij,ij->", dtot, j_tot) - 0.5 * (
            np.einsum("ij,ij->", da, ka_rsh) + np.einsum("ij,ij->", db, kb_rsh)
        )

    h = 1e-5
    g_fd_rhf = np.zeros_like(g_rhf)
    g_fd_uhf = np.zeros_like(g_uhf)
    for ia in range(3):
        for ax in range(3):
            cp = coords_0.copy()
            cm = coords_0.copy()
            cp[ia, ax] += h
            cm[ia, ax] -= h
            g_fd_rhf[ia, ax] = (eval_e2e_pyscf(cp) - eval_e2e_pyscf(cm)) / (2.0 * h)
            g_fd_uhf[ia, ax] = (eval_e2e_uhf_pyscf(cp) - eval_e2e_uhf_pyscf(cm)) / (2.0 * h)

    np.testing.assert_allclose(g_rhf, g_fd_rhf, atol=1e-7)
    np.testing.assert_allclose(g_uhf, g_fd_uhf, atol=1e-7)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("xc", ["cam-b3lyp", "wb97x"])
@pytest.mark.parametrize("use_df", [False, True])
def test_rks_rsh_scf_vs_pyscf(backend, xc, use_df):
    mol = _build_h2o_mol(basis="cc-pvdz")

    # PySCF reference on dense grid
    pmf = dft.RKS(mol, xc=xc)
    if use_df:
        pmf = pmf.density_fit(auxbasis="def2-universal-jkfit")
    pmf.grids.level = 4
    pmf.conv_tol = 1e-11
    pmf.kernel()
    assert pmf.converged

    mf = gpu4dft.RKS(mol, xc=xc, grids=pmf.grids, backend=backend)
    mf.conv_tol = 1e-10
    mf.diis_tol = 1e-8
    if use_df:
        mf = mf.density_fit("def2-universal-jkfit")
    e_got = mf.kernel()
    assert mf.converged
    assert abs(e_got - pmf.e_tot) < 1e-6
    if backend == "cuda":
        assert mf.result.scf.steady_state_allocations == 0


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("xc", ["camb3lyp", "wb97x"])
@pytest.mark.parametrize("use_df", [False, True])
def test_uks_rsh_scf_vs_pyscf(backend, xc, use_df):
    mol = _build_h2o_mol(
        basis="cc-pvdz",
        spin=1,
        atoms="O 0 0 0; O 1.38 0.03 0; H -0.21 0.99 0.04",
    )
    pyscf_xc = "cam-b3lyp" if xc == "camb3lyp" else xc

    pmf = dft.UKS(mol, xc=pyscf_xc)
    if use_df:
        pmf = pmf.density_fit(auxbasis="def2-universal-jkfit")
    pmf.grids.level = 4
    pmf.max_cycle = 100
    pmf.conv_tol = 1e-11
    pmf.kernel()
    assert pmf.converged
    s2_ref, _ = pmf.spin_square()

    mf = gpu4dft.UKS(mol, xc=xc, grids=pmf.grids, backend=backend)
    mf.conv_tol = 1e-10
    mf.diis_tol = 1e-8
    if use_df:
        mf = mf.density_fit("def2-universal-jkfit")
    e_got = mf.kernel()
    assert mf.converged
    assert abs(e_got - pmf.e_tot) < 1e-6
    assert abs(mf.result.scf.s2 - s2_ref) < 1e-5
    if backend == "cuda":
        assert mf.result.scf.steady_state_allocations == 0


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("xc", ["cam-b3lyp", "wb97x"])
def test_rsh_gradients_vs_pyscf_and_fd(backend, xc):
    mol = _build_h2o_mol(basis="6-31g")
    pmf = dft.RKS(mol, xc=xc)
    pmf.grids.level = 4
    pmf.conv_tol = 1e-12
    pmf.kernel()
    pg = pmf.nuc_grad_method()
    pg.grid_response = True
    g_ref = pg.kernel()

    mf = gpu4dft.RKS(mol, xc=xc, grids=pmf.grids, backend=backend)
    mf.conv_tol = 1e-11
    mf.diis_tol = 1e-9
    mf.kernel()
    g_got = mf.nuc_grad_method().kernel()
    # External-grid MolecularGrid::from_points has no parent_atom metadata; compare against
    # PySCF fixed-grid gradient and validate full moving-grid + translational invariance below.
    pg_nresp = pmf.nuc_grad_method()
    pg_nresp.grid_response = False
    np.testing.assert_allclose(g_got, pg_nresp.kernel(), atol=1e-6)
    np.testing.assert_allclose(g_got, g_ref, atol=3e-5)

    # Moving-grid SCF finite-difference check on H1-z (GPU4DFT internal grid)
    mf_int = gpu4dft.RKS(mol, xc=xc, backend=backend)
    mf_int.grids.level = 4
    mf_int.conv_tol = 1e-11
    mf_int.diis_tol = 1e-9
    mf_int.kernel()
    g_int = mf_int.nuc_grad_method().kernel()
    np.testing.assert_allclose(g_int.sum(axis=0), 0.0, atol=1e-6)

    coords_b = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    syms = [mol.atom_symbol(i) for i in range(mol.natm)]
    h = 1e-4

    def eval_scf_dz(dz):
        c = coords_b.copy()
        c[1, 2] += dz
        m_p = pyscf.gto.M(atom=[(s, tuple(xyz)) for s, xyz in zip(syms, c)], basis="6-31g", unit="Bohr", verbose=0)
        mf_p = gpu4dft.RKS(m_p, xc=xc, backend=backend)
        mf_p.grids.level = 4
        mf_p.conv_tol = 1e-11
        mf_p.diis_tol = 1e-9
        return mf_p.kernel()

    g_fd_z = (eval_scf_dz(h) - eval_scf_dz(-h)) / (2.0 * h)
    assert abs(g_int[1, 2] - g_fd_z) < 2e-5


@pytest.mark.parametrize("backend", _backends())
def test_rsh_uks_gradient_and_d3bj_calculator_and_guards(backend):
    # Open-shell gradient vs PySCF
    mol_oh = _build_h2o_mol(basis="6-31g", spin=1, atoms="O 0 0 0; H 0 0 0.97")
    pmf_u = dft.UKS(mol_oh, xc="cam-b3lyp")
    pmf_u.grids.level = 4
    pmf_u.conv_tol = 1e-12
    pmf_u.kernel()
    pg_u = pmf_u.nuc_grad_method()
    pg_u.grid_response = True
    g_u_ref = pg_u.kernel()

    mf_u = gpu4dft.UKS(mol_oh, xc="cam-b3lyp", backend=backend)
    mf_u.grids.level = 4
    mf_u.conv_tol = 1e-11
    mf_u.diis_tol = 1e-9
    mf_u.kernel()
    g_u_got = mf_u.nuc_grad_method().kernel()
    np.testing.assert_allclose(g_u_got, g_u_ref, atol=1e-5)
    np.testing.assert_allclose(g_u_got.sum(axis=0), 0.0, atol=1e-6)

    # D3(BJ) energy + gradient + ASE-compatible GPU4DFTCalculator + stale-result clear
    mol = _build_h2o_mol(basis="sto-3g")
    mf = gpu4dft.RKS(mol, xc="cam-b3lyp", backend=backend, dispersion="d3bj")
    e0 = mf.kernel()
    assert mf.result is not None
    assert mf.result.scf.dispersion_energy < 0.0
    mf_df = mf.density_fit("def2-universal-jkfit")
    assert mf_df.result is None
    e_df = mf_df.kernel()
    assert abs(e_df - e0) < 5e-3

    # .snk() NotImplementedError guard
    mf_snk = gpu4dft.RKS(mol, xc="cam-b3lyp", backend=backend).snk()
    assert mf_snk.result is None
    with pytest.raises(NotImplementedError):
        mf_snk.kernel()

    # GPU4DFTCalculator with RSH + d3bj
    ase = pytest.importorskip("ase")
    atoms = ase.Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]])
    calc = gpu4dft.GPU4DFTCalculator(
        xc="cam-b3lyp", basis="sto-3g", dispersion="d3bj", backend=backend
    )
    atoms.calc = calc
    e_calc = atoms.get_potential_energy()
    f_calc = atoms.get_forces()
    assert np.isfinite(e_calc)
    assert f_calc.shape == (3, 3)
    np.testing.assert_allclose(f_calc.sum(axis=0), 0.0, atol=1e-5)
