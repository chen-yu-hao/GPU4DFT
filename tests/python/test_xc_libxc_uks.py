"""Tests for Libxc 5.2.3 XC evaluation, RSH metadata, spin-polarized UKS XC, and grid_response."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import dft, gto
from pyscf.dft import libxc as pyscf_libxc
from pyscf.grad import rks as pyscf_grad_rks
from pyscf.grad import uks as pyscf_grad_uks

import gpu4dft


def _backends() -> list[str]:
    return ["host", "cuda"] if gpu4dft.backend_available("cuda") else ["host"]


def _make_open_shell_mol(kind: str = "o2") -> gto.Mole:
    if kind == "o2":
        return gto.M(
            atom="O 0.0 0.0 0.0; O 0.0 0.0 1.21",
            basis="def2-svp",
            spin=2,
            charge=0,
            cart=False,
            verbose=0,
        )
    if kind == "oh":
        return gto.M(
            atom="O 0.0 0.0 0.0; H 0.0 0.0 0.97",
            basis="def2-svp",
            spin=1,
            charge=0,
            cart=False,
            verbose=0,
        )
    raise ValueError(kind)


@pytest.mark.parametrize(
    "xc_name",
    ["cam-b3lyp", "wb97x", "wb97x-d", "b3lyp", "pbe0", "pbe", "hf"],
)
def test_rsh_coefficients_match_pyscf(xc_name: str):
    rsh = gpu4dft.xc_rsh_coefficients(xc_name)
    if xc_name == "hf":
        assert rsh.omega == 0.0
        assert rsh.alpha == pytest.approx(1.0, abs=1e-12)
        assert rsh.beta == 0.0
        assert not rsh.is_rsh
        return

    om_ref, al_ref, be_ref = pyscf_libxc.rsh_coeff(xc_name)
    assert rsh.omega == pytest.approx(om_ref, abs=1e-12)
    assert rsh.alpha == pytest.approx(al_ref, abs=1e-12)
    assert rsh.beta == pytest.approx(be_ref, abs=1e-12)
    assert rsh.c_lr == pytest.approx(al_ref, abs=1e-12)
    assert rsh.c_sr == pytest.approx(al_ref + be_ref, abs=1e-12)
    assert rsh.is_rsh == (om_ref > 0.0 and abs(be_ref) > 0.0)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize(
    "xc_name",
    ["svwn", "pbe", "blyp", "b3lyp", "b3lyp5", "pbe0", "tpss", "cam-b3lyp", "wb97x"],
)
def test_eval_vxc_uks_matches_pyscf(xc_name: str, backend: str):
    mol = _make_open_shell_mol("oh")
    mf_ref = dft.UKS(mol, xc=xc_name)
    mf_ref.grids.level = 2
    mf_ref.grids.build()

    dm_a0, dm_b0 = mf_ref.get_init_guess()
    # Perturb slightly to break trivial symmetry and verify asymmetric spin densities
    dm_a = np.ascontiguousarray(0.5 * (dm_a0 + dm_a0.T), dtype=np.float64)
    dm_b = np.ascontiguousarray(0.5 * (dm_b0 + dm_b0.T), dtype=np.float64)

    nelec_ref, exc_ref, vxc_ref = mf_ref._numint.nr_uks(mol, mf_ref.grids, xc_name, (dm_a, dm_b))

    basis = gpu4dft.basis_from_pyscf(mol)
    grid = gpu4dft.MolecularGrid.from_points(
        np.ascontiguousarray(mf_ref.grids.coords, dtype=np.float64),
        np.ascontiguousarray(mf_ref.grids.weights, dtype=np.float64),
        basis,
    )
    opts = gpu4dft.XcOptions()
    opts.backend = gpu4dft.Backend.cuda if backend == "cuda" else gpu4dft.Backend.host
    opts.spin_polarized = True
    opts.use_libxc = True
    eng = gpu4dft.XcEngine(basis, grid, xc_name, opts)

    res = eng.eval_vxc_uks(dm_a, dm_b)
    assert res.exc == pytest.approx(exc_ref, rel=1e-11, abs=1e-11)
    assert res.nelec_alpha == pytest.approx(nelec_ref[0], rel=1e-12, abs=1e-12)
    assert res.nelec_beta == pytest.approx(nelec_ref[1], rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(res.vxc_alpha, vxc_ref[0], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(res.vxc_beta, vxc_ref[1], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("xc_name", ["svwn", "pbe", "b3lyp", "tpss", "cam-b3lyp", "wb97x"])
def test_nuclear_grad_uks_fixed_grid_matches_pyscf(xc_name: str, backend: str):
    mol = _make_open_shell_mol("o2")
    mf_ref = dft.UKS(mol, xc=xc_name)
    mf_ref.grids.level = 2
    mf_ref.grids.build()
    dm_a, dm_b = mf_ref.get_init_guess()
    dm_a = np.ascontiguousarray(0.5 * (dm_a + dm_a.T), dtype=np.float64)
    dm_b = np.ascontiguousarray(0.5 * (dm_b + dm_b.T), dtype=np.float64)

    _, vxc_grad = pyscf_grad_uks.get_vxc(mf_ref._numint, mol, mf_ref.grids, xc_name, (dm_a, dm_b))
    aoslices = mol.aoslice_by_atom()
    g_ref = np.zeros((mol.natm, 3), dtype=np.float64)
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        g_ref[ia] = 2.0 * (
            np.einsum("xuv,uv->x", vxc_grad[0][:, p0:p1, :], dm_a[p0:p1, :])
            + np.einsum("xuv,uv->x", vxc_grad[1][:, p0:p1, :], dm_b[p0:p1, :])
        )

    atoms = gpu4dft.atoms_from_pyscf(mol)
    basis = gpu4dft.basis_from_pyscf(mol)
    grid = gpu4dft.MolecularGrid.from_points(
        np.ascontiguousarray(mf_ref.grids.coords, dtype=np.float64),
        np.ascontiguousarray(mf_ref.grids.weights, dtype=np.float64),
        basis,
    )
    opts = gpu4dft.XcOptions()
    opts.backend = gpu4dft.Backend.cuda if backend == "cuda" else gpu4dft.Backend.host
    opts.spin_polarized = True
    opts.use_libxc = True
    eng = gpu4dft.XcEngine(basis, grid, xc_name, opts)
    g_got = eng.nuclear_grad_uks(atoms, dm_a, dm_b)
    np.testing.assert_allclose(g_got, g_ref, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("partition", ["stratmann", "becke"])
@pytest.mark.parametrize("radii_adjust", [True, False])
def test_xc_grid_response_matches_moving_grid_fd(
    backend: str, partition: str, radii_adjust: bool
):
    mol0 = gto.M(
        atom="O 0.0 0.0 0.12; H 0.0 0.76 -0.47; H 0.09 -0.73 -0.43",
        basis="def2-svp",
        cart=False,
        verbose=0,
    )
    # Use a fixed AO density matrix on a moving grid + moving shell centers
    mf_tmp = dft.RKS(mol0, xc="pbe").run()
    dm_fixed = np.ascontiguousarray(mf_tmp.make_rdm1(), dtype=np.float64)

    be_enum = gpu4dft.Backend.cuda if backend == "cuda" else gpu4dft.Backend.host
    part_enum = (
        gpu4dft.PartitionScheme.stratmann
        if partition == "stratmann"
        else gpu4dft.PartitionScheme.becke
    )

    def make_eng(mol_obj: gto.Mole) -> tuple[list[gpu4dft.Atom], gpu4dft.XcEngine]:
        at = gpu4dft.atoms_from_pyscf(mol_obj)
        bas = gpu4dft.basis_from_pyscf(mol_obj)
        gopts = gpu4dft.GridOptions()
        gopts.level = 2
        gopts.partition = part_enum
        gopts.becke_radii_adjust = radii_adjust
        gopts.backend = be_enum
        grid = gpu4dft.MolecularGrid(at, bas, gopts)
        assert grid.has_partition_info
        xopts = gpu4dft.XcOptions()
        xopts.backend = be_enum
        xopts.grid_response = True
        xopts.use_libxc = True
        return at, gpu4dft.XcEngine(bas, grid, "b3lyp", xopts)

    atoms0, eng0 = make_eng(mol0)
    g_ana = eng0.nuclear_grad(atoms0, dm_fixed)
    # Exact translational invariance to machine precision
    np.testing.assert_allclose(np.sum(g_ana, axis=0), np.zeros(3), rtol=0.0, atol=1e-12)

    coords0 = np.asarray(mol0.atom_coords(unit="Bohr"), dtype=np.float64)
    charges = [int(mol0.atom_charge(i)) for i in range(mol0.natm)]
    h = 1e-4
    g_fd = np.zeros_like(g_ana)
    for ia in range(mol0.natm):
        for ix in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, ix] += h
            cm[ia, ix] -= h
            mol_p = gto.M(
                atom=[(charges[k], tuple(cp[k])) for k in range(mol0.natm)],
                basis="def2-svp",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            mol_m = gto.M(
                atom=[(charges[k], tuple(cm[k])) for k in range(mol0.natm)],
                basis="def2-svp",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            _, ep_eng = make_eng(mol_p)
            _, em_eng = make_eng(mol_m)
            ep = ep_eng.eval_vxc(dm_fixed).exc
            em = em_eng.eval_vxc(dm_fixed).exc
            g_fd[ia, ix] = (ep - em) / (2.0 * h)

    np.testing.assert_allclose(g_ana, g_fd, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("xc_name", ["svwn", "pbe", "b3lyp", "pbe0", "tpss"])
def test_rks_native_grid_response_gradients(xc_name: str, backend: str):
    mol = gto.M(
        atom="O 0.0 0.0 0.12; H 0.0 0.76 -0.47; H 0.08 -0.74 -0.44",
        basis="6-31g*",
        cart=False,
        verbose=0,
    )
    mf = gpu4dft.RKS(mol, xc=xc_name, backend=backend, grids_level=3, conv_tol=1e-12, diis_tol=1e-9).run()
    assert mf.converged
    if backend == "cuda":
        assert mf.result.scf.steady_state_allocations == 0

    gobj = mf.nuc_grad_method()
    assert gobj.grid_response is True
    g_ana = gobj.kernel()

    # 1. Translational invariance <= 1e-6 Eh/a0 (in practice ~1e-13)
    np.testing.assert_allclose(np.sum(g_ana, axis=0), np.zeros(3), rtol=0.0, atol=1e-6)

    # 2. Match moving-grid central finite differences <= 1e-5 Eh/a0
    coords0 = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    charges = [int(mol.atom_charge(i)) for i in range(mol.natm)]
    h = 1e-4
    g_fd = np.zeros_like(g_ana)
    for ia in range(mol.natm):
        for ix in range(3):
            cp = coords0.copy()
            cm = coords0.copy()
            cp[ia, ix] += h
            cm[ia, ix] -= h
            mol_p = gto.M(
                atom=[(charges[k], tuple(cp[k])) for k in range(mol.natm)],
                basis="6-31g*",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            mol_m = gto.M(
                atom=[(charges[k], tuple(cm[k])) for k in range(mol.natm)],
                basis="6-31g*",
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            ep = gpu4dft.RKS(mol_p, xc=xc_name, backend=backend, grids_level=3, conv_tol=1e-12, diis_tol=1e-9).kernel()
            em = gpu4dft.RKS(mol_m, xc=xc_name, backend=backend, grids_level=3, conv_tol=1e-12, diis_tol=1e-9).kernel()
            g_fd[ia, ix] = (ep - em) / (2.0 * h)
    np.testing.assert_allclose(g_ana, g_fd, rtol=0.0, atol=1e-5)

    # 3. Match PySCF grid_response=True on converged RKS (use dense native grid
    # to eliminate M4 vs Treutler quadrature discretization differences)
    mf_hi = gpu4dft.RKS(mol, xc=xc_name, backend=backend, conv_tol=1e-12, diis_tol=1e-9)
    mf_hi.grids.n_rad = 200
    mf_hi.grids.n_ang = 302
    mf_hi.grids.prune = False
    mf_hi.run()
    g_hi = mf_hi.nuc_grad_method().kernel()

    mf_ref = dft.RKS(mol, xc=xc_name)
    mf_ref.grids.level = 5
    mf_ref.conv_tol = 1e-12
    mf_ref.kernel()
    g_ref_obj = pyscf_grad_rks.Gradients(mf_ref)
    g_ref_obj.grid_response = True
    g_pyscf = g_ref_obj.kernel()
    np.testing.assert_allclose(g_hi, g_pyscf, rtol=0.0, atol=1e-5)

    # 4. Also verify toggling grid_response = False works on native grid
    if xc_name == "b3lyp":
        gobj.grid_response = False
        g_no_resp = gobj.kernel()
        assert np.max(np.abs(g_no_resp - g_ana)) > 1e-6
