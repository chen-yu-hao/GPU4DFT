"""Unit & integration tests for F8.1 (.as_scanner(), geomopt, release, GridsConfig)
and F8.2 (GPU4DFTCalculator upgrades, trim_device_pool, dm0 reuse/fallback).
"""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import pytest

import gpu4dft

pyscf = pytest.importorskip("pyscf")
from pyscf import df as pyscf_df  # noqa: E402
from pyscf import dft as pyscf_dft  # noqa: E402
from pyscf import gto  # noqa: E402
from pyscf import scf as pyscf_scf  # noqa: E402


def _available_backends() -> list[str]:
    out = ["host"]
    if gpu4dft.backend_available("cuda"):
        out.append("cuda")
    return out


BACKENDS = _available_backends()
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def _make_h2o(basis: str = "def2-svp", shift_x: float = 0.0, distorted: bool = False) -> gto.Mole:
    if distorted:
        atom = [
            ("O", (shift_x, 0.0, 0.0)),
            ("H", (shift_x, 0.92, -0.45)),
            ("H", (shift_x, -0.92, -0.45)),
        ]
    else:
        atom = [
            ("O", (shift_x, 0.0, 0.1173)),
            ("H", (shift_x, 0.7572, -0.4692)),
            ("H", (shift_x, -0.7572, -0.4692)),
        ]
    return gto.M(atom=atom, basis=basis, charge=0, spin=0, unit="Angstrom", verbose=0)


def _make_oh(basis: str = "def2-svp", shift_z: float = 0.0, bond_len: float = 0.97) -> gto.Mole:
    return gto.M(
        atom=[("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond_len + shift_z))],
        basis=basis,
        charge=0,
        spin=1,
        unit="Angstrom",
        verbose=0,
    )


def _make_ho2(basis: str = "def2-svp", shift_x: float = 0.0) -> gto.Mole:
    return gto.M(
        atom=[
            ("H", (-0.876 + shift_x, 0.878, 0.0)),
            ("O", (shift_x, 0.955, 0.0)),
            ("O", (0.105 + shift_x, -0.350, 0.0)),
        ],
        basis=basis,
        charge=0,
        spin=1,
        unit="Angstrom",
        verbose=0,
    )


def _make_mol(kind: str, basis: str = "def2-svp", shift: float = 0.0) -> gto.Mole:
    if kind == "h2o":
        return _make_h2o(basis=basis, shift_x=shift)
    if kind == "oh":
        return _make_oh(basis=basis, shift_z=shift)
    if kind == "ho2":
        return _make_ho2(basis=basis, shift_x=shift)
    raise ValueError(f"unknown molecule kind {kind!r}")


@pytest.mark.parametrize("kind", ["rhf", "rks", "uhf", "uks"])
def test_scanner_isinstance_and_attributes(kind: str):
    from pyscf.lib import GradScanner as PySCFGradScanner
    from pyscf.lib import SinglePointScanner as PySCFSinglePointScanner

    if kind in ("rhf", "rks"):
        mol = _make_h2o(basis="sto-3g")
        mf = (
            gpu4dft.RHF(mol, backend="host", conv_tol=1e-10)
            if kind == "rhf"
            else gpu4dft.RKS(mol, xc="b3lyp", backend="host", conv_tol=1e-10)
        )
    else:
        mol = _make_ho2(basis="sto-3g")
        mf = (
            gpu4dft.UHF(mol, backend="host", conv_tol=1e-10)
            if kind == "uhf"
            else gpu4dft.UKS(mol, xc="b3lyp", backend="host", conv_tol=1e-10)
        )

    scf_scanner = mf.as_scanner()
    grad_scanner = mf.nuc_grad_method().as_scanner()

    assert isinstance(scf_scanner, gpu4dft.SCFScanner)
    assert isinstance(scf_scanner, PySCFSinglePointScanner)
    assert isinstance(grad_scanner, gpu4dft.GradScanner)
    assert isinstance(grad_scanner, PySCFGradScanner)

    for attr in ("mol", "e_tot", "converged", "stdout", "verbose", "mo_coeff", "mo_occ", "mo_energy"):
        assert hasattr(scf_scanner, attr)
    for attr in ("mol", "base", "e_tot", "de", "converged", "stdout", "verbose", "grid_response", "auxbasis_response"):
        assert hasattr(grad_scanner, attr)

    e_scf = scf_scanner(mol)
    assert np.isfinite(e_scf)
    assert scf_scanner.converged is True
    assert scf_scanner.e_tot == pytest.approx(e_scf)
    assert scf_scanner.mo_coeff is not None
    assert scf_scanner.mo_occ is not None
    assert scf_scanner.mo_energy is not None

    e_g, de = grad_scanner(mol)
    assert e_g == pytest.approx(e_scf, abs=1e-9)
    assert grad_scanner.converged is True
    assert grad_scanner.e_tot == pytest.approx(e_g)
    assert grad_scanner.de is not None and grad_scanner.de.shape == (mol.natm, 3)
    assert np.allclose(de, grad_scanner.de)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "method,xc,mol_kind,use_df,disp",
    [
        ("rhf", None, "h2o", False, None),
        ("rks", "b3lyp", "h2o", True, "d3bj"),
        ("uhf", None, "oh", True, None),
        ("uks", "cam-b3lyp", "ho2", True, None),
        ("uks", "wb97x", "ho2", True, None),
    ],
)
def test_scf_and_grad_scanner_parity_all_methods(
    backend: str,
    method: str,
    xc: str | None,
    mol_kind: str,
    use_df: bool,
    disp: str | None,
):
    mol1 = _make_mol(mol_kind, basis="def2-svp", shift=0.0)
    mol2 = _make_mol(mol_kind, basis="def2-svp", shift=0.05)

    def _build_mf(m: gto.Mole):
        cls_map = {
            "rhf": gpu4dft.RHF,
            "rks": gpu4dft.RKS,
            "uhf": gpu4dft.UHF,
            "uks": gpu4dft.UKS,
        }
        kw: dict[str, Any] = dict(
            backend=backend, precision="fp64", conv_tol=1e-11, diis_tol=1e-10
        )
        if xc is not None:
            kw["xc"] = xc
            kw["grids_level"] = 3
        if disp is not None:
            kw["dispersion"] = disp
        obj = cls_map[method](m, **kw)
        if use_df:
            obj.density_fit(auxbasis="def2-universal-jkfit")
        return obj

    base = _build_mf(mol1)
    scf_scanner = base.as_scanner()
    grad_scanner = base.nuc_grad_method().as_scanner()

    for m_cur in (mol1, mol2):
        e_scan = float(scf_scanner(m_cur))
        e_gscan, g_scan = grad_scanner(m_cur)

        mf_fresh = _build_mf(m_cur)
        e_fresh = float(mf_fresh.kernel())
        g_fresh = np.asarray(mf_fresh.nuc_grad_method().kernel(), dtype=np.float64)

        assert abs(e_scan - e_fresh) <= 1e-9
        assert abs(float(e_gscan) - e_fresh) <= 1e-9
        assert np.max(np.abs(np.asarray(g_scan, dtype=np.float64) - g_fresh)) <= 1e-9

    # Coordinate array and atom-string inputs on scanner
    coords2_ang = np.asarray(mol2.atom_coords(unit="Angstrom"), dtype=np.float64)
    e_arr = float(scf_scanner(coords2_ang))
    assert abs(e_arr - float(_build_mf(mol2).kernel())) <= 1e-9

    atom_lines = "\n".join(
        f"{mol2.atom_symbol(i)} {coords2_ang[i, 0]:.12f} {coords2_ang[i, 1]:.12f} {coords2_ang[i, 2]:.12f}"
        for i in range(mol2.natm)
    )
    e_str, g_str = grad_scanner(atom_lines)
    assert abs(e_str - e_arr) <= 1e-9
    assert g_str.shape == (mol2.natm, 3)

    # Propagate grid_response & auxbasis_response
    if use_df:
        grad_bad = base.nuc_grad_method()
        grad_bad.auxbasis_response = False
        gscan_bad = grad_bad.as_scanner()
        assert gscan_bad.auxbasis_response is False
        with pytest.raises(NotImplementedError):
            gscan_bad(mol1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "method,xc,mol_kind,check_pyscf",
    [
        ("rks", "b3lyp", "h2o", True),
        ("uks", "cam-b3lyp", "ho2", True),
        ("uks", "wb97x", "ho2", False),
    ],
)
def test_grad_scanner_moving_grid_fd_and_pyscf_parity(
    backend: str,
    method: str,
    xc: str,
    mol_kind: str,
    check_pyscf: bool,
):
    basis_name = "def2-svp" if (backend == "cuda" or mol_kind == "h2o") else "6-31g"
    mol = _make_mol(mol_kind, basis=basis_name)
    cls = gpu4dft.RKS if method == "rks" else gpu4dft.UKS
    mf = cls(mol, xc=xc, backend=backend, grids_level=4, conv_tol=1e-11).density_fit(
        "def2-universal-jkfit"
    )
    scf_scanner = mf.as_scanner()
    grad_drv = mf.nuc_grad_method()
    grad_drv.grid_response = True
    grad_drv.auxbasis_response = True
    grad_scanner = grad_drv.as_scanner()

    _, de = grad_scanner(mol)
    # Translational invariance
    np.testing.assert_allclose(np.sum(de, axis=0), np.zeros(3), rtol=0.0, atol=1e-6)

    # Moving-grid central finite differences in Bohr via scf_scanner
    # (Full all-atom FD on CUDA; atom 0 FD on 2-vCPU host to keep test runtime bounded)
    coords0_ang = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    step_bohr = 1e-4
    step_ang = step_bohr / BOHR_PER_ANGSTROM
    atom_indices = range(mol.natm) if backend == "cuda" else range(1)
    g_fd = np.zeros((len(list(atom_indices)), 3), dtype=np.float64)
    for row_idx, a in enumerate(atom_indices):
        for k in range(3):
            c_plus = coords0_ang.copy()
            c_minus = coords0_ang.copy()
            c_plus[a, k] += step_ang
            c_minus[a, k] -= step_ang
            ep = float(scf_scanner(c_plus))
            em = float(scf_scanner(c_minus))
            g_fd[row_idx, k] = (ep - em) / (2.0 * step_bohr)
    assert np.max(np.abs(de[list(atom_indices)] - g_fd)) <= 1e-5

    if check_pyscf:
        ref_cls = pyscf_dft.RKS if method == "rks" else pyscf_dft.UKS
        ref_mf = ref_cls(mol, xc=xc).density_fit(auxbasis="def2-universal-jkfit")
        ref_mf.grids.level = 4
        ref_mf.conv_tol = 1e-11
        ref_mf.kernel()
        ref_grad = ref_mf.nuc_grad_method()
        ref_grad.grid_response = True
        ref_grad.auxbasis_response = True
        de_ref = ref_grad.kernel()
        pyscf_tol = 2e-5 if mol_kind == "ho2" else 1e-5
        assert np.max(np.abs(de - de_ref)) <= pyscf_tol

        # Verify exact <= 1e-6 Eh/a0 PySCF parity when grid coords/weights are matched (grid_response=False)
        mf_mg = cls(mol, xc=xc, backend=backend, conv_tol=1e-11, diis_tol=1e-10).density_fit(
            "def2-universal-jkfit"
        )
        mf_mg.grids.coords = np.ascontiguousarray(ref_mf.grids.coords, dtype=np.float64)
        mf_mg.grids.weights = np.ascontiguousarray(ref_mf.grids.weights, dtype=np.float64)
        gm_mg = mf_mg.nuc_grad_method()
        gm_mg.grid_response = False
        gm_mg.auxbasis_response = True
        de_mg = gm_mg.kernel()
        ref_grad.grid_response = False
        de_ref_no_gr = ref_grad.kernel()
        assert np.max(np.abs(de_mg - de_ref_no_gr)) <= 1e-6


def test_geomopt_geometric_solver_mf_and_scanner():
    geometric = pytest.importorskip("geometric")
    _ = geometric
    geometric_solver = pytest.importorskip("pyscf.geomopt.geometric_solver")

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"

    # 1. Closed-shell distorted H2O B3LYP-D3(BJ)/def2-svp via both mf and GradScanner
    mol_h2o = _make_h2o(basis="def2-svp", distorted=True)
    mf_h2o = gpu4dft.RKS(
        mol_h2o, xc="b3lyp", dispersion="d3bj", backend=backend, grids_level=4, conv_tol=1e-10
    ).density_fit("def2-universal-jkfit")
    mol_opt_mf = geometric_solver.optimize(mf_h2o, maxsteps=50)

    g_scanner = mf_h2o.nuc_grad_method().as_scanner()
    mol_opt_scan = geometric_solver.optimize(g_scanner, maxsteps=50)
    assert g_scanner.converged is True
    assert g_scanner.de is not None
    assert np.max(np.abs(g_scanner.de)) <= 4.5e-4
    assert np.isfinite(g_scanner.e_tot)
    assert g_scanner.mol.natm == mol_h2o.natm

    # Verify bond lengths match between passing mf and passing GradScanner, and against PySCF
    c_mf = mol_opt_mf.atom_coords(unit="Angstrom")
    c_sc = mol_opt_scan.atom_coords(unit="Angstrom")
    oh_mf = np.linalg.norm(c_mf[1] - c_mf[0])
    oh_sc = np.linalg.norm(c_sc[1] - c_sc[0])
    assert abs(oh_mf - oh_sc) <= 1e-4

    # Open-shell OH radical UKS(CAM-B3LYP)/def2-svp via GradScanner + PySCF parity
    mol_oh = _make_oh(basis="def2-svp", bond_len=1.08)
    mf_oh = gpu4dft.UKS(
        mol_oh, xc="cam-b3lyp", backend=backend, grids_level=4, conv_tol=1e-10
    ).density_fit("def2-universal-jkfit")
    oh_scan = mf_oh.nuc_grad_method().as_scanner()
    mol_oh_g4d = geometric_solver.optimize(oh_scan, maxsteps=50)
    assert oh_scan.converged is True
    assert np.max(np.abs(oh_scan.de)) <= 4.5e-4

    ref_oh = pyscf_dft.UKS(mol_oh, xc="cam-b3lyp").density_fit("def2-universal-jkfit")
    ref_oh.grids.level = 4
    ref_oh.conv_tol = 1e-10
    mol_oh_ref = geometric_solver.optimize(ref_oh, maxsteps=50)

    r_g4d = float(np.linalg.norm(mol_oh_g4d.atom_coords(unit="Angstrom")[1] - mol_oh_g4d.atom_coords(unit="Angstrom")[0]))
    r_ref = float(np.linalg.norm(mol_oh_ref.atom_coords(unit="Angstrom")[1] - mol_oh_ref.atom_coords(unit="Angstrom")[0]))
    assert abs(r_g4d - r_ref) <= 1e-3


def test_geomopt_berny_solver_mf_and_scanner():
    berny = pytest.importorskip("berny")
    _ = berny
    berny_solver = pytest.importorskip("pyscf.geomopt.berny_solver")

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol_h2o = _make_h2o(basis="sto-3g", distorted=True)

    mf = gpu4dft.RKS(mol_h2o, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-10)
    e_init = float(mf.kernel())
    mol_opt_mf = berny_solver.optimize(mf, maxsteps=40)

    g_scanner = mf.nuc_grad_method().as_scanner()
    mol_opt_sc = berny_solver.optimize(g_scanner, maxsteps=40)
    assert g_scanner.converged is True
    assert g_scanner.e_tot < e_init
    assert np.max(np.abs(g_scanner.de)) <= 4.5e-4

    ref_mf = pyscf_dft.RKS(mol_h2o, xc="b3lyp")
    ref_mf.grids.level = 3
    ref_mf.conv_tol = 1e-10
    mol_opt_ref = berny_solver.optimize(ref_mf, maxsteps=40)

    for m_g4d in (mol_opt_mf, mol_opt_sc):
        c1 = m_g4d.atom_coords(unit="Angstrom")
        c2 = mol_opt_ref.atom_coords(unit="Angstrom")
        r1 = np.linalg.norm(c1[1] - c1[0])
        r2 = np.linalg.norm(c2[1] - c2[0])
        assert abs(r1 - r2) <= 1e-3


def test_scanner_dm0_reuse_and_isolation_from_base():
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol1 = _make_h2o(basis="def2-svp", shift_x=0.0)
    mol_shifted = _make_h2o(basis="def2-svp", shift_x=0.02)
    mol_shifted_2 = _make_h2o(basis="def2-svp", shift_x=-0.02)

    base = gpu4dft.RKS(
        mol1, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-10
    ).density_fit("def2-universal-jkfit")
    base_e = float(base.kernel())
    base_mol = base.mol
    base_eng = base._engine

    scf_scanner = base.as_scanner()
    grad_scanner = base.nuc_grad_method().as_scanner()

    # Warm step 1 on mol1 then small 0.02 A shift
    _ = scf_scanner(mol1)
    _ = scf_scanner(mol_shifted)
    iters_warm = int(scf_scanner.iterations)

    mf_cold = gpu4dft.RKS(
        mol_shifted, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-10
    ).density_fit("def2-universal-jkfit")
    _ = mf_cold.kernel()
    iters_cold = int(mf_cold.iterations)
    assert iters_warm < iters_cold or iters_warm <= max(4, iters_cold // 2)

    # GradScanner runs independently
    _ = grad_scanner(mol_shifted_2)

    assert base.mol is base_mol
    assert base.e_tot == base_e
    assert base._engine is base_eng
    assert scf_scanner.mol is not grad_scanner.mol

    # Different molecule nao/charge/atom count safely ignores mismatched dm0
    mol_nh3 = gto.M(
        atom=[
            ("N", (0.0, 0.0, 0.0)),
            ("H", (0.0, -0.94, 0.38)),
            ("H", (0.81, 0.47, 0.38)),
            ("H", (-0.81, 0.47, 0.38)),
        ],
        basis="def2-svp",
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    e_nh3 = scf_scanner(mol_nh3)
    assert scf_scanner.converged is True
    e_nh3_fresh = gpu4dft.RKS(
        mol_nh3, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-10
    ).density_fit("def2-universal-jkfit").kernel()
    assert abs(e_nh3 - e_nh3_fresh) <= 1e-9

    # Standalone MoleculeSpec scanner with SimpleAtoms / ndarray / str
    spec = gpu4dft.MoleculeSpec(
        basis=gpu4dft.basis_from_pyscf(mol1),
        atoms=gpu4dft.atoms_from_pyscf(mol1),
        nelectron=int(mol1.nelectron),
        aux_basis=gpu4dft.aux_basis_from_pyscf(mol1, "def2-universal-jkfit"),
        spin=0,
    )
    spec_mf = gpu4dft.RKS(spec, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-10).density_fit()
    spec_scan = spec_mf.as_scanner()
    e_spec1 = spec_scan(spec)
    assert abs(e_spec1 - base_e) <= 1e-9

    sa_shift = gpu4dft.SimpleAtoms(
        [8, 1, 1], np.asarray(mol_shifted.atom_coords(unit="Angstrom"), dtype=np.float64)
    )
    e_spec2 = spec_scan(sa_shift)
    assert abs(e_spec2 - mf_cold.e_tot) <= 1e-9

    coords_bohr = np.asarray(mol_shifted_2.atom_coords(unit="Bohr"), dtype=np.float64)
    e_spec3 = spec_scan(coords_bohr)
    assert np.isfinite(e_spec3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_config_signature_invalidation_set_geom_and_mol_reassign(backend: str):
    mol1 = _make_h2o(basis="sto-3g", shift_x=0.0)
    mf = gpu4dft.RKS(mol1, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-11)
    e1 = float(mf.kernel())
    sig1 = mf._setup_sig

    # In-place mutation of mol1 keeping id(mol1) identical
    coords2 = np.asarray(mol1.atom_coords(unit="Angstrom"), dtype=np.float64)
    coords2[1, 0] += 0.15
    mol1.set_geom_(coords2, unit="Angstrom", inplace=True)
    e2 = float(mf.kernel())
    assert mf._setup_sig != sig1

    mol2_fresh = gto.M(
        atom=[(mol1.atom_symbol(i), tuple(coords2[i])) for i in range(mol1.natm)],
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    e2_fresh = float(
        gpu4dft.RKS(mol2_fresh, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-11).kernel()
    )
    assert abs(e2 - e2_fresh) <= 1e-10
    assert abs(e2 - e1) > 1e-5

    # Reassigning mf.mol to a different molecule
    mol3 = _make_h2o(basis="sto-3g", distorted=True)
    mf.mol = mol3
    e3 = float(mf.kernel())
    e3_fresh = float(
        gpu4dft.RKS(mol3, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-11).kernel()
    )
    assert abs(e3 - e3_fresh) <= 1e-10


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kind", ["rks", "uks"])
def test_release_preserves_host_state_and_post_release_recovery(backend: str, kind: str):
    if kind == "rks":
        mol = _make_h2o(basis="sto-3g")
        mf = gpu4dft.RKS(mol, xc="b3lyp", backend=backend, grids_level=3, conv_tol=1e-11)
    else:
        mol = _make_oh(basis="sto-3g")
        mf = gpu4dft.UKS(
            mol, xc="cam-b3lyp", backend=backend, grids_level=3, conv_tol=1e-11
        ).density_fit("def2-universal-jkfit")

    e_ref = float(mf.kernel())
    dm_ref = mf.make_rdm1().copy()
    s2_ref = mf.spin_square()
    mu_ref = mf.dipole_moment().copy()
    g_ref = mf.nuc_grad_method().kernel().copy()

    mf.release()
    assert mf._engine is None
    assert mf._setup_sig is None
    assert mf.result is not None
    assert mf.converged is True
    assert mf.e_tot == e_ref
    np.testing.assert_allclose(mf.make_rdm1(), dm_ref, rtol=0.0, atol=1e-12)
    assert mf.spin_square() == s2_ref
    np.testing.assert_allclose(mf.dipole_moment(), mu_ref, rtol=0.0, atol=1e-12)

    # Post-release NuclearGradients recovery
    g_after = mf.nuc_grad_method().kernel()
    assert mf._engine is not None
    np.testing.assert_allclose(g_after, g_ref, rtol=0.0, atol=1e-9)

    if kind == "rks":
        # Post-release TDA and Hessian recovery
        mf.release()
        e_ex, _ = mf.TDA().kernel(nstates=2)
        assert len(e_ex) == 2 and np.all(np.isfinite(e_ex))
        mf.release()
        h = mf.Hessian(step_bohr=5e-3).kernel()
        assert h.shape == (mol.natm, mol.natm, 3, 3)


def test_grids_config_becke_radii_adjust_and_weight_cutoff():
    g = gpu4dft.GridsConfig()
    assert g.becke_radii_adjust is True
    assert g.weight_cutoff == pytest.approx(1e-14)

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _make_h2o(basis="sto-3g")
    mf = gpu4dft.RKS(mol, xc="b3lyp", backend=backend, grids_level=2, conv_tol=1e-11)
    e_default = float(mf.kernel())
    sig_default = mf._setup_sig
    npts_default = int(mf.result.grid_npoints)

    mf.grids.becke_radii_adjust = False
    e_no_adj = float(mf.kernel())
    assert mf._setup_sig != sig_default
    assert abs(e_no_adj - e_default) > 1e-7

    mf_fresh = gpu4dft.RKS(mol, xc="b3lyp", backend=backend, grids_level=2, conv_tol=1e-11)
    mf_fresh.grids.becke_radii_adjust = False
    e_fresh = float(mf_fresh.kernel())
    assert abs(e_no_adj - e_fresh) <= 1e-12

    mf.grids.weight_cutoff = 1e-4
    _ = mf.kernel()
    assert int(mf.result.grid_npoints) < npts_default


def test_scanner_negative_and_invalid_inputs():
    mol_h2o = _make_h2o(basis="sto-3g")
    mf_rks = gpu4dft.RKS(mol_h2o, xc="b3lyp", backend="host")
    rks_scf = mf_rks.as_scanner()
    rks_grad = mf_rks.nuc_grad_method().as_scanner()

    for bad in (
        None,
        "not_a_mole",
        "",
        123,
        np.zeros((2, 3)),
        np.zeros((3, 2)),
        np.zeros((3, 3, 1)),
        np.full((3, 3), np.nan),
    ):
        with pytest.raises((TypeError, ValueError)):
            rks_scf(bad)
        with pytest.raises((TypeError, ValueError)):
            rks_grad(bad)

    mol_oh = _make_oh(basis="sto-3g")
    with pytest.raises((ValueError, RuntimeError)):
        rks_scf(mol_oh)
    with pytest.raises((ValueError, RuntimeError)):
        rks_grad(mol_oh)

    # Preserve spin=None on UKS scanner so invalid spin parity on incoming Mole fails cleanly
    mf_uks = gpu4dft.UKS(mol_oh, xc="b3lyp", backend="host")
    assert mf_uks.spin is None
    uks_scf = mf_uks.as_scanner()
    uks_grad = mf_uks.nuc_grad_method().as_scanner()
    assert uks_scf.spin is None

    mol_bad_parity = _make_oh(basis="sto-3g")
    mol_bad_parity.spin = 0
    with pytest.raises((ValueError, RuntimeError)):
        uks_scf(mol_bad_parity)
    with pytest.raises((ValueError, RuntimeError)):
        uks_grad(mol_bad_parity)


def test_calculator_system_changes_empty_fast_path():
    ase = pytest.importorskip("ase")
    from ase.calculators.calculator import PropertyNotImplementedError

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    mol = _make_h2o(basis="def2-svp")
    coords_ang = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    atoms = ase.Atoms(symbols=["O", "H", "H"], positions=coords_ang)
    calc = gpu4dft.GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=backend,
        conv_tol=1e-10,
    )
    atoms.calc = calc

    # Sequence 1: energy -> forces -> energy (no redundant SCF)
    e1 = float(atoms.get_potential_energy())
    mf1 = calc.last_mf
    assert mf1 is not None
    iters1 = int(mf1.iterations)

    f1 = np.asarray(atoms.get_forces(), dtype=np.float64)
    assert calc.check_state(atoms) == []
    assert calc.last_mf is mf1
    assert calc.last_mf.iterations == iters1

    e2 = float(atoms.get_potential_energy())
    assert e2 == e1
    assert calc.last_mf is mf1

    from gpu4dft.ase_calculator import _BOHR_IN_ANGSTROM, _HARTREE_IN_EV

    g_direct = mf1.nuc_grad_method().kernel()
    np.testing.assert_allclose(f1, -g_direct * (_HARTREE_IN_EV / _BOHR_IN_ANGSTROM), atol=1e-7)

    # Sequence 2: forces first -> energy
    atoms2 = ase.Atoms(symbols=["O", "H", "H"], positions=coords_ang)
    calc2 = gpu4dft.GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=backend,
        conv_tol=1e-10,
    )
    atoms2.calc = calc2
    f2 = np.asarray(atoms2.get_forces(), dtype=np.float64)
    mf2 = calc2.last_mf
    e2b = float(atoms2.get_potential_energy())
    assert calc2.last_mf is mf2
    assert abs(e2b - e1) <= 1e-8
    np.testing.assert_allclose(f2, f1, atol=1e-7)

    # Unsupported property raises PropertyNotImplementedError
    with pytest.raises((PropertyNotImplementedError, NotImplementedError)):
        calc.calculate(atoms, properties=["stress"], system_changes=[])

    # Post-release fast path recovery when _engine was manually released
    calc.last_mf.release()
    assert calc.last_mf._engine is None
    calc.calculate(atoms, properties=["forces"], system_changes=[])
    np.testing.assert_allclose(calc.results["forces"], f1, atol=1e-7)

    # SimpleAtoms standalone fast path
    sa = gpu4dft.SimpleAtoms([8, 1, 1], coords_ang)
    sa_calc = gpu4dft.GPU4DFTCalculator(
        method="rks", xc="b3lyp", basis="sto-3g", backend=backend, conv_tol=1e-10
    )
    sa.calc = sa_calc
    e_sa1 = sa.get_potential_energy()
    mf_sa = sa_calc.last_mf
    f_sa = sa.get_forces()
    assert sa_calc.last_mf is mf_sa
    assert f_sa.shape == (3, 3) and np.isfinite(e_sa1)


def _cuda_free_bytes() -> int | None:
    try:
        cudart = ctypes.CDLL("libcudart.so")
        free_b = ctypes.c_size_t()
        total_b = ctypes.c_size_t()
        rc = cudart.cudaMemGetInfo(ctypes.byref(free_b), ctypes.byref(total_b))
        if rc == 0:
            return int(free_b.value)
    except OSError:
        return None
    return None


def test_calculator_vram_non_leak_release_and_trim_pool():
    gpu4dft.trim_device_pool()
    gpu4dft._core.trim_device_pool(-1)

    ase = pytest.importorskip("ase")
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"

    mol = _make_ho2(basis="def2-svp")
    pos0 = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    atoms = ase.Atoms(symbols=["H", "O", "O"], positions=pos0)
    calc = gpu4dft.GPU4DFTCalculator(
        method="uks",
        xc="b3lyp",
        basis="def2-svp",
        spin=1,
        dispersion="d3bj",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=backend,
        conv_tol=1e-10,
    )
    atoms.calc = calc

    mf_history = []
    for step in range(5):
        p = pos0.copy()
        p[0, 0] += 0.01 * step
        atoms.set_positions(p)
        _ = atoms.get_potential_energy()
        _ = atoms.get_forces()
        assert calc.last_mf is not None
        mf_history.append(calc.last_mf)
        if step >= 1:
            prev_mf = mf_history[step - 1]
            assert prev_mf._engine is None
            assert prev_mf.result is not None
            assert prev_mf.make_rdm1().shape == (2, mol.nao_nr(), mol.nao_nr())

    # reset() releases _engine while keeping last_mf (for dm0 reuse across ASE BFGS)
    calc.reset()
    assert calc.last_mf is not None
    assert calc.last_mf._engine is None
    assert calc.last_mf.result is not None

    # close() releases _engine AND clears last_mf
    calc.close()
    assert calc.last_mf is None
    assert calc._last_dm_key is None
    _ = atoms.get_potential_energy()
    assert calc.last_mf is not None
    calc.close()

    if backend == "cuda":
        # Switch molecule sizes and check repeated close() does not leak VRAM
        h2o = ase.Atoms("H2O", positions=_make_h2o("def2-svp").atom_coords(unit="Angstrom"))
        h2o_2 = h2o.copy()
        h2o_2.translate([3.5, 0.0, 0.0])
        dimer = h2o + h2o_2

        free_samples: list[int] = []
        for cycle in range(3):
            c = gpu4dft.GPU4DFTCalculator(
                method="rks",
                xc="b3lyp",
                basis="def2-svp",
                density_fit=True,
                backend="cuda",
                conv_tol=1e-9,
            )
            _ = c.get_potential_energy(h2o)
            _ = c.get_potential_energy(dimer)
            _ = c.get_potential_energy(h2o)
            c.close()
            f_b = _cuda_free_bytes()
            if f_b is not None:
                free_samples.append(f_b)
        if len(free_samples) >= 2:
            # Allow small driver/context jitter (< 32 MiB) across cycles
            assert free_samples[0] - free_samples[-1] < 32 * 1024 * 1024


def test_calculator_kwargs_auxbasis_types_and_charge_none():
    ase = pytest.importorskip("ase")
    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"

    mol = _make_h2o(basis="def2-svp")
    coords_ang = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    atoms = ase.Atoms("OH2", positions=coords_ang)

    # 1. All extra **kwargs forwarded cleanly
    calc = gpu4dft.GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        charge=None,
        backend=backend,
        conv_tol=1e-9,
        diis_tol=1e-8,
        diis_space=6,
        damping=0.05,
        level_shift=0.0,
        lindep_tol=1e-6,
        schwarz_cutoff=1e-12,
        fp32_threshold=1e-5,
        ozaki_slices=8,
        early_gemm_mode="fp64",
        incremental=False,
        use_cuda_graph=False,
        init_guess="hcore",
        grids_partition="becke",
        grids_prune=False,
    )
    atoms.info["charge"] = None
    e_ev = calc.get_potential_energy(atoms)
    assert np.isfinite(e_ev)
    mf = calc.last_mf
    assert mf is not None
    assert mf.diis_tol == pytest.approx(1e-8)
    assert mf.diis_space == 6
    assert mf.damping == pytest.approx(0.05)
    assert mf.level_shift == pytest.approx(0.0)
    assert mf.lindep_tol == pytest.approx(1e-6)
    assert mf.schwarz_cutoff == pytest.approx(1e-12)
    assert mf.fp32_threshold == pytest.approx(1e-5)
    assert mf.ozaki_slices == 8
    assert str(mf.early_gemm_mode).endswith("fp64")
    assert mf.incremental is False
    assert mf.use_cuda_graph is False
    assert mf.init_guess == "hcore"
    assert str(mf.grids.partition).lower().endswith("becke")
    assert mf.grids.prune is False

    # SimpleAtoms charge=None
    sa_none = gpu4dft.SimpleAtoms([8, 1, 1], coords_ang, charge=None)
    assert sa_none.info["charge"] == 0
    assert sa_none.copy().info["charge"] == 0

    # 2. dict and Basis auxbasis support
    from gpu4dft.ase_calculator import _HARTREE_IN_EV

    aux_dict = {"O": "def2-universal-jkfit", "H": "def2-svp-jkfit"}
    calc_dict = gpu4dft.GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis=aux_dict,
        backend=backend,
        conv_tol=1e-10,
    )
    e_dict = calc_dict.get_potential_energy(atoms) / _HARTREE_IN_EV
    e_dict_ref = (
        gpu4dft.RKS(mol, xc="b3lyp", backend=backend, conv_tol=1e-10)
        .density_fit(auxbasis=aux_dict)
        .kernel()
    )
    assert abs(e_dict - e_dict_ref) <= 1e-9

    aux_basis_obj = gpu4dft.aux_basis_from_pyscf(mol, "def2-universal-jkfit")
    calc_obj = gpu4dft.GPU4DFTCalculator(
        method="rks",
        xc="b3lyp",
        basis="def2-svp",
        density_fit=True,
        auxbasis=aux_basis_obj,
        backend=backend,
        conv_tol=1e-10,
    )
    e_obj = calc_obj.get_potential_energy(atoms) / _HARTREE_IN_EV
    e_obj_ref = (
        gpu4dft.RKS(mol, xc="b3lyp", backend=backend, conv_tol=1e-10)
        .density_fit(auxbasis=aux_basis_obj)
        .kernel()
    )
    assert abs(e_obj - e_obj_ref) <= 1e-9


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_calculator_methods_d3bj_ri_and_dm0_fallback_retry(method: str, monkeypatch):
    ase = pytest.importorskip("ase")
    from gpu4dft.ase_calculator import _BOHR_IN_ANGSTROM, _HARTREE_IN_EV

    backend = "cuda" if gpu4dft.backend_available("cuda") else "host"
    is_open = method in ("uhf", "uks")
    is_dft = method in ("rks", "uks")

    mol = _make_ho2(basis="def2-svp") if is_open else _make_h2o(basis="def2-svp")
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    coords_ang = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    zs = [int(mol.atom_charge(i)) for i in range(mol.natm)]

    xc = "b3lyp" if is_dft else "hf"
    atoms = ase.Atoms(symbols=symbols, positions=coords_ang)
    calc = gpu4dft.GPU4DFTCalculator(
        method=method,
        xc=xc,
        basis="def2-svp",
        spin=1 if is_open else 0,
        dispersion="d3bj",
        density_fit=True,
        auxbasis="def2-universal-jkfit",
        backend=backend,
        grids_level=4,
        conv_tol=1e-10,
    )
    atoms.calc = calc
    e_calc = float(atoms.get_potential_energy()) / _HARTREE_IN_EV
    g_calc = -np.asarray(atoms.get_forces(), dtype=np.float64) * (_BOHR_IN_ANGSTROM / _HARTREE_IN_EV)

    if method == "rhf":
        ref_mf = pyscf_scf.RHF(mol).density_fit("def2-universal-jkfit")
    elif method == "rks":
        ref_mf = pyscf_dft.RKS(mol, xc=xc).density_fit("def2-universal-jkfit")
        ref_mf.grids.level = 4
    elif method == "uhf":
        ref_mf = pyscf_scf.UHF(mol).density_fit("def2-universal-jkfit")
    else:
        ref_mf = pyscf_dft.UKS(mol, xc=xc).density_fit("def2-universal-jkfit")
        ref_mf.grids.level = 4
    ref_mf.conv_tol = 1e-11
    e_ref = float(ref_mf.kernel())
    ref_g = ref_mf.nuc_grad_method()
    if is_dft:
        ref_g.grid_response = True
    ref_g.auxbasis_response = True
    g_ref = np.asarray(ref_g.kernel(), dtype=np.float64)

    e_d3, g_d3 = gpu4dft.dftd3_energy_grad(zs, coords_bohr, xc=xc)
    assert abs(e_calc - (e_ref + e_d3)) <= (1e-6 if not is_dft else 5e-5)
    if method != "uks":
        assert np.max(np.abs(g_calc - (g_ref + g_d3))) <= 1e-5
    else:
        # Native GPU4DFT grids_level=4 (max_ang=302, Stratmann) vs PySCF (434, Becke) differs by ~1.6e-5 on HO2;
        # verify both loose PySCF parity and exact moving-grid central FD <= 1e-5 Eh/a0.
        assert np.max(np.abs(g_calc - (g_ref + g_d3))) <= 5e-5
        step_bohr = 1e-4
        step_ang = step_bohr / BOHR_PER_ANGSTROM
        g_fd = np.zeros_like(g_calc)
        for a in range(len(symbols)):
            for k in range(3):
                cp = coords_ang.copy()
                cm = coords_ang.copy()
                cp[a, k] += step_ang
                cm[a, k] -= step_ang
                ep = float(calc.get_potential_energy(ase.Atoms(symbols=symbols, positions=cp))) / _HARTREE_IN_EV
                em = float(calc.get_potential_energy(ase.Atoms(symbols=symbols, positions=cm))) / _HARTREE_IN_EV
                g_fd[a, k] = (ep - em) / (2.0 * step_bohr)
        assert np.max(np.abs(g_calc - g_fd)) <= 1e-5

    # dm0=None fallback retry when warm dm0 fails to converge (only tested once for RHF)
    if method == "rhf":
        orig_kernel = gpu4dft.scf._SCFBase.kernel
        calls: list[bool] = []

        def _flaky_kernel(self, dm0=None):
            calls.append(dm0 is not None)
            if dm0 is not None and len(calls) == 1:
                self.setup()
                res = orig_kernel(self, dm0=dm0)
                self.converged = False
                return res
            return orig_kernel(self, dm0=dm0)

        monkeypatch.setattr(gpu4dft.scf._SCFBase, "kernel", _flaky_kernel)
        pos_new = coords_ang.copy()
        pos_new[0, 0] += 0.01
        atoms.set_positions(pos_new)
        e_retry = float(atoms.get_potential_energy())
        assert np.isfinite(e_retry)
        assert calls == [True, False]
        assert calc.last_mf is not None and calc.last_mf.converged is True

        # If dm0=None also fails to converge, RuntimeError must be raised
        def _always_fail(self, dm0=None):
            self.setup()
            res = orig_kernel(self, dm0=dm0)
            self.converged = False
            return res

        monkeypatch.setattr(gpu4dft.scf._SCFBase, "kernel", _always_fail)
        pos_new2 = pos_new.copy()
        pos_new2[0, 0] += 0.01
        atoms.set_positions(pos_new2)
        with pytest.raises(RuntimeError, match="did not converge"):
            atoms.get_potential_energy()
