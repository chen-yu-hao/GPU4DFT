"""Tests for DFT-D3(BJ) dispersion energies, analytical gradients, and SCF/RKS integration."""

from __future__ import annotations

import numpy as np
import pytest
from pyscf import gto

import gpu4dft

dftd3_interface = pytest.importorskip("dftd3.interface")


def _backends() -> list[str]:
    return ["host", "cuda"] if gpu4dft.backend_available("cuda") else ["host"]


def _s22_water_dimer() -> tuple[list[int], np.ndarray]:
    # Bohr coordinates of water dimer
    z = [8, 1, 1, 8, 1, 1]
    coords_ang = np.array(
        [
            [-1.551007, -0.114520, 0.000000],
            [-1.934259, 0.762503, 0.000000],
            [-0.599677, 0.040712, 0.000000],
            [1.350625, 0.111469, 0.000000],
            [1.680398, -0.373741, -0.758561],
            [1.680398, -0.373741, 0.758561],
        ],
        dtype=np.float64,
    )
    coords_bohr = coords_ang / 0.529177210903
    return z, coords_bohr


def _formic_acid_dimer() -> tuple[list[str], np.ndarray]:
    syms = ["C", "O", "O", "H", "H", "C", "O", "O", "H", "H"]
    coords_ang = np.array(
        [
            [-1.889186, -0.181623, 0.000000],
            [-1.529057, 1.076479, 0.000000],
            [-1.168421, -1.160587, 0.000000],
            [-0.531849, 1.158367, 0.000000],
            [-2.976298, -0.305419, 0.000000],
            [1.889186, 0.181623, 0.000000],
            [1.529057, -1.076479, 0.000000],
            [1.168421, 1.160587, 0.000000],
            [0.531849, -1.158367, 0.000000],
            [2.976298, 0.305419, 0.000000],
        ],
        dtype=np.float64,
    )
    return syms, coords_ang / 0.529177210903


def _s_dftd3_ref(atomic_numbers: list[int], coords_bohr: np.ndarray, xc: str) -> tuple[float, np.ndarray]:
    xc_map = {"cam-b3lyp": "camb3lyp"}
    method = xc_map.get(xc.lower(), xc.lower())
    model = dftd3_interface.DispersionModel(
        np.asarray(atomic_numbers, dtype=np.int32),
        np.asarray(coords_bohr, dtype=np.float64),
    )
    param = dftd3_interface.RationalDampingParam(method=method)
    res = model.get_dispersion(param, grad=True)
    return float(res["energy"]), np.asarray(res["gradient"], dtype=np.float64)


@pytest.mark.parametrize(
    "xc",
    ["b3lyp", "pbe", "pbe0", "cam-b3lyp", "wb97x", "blyp", "bp86", "tpss", "hf"],
)
def test_dftd3_matches_sdftd3_and_fd(xc: str):
    z_w, c_w = _s22_water_dimer()
    e_got, g_got = gpu4dft.dftd3_energy_grad(z_w, c_w, xc=xc)
    e_ref, g_ref = _s_dftd3_ref(z_w, c_w, xc)
    assert abs(e_got - e_ref) <= 1e-8
    assert np.max(np.abs(g_got - g_ref)) <= 1e-7

    # Also test element symbol input + multi-atom CN chain-rule on formic acid dimer
    syms, c_f = _formic_acid_dimer()
    z_f = [6, 8, 8, 1, 1, 6, 8, 8, 1, 1]
    ef_got, gf_got = gpu4dft.dftd3_energy_grad(syms, c_f, xc=xc)
    ef_ref, gf_ref = _s_dftd3_ref(z_f, c_f, xc)
    assert abs(ef_got - ef_ref) <= 1e-8
    assert np.max(np.abs(gf_got - gf_ref)) <= 1e-7

    # Central finite difference check of analytical gradient
    h = 1e-5
    gf_fd = np.zeros_like(gf_got)
    for ia in range(len(syms)):
        for ix in range(3):
            cp = c_f.copy()
            cm = c_f.copy()
            cp[ia, ix] += h
            cm[ia, ix] -= h
            ep = gpu4dft.dftd3_energy(syms, cp, xc=xc)
            em = gpu4dft.dftd3_energy(syms, cm, xc=xc)
            gf_fd[ia, ix] = (ep - em) / (2.0 * h)
    np.testing.assert_allclose(gf_got, gf_fd, rtol=0.0, atol=1e-8)

    # Translational & rotational invariance
    np.testing.assert_allclose(np.sum(gf_got, axis=0), np.zeros(3), rtol=0.0, atol=1e-13)
    torque = np.sum(np.cross(c_f, gf_got), axis=0)
    np.testing.assert_allclose(torque, np.zeros(3), rtol=0.0, atol=1e-13)


def test_dftd3_wb97xd_not_aliased_and_mole_input():
    with pytest.raises(Exception):
        gpu4dft.dftd3_params("wb97x-d")
    with pytest.raises(Exception):
        gpu4dft.dftd3_params("wb97xd")

    mol = gto.M(
        atom="O 0.0 0.0 0.12; H 0.0 0.76 -0.47; H 0.08 -0.74 -0.44",
        basis="def2-svp",
        verbose=0,
    )
    e_mol, g_mol = gpu4dft.dftd3_energy_grad(mol, xc="b3lyp")
    z = [int(z_val) for z_val in mol.atom_charges()]
    c = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    e_arr, g_arr = gpu4dft.dftd3_energy_grad(z, c, xc="b3lyp")
    assert e_mol == pytest.approx(e_arr, abs=1e-14)
    np.testing.assert_allclose(g_mol, g_arr, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("backend", _backends())
@pytest.mark.parametrize("xc", ["b3lyp", "pbe", "pbe0", "hf"])
def test_rks_rhf_dispersion_d3bj_energy_and_grad(xc: str, backend: str):
    mol = gto.M(
        atom="O 0.0 0.0 0.12; H 0.0 0.76 -0.47; H 0.08 -0.74 -0.44",
        basis="6-31g*",
        cart=False,
        verbose=0,
    )
    if xc == "hf":
        mf0 = gpu4dft.RHF(mol, backend=backend, conv_tol=1e-12, diis_tol=1e-9).run()
        mf_d3 = gpu4dft.RHF(mol, backend=backend, dispersion="d3bj", conv_tol=1e-12, diis_tol=1e-9).run()
    else:
        mf0 = gpu4dft.RKS(mol, xc=xc, backend=backend, grids_level=3, conv_tol=1e-12, diis_tol=1e-9).run()
        mf_d3 = gpu4dft.RKS(mol, xc=xc, backend=backend, grids_level=3, dispersion="d3bj", conv_tol=1e-12, diis_tol=1e-9).run()

    e_disp_ref, g_disp_ref = gpu4dft.dftd3_energy_grad(mol, xc=xc)
    assert mf_d3.e_disp == pytest.approx(e_disp_ref, abs=1e-12)
    assert mf_d3.scf_summary["e_disp"] == pytest.approx(e_disp_ref, abs=1e-12)
    assert mf_d3.e_elec == pytest.approx(mf0.e_elec, abs=1e-11)
    assert mf_d3.e_tot == pytest.approx(mf0.e_tot + e_disp_ref, abs=1e-11)

    g0 = mf0.nuc_grad_method().kernel()
    g_d3 = mf_d3.nuc_grad_method().kernel()
    np.testing.assert_allclose(g_d3, g0 + g_disp_ref, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(np.sum(g_d3, axis=0), np.zeros(3), rtol=0.0, atol=1e-6)

    # Central FD check of total energy with dispersion="d3bj"
    if xc == "b3lyp":
        coords0 = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
        charges = [int(mol.atom_charge(i)) for i in range(mol.natm)]
        h = 1e-4
        g_fd = np.zeros_like(g_d3)
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
                ep = gpu4dft.RKS(mol_p, xc=xc, backend=backend, grids_level=3, dispersion="d3bj", conv_tol=1e-12, diis_tol=1e-9).kernel()
                em = gpu4dft.RKS(mol_m, xc=xc, backend=backend, grids_level=3, dispersion="d3bj", conv_tol=1e-12, diis_tol=1e-9).kernel()
                g_fd[ia, ix] = (ep - em) / (2.0 * h)
        np.testing.assert_allclose(g_d3, g_fd, rtol=0.0, atol=1e-5)
