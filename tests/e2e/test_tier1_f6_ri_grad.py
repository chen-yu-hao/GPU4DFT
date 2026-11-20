"""Tier 1 E2E tests for Feature F6: RI-JK Analytical Nuclear Gradients (T1-F6-01..05)."""

from __future__ import annotations

import numpy as np
import pytest

from data.molecules import get_molecule, get_water_cluster_coords
from oracles import (
    central_fd_grad,
    rotational_invariance_norm,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


@pytest.mark.tier1
@pytest.mark.f6
@pytest.mark.gpu
def test_f6_ri_jk_grad_closed_shell(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F6-01: Closed-shell RI-JK SCF (matched grid <=1e-6) & native grad (vs FD <=1e-5, PySCF, trans)."""
    from gpu4dft import RHF, RKS

    auxbasis = "def2-universal-jkfit"
    cases = [
        ("ch2o", "def2-svp", "rhf", None),
        ("ch2o", "def2-svp", "rks", "pbe"),
        ("ch2o", "def2-svp", "rks", "b3lyp"),
        ("h2o_dimer", "6-31g*", "rks", "pbe0"),
    ]
    for mol_name, basis, method, xc in cases:
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis=basis)

        ref = run_pyscf_scf(
            mol,
            method,
            xc=xc,
            density_fit=True,
            auxbasis=auxbasis,
            grids_level=5,
            conv_tol=1e-11,
        )
        assert ref["converged"]
        g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

        if method == "rhf":
            mf = RHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
                auxbasis=auxbasis
            )
            mf.kernel()
            assert mf.converged
            g_got = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)
            assert g_got.shape == (mol.natm, 3)
            assert np.max(np.abs(g_got - g_ref)) <= 1e-5
            assert translational_invariance_norm(g_got) <= 1e-6

            ref_4c = run_pyscf_scf(mol, "rhf", density_fit=False, conv_tol=1e-11)
            g_4c = run_pyscf_grad(ref_4c["mf"])
            assert np.max(np.abs(g_got - g_ref)) < np.max(np.abs(g_got - g_4c))
        else:
            mf_scf = RKS(
                mol, xc=xc or "b3lyp", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf_scf, ref["mf"])
            e_got = float(mf_scf.kernel())
            assert mf_scf.converged
            assert abs(e_got - ref["e_tot"]) <= 1e-6

            # Native grids_level=5 analytical gradient (no match_grids so grid_response=True is active)
            syms = spec.symbols
            coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

            def _build_native_rks(
                c_bohr: np.ndarray, _syms=syms, _basis=basis, _xc=xc or "b3lyp"
            ):
                m_s = make_pyscf_mol(
                    [(_syms[i], tuple(c_bohr[i])) for i in range(len(_syms))],
                    basis=_basis,
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

            mf_grad = _build_native_rks(coords_bohr)
            mf_grad.kernel()
            assert mf_grad.converged
            g_got = np.asarray(mf_grad.nuc_grad_method().kernel(), dtype=np.float64)
            assert g_got.shape == (mol.natm, 3)
            assert translational_invariance_norm(g_got) <= 1e-6
            assert np.max(np.abs(g_got - g_ref)) <= 1e-4

            def _e_fn_rks(c_bohr: np.ndarray, _b=_build_native_rks) -> float:
                m_d = _b(c_bohr)
                e_d = float(m_d.kernel())
                assert m_d.converged
                return e_d

            g_fd = central_fd_grad(_e_fn_rks, coords_bohr, step=1e-4)
            assert np.max(np.abs(g_got - g_fd)) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f6
@pytest.mark.gpu
def test_f6_ri_jk_grad_open_shell(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F6-02: Open-shell UHF/UKS RI-JK SCF (matched <=1e-6) & native grad (vs FD <=1e-5, trans <=1e-6)."""
    from gpu4dft import UHF, UKS

    auxbasis = "def2-universal-jkfit"
    cases = [
        ("ho2_radical", "uhf", None),
        ("o2_triplet", "uhf", None),
        ("ho2_radical", "uks", "b3lyp"),
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
            grids_level=5,
            conv_tol=1e-11,
        )
        assert ref["converged"]
        g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

        if method == "uhf":
            mf = UHF(mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10).density_fit(
                auxbasis=auxbasis
            )
            mf.kernel()
            assert mf.converged
            g_got = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)
            assert g_got.shape == (mol.natm, 3)
            assert np.max(np.abs(g_got - g_ref)) <= 1e-5
            assert translational_invariance_norm(g_got) <= 1e-6
        else:
            mf_scf = UKS(
                mol, xc=xc or "b3lyp", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf_scf, ref["mf"])
            e_got = float(mf_scf.kernel())
            assert mf_scf.converged
            assert abs(e_got - ref["e_tot"]) <= 1e-6
            assert ref["spin_square"] is not None
            assert abs(mf_scf.spin_square()[0] - ref["spin_square"][0]) <= 1e-5

            syms = spec.symbols
            coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

            def _build_native_uks(
                c_bohr: np.ndarray, _syms=syms, _spec=spec, _xc=xc or "b3lyp"
            ):
                m_s = make_pyscf_mol(
                    [(_syms[i], tuple(c_bohr[i])) for i in range(len(_syms))],
                    basis="def2-svp",
                    charge=_spec.charge,
                    spin=_spec.spin,
                    unit="Bohr",
                )
                return UKS(
                    m_s,
                    xc=_xc,
                    backend=gpu4dft_backend,
                    precision="fp64",
                    grids_level=5,
                    conv_tol=1e-11,
                    diis_tol=1e-8,
                ).density_fit(auxbasis=auxbasis)

            mf_grad = _build_native_uks(coords_bohr)
            mf_grad.kernel()
            assert mf_grad.converged
            g_got = np.asarray(mf_grad.nuc_grad_method().kernel(), dtype=np.float64)
            assert g_got.shape == (mol.natm, 3)
            assert translational_invariance_norm(g_got) <= 1e-6
            assert np.max(np.abs(g_got - g_ref)) <= 1e-4

            def _e_fn_uks(c_bohr: np.ndarray, _b=_build_native_uks) -> float:
                m_d = _b(c_bohr)
                e_d = float(m_d.kernel())
                assert m_d.converged
                return e_d

            g_fd = central_fd_grad(_e_fn_uks, coords_bohr, step=1e-4)
            assert np.max(np.abs(g_got - g_fd)) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f6
@pytest.mark.gpu
def test_f6_ri_jk_grad_rsh(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F6-03: RSH (CAM-B3LYP, wB97X) RKS/UKS RI-JK SCF (matched <=1e-6) & native grad (vs FD <=1e-5)."""
    from gpu4dft import RKS, UKS

    auxbasis = "def2-universal-jkfit"
    for xc in ("cam-b3lyp", "wb97x"):
        for mol_name, method in (("ch2o", "rks"), ("ho2_radical", "uks")):
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
                grids_level=5,
                conv_tol=1e-11,
            )
            assert ref["converged"]
            g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

            cls = RKS if method == "rks" else UKS
            mf_scf = cls(
                mol, xc=xc, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
            ).density_fit(auxbasis=auxbasis)
            match_grids(mf_scf, ref["mf"])
            e_got = float(mf_scf.kernel())
            assert mf_scf.converged
            assert abs(e_got - ref["e_tot"]) <= 1e-6
            if method == "uks":
                assert ref["spin_square"] is not None
                assert abs(mf_scf.spin_square()[0] - ref["spin_square"][0]) <= 1e-5

            syms = spec.symbols
            coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

            def _build_native(
                c_bohr: np.ndarray, _cls=cls, _syms=syms, _spec=spec, _xc=xc
            ):
                m_s = make_pyscf_mol(
                    [(_syms[i], tuple(c_bohr[i])) for i in range(len(_syms))],
                    basis="def2-svp",
                    charge=_spec.charge,
                    spin=_spec.spin,
                    unit="Bohr",
                )
                return _cls(
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


@pytest.mark.tier1
@pytest.mark.f6
@pytest.mark.gpu
def test_f6_ri_jk_grad_vs_finite_diff(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F6-04: RI-JK analytical gradients match central FD of RI-JK energy (proves auxbasis_response)."""
    from gpu4dft import RHF, RKS, UKS

    auxbasis = "def2-universal-jkfit"
    cases = [
        ("h2o", "rhf", None),
        ("h2o", "rks", "b3lyp"),
        ("ho2_radical", "uks", "wb97x"),
    ]
    for mol_name, method, xc in cases:
        spec = get_molecule(mol_name)
        coords_bohr = spec.coords_bohr.copy()
        coords_bohr[0, 0] += 0.04
        coords_bohr[1, 2] -= 0.03
        syms = spec.symbols

        def _build_mf(c_bohr: np.ndarray, _spec=spec, _syms=syms, _method=method, _xc=xc):
            atom_list = [(_syms[i], tuple(c_bohr[i])) for i in range(len(_syms))]
            mol = make_pyscf_mol(
                atom_list, basis="6-31g*", charge=_spec.charge, spin=_spec.spin, unit="Bohr"
            )
            if _method == "rhf":
                return RHF(
                    mol, backend=gpu4dft_backend, precision="fp64", conv_tol=1e-11
                ).density_fit(auxbasis=auxbasis)
            cls = RKS if _method == "rks" else UKS
            return cls(
                mol,
                xc=_xc or "b3lyp",
                backend=gpu4dft_backend,
                precision="fp64",
                grids_level=4,
                conv_tol=1e-11,
            ).density_fit(auxbasis=auxbasis)

        mf0 = _build_mf(coords_bohr)
        mf0.kernel()
        assert mf0.converged
        g_an = np.asarray(mf0.nuc_grad_method().kernel(), dtype=np.float64)

        def _energy_fn(c_bohr: np.ndarray, _builder=_build_mf) -> float:
            mf_s = _builder(c_bohr)
            e_s = mf_s.kernel()
            assert mf_s.converged
            return float(e_s)

        g_fd = central_fd_grad(_energy_fn, coords_bohr, step=1e-4)
        assert np.max(np.abs(g_an - g_fd)) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f6
@pytest.mark.gpu
def test_f6_ri_jk_grad_translational_invariance(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F6-05: Asymmetric cluster RI-RKS(B3LYP) gradient satisfies |sum_A g_A|_inf <= 1e-6 Eh/a0."""
    from gpu4dft import RKS

    atoms = get_water_cluster_coords(2, 2, 1)
    mol = make_pyscf_mol(atoms, basis="def2-svp")
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    assert coords_bohr.shape == (12, 3)

    mf = RKS(
        mol,
        xc="b3lyp",
        backend=gpu4dft_backend,
        precision="fp64",
        grids_level=3,
        conv_tol=1e-10,
    ).density_fit(auxbasis="def2-universal-jkfit")
    mf.kernel()
    assert mf.converged

    grad = np.asarray(mf.nuc_grad_method().kernel(), dtype=np.float64)
    assert grad.shape == (12, 3)
    assert translational_invariance_norm(grad) <= 1e-6
    assert rotational_invariance_norm(coords_bohr, grad) <= 1e-6
