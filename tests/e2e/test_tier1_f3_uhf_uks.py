"""Tier 1 E2E tests for Feature F3: Open-shell UHF/UKS SCF, <S^2>, and Direct nuclear gradients."""

from __future__ import annotations

import numpy as np
import pytest

from data.molecules import get_molecule
from oracles import (
    central_fd_grad,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


@pytest.mark.tier1
@pytest.mark.f3
@pytest.mark.gpu
def test_f3_uhf_doublet_scf_and_s2(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F3-01: UHF on doublet radicals matches PySCF <= 1e-6 Eh, <S^2> <= 1e-5, DM shape (2, nao, nao)."""
    from gpu4dft import UHF

    for name in ("oh_radical", "ch3_radical", "no_radical"):
        mol_data = get_molecule(name)
        mol = make_pyscf_mol(
            mol_data.pyscf_atom,
            basis="def2-svp",
            charge=mol_data.charge,
            spin=mol_data.spin,
        )
        nao = mol.nao_nr()

        ref = run_pyscf_scf(mol, "uhf", conv_tol=1e-11)
        assert ref["converged"], f"PySCF UHF did not converge for {name}"

        mf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-10)
        e_got = mf.kernel()

        assert mf.converged, f"GPU4DFT UHF did not converge for {name}"
        assert abs(e_got - ref["e_tot"]) <= 1e-6

        ss, mult = mf.spin_square()
        ss_ref, mult_ref = ref["spin_square"]
        assert abs(ss - ss_ref) <= 1e-5
        assert abs(mult - mult_ref) <= 1e-5

        dm = mf.make_rdm1()
        assert dm.shape == (2, nao, nao)


@pytest.mark.tier1
@pytest.mark.f3
@pytest.mark.gpu
def test_f3_uhf_triplet_scf_and_s2(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F3-02: UHF on triplet O2 and triplet CH2 matches PySCF <= 1e-6 Eh and <S^2> <= 1e-5."""
    from gpu4dft import UHF

    o2 = get_molecule("o2_triplet")
    ch2_triplet = [("C", (0.0, 0.0, 0.0)), ("H", (0.0, 0.99, 0.44)), ("H", (0.0, -0.99, 0.44))]

    cases = (
        ("o2_triplet", o2.pyscf_atom, o2.charge, o2.spin),
        ("ch2_triplet", ch2_triplet, 0, 2),
    )

    for name, atom_spec, charge, spin in cases:
        mol = make_pyscf_mol(atom_spec, basis="def2-svp", charge=charge, spin=spin)
        ref = run_pyscf_scf(mol, "uhf", conv_tol=1e-11)
        assert ref["converged"], f"PySCF UHF did not converge for {name}"

        mf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-10)
        e_got = mf.kernel()

        assert mf.converged, f"GPU4DFT UHF did not converge for {name}"
        assert abs(e_got - ref["e_tot"]) <= 1e-6

        ss, _ = mf.spin_square()
        assert abs(ss - ref["spin_square"][0]) <= 1e-5
        assert 2.0 <= ss < 2.2, f"Unexpected triplet <S^2>={ss:.6f} for {name}"


@pytest.mark.tier1
@pytest.mark.f3
@pytest.mark.gpu
def test_f3_uks_doublet_triplet_functionals(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F3-03: UKS (PBE, B3LYP, PBE0) on doublet OH and triplet O2 matches PySCF <= 1e-6 Eh & <S^2> <= 1e-5."""
    from gpu4dft import UKS

    for name in ("oh_radical", "o2_triplet"):
        mol_data = get_molecule(name)
        mol = make_pyscf_mol(
            mol_data.pyscf_atom,
            basis="def2-svp",
            charge=mol_data.charge,
            spin=mol_data.spin,
        )
        for xc in ("pbe", "b3lyp", "pbe0"):
            ref = run_pyscf_scf(mol, "uks", xc=xc, grids_level=3, conv_tol=1e-11)
            assert ref["converged"], f"PySCF UKS({xc}) failed on {name}"

            mf = UKS(mol, xc=xc, backend=gpu4dft_backend, conv_tol=1e-10)
            match_grids(mf, ref["mf"])
            e_got = mf.kernel()

            assert mf.converged, f"GPU4DFT UKS({xc}) failed on {name}"
            assert abs(e_got - ref["e_tot"]) <= 1e-6
            ss, _ = mf.spin_square()
            assert abs(ss - ref["spin_square"][0]) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f3
@pytest.mark.gpu
def test_f3_uhf_uks_charged_radicals(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T1-F3-04: Charged open-shell radicals [H2O]+ and [O2]- match PySCF UHF & UKS(B3LYP)."""
    from gpu4dft import UHF, UKS

    for name in ("h2o_cation", "o2_anion"):
        mol_data = get_molecule(name)
        mol = make_pyscf_mol(
            mol_data.pyscf_atom,
            basis="6-31+g*",
            charge=mol_data.charge,
            spin=mol_data.spin,
        )

        ref_uhf = run_pyscf_scf(mol, "uhf", conv_tol=1e-11)
        assert ref_uhf["converged"]
        mf_uhf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-10)
        e_uhf = mf_uhf.kernel()
        assert mf_uhf.converged
        assert abs(e_uhf - ref_uhf["e_tot"]) <= 1e-6
        assert abs(mf_uhf.spin_square()[0] - ref_uhf["spin_square"][0]) <= 1e-5

        ref_uks = run_pyscf_scf(mol, "uks", xc="b3lyp", grids_level=3, conv_tol=1e-11)
        assert ref_uks["converged"]
        mf_uks = UKS(mol, xc="b3lyp", backend=gpu4dft_backend, conv_tol=1e-10)
        match_grids(mf_uks, ref_uks["mf"])
        e_uks = mf_uks.kernel()
        assert mf_uks.converged
        assert abs(e_uks - ref_uks["e_tot"]) <= 1e-6
        assert abs(mf_uks.spin_square()[0] - ref_uks["spin_square"][0]) <= 1e-5


@pytest.mark.tier1
@pytest.mark.f3
@pytest.mark.gpu
def test_f3_uhf_uks_direct_gradients(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T1-F3-05: Direct UHF & UKS(PBE, B3LYP) gradients on HO2 & O2 match PySCF and central FD <= 1e-5."""
    from gpu4dft import UHF, UKS

    ho2_distorted = [
        ("O", (0.0000, 0.0000, 0.0000)),
        ("O", (1.3800, 0.0300, 0.0000)),
        ("H", (-0.2100, 0.9900, 0.0400)),
    ]
    o2_stretched = [
        ("O", (0.0000, 0.0000, -0.6750)),
        ("O", (0.0000, 0.0000, 0.6750)),
    ]

    cases = (
        ("ho2", ho2_distorted, 0, 1),
        ("o2", o2_stretched, 0, 2),
    )

    for name, atom_spec, charge, spin in cases:
        mol = make_pyscf_mol(atom_spec, basis="def2-svp", charge=charge, spin=spin)
        symbols = [sym for sym, _ in atom_spec]
        coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

        for method, xc in (("uhf", None), ("uks", "pbe"), ("uks", "b3lyp")):
            ref = run_pyscf_scf(mol, method, xc=xc, grids_level=5, conv_tol=1e-11)
            assert ref["converged"]
            g_pyscf = run_pyscf_grad(ref["mf"], grid_response=True)

            if method == "uhf":
                mf = UHF(mol, backend=gpu4dft_backend, conv_tol=1e-11, diis_tol=1e-8)
            else:
                mf = UKS(
                    mol,
                    xc=xc,
                    backend=gpu4dft_backend,
                    grids_level=5,
                    conv_tol=1e-11,
                    diis_tol=1e-8,
                )
            mf.kernel()
            assert mf.converged
            g_ana = mf.nuc_grad_method().kernel()

            assert np.max(np.abs(g_ana)) > 1e-3, f"Expected non-zero gradient on {name}"
            diff_pyscf = float(np.max(np.abs(g_ana - g_pyscf)))
            assert diff_pyscf <= 1e-5, f"{name} {method}({xc}) vs PySCF diff {diff_pyscf:.3e} > 1e-5"
            assert translational_invariance_norm(g_ana) <= 1e-6

            if name == "ho2" and method in ("uhf", "uks") and xc in (None, "b3lyp"):
                def _energy_fn(c_bohr: np.ndarray, _m: str = method, _xc: str | None = xc) -> float:
                    mol_d = make_pyscf_mol(
                        [(symbols[i], tuple(c_bohr[i])) for i in range(len(symbols))],
                        basis="def2-svp",
                        charge=charge,
                        spin=spin,
                        unit="Bohr",
                    )
                    if _m == "uhf":
                        mf_d = UHF(mol_d, backend=gpu4dft_backend, conv_tol=1e-11, diis_tol=1e-8)
                    else:
                        mf_d = UKS(
                            mol_d,
                            xc=_xc,
                            backend=gpu4dft_backend,
                            grids_level=5,
                            conv_tol=1e-11,
                            diis_tol=1e-8,
                        )
                    e_d = mf_d.kernel()
                    assert mf_d.converged
                    return float(e_d)

                g_fd = central_fd_grad(_energy_fn, coords_bohr, step=1e-4)
                diff_fd = float(np.max(np.abs(g_ana - g_fd)))
                assert diff_fd <= 1e-5, f"{name} {method}({xc}) vs FD diff {diff_fd:.3e} > 1e-5"
