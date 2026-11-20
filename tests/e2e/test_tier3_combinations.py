"""Tier 3 cross-feature pairwise and multi-feature combination E2E tests (T3-01..10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import BOHR_TO_ANGSTROM, HARTREE_TO_KCAL_MOL, track_peak_vram_mib
from data.bh76_loader import get_bh76_openshell_reactions, get_bh76_species
from data.molecules import get_molecule
from data.s22_loader import get_s22_complex
from oracles import (
    central_fd_grad,
    rotational_invariance_norm,
    run_dftd3_bj,
    run_pyscf_grad,
    run_pyscf_scf,
    translational_invariance_norm,
)


def _run_geometric_opt(
    mol_init: Any,
    grad_scanner: Any,
    work_dir: Path,
    *,
    transition: bool = False,
    maxiter: int = 100,
) -> tuple[Any, float, np.ndarray]:
    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from pyscf import gto

    symbols = [mol_init.atom_symbol(i) for i in range(mol_init.natm)]
    coords_ang = np.asarray(mol_init.atom_coords(unit="Angstrom"), dtype=np.float64)

    gmol = geometric.molecule.Molecule()
    gmol.elem = symbols
    gmol.xyzs = [coords_ang]

    class _ScannerEngine(geometric.engine.Engine):
        def calc_new(self, coords: np.ndarray, dirname: str) -> dict[str, Any]:
            c_ang = np.asarray(coords, dtype=np.float64).reshape(-1, 3) * BOHR_TO_ANGSTROM
            mol_step = gto.M(
                atom=[(symbols[i], tuple(c_ang[i])) for i in range(len(symbols))],
                basis=mol_init.basis,
                charge=mol_init.charge,
                spin=mol_init.spin,
                unit="Angstrom",
                cart=False,
                verbose=0,
            )
            e_tot, grad = grad_scanner(mol_step)
            return {
                "energy": float(e_tot),
                "gradient": np.asarray(grad, dtype=np.float64).ravel(),
            }

    work_dir.mkdir(parents=True, exist_ok=True)
    input_xyz = work_dir / "opt_init.xyz"
    lines = [str(len(symbols)), "geometric opt init"]
    for sym, xyz in zip(symbols, coords_ang):
        lines.append(f"{sym} {xyz[0]:.10f} {xyz[1]:.10f} {xyz[2]:.10f}")
    input_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")

    engine = _ScannerEngine(gmol)
    m_opt = geometric.optimize.run_optimizer(
        customengine=engine,
        input=str(input_xyz),
        transition=bool(transition),
        maxiter=int(maxiter),
        prefix=str(work_dir / "geom_opt"),
    )
    final_ang = np.asarray(m_opt.xyzs[-1], dtype=np.float64)
    mol_final = gto.M(
        atom=[(symbols[i], tuple(final_ang[i])) for i in range(len(symbols))],
        basis=mol_init.basis,
        charge=mol_init.charge,
        spin=mol_init.spin,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    e_final, g_final = grad_scanner(mol_final)
    return mol_final, float(e_final), np.asarray(g_final, dtype=np.float64)


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.gpu
def test_t3_uks_camb3lyp_d3bj_rijk_grad_and_fd(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T3-01: UKS(CAM-B3LYP-D3(BJ)) + RI-JK SCF (matched <=1e-6) & native lvl5 grad (vs FD <=1e-5)."""
    from gpu4dft import UKS

    atoms_oh_ch4 = [
        ("C", (0.0000, 0.0000, 0.0000)),
        ("H", (0.6291, 0.6291, 0.6291)),
        ("H", (-0.6291, -0.6291, 0.6291)),
        ("H", (-0.6291, 0.6291, -0.6291)),
        ("H", (0.6291, -0.6291, -0.6291)),
        ("O", (2.6500, 0.1200, -0.0500)),
        ("H", (1.7200, 0.0500, 0.0200)),
    ]
    mol = make_pyscf_mol(atoms_oh_ch4, basis="def2-svp", charge=0, spin=1)
    auxbasis = "def2-universal-jkfit"

    # 1. Exact matched-grid SCF comparison vs PySCF DF + s-dftd3
    ref = run_pyscf_scf(
        mol, "uks", xc="cam-b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
    )
    g_ref_xc = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    z_nums = mol.atom_charges()
    e_d3, g_d3 = run_dftd3_bj(z_nums, coords_bohr, xc="cam-b3lyp")

    mf_scf = UKS(
        mol, xc="cam-b3lyp", dispersion="d3bj", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    match_grids(mf_scf, ref["mf"])
    e_got = float(mf_scf.kernel())
    assert mf_scf.converged
    assert abs(e_got - (ref["e_tot"] + e_d3)) <= 1e-6

    ss_got, _ = mf_scf.spin_square()
    assert ref["spin_square"] is not None
    assert abs(ss_got - ref["spin_square"][0]) <= 1e-5

    # 2. Native grids_level=5 analytical gradient vs moving-grid central FD, PySCF+D3, and trans invariance
    syms = [mol.atom_symbol(i) for i in range(mol.natm)]

    def _build_native(c_bohr: np.ndarray):
        m_s = make_pyscf_mol(
            [(syms[i], tuple(c_bohr[i])) for i in range(len(syms))],
            basis="def2-svp",
            charge=0,
            spin=1,
            unit="Bohr",
        )
        return UKS(
            m_s,
            xc="cam-b3lyp",
            dispersion="d3bj",
            backend=gpu4dft_backend,
            precision="fp64",
            grids_level=5,
            conv_tol=1e-11,
            diis_tol=1e-8,
        ).density_fit(auxbasis=auxbasis)

    mf_nat = _build_native(coords_bohr)
    mf_nat.kernel()
    assert mf_nat.converged
    g_nat = np.asarray(mf_nat.nuc_grad_method().kernel(), dtype=np.float64)
    assert translational_invariance_norm(g_nat) <= 1e-6
    assert np.max(np.abs(g_nat - (g_ref_xc + g_d3))) <= 1e-4

    def _e_fn(c_bohr: np.ndarray) -> float:
        m_d = _build_native(c_bohr)
        e_d = float(m_d.kernel())
        assert m_d.converged
        return e_d

    g_fd = central_fd_grad(_e_fn, coords_bohr, step=1e-4)
    assert np.max(np.abs(g_nat - g_fd)) <= 1e-5


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.gpu
def test_t3_uks_wb97x_d3bj_rijk_grad_invariance(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T3-02: UKS(wB97X-D3(BJ)) + RI-JK SCF (matched <=1e-6) & native lvl5 grad (FD <=1e-5, invariants)."""
    from gpu4dft import UKS

    atoms_ho2_h2o = [
        ("O", (0.0000, 0.0000, 0.0000)),
        ("O", (1.3300, 0.0000, 0.0000)),
        ("H", (-0.2300, 0.9450, 0.0000)),
        ("O", (-0.6500, 2.7200, 0.1500)),
        ("H", (-1.5800, 2.9500, 0.0500)),
        ("H", (-0.1800, 3.3200, 0.7200)),
    ]
    mol = make_pyscf_mol(atoms_ho2_h2o, basis="def2-svp", charge=0, spin=1)
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(
        mol, "uks", xc="wb97x", density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
    )
    g_ref_xc = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)
    e_d3, g_d3 = run_dftd3_bj(mol.atom_charges(), coords_bohr, xc="wb97x")

    mf_scf = UKS(
        mol, xc="wb97x", dispersion="d3bj", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    match_grids(mf_scf, ref["mf"])
    e_got = float(mf_scf.kernel())
    assert mf_scf.converged
    assert abs(e_got - (ref["e_tot"] + e_d3)) <= 1e-6
    ss_got, _ = mf_scf.spin_square()
    assert ref["spin_square"] is not None
    assert abs(ss_got - ref["spin_square"][0]) <= 1e-5

    # Native grids_level=5 analytical gradient, invariants, PySCF+D3 parity, and central FD
    syms = [mol.atom_symbol(i) for i in range(mol.natm)]

    def _build_native(c_bohr: np.ndarray):
        m_s = make_pyscf_mol(
            [(syms[i], tuple(c_bohr[i])) for i in range(len(syms))],
            basis="def2-svp",
            charge=0,
            spin=1,
            unit="Bohr",
        )
        return UKS(
            m_s,
            xc="wb97x",
            dispersion="d3bj",
            backend=gpu4dft_backend,
            precision="fp64",
            grids_level=5,
            conv_tol=1e-11,
            diis_tol=1e-8,
        ).density_fit(auxbasis=auxbasis)

    mf_nat = _build_native(coords_bohr)
    mf_nat.kernel()
    assert mf_nat.converged
    g_nat = np.asarray(mf_nat.nuc_grad_method().kernel(), dtype=np.float64)
    assert translational_invariance_norm(g_nat) <= 1e-6
    assert rotational_invariance_norm(coords_bohr, g_nat) <= 1e-6
    assert np.max(np.abs(g_nat - (g_ref_xc + g_d3))) <= 1e-4

    def _e_fn(c_bohr: np.ndarray) -> float:
        m_d = _build_native(c_bohr)
        e_d = float(m_d.kernel())
        assert m_d.converged
        return e_d

    g_fd = central_fd_grad(_e_fn, coords_bohr, step=1e-4)
    assert np.max(np.abs(g_nat - g_fd)) <= 1e-5


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f8
@pytest.mark.gpu
def test_t3_rks_b3lyp_d3bj_rijk_scanner_s22_opt(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T3-03: S22 Formic_acid_dimer B3LYP-D3(BJ)/def2-svp RI-JK .as_scanner() opt vs PySCF."""
    from gpu4dft import RKS

    cpx = get_s22_complex("Formic_acid_dimer")
    mol_d = make_pyscf_mol(cpx.dimer, basis="def2-svp")
    mol_a = make_pyscf_mol(cpx.monomer_a, basis="def2-svp")
    mol_b = make_pyscf_mol(cpx.monomer_b, basis="def2-svp")
    auxbasis = "def2-universal-jkfit"

    mf_d = RKS(
        mol_d, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mol_d_opt, e_d_opt, g_d_opt = _run_geometric_opt(
        mol_d, mf_d.nuc_grad_method().as_scanner(), tmp_path / "fad_dimer", maxiter=100
    )
    assert np.max(np.abs(g_d_opt)) <= 4.5e-4

    mf_a = RKS(
        mol_a, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mol_a_opt, e_a_opt, g_a_opt = _run_geometric_opt(
        mol_a, mf_a.nuc_grad_method().as_scanner(), tmp_path / "fad_monA", maxiter=80
    )
    assert np.max(np.abs(g_a_opt)) <= 4.5e-4

    mf_b = RKS(
        mol_b, xc="b3lyp", dispersion="d3bj", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    mol_b_opt, e_b_opt, g_b_opt = _run_geometric_opt(
        mol_b, mf_b.nuc_grad_method().as_scanner(), tmp_path / "fad_monB", maxiter=80
    )
    assert np.max(np.abs(g_b_opt)) <= 4.5e-4

    e_int_g4d = (e_d_opt - e_a_opt - e_b_opt) * HARTREE_TO_KCAL_MOL

    def _pyscf_e(m_x) -> float:
        res = run_pyscf_scf(m_x, "rks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=4)
        e_d3, _ = run_dftd3_bj(m_x.atom_charges(), np.asarray(m_x.atom_coords(unit="Bohr")), xc="b3lyp")
        return float(res["e_tot"] + e_d3)

    e_int_pyscf = (_pyscf_e(mol_d_opt) - _pyscf_e(mol_a_opt) - _pyscf_e(mol_b_opt)) * HARTREE_TO_KCAL_MOL
    assert abs(e_int_g4d - e_int_pyscf) <= 0.05


@pytest.mark.tier3
@pytest.mark.f2
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f8
@pytest.mark.gpu
def test_t3_rks_wb97x_d3bj_rijk_ase_disp_dimer_opt(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T3-04: S22 Benzene-methane_complex ASE BFGS opt with wB97X-D3(BJ) + RI-JK vs PySCF."""
    from ase.optimize import BFGS
    from ase.units import Bohr as ASE_BOHR
    from ase.units import Hartree as ASE_HARTREE
    from gpu4dft import GPU4DFTCalculator

    cpx = get_s22_complex("Benzene-methane_complex")
    auxbasis = "def2-universal-jkfit"
    fmax_ev_ang = 4.5e-4 * (ASE_HARTREE / ASE_BOHR)

    def _opt_ase(atoms_in, tag: str):
        at = atoms_in.copy()
        at.calc = GPU4DFTCalculator(
            method="rks",
            xc="wb97x",
            basis="def2-svp",
            dispersion="d3bj",
            density_fit=True,
            auxbasis=auxbasis,
            backend=gpu4dft_backend,
            grids_level=4,
            conv_tol=1e-10,
        )
        opt = BFGS(at, logfile=str(tmp_path / f"{tag}.log"))
        converged = opt.run(fmax=fmax_ev_ang, steps=120)
        assert converged
        forces = at.get_forces()
        grad_eh_bohr = -forces * (ASE_BOHR / ASE_HARTREE)
        assert np.max(np.abs(grad_eh_bohr)) <= 4.5e-4
        e_eh = float(at.get_potential_energy()) / ASE_HARTREE
        return at, e_eh

    dimer_opt, ed_g4d = _opt_ase(cpx.dimer, "bm_dimer")
    mona_opt, ea_g4d = _opt_ase(cpx.monomer_a, "bm_monA")
    monb_opt, eb_g4d = _opt_ase(cpx.monomer_b, "bm_monB")
    e_int_g4d = (ed_g4d - ea_g4d - eb_g4d) * HARTREE_TO_KCAL_MOL

    def _pyscf_wb97x_d3(atoms_in) -> float:
        m = make_pyscf_mol(atoms_in, basis="def2-svp")
        res = run_pyscf_scf(m, "rks", xc="wb97x", density_fit=True, auxbasis=auxbasis, grids_level=4)
        e_d3, _ = run_dftd3_bj(m.atom_charges(), np.asarray(m.atom_coords(unit="Bohr")), xc="wb97x")
        return float(res["e_tot"] + e_d3)

    e_int_pyscf = (
        _pyscf_wb97x_d3(dimer_opt) - _pyscf_wb97x_d3(mona_opt) - _pyscf_wb97x_d3(monb_opt)
    ) * HARTREE_TO_KCAL_MOL
    assert abs(e_int_g4d - e_int_pyscf) <= 0.05


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f8
@pytest.mark.gpu
def test_t3_uks_rsh_rijk_scanner_bh76_barrier_opt(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    tmp_path: Path,
) -> None:
    """T3-05: BH76 F + H2 -> HF + H UKS(wB97X) RI-JK .as_scanner() opt + barrier vs PySCF."""
    from gpu4dft import UKS

    core_rxns = get_bh76_openshell_reactions("core")
    species = get_bh76_species()
    rxn_fwd = next(r for r in core_rxns if r.id == "bh76_RKT10_fwd")
    rxn_rev = next(r for r in core_rxns if r.id == "bh76_RKT10_rev")
    auxbasis = "def2-universal-jkfit"

    all_sp = set(rxn_fwd.species_names) | set(rxn_rev.species_names)
    opt_mols: dict[str, Any] = {}
    opt_e_g4d: dict[str, float] = {}

    for sp_name in sorted(all_sp):
        sp = species[sp_name]
        mol = make_pyscf_mol(sp.pyscf_atom, basis="def2-svp", charge=sp.charge, spin=sp.spin)
        mf = UKS(
            mol, xc="wb97x", backend=gpu4dft_backend, grids_level=4, conv_tol=1e-10
        ).density_fit(auxbasis=auxbasis)
        if sp.n_atoms == 1:
            e_sp = float(mf.kernel())
            assert mf.converged
            opt_mols[sp_name] = mol
            opt_e_g4d[sp_name] = e_sp
        else:
            is_ts = sp_name == rxn_fwd.ts_name
            m_opt, e_opt, g_opt = _run_geometric_opt(
                mol,
                mf.nuc_grad_method().as_scanner(),
                tmp_path / f"rkt10_{sp_name}",
                transition=is_ts,
                maxiter=100,
            )
            assert np.max(np.abs(g_opt)) <= 4.5e-4
            opt_mols[sp_name] = m_opt
            opt_e_g4d[sp_name] = e_opt

    opt_e_pyscf: dict[str, float] = {}
    for sp_name, m_opt in opt_mols.items():
        res = run_pyscf_scf(m_opt, "uks", xc="wb97x", density_fit=True, auxbasis=auxbasis, grids_level=4)
        assert res["converged"]
        opt_e_pyscf[sp_name] = float(res["e_tot"])

    for rxn in (rxn_fwd, rxn_rev):
        b_g4d = sum(c * opt_e_g4d[s] for s, c in zip(rxn.species_names, rxn.coeffs)) * HARTREE_TO_KCAL_MOL
        b_pyscf = sum(c * opt_e_pyscf[s] for s, c in zip(rxn.species_names, rxn.coeffs)) * HARTREE_TO_KCAL_MOL
        assert abs(b_g4d - b_pyscf) <= 0.1


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.gpu
def test_t3_uks_ri_jk_ozaki_int8_and_sp2(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T3-06: UKS(B3LYP) RI-JK with Ozaki INT8 + mixed precision + SP2 SCF & native lvl5 grad."""
    from gpu4dft import UKS

    auxbasis = "def2-universal-jkfit"
    for mol_name in ("ch3_radical", "ho2_radical"):
        spec = get_molecule(mol_name)
        mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)

        ref = run_pyscf_scf(
            mol, "uks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
        )
        g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

        mf_scf = UKS(
            mol,
            xc="b3lyp",
            backend=gpu4dft_backend,
            precision="mixed",
            dynamic_precision=True,
            gemm_mode="ozaki_int8",
            solver="sp2",
            conv_tol=1e-10,
        ).density_fit(auxbasis=auxbasis)
        match_grids(mf_scf, ref["mf"])

        e_got = float(mf_scf.kernel())
        assert mf_scf.converged
        assert abs(e_got - ref["e_tot"]) <= 1e-6

        ss_got, _ = mf_scf.spin_square()
        assert ref["spin_square"] is not None
        assert abs(ss_got - ref["spin_square"][0]) <= 1e-5

        syms = spec.symbols
        coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

        def _build_native(c_bohr: np.ndarray, _syms=syms, _spec=spec):
            m_s = make_pyscf_mol(
                [(_syms[i], tuple(c_bohr[i])) for i in range(len(_syms))],
                basis="def2-svp",
                charge=_spec.charge,
                spin=_spec.spin,
                unit="Bohr",
            )
            return UKS(
                m_s,
                xc="b3lyp",
                backend=gpu4dft_backend,
                precision="mixed",
                dynamic_precision=True,
                gemm_mode="ozaki_int8",
                solver="sp2",
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


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.gpu
def test_t3_rsh_ri_jk_def2_tzvp_fg_shells(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T3-07: RSH RI-JK SCF & native lvl5 grad in def2-tzvp (f AOs + g aux shells)."""
    from gpu4dft import RKS, UKS

    auxbasis = "def2-universal-jkfit"
    for xc in ("cam-b3lyp", "wb97x"):
        for mol_name, method in (("ch2o", "rks"), ("ho2_radical", "uks")):
            spec = get_molecule(mol_name)
            mol = make_pyscf_mol(
                spec.pyscf_atom, basis="def2-tzvp", charge=spec.charge, spin=spec.spin
            )

            ref = run_pyscf_scf(
                mol, method, xc=xc, density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
            )
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
                ss_got, _ = mf_scf.spin_square()
                assert ref["spin_square"] is not None
                assert abs(ss_got - ref["spin_square"][0]) <= 1e-5

            syms = spec.symbols
            coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

            def _build_native(
                c_bohr: np.ndarray, _cls=cls, _syms=syms, _spec=spec, _xc=xc
            ):
                m_s = make_pyscf_mol(
                    [(_syms[i], tuple(c_bohr[i])) for i in range(len(_syms))],
                    basis="def2-tzvp",
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


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f2
@pytest.mark.f3
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.gpu
def test_t3_charged_radical_dimer_pbe0_d3bj_rijk(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T3-08: Charged radical dimer [NH3...NH3]+ UKS(PBE0-D3(BJ)) RI-JK SCF & native lvl5 grad."""
    from gpu4dft import UKS

    spec = get_molecule("nh3_dimer_cation")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp", charge=spec.charge, spin=spec.spin)
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(
        mol, "uks", xc="pbe0", density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
    )
    g_ref_xc = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)
    e_d3, g_d3 = run_dftd3_bj(spec.atomic_numbers, spec.coords_bohr, xc="pbe0")

    mf_scf = UKS(
        mol, xc="pbe0", dispersion="d3bj", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-10
    ).density_fit(auxbasis=auxbasis)
    match_grids(mf_scf, ref["mf"])

    e_got = float(mf_scf.kernel())
    assert mf_scf.converged
    assert abs(e_got - (ref["e_tot"] + e_d3)) <= 1e-6

    ss_got, _ = mf_scf.spin_square()
    assert ref["spin_square"] is not None
    assert abs(ss_got - ref["spin_square"][0]) <= 1e-5

    syms = spec.symbols
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    def _build_native(c_bohr: np.ndarray):
        m_s = make_pyscf_mol(
            [(syms[i], tuple(c_bohr[i])) for i in range(len(syms))],
            basis="def2-svp",
            charge=spec.charge,
            spin=spec.spin,
            unit="Bohr",
        )
        return UKS(
            m_s,
            xc="pbe0",
            dispersion="d3bj",
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
    assert np.max(np.abs(g_got - (g_ref_xc + g_d3))) <= 1e-4

    def _e_fn(c_bohr: np.ndarray) -> float:
        m_d = _build_native(c_bohr)
        e_d = float(m_d.kernel())
        assert m_d.converged
        return e_d

    g_fd = central_fd_grad(_e_fn, coords_bohr, step=1e-4)
    assert np.max(np.abs(g_got - g_fd)) <= 1e-5


@pytest.mark.tier3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f7
@pytest.mark.gpu
@pytest.mark.slow
def test_t3_large_mol_rsh_rijk_scf_and_grad(
    require_gpu4dft,
    require_pyscf,
    gpu4dft_backend: str,
    make_pyscf_mol,
    match_grids,
) -> None:
    """T3-09: (H2O)_32 (NAO=768) wB97X RI-JK matched SCF & native lvl5 grad within VRAM & invariants."""
    from gpu4dft import RKS

    spec = get_molecule("h2o_32")
    mol = make_pyscf_mol(spec.pyscf_atom, basis="def2-svp")
    assert mol.nao_nr() == 768
    auxbasis = "def2-universal-jkfit"

    ref = run_pyscf_scf(
        mol, "rks", xc="wb97x", density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-9
    )
    g_ref = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)

    mf_scf = RKS(
        mol, xc="wb97x", backend=gpu4dft_backend, precision="fp64", conv_tol=1e-9
    ).density_fit(auxbasis=auxbasis)
    match_grids(mf_scf, ref["mf"])
    e_got = float(mf_scf.kernel())
    assert mf_scf.converged
    assert abs(e_got - ref["e_tot"]) <= 1e-6

    with track_peak_vram_mib() as vram:
        mf_grad = RKS(
            mol, xc="wb97x", backend=gpu4dft_backend, precision="fp64", grids_level=5, conv_tol=1e-9
        ).density_fit(auxbasis=auxbasis)
        mf_grad.kernel()
        assert mf_grad.converged
        grad = np.asarray(mf_grad.nuc_grad_method().kernel(), dtype=np.float64)

    assert grad.shape == (spec.n_atoms, 3)
    assert np.max(np.abs(grad - g_ref)) <= 1e-4
    assert translational_invariance_norm(grad) <= 1e-6
    assert rotational_invariance_norm(spec.coords_bohr, grad) <= 1e-5
    if gpu4dft_backend == "cuda" and vram["total_mib"] > 0.0:
        assert vram["used_after_mib"] < vram["total_mib"]


@pytest.mark.tier3
@pytest.mark.f2
@pytest.mark.f3
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f8
@pytest.mark.gpu
def test_t3_ase_calculator_uks_d3bj_dm_reuse_traj(
    require_gpu4dft,
    require_pyscf,
    require_dftd3,
    require_ase,
    gpu4dft_backend: str,
    make_pyscf_mol,
) -> None:
    """T3-10: 5-step GPU4DFTCalculator(UKS, B3LYP-D3(BJ), DF) trajectory verifies DM reuse & step parity."""
    import ase
    from ase.units import Bohr as ASE_BOHR
    from ase.units import Hartree as ASE_HARTREE
    from gpu4dft import GPU4DFTCalculator

    spec = get_molecule("ho2_radical")
    base_pos = spec.coords_angstrom.copy()
    auxbasis = "def2-universal-jkfit"

    calc = GPU4DFTCalculator(
        method="uks",
        xc="b3lyp",
        basis="def2-svp",
        charge=0,
        spin=1,
        dispersion="d3bj",
        density_fit=True,
        auxbasis=auxbasis,
        backend=gpu4dft_backend,
        grids_level=5,
        conv_tol=1e-10,
    )
    atoms = ase.Atoms(symbols=spec.symbols, positions=base_pos)
    atoms.calc = calc

    iters_history: list[int] = []
    for step in range(5):
        pos = base_pos.copy()
        pos[0, 0] += step * 0.015
        pos[2, 1] -= step * 0.010
        atoms.set_positions(pos)

        e_ev = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        assert calc.last_mf is not None
        iters_history.append(int(calc.last_mf.iterations))

        e_g4d_eh = e_ev / ASE_HARTREE
        g_g4d_eh_bohr = -forces * (ASE_BOHR / ASE_HARTREE)

        mol_ref = make_pyscf_mol(atoms, basis="def2-svp", charge=0, spin=1)
        ref = run_pyscf_scf(
            mol_ref, "uks", xc="b3lyp", density_fit=True, auxbasis=auxbasis, grids_level=5, conv_tol=1e-11
        )
        g_ref_xc = run_pyscf_grad(ref["mf"], grid_response=True, auxbasis_response=True)
        c_bohr = pos / BOHR_TO_ANGSTROM
        e_d3, g_d3 = run_dftd3_bj(spec.atomic_numbers, c_bohr, xc="b3lyp")

        assert abs(e_g4d_eh - (ref["e_tot"] + e_d3)) <= 1.5e-4
        assert np.max(np.abs(g_g4d_eh_bohr - (g_ref_xc + g_d3))) <= 1e-5

    # Warm steps (steps 1..4) reuse prior DM and should not exceed initial cold iteration count
    assert max(iters_history[1:]) <= iters_history[0]
