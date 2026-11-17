"""Comprehensive pytest suite for GPU4DFT's Linear-Response Excited-State Engine
(CIS, TDA, TDHF, TDDFT / Casida RPA, Transition Dipoles & Oscillator Strengths,
Ozaki INT8 & Dynamic-Precision GEMM modes, CPHF/Z-Vector Polarizability) and
Seminumerical Molecular Nuclear Hessian & Harmonic Vibrational Frequency Engine.

Works standalone via ctypes (`GPU4DFT_RESP_LIB` / `GPU4DFT_TEST_LIB`) AND
automatically discovers `libgpu4dft_core.so` when integrated into CMake/CTest.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, hessian, scf, tdscf  # noqa: E402
from pyscf.hessian import thermo as pyscf_thermo  # noqa: E402

_DOUBLE_PTR = ctypes.POINTER(ctypes.c_double)
_INT_PTR = ctypes.POINTER(ctypes.c_int)
_FXC_CB_TYPE = ctypes.CFUNCTYPE(None, _DOUBLE_PTR, _DOUBLE_PTR, ctypes.c_void_p)

HARTREE_TO_EV = 27.21138602


def _find_resp_cdll() -> ctypes.CDLL:
    candidates: list[Path] = []
    for env_key in ("GPU4DFT_RESP_LIB", "GPU4DFT_TEST_LIB", "GPU4DFT_DRIVER_LIB"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(Path(env_val))

    repo_root = Path(__file__).resolve().parents[2]
    for sub in ("build", "build_cuda", "build_resp"):
        bdir = repo_root / sub
        for name in (
            "libgpu4dft_resp.so",
            "libgpu4dft_core.so",
            "libgpu4dft_resp.dylib",
            "libgpu4dft_core.dylib",
        ):
            candidates.append(bdir / name)

    try:
        import gpu4dft._core as _core_mod

        core_path = Path(_core_mod.__file__).resolve()
        for name in ("libgpu4dft_core.so", "libgpu4dft_core.dylib"):
            candidates.append(core_path.parents[2] / name)
            candidates.append(core_path.parent / name)
        candidates.append(core_path)
    except Exception:
        pass

    for path in candidates:
        if path.exists():
            try:
                lib = ctypes.CDLL(str(path))
                if hasattr(lib, "gpu4dft_resp_c_run_molecular_response"):
                    _bind_resp_signatures(lib)
                    return lib
            except OSError:
                continue

    pytest.skip(
        "GPU4DFT response shared library not found; set GPU4DFT_RESP_LIB or build libgpu4dft_core."
    )


def _bind_resp_signatures(lib: ctypes.CDLL) -> None:
    lib.gpu4dft_resp_c_cuda_available.argtypes = []
    lib.gpu4dft_resp_c_cuda_available.restype = ctypes.c_int

    lib.gpu4dft_resp_c_dipole_integrals.argtypes = [
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_resp_c_dipole_integrals.restype = ctypes.c_int

    lib.gpu4dft_resp_c_run_response_eri.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_resp_c_run_response_eri.restype = ctypes.c_int

    lib.gpu4dft_resp_c_run_molecular_response.argtypes = [
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_char_p,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_resp_c_run_molecular_response.restype = ctypes.c_int

    lib.gpu4dft_resp_c_polarizability_eri.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_resp_c_polarizability_eri.restype = ctypes.c_int

    lib.gpu4dft_resp_c_compute_molecular_hessian.argtypes = [
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_char_p,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        _DOUBLE_PTR,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_resp_c_compute_molecular_hessian.restype = ctypes.c_int

    lib.gpu4dft_resp_c_analyze_hessian.argtypes = [
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        ctypes.c_int,
        _INT_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
        _DOUBLE_PTR,
    ]
    lib.gpu4dft_resp_c_analyze_hessian.restype = ctypes.c_int


@pytest.fixture(scope="module")
def resp_lib() -> ctypes.CDLL:
    return _find_resp_cdll()


def _pack_mol(mol: gto.Mole):
    """Extract C-ABI packed shell and atom arrays from a PySCF Mole."""
    shell_l = []
    shell_centers = []
    prim_offsets = [0]
    prim_exps = []
    prim_coeffs = []

    for ib in range(mol.nbas):
        l_val = int(mol.bas_angular(ib))
        center = mol.bas_coord(ib)
        exps = mol.bas_exp(ib)
        ctr = mol.bas_ctr_coeff(ib)
        for ic in range(ctr.shape[1]):
            shell_l.append(l_val)
            shell_centers.extend([float(center[0]), float(center[1]), float(center[2])])
            prim_exps.extend([float(x) for x in exps])
            prim_coeffs.extend([float(c) for c in ctr[:, ic]])
            prim_offsets.append(len(prim_exps))

    atom_z = np.asarray([int(mol.atom_charge(i)) for i in range(mol.natm)], dtype=np.int32)
    atom_coords = np.ascontiguousarray(mol.atom_coords(unit="Bohr"), dtype=np.float64)

    return (
        len(shell_l),
        np.ascontiguousarray(shell_l, dtype=np.int32),
        np.ascontiguousarray(shell_centers, dtype=np.float64),
        np.ascontiguousarray(prim_offsets, dtype=np.int32),
        np.ascontiguousarray(prim_exps, dtype=np.float64),
        np.ascontiguousarray(prim_coeffs, dtype=np.float64),
        mol.natm,
        atom_z,
        atom_coords,
    )


def _dptr(arr: np.ndarray):
    return arr.ctypes.data_as(_DOUBLE_PTR)


def _iptr(arr: np.ndarray):
    return arr.ctypes.data_as(_INT_PTR)


def _mol_h2o(basis: str = "6-31g*") -> gto.Mole:
    return gto.M(
        atom="O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692",
        basis=basis,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_ch2o(basis: str = "def2-svp") -> gto.Mole:
    return gto.M(
        atom=(
            "C  0.0000  0.0000 -0.5340; "
            "O  0.0000  0.0000  0.6750; "
            "H  0.0000  0.9410 -1.1220; "
            "H  0.0000 -0.9410 -1.1220"
        ),
        basis=basis,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _mol_c2h4(basis: str = "def2-svp") -> gto.Mole:
    return gto.M(
        atom=(
            "C  0.0000  0.0000  0.6695; "
            "C  0.0000  0.0000 -0.6695; "
            "H  0.0000  0.9289  1.2321; "
            "H  0.0000 -0.9289  1.2321; "
            "H  0.0000  0.9289 -1.2321; "
            "H  0.0000 -0.9289 -1.2321"
        ),
        basis=basis,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )


def _backends_for(lib: ctypes.CDLL) -> list[int]:
    out = [0]
    if lib.gpu4dft_resp_c_cuda_available() == 1:
        out.append(1)
    return out


# ===========================================================================
# 1. AO Electric Dipole Integrals (s, p, d, f, g shells) on Host & CUDA
# ===========================================================================


@pytest.mark.parametrize("basis_name", ["6-31g*", "def2-svp", "def2-qzvp"])
def test_ao_dipole_integrals_up_to_g_shells(resp_lib: ctypes.CDLL, basis_name: str):
    mol = _mol_h2o(basis=basis_name)
    origin = np.array([0.15, -0.25, 0.40], dtype=np.float64)
    with mol.with_common_orig(origin):
        ref_dip = -mol.intor("int1e_r")  # shape (3, nao, nao)

    nsh, sh_l, sh_c, p_off, p_exp, p_coef, _, _, _ = _pack_mol(mol)
    nao = mol.nao_nr()

    for be in _backends_for(resp_lib):
        got = np.zeros((3, nao, nao), dtype=np.float64)
        rc = resp_lib.gpu4dft_resp_c_dipole_integrals(
            nsh,
            _iptr(sh_l),
            _dptr(sh_c),
            _iptr(p_off),
            _dptr(p_exp),
            _dptr(p_coef),
            _dptr(origin),
            be,
            _dptr(got),
        )
        assert rc == 0
        np.testing.assert_allclose(got, ref_dip, rtol=0.0, atol=1e-13)


# ===========================================================================
# 2. Full End-to-End CIS / TDHF & TDA / TDDFT (SVWN, PBE, B3LYP, PBE0)
#    Lowest 5 Singlet States + Oscillator Strengths on H2O, CH2O, C2H4
# ===========================================================================


@pytest.fixture(scope="module")
def pyscf_td_references():
    """Compute high-precision PySCF TDA & TDDFT/TDHF reference excited states once."""
    systems = {
        "h2o_631gs": _mol_h2o("6-31g*"),
        "ch2o_def2svp": _mol_ch2o("def2-svp"),
        "c2h4_def2svp": _mol_c2h4("def2-svp"),
    }
    methods = ["hf", "svwn", "pbe", "b3lyp", "pbe0"]
    refs: dict[tuple[str, str, int], dict] = {}

    for sys_key, mol in systems.items():
        for xc in methods:
            if xc == "hf":
                mf = scf.RHF(mol)
                mf.conv_tol = 1e-13
                mf.conv_tol_grad = 1e-11
                mf.kernel()
                g_coords = None
                g_weights = None
            else:
                mf = dft.RKS(mol, xc=xc)
                mf.grids.level = 4
                mf.grids.build()
                mf.conv_tol = 1e-13
                mf.conv_tol_grad = 1e-11
                mf.kernel()
                g_coords = np.ascontiguousarray(mf.grids.coords, dtype=np.float64)
                g_weights = np.ascontiguousarray(mf.grids.weights, dtype=np.float64)

            for method_id in (0, 1):  # 0 = TDA/CIS, 1 = TDDFT/TDHF
                td = tdscf.TDA(mf) if method_id == 0 else tdscf.TDDFT(mf)
                td.nstates = 5
                td.conv_tol = 1e-8
                td.max_space = 80
                td.max_cycle = 150
                td.kernel()
                assert td.e is not None and len(td.e) == 5
                refs[(sys_key, xc, method_id)] = {
                    "mol": mol,
                    "e_scf": float(mf.e_tot),
                    "e_exc": np.asarray(td.e, dtype=np.float64),
                    "f_osc": np.asarray(td.oscillator_strength(), dtype=np.float64),
                    "g_coords": g_coords,
                    "g_weights": g_weights,
                }
    return refs


def _run_molecular_td(
    lib: ctypes.CDLL,
    mol: gto.Mole,
    xc: str,
    backend: int,
    method_id: int,
    gemm_mode: int = 0,
    n_states: int = 5,
    g_coords: np.ndarray | None = None,
    g_weights: np.ndarray | None = None,
):
    nsh, sh_l, sh_c, p_off, p_exp, p_coef, natm, at_z, at_c = _pack_mol(mol)
    e_scf = ctypes.c_double(0.0)
    e_eh = np.zeros(n_states, dtype=np.float64)
    e_ev = np.zeros(n_states, dtype=np.float64)
    f_osc = np.zeros(n_states, dtype=np.float64)
    tdip = np.zeros((n_states, 3), dtype=np.float64)
    stats = np.zeros(4, dtype=np.float64)

    ngrid = 0 if g_coords is None else int(g_weights.shape[0])
    gc_ptr = _dptr(g_coords) if g_coords is not None else ctypes.cast(0, _DOUBLE_PTR)
    gw_ptr = _dptr(g_weights) if g_weights is not None else ctypes.cast(0, _DOUBLE_PTR)

    rc = lib.gpu4dft_resp_c_run_molecular_response(
        nsh,
        _iptr(sh_l),
        _dptr(sh_c),
        _iptr(p_off),
        _dptr(p_exp),
        _dptr(p_coef),
        natm,
        _iptr(at_z),
        _dptr(at_c),
        mol.nelectron,
        xc.encode("utf-8"),
        gc_ptr,
        gw_ptr,
        ngrid,
        backend,
        method_id,
        gemm_mode,
        8,
        n_states,
        1e-8,
        ctypes.byref(e_scf),
        _dptr(e_eh),
        _dptr(e_ev),
        _dptr(f_osc),
        _dptr(tdip),
        _dptr(stats),
    )
    assert rc == 0, f"gpu4dft_resp_c_run_molecular_response failed with rc={rc}"
    assert stats[0] == 1.0, f"Davidson solver did not converge ({xc}, m={method_id}, be={backend})"
    return e_scf.value, e_eh, e_ev, f_osc, tdip, stats


@pytest.mark.parametrize("sys_key", ["h2o_631gs", "ch2o_def2svp", "c2h4_def2svp"])
@pytest.mark.parametrize("xc", ["hf", "svwn", "pbe", "b3lyp", "pbe0"])
@pytest.mark.parametrize("method_id", [0, 1])  # 0 = CIS/TDA, 1 = TDHF/TDDFT
def test_excited_states_and_oscillator_strengths_vs_pyscf(
    resp_lib: ctypes.CDLL, pyscf_td_references, sys_key: str, xc: str, method_id: int
):
    ref = pyscf_td_references[(sys_key, xc, method_id)]
    mol = ref["mol"]
    tol_eh = 1e-8 if sys_key in ("h2o_631gs", "ch2o_def2svp") else 1e-7

    for be in _backends_for(resp_lib):
        e_scf, e_eh, e_ev, f_osc, _tdip, _stats = _run_molecular_td(
            resp_lib,
            mol,
            xc,
            backend=be,
            method_id=method_id,
            gemm_mode=0,
            n_states=5,
            g_coords=ref["g_coords"],
            g_weights=ref["g_weights"],
        )
        assert abs(e_scf - ref["e_scf"]) <= 1e-9
        np.testing.assert_allclose(e_eh, ref["e_exc"], rtol=0.0, atol=tol_eh)
        np.testing.assert_allclose(
            e_ev, ref["e_exc"] * HARTREE_TO_EV, rtol=0.0, atol=tol_eh * HARTREE_TO_EV
        )
        np.testing.assert_allclose(f_osc, ref["f_osc"], rtol=0.0, atol=1e-6)


# ===========================================================================
# 3. Ozaki INT8 GEMM Mode & Dynamic FP32->FP64 Precision Davidson Tests
# ===========================================================================


@pytest.mark.parametrize("sys_key", ["h2o_631gs", "ch2o_def2svp", "c2h4_def2svp"])
@pytest.mark.parametrize("xc", ["hf", "pbe", "b3lyp"])
@pytest.mark.parametrize("method_id", [0, 1])
def test_ozaki_int8_gemm_mode_response(
    resp_lib: ctypes.CDLL, pyscf_td_references, sys_key: str, xc: str, method_id: int
):
    ref = pyscf_td_references[(sys_key, xc, method_id)]
    mol = ref["mol"]

    for be in _backends_for(resp_lib):
        _, e_eh, _, f_osc, _, _ = _run_molecular_td(
            resp_lib,
            mol,
            xc,
            backend=be,
            method_id=method_id,
            gemm_mode=2,  # GemmMode::OzakiInt8
            n_states=5,
            g_coords=ref["g_coords"],
            g_weights=ref["g_weights"],
        )
        np.testing.assert_allclose(e_eh, ref["e_exc"], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(f_osc, ref["f_osc"], rtol=0.0, atol=1e-6)


def test_in_core_eri_dynamic_precision_and_polarizability(resp_lib: ctypes.CDLL):
    """Test standalone in-core ERI callbacks, dynamic FP32->FP64 Davidson promotion,
    and CPHF static electric dipole polarizability vs exact finite-field RHF.
    """
    mol = _mol_h2o("6-31g*")
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    nao = mol.nao_nr()
    nmo = mf.mo_coeff.shape[1]
    n_occ = mol.nelectron // 2
    mo_e = np.ascontiguousarray(mf.mo_energy, dtype=np.float64)
    mo_c = np.ascontiguousarray(mf.mo_coeff, dtype=np.float64)
    eri = np.ascontiguousarray(mol.intor("int2e"), dtype=np.float64)
    ao_dip = np.ascontiguousarray(-mol.intor("int1e_r"), dtype=np.float64)

    td_ref = tdscf.TDHF(mf)
    td_ref.nstates = 5
    td_ref.conv_tol = 1e-7
    td_ref.max_space = 80
    td_ref.kernel()

    # Exact 4-point Richardson finite-field static polarizability d\mu_a / dF_b for RHF
    field_step = 2e-4
    alpha_ff = np.zeros((3, 3), dtype=np.float64)
    h0 = mf.get_hcore()
    r_int = mol.intor("int1e_r")  # +r; H(F) = H0 + F . r -> d = -<r>
    for b in range(3):
        dips = {}
        for mult in (-2, -1, 1, 2):
            mf_f = scf.RHF(mol)
            mf_f.conv_tol = 1e-13
            mf_f.conv_tol_grad = 1e-11
            mf_f.get_hcore = lambda *args, f=(mult * field_step), axis=b: h0 + f * r_int[axis]
            mf_f.kernel()
            dm_f = mf_f.make_rdm1()
            dips[mult] = np.einsum("xij,ji->x", ao_dip, dm_f)
        alpha_ff[:, b] = (8.0 * (dips[1] - dips[-1]) - (dips[2] - dips[-2])) / (12.0 * field_step)

    for be in _backends_for(resp_lib):
        e_eh = np.zeros(5, dtype=np.float64)
        f_osc = np.zeros(5, dtype=np.float64)
        tdip = np.zeros((5, 3), dtype=np.float64)
        x_vec = np.zeros((5, n_occ, nmo - n_occ), dtype=np.float64)
        y_vec = np.zeros((5, n_occ, nmo - n_occ), dtype=np.float64)
        stats = np.zeros(4, dtype=np.float64)

        rc = resp_lib.gpu4dft_resp_c_run_response_eri(
            nao,
            nmo,
            n_occ,
            _dptr(mo_e),
            _dptr(mo_c),
            _dptr(eri),
            _dptr(ao_dip),
            be,
            1,  # TDDFT/TDHF
            0,  # FP64 final
            8,
            1,  # dynamic_precision = True (FP32 early -> FP64 final)
            5,
            1.0,
            1e-9,
            None,
            None,
            _dptr(e_eh),
            _dptr(f_osc),
            _dptr(tdip),
            _dptr(x_vec),
            _dptr(y_vec),
            _dptr(stats),
        )
        assert rc == 0 and stats[0] == 1.0
        np.testing.assert_allclose(e_eh, td_ref.e, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(f_osc, td_ref.oscillator_strength(), rtol=0.0, atol=1e-7)
        # Verify RPA biorthonormality X_k^T X_k - Y_k^T Y_k == 1
        norms = np.sum(x_vec**2 - y_vec**2, axis=(1, 2))
        np.testing.assert_allclose(norms, np.ones(5), rtol=0.0, atol=1e-10)

        polar = np.zeros((3, 3), dtype=np.float64)
        rc_pol = resp_lib.gpu4dft_resp_c_polarizability_eri(
            nao,
            nmo,
            n_occ,
            _dptr(mo_e),
            _dptr(mo_c),
            _dptr(eri),
            _dptr(ao_dip),
            be,
            1.0,
            1e-11,
            None,
            None,
            _dptr(polar),
        )
        assert rc_pol == 0
        np.testing.assert_allclose(polar, alpha_ff, rtol=0.0, atol=1e-6)


# ===========================================================================
# 4. Molecular Nuclear Hessian & Harmonic Vibrational Frequencies vs PySCF
# ===========================================================================


def _pyscf_hess_and_freqs(mol: gto.Mole, xc: str):
    if xc == "hf":
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        h4 = mf.Hessian().kernel()
        g_coords, g_weights = None, None
    else:
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = 4
        mf.grids.build()
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        hobj = mf.Hessian()
        hobj.grid_response = False
        h4 = hobj.kernel()
        g_coords = np.ascontiguousarray(mf.grids.coords, dtype=np.float64)
        g_weights = np.ascontiguousarray(mf.grids.weights, dtype=np.float64)

    n = mol.natm
    h_2d = np.ascontiguousarray(h4.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n), dtype=np.float64)
    harm = pyscf_thermo.harmonic_analysis(mol, h4)
    freqs_ref = np.asarray(np.real(harm["freq_wavenumber"]), dtype=np.float64)
    red_mass_ref = np.asarray(harm["reduced_mass"], dtype=np.float64)
    masses_pyscf = np.ascontiguousarray(mol.atom_mass_list(isotope_avg=True), dtype=np.float64)
    return h_2d, freqs_ref, red_mass_ref, masses_pyscf, g_coords, g_weights


@pytest.mark.parametrize(
    "mol_fn,basis_name,xc",
    [
        (_mol_h2o, "6-31g*", "hf"),
        (_mol_h2o, "6-31g*", "b3lyp"),
        (_mol_ch2o, "6-31g*", "hf"),
        (_mol_ch2o, "6-31g*", "b3lyp"),
    ],
)
def test_nuclear_hessian_and_vibrational_frequencies_vs_pyscf(
    resp_lib: ctypes.CDLL, mol_fn, basis_name: str, xc: str
):
    mol = mol_fn(basis_name)
    h_ref, freqs_ref, rmass_ref, masses_avg, g_coords, g_weights = _pyscf_hess_and_freqs(mol, xc)
    nsh, sh_l, sh_c, p_off, p_exp, p_coef, natm, at_z, at_c = _pack_mol(mol)
    n3 = 3 * natm

    # Prefer CUDA if available for full seminumerical loop, AND verify Host analyze_hessian too
    be_list = _backends_for(resp_lib)
    run_be = 1 if (1 in be_list) else 0

    hess_got = np.zeros((n3, n3), dtype=np.float64)
    nvib = ctypes.c_int(0)
    freqs_got = np.zeros(n3, dtype=np.float64)
    rmass_got = np.zeros(n3, dtype=np.float64)
    modes_got = np.zeros((n3, n3), dtype=np.float64)
    zpe_got = ctypes.c_double(0.0)

    masses_ptr = ctypes.cast(0, _DOUBLE_PTR)
    _ = masses_avg
    ngrid = 0 if g_coords is None else int(g_weights.shape[0])
    gc_ptr = _dptr(g_coords) if g_coords is not None else ctypes.cast(0, _DOUBLE_PTR)
    gw_ptr = _dptr(g_weights) if g_weights is not None else ctypes.cast(0, _DOUBLE_PTR)

    rc = resp_lib.gpu4dft_resp_c_compute_molecular_hessian(
        nsh,
        _iptr(sh_l),
        _dptr(sh_c),
        _iptr(p_off),
        _dptr(p_exp),
        _dptr(p_coef),
        natm,
        _iptr(at_z),
        _dptr(at_c),
        masses_ptr,
        mol.nelectron,
        xc.encode("utf-8"),
        gc_ptr,
        gw_ptr,
        ngrid,
        run_be,
        1e-3,  # step_bohr = 0.001 Bohr
        1 if xc != "hf" else 0,
        _dptr(hess_got),
        ctypes.byref(nvib),
        _dptr(freqs_got),
        _dptr(rmass_got),
        _dptr(modes_got),
        ctypes.byref(zpe_got),
    )
    assert rc == 0
    assert nvib.value == 3 * natm - 6

    np.testing.assert_allclose(hess_got, h_ref, rtol=0.0, atol=1e-5)
    np.testing.assert_allclose(freqs_got[: nvib.value], freqs_ref, rtol=0.0, atol=0.5)
    np.testing.assert_allclose(rmass_got[: nvib.value], rmass_ref, rtol=0.0, atol=1e-3)

    # Also verify analyze_hessian consistency across ALL backends (Host & CUDA)
    for be in be_list:
        nv2 = ctypes.c_int(0)
        f2 = np.zeros(n3, dtype=np.float64)
        rm2 = np.zeros(n3, dtype=np.float64)
        m2 = np.zeros((n3, n3), dtype=np.float64)
        z2 = ctypes.c_double(0.0)
        rc2 = resp_lib.gpu4dft_resp_c_analyze_hessian(
            natm,
            _iptr(at_z),
            _dptr(at_c),
            masses_ptr,
            _dptr(h_ref),
            be,
            ctypes.byref(nv2),
            _dptr(f2),
            _dptr(rm2),
            _dptr(m2),
            ctypes.byref(z2),
        )
        assert rc2 == 0 and nv2.value == nvib.value
        np.testing.assert_allclose(f2[: nv2.value], freqs_ref, rtol=0.0, atol=0.05)
