"""Tests for symbolic XC codegen, GPU numerical XC integration, XC gradients, and sn-K (COSX)."""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import dft, gto, scf
from pyscf.grad import rks as grad_rks

REPO_ROOT = Path(__file__).resolve().parents[2]
CODEGEN_DIR = REPO_ROOT / "codegen"
if str(CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(CODEGEN_DIR))

from xc_codegen.functionals import (  # noqa: E402
    NAMED_FUNCTIONALS,
    build_base_functionals,
    differentiate_functional,
)
from xc_codegen.sym_diff import c, cse, diff, eval_numeric, log1p, sqrt, var  # noqa: E402

_F64_P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_I32_P = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_CACHED_LIB = None


def _find_xc_shared_lib() -> ctypes.CDLL:
    global _CACHED_LIB
    if _CACHED_LIB is not None:
        return _CACHED_LIB

    candidates: list[Path] = []
    env_path = os.environ.get("GPU4DFT_XC_LIB")
    if env_path:
        candidates.append(Path(env_path))

    # Search common build locations (standalone libgpu4dft_xc.so or integrated _core*.so)
    for base in (REPO_ROOT, Path.cwd()):
        for pat in (
            "build*/libgpu4dft_xc.so",
            "build*/**/libgpu4dft_xc.so",
            "build*/python/gpu4dft/_core*.so",
            "build*/**/_core*.so",
        ):
            candidates.extend(sorted(base.glob(pat)))

    try:
        import gpu4dft._core as core_mod

        if getattr(core_mod, "__file__", None):
            candidates.append(Path(core_mod.__file__))
    except Exception:
        pass

    for path in candidates:
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            if hasattr(lib, "gpu4dft_xc_c_eval_vxc"):
                _bind_signatures(lib)
                _CACHED_LIB = lib
                return lib
        except OSError:
            continue

    pytest.skip("Compiled libgpu4dft_xc.so / _core.so with gpu4dft_xc_c_* symbols not found")


def _bind_signatures(lib: ctypes.CDLL) -> None:
    lib.gpu4dft_xc_c_cuda_available.argtypes = []
    lib.gpu4dft_xc_c_cuda_available.restype = ctypes.c_int

    lib.gpu4dft_xc_c_exx_fraction.argtypes = [ctypes.c_char_p]
    lib.gpu4dft_xc_c_exx_fraction.restype = ctypes.c_double

    lib.gpu4dft_xc_c_lebedev.argtypes = [ctypes.c_int, _F64_P, _F64_P]
    lib.gpu4dft_xc_c_lebedev.restype = ctypes.c_int

    lib.gpu4dft_xc_c_eval_xc.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        _F64_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_eval_xc.restype = ctypes.c_int

    lib.gpu4dft_xc_c_eval_ao.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        _F64_P,
        ctypes.c_int,
        ctypes.c_int,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_eval_ao.restype = ctypes.c_int

    lib.gpu4dft_xc_c_build_grid.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
        _F64_P,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_build_grid.restype = ctypes.c_int

    lib.gpu4dft_xc_c_eval_vxc.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        _F64_P,
        _F64_P,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _F64_P,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        _F64_P,
    ]
    lib.gpu4dft_xc_c_eval_vxc.restype = ctypes.c_int

    lib.gpu4dft_xc_c_nuclear_grad.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        _F64_P,
        _F64_P,
        ctypes.c_char_p,
        ctypes.c_int,
        _F64_P,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_nuclear_grad.restype = ctypes.c_int

    lib.gpu4dft_xc_c_snk_overlap.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_snk_overlap.restype = ctypes.c_int

    lib.gpu4dft_xc_c_snk_esp.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_snk_esp.restype = ctypes.c_int

    lib.gpu4dft_xc_c_snk_compute_k.argtypes = [
        ctypes.c_int,
        _I32_P,
        _F64_P,
        _I32_P,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        _F64_P,
        _F64_P,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _F64_P,
        _F64_P,
    ]
    lib.gpu4dft_xc_c_snk_compute_k.restype = ctypes.c_int


def _pack_mol_basis(mol):
    sh_l, sh_center, sh_nprim, prim_exp, prim_coeff = [], [], [], [], []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        coord = mol.bas_coord(ib)
        exps = mol.bas_exp(ib)
        ctrs = mol.bas_ctr_coeff(ib)
        for ic in range(ctrs.shape[1]):
            sh_l.append(l)
            sh_center.extend([float(coord[0]), float(coord[1]), float(coord[2])])
            sh_nprim.append(len(exps))
            prim_exp.extend(float(e) for e in exps)
            prim_coeff.extend(float(c_val) for c_val in ctrs[:, ic])
    return (
        len(sh_l),
        np.asarray(sh_l, dtype=np.int32),
        np.asarray(sh_center, dtype=np.float64),
        np.asarray(sh_nprim, dtype=np.int32),
        np.asarray(prim_exp, dtype=np.float64),
        np.asarray(prim_coeff, dtype=np.float64),
    )


def _pack_mol_atoms(mol):
    charges = np.asarray([int(mol.atom_charge(i)) for i in range(mol.natm)], dtype=np.int32)
    coords = np.ascontiguousarray(mol.atom_coords(unit="Bohr"), dtype=np.float64).ravel()
    return mol.natm, charges, coords


# ===========================================================================
# Part 1: Pure-Stdlib Symbolic XC Codegen & Libxc Point-Wise Verification
# ===========================================================================
def test_xc_codegen_cli(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(CODEGEN_DIR)}
    listed = subprocess.check_output(
        [sys.executable, "-m", "xc_codegen", "--list-outputs"],
        env=env,
        text=True,
    )
    assert listed == "include/gpu4dft/generated/xc_kernels.hpp"

    subprocess.check_call(
        [sys.executable, "-m", "xc_codegen", "--out-dir", str(tmp_path)],
        env=env,
    )
    hdr = tmp_path / "include/gpu4dft/generated/xc_kernels.hpp"
    assert hdr.is_file()
    text = hdr.read_text()
    assert "GPU4DFT_HD inline void eval_xc_point(" in text
    assert "eval_mgga_c_tpss_d2" in text
    assert "const double t0 =" in text


def test_sym_diff_and_cse_exactness():
    x = var("rho")
    y = var("sigma")
    expr = log1p(x * x + c(0.3) * sqrt(y)) / (c(1.0) + x)
    dx = diff(expr, "rho")
    dxy = diff(dx, "sigma")

    pt = {"rho": 0.73, "sigma": 0.19}
    h = 1e-6
    fd_x = (
        eval_numeric(expr, {"rho": pt["rho"] + h, "sigma": pt["sigma"]})
        - eval_numeric(expr, {"rho": pt["rho"] - h, "sigma": pt["sigma"]})
    ) / (2.0 * h)
    assert abs(eval_numeric(dx, pt) - fd_x) < 1e-10

    fd_xy = (
        eval_numeric(dx, {"rho": pt["rho"], "sigma": pt["sigma"] + h})
        - eval_numeric(dx, {"rho": pt["rho"], "sigma": pt["sigma"] - h})
    ) / (2.0 * h)
    assert abs(eval_numeric(dxy, pt) - fd_xy) < 1e-10

    bindings, roots = cse([expr, dx, dxy])
    assert len(bindings) > 0
    assert len(roots) == 3


XC_SPEC_TABLE = [
    ("lda_x", "lda_x", "LDA", 0.0),
    ("lda_c_vwn5", "lda_c_vwn", "LDA", 0.0),
    ("lda_c_vwn_rpa", "lda_c_vwn_rpa", "LDA", 0.0),
    ("lda_c_pz", "lda_c_pz", "LDA", 0.0),
    ("gga_x_b88", "gga_x_b88", "GGA", 0.0),
    ("gga_c_lyp", "gga_c_lyp", "GGA", 0.0),
    ("gga_x_pbe", "gga_x_pbe", "GGA", 0.0),
    ("gga_c_pbe", "gga_c_pbe", "GGA", 0.0),
    ("mgga_x_tpss", "mgga_x_tpss", "MGGA", 0.0),
    ("mgga_c_tpss", "mgga_c_tpss", "MGGA", 0.0),
    ("lda", "svwn", "LDA", 0.0),
    ("svwn", "svwn", "LDA", 0.0),
    ("svwn_rpa", "lda_x + lda_c_vwn_rpa", "LDA", 0.0),
    ("spz81", "lda_x + lda_c_pz", "LDA", 0.0),
    ("pbe", "pbe", "GGA", 0.0),
    ("blyp", "blyp", "GGA", 0.0),
    ("b3lyp", "b3lyp", "GGA", 0.20),
    ("b3lyp5", "b3lyp5", "GGA", 0.20),
    ("pbe0", "pbe0", "GGA", 0.25),
    ("tpss", "tpss", "MGGA", 0.0),
]


@pytest.mark.parametrize("our_name,pyscf_code,family,expected_exx", XC_SPEC_TABLE)
def test_symbolic_xc_matches_libxc_pointwise(our_name, pyscf_code, family, expected_exx):
    base_map = {s.name: differentiate_functional(s) for s in build_base_functionals()}
    nf_spec = next(
        nf for nf in NAMED_FUNCTIONALS if our_name == nf.name or our_name in nf.aliases
    )
    assert abs(nf_spec.hyb_exx - expected_exx) < 1e-15

    rng = np.random.default_rng(2026)
    n_pts = 24
    rho = 10.0 ** rng.uniform(-3.0, 1.5, size=n_pts)
    s_red = 10.0 ** rng.uniform(-2.5, 0.8, size=n_pts)
    s_red[0] = 0.0
    dx = s_red * 2.0 * ((3.0 * math.pi**2) ** (1.0 / 3.0)) * (rho ** (4.0 / 3.0))
    dx[0] = 0.0
    sigma = dx * dx  # match exact IEEE-754 dx*dx computed inside PySCF eval_xc
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * ((3.0 * math.pi**2) ** (2.0 / 3.0)) * (rho ** (5.0 / 3.0))
    tau = tau_w + rng.uniform(0.1, 3.5, size=n_pts) * tau_unif

    if family == "LDA":
        r_in = rho
    elif family == "GGA":
        r_in = np.vstack([rho, dx, np.zeros_like(rho), np.zeros_like(rho)])
    else:
        r_in = np.vstack(
            [rho, dx, np.zeros_like(rho), np.zeros_like(rho), np.zeros_like(rho), tau]
        )

    exc_ref, vxc_ref, fxc_ref, _ = dft.libxc.eval_xc(pyscf_code, r_in, spin=0, deriv=2)

    keys = ("zk", "vrho", "vsigma", "vtau", "v2rho2", "v2rhosigma", "v2sigma2")
    vals = {k: np.zeros(n_pts) for k in keys}
    for i in range(n_pts):
        env = {"rho": float(rho[i]), "sigma": float(sigma[i]), "tau": float(tau[i])}
        for comp_name, weight in nf_spec.components:
            dmap = base_map[comp_name]
            for k in keys:
                vals[k][i] += weight * eval_numeric(dmap[k], env)

    def rel_err(a, b):
        b_arr = np.asarray(b)
        return float(np.max(np.abs(a - b_arr) / np.maximum(1.0, np.abs(b_arr))))

    assert rel_err(vals["zk"], exc_ref) < 1e-12
    assert rel_err(vals["vrho"], vxc_ref[0]) < 1e-12
    if family == "GGA":
        assert rel_err(vals["vsigma"], vxc_ref[1]) < 1e-12
    elif family == "MGGA":
        assert rel_err(vals["vsigma"], vxc_ref[1]) < 5e-11
        assert rel_err(vals["vtau"], vxc_ref[3]) < 1e-12

    assert rel_err(vals["v2rho2"][1:], fxc_ref[0][1:]) < 1e-10
    if family == "GGA":
        assert rel_err(vals["v2rhosigma"][1:], fxc_ref[1][1:]) < 1e-10
        assert rel_err(vals["v2sigma2"][1:], fxc_ref[2][1:]) < 1e-8
    elif family == "MGGA":
        assert rel_err(vals["v2rhosigma"][1:], fxc_ref[1][1:]) < 5e-10
        assert rel_err(vals["v2sigma2"][1:], fxc_ref[2][1:]) < 1e-8
    assert np.all(np.isfinite(vals["v2sigma2"]))


# ===========================================================================
# Part 2: Native C++ / CUDA GPU XC, Grid, Gradient & Semi-Numerical Exchange
# ===========================================================================
@pytest.mark.parametrize("our_name,pyscf_code,family,expected_exx", XC_SPEC_TABLE)
def test_compiled_xc_kernels_host_and_cuda(our_name, pyscf_code, family, expected_exx):
    lib = _find_xc_shared_lib()
    assert abs(lib.gpu4dft_xc_c_exx_fraction(our_name.encode()) - expected_exx) < 1e-15

    rng = np.random.default_rng(7)
    n_pts = 32
    rho = np.ascontiguousarray(10.0 ** rng.uniform(-3.0, 1.5, size=n_pts))
    s_red = 10.0 ** rng.uniform(-2.5, 0.8, size=n_pts)
    s_red[0] = 0.0
    dx = np.ascontiguousarray(
        s_red * 2.0 * ((3.0 * math.pi**2) ** (1.0 / 3.0)) * (rho ** (4.0 / 3.0))
    )
    dx[0] = 0.0
    sigma = np.ascontiguousarray(dx * dx)
    tau_w = sigma / (8.0 * rho)
    tau_unif = 0.3 * ((3.0 * math.pi**2) ** (2.0 / 3.0)) * (rho ** (5.0 / 3.0))
    tau = np.ascontiguousarray(tau_w + rng.uniform(0.1, 3.5, size=n_pts) * tau_unif)

    r_in = (
        rho
        if family == "LDA"
        else (
            np.vstack([rho, dx, np.zeros_like(rho), np.zeros_like(rho)])
            if family == "GGA"
            else np.vstack(
                [rho, dx, np.zeros_like(rho), np.zeros_like(rho), np.zeros_like(rho), tau]
            )
        )
    )
    exc_ref, vxc_ref, _, _ = dft.libxc.eval_xc(pyscf_code, r_in, spin=0, deriv=2)

    backends = [0] + ([1] if lib.gpu4dft_xc_c_cuda_available() else [])
    for bk in backends:
        out = np.zeros((10, n_pts), dtype=np.float64)
        rc = lib.gpu4dft_xc_c_eval_xc(our_name.encode(), n_pts, rho, sigma, tau, 2, bk, 1e-15, out.ravel())
        assert rc == 0
        np.testing.assert_allclose(out[0], exc_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out[1], vxc_ref[0], rtol=1e-12, atol=1e-12)
        if family != "LDA":
            np.testing.assert_allclose(out[2], vxc_ref[1], rtol=1e-11, atol=1e-12)
        if family == "MGGA":
            np.testing.assert_allclose(out[3], vxc_ref[3], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n_ang,deg", [(14, 5), (26, 7), (50, 11), (110, 17), (194, 23), (302, 29)])
def test_lebedev_quadrature_exactness(n_ang, deg):
    lib = _find_xc_shared_lib()
    xyz = np.zeros((n_ang, 3), dtype=np.float64)
    w = np.zeros(n_ang, dtype=np.float64)
    assert lib.gpu4dft_xc_c_lebedev(n_ang, xyz.ravel(), w) == 0
    np.testing.assert_allclose(np.sum(xyz**2, axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sum(w), 4.0 * math.pi, atol=1e-14)
    if deg >= 6:
        val = np.sum(w * (xyz[:, 0] ** 2) * (xyz[:, 1] ** 2) * (xyz[:, 2] ** 2))
        np.testing.assert_allclose(val, 4.0 * math.pi / 105.0, atol=1e-14)


def test_ao_eval_up_to_g_shells_host_and_cuda():
    lib = _find_xc_shared_lib()
    mol = gto.M(
        atom="O 0 0 0.1; H 0 0.75 -0.45; H 0 -0.75 -0.45",
        basis={"O": gto.basis.load("def2-tzvp", "O") + [[4, [0.9, 1.0]]], "H": "def2-tzvp"},
        verbose=0,
    )
    b_args = _pack_mol_basis(mol)
    pts = np.ascontiguousarray(np.random.default_rng(77).uniform(-2.0, 2.0, size=(35, 3)))
    ao_ref = dft.numint.eval_ao(mol, pts, deriv=2)

    backends = [0] + ([1] if lib.gpu4dft_xc_c_cuda_available() else [])
    for bk in backends:
        ao_out = np.zeros((10, len(pts), mol.nao), dtype=np.float64)
        rc = lib.gpu4dft_xc_c_eval_ao(*b_args, len(pts), pts.ravel(), 2, bk, ao_out.ravel())
        assert rc == 0
        np.testing.assert_allclose(ao_out, ao_ref, rtol=0, atol=1e-13)


@pytest.mark.parametrize("partition_scheme", [0, 1])  # 0 = Stratmann (SSF), 1 = Becke
def test_gpu_molecular_grid_and_screening(partition_scheme):
    lib = _find_xc_shared_lib()
    if not lib.gpu4dft_xc_c_cuda_available():
        pytest.skip("CUDA device required")

    mol = gto.M(
        atom=(
            "O 0 0 0; H 0 0.75 0.58; H 0 -0.75 0.58; "
            "O 0 0 6.0; H 0 0.75 6.58; H 0 -0.75 6.58; "
            "O 0 0 12.0; H 0 0.75 12.58; H 0 -0.75 12.58"
        ),
        basis="def2-svp",
        verbose=0,
    )
    a_args = _pack_mol_atoms(mol)
    b_args = _pack_mol_basis(mol)
    max_pts = 300000

    npts_h, nbat_h = ctypes.c_int(0), ctypes.c_int(0)
    avg_h = ctypes.c_double(0.0)
    coords_h = np.zeros(max_pts * 3, dtype=np.float64)
    weights_h = np.zeros(max_pts, dtype=np.float64)
    assert (
        lib.gpu4dft_xc_c_build_grid(
            *a_args, *b_args, 2, partition_scheme, 0, max_pts,
            ctypes.byref(npts_h), ctypes.byref(nbat_h), ctypes.byref(avg_h),
            coords_h, weights_h,
        )
        == 0
    )

    npts_d, nbat_d = ctypes.c_int(0), ctypes.c_int(0)
    avg_d = ctypes.c_double(0.0)
    coords_d = np.zeros(max_pts * 3, dtype=np.float64)
    weights_d = np.zeros(max_pts, dtype=np.float64)
    assert (
        lib.gpu4dft_xc_c_build_grid(
            *a_args, *b_args, 2, partition_scheme, 1, max_pts,
            ctypes.byref(npts_d), ctypes.byref(nbat_d), ctypes.byref(avg_d),
            coords_d, weights_d,
        )
        == 0
    )

    assert npts_h.value == npts_d.value and npts_d.value > 0
    assert nbat_h.value == nbat_d.value and nbat_d.value >= 4
    n = npts_d.value
    np.testing.assert_allclose(coords_d[: 3 * n], coords_h[: 3 * n], rtol=0, atol=1e-14)
    np.testing.assert_allclose(weights_d[:n], weights_h[:n], rtol=1e-14, atol=1e-13)
    assert avg_d.value < 0.85 * mol.nao  # active-AO screening drops distant shells!


@pytest.mark.parametrize("xc_name", ["svwn", "pbe", "blyp", "b3lyp", "pbe0", "tpss"])
def test_vxc_matches_pyscf_to_1e12_host_and_cuda(xc_name):
    lib = _find_xc_shared_lib()
    mol = gto.M(
        atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; H 0.05 -0.7510 -0.4650",
        basis="def2-tzvp",
        verbose=0,
    )
    mf = dft.RKS(mol, xc=xc_name)
    mf.grids.level = 2
    mf.grids.build()
    dm = np.ascontiguousarray(mf.get_init_guess(), dtype=np.float64)
    nelec_ref, exc_ref, vxc_ref = mf._numint.nr_rks(mol, mf.grids, xc_name, dm)

    b_args = _pack_mol_basis(mol)
    g_coords = np.ascontiguousarray(mf.grids.coords, dtype=np.float64).ravel()
    g_weights = np.ascontiguousarray(mf.grids.weights, dtype=np.float64)
    npts = len(g_weights)

    backends = [0] + ([1] if lib.gpu4dft_xc_c_cuda_available() else [])
    for bk in backends:
        exc_out = ctypes.c_double(0.0)
        nelec_out = ctypes.c_double(0.0)
        vxc_out = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        rc = lib.gpu4dft_xc_c_eval_vxc(
            *b_args, npts, g_coords, g_weights, xc_name.encode(),
            bk, 0, 0, dm.ravel(),
            ctypes.byref(exc_out), ctypes.byref(nelec_out), vxc_out.ravel(),
        )
        assert rc == 0
        np.testing.assert_allclose(nelec_out.value, nelec_ref, rtol=1e-13, atol=1e-12)
        np.testing.assert_allclose(exc_out.value, exc_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vxc_out, vxc_ref, rtol=1e-12, atol=1e-12)

    if lib.gpu4dft_xc_c_cuda_available():
        # Verify GPU device-pointer API (eval_vxc_device)
        exc_dev = ctypes.c_double(0.0)
        nelec_dev = ctypes.c_double(0.0)
        vxc_dev = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        assert (
            lib.gpu4dft_xc_c_eval_vxc(
                *b_args, npts, g_coords, g_weights, xc_name.encode(),
                1, 0, 1, dm.ravel(),
                ctypes.byref(exc_dev), ctypes.byref(nelec_dev), vxc_dev.ravel(),
            )
            == 0
        )
        np.testing.assert_allclose(exc_dev.value, exc_ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(vxc_dev, vxc_ref, rtol=1e-12, atol=1e-12)

        # Verify FP32 Tensor Core GEMM mode
        exc_32 = ctypes.c_double(0.0)
        nelec_32 = ctypes.c_double(0.0)
        vxc_32 = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        assert (
            lib.gpu4dft_xc_c_eval_vxc(
                *b_args, npts, g_coords, g_weights, xc_name.encode(),
                1, 1, 0, dm.ravel(),
                ctypes.byref(exc_32), ctypes.byref(nelec_32), vxc_32.ravel(),
            )
            == 0
        )
        assert abs(exc_32.value - exc_ref) < 1e-5
        np.testing.assert_allclose(vxc_32, vxc_ref, rtol=0, atol=5e-5)


@pytest.mark.parametrize("xc_name", ["svwn", "pbe", "b3lyp", "tpss"])
def test_xc_nuclear_grad_matches_pyscf(xc_name):
    lib = _find_xc_shared_lib()
    mol = gto.M(
        atom="O 0.0 0.0 0.12; H 0.0 0.76 -0.47; H 0.08 -0.74 -0.44",
        basis="def2-tzvp",
        verbose=0,
    )
    mf = dft.RKS(mol, xc=xc_name)
    mf.grids.level = 2
    mf.conv_tol = 1e-10
    mf.kernel()
    dm = np.ascontiguousarray(mf.make_rdm1(), dtype=np.float64)

    vxc_grad_pyscf = grad_rks.get_vxc(mf._numint, mol, mf.grids, xc_name, dm)[1]
    aoslices = mol.aoslice_by_atom()
    g_ref = np.zeros((mol.natm, 3))
    for ia in range(mol.natm):
        p0, p1 = aoslices[ia, 2], aoslices[ia, 3]
        g_ref[ia] = 2.0 * np.einsum("xuv,uv->x", vxc_grad_pyscf[:, p0:p1, :], dm[p0:p1, :])

    a_args = _pack_mol_atoms(mol)
    b_args = _pack_mol_basis(mol)
    g_coords = np.ascontiguousarray(mf.grids.coords, dtype=np.float64).ravel()
    g_weights = np.ascontiguousarray(mf.grids.weights, dtype=np.float64)

    backends = [0] + ([1] if lib.gpu4dft_xc_c_cuda_available() else [])
    for bk in backends:
        g_out = np.zeros((mol.natm, 3), dtype=np.float64)
        rc = lib.gpu4dft_xc_c_nuclear_grad(
            *a_args, *b_args, len(g_weights), g_coords, g_weights,
            xc_name.encode(), bk, dm.ravel(), g_out.ravel(),
        )
        assert rc == 0
        np.testing.assert_allclose(g_out, g_ref, rtol=0, atol=1e-12)


def test_snk_esp_and_cosx_exchange_host_and_cuda():
    lib = _find_xc_shared_lib()
    # 1. Exact 1e overlap & 1e grid electrostatic potential up to g (l=4) shells
    mol_g = gto.M(
        atom="O 0 0 0; H 0 0.75 0.58; H 0 -0.75 0.58",
        basis={"O": gto.basis.load("def2-tzvp", "O") + [[4, [0.85, 1.0]]], "H": "def2-svp"},
        verbose=0,
    )
    b_g = _pack_mol_basis(mol_g)
    s_ref = mol_g.intor("int1e_ovlp")
    s_ours = np.zeros((mol_g.nao, mol_g.nao), dtype=np.float64)
    assert lib.gpu4dft_xc_c_snk_overlap(*b_g, s_ours.ravel()) == 0
    np.testing.assert_allclose(s_ours, s_ref, rtol=0, atol=1e-13)

    pt = np.array([0.35, -0.42, 0.61], dtype=np.float64)
    esp_ref = mol_g.intor("int1e_grids", grids=pt.reshape(1, 3))[0]
    for bk in [0] + ([1] if lib.gpu4dft_xc_c_cuda_available() else []):
        esp_out = np.zeros((mol_g.nao, mol_g.nao), dtype=np.float64)
        assert lib.gpu4dft_xc_c_snk_esp(*b_g, pt, bk, esp_out.ravel()) == 0
        np.testing.assert_allclose(esp_out, esp_ref, rtol=0, atol=1e-13)

    # 2. Full SemiNumericalK (COSX) vs exact 4-center HF exchange K(D) on H2O / def2-svp
    mol = gto.M(
        atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
        basis="def2-svp",
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    dm = np.ascontiguousarray(mf.make_rdm1(), dtype=np.float64)
    _, vk_exact = mf.get_jk(mol, dm)

    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.prune = None
    grids.build()
    g_coords = np.ascontiguousarray(grids.coords, dtype=np.float64).ravel()
    g_weights = np.ascontiguousarray(grids.weights, dtype=np.float64)
    b_args = _pack_mol_basis(mol)

    vk_h = np.zeros((mol.nao, mol.nao), dtype=np.float64)
    assert (
        lib.gpu4dft_xc_c_snk_compute_k(
            *b_args, len(g_weights), g_coords, g_weights, 0, 0, 1, 0, dm.ravel(), vk_h.ravel()
        )
        == 0
    )
    np.testing.assert_allclose(vk_h, vk_h.T, rtol=0, atol=1e-14)
    np.testing.assert_allclose(vk_h, vk_exact, rtol=0, atol=5e-6)

    if lib.gpu4dft_xc_c_cuda_available():
        vk_d = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        assert (
            lib.gpu4dft_xc_c_snk_compute_k(
                *b_args, len(g_weights), g_coords, g_weights, 1, 0, 1, 0, dm.ravel(), vk_d.ravel()
            )
            == 0
        )
        np.testing.assert_allclose(vk_d, vk_h, rtol=0, atol=1e-12)
        np.testing.assert_allclose(vk_d, vk_exact, rtol=0, atol=5e-6)

        # Device-pointer API
        vk_dev = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        assert (
            lib.gpu4dft_xc_c_snk_compute_k(
                *b_args, len(g_weights), g_coords, g_weights, 1, 0, 1, 1, dm.ravel(), vk_dev.ravel()
            )
            == 0
        )
        np.testing.assert_allclose(vk_dev, vk_d, rtol=0, atol=1e-13)

        # FP32 Tensor Core GEMM mode
        vk_32 = np.zeros((mol.nao, mol.nao), dtype=np.float64)
        assert (
            lib.gpu4dft_xc_c_snk_compute_k(
                *b_args, len(g_weights), g_coords, g_weights, 1, 1, 1, 0, dm.ravel(), vk_32.ravel()
            )
            == 0
        )
        np.testing.assert_allclose(vk_32, vk_exact, rtol=0, atol=5e-5)
