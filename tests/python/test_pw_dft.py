"""Comprehensive tests for Plane-Wave Kohn-Sham & Hybrid ACE DFT (`gpu4dft::pw`)."""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf.pbc import dft as pbc_dft
from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import scf as pbc_scf
from pyscf.pbc import tools as pbc_tools
from pyscf.pbc.gto.pseudo import pp as pbc_pp

REPO_ROOT = Path(__file__).resolve().parents[2]
BOHR_PER_ANGSTROM = 1.8897259885789233

_F64_P = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_I32_P = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
_CACHED_PW_LIB = None


def _find_pw_shared_lib() -> ctypes.CDLL:
    global _CACHED_PW_LIB
    if _CACHED_PW_LIB is not None:
        return _CACHED_PW_LIB

    candidates: list[Path] = []
    for env_var in ("GPU4DFT_PW_LIB", "GPU4DFT_TEST_LIB"):
        env_path = os.environ.get(env_var)
        if env_path:
            candidates.append(Path(env_path))

    for base in (REPO_ROOT, Path.cwd()):
        for pat in (
            "build*/libgpu4dft_pw.so",
            "build*/**/libgpu4dft_pw.so",
            "build*/libgpu4dft_core.so",
            "build*/**/libgpu4dft_core.so",
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
            if hasattr(lib, "gpu4dft_pw_c_run_scf"):
                _bind_pw_signatures(lib)
                _CACHED_PW_LIB = lib
                return lib
        except OSError:
            continue

    pytest.skip("Compiled libgpu4dft_pw.so / libgpu4dft_core.so with gpu4dft_pw_c_* symbols not found")


def _bind_pw_signatures(lib: ctypes.CDLL) -> None:
    lib.gpu4dft_pw_c_last_error.argtypes = []
    lib.gpu4dft_pw_c_last_error.restype = ctypes.c_char_p

    lib.gpu4dft_pw_c_cuda_available.argtypes = []
    lib.gpu4dft_pw_c_cuda_available.restype = ctypes.c_int

    lib.gpu4dft_pw_c_ewald.argtypes = [
        _F64_P,
        ctypes.c_int,
        _I32_P,
        _F64_P,
        ctypes.c_char_p,
        ctypes.c_void_p,
        _F64_P,
        _F64_P,
    ]
    lib.gpu4dft_pw_c_ewald.restype = ctypes.c_int

    lib.gpu4dft_pw_c_fft3d.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _F64_P,
        _F64_P,
    ]
    lib.gpu4dft_pw_c_fft3d.restype = ctypes.c_int

    lib.gpu4dft_pw_c_basis_info.argtypes = [
        _F64_P,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        _I32_P,
        _I32_P,
    ]
    lib.gpu4dft_pw_c_basis_info.restype = ctypes.c_int

    lib.gpu4dft_pw_c_solve_eigen.argtypes = [
        _F64_P,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        _F64_P,
        _I32_P,
    ]
    lib.gpu4dft_pw_c_solve_eigen.restype = ctypes.c_int

    lib.gpu4dft_pw_c_run_scf.argtypes = [
        _F64_P,          # a_vecs_9
        ctypes.c_int,    # natom
        _I32_P,          # atomic_numbers
        _F64_P,          # atom_pos_3n
        ctypes.c_char_p, # pseudo_family
        ctypes.c_char_p, # xc_name
        ctypes.c_double, # ecut_wfc
        ctypes.c_double, # ecut_rho
        ctypes.c_void_p, # fft_grid_in_3
        ctypes.c_int,    # nkpts
        ctypes.c_void_p, # kpts_3nk
        ctypes.c_void_p, # kweights_nk
        ctypes.c_void_p, # kmesh_3
        ctypes.c_int,    # charge
        ctypes.c_int,    # nbands
        ctypes.c_int,    # solver_id (0=Davidson, 1=LOBPCG)
        ctypes.c_int,    # backend_id (0=Host, 1=Cuda)
        ctypes.c_int,    # exxdiv_mode (0=None, 1=Ewald)
        ctypes.c_int,    # use_ace (0/1)
        ctypes.c_int,    # max_scf_iter
        ctypes.c_double, # conv_tol
        _F64_P,          # out_metrics_18
        ctypes.c_void_p, # out_eigenvalues
        ctypes.c_int,    # max_out_eigs
    ]
    lib.gpu4dft_pw_c_run_scf.restype = ctypes.c_int


def _check_rc(lib: ctypes.CDLL, rc: int) -> None:
    if rc != 0:
        err = lib.gpu4dft_pw_c_last_error()
        msg = err.decode("utf-8", errors="replace") if err else "unknown PW error"
        raise RuntimeError(f"gpu4dft::pw C-ABI call failed ({rc}): {msg}")


def _available_backends(lib: ctypes.CDLL) -> list[tuple[str, int]]:
    out = [("host", 0)]
    if lib.gpu4dft_pw_c_cuda_available() == 1:
        out.append(("cuda", 1))
    return out


# ============================================================================
# Exact Plane-Wave Adapter for `pyscf.pbc.dft.RKS` / `KRKS` / `RHF` (FFTDF)
# ============================================================================


class PyscfExactPWCell(pbc_gto.Cell):
    """Presents the exact spherical plane-wave basis 0.5*|k+G|^2 <= ecut_wfc to PySCF FFTDF."""

    def __init__(self, ecut_wfc: float, kpts: np.ndarray | None = None, **kwargs):
        super().__init__(**kwargs)
        self.ecut_wfc = float(ecut_wfc)
        self._ref_kpts = np.zeros((1, 3)) if kpts is None else np.atleast_2d(kpts)
        self._pw_cache: dict[str, np.ndarray] = {}

    def build(self, *args, **kwargs):
        res = super().build(*args, **kwargs)
        gv = self.get_Gv(self.mesh)
        self._max_npw = max(
            int(np.sum(0.5 * np.einsum("gx,gx->g", gv + k, gv + k) <= self.ecut_wfc + 1e-12))
            for k in self._ref_kpts
        )
        return res

    def active_pw_for_k(self, kpt: np.ndarray):
        gv = self.get_Gv(self.mesh)
        kg = gv + np.asarray(kpt, dtype=np.float64)
        kin = 0.5 * np.einsum("gx,gx->g", kg, kg)
        idx = np.where(kin <= self.ecut_wfc + 1e-12)[0]
        return idx, gv[idx], kg[idx], kin[idx]

    def nao_nr(self, cart=None):
        return self._max_npw

    def _positive_half_g(self, gv: np.ndarray) -> np.ndarray:
        c0 = gv[:, 0] > 1e-12
        c1 = (np.abs(gv[:, 0]) <= 1e-12) & (gv[:, 1] > 1e-12)
        c2 = (np.abs(gv[:, 0]) <= 1e-12) & (np.abs(gv[:, 1]) <= 1e-12) & (gv[:, 2] > 1e-12)
        return gv[c0 | c1 | c2]

    def _full_ao(self, k: np.ndarray, want_deriv1: bool, force_complex: bool) -> np.ndarray:
        key = f"{np.round(k, 12).tolist()}_{int(want_deriv1)}_{int(force_complex)}"
        if key in self._pw_cache:
            return self._pw_cache[key]
        coords = self.get_uniform_grids()
        nao = self._max_npw
        inv_sqrt_vol = 1.0 / math.sqrt(self.vol)
        sqrt2_vol = math.sqrt(2.0 / self.vol)

        if np.allclose(k, 0.0, atol=1e-12) and not force_complex:
            _, gv, _, _ = self.active_pw_for_k(k)
            pos = self._positive_half_g(gv)
            npos = len(pos)
            ao0 = np.zeros((len(coords), nao), dtype=np.float64)
            ao0[:, 0] = inv_sqrt_vol
            phase = coords @ pos.T
            cp = sqrt2_vol * np.cos(phase)
            sp = sqrt2_vol * np.sin(phase)
            ao0[:, 1 : 1 + npos] = cp
            ao0[:, 1 + npos : 1 + 2 * npos] = sp
            if want_deriv1:
                ao1 = np.zeros((4, len(coords), nao), dtype=np.float64)
                ao1[0] = ao0
                for d in range(3):
                    ao1[1 + d, :, 1 : 1 + npos] = -sp * pos[None, :, d]
                    ao1[1 + d, :, 1 + npos : 1 + 2 * npos] = cp * pos[None, :, d]
                res = ao1
            else:
                res = ao0
        else:
            _, _, kg, _ = self.active_pw_for_k(k)
            npw = len(kg)
            ao0 = np.zeros((len(coords), nao), dtype=np.complex128)
            ao0[:, :npw] = inv_sqrt_vol * np.exp(1j * (coords @ kg.T))
            if want_deriv1:
                ao1 = np.zeros((4, len(coords), nao), dtype=np.complex128)
                ao1[0] = ao0
                for d in range(3):
                    ao1[1 + d, :, :npw] = ao0[:, :npw] * (1j * kg[None, :, d])
                res = ao1
            else:
                res = ao0
        self._pw_cache[key] = res
        return res

    def pbc_eval_gto(self, eval_name, coords, comp=None, kpts=None, kpt=None, **kwargs):
        single = kpts is None
        if single:
            kpts_list = [np.zeros(3) if kpt is None else np.asarray(kpt)]
        elif np.ndim(kpts) == 1:
            single = True
            kpts_list = [np.asarray(kpts)]
        else:
            kpts_list = [np.asarray(k) for k in kpts]
        force_complex = (len(self._ref_kpts) > 1) or (
            not all(np.allclose(k, 0.0, atol=1e-12) for k in kpts_list)
        )
        want_d1 = ("deriv1" in eval_name) or ("ip" in eval_name)
        ugrids = self.get_uniform_grids()
        if len(coords) == len(ugrids):
            sl = slice(None)
        else:
            diff = np.sum(np.abs(ugrids - coords[0]), axis=1)
            p0 = int(np.argmin(diff))
            sl = slice(p0, p0 + len(coords))
        out = []
        for k in kpts_list:
            full = self._full_ao(k, want_d1, force_complex)
            out.append(full[:, sl, :] if want_d1 else full[sl, :])
        return out[0] if single else out

    def pbc_intor(self, intor, hermi=1, kpts=None, kpt=None, **kwargs):
        single = kpts is None
        if single:
            kpts_list = [np.zeros(3) if kpt is None else np.asarray(kpt)]
        elif np.ndim(kpts) == 1:
            single = True
            kpts_list = [np.asarray(kpts)]
        else:
            kpts_list = [np.asarray(k) for k in kpts]
        use_real = (len(self._ref_kpts) == 1) and all(
            np.allclose(k, 0.0, atol=1e-12) for k in kpts_list
        )
        dtype = np.float64 if use_real else np.complex128
        nao = self._max_npw
        out = []
        for k in kpts_list:
            if use_real and np.allclose(k, 0.0, atol=1e-12):
                _, gv, _, _ = self.active_pw_for_k(k)
                pos = self._positive_half_g(gv)
                npos = len(pos)
                kin_pos = 0.5 * np.einsum("gx,gx->g", pos, pos)
                d = np.full(nao, 1e6, dtype=np.float64)
                d[0] = 0.0
                d[1 : 1 + npos] = kin_pos
                d[1 + npos : 1 + 2 * npos] = kin_pos
            else:
                _, _, _, kin = self.active_pw_for_k(k)
                d = np.full(nao, 1e6, dtype=np.float64)
                d[: len(kin)] = kin
            out.append(np.eye(nao, dtype=dtype) if "ovlp" in intor else np.diag(d).astype(dtype))
        return out[0] if single else np.asarray(out)


def _run_pyscf_pw_reference(
    a_bohr: np.ndarray,
    atoms_bohr: list[tuple[str, list[float]]],
    pseudo: str,
    xc: str,
    ecut_wfc: float,
    fft_mesh: tuple[int, int, int],
    kmesh: tuple[int, int, int] = (1, 1, 1),
    exxdiv: str | None = "ewald",
) -> tuple[float, np.ndarray]:
    is_kpts = kmesh != (1, 1, 1)
    base_cell = pbc_gto.Cell()
    base_cell.a = a_bohr
    base_cell.atom = atoms_bohr
    base_cell.unit = "B"
    base_cell.pseudo = pseudo
    base_cell.mesh = list(fft_mesh)
    base_cell.verbose = 0
    base_cell.build()

    kpts = base_cell.make_kpts(list(kmesh)) if is_kpts else None
    pw_cell = PyscfExactPWCell(
        ecut_wfc=ecut_wfc,
        kpts=kpts,
        a=a_bohr,
        atom=atoms_bohr,
        unit="B",
        pseudo=pseudo,
        mesh=list(fft_mesh),
        verbose=0,
    )
    pw_cell.build()

    if not is_kpts:
        hcore = pw_cell.pbc_intor("int1e_kin") + pbc_pp.get_pp(pw_cell)
        if xc.lower() in ("hf", "rhf"):
            mf = pbc_scf.RHF(pw_cell, exxdiv=exxdiv)
        else:
            mf = pbc_dft.RKS(pw_cell, xc=xc)
            mf.exxdiv = exxdiv
        mf.chkfile = None
        mf.get_hcore = lambda *a, **kw: hcore
        mf.conv_tol = 1e-10
        e0, u0 = mf.eig(hcore, np.eye(pw_cell.nao_nr()))
        dm0 = mf.make_rdm1(u0, mf.get_occ(e0, u0))
        etot = float(mf.kernel(dm0=dm0))
        assert mf.converged
        return etot, np.asarray(mf.mo_energy)

    assert kpts is not None
    hcore_k = np.asarray(
        [pw_cell.pbc_intor("int1e_kin", kpt=k) + pbc_pp.get_pp(pw_cell, kpt=k).astype(np.complex128) for k in kpts]
    )
    kmf = pbc_dft.KRKS(pw_cell, kpts=kpts, xc=xc)
    kmf.exxdiv = exxdiv
    kmf.chkfile = None
    kmf.get_hcore = lambda *a, **kw: hcore_k
    kmf.conv_tol = 1e-10
    ke0, ku0 = kmf.eig(hcore_k, kmf.get_ovlp())
    kdm0 = kmf.make_rdm1(ku0, kmf.get_occ(ke0, ku0))
    etot = float(kmf.kernel(dm0=kdm0))
    assert kmf.converged
    return etot, np.asarray(kmf.mo_energy)


def _run_gpu4dft_pw_scf(
    lib: ctypes.CDLL,
    a_bohr: np.ndarray,
    atomic_numbers: list[int],
    coords_bohr: np.ndarray,
    pseudo: str,
    xc: str,
    ecut_wfc: float,
    fft_mesh: tuple[int, int, int] = (0, 0, 0),
    kpts: np.ndarray | None = None,
    kmesh: tuple[int, int, int] = (1, 1, 1),
    nbands: int = 0,
    solver: str = "davidson",
    backend_id: int = 0,
    exxdiv: str | None = "ewald",
    use_ace: bool = True,
    conv_tol: float = 1e-10,
) -> dict:
    a_c = np.ascontiguousarray(a_bohr, dtype=np.float64).ravel()
    z_c = np.ascontiguousarray(atomic_numbers, dtype=np.int32)
    pos_c = np.ascontiguousarray(coords_bohr, dtype=np.float64).ravel()
    grid_c = np.ascontiguousarray(fft_mesh, dtype=np.int32)
    kmesh_c = np.ascontiguousarray(kmesh, dtype=np.int32)

    nk = 0 if kpts is None else int(len(kpts))
    kpts_ptr = None
    kw_ptr = None
    if kpts is not None:
        kpts_c = np.ascontiguousarray(kpts, dtype=np.float64).ravel()
        kw_c = np.full(nk, 1.0 / nk, dtype=np.float64)
        kpts_ptr = kpts_c.ctypes.data_as(ctypes.c_void_p)
        kw_ptr = kw_c.ctypes.data_as(ctypes.c_void_p)

    solver_id = 0 if solver.lower() in ("davidson", "block_davidson") else 1
    exxdiv_id = 0 if exxdiv is None else 1
    metrics = np.zeros(18, dtype=np.float64)
    max_eigs = max(1, nk if nk > 0 else 1) * max(nbands, 16)
    eigs = np.zeros(max_eigs, dtype=np.float64)

    rc = lib.gpu4dft_pw_c_run_scf(
        a_c,
        len(atomic_numbers),
        z_c,
        pos_c,
        pseudo.encode("utf-8"),
        xc.encode("utf-8"),
        float(ecut_wfc),
        0.0,
        grid_c.ctypes.data_as(ctypes.c_void_p),
        nk,
        kpts_ptr,
        kw_ptr,
        kmesh_c.ctypes.data_as(ctypes.c_void_p),
        0,
        int(nbands),
        solver_id,
        int(backend_id),
        exxdiv_id,
        1 if use_ace else 0,
        80,
        float(conv_tol),
        metrics,
        eigs.ctypes.data_as(ctypes.c_void_p),
        max_eigs,
    )
    _check_rc(lib, rc)
    return {
        "total_energy": float(metrics[0]),
        "kinetic_energy": float(metrics[1]),
        "local_pp_energy": float(metrics[2]),
        "nonlocal_pp_energy": float(metrics[3]),
        "hartree_energy": float(metrics[4]),
        "xc_energy": float(metrics[5]),
        "exx_energy": float(metrics[6]),
        "ewald_energy": float(metrics[7]),
        "homo": float(metrics[8]),
        "lumo": float(metrics[9]),
        "density_rms": float(metrics[10]),
        "converged": bool(metrics[11] > 0.5),
        "scf_iterations": int(round(metrics[12])),
        "ace_iterations": int(round(metrics[13])),
        "eig_iterations": int(round(metrics[14])),
        "hpsi_calls": int(round(metrics[15])),
        "poisson_ffts": int(round(metrics[16])),
        "elapsed_ms": float(metrics[17]),
        "raw_eigs": eigs,
    }


def _diamond_geometry(a_ang: float) -> tuple[np.ndarray, np.ndarray]:
    a_b = a_ang * BOHR_PER_ANGSTROM
    a_vecs = np.array(
        [
            [0.0, 0.5 * a_b, 0.5 * a_b],
            [0.5 * a_b, 0.0, 0.5 * a_b],
            [0.5 * a_b, 0.5 * a_b, 0.0],
        ],
        dtype=np.float64,
    )
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.25 * a_b, 0.25 * a_b, 0.25 * a_b],
        ],
        dtype=np.float64,
    )
    return a_vecs, coords


# ============================================================================
# 1. Ewald Ion-Ion Repulsion & Madelung Constant Tests (< 1e-10 Eh)
# ============================================================================


@pytest.mark.parametrize(
    "case_name,a_ang,symbols,z_list,coords_ang,pseudo,kmesh",
    [
        (
            "diamond_si_pade",
            [[0.0, 2.715, 2.715], [2.715, 0.0, 2.715], [2.715, 2.715, 0.0]],
            ["Si", "Si"],
            [14, 14],
            [[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]],
            "gth-pade",
            (1, 1, 1),
        ),
        (
            "diamond_c_pbe_222",
            [[0.0, 1.7835, 1.7835], [1.7835, 0.0, 1.7835], [1.7835, 1.7835, 0.0]],
            ["C", "C"],
            [6, 6],
            [[0.0, 0.0, 0.0], [0.89175, 0.89175, 0.89175]],
            "gth-pbe",
            (2, 2, 2),
        ),
        (
            "h2o_triclinic",
            [[6.0, 0.4, 0.2], [0.1, 5.8, 0.3], [0.2, 0.1, 6.2]],
            ["O", "H", "H"],
            [8, 1, 1],
            [[2.5, 2.5, 2.5], [3.257, 3.086, 2.5], [1.743, 3.086, 2.5]],
            "gth-pade",
            (1, 1, 1),
        ),
        (
            "alp_rocksalt_like",
            [[0.0, 2.73, 2.73], [2.73, 0.0, 2.73], [2.73, 2.73, 0.0]],
            ["Al", "P"],
            [13, 15],
            [[0.0, 0.0, 0.0], [1.365, 1.365, 1.365]],
            "gth-pbe",
            (2, 2, 2),
        ),
    ],
)
def test_ewald_and_madelung_vs_pyscf(
    case_name: str,
    a_ang: list[list[float]],
    symbols: list[str],
    z_list: list[int],
    coords_ang: list[list[float]],
    pseudo: str,
    kmesh: tuple[int, int, int],
) -> None:
    lib = _find_pw_shared_lib()
    a_bohr = np.asarray(a_ang, dtype=np.float64) * BOHR_PER_ANGSTROM
    pos_bohr = np.asarray(coords_ang, dtype=np.float64) * BOHR_PER_ANGSTROM

    cell = pbc_gto.Cell()
    cell.a = a_bohr
    cell.atom = [[s, p.tolist()] for s, p in zip(symbols, pos_bohr)]
    cell.unit = "B"
    cell.pseudo = pseudo
    cell.precision = 1e-14
    cell.verbose = 0
    cell.build()

    ref_ewald = float(cell.ewald())
    kpts = cell.make_kpts(list(kmesh))
    ref_madelung = float(pbc_tools.pbc.madelung(cell, kpts))

    out_ew = np.zeros(1, dtype=np.float64)
    out_mad = np.zeros(1, dtype=np.float64)
    km_arr = np.asarray(kmesh, dtype=np.int32)
    rc = lib.gpu4dft_pw_c_ewald(
        np.ascontiguousarray(a_bohr).ravel(),
        len(z_list),
        np.asarray(z_list, dtype=np.int32),
        np.ascontiguousarray(pos_bohr).ravel(),
        pseudo.encode("utf-8"),
        km_arr.ctypes.data_as(ctypes.c_void_p),
        out_ew,
        out_mad,
    )
    _check_rc(lib, rc)

    assert abs(out_ew[0] - ref_ewald) < 1e-11, f"{case_name}: {out_ew[0]} vs {ref_ewald}"
    assert abs(out_mad[0] - ref_madelung) < 1e-11, f"{case_name}: {out_mad[0]} vs {ref_madelung}"


# ============================================================================
# 2. Batched 3D cuFFT vs Host Mixed-Radix/Bluestein FFT & Basis Sizing (< 1e-12)
# ============================================================================


@pytest.mark.parametrize(
    "grid,nbatch",
    [
        ((12, 15, 18), 6),  # Smooth 2^a * 3^b * 5^c mixed-radix
        ((16, 20, 24), 8),  # Larger batched smooth grid
        ((11, 13, 7), 4),   # Prime dimensions exercising Bluestein Chirp-Z on Host & cuFFT
    ],
)
def test_batched_3d_fft_host_and_cuda(grid: tuple[int, int, int], nbatch: int) -> None:
    lib = _find_pw_shared_lib()
    rng = np.random.default_rng(42 + grid[0])
    shape = (nbatch, grid[0], grid[1], grid[2])
    data = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    data = np.ascontiguousarray(data, dtype=np.complex128)

    ref_fwd = np.fft.fftn(data, axes=(1, 2, 3))
    ref_inv = np.fft.ifftn(data, axes=(1, 2, 3)) * float(np.prod(grid))

    in_f64 = data.view(np.float64).ravel()
    for backend_name, backend_id in _available_backends(lib):
        out_fwd_f64 = np.empty_like(in_f64)
        rc = lib.gpu4dft_pw_c_fft3d(
            grid[0], grid[1], grid[2], nbatch, -1, backend_id, in_f64, out_fwd_f64
        )
        _check_rc(lib, rc)
        got_fwd = out_fwd_f64.view(np.complex128).reshape(shape)
        assert np.max(np.abs(got_fwd - ref_fwd)) < 1e-12, f"{backend_name} forward FFT mismatch"

        out_inv_f64 = np.empty_like(in_f64)
        rc = lib.gpu4dft_pw_c_fft3d(
            grid[0], grid[1], grid[2], nbatch, +1, backend_id, in_f64, out_inv_f64
        )
        _check_rc(lib, rc)
        got_inv = out_inv_f64.view(np.complex128).reshape(shape)
        assert np.max(np.abs(got_inv - ref_inv)) < 1e-12, f"{backend_name} inverse FFT mismatch"

        # Round-trip check
        rt_f64 = np.empty_like(in_f64)
        rc = lib.gpu4dft_pw_c_fft3d(
            grid[0], grid[1], grid[2], nbatch, +1, backend_id, out_fwd_f64, rt_f64
        )
        _check_rc(lib, rc)
        got_rt = rt_f64.view(np.complex128).reshape(shape) / float(np.prod(grid))
        assert np.max(np.abs(got_rt - data)) < 1e-13


def test_pw_basis_smooth_grid_and_free_local_eigenvalues() -> None:
    lib = _find_pw_shared_lib()
    a_bohr = np.diag([7.0, 7.5, 8.0]).astype(np.float64)
    kpt = np.array([0.13, -0.21, 0.07], dtype=np.float64)
    ecut = 8.0

    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    rc = lib.gpu4dft_pw_c_basis_info(
        a_bohr.ravel(),
        ecut,
        0.0,
        kpt.ctypes.data_as(ctypes.c_void_p),
        None,
        grid_out,
        npw_out,
    )
    _check_rc(lib, rc)
    for n in grid_out:
        rem = int(n)
        for p in (2, 3, 5):
            while rem % p == 0:
                rem //= p
        assert rem == 1, f"FFT grid size {n} is not 2^a*3^b*5^c smooth"

    # Analytical free-electron kinetic eigenvalues 0.5 * |k + G|^2
    b_vecs = 2.0 * math.pi * np.linalg.inv(a_bohr).T
    rx = np.fft.fftfreq(grid_out[0], 1.0 / grid_out[0])
    ry = np.fft.fftfreq(grid_out[1], 1.0 / grid_out[1])
    rz = np.fft.fftfreq(grid_out[2], 1.0 / grid_out[2])
    m_grid = np.stack(np.meshgrid(rx, ry, rz, indexing="ij"), axis=-1).reshape(-1, 3)
    gv = m_grid @ b_vecs
    kg = gv + kpt[None, :]
    kin_all = 0.5 * np.einsum("gx,gx->g", kg, kg)
    active_mask = kin_all <= ecut + 1e-12
    assert int(np.sum(active_mask)) == int(npw_out[0])
    ref_free = np.sort(kin_all[active_mask])[:8]

    # Smooth 3D coupled cosine local potential V(r) -> exact dense reference in active PW basis
    i0 = np.arange(grid_out[0])[:, None, None] / grid_out[0]
    i1 = np.arange(grid_out[1])[None, :, None] / grid_out[1]
    i2 = np.arange(grid_out[2])[None, None, :] / grid_out[2]
    vloc_3d = (
        -0.65 * np.cos(2.0 * math.pi * i0)
        - 0.45 * np.cos(2.0 * math.pi * i1)
        - 0.35 * np.cos(2.0 * math.pi * i2)
        + 0.25 * np.cos(2.0 * math.pi * (i0 + i1 - i2))
    )
    vloc_r = np.ascontiguousarray(vloc_3d.ravel(), dtype=np.float64)

    # Exact dense Hamiltonian H_{G, G'} = T_G delta_{G,G'} + V_loc(G - G')
    ngrid = int(np.prod(grid_out))
    idx_act = np.where(active_mask)[0]
    npw = len(idx_act)
    h_dense = np.diag(kin_all[idx_act]).astype(np.complex128)
    for j in range(npw):
        box = np.zeros(ngrid, dtype=np.complex128)
        box[idx_act[j]] = 1.0
        u_r = np.fft.ifftn(box.reshape(grid_out)).ravel() * ngrid
        vu_g = np.fft.fftn(((vloc_r / ngrid) * u_r).reshape(grid_out)).ravel()
        h_dense[:, j] += vu_g[idx_act]
    ref_local = np.linalg.eigvalsh(h_dense)[:8]

    for backend_name, backend_id in _available_backends(lib):
        for solver_name, solver_id in (("davidson", 0), ("lobpcg", 1)):
            evals_free = np.zeros(8, dtype=np.float64)
            iters = np.zeros(1, dtype=np.int32)
            rc = lib.gpu4dft_pw_c_solve_eigen(
                a_bohr.ravel(),
                ecut,
                kpt.ctypes.data_as(ctypes.c_void_p),
                grid_out.ctypes.data_as(ctypes.c_void_p),
                None,
                8,
                solver_id,
                backend_id,
                1e-12,
                80,
                evals_free,
                iters,
            )
            _check_rc(lib, rc)
            assert np.max(np.abs(evals_free - ref_free)) < 1e-12, (
                f"Free-electron {backend_name}/{solver_name} error: {np.max(np.abs(evals_free - ref_free))}"
            )

            evals_loc = np.zeros(8, dtype=np.float64)
            rc = lib.gpu4dft_pw_c_solve_eigen(
                a_bohr.ravel(),
                ecut,
                kpt.ctypes.data_as(ctypes.c_void_p),
                grid_out.ctypes.data_as(ctypes.c_void_p),
                vloc_r.ctypes.data_as(ctypes.c_void_p),
                8,
                solver_id,
                backend_id,
                1e-12,
                80,
                evals_loc,
                iters,
            )
            _check_rc(lib, rc)
            assert np.max(np.abs(evals_loc - ref_local)) < 1e-12, (
                f"Local-potential {backend_name}/{solver_name} error: {np.max(np.abs(evals_loc - ref_local))}"
            )


# ============================================================================
# 3. Periodic Kohn-Sham DFT: Diamond Si, Diamond C, H2 & H2O (LDA & PBE, Gamma & 2x2x2)
# ============================================================================


@pytest.mark.parametrize(
    "pseudo,xc_gpu4dft,xc_pyscf",
    [
        ("gth-pade", "svwn", "lda,vwn5"),
        ("gth-pade", "spz81", "lda,pz"),
        ("gth-pbe", "pbe", "pbe"),
    ],
)
def test_diamond_si_gamma_lda_and_pbe_davidson_lobpcg(
    pseudo: str, xc_gpu4dft: str, xc_pyscf: str
) -> None:
    lib = _find_pw_shared_lib()
    a_bohr, coords_bohr = _diamond_geometry(5.43)
    ecut = 10.0

    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    _check_rc(lib, lib.gpu4dft_pw_c_basis_info(a_bohr.ravel(), ecut, 0.0, None, None, grid_out, npw_out))
    mesh = (int(grid_out[0]), int(grid_out[1]), int(grid_out[2]))

    atoms_pyscf = [("Si", coords_bohr[0].tolist()), ("Si", coords_bohr[1].tolist())]
    e_ref, mo_ref = _run_pyscf_pw_reference(a_bohr, atoms_pyscf, pseudo, xc_pyscf, ecut, mesh)

    for backend_name, backend_id in _available_backends(lib):
        for solver in ("davidson", "lobpcg"):
            out = _run_gpu4dft_pw_scf(
                lib,
                a_bohr,
                [14, 14],
                coords_bohr,
                pseudo,
                xc_gpu4dft,
                ecut,
                fft_mesh=mesh,
                nbands=6,
                solver=solver,
                backend_id=backend_id,
                conv_tol=1e-10,
            )
            assert out["converged"], f"Diamond Si {pseudo}/{xc_gpu4dft} {backend_name}/{solver} did not converge"
            assert abs(out["total_energy"] - e_ref) < 1e-8, (
                f"Diamond Si {xc_gpu4dft} {backend_name}/{solver}: {out['total_energy']} vs {e_ref}"
            )
            assert np.max(np.abs(out["raw_eigs"][:4] - mo_ref[:4])) < 1e-7


@pytest.mark.parametrize(
    "system,a_ang,z_elem,pseudo,xc_gpu4dft,xc_pyscf,ecut",
    [
        ("diamond_si_222_lda", 5.43, 14, "gth-pade", "svwn", "lda,vwn5", 8.0),
        ("diamond_si_222_pbe", 5.43, 14, "gth-pbe", "pbe", "pbe", 8.0),
        ("diamond_c_222_pbe", 3.567, 6, "gth-pbe", "pbe", "pbe", 11.0),
    ],
)
def test_diamond_si_and_c_2x2x2_monkhorst_pack_krks(
    system: str,
    a_ang: float,
    z_elem: int,
    pseudo: str,
    xc_gpu4dft: str,
    xc_pyscf: str,
    ecut: float,
) -> None:
    lib = _find_pw_shared_lib()
    a_bohr, coords_bohr = _diamond_geometry(a_ang)
    sym = "Si" if z_elem == 14 else "C"

    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    _check_rc(lib, lib.gpu4dft_pw_c_basis_info(a_bohr.ravel(), ecut, 0.0, None, None, grid_out, npw_out))
    mesh = (int(grid_out[0]), int(grid_out[1]), int(grid_out[2]))

    kmesh = (2, 2, 2)
    tmp_cell = pbc_gto.M(
        a=a_bohr,
        atom=[(sym, coords_bohr[0].tolist()), (sym, coords_bohr[1].tolist())],
        unit="B",
        pseudo=pseudo,
        mesh=list(mesh),
        verbose=0,
    )
    kpts_222 = tmp_cell.make_kpts(list(kmesh))

    e_ref, _ = _run_pyscf_pw_reference(
        a_bohr,
        [(sym, coords_bohr[0].tolist()), (sym, coords_bohr[1].tolist())],
        pseudo,
        xc_pyscf,
        ecut,
        mesh,
        kmesh=kmesh,
    )

    energies_by_backend = {}
    for backend_name, backend_id in _available_backends(lib):
        out = _run_gpu4dft_pw_scf(
            lib,
            a_bohr,
            [z_elem, z_elem],
            coords_bohr,
            pseudo,
            xc_gpu4dft,
            ecut,
            fft_mesh=mesh,
            kpts=kpts_222,
            kmesh=kmesh,
            nbands=5,
            solver="davidson",
            backend_id=backend_id,
            conv_tol=1e-10,
        )
        assert out["converged"], f"{system} ({backend_name}) failed to converge"
        assert abs(out["total_energy"] - e_ref) < 1e-8, (
            f"{system} ({backend_name}): {out['total_energy']} vs PySCF KRKS {e_ref}"
        )
        energies_by_backend[backend_name] = out["total_energy"]

    if "cuda" in energies_by_backend:
        assert abs(energies_by_backend["cuda"] - energies_by_backend["host"]) < 1e-10


def test_diamond_c_gamma_lda_and_pbe() -> None:
    lib = _find_pw_shared_lib()
    a_bohr, coords_bohr = _diamond_geometry(3.567)
    ecut = 14.0
    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    _check_rc(lib, lib.gpu4dft_pw_c_basis_info(a_bohr.ravel(), ecut, 0.0, None, None, grid_out, npw_out))
    mesh = (int(grid_out[0]), int(grid_out[1]), int(grid_out[2]))

    atoms_pyscf = [("C", coords_bohr[0].tolist()), ("C", coords_bohr[1].tolist())]
    for pseudo, xc_g, xc_p in (("gth-pade", "svwn", "lda,vwn5"), ("gth-pbe", "pbe", "pbe")):
        e_ref, _ = _run_pyscf_pw_reference(a_bohr, atoms_pyscf, pseudo, xc_p, ecut, mesh)
        for backend_name, backend_id in _available_backends(lib):
            out = _run_gpu4dft_pw_scf(
                lib,
                a_bohr,
                [6, 6],
                coords_bohr,
                pseudo,
                xc_g,
                ecut,
                fft_mesh=mesh,
                solver="lobpcg",
                backend_id=backend_id,
                conv_tol=1e-10,
            )
            assert out["converged"]
            assert abs(out["total_energy"] - e_ref) < 1e-8, (
                f"Diamond C {xc_g} ({backend_name}): {out['total_energy']} vs {e_ref}"
            )


@pytest.mark.parametrize(
    "mol_name,box_bohr,symbols,z_list,coords_bohr,pseudo,xc_gpu4dft,xc_pyscf,ecut",
    [
        (
            "H2_box_lda",
            6.5,
            ["H", "H"],
            [1, 1],
            [[2.55, 3.25, 3.25], [3.95, 3.25, 3.25]],
            "gth-pade",
            "svwn",
            "lda,vwn5",
            10.0,
        ),
        (
            "H2O_box_pbe",
            6.5,
            ["O", "H", "H"],
            [8, 1, 1],
            [[3.25, 3.25, 3.00], [4.68, 3.25, 4.10], [1.82, 3.25, 4.10]],
            "gth-pbe",
            "pbe",
            "pbe",
            9.0,
        ),
    ],
)
def test_h2_and_h2o_periodic_box_rks(
    mol_name: str,
    box_bohr: float,
    symbols: list[str],
    z_list: list[int],
    coords_bohr: list[list[float]],
    pseudo: str,
    xc_gpu4dft: str,
    xc_pyscf: str,
    ecut: float,
) -> None:
    lib = _find_pw_shared_lib()
    a_bohr = np.eye(3, dtype=np.float64) * box_bohr
    pos_arr = np.asarray(coords_bohr, dtype=np.float64)

    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    _check_rc(lib, lib.gpu4dft_pw_c_basis_info(a_bohr.ravel(), ecut, 0.0, None, None, grid_out, npw_out))
    mesh = (int(grid_out[0]), int(grid_out[1]), int(grid_out[2]))

    atoms_pyscf = [(s, c) for s, c in zip(symbols, coords_bohr)]
    e_ref, _ = _run_pyscf_pw_reference(a_bohr, atoms_pyscf, pseudo, xc_pyscf, ecut, mesh)

    for backend_name, backend_id in _available_backends(lib):
        for solver in ("davidson", "lobpcg"):
            out = _run_gpu4dft_pw_scf(
                lib,
                a_bohr,
                z_list,
                pos_arr,
                pseudo,
                xc_gpu4dft,
                ecut,
                fft_mesh=mesh,
                solver=solver,
                backend_id=backend_id,
                conv_tol=1e-10,
            )
            assert out["converged"]
            assert abs(out["total_energy"] - e_ref) < 1e-8, (
                f"{mol_name} ({backend_name}/{solver}): {out['total_energy']} vs {e_ref}"
            )


# ============================================================================
# 4. Hybrid ACE Exact Exchange (`PBE0` & `HF`, `exxdiv=None` & `'ewald'`) + Speedup
# ============================================================================


@pytest.mark.parametrize(
    "case_name,xc_name,pseudo,exxdiv",
    [
        ("si_pbe0_ewald", "pbe0", "gth-pbe", "ewald"),
        ("si_pbe0_none", "pbe0", "gth-pbe", None),
        ("si_hf_ewald", "hf", "gth-pade", "ewald"),
        ("si_hf_none", "hf", "gth-pade", None),
    ],
)
def test_ace_hybrid_pbe0_and_hf_vs_pyscf(
    case_name: str, xc_name: str, pseudo: str, exxdiv: str | None
) -> None:
    lib = _find_pw_shared_lib()
    a_bohr, coords_bohr = _diamond_geometry(5.43)
    ecut = 8.0

    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    _check_rc(lib, lib.gpu4dft_pw_c_basis_info(a_bohr.ravel(), ecut, 0.0, None, None, grid_out, npw_out))
    mesh = (int(grid_out[0]), int(grid_out[1]), int(grid_out[2]))

    atoms_pyscf = [("Si", coords_bohr[0].tolist()), ("Si", coords_bohr[1].tolist())]
    e_ref, _ = _run_pyscf_pw_reference(
        a_bohr, atoms_pyscf, pseudo, xc_name, ecut, mesh, exxdiv=exxdiv
    )

    energies = {}
    for backend_name, backend_id in _available_backends(lib):
        out = _run_gpu4dft_pw_scf(
            lib,
            a_bohr,
            [14, 14],
            coords_bohr,
            pseudo,
            xc_name,
            ecut,
            fft_mesh=mesh,
            solver="davidson",
            backend_id=backend_id,
            exxdiv=exxdiv,
            use_ace=True,
            conv_tol=1e-10,
        )
        assert out["converged"], f"{case_name} ({backend_name}) ACE did not converge"
        assert abs(out["total_energy"] - e_ref) < 1e-7, (
            f"{case_name} ({backend_name}): {out['total_energy']} vs PySCF {e_ref}"
        )
        energies[backend_name] = out["total_energy"]

    if "cuda" in energies:
        assert abs(energies["cuda"] - energies["host"]) < 1e-9


def test_h2o_ace_pbe0_and_ace_speedup_benchmark() -> None:
    lib = _find_pw_shared_lib()
    a_bohr = np.eye(3, dtype=np.float64) * 6.8
    coords_bohr = np.array(
        [[3.4, 3.4, 3.1], [4.83, 3.4, 4.2], [1.97, 3.4, 4.2]], dtype=np.float64
    )
    ecut = 9.0

    grid_out = np.zeros(3, dtype=np.int32)
    npw_out = np.zeros(1, dtype=np.int32)
    _check_rc(lib, lib.gpu4dft_pw_c_basis_info(a_bohr.ravel(), ecut, 0.0, None, None, grid_out, npw_out))
    mesh = (int(grid_out[0]), int(grid_out[1]), int(grid_out[2]))

    atoms_pyscf = [
        ("O", coords_bohr[0].tolist()),
        ("H", coords_bohr[1].tolist()),
        ("H", coords_bohr[2].tolist()),
    ]
    e_ref, _ = _run_pyscf_pw_reference(
        a_bohr, atoms_pyscf, "gth-pbe", "pbe0", ecut, mesh, exxdiv="ewald"
    )

    backends = dict(_available_backends(lib))
    bench_backend_id = backends.get("cuda", 0)

    # 1. Exact H2O PBE0 accuracy verification against pyscf.pbc (LOBPCG + ACE, Davidson + ACE, Uncompressed)
    lobpcg_ace = _run_gpu4dft_pw_scf(
        lib,
        a_bohr,
        [8, 1, 1],
        coords_bohr,
        "gth-pbe",
        "pbe0",
        ecut,
        fft_mesh=mesh,
        solver="lobpcg",
        backend_id=bench_backend_id,
        exxdiv="ewald",
        use_ace=True,
        conv_tol=1e-9,
    )
    ace_res = _run_gpu4dft_pw_scf(
        lib,
        a_bohr,
        [8, 1, 1],
        coords_bohr,
        "gth-pbe",
        "pbe0",
        ecut,
        fft_mesh=mesh,
        solver="davidson",
        backend_id=bench_backend_id,
        exxdiv="ewald",
        use_ace=True,
        conv_tol=1e-9,
    )
    uncomp_res = _run_gpu4dft_pw_scf(
        lib,
        a_bohr,
        [8, 1, 1],
        coords_bohr,
        "gth-pbe",
        "pbe0",
        ecut,
        fft_mesh=mesh,
        solver="davidson",
        backend_id=bench_backend_id,
        exxdiv="ewald",
        use_ace=False,
        conv_tol=1e-9,
    )

    assert ace_res["converged"] and lobpcg_ace["converged"] and uncomp_res["converged"]
    assert abs(ace_res["total_energy"] - e_ref) < 1e-7
    assert abs(lobpcg_ace["total_energy"] - e_ref) < 1e-7
    assert abs(uncomp_res["total_energy"] - e_ref) < 1e-7
    assert abs(ace_res["total_energy"] - uncomp_res["total_energy"]) < 1e-8
    assert ace_res["poisson_ffts"] * 5 < uncomp_res["poisson_ffts"]

    # 2. Steady-state ACE wall-clock speedup benchmark on 2x1x1 Diamond Si supercell (4 Si, n_occ=8, Ecut=18 Ha)
    a_si, pos_si = _diamond_geometry(5.43)
    a_super = a_si.copy()
    a_super[0] *= 2.0
    pos_super = np.vstack([pos_si, pos_si + a_si[0]])
    z_super = [14, 14, 14, 14]
    ecut_bench = 18.0
    # Warm up cuFFT plans on the supercell grid for both solvers so timing reflects steady-state SCF execution
    _run_gpu4dft_pw_scf(
        lib, a_super, z_super, pos_super, "gth-pbe", "pbe0", ecut_bench,
        solver="davidson", backend_id=bench_backend_id, exxdiv="ewald", use_ace=False, conv_tol=1e-3,
    )
    ace_bench = _run_gpu4dft_pw_scf(
        lib, a_super, z_super, pos_super, "gth-pbe", "pbe0", ecut_bench,
        solver="davidson", backend_id=bench_backend_id, exxdiv="ewald", use_ace=True, conv_tol=1e-9,
    )
    uncomp_bench = _run_gpu4dft_pw_scf(
        lib, a_super, z_super, pos_super, "gth-pbe", "pbe0", ecut_bench,
        solver="davidson", backend_id=bench_backend_id, exxdiv="ewald", use_ace=False, conv_tol=1e-9,
    )
    assert ace_bench["converged"] and uncomp_bench["converged"]
    assert abs(ace_bench["total_energy"] - uncomp_bench["total_energy"]) < 1e-8
    assert ace_bench["poisson_ffts"] * 5 < uncomp_bench["poisson_ffts"], (
        f"Expected >5x fewer 3D Poisson FFTs with ACE: {ace_bench['poisson_ffts']} vs {uncomp_bench['poisson_ffts']}"
    )
    assert ace_bench["elapsed_ms"] < uncomp_bench["elapsed_ms"], (
        f"Expected ACE wall-clock speedup: {ace_bench['elapsed_ms']:.2f} ms vs {uncomp_bench['elapsed_ms']:.2f} ms"
    )


def test_nanobind_pw_submodule_if_wired() -> None:
    """Exercises `gpu4dft.pw` Python bindings once wired into `nanobind`."""
    try:
        import gpu4dft
    except Exception:
        pytest.skip("gpu4dft Python package not in PYTHONPATH")

    pw_mod = getattr(gpu4dft, "pw", None)
    if pw_mod is None:
        pytest.skip("gpu4dft.pw nanobind submodule not wired yet")

    a_bohr, coords_bohr = _diamond_geometry(5.43)
    cell = pbc_gto.M(
        a=a_bohr,
        atom=[("Si", coords_bohr[0].tolist()), ("Si", coords_bohr[1].tolist())],
        unit="B",
        pseudo="gth-pade",
        verbose=0,
    )
    if hasattr(pw_mod, "ewald_energy"):
        ew = pw_mod.ewald_energy(a_bohr, [14, 14], coords_bohr, "gth-pade")
        assert abs(float(ew) - float(cell.ewald())) < 1e-11

