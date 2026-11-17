"""Comprehensive PySCF verification of arbitrary-l 1e/2e integrals, Schwarz
bounds, DirectJKEngine (FP64, FP32, Mixed, Hermite J-engine, Incremental Fock,
device-resident 2D-tiled GPU contraction, non-symmetric TDDFT transition DMs),
RIJKEngine (GPU-native 3c/2c + Cholesky/eigensolver + chunked GEMM, symmetric &
non-symmetric DMs), and 1e + 2e analytical nuclear gradients.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

import gpu4dft

gto = pytest.importorskip("pyscf.gto")
scf = pytest.importorskip("pyscf.scf")
df = pytest.importorskip("pyscf.df")
pyscf_grad = pytest.importorskip("pyscf.grad")


def _load_c_bridge():
    candidates = []
    env_so = os.environ.get("GPU4DFT_TEST_LIB")
    if env_so:
        candidates.append(Path(env_so))
    core_file = getattr(gpu4dft._core, "__file__", None)
    if core_file:
        p = Path(core_file)
        candidates.append(p)
        for sib in p.parent.parent.parent.rglob("libgpu4dft*.so"):
            candidates.append(sib)
    for path in candidates:
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            if hasattr(lib, "gpu4dft_c_direct_jk"):
                return lib
        except OSError:
            continue
    return None


_CLIB = _load_c_bridge()

_BACKEND_CODE = {"host": 0, "cuda": 1, "hip": 2}
_PREC_CODE = {"fp64": 0, "fp32": 1, "mixed": 2}


def _pack_mol_basis(mol):
    shell_l = []
    centers = []
    prim_offsets = [0]
    exps = []
    coeffs = []
    for ib in range(mol.nbas):
        l = int(mol.bas_angular(ib))
        coord = [float(x) for x in mol.bas_coord(ib)]
        b_exp = [float(x) for x in mol.bas_exp(ib)]
        ctr = mol.bas_ctr_coeff(ib)
        for ic in range(ctr.shape[1]):
            shell_l.append(l)
            centers.extend(coord)
            exps.extend(b_exp)
            coeffs.extend(float(c) for c in ctr[:, ic])
            prim_offsets.append(len(exps))
    return (
        len(shell_l),
        np.asarray(shell_l, dtype=np.int32),
        np.asarray(centers, dtype=np.float64),
        np.asarray(prim_offsets, dtype=np.int32),
        np.asarray(exps, dtype=np.float64),
        np.asarray(coeffs, dtype=np.float64),
    )


def _pack_mol_atoms(mol):
    z = np.asarray([int(mol.atom_charge(i)) for i in range(mol.natm)], dtype=np.int32)
    coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=np.float64).reshape(-1)
    return mol.natm, z, coords


def _ptr_i32(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


def _ptr_f64(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _eval_1e(mol, op_name: str, backend: str) -> np.ndarray:
    basis = gpu4dft.basis_from_pyscf(mol)
    if hasattr(gpu4dft, op_name):
        fn = getattr(gpu4dft, op_name)
        if op_name == "nuclear_attraction_matrix":
            atoms = [
                gpu4dft.Atom(int(mol.atom_charge(i)), list(mol.atom_coords(unit="Bohr")[i]))
                for i in range(mol.natm)
            ]
            return np.asarray(fn(basis, atoms, backend))
        return np.asarray(fn(basis, backend))

    if _CLIB is None:
        pytest.skip("1e bindings / C-ABI bridge not linked in this target")
    op_code = {"overlap_matrix": 0, "kinetic_matrix": 1, "nuclear_attraction_matrix": 2}[op_name]
    nsh, sl, sc, po, ex, co = _pack_mol_basis(mol)
    nat, az, ac = _pack_mol_atoms(mol)
    out = np.zeros((mol.nao, mol.nao), dtype=np.float64)
    rc = _CLIB.gpu4dft_c_one_electron(
        ctypes.c_int(op_code),
        ctypes.c_int(nsh),
        _ptr_i32(sl),
        _ptr_f64(sc),
        _ptr_i32(po),
        _ptr_f64(ex),
        _ptr_f64(co),
        ctypes.c_int(nat),
        _ptr_i32(az),
        _ptr_f64(ac),
        ctypes.c_int(_BACKEND_CODE[backend]),
        _ptr_f64(out),
    )
    assert rc == 0
    return out


def _eval_nuc_rep(mol):
    if hasattr(gpu4dft, "nuclear_repulsion_energy") and hasattr(gpu4dft, "nuclear_repulsion_grad"):
        atoms = [
            gpu4dft.Atom(int(mol.atom_charge(i)), list(mol.atom_coords(unit="Bohr")[i]))
            for i in range(mol.natm)
        ]
        return (
            float(gpu4dft.nuclear_repulsion_energy(atoms)),
            np.asarray(gpu4dft.nuclear_repulsion_grad(atoms)).reshape(mol.natm, 3),
        )
    if _CLIB is None:
        pytest.skip("C-ABI bridge not linked")
    nat, az, ac = _pack_mol_atoms(mol)
    e = ctypes.c_double(0.0)
    g = np.zeros((nat, 3), dtype=np.float64)
    rc = _CLIB.gpu4dft_c_nuc_repulsion(
        ctypes.c_int(nat), _ptr_i32(az), _ptr_f64(ac), ctypes.byref(e), _ptr_f64(g)
    )
    assert rc == 0
    return float(e.value), g


def _eval_direct_jk(
    mol,
    dm: np.ndarray,
    backend: str = "host",
    precision: str = "fp64",
    schwarz_cutoff: float = 1e-12,
    fp32_threshold: float = 1e-6,
    compute_j: bool = True,
    compute_k: bool = True,
    k_scale: float = 1.0,
    symmetrize_dm: bool = True,
    use_device_api: bool = False,
):
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    can_use_py = (
        hasattr(gpu4dft, "DirectJKEngine")
        and not use_device_api
        and (symmetrize_dm or hasattr(gpu4dft.JKOptions(), "symmetrize_dm"))
    )
    if can_use_py:
        basis = gpu4dft.basis_from_pyscf(mol)
        eng = gpu4dft.DirectJKEngine(basis)
        opts = gpu4dft.JKOptions()
        opts.backend = getattr(gpu4dft.Backend, backend)
        opts.precision = getattr(gpu4dft.Precision, precision)
        opts.schwarz_cutoff = schwarz_cutoff
        opts.fp32_threshold = fp32_threshold
        opts.compute_j = compute_j
        opts.compute_k = compute_k
        opts.k_scale = k_scale
        if hasattr(opts, "symmetrize_dm"):
            opts.symmetrize_dm = symmetrize_dm
        return eng.compute_jk(dm_c, opts)

    if _CLIB is None or not hasattr(_CLIB, "gpu4dft_c_direct_jk_ex"):
        pytest.skip("DirectJKEngine extended C-ABI bridge not linked")
    nsh, sl, sc, po, ex, co = _pack_mol_basis(mol)
    vj = np.zeros_like(dm_c)
    vk = np.zeros_like(dm_c)
    rc = _CLIB.gpu4dft_c_direct_jk_ex(
        ctypes.c_int(nsh),
        _ptr_i32(sl),
        _ptr_f64(sc),
        _ptr_i32(po),
        _ptr_f64(ex),
        _ptr_f64(co),
        _ptr_f64(dm_c),
        ctypes.c_int(_BACKEND_CODE[backend]),
        ctypes.c_int(_PREC_CODE[precision]),
        ctypes.c_double(schwarz_cutoff),
        ctypes.c_double(fp32_threshold),
        ctypes.c_int(1 if compute_j else 0),
        ctypes.c_int(1 if compute_k else 0),
        ctypes.c_double(k_scale),
        ctypes.c_int(1 if symmetrize_dm else 0),
        ctypes.c_int(1 if use_device_api else 0),
        ctypes.c_int(1),
        None,
        None,
        _ptr_f64(vj) if compute_j else None,
        _ptr_f64(vk) if compute_k else None,
    )
    assert rc == 0
    return vj, vk


def _eval_ri_jk(
    mol,
    auxmol,
    dm: np.ndarray,
    backend: str = "host",
    precision: str = "fp64",
    compute_j: bool = True,
    compute_k: bool = True,
    k_scale: float = 1.0,
    symmetrize_dm: bool = True,
    use_device_api: bool = False,
):
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    can_use_py = (
        hasattr(gpu4dft, "RIJKEngine")
        and not use_device_api
        and (symmetrize_dm or hasattr(gpu4dft.JKOptions(), "symmetrize_dm"))
    )
    if can_use_py:
        basis = gpu4dft.basis_from_pyscf(mol)
        aux_basis = gpu4dft.basis_from_pyscf(auxmol)
        be = getattr(gpu4dft.Backend, backend)
        eng = gpu4dft.RIJKEngine(basis, aux_basis, be)
        opts = gpu4dft.JKOptions()
        opts.backend = be
        opts.precision = getattr(gpu4dft.Precision, precision)
        opts.compute_j = compute_j
        opts.compute_k = compute_k
        opts.k_scale = k_scale
        if hasattr(opts, "symmetrize_dm"):
            opts.symmetrize_dm = symmetrize_dm
        return eng.compute_jk(dm_c, opts)

    if _CLIB is None or not hasattr(_CLIB, "gpu4dft_c_ri_jk_ex"):
        pytest.skip("RIJKEngine extended C-ABI bridge not linked")
    nsh, sl, sc, po, ex, co = _pack_mol_basis(mol)
    ansh, asl, asc, apo, aex, aco = _pack_mol_basis(auxmol)
    vj = np.zeros_like(dm_c)
    vk = np.zeros_like(dm_c)
    rc = _CLIB.gpu4dft_c_ri_jk_ex(
        ctypes.c_int(nsh),
        _ptr_i32(sl),
        _ptr_f64(sc),
        _ptr_i32(po),
        _ptr_f64(ex),
        _ptr_f64(co),
        ctypes.c_int(ansh),
        _ptr_i32(asl),
        _ptr_f64(asc),
        _ptr_i32(apo),
        _ptr_f64(aex),
        _ptr_f64(aco),
        _ptr_f64(dm_c),
        ctypes.c_int(_BACKEND_CODE[backend]),
        ctypes.c_int(_PREC_CODE[precision]),
        ctypes.c_int(1 if compute_j else 0),
        ctypes.c_int(1 if compute_k else 0),
        ctypes.c_double(k_scale),
        ctypes.c_int(1 if symmetrize_dm else 0),
        ctypes.c_int(1 if use_device_api else 0),
        ctypes.c_int(1),
        None,
        None,
        _ptr_f64(vj) if compute_j else None,
        _ptr_f64(vk) if compute_k else None,
    )
    assert rc == 0
    return vj, vk


def _eval_1e_and_2e_grad(mol, dm: np.ndarray, wdm: np.ndarray, k_fraction: float, backend: str):
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    wdm_c = np.ascontiguousarray(wdm, dtype=np.float64)
    if hasattr(gpu4dft, "one_electron_grad") and hasattr(gpu4dft, "jk_nuclear_grad"):
        basis = gpu4dft.basis_from_pyscf(mol)
        atoms = [
            gpu4dft.Atom(int(mol.atom_charge(i)), list(mol.atom_coords(unit="Bohr")[i]))
            for i in range(mol.natm)
        ]
        g1 = np.asarray(gpu4dft.one_electron_grad(basis, atoms, dm_c, wdm_c, backend)).reshape(
            mol.natm, 3
        )
        g2 = np.asarray(gpu4dft.jk_nuclear_grad(basis, dm_c, k_fraction, backend)).reshape(
            mol.natm, 3
        )
        return g1, g2

    if _CLIB is None:
        pytest.skip("Gradient bindings / C-ABI bridge not linked")
    nsh, sl, sc, po, ex, co = _pack_mol_basis(mol)
    nat, az, ac = _pack_mol_atoms(mol)
    g1 = np.zeros((nat, 3), dtype=np.float64)
    g2 = np.zeros((nat, 3), dtype=np.float64)
    nat_out = ctypes.c_int(0)
    rc1 = _CLIB.gpu4dft_c_one_elec_grad(
        ctypes.c_int(nsh),
        _ptr_i32(sl),
        _ptr_f64(sc),
        _ptr_i32(po),
        _ptr_f64(ex),
        _ptr_f64(co),
        ctypes.c_int(nat),
        _ptr_i32(az),
        _ptr_f64(ac),
        _ptr_f64(dm_c),
        _ptr_f64(wdm_c),
        ctypes.c_int(_BACKEND_CODE[backend]),
        _ptr_f64(g1),
    )
    rc2 = _CLIB.gpu4dft_c_jk_grad(
        ctypes.c_int(nsh),
        _ptr_i32(sl),
        _ptr_f64(sc),
        _ptr_i32(po),
        _ptr_f64(ex),
        _ptr_f64(co),
        _ptr_f64(dm_c),
        ctypes.c_double(k_fraction),
        ctypes.c_int(_BACKEND_CODE[backend]),
        ctypes.c_double(1e-13),
        _ptr_f64(g2),
        ctypes.byref(nat_out),
    )
    assert rc1 == 0 and rc2 == 0 and nat_out.value == nat
    return g1, g2


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize(
    "atom_spec, basis_name",
    [
        ("O 0.0 0.0 0.11; H 0.0 0.76 -0.47; H 0.0 -0.76 -0.47", "sto-3g"),
        ("O 0.0 0.0 0.11; H 0.0 0.76 -0.47; H 0.0 -0.76 -0.47", "def2-svp"),
        ("F 0.0 0.0 0.0; H 0.0 0.0 0.92", "def2-tzvp"),  # exercises s, p, d, f!
    ],
)
def test_eri_tensor_spdf_matches_pyscf(atom_spec, basis_name, backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(atom=atom_spec, basis=basis_name, unit="Angstrom", cart=False, verbose=0)
    ref = mol.intor("int2e", aosym="s1")
    basis = gpu4dft.basis_from_pyscf(mol)

    got64 = gpu4dft.eri_tensor(basis, backend=backend, precision="fp64")
    scale = max(1.0, float(np.max(np.abs(ref))))
    np.testing.assert_allclose(got64, ref, rtol=0, atol=1e-12 * scale)

    got32 = gpu4dft.eri_tensor(basis, backend=backend, precision="fp32")
    np.testing.assert_allclose(got32, ref, rtol=0, atol=2e-6 * scale)


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_eri_tensor_g_shell_matches_pyscf(backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(
        atom="O 0 0 0; H 0.3 -0.2 0.85",
        charge=-1,
        basis={
            "O": [[0, [1.5, 1.0]], [2, [0.9, 1.0]], [4, [0.75, 1.0]]],
            "H": [[0, [0.8, 1.0]], [1, [0.6, 1.0]]],
        },
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    ref = mol.intor("int2e", aosym="s1")
    got = gpu4dft.eri_tensor(gpu4dft.basis_from_pyscf(mol), backend=backend, precision="fp64")
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12 * max(1.0, float(np.max(np.abs(ref)))))


def test_shells_beyond_g_not_implemented():
    mol = gto.M(atom="Ne 0 0 0", basis={"Ne": [[5, [1.0, 1.0]]]}, cart=False, verbose=0)
    with pytest.raises(NotImplementedError):
        gpu4dft.eri_tensor(gpu4dft.basis_from_pyscf(mol))


def test_schwarz_bounds_exact_and_sorted():
    if _CLIB is None:
        pytest.skip("C-ABI bridge not linked")
    mol = gto.M(
        atom="O 0.0 0.0 0.1; H 0.0 0.76 -0.45; H 0.1 -0.75 -0.42",
        basis="def2-svp",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    eri = mol.intor("int2e", aosym="s1")
    basis = gpu4dft.basis_from_pyscf(mol)
    nsh, sl, sc, po, ex, co = _pack_mol_basis(mol)

    for la, lb in [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)]:
        npair = ctypes.c_int(0)
        rc = _CLIB.gpu4dft_c_schwarz_check(
            ctypes.c_int(nsh),
            _ptr_i32(sl),
            _ptr_f64(sc),
            _ptr_i32(po),
            _ptr_f64(ex),
            _ptr_f64(co),
            ctypes.c_int(la),
            ctypes.c_int(lb),
            ctypes.byref(npair),
            None,
            None,
            None,
        )
        assert rc == 0 and npair.value > 0
        n = npair.value
        bounds = np.zeros(n, dtype=np.float64)
        ish = np.zeros(n, dtype=np.int32)
        jsh = np.zeros(n, dtype=np.int32)
        _CLIB.gpu4dft_c_schwarz_check(
            ctypes.c_int(nsh),
            _ptr_i32(sl),
            _ptr_f64(sc),
            _ptr_i32(po),
            _ptr_f64(ex),
            _ptr_f64(co),
            ctypes.c_int(la),
            ctypes.c_int(lb),
            ctypes.byref(npair),
            _ptr_f64(bounds),
            _ptr_i32(ish),
            _ptr_i32(jsh),
        )
        assert np.all(np.diff(bounds) <= 1e-15)
        for p in range(n):
            i0 = basis.ao_offset(int(ish[p]))
            j0 = basis.ao_offset(int(jsh[p]))
            ni = 2 * la + 1
            nj = 2 * lb + 1
            sub = eri[i0 : i0 + ni, j0 : j0 + nj, i0 : i0 + ni, j0 : j0 + nj]
            diag = np.diagonal(sub.reshape(ni * nj, ni * nj))
            ref_q = np.sqrt(np.max(np.abs(diag)))
            np.testing.assert_allclose(bounds[p], ref_q, rtol=0, atol=1e-13)


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize("basis_name", ["6-31g*", "def2-tzvp"])
def test_one_electron_integrals_match_pyscf(basis_name, backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(
        atom="O 0.0 0.0 0.12; H 0.0 0.757 -0.469; H 0.05 -0.752 -0.465",
        basis=basis_name,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    S = _eval_1e(mol, "overlap_matrix", backend)
    T = _eval_1e(mol, "kinetic_matrix", backend)
    V = _eval_1e(mol, "nuclear_attraction_matrix", backend)

    np.testing.assert_allclose(S, mol.intor("int1e_ovlp"), rtol=0, atol=1e-12)
    np.testing.assert_allclose(T, mol.intor("int1e_kin"), rtol=0, atol=1e-12)
    np.testing.assert_allclose(V, mol.intor("int1e_nuc"), rtol=0, atol=1e-12)

    e_nuc, g_nuc = _eval_nuc_rep(mol)
    np.testing.assert_allclose(e_nuc, mol.energy_nuc(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(g_nuc, pyscf_grad.rhf.grad_nuc(mol), rtol=0, atol=1e-13)


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_direct_jk_fp64_mixed_hermite_and_incremental(backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(
        atom="C 0.0 0.0 0.0; O 0.0 0.0 1.21; H 0.0 0.94 -0.58; H 0.0 -0.94 -0.58",
        basis="def2-svp",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol).run()
    dm = mf.make_rdm1()
    vj_ref, vk_ref = scf.hf.get_jk(mol, dm, hermi=1)

    # 1. FP64 full J and K
    vj_64, vk_64 = _eval_direct_jk(mol, dm, backend=backend, precision="fp64", schwarz_cutoff=1e-13)
    np.testing.assert_allclose(vj_64, vj_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(vk_64, vk_ref, rtol=0, atol=1e-10)

    # 2. Device-resident API (`compute_jk_device`) on CUDA
    if backend == "cuda" and _CLIB is not None and hasattr(_CLIB, "gpu4dft_c_direct_jk_ex"):
        vj_dev, vk_dev = _eval_direct_jk(
            mol, dm, backend="cuda", precision="fp64", schwarz_cutoff=1e-13, use_device_api=True
        )
        np.testing.assert_allclose(vj_dev, vj_ref, rtol=0, atol=1e-10)
        np.testing.assert_allclose(vk_dev, vk_ref, rtol=0, atol=1e-10)

    # 3. Dynamic Mixed Precision (FP64 for Q_ab*Q_cd*D_max >= 1e-6, FP32+FP64-accum below)
    vj_mix, vk_mix = _eval_direct_jk(
        mol, dm, backend=backend, precision="mixed", schwarz_cutoff=1e-12, fp32_threshold=1e-6
    )
    np.testing.assert_allclose(vj_mix, vj_ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(vk_mix, vk_ref, rtol=0, atol=1e-9)

    # 4. Hermite J-engine path (compute_j=True, compute_k=False)
    vj_herm, _ = _eval_direct_jk(
        mol, dm, backend=backend, precision="fp64", compute_j=True, compute_k=False
    )
    np.testing.assert_allclose(vj_herm, vj_ref, rtol=0, atol=1e-10)

    # 5. Hybrid k_scale (e.g. B3LYP 0.20)
    _, vk_scaled = _eval_direct_jk(
        mol, dm, backend=backend, precision="fp64", compute_j=False, compute_k=True, k_scale=0.20
    )
    np.testing.assert_allclose(vk_scaled, 0.20 * vk_ref, rtol=0, atol=1e-10)

    # 6. Incremental Fock build: J(D_0) + Delta J(D - D_0) == J(D)
    dm0 = scf.hf.init_guess_by_minao(mol)
    vj0, vk0 = _eval_direct_jk(mol, dm0, backend=backend, precision="fp64", schwarz_cutoff=1e-13)
    dj, dk = _eval_direct_jk(
        mol, dm - dm0, backend=backend, precision="fp64", schwarz_cutoff=1e-12
    )
    np.testing.assert_allclose(vj0 + dj, vj_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(vk0 + dk, vk_ref, rtol=0, atol=1e-10)


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize(
    "atom_spec, basis_name",
    [
        ("C 0.0 0.0 0.0; O 0.0 0.0 1.21; H 0.0 0.94 -0.58; H 0.0 -0.94 -0.58", "def2-svp"),
        ("F 0.0 0.0 0.0; H 0.0 0.0 0.92", "def2-tzvp"),  # exercises s, p, d, f shells!
    ],
)
def test_nonsymmetric_dm_direct_jk_matches_exact_einsum(atom_spec, basis_name, backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(atom=atom_spec, basis=basis_name, unit="Angstrom", cart=False, verbose=0)
    eri = mol.intor("int2e", aosym="s1")
    rng = np.random.default_rng(42)

    # 1. General non-symmetric transition density matrix (X != X^T)
    dm_nonsym = rng.standard_normal((mol.nao, mol.nao))
    vj_ref = np.einsum("uvls,ls->uv", eri, dm_nonsym, optimize=True)
    vk_ref = np.einsum("ulvs,ls->uv", eri, dm_nonsym, optimize=True)
    assert not np.allclose(vk_ref, vk_ref.T, atol=1e-3)

    vj_got, vk_got = _eval_direct_jk(
        mol,
        dm_nonsym,
        backend=backend,
        precision="fp64",
        schwarz_cutoff=1e-14,
        symmetrize_dm=False,
    )
    np.testing.assert_allclose(vj_got, vj_ref, rtol=0, atol=1e-11)
    np.testing.assert_allclose(vk_got, vk_ref, rtol=0, atol=1e-11)

    # Verify device-resident `compute_jk_device` path on CUDA as well!
    if backend == "cuda":
        vj_dev, vk_dev = _eval_direct_jk(
            mol,
            dm_nonsym,
            backend="cuda",
            precision="fp64",
            schwarz_cutoff=1e-14,
            symmetrize_dm=False,
            use_device_api=True,
        )
        np.testing.assert_allclose(vj_dev, vj_ref, rtol=0, atol=1e-11)
        np.testing.assert_allclose(vk_dev, vk_ref, rtol=0, atol=1e-11)

    # Pure-J Hermite path with non-symmetric DM
    vj_only, _ = _eval_direct_jk(
        mol,
        dm_nonsym,
        backend=backend,
        precision="fp64",
        schwarz_cutoff=1e-14,
        compute_j=True,
        compute_k=False,
        symmetrize_dm=False,
    )
    np.testing.assert_allclose(vj_only, vj_ref, rtol=0, atol=1e-11)

    # 2. Strictly antisymmetric transition density (D^T = -D, used in Casida RPA A-B block!):
    #    J(D_anti) must vanish identically and K(D_anti) must be antisymmetric!
    raw = rng.standard_normal((mol.nao, mol.nao))
    dm_anti = raw - raw.T
    vk_anti_ref = np.einsum("ulvs,ls->uv", eri, dm_anti, optimize=True)
    vj_anti, vk_anti = _eval_direct_jk(
        mol,
        dm_anti,
        backend=backend,
        precision="fp64",
        schwarz_cutoff=1e-14,
        k_scale=0.25,
        symmetrize_dm=False,
    )
    np.testing.assert_allclose(vj_anti, np.zeros_like(vj_anti), rtol=0, atol=1e-12)
    np.testing.assert_allclose(vk_anti, 0.25 * vk_anti_ref, rtol=0, atol=1e-11)
    np.testing.assert_allclose(vk_anti + vk_anti.T, np.zeros_like(vk_anti), rtol=0, atol=1e-11)


@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_ri_jk_matches_pyscf_df(backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(
        atom="O 0.0 0.0 0.11; H 0.0 0.76 -0.47; H 0.0 -0.76 -0.47",
        basis="def2-svp",
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    auxmol = df.addons.make_auxmol(mol, auxbasis="def2-svp-jkfit")
    mf_df = scf.RHF(mol).density_fit(auxbasis="def2-svp-jkfit").run()
    dm = mf_df.make_rdm1()
    vj_df_ref, vk_df_ref = mf_df.get_jk(mol, dm, hermi=1)

    vj_ri, vk_ri = _eval_ri_jk(mol, auxmol, dm, backend=backend, precision="fp64")
    np.testing.assert_allclose(vj_ri, vj_df_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(vk_ri, vk_df_ref, rtol=0, atol=1e-10)

    if backend == "cuda" and _CLIB is not None and hasattr(_CLIB, "gpu4dft_c_ri_jk_ex"):
        vj_dev, vk_dev = _eval_ri_jk(
            mol, auxmol, dm, backend="cuda", precision="fp64", use_device_api=True
        )
        np.testing.assert_allclose(vj_dev, vj_df_ref, rtol=0, atol=1e-10)
        np.testing.assert_allclose(vk_dev, vk_df_ref, rtol=0, atol=1e-10)


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize(
    "atom_spec, basis_name, aux_name",
    [
        ("O 0.0 0.0 0.11; H 0.0 0.76 -0.47; H 0.0 -0.76 -0.47", "def2-svp", "def2-svp-jkfit"),
        ("F 0.0 0.0 0.0; H 0.0 0.0 0.92", "def2-tzvp", "def2-tzvp-jkfit"),  # f & g shells!
    ],
)
def test_nonsymmetric_dm_ri_jk_matches_exact_3c_einsum(atom_spec, basis_name, aux_name, backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(atom=atom_spec, basis=basis_name, unit="Angstrom", cart=False, verbose=0)
    auxmol = df.addons.make_auxmol(mol, auxbasis=aux_name)

    import scipy.linalg

    int3c = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    int2c = auxmol.intor("int2c2e", aosym="s1")
    chol = np.linalg.cholesky(int2c)
    b_ref = scipy.linalg.solve_triangular(
        chol, int3c.transpose(2, 0, 1).reshape(auxmol.nao, -1), lower=True
    ).reshape(auxmol.nao, mol.nao, mol.nao)
    eri_ri = np.einsum("Puv,Pls->uvls", b_ref, b_ref, optimize=True)

    rng = np.random.default_rng(123)
    dm_nonsym = rng.standard_normal((mol.nao, mol.nao))
    vj_ref = np.einsum("uvls,ls->uv", eri_ri, dm_nonsym, optimize=True)
    vk_ref = np.einsum("ulvs,ls->uv", eri_ri, dm_nonsym, optimize=True)

    vj_got, vk_got = _eval_ri_jk(
        mol, auxmol, dm_nonsym, backend=backend, precision="fp64", symmetrize_dm=False
    )
    np.testing.assert_allclose(vj_got, vj_ref, rtol=0, atol=1e-11)
    np.testing.assert_allclose(vk_got, vk_ref, rtol=0, atol=1e-11)

    if backend == "cuda":
        vj_dev, vk_dev = _eval_ri_jk(
            mol,
            auxmol,
            dm_nonsym,
            backend="cuda",
            precision="fp64",
            symmetrize_dm=False,
            use_device_api=True,
        )
        np.testing.assert_allclose(vj_dev, vj_ref, rtol=0, atol=1e-11)
        np.testing.assert_allclose(vk_dev, vk_ref, rtol=0, atol=1e-11)

        # Also check FP32 non-symmetric RI-K on CUDA
        _, vk_fp32 = _eval_ri_jk(
            mol,
            auxmol,
            dm_nonsym,
            backend="cuda",
            precision="fp32",
            compute_j=False,
            compute_k=True,
            k_scale=0.5,
            symmetrize_dm=False,
            use_device_api=True,
        )
        np.testing.assert_allclose(vk_fp32, 0.5 * vk_ref, rtol=0, atol=5e-5)


def test_ri_jk_near_singular_aux_metric_fallback():
    mol = gto.M(atom="O 0 0 0; H 0 0.76 0.58; H 0 -0.76 0.58", basis="sto-3g", verbose=0)
    # Deliberately duplicate an auxiliary shell to make the 2-center metric singular (rank-deficient),
    # exercising the GPU `cusolverDnDsyevd` inverse-sqrt fallback vs Host Jacobi fallback!
    dup_aux = {
        "O": [[0, [2.0, 1.0]], [0, [2.0, 1.0]], [1, [1.2, 1.0]]],
        "H": [[0, [1.0, 1.0]]],
    }
    auxmol = gto.M(atom=mol.atom, basis=dup_aux, verbose=0)
    dm = scf.RHF(mol).run().make_rdm1()

    vj_h, vk_h = _eval_ri_jk(mol, auxmol, dm, backend="host", precision="fp64")
    if gpu4dft.backend_available("cuda"):
        vj_c, vk_c = _eval_ri_jk(mol, auxmol, dm, backend="cuda", precision="fp64")
        np.testing.assert_allclose(vj_c, vj_h, rtol=0, atol=1e-10)
        np.testing.assert_allclose(vk_c, vk_h, rtol=0, atol=1e-10)


@pytest.mark.parametrize("backend", ["host", "cuda"])
@pytest.mark.parametrize("basis_name", ["def2-svp", "def2-tzvp"])
def test_analytical_nuclear_gradients_match_pyscf(basis_name, backend):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol = gto.M(
        atom="O 0.02 -0.01 0.14; H 0.05 0.81 -0.44; H -0.04 -0.72 -0.51",
        basis=basis_name,
        unit="Angstrom",
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    dm = mf.make_rdm1()
    g_scanner = pyscf_grad.RHF(mf)
    wdm = g_scanner.make_rdm1e()
    de_ref = g_scanner.kernel()

    _, g_nuc = _eval_nuc_rep(mol)
    g_1e, g_2e = _eval_1e_and_2e_grad(mol, dm, wdm, k_fraction=1.0, backend=backend)
    de_got = g_nuc + g_1e + g_2e

    np.testing.assert_allclose(de_got, de_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(np.sum(de_got, axis=0), np.zeros(3), rtol=0, atol=1e-12)
