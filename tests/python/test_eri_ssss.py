"""(ss|ss) integrals against PySCF/libcint."""

import numpy as np
import pytest

import gpu4dft

gto = pytest.importorskip("pyscf.gto")

# One generally contracted shell (two contractions sharing primitives).
GENERAL_CONTRACTION_H = """
H    S
     13.0107010         0.0334986   0.0
      1.9622572         0.2347270   0.0
      0.4445380         0.8137573   0.0
      0.1219496         0.0         1.0
"""


def _random_h_cluster(natom, seed):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-2.5, 2.5, size=(natom, 3))
    return "; ".join(f"H {x:.6f} {y:.6f} {z:.6f}" for x, y, z in coords)


MOLECULES = {
    "H2/sto-3g": dict(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g"),
    "H4/6-31g": dict(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.1; H 0.3 0 3.0", basis="6-31g"),
    "HeH+/3-21g": dict(atom="He 0 0 0; H 0 0 0.77", basis="3-21g", charge=1),
    "H2/general-contraction": dict(
        atom="H 0 0 0; H 0.2 0.1 0.9",
        basis={"H": gto.basis.parse(GENERAL_CONTRACTION_H)},
    ),
    # Uncontracted even-tempered set: exponents 0.03 ... 7.3, i.e. diffuse to
    # tight, which exercises both branches of the Boys function.
    "H8/even-tempered": dict(
        atom=_random_h_cluster(8, seed=7),
        basis={"H": gto.etbs([(0, 6, 0.03, 3.0)])},
    ),
}


@pytest.fixture(scope="module", params=sorted(MOLECULES))
def system(request):
    mol = gto.M(**MOLECULES[request.param], unit="Angstrom", verbose=0)
    return mol, gpu4dft.basis_from_pyscf(mol), mol.intor("int2e", aosym="s1")


# Absolute tolerances relative to max |(ij|kl)|. Observed on the host backend:
# ~1e-14 (fp64) and ~1e-7 (fp32).
TOLERANCES = {"fp64": 1e-13, "fp32": 1e-6}


@pytest.mark.parametrize("precision", sorted(TOLERANCES))
@pytest.mark.parametrize("backend", ["host", "cuda"])
def test_matches_pyscf(system, backend, precision):
    if not gpu4dft.backend_available(backend):
        pytest.skip(f"{backend} backend not available")
    mol, basis, ref = system
    assert basis.nao == mol.nao

    eri = gpu4dft.eri_tensor(basis, backend=backend, precision=precision)

    assert eri.shape == (mol.nao,) * 4
    scale = np.abs(ref).max()
    np.testing.assert_allclose(eri, ref, rtol=0, atol=TOLERANCES[precision] * scale)


def test_permutational_symmetry(system):
    _, basis, _ = system
    eri = gpu4dft.eri_tensor(basis)
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        np.testing.assert_allclose(eri, eri.transpose(perm), rtol=0, atol=1e-14)


def test_shells_beyond_s_not_implemented():
    # 2p shells on O/sto-3g are now implemented up to g (l <= 4) and match PySCF;
    # angular momentum beyond g (l = 5, h-shell) raises NotImplementedError.
    mol = gto.M(atom="O 0 0 0", basis="sto-3g", verbose=0)  # has a 2p shell
    ref = mol.intor("int2e", aosym="s1")
    eri = gpu4dft.eri_tensor(gpu4dft.basis_from_pyscf(mol))
    np.testing.assert_allclose(eri, ref, rtol=0, atol=1e-13 * np.abs(ref).max())

    h_basis = gpu4dft.Basis([gpu4dft.Shell(5, [0.0, 0.0, 0.0], [1.0], [1.0])])
    with pytest.raises(NotImplementedError):
        gpu4dft.eri_tensor(h_basis)


def test_unknown_backend_name():
    with pytest.raises(ValueError, match="unknown Backend"):
        gpu4dft.backend_available("opencl")


@pytest.mark.parametrize(
    "shell_args, message",
    [
        ((0, [0, 0, 0], [], []), "no primitives"),
        ((0, [0, 0, 0], [1.0, 2.0], [1.0]), "2 exponents but 1 coefficients"),
        ((0, [0, 0, 0], [-1.0], [1.0]), "positive"),
        ((0, [0, 0, 0], [1.0], [0.0]), "zero norm"),
        ((-1, [0, 0, 0], [1.0], [1.0]), "angular momentum"),
    ],
)
def test_invalid_shells_rejected(shell_args, message):
    with pytest.raises(ValueError, match=message):
        gpu4dft.Basis([gpu4dft.Shell(*shell_args)])


def test_kernel_registry_lists_host_kernels():
    names = gpu4dft.kernel_names()
    assert {"eri_ssss_host_fp64", "eri_ssss_host_fp32"} <= set(names)
