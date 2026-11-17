"""Atomic Simulation Environment (ASE) calculator interface for GPU4DFT.

Works seamlessly with ``ase.Atoms`` when ``ase`` is installed, and also provides
a clean standalone calculator interface testable without ``ase`` installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._core import Atom, Basis, Shell
from .scf import MoleculeSpec, RHF, RKS

try:
    from ase.calculators.calculator import Calculator as _ASECalculatorBase
    from ase.calculators.calculator import all_changes as _ALL_CHANGES
    from ase.units import Bohr as _BOHR_IN_ANGSTROM
    from ase.units import Hartree as _HARTREE_IN_EV

    HAVE_ASE = True
except ImportError:  # pragma: no cover - clean standalone fallback when ASE is absent
    HAVE_ASE = False
    _ALL_CHANGES = [
        "positions",
        "numbers",
        "cell",
        "pbc",
        "initial_charges",
        "initial_magmoms",
    ]
    # CODATA 2018 recommended values matching ASE / PySCF
    _HARTREE_IN_EV = 27.211386245988
    _BOHR_IN_ANGSTROM = 0.529177210903

    class _ASECalculatorBase:  # type: ignore[no-redef]
        implemented_properties: list[str] = ["energy", "forces"]

        def __init__(self, **kwargs: Any) -> None:
            self.parameters: dict[str, Any] = dict(kwargs)
            self.results: dict[str, Any] = {}
            self.atoms: Any = None

        def check_state(self, atoms: Any, tol: float = 1e-15) -> list[str]:
            if self.atoms is None:
                return list(_ALL_CHANGES)
            old_z = np.asarray(self.atoms.get_atomic_numbers(), dtype=int)
            new_z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            if old_z.shape != new_z.shape or not np.array_equal(old_z, new_z):
                return ["numbers", "positions"]
            old_pos = np.asarray(self.atoms.get_positions(), dtype=np.float64)
            new_pos = np.asarray(atoms.get_positions(), dtype=np.float64)
            if not np.allclose(old_pos, new_pos, rtol=0.0, atol=tol):
                return ["positions"]
            return []

        def calculate(
            self,
            atoms: Any = None,
            properties: list[str] | tuple[str, ...] = ("energy",),
            system_changes: list[str] = _ALL_CHANGES,
        ) -> None:
            if atoms is not None:
                self.atoms = atoms.copy() if hasattr(atoms, "copy") else atoms

        def get_potential_energy(self, atoms: Any = None, force_consistent: bool = False) -> float:
            _ = force_consistent
            target = atoms if atoms is not None else self.atoms
            changes = self.check_state(target)
            if changes or "energy" not in self.results:
                self.calculate(atoms=target, properties=["energy"], system_changes=changes)
            return float(self.results["energy"])

        def get_forces(self, atoms: Any = None) -> np.ndarray:
            target = atoms if atoms is not None else self.atoms
            changes = self.check_state(target)
            if changes or "forces" not in self.results:
                self.calculate(
                    atoms=target, properties=["energy", "forces"], system_changes=changes
                )
            return np.asarray(self.results["forces"], dtype=np.float64)


# Compact embedded STO-3G table (H, He, C, N, O, F) for zero-PySCF fallback testing.
_EMBEDDED_STO3G: dict[int, list[tuple[int, list[float], list[float]]]] = {
    1: [(0, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454])],
    2: [(0, [6.36242139, 1.15892300, 0.31364979], [0.15432897, 0.53532814, 0.44463454])],
    6: [
        (0, [71.6168370, 13.0450960, 3.53051220], [0.15432897, 0.53532814, 0.44463454]),
        (0, [2.94124940, 0.68348310, 0.22228990], [-0.09996723, 0.39951283, 0.70011547]),
        (1, [2.94124940, 0.68348310, 0.22228990], [0.15591627, 0.60768372, 0.39195739]),
    ],
    7: [
        (0, [99.1061690, 18.0523120, 4.88566020], [0.15432897, 0.53532814, 0.44463454]),
        (0, [3.78045590, 0.87849660, 0.28571440], [-0.09996723, 0.39951283, 0.70011547]),
        (1, [3.78045590, 0.87849660, 0.28571440], [0.15591627, 0.60768372, 0.39195739]),
    ],
    8: [
        (0, [130.709320, 23.8088610, 6.44360830], [0.15432897, 0.53532814, 0.44463454]),
        (0, [5.03315130, 1.16959610, 0.38038900], [-0.09996723, 0.39951283, 0.70011547]),
        (1, [5.03315130, 1.16959610, 0.38038900], [0.15591627, 0.60768372, 0.39195739]),
    ],
    9: [
        (0, [166.679130, 30.3608120, 8.21682070], [0.15432897, 0.53532814, 0.44463454]),
        (0, [6.46480320, 1.50228120, 0.48858850], [-0.09996723, 0.39951283, 0.70011547]),
        (1, [6.46480320, 1.50228120, 0.48858850], [0.15591627, 0.60768372, 0.39195739]),
    ],
}


class SimpleAtoms:
    """Minimal duck-typed ``ase.Atoms`` replacement for environments without ``ase`` installed."""

    def __init__(
        self,
        numbers: list[int] | np.ndarray,
        positions: list[list[float]] | np.ndarray,
        charge: int = 0,
    ) -> None:
        self._numbers = np.asarray(numbers, dtype=int)
        self._positions = np.asarray(positions, dtype=np.float64)
        if self._positions.ndim != 2 or self._positions.shape != (len(self._numbers), 3):
            raise ValueError("positions must have shape (natom, 3) in Angstrom")
        self.info: dict[str, Any] = {"charge": int(charge)}
        self.calc: Any = None

    def get_atomic_numbers(self) -> np.ndarray:
        return self._numbers.copy()

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()

    def set_positions(self, new_positions: list[list[float]] | np.ndarray) -> None:
        arr = np.asarray(new_positions, dtype=np.float64)
        if arr.shape != self._positions.shape:
            raise ValueError("new_positions shape mismatch")
        self._positions = arr.copy()

    def copy(self) -> SimpleAtoms:
        out = SimpleAtoms(self._numbers.copy(), self._positions.copy(), int(self.info.get("charge", 0)))
        out.info = dict(self.info)
        return out

    def get_potential_energy(self) -> float:
        if self.calc is None:
            raise RuntimeError("No calculator attached to SimpleAtoms")
        return float(self.calc.get_potential_energy(self))

    def get_forces(self) -> np.ndarray:
        if self.calc is None:
            raise RuntimeError("No calculator attached to SimpleAtoms")
        return np.asarray(self.calc.get_forces(self), dtype=np.float64)


def _build_molecule_for_atoms(atoms_obj: Any, basis_name: str, charge: int) -> Any:
    numbers = [int(z) for z in atoms_obj.get_atomic_numbers()]
    pos_ang = np.asarray(atoms_obj.get_positions(), dtype=np.float64)

    try:
        from pyscf import gto

        atom_spec = [
            (z, (float(pos_ang[i, 0]), float(pos_ang[i, 1]), float(pos_ang[i, 2])))
            for i, z in enumerate(numbers)
        ]
        return gto.M(atom=atom_spec, basis=basis_name, charge=charge, unit="Angstrom", cart=False, verbose=0)
    except ImportError:
        if basis_name.strip().lower() != "sto-3g":
            raise ImportError(
                f"pyscf is required to parse basis {basis_name!r}; only 'sto-3g' is embedded standalone"
            ) from None
        pos_bohr = pos_ang / _BOHR_IN_ANGSTROM
        shells: list[Shell] = []
        atom_list: list[Atom] = []
        for i, z in enumerate(numbers):
            if z not in _EMBEDDED_STO3G:
                raise ValueError(f"Element Z={z} is not in embedded standalone STO-3G table")
            c_bohr = [float(pos_bohr[i, 0]), float(pos_bohr[i, 1]), float(pos_bohr[i, 2])]
            atom_list.append(Atom(z, c_bohr))
            for l_ang, exps, coeffs in _EMBEDDED_STO3G[z]:
                shells.append(Shell(l_ang, c_bohr, list(exps), list(coeffs)))
        nelec = int(sum(numbers) - charge)
        return MoleculeSpec(basis=Basis(shells), atoms=atom_list, nelectron=nelec)


class GPU4DFTCalculator(_ASECalculatorBase):
    """ASE-compatible calculator driving GPU4DFT's C++20/CUDA RHF & RKS engines.

    Energies are returned in **eV** and forces in **eV / Angstrom**, matching ASE conventions.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        xc: str = "b3lyp",
        basis: str = "def2-svp",
        charge: int = 0,
        density_fit: bool = False,
        auxbasis: str = "def2-universal-jkfit",
        snk: bool = False,
        backend: str | None = None,
        precision: str = "fp64",
        dynamic_precision: bool | None = None,
        gemm_mode: str = "fp64",
        solver: str = "diag",
        grids_level: int = 3,
        conv_tol: float = 1e-9,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            xc=xc,
            basis=basis,
            charge=charge,
            density_fit=density_fit,
            auxbasis=auxbasis,
            snk=snk,
            backend=backend,
            precision=precision,
            dynamic_precision=dynamic_precision,
            gemm_mode=gemm_mode,
            solver=solver,
            grids_level=grids_level,
            conv_tol=conv_tol,
            **kwargs,
        )
        self.last_mf: RHF | RKS | None = None

    def check_state(self, atoms: Any, tol: float = 1e-15) -> list[str]:
        if self.atoms is None:
            return list(_ALL_CHANGES)
        if (
            not HAVE_ASE
            or isinstance(atoms, SimpleAtoms)
            or isinstance(self.atoms, SimpleAtoms)
            or not hasattr(atoms, "cell")
        ):
            changes: list[str] = []
            old_z = np.asarray(self.atoms.get_atomic_numbers(), dtype=int)
            new_z = np.asarray(atoms.get_atomic_numbers(), dtype=int)
            if old_z.shape != new_z.shape or not np.array_equal(old_z, new_z):
                changes.append("numbers")
            old_pos = np.asarray(self.atoms.get_positions(), dtype=np.float64)
            new_pos = np.asarray(atoms.get_positions(), dtype=np.float64)
            if old_pos.shape != new_pos.shape or not np.allclose(
                old_pos, new_pos, rtol=0.0, atol=tol
            ):
                changes.append("positions")
            old_q = int((getattr(self.atoms, "info", {}) or {}).get("charge", 0))
            new_q = int((getattr(atoms, "info", {}) or {}).get("charge", 0))
            if old_q != new_q:
                changes.append("initial_charges")
            return changes
        return list(super().check_state(atoms, tol=tol))

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = _ALL_CHANGES,
    ) -> None:
        if isinstance(atoms, SimpleAtoms):
            self.atoms = atoms.copy()
        else:
            super().calculate(atoms, properties, system_changes)
        target = self.atoms if atoms is None else atoms
        if target is None:
            raise ValueError("GPU4DFTCalculator.calculate requires an atoms object")

        p = self.parameters
        info_dict = getattr(target, "info", {}) or {}
        charge = int(info_dict.get("charge", p.get("charge", 0)))
        mol = _build_molecule_for_atoms(target, str(p["basis"]), charge)

        xc_name = str(p["xc"]).strip()
        is_hf = xc_name.lower() in ("", "hf", "rhf")
        scf_kwargs = dict(
            backend=p["backend"],
            precision=p["precision"],
            dynamic_precision=p["dynamic_precision"],
            gemm_mode=p["gemm_mode"],
            solver=p["solver"],
            grids_level=int(p["grids_level"]),
            conv_tol=float(p["conv_tol"]),
        )
        mf: RHF | RKS = RHF(mol, **scf_kwargs) if is_hf else RKS(mol, xc=xc_name, **scf_kwargs)

        if p["density_fit"]:
            mf.density_fit(auxbasis=str(p["auxbasis"]))
        if p["snk"]:
            mf.snk()

        e_hartree = mf.kernel()
        if not mf.converged:
            raise RuntimeError(
                f"GPU4DFT SCF did not converge in {mf.iterations} iterations "
                f"(final DIIS RMS = {mf.result.scf.final_diis_rms:.3e})"
            )
        self.last_mf = mf

        energy_ev = float(e_hartree) * _HARTREE_IN_EV
        self.results["energy"] = energy_ev
        self.results["free_energy"] = energy_ev
        self.results["energy_hartree"] = float(e_hartree)

        if "forces" in properties:
            grad_eh_bohr = mf.nuc_grad_method().kernel()
            forces_ev_ang = -np.asarray(grad_eh_bohr, dtype=np.float64) * (
                _HARTREE_IN_EV / _BOHR_IN_ANGSTROM
            )
            self.results["forces"] = forces_ev_ang
            self.results["gradient_eh_bohr"] = grad_eh_bohr
