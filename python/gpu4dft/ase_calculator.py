"""Atomic Simulation Environment (ASE) calculator interface for GPU4DFT.

Works seamlessly with ``ase.Atoms`` when ``ase`` is installed, and also provides
a clean standalone calculator interface testable without ``ase`` installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _core
from ._core import Atom, Basis, Shell
from .scf import MoleculeSpec, RHF, RKS, UHF, UKS

try:
    from ase.calculators.calculator import Calculator as _ASECalculatorBase
    from ase.calculators.calculator import PropertyNotImplementedError
    from ase.calculators.calculator import all_changes as _ALL_CHANGES
    from ase.units import Bohr as _BOHR_IN_ANGSTROM
    from ase.units import Hartree as _HARTREE_IN_EV

    HAVE_ASE = True
except ImportError:  # pragma: no cover - clean standalone fallback when ASE is absent
    HAVE_ASE = False
    PropertyNotImplementedError = NotImplementedError  # type: ignore[misc,assignment]
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
        default_parameters: dict[str, Any] = {}

        def __init__(self, **kwargs: Any) -> None:
            self.parameters: dict[str, Any] = self.get_default_parameters()
            self.results: dict[str, Any] = {}
            self.atoms: Any = None
            if kwargs:
                self.set(**kwargs)

        def get_default_parameters(self) -> dict[str, Any]:
            return dict(self.default_parameters)

        def reset(self) -> None:
            self.atoms = None
            self.results = {}

        def set(self, **kwargs: Any) -> dict[str, Any]:
            changed: dict[str, Any] = {}
            for key, val in kwargs.items():
                if key not in self.parameters or self.parameters[key] != val:
                    changed[key] = val
                self.parameters[key] = val
            if changed:
                self.reset()
            return changed

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
        charge: int | None = 0,
        spin: int | None = None,
    ) -> None:
        self._numbers = np.asarray(numbers, dtype=int)
        self._positions = np.asarray(positions, dtype=np.float64)
        if self._positions.ndim != 2 or self._positions.shape != (len(self._numbers), 3):
            raise ValueError("positions must have shape (natom, 3) in Angstrom")
        q = 0 if charge is None else int(charge)
        self.info: dict[str, Any] = {"charge": q}
        if spin is not None:
            self.info["spin"] = int(spin)
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
        raw_q = self.info.get("charge", 0)
        q = 0 if raw_q is None else int(raw_q)
        out = SimpleAtoms(self._numbers.copy(), self._positions.copy(), q)
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


def _freeze_value(val: Any) -> Any:
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, Basis):
        return ("Basis", id(val))
    if isinstance(val, dict):
        return ("dict", tuple(sorted((str(k), _freeze_value(v)) for k, v in val.items())))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze_value(x) for x in val)
    return ("repr", repr(val))


def _hashable_basis_key(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, str):
        return ("str", val)
    if isinstance(val, Basis):
        return ("Basis", id(val))
    if isinstance(val, dict):
        return ("dict", tuple(sorted((str(k), _freeze_value(v)) for k, v in val.items())))
    return _freeze_value(val)


def _build_molecule_for_atoms(
    atoms_obj: Any, basis_name: Any, charge: int, spin: int = 0
) -> Any:
    numbers = [int(z) for z in atoms_obj.get_atomic_numbers()]
    pos_ang = np.asarray(atoms_obj.get_positions(), dtype=np.float64)

    try:
        from pyscf import gto

        atom_spec = [
            (z, (float(pos_ang[i, 0]), float(pos_ang[i, 1]), float(pos_ang[i, 2])))
            for i, z in enumerate(numbers)
        ]
        return gto.M(
            atom=atom_spec,
            basis=basis_name,
            charge=charge,
            spin=int(spin),
            unit="Angstrom",
            cart=False,
            verbose=0,
        )
    except ImportError:
        if not isinstance(basis_name, str) or basis_name.strip().lower() != "sto-3g":
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
        return MoleculeSpec(
            basis=Basis(shells), atoms=atom_list, nelectron=nelec, spin=int(spin)
        )


class GPU4DFTCalculator(_ASECalculatorBase):
    """ASE-compatible calculator driving GPU4DFT's C++20/CUDA RHF/RKS/UHF/UKS engines.

    Energies are returned in **eV** and forces in **eV / Angstrom**, matching ASE conventions.
    """

    implemented_properties = ["energy", "forces"]
    default_parameters: dict[str, Any] = {
        "method": None,
        "xc": "b3lyp",
        "basis": "def2-svp",
        "charge": 0,
        "spin": None,
        "dispersion": None,
        "density_fit": False,
        "auxbasis": "def2-universal-jkfit",
        "snk": False,
        "backend": None,
        "precision": "fp64",
        "dynamic_precision": None,
        "gemm_mode": "fp64",
        "solver": "diag",
        "grids_level": 3,
        "conv_tol": 1e-9,
        "max_cycle": 100,
    }

    def __init__(
        self,
        *,
        method: str | None = None,
        xc: str = "b3lyp",
        basis: str | dict = "def2-svp",
        charge: int | None = 0,
        spin: int | None = None,
        dispersion: str | None = None,
        density_fit: bool = False,
        auxbasis: str | dict | Basis | None = "def2-universal-jkfit",
        snk: bool = False,
        backend: str | None = None,
        precision: str = "fp64",
        dynamic_precision: bool | None = None,
        gemm_mode: str = "fp64",
        solver: str = "diag",
        grids_level: int = 3,
        conv_tol: float = 1e-9,
        max_cycle: int = 100,
        **kwargs: Any,
    ) -> None:
        self.last_mf: RHF | RKS | UHF | UKS | None = None
        self._last_dm_key: tuple | None = None
        self._last_coords_ang: np.ndarray | None = None
        self._last_full_sig: tuple | None = None
        super().__init__(
            method=method,
            xc=xc,
            basis=basis,
            charge=charge,
            spin=spin,
            dispersion=dispersion,
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
            max_cycle=max_cycle,
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        mf = getattr(self, "last_mf", None)
        if mf is not None:
            if hasattr(mf, "release"):
                mf.release()
            else:
                mf._engine = None
        self._last_coords_ang = None
        self._last_full_sig = None
        _core.trim_device_pool()

    def close(self) -> None:
        mf = getattr(self, "last_mf", None)
        if mf is not None:
            if hasattr(mf, "release"):
                mf.release()
            else:
                mf._engine = None
        self.last_mf = None
        self._last_dm_key = None
        self._last_coords_ang = None
        self._last_full_sig = None
        super().reset()
        _core.trim_device_pool()

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
        else:
            changes = list(super().check_state(atoms, tol=tol))

        old_info = getattr(self.atoms, "info", {}) or {}
        new_info = getattr(atoms, "info", {}) or {}
        old_q = old_info.get("charge", 0)
        new_q = new_info.get("charge", 0)
        if (0 if old_q is None else int(old_q)) != (0 if new_q is None else int(new_q)):
            if "initial_charges" not in changes:
                changes.append("initial_charges")
        old_s = old_info.get("spin", None)
        new_s = new_info.get("spin", None)
        if (None if old_s is None else int(old_s)) != (None if new_s is None else int(new_s)):
            if "initial_magmoms" not in changes:
                changes.append("initial_magmoms")
        return changes

    def calculate(
        self,
        atoms: Any = None,
        properties: list[str] | tuple[str, ...] = ("energy",),
        system_changes: list[str] = _ALL_CHANGES,
    ) -> None:
        for prop in properties:
            if prop not in ("energy", "free_energy", "forces"):
                raise PropertyNotImplementedError(
                    f"Property {prop!r} is not implemented by GPU4DFTCalculator"
                )
        prev_results = dict(self.results)
        if isinstance(atoms, SimpleAtoms):
            if system_changes:
                self.results = {}
            self.atoms = atoms.copy()
        else:
            super().calculate(atoms, properties, system_changes)
        target = self.atoms if atoms is None else atoms
        if target is None:
            raise ValueError("GPU4DFTCalculator.calculate requires an atoms object")

        p = self.parameters
        info_dict = getattr(target, "info", {}) or {}
        if "charge" in info_dict and info_dict["charge"] is not None:
            charge = int(info_dict["charge"])
        elif p.get("charge") is not None:
            charge = int(p["charge"])
        else:
            charge = 0
        if "spin" in info_dict and info_dict["spin"] is not None:
            spin = int(info_dict["spin"])
        elif p.get("spin") is not None:
            spin = int(p["spin"])
        elif hasattr(target, "get_initial_magnetic_moments"):
            spin = int(round(float(np.sum(target.get_initial_magnetic_moments()))))
        else:
            spin = 0
        if spin < 0:
            raise ValueError(
                f"GPU4DFTCalculator requires non-negative spin (N_alpha - N_beta >= 0); got spin={spin}"
            )

        xc_name = str(p["xc"]).strip()
        is_hf = xc_name.lower() in ("", "hf", "rhf", "uhf")
        raw_method = p.get("method")
        if raw_method is not None:
            method = str(raw_method).strip().lower()
        else:
            method = ("uhf" if is_hf else "uks") if spin != 0 else ("rhf" if is_hf else "rks")
        if method not in ("rhf", "rks", "uhf", "uks"):
            raise ValueError(f"Unsupported method {method!r}; choose 'rhf', 'rks', 'uhf', or 'uks'")

        scf_kwargs: dict[str, Any] = dict(
            backend=p["backend"],
            precision=p["precision"],
            dynamic_precision=p["dynamic_precision"],
            gemm_mode=p["gemm_mode"],
            solver=p["solver"],
            grids_level=int(p["grids_level"]),
            conv_tol=float(p["conv_tol"]),
            max_cycle=int(p.get("max_cycle", 100)),
            dispersion=p.get("dispersion"),
        )
        for key in (
            "diis_tol",
            "diis_space",
            "damping",
            "level_shift",
            "lindep_tol",
            "schwarz_cutoff",
            "fp32_threshold",
            "ozaki_slices",
            "early_gemm_mode",
            "incremental",
            "use_cuda_graph",
            "init_guess",
        ):
            if key in p and p[key] is not None:
                scf_kwargs[key] = p[key]
        grids_partition = p.get("grids_partition")
        grids_prune = p.get("grids_prune")

        basis_val = p.get("basis", "def2-svp")
        aux_val = p.get("auxbasis", "def2-universal-jkfit")
        numbers = tuple(int(z) for z in target.get_atomic_numbers())
        dm_key = (
            numbers,
            charge,
            spin,
            method,
            xc_name.lower(),
            _hashable_basis_key(basis_val),
            bool(p["density_fit"]),
            _hashable_basis_key(aux_val),
            bool(p["snk"]),
        )
        full_sig = (
            dm_key,
            tuple(sorted((k, _freeze_value(v)) for k, v in scf_kwargs.items())),
            _freeze_value(grids_partition),
            None if grids_prune is None else bool(grids_prune),
            tuple(
                sorted(
                    (k, _freeze_value(p.get(k)))
                    for k in ("snk_overlap_fitting", "snk_pair_cutoff", "snk_lindep_tol", "snk_precision")
                )
            ),
        )

        pos_ang = np.asarray(target.get_positions(), dtype=np.float64)
        if (
            not system_changes
            and self.last_mf is not None
            and self.last_mf.converged
            and self.last_mf.result is not None
            and self._last_dm_key == dm_key
            and self._last_full_sig == full_sig
            and self._last_coords_ang is not None
            and self._last_coords_ang.shape == pos_ang.shape
            and np.allclose(self._last_coords_ang, pos_ang, rtol=0.0, atol=1e-15)
        ):
            e_hartree = float(self.last_mf.e_tot)
            energy_ev = e_hartree * _HARTREE_IN_EV
            self.results["energy"] = energy_ev
            self.results["free_energy"] = energy_ev
            self.results["energy_hartree"] = e_hartree
            if "forces" in properties:
                if "forces" in prev_results and "gradient_eh_bohr" in prev_results:
                    self.results["forces"] = prev_results["forces"]
                    self.results["gradient_eh_bohr"] = prev_results["gradient_eh_bohr"]
                else:
                    if self.last_mf._engine is None:
                        self.last_mf.kernel(dm0=self.last_mf.make_rdm1())
                    grad_eh_bohr = self.last_mf.nuc_grad_method().kernel()
                    forces_ev_ang = -np.asarray(grad_eh_bohr, dtype=np.float64) * (
                        _HARTREE_IN_EV / _BOHR_IN_ANGSTROM
                    )
                    self.results["forces"] = forces_ev_ang
                    self.results["gradient_eh_bohr"] = grad_eh_bohr
            return

        dm0 = None
        if self.last_mf is not None:
            if (
                self.last_mf.converged
                and self._last_dm_key == dm_key
                and self.last_mf.result is not None
            ):
                dm0 = self.last_mf.make_rdm1()
            if hasattr(self.last_mf, "release"):
                self.last_mf.release()
            else:
                self.last_mf._engine = None
        if self._last_dm_key is not None and self._last_dm_key != dm_key:
            _core.trim_device_pool()

        mol = _build_molecule_for_atoms(target, basis_val, charge, spin=spin)

        mf: RHF | RKS | UHF | UKS
        if method == "rhf":
            mf = RHF(mol, **scf_kwargs)
        elif method == "rks":
            mf = RKS(mol, xc=xc_name, **scf_kwargs)
        elif method == "uhf":
            mf = UHF(mol, spin=spin, **scf_kwargs)
        else:
            mf = UKS(mol, xc=xc_name, spin=spin, **scf_kwargs)

        if p["density_fit"]:
            mf.density_fit(auxbasis=aux_val)
        if p["snk"]:
            snk_kwargs: dict[str, Any] = {}
            for sk_src, sk_dst in (
                ("snk_overlap_fitting", "overlap_fitting"),
                ("snk_pair_cutoff", "pair_cutoff"),
                ("snk_lindep_tol", "lindep_tol"),
                ("snk_precision", "precision"),
            ):
                if sk_src in p and p[sk_src] is not None:
                    snk_kwargs[sk_dst] = p[sk_src]
            mf.snk(**snk_kwargs)
        if grids_partition is not None:
            mf.grids.partition = grids_partition
        if grids_prune is not None:
            mf.grids.prune = bool(grids_prune)

        e_hartree = mf.kernel(dm0=dm0)
        if not mf.converged and dm0 is not None:
            e_hartree = mf.kernel(dm0=None)
        if not mf.converged:
            if hasattr(mf, "release"):
                mf.release()
            else:
                mf._engine = None
            raise RuntimeError(
                f"GPU4DFT SCF did not converge in {mf.iterations} iterations "
                f"(final DIIS RMS = {mf.result.scf.final_diis_rms:.3e})"
            )
        self.last_mf = mf
        self._last_dm_key = dm_key
        self._last_coords_ang = pos_ang.copy()
        self._last_full_sig = full_sig

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
