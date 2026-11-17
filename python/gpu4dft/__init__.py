"""GPU4DFT: GPU-first density functional theory.

Unimplemented features raise a subclass of the builtin ``NotImplementedError``.
"""

from __future__ import annotations

from . import _core
from ._core import Backend, Basis, Precision, Shell, kernel_names
from .pyscf_interface import basis_from_pyscf

__all__ = [
    "Backend",
    "Basis",
    "Precision",
    "Shell",
    "backend_available",
    "basis_from_pyscf",
    "eri_tensor",
    "kernel_names",
]


def _as_enum(enum_type, value):
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type[value]
    except KeyError:
        names = ", ".join(member.name for member in enum_type)
        raise ValueError(f"unknown {enum_type.__name__} {value!r}; choose from {names}") from None


def backend_available(backend: Backend | str) -> bool:
    """True if ``backend`` is compiled in and has a usable device."""
    return _core.backend_available(_as_enum(Backend, backend))


def eri_tensor(basis: Basis, backend: Backend | str = "host", precision: Precision | str = "fp64"):
    """Full (ij|kl) ERI tensor as a (nao, nao, nao, nao) NumPy array.

    Intended for validation on small systems; production code paths contract
    integrals on the fly instead of storing them.
    """
    return _core.eri_tensor(basis, _as_enum(Backend, backend), _as_enum(Precision, precision))
