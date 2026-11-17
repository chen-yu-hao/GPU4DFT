"""Assemble the full set of generated files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .backends import (
    Backend,
    CudaBackend,
    HipBackend,
    HostBackend,
    emit_launchers_header,
    emit_registry,
)
from .config import ERI_CLASSES, CudaTuning, EriClass
from .deriv import DERIV1_HEADER_RELPATH, emit_deriv1_header
from .eri import GENERAL_HEADER_RELPATH, emit_class_header, emit_general_header, header_relpath

BACKENDS: dict[str, Backend] = {
    b.name: b for b in (HostBackend(), CudaBackend(), HipBackend())
}

LAUNCHERS_HEADER = "include/gpu4dft/generated/eri_launchers.hpp"
REGISTRY_SOURCE = "src/eri_registry.cpp"


def _resolve_backends(names: Iterable[str]) -> list[Backend]:
    resolved: list[Backend] = []
    for name in names:
        if name not in BACKENDS:
            raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)}")
        if BACKENDS[name] not in resolved:
            resolved.append(BACKENDS[name])
    if not resolved:
        raise ValueError("at least one backend is required")
    return resolved


def generate(
    backends: Sequence[str],
    classes: Mapping[EriClass, CudaTuning] = ERI_CLASSES,
    deriv_order: int = 1,
) -> dict[str, str]:
    """Return {path relative to the output directory: file contents}."""
    if deriv_order not in (0, 1):
        raise ValueError(f"unsupported deriv_order {deriv_order}; expected 0 or 1")
    selected = _resolve_backends(backends)
    class_list = list(classes)
    files: dict[str, str] = {}
    for cls, tuning in classes.items():
        files[header_relpath(cls)] = emit_class_header(cls, tuning)
    files[GENERAL_HEADER_RELPATH] = emit_general_header()
    if deriv_order >= 1:
        files[DERIV1_HEADER_RELPATH] = emit_deriv1_header()
    for backend in selected:
        for cls, tuning in classes.items():
            files[backend.source_relpath(cls)] = backend.emit_source(cls, tuning)
        files[backend.general_source_relpath()] = backend.emit_general_source()
    files[LAUNCHERS_HEADER] = emit_launchers_header(class_list, selected)
    files[REGISTRY_SOURCE] = emit_registry(class_list, selected)
    return files


def list_outputs(
    backends: Sequence[str],
    classes: Mapping[EriClass, CudaTuning] = ERI_CLASSES,
    deriv_order: int = 1,
) -> list[str]:
    return list(generate(backends, classes=classes, deriv_order=deriv_order))


def write_outputs(out_dir: Path, files: Mapping[str, str]) -> None:
    for relpath, text in files.items():
        path = out_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
