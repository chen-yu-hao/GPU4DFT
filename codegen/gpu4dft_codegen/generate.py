"""Assemble the full set of generated files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .backends import (
    Backend,
    CudaBackend,
    HostBackend,
    emit_launchers_header,
    emit_registry,
)
from .config import ERI_CLASSES, CudaTuning, EriClass
from .eri import emit_class_header, header_relpath

BACKENDS: dict[str, Backend] = {b.name: b for b in (HostBackend(), CudaBackend())}

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
) -> dict[str, str]:
    """Return {path relative to the output directory: file contents}."""
    selected = _resolve_backends(backends)
    class_list = list(classes)
    files: dict[str, str] = {}
    for cls in class_list:
        files[header_relpath(cls)] = emit_class_header(cls)
    for backend in selected:
        for cls, tuning in classes.items():
            files[backend.source_relpath(cls)] = backend.emit_source(cls, tuning)
    files[LAUNCHERS_HEADER] = emit_launchers_header(class_list, selected)
    files[REGISTRY_SOURCE] = emit_registry(class_list, selected)
    return files


def list_outputs(backends: Sequence[str]) -> list[str]:
    return list(generate(backends))


def write_outputs(out_dir: Path, files: Mapping[str, str]) -> None:
    for relpath, text in files.items():
        path = out_dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
