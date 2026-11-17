"""Autotuning search space and CUDA microbenchmark harness for ERI kernels.

Enumerates valid :class:`CudaTuning` configurations across ``block_size``,
``thread_mapping`` (``"thread"`` vs ``"warp"`` cooperative primitive shuffles),
``prim_unroll``, and ``algorithm`` (``"md"`` vs ``"rys"``), and optionally
compiles + benchmarks each candidate with ``nvcc`` when a CUDA GPU is present.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .backends import CudaBackend, emit_launchers_header
from .config import CudaTuning, EriClass
from .eri import emit_class_header, header_relpath


@dataclass(frozen=True)
class AutotuneResult:
    cls: EriClass
    tuning: CudaTuning
    elapsed_ms: float
    quartets_per_sec: float


def candidate_tunings(
    cls: EriClass,
    block_sizes: Sequence[int] = (64, 128, 256),
    thread_mappings: Sequence[str] = ("thread", "warp"),
    prim_unrolls: Sequence[int] = (1, 2, 4),
    algorithms: Sequence[str] | None = None,
) -> list[CudaTuning]:
    """Generate the candidate tuning space for ``cls``."""
    if algorithms is None:
        algorithms = ("md", "rys") if cls.total_l <= 4 else ("rys",)
    out: list[CudaTuning] = []
    for alg in algorithms:
        for mapping in thread_mappings:
            unrolls = (1,) if mapping == "warp" else prim_unrolls
            for bs in block_sizes:
                if mapping == "warp" and bs % 32 != 0:
                    continue
                for unroll in unrolls:
                    out.append(
                        CudaTuning(
                            block_size=bs,
                            thread_mapping=mapping,
                            prim_unroll=unroll,
                            algorithm=alg,
                        )
                    )
    return out


def _emit_bench_driver(cls: EriClass, npair: int, nprim: int, repeats: int) -> str:
    nsph = cls.n_sph_components
    return f"""\
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include "gpu4dft/generated/eri_launchers.hpp"

int main() {{
  int dev_count = 0;
  if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) return 2;

  const int npair = {npair};
  const int nprim = {nprim};
  const int tot_prim = npair * nprim;

  std::vector<int> h_off(npair + 1);
  for (int i = 0; i <= npair; ++i) h_off[i] = i * nprim;
  std::vector<double> h_exp(tot_prim, 1.2), h_P(tot_prim, 0.1), h_pref(tot_prim, 0.05);
  std::vector<double> h_A(npair, 0.0), h_B(npair, 0.7), h_alpha(tot_prim, 0.6);

  int* d_off = nullptr;
  double *d_exp = nullptr, *d_Px = nullptr, *d_Py = nullptr, *d_Pz = nullptr, *d_pref = nullptr;
  double *d_Ax = nullptr, *d_Ay = nullptr, *d_Az = nullptr;
  double *d_Bx = nullptr, *d_By = nullptr, *d_Bz = nullptr;
  double *d_aa = nullptr, *d_ab = nullptr, *d_out = nullptr;

  cudaMalloc(&d_off, h_off.size() * sizeof(int));
  cudaMemcpy(d_off, h_off.data(), h_off.size() * sizeof(int), cudaMemcpyHostToDevice);
  auto up = [](double** p, const std::vector<double>& v) {{
    cudaMalloc(p, v.size() * sizeof(double));
    cudaMemcpy(*p, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice);
  }};
  up(&d_exp, h_exp); up(&d_Px, h_P); up(&d_Py, h_P); up(&d_Pz, h_P); up(&d_pref, h_pref);
  up(&d_Ax, h_A); up(&d_Ay, h_A); up(&d_Az, h_A);
  up(&d_Bx, h_B); up(&d_By, h_B); up(&d_Bz, h_B);
  up(&d_aa, h_alpha); up(&d_ab, h_alpha);
  cudaMalloc(&d_out, static_cast<std::size_t>(npair) * npair * {nsph} * sizeof(double));

  gpu4dft::ShellPairView view{{
      npair, d_off, d_exp, d_Px, d_Py, d_Pz, d_pref,
      d_Ax, d_Ay, d_Az, d_Bx, d_By, d_Bz, d_aa, d_ab, nullptr, nullptr, nullptr, nullptr, nullptr}};

  // Warmup
  gpu4dft::generated::eri_{cls.name}_cuda_fp64(view, view, d_out, nullptr);
  cudaDeviceSynchronize();

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  for (int r = 0; r < {repeats}; ++r) {{
    gpu4dft::generated::eri_{cls.name}_cuda_fp64(view, view, d_out, nullptr);
  }}
  cudaEventRecord(ev1);
  cudaEventSynchronize(ev1);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev0, ev1);
  std::printf("%.6f\\n", double(ms) / {repeats});
  return 0;
}}
"""


def benchmark_class(
    cls: EriClass,
    repo_root: Path,
    candidates: Sequence[CudaTuning] | None = None,
    npair: int = 64,
    nprim: int = 3,
    repeats: int = 5,
    cuda_arch: str = "native",
) -> list[AutotuneResult]:
    """Compile and benchmark ``candidates`` on the active CUDA GPU via ``nvcc``."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found on PATH; run autotuning on a CUDA machine/Colab")

    tunings = list(candidates) if candidates is not None else candidate_tunings(cls)
    backend = CudaBackend()
    results: list[AutotuneResult] = []

    with tempfile.TemporaryDirectory(prefix=f"gpu4dft_tune_{cls.name}_") as tmp:
        tmp_dir = Path(tmp)
        launchers = tmp_dir / "include/gpu4dft/generated/eri_launchers.hpp"
        launchers.parent.mkdir(parents=True, exist_ok=True)
        launchers.write_text(emit_launchers_header([cls], [backend]))

        bench_cu = tmp_dir / "bench.cu"
        bench_cu.write_text(_emit_bench_driver(cls, npair=npair, nprim=nprim, repeats=repeats))

        for idx, tuning in enumerate(tunings):
            hdr = tmp_dir / header_relpath(cls)
            hdr.parent.mkdir(parents=True, exist_ok=True)
            hdr.write_text(emit_class_header(cls, tuning))

            kern_cu = tmp_dir / f"kern_{idx}.cu"
            kern_cu.write_text(backend.emit_source(cls, tuning))
            exe = tmp_dir / f"bench_{idx}"

            cmd = [
                nvcc,
                "-O3",
                "-std=c++20",
                f"-arch={cuda_arch}",
                f"-I{repo_root / 'include'}",
                f"-I{tmp_dir / 'include'}",
                str(kern_cu),
                str(bench_cu),
                "-o",
                str(exe),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            run = subprocess.run([str(exe)], check=True, capture_output=True, text=True)
            ms = float(run.stdout.strip())
            qps = (float(npair) * float(npair)) / max(ms * 1e-3, 1e-12)
            results.append(
                AutotuneResult(cls=cls, tuning=tuning, elapsed_ms=ms, quartets_per_sec=qps)
            )

    results.sort(key=lambda r: r.elapsed_ms)
    return results
