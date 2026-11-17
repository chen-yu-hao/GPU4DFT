"""Code generator for GPU4DFT integral kernels.

The generator turns a declarative description of the integral classes we want
(angular momenta, precisions, per-backend tuning parameters, derivative order)
into C++/CUDA/HIP sources. Every class gets:

* one backend-neutral header holding the integral math as ``GPU4DFT_HD``
  functions, shared verbatim by the host and the device builds, so the CPU
  reference backend validates exactly the arithmetic the GPU runs;
* one driver per backend (Host OpenMP, CUDA, AMD ROCm HIP) that maps the work
  onto that backend's parallelism (thread-per-quartet or warp-cooperative
  primitive shuffle reduction);
* an entry in a generated kernel registry plus a general Rys evaluator up to
  ``g`` shells (``l <= 4``) and analytical 1st nuclear derivatives.

Run ``python -m gpu4dft_codegen --help`` for the command-line interface.
"""

from .autotune import AutotuneResult, benchmark_class, candidate_tunings
from .backends import CudaBackend, HipBackend, HostBackend
from .config import (
    ERI_CLASSES,
    MAX_SUPPORTED_L,
    PRECISIONS,
    CudaTuning,
    EriClass,
    Precision,
    canonical_classes_up_to,
)
from .deriv import emit_deriv1_header
from .generate import BACKENDS, generate, list_outputs, write_outputs

__all__ = [
    "AutotuneResult",
    "BACKENDS",
    "CudaBackend",
    "CudaTuning",
    "ERI_CLASSES",
    "EriClass",
    "HipBackend",
    "HostBackend",
    "MAX_SUPPORTED_L",
    "PRECISIONS",
    "Precision",
    "benchmark_class",
    "candidate_tunings",
    "canonical_classes_up_to",
    "emit_deriv1_header",
    "generate",
    "list_outputs",
    "write_outputs",
]
