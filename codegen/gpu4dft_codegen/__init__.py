"""Code generator for GPU4DFT integral kernels.

The generator turns a declarative description of the integral classes we want
(angular momenta, precisions, per-backend tuning parameters) into C++/CUDA
sources. Every class gets:

* one backend-neutral header holding the integral math as ``GPU4DFT_HD``
  functions, shared verbatim by the host and the device builds, so the CPU
  reference backend validates exactly the arithmetic the GPU runs;
* one driver per backend (host loop, CUDA kernel + launcher) that maps the
  work onto that backend's parallelism;
* an entry in a generated kernel registry that the runtime dispatches on.

Run ``python -m gpu4dft_codegen --help`` for the command-line interface.
"""

from .config import ERI_CLASSES, PRECISIONS, CudaTuning, EriClass, Precision
from .generate import BACKENDS, generate, list_outputs, write_outputs

__all__ = [
    "BACKENDS",
    "ERI_CLASSES",
    "PRECISIONS",
    "CudaTuning",
    "EriClass",
    "Precision",
    "generate",
    "list_outputs",
    "write_outputs",
]
