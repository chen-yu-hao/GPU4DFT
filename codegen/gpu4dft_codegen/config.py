"""What to generate: integral classes, precisions and tuning parameters."""

from __future__ import annotations

from dataclasses import dataclass

ANGULAR_LETTERS = "spdfghik"


@dataclass(frozen=True)
class EriClass:
    """A two-electron integral class (la lb | lc ld)."""

    la: int
    lb: int
    lc: int
    ld: int

    @property
    def angular(self) -> tuple[int, int, int, int]:
        return (self.la, self.lb, self.lc, self.ld)

    @property
    def name(self) -> str:
        return "".join(ANGULAR_LETTERS[l] for l in self.angular)


@dataclass(frozen=True)
class Precision:
    """Arithmetic precision of a kernel variant.

    ``ctype`` is the type the integral math is evaluated in. Accumulation over
    primitives is always done in double, so FP32 variants trade accuracy only
    in the per-primitive evaluation.
    """

    name: str
    ctype: str
    enum: str


PRECISIONS: tuple[Precision, ...] = (
    Precision(name="fp64", ctype="double", enum="FP64"),
    Precision(name="fp32", ctype="float", enum="FP32"),
)


@dataclass(frozen=True)
class CudaTuning:
    """Launch parameters for the CUDA driver of one integral class.

    These are the knobs an autotuner will search over once it exists.
    """

    block_size: int = 128


# Integral classes to generate, with their per-class tuning parameters.
ERI_CLASSES: dict[EriClass, CudaTuning] = {
    EriClass(0, 0, 0, 0): CudaTuning(block_size=128),
}
