"""What to generate: integral classes, precisions and tuning parameters."""

from __future__ import annotations

from dataclasses import dataclass

ANGULAR_LETTERS = "spdfghik"
MAX_SUPPORTED_L = 4  # s, p, d, f, g


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
    def total_l(self) -> int:
        return self.la + self.lb + self.lc + self.ld

    @property
    def n_sph_components(self) -> int:
        return (2 * self.la + 1) * (2 * self.lb + 1) * (2 * self.lc + 1) * (2 * self.ld + 1)

    @property
    def n_cart_components(self) -> int:
        def nc(l: int) -> int:
            return (l + 1) * (l + 2) // 2

        return nc(self.la) * nc(self.lb) * nc(self.lc) * nc(self.ld)

    @property
    def name(self) -> str:
        return "".join(
            ANGULAR_LETTERS[l] if 0 <= l < len(ANGULAR_LETTERS) else f"l{l}"
            for l in self.angular
        )


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
    """Launch and codegen parameters for GPU/host drivers of one integral class.

    Parameters
    ----------
    block_size:
        Threads per GPU thread block (must be a multiple of 32 when
        ``thread_mapping == "warp"``).
    thread_mapping:
        ``"thread"`` maps one thread to one shell quartet; ``"warp"`` maps one
        32-lane warp (or wavefront subwarp) cooperatively across primitive
        quartets using warp-shuffle reductions.
    prim_unroll:
        Unroll factor hint ``#pragma unroll`` on inner primitive loops.
    algorithm:
        ``"auto"`` selects McMurchie-Davidson (``"md"``) for ``L <= 3`` and
        Rys quadrature + 2D VRR (``"rys"``) for ``L >= 4``; may also be set
        explicitly to ``"md"`` or ``"rys"``.
    """

    block_size: int = 128
    thread_mapping: str = "thread"
    prim_unroll: int = 1
    algorithm: str = "auto"

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.thread_mapping not in ("thread", "warp"):
            raise ValueError(
                f"thread_mapping must be 'thread' or 'warp', got {self.thread_mapping!r}"
            )
        if self.thread_mapping == "warp" and self.block_size % 32 != 0:
            raise ValueError("block_size must be a multiple of 32 for warp thread_mapping")
        if self.prim_unroll < 1:
            raise ValueError(f"prim_unroll must be >= 1, got {self.prim_unroll}")
        if self.algorithm not in ("auto", "md", "rys"):
            raise ValueError(f"algorithm must be 'auto', 'md', or 'rys', got {self.algorithm!r}")

    def resolved_algorithm(self, cls: EriClass) -> str:
        if self.algorithm != "auto":
            return self.algorithm
        return "md" if cls.total_l <= 3 else "rys"


def canonical_classes_up_to(max_l: int) -> list[EriClass]:
    """Return all canonical (la >= lb, lc >= ld, (la,lb) >= (lc,ld)) classes."""
    pair_classes = [(la, lb) for la in range(max_l + 1) for lb in range(la + 1)]
    out: list[EriClass] = []
    for i, (la, lb) in enumerate(pair_classes):
        for lc, ld in pair_classes[: i + 1]:
            out.append(EriClass(la, lb, lc, ld))
    return out


def _default_tuning(cls: EriClass) -> CudaTuning:
    if cls.angular == (0, 0, 0, 0):
        return CudaTuning(block_size=128, thread_mapping="thread", prim_unroll=1, algorithm="md")
    if cls.total_l <= 3:
        return CudaTuning(block_size=128, thread_mapping="thread", prim_unroll=2, algorithm="md")
    return CudaTuning(block_size=128, thread_mapping="thread", prim_unroll=1, algorithm="rys")


# Specialized unrolled canonical integral classes up to d shells (l <= 2: 21 classes).
# Higher angular momenta up to g shells (l <= 4) are handled seamlessly via the
# generated general Rys fallback kernel (eri_general).
ERI_CLASSES: dict[EriClass, CudaTuning] = {
    cls: _default_tuning(cls) for cls in canonical_classes_up_to(2)
}
