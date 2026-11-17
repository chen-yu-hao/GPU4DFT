"""Command-line entry point: ``python -m xc_codegen``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .emit_xc import generate, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="xc_codegen",
        description="Generate GPU4DFT symbolic XC functional kernels (Host/CUDA/HIP).",
    )
    parser.add_argument(
        "--backends",
        default="host,cuda,hip",
        help="optional backend list (xc_kernels.hpp is backend-neutral GPU4DFT_HD)",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--out-dir", type=Path, help="directory to write generated files into"
    )
    action.add_argument(
        "--list-outputs",
        action="store_true",
        help="print the generated file paths (relative, ';'-separated) and exit",
    )
    args = parser.parse_args(argv)

    files = generate()
    if args.list_outputs:
        sys.stdout.write(";".join(files))
        return 0
    write_outputs(args.out_dir, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
