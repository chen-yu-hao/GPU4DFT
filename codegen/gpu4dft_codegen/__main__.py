"""Command-line entry point: ``python -m gpu4dft_codegen``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .generate import BACKENDS, generate, write_outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu4dft_codegen", description="Generate GPU4DFT integral kernels."
    )
    parser.add_argument(
        "--backends",
        default="host",
        help=f"comma-separated backends to generate ({', '.join(BACKENDS)}); default: host",
    )
    parser.add_argument(
        "--deriv-order",
        type=int,
        default=1,
        choices=(0, 1),
        help="maximum nuclear derivative order to emit (0 or 1; default: 1)",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--out-dir", type=Path, help="directory to write generated files into")
    action.add_argument(
        "--list-outputs",
        action="store_true",
        help="print the generated file paths (relative, ';'-separated) and exit",
    )
    args = parser.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    try:
        files = generate(backends, deriv_order=args.deriv_order)
    except ValueError as exc:
        parser.error(str(exc))

    if args.list_outputs:
        sys.stdout.write(";".join(files))
        return 0
    write_outputs(args.out_dir, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())
