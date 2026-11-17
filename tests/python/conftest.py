import sys
from pathlib import Path

# Make the kernel generator importable for the codegen tests.
CODEGEN_DIR = Path(__file__).resolve().parents[2] / "codegen"
if str(CODEGEN_DIR) not in sys.path:
    sys.path.insert(0, str(CODEGEN_DIR))
