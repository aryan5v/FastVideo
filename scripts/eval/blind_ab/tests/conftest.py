"""Make ``blind_ab`` importable as a package (it lives under scripts/eval/)."""
import sys
from pathlib import Path

SCRIPTS_EVAL = Path(__file__).resolve().parents[2]
if str(SCRIPTS_EVAL) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_EVAL))
