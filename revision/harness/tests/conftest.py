
import os
import sys
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

REVISION_DIR = Path(__file__).resolve().parents[2]

if str(REVISION_DIR) not in sys.path:
    sys.path.insert(0, str(REVISION_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running gate tests (G2 law equivalence; ~20 min)")
