"""
Makes `src/` importable as the package root (e.g. `import pinn`,
`import common`) regardless of how pytest is invoked or whether
PYTHONPATH=src has been set manually.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
