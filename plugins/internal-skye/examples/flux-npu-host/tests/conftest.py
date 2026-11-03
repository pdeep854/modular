"""Puts the package and this directory on ``sys.path``.

The scaffold is intentionally not installed -- it is a numpy-only source tree
run in place, so tests import it from the checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parent
for entry in (_TESTS.parent, _TESTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))
