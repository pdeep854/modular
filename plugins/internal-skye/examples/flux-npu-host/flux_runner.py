#!/usr/bin/env python3
"""One command: prompt to image, every compute stage a MAX MEF on the NPU.

The P1.3 thin runner. All behavior lives in :mod:`flux_klein_host.runner`;
this shim only makes the package importable when the file is run in place
from a checkout, the way the lane box runs it:

.. code-block:: bash

    python3 flux_runner.py --prompt "a red bicycle" --out image.png \\
        --tokenizer /path/to/tokenizer.json

Run ``--help`` for the backend matrix (``--encoder``/``--vae`` mef|hand) and
the knobs, and ``--dry-run`` to walk the whole orchestration with no device.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flux_klein_host.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
