# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Drives `npu_op_disposition_probe` once per op and writes the census."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_PROBE = Path(__file__).resolve().parent / "npu_op_disposition_probe.py"

_BOOTSTRAP = """
import json, runpy, sys
sys.path[:] = json.loads(sys.argv[1])
sys.argv = [sys.argv[2], sys.argv[3]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def _run(case: str) -> dict:
    env = os.environ.copy()
    # A process of its own is not enough: the compile cache is on disk and
    # shared, so a kernel another case built is served back and lands in this
    # case's MEF. That turns an absence claim into a false negative -- an op
    # that names no selector looks as though it named one. A cache directory
    # per case is what actually isolates them.
    with tempfile.TemporaryDirectory(prefix=f"npu_probe_{case}_") as cache:
        env["MODULAR_HOME"] = cache
        env["TMPDIR"] = cache
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                _BOOTSTRAP,
                json.dumps(sys.path),
                str(_PROBE),
                case,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=900,
        )
    staged: dict | None = None
    for line in proc.stdout.splitlines():
        if line.startswith("###JSON###"):
            return json.loads(line[len("###JSON###") :])
        if line.startswith("###STAGED###"):
            staged = json.loads(line[len("###STAGED###") :])
    if staged is not None:
        # The compiler aborted the process. Staging still happened, and its
        # verdict is the one this census is about.
        staged["outcome"] = "COMPILER_ABORTED"
        staged["error"] = proc.stderr.strip().splitlines()[-1][:200]
        return staged
    return {
        "case": case,
        "outcome": "PROBE_NO_OUTPUT",
        "returncode": proc.returncode,
        "stderr": proc.stderr[-3000:],
        "stdout": proc.stdout[-1500:],
    }


def test_census() -> None:
    cases = json.loads(
        subprocess.run(
            [
                sys.executable,
                "-c",
                _BOOTSTRAP,
                json.dumps(sys.path),
                str(_PROBE),
                "--list",
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=True,
        )
        .stdout.strip()
        .splitlines()[-1]
    )
    only = os.environ.get("NPU_CENSUS_ONLY")
    if only:
        wanted = set(only.split(","))
        cases = [c for c in cases if c in wanted]

    sys.path.insert(0, str(_PROBE.parent))
    from npu_op_disposition_probe import classify

    records = []
    for case in cases:
        rec = _run(case)
        rec["rung"] = classify(rec)
        records.append(rec)
        print(f"[{case}] {rec['rung']}", flush=True)

    out = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp")
    path = Path(out) / "npu_op_disposition.json"
    path.write_text(json.dumps(records, indent=2))
    print(f"WROTE {path}")
    print("###CENSUS###" + json.dumps(records))
