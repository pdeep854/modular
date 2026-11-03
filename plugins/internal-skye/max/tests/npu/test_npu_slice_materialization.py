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
"""Drives the slice-materialization probe and prints its table.

Also pins the fuse-views contiguity verdict per slice form: a byte-contiguous
slice behind static unit dims folds to zero-copy `mogg.slice`, and the strided
forms (non-unit step, two sliced axes) keep their copying materialize. The
verdict is asserted, not just printed, because the contiguity predicates
(`isContiguousSliceKernel` and its slice-of-slice sibling
`isContiguousSliceChainKernel`, FuseViews.cpp) are gates: widening one past
the byte-range contract is the gate-loosening failure class, and narrowing
one re-buys `npu_view_copy` host walks.

Each verdict case runs in a subprocess of its own, like the op disposition
census: the instrument here is the compile's IR dumps, and a process that has
already run dozens of sibling `InferenceSession`s stops writing them -- the
`batch2_step2` compile came back with zero IR stages late in a full-suite
process while passing standalone. Isolation keeps the instrument readable at
the point of the verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from npu_slice_materialization_probe import (
    VERDICT_CASES,
    probe,
    render,
)
from npu_slice_materialization_probe import (
    virtual_npu as _virtual_npu,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

_PROBE = Path(__file__).resolve().parent / "npu_slice_materialization_probe.py"

_BOOTSTRAP = """
import json, runpy, sys
sys.path[:] = json.loads(sys.argv[1])
sys.argv = [sys.argv[2], sys.argv[3]]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    yield from _virtual_npu()


def test_slice_materialization_census(
    virtual_npu: None, tmp_path: Path
) -> None:
    for symbolic in (False, True):
        records = probe(
            _KERNELS, tmp_path / f"sym_{symbolic}", symbolic=symbolic
        )
        print(f"\n=== symbolic sequence axis: {symbolic} ===")
        print(render(records) + "\n")
        print(json.dumps(records, indent=1))


def _run_verdict(case: str) -> dict:
    env = os.environ.copy()
    # A process of its own is not enough: the compile cache is on disk and
    # shared, and a compile served from it may skip the IR dumps this verdict
    # is read from. A cache directory per case is what isolates them.
    with tempfile.TemporaryDirectory(prefix=f"npu_verdict_{case}_") as cache:
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
    for line in proc.stdout.splitlines():
        if line.startswith("###JSON###"):
            return json.loads(line[len("###JSON###") :])
    # Unlike the census, which records a dead probe as an outcome, this is a
    # falsifier: a case that produced no record fails, loudly and with the
    # child's own words.
    pytest.fail(
        f"{case}: the verdict probe died without a record"
        f" (rc={proc.returncode}); stderr tail:\n{proc.stderr[-3000:]}"
    )


def test_unit_batch_contiguity_verdicts() -> None:
    for (
        name,
        _shape,
        _slice_fn,
        _control_shape,
        want,
        min_slices,
    ) in VERDICT_CASES:
        rec = _run_verdict(name)
        assert "case_dumped" in rec, (
            f"{name}: the verdict probe crashed before compiling:"
            f" {json.dumps(rec, indent=1)}"
        )
        for leg in ("control", "case"):
            assert rec[f"{leg}_dumped"], (
                f"{name}: no fuse-views dump was written for the {leg}"
                " compile; its compile diagnostic: "
                + (rec[f"{leg}_error"] or "(the compile raised nothing)")
            )
        assert rec["case_pre_slice_ops"] >= min_slices, (
            f"{name}: {rec['case_pre_slice_ops']} mo.slice(s) staged where"
            f" the case needs {min_slices}, so the verdict below would be"
            " vacuous"
        )
        base = rec["control_materialize"]
        delta = rec["case_materialize"] - base
        if want == "fold":
            assert delta == 0, (
                f"{name}: a byte-contiguous slice must cross zero-copy"
                f" (fused or raised to mogg.slice), but it costs {delta}"
                f" materialize(s) over the none control's {base}"
            )
        else:
            assert delta == 1, (
                f"{name}: a strided slice must stay exactly one copying"
                f" kernel, got {delta} materialize(s) over the none"
                f" control's {base}"
            )
