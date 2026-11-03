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
"""Cross-compiles the host-walk device-producer discriminator for Windows ARM64.

Every silicon proof of a host-walk transport (`npu_view_copy`, the concat and
split host copies) fed the walk from GRAPH INPUTS. The first whole-model graph
run interleaved walks with NPU-written intermediates and produced a
deterministic wrong output, and one direct failure: a host-produced
intermediate reached its NPU consumer unwritten (`position_ids[0] is
2147483648`). Whether a walk delivers when its producer is an NPU kernel is
exactly what no per-op proof establishes — the KB entry
`runtime/mid-graph-host-walks-are-unproven-with-device-written-producers`
carries the evidence, and this file builds the designed discriminator: the
smallest graphs that can confirm or falsify the transport class on silicon.

Five arms, each its own MEF, each run three-legged on the device (signed skel
present, withheld, restored) by `npu_hostwalk_probe_exec.c`:

- ``prodout``  X, W -> matmul -> A.                 Producer control: proves
  the matmul at this shape before any walk is blamed. Zero walks by
  construction, which is this file's in-test failing control for the IR
  instrument (a walk count that could not read zero could not be trusted to
  read one).
- ``tap``      X, W -> matmul -> slice -> S.        Device-write -> host-read
  in isolation: the walk materializes an NPU-written intermediate into a
  graph output.
- ``input``    A, V -> slice -> matmul -> out.      Host-read -> device-read
  with a graph-input producer: the walk's OUTPUT is a host-written
  intermediate consumed by an NPU kernel — the direction the `mo.range`
  failure indicts, from the transport the per-op proofs used.
- ``device``   X, W, V -> matmul -> slice -> matmul -> out. The designed
  three-op discriminator: NPU kernel writes A, `npu_view_copy` materializes
  A[:, 32:96] mid-graph, an NPU kernel consumes the walked bytes.
- ``device_big`` the same sandwich at a 256 KiB walk (512x256 producer), so a
  pass on the 512-byte arm cannot hide a size-dependent transport bug.

The slice is a COLUMN slice, so it is never a contiguous run the compiler
could fold to a pointer bump: it either materializes through this device's
walk or the probe is not probing. The IR is the instrument for that
(`mogg.call.materialize["…npu_view_copy…"]` in `.mogg-pre-mgp.mlir`, the
technique `test_npu_graph_movement.py` established); the MEF cannot carry it,
because a registration that calls no `load_function` puts no symbol there.

Values are exact by construction and injective, so a wrong byte on the device
names its own mechanism: `X[i,k] = 3*(64i+k)+1` (odd integers, ≡1 mod 3, max
766), `W` one-hot rows with the second half doubled (`A`'s halves are ≡1 and
≡2 mod 3, so every element of `A` is globally unique), `V` a signed one-hot
permutation (every element of the slice is read by exactly one output
element). Every value is an integer of magnitude ≤ 1532, inside fp16's exact
range, and every dot product has exactly one nonzero term, so accumulation
order cannot matter. Zero never legitimately occurs anywhere: a zero output
byte is unwritten memory, not arithmetic.

The C driver re-derives the same constructions and compares bitwise, no
epsilon, classifying each mismatch against the enumerable wrong mechanisms
(unwritten-zero, slice-from-offset-0, dense-stride confusion, walked-the-X-
buffer, foreign). This file and that driver must agree; the shared constants
are pinned by `_selfcheck` below.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
from max.driver import (
    NPU,
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DevicePlacementPolicy,
    DeviceRef,
    Graph,
    TensorType,
    ops,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# Windows ABI, ELF container — the same pair every windows sibling pins, and
# for the same reasons (`test_npu_graph_matmul_windows.py`).
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

# The small arms. K = 64 is the selector's granule; M and N are free.
_M, _K, _N = 4, 64, 128
_SLICE_LO, _SLICE_HI = 32, 96
_SW = _SLICE_HI - _SLICE_LO  # slice width, and the consumer's K and N

# The big arm: the same sandwich with a 256 KiB producer output and a 128 KiB
# walk, spanning many pages where the small arm spans a fraction of one.
_MB, _KB, _NB = 512, 128, 256
_SLICE_LO_B, _SLICE_HI_B = 64, 192
_SWB = _SLICE_HI_B - _SLICE_LO_B


def _x_small() -> np.ndarray:
    i = np.arange(_M)[:, None]
    k = np.arange(_K)[None, :]
    return (3 * (_K * i + k) + 1).astype(np.float16)


def _w_small() -> np.ndarray:
    w = np.zeros((_N, _K), dtype=np.float16)
    for j in range(_N):
        w[j, j % _K] = 1.0 if j < _K else 2.0
    return w


def _v(width: int) -> np.ndarray:
    v = np.zeros((width, width), dtype=np.float16)
    for q in range(width):
        v[q, (37 * q + 11) % width] = 1.0 if q % 2 == 0 else -1.0
    return v


def _x_big() -> np.ndarray:
    i = np.arange(_MB)[:, None]
    k = np.arange(_KB)[None, :]
    return (3 * ((7 * i + k) % 512) + 1).astype(np.float16)


def _w_big() -> np.ndarray:
    w = np.zeros((_NB, _KB), dtype=np.float16)
    for j in range(_NB):
        w[j, j % _KB] = 1.0 if j < _KB else 2.0
    return w


def _expected() -> dict[str, np.ndarray]:
    """Every arm's exact expectation, in float64 over exactly representable
    integers.

    numpy fp32 matmul on this box returns garbage (OpenBLAS DYNAMIC_ARCH), so
    nothing here multiplies matrices: each construction is one-hot by design
    and the expectations are gathers.
    """
    x = _x_small().astype(np.float64)
    s_j = np.where(np.arange(_N) < _K, 1.0, 2.0)
    a = s_j[None, :] * x[:, np.arange(_N) % _K]
    s = a[:, _SLICE_LO:_SLICE_HI]
    t_q = np.where(np.arange(_SW) % 2 == 0, 1.0, -1.0)
    out = t_q[None, :] * s[:, (37 * np.arange(_SW) + 11) % _SW]

    xb = _x_big().astype(np.float64)
    s_jb = np.where(np.arange(_NB) < _KB, 1.0, 2.0)
    ab = s_jb[None, :] * xb[:, np.arange(_NB) % _KB]
    sb = ab[:, _SLICE_LO_B:_SLICE_HI_B]
    t_qb = np.where(np.arange(_SWB) % 2 == 0, 1.0, -1.0)
    outb = t_qb[None, :] * sb[:, (37 * np.arange(_SWB) + 11) % _SWB]

    return {
        "prodout": a,
        "tap": s,
        "input": out,
        "device": out,
        "device_big": outb,
    }


def _selfcheck() -> None:
    """Pins the properties the driver's byte-level classifiers rely on."""
    exp = _expected()
    for name, arr in exp.items():
        assert np.all(arr == arr.astype(np.float16).astype(np.float64)), (
            f"{name}: a value is not exactly representable in fp16"
        )
        assert np.all(arr != 0.0), f"{name}: zero occurs legitimately"
    a = exp["prodout"]
    assert len(np.unique(a)) == a.size, "A's values are not globally unique"
    # The classifiers must be distinguishable from the truth where they are
    # counted: a slice taken from column 0 differs from the real slice
    # everywhere, on both geometries.
    assert np.all(a[:, 0:_SW] != exp["tap"]), "offset-0 collides with truth"


def _build(arm: str) -> Graph:
    dev = DeviceRef.NPU()
    f16 = DType.float16

    def t(shape: list[int]) -> TensorType:
        return TensorType(f16, shape, dev)

    inputs = {
        "prodout": [t([_M, _K]), t([_N, _K])],
        "tap": [t([_M, _K]), t([_N, _K])],
        "input": [t([_M, _N]), t([_SW, _SW])],
        "device": [t([_M, _K]), t([_N, _K]), t([_SW, _SW])],
        "device_big": [t([_MB, _KB]), t([_NB, _KB]), t([_SWB, _SWB])],
    }[arm]

    with Graph(
        f"npu_hostwalk_{arm}_win",
        input_types=inputs,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        if arm == "prodout":
            x, w = (v.tensor for v in graph.inputs)
            graph.output(x @ ops.transpose(w, -1, -2))
        elif arm == "tap":
            x, w = (v.tensor for v in graph.inputs)
            a = x @ ops.transpose(w, -1, -2)
            graph.output(a[:, _SLICE_LO:_SLICE_HI])
        elif arm == "input":
            a, v = (i.tensor for i in graph.inputs)
            s = a[:, _SLICE_LO:_SLICE_HI]
            graph.output(s @ ops.transpose(v, -1, -2))
        elif arm == "device":
            x, w, v = (i.tensor for i in graph.inputs)
            a = x @ ops.transpose(w, -1, -2)
            s = a[:, _SLICE_LO:_SLICE_HI]
            graph.output(s @ ops.transpose(v, -1, -2))
        else:  # device_big
            x, w, v = (i.tensor for i in graph.inputs)
            a = x @ ops.transpose(w, -1, -2)
            s = a[:, _SLICE_LO_B:_SLICE_HI_B]
            graph.output(s @ ops.transpose(v, -1, -2))
    return graph


# One walk in every arm that has a slice, and none in the producer control.
_WALKS = {"prodout": 0, "tap": 1, "input": 1, "device": 1, "device_big": 1}


def _materializers(dumps: Path) -> list[str]:
    """Returns the resolved symbol of every materialize that survived lowering.

    Read off `.mogg-pre-mgp.mlir`, after `mogg-fuse-views` erased the foldable
    ones — the count surviving there is what this graph walks at runtime, and
    the symbol says whose materializer serves it
    (`test_the_npu_materializer_is_the_one_that_runs` is the model).
    """
    dumped = list(dumps.glob("*.mogg-pre-mgp.mlir"))
    assert dumped, "served from the compile cache; nothing was measured"
    called: list[str] = []
    for f in dumped:
        called.extend(
            re.findall(r'mogg\.call\.materialize\["([^"]*)"\]', f.read_text())
        )
    return called


def test_hostwalk_probe_arms_cross_compile_for_windows_arm64(
    tmp_path: Path,
) -> None:
    _selfcheck()

    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", str(tmp_path)))
    record: dict[str, dict[str, object]] = {}
    for arm, walks in _WALKS.items():
        graph = _build(arm)
        case_dir = tmp_path / arm
        case_dir.mkdir(parents=True, exist_ok=True)
        session = InferenceSession(devices=[NPU()])
        session.debug.ir_output_dir = str(case_dir)
        compiled = session.compile(graph)

        called = _materializers(case_dir)
        assert len(called) == walks, (
            f"{arm}: {len(called)} materializing walks survived lowering,"
            f" designed {walks}: {called}"
        )
        assert all("npu_view_copy" in s for s in called), (
            f"{arm}: a walk resolved to a foreign materializer: {called}"
        )

        mef = out / f"hostwalk_{arm}.mef"
        compiled.export_mef(mef)
        blob = mef.read_bytes()
        assert b"hexagon_ondevice" in blob, arm
        assert b'{"label":"npu","id":0}' in blob, arm
        # The dispatch stub's entry points plus the Windows-only stack probe:
        # what separates "a MEF came out" from "a MEF that dispatches on the
        # NPU came out, built for the right ABI".
        for symbol in (
            b"AsyncRT_DeviceContext_loadFunction",
            b"AsyncRT_DeviceContext_enqueueFunctionDirect",
            b"__chkstk",
        ):
            assert symbol in blob, (arm, symbol)

        for f in case_dir.glob("*.mogg-pre-mgp.mlir"):
            shutil.copy(f, out / f"{arm}.{f.name}")
        record[arm] = {
            "mef_bytes": len(blob),
            "walks": called,
        }

    (out / "hostwalk_probe_record.json").write_text(
        json.dumps(record, indent=1)
    )
    print(json.dumps(record, indent=1))
