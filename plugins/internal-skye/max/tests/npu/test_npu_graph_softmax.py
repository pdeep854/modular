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
"""Compiles an fp16 `mo.reduce.softmax` onto the on-device NPU.

The registration has been in the tree since the `softmax` selector was signed,
but no graph had ever been built against it, so nothing established that a
softmax reaches it rather than the built-in. That is what these compile against
a virtual NPU.

Which selector the stub named is the thing worth proving, since a built-in
softmax would also compile an fp16 graph and would say nothing about the NPU.
The stub's two `comptime assert`s are the discriminators, and the built-in
accepts what each of them refuses: bf16, and a reduction over anything but the
last axis.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from max.driver import (
    CPU,
    NPU,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# Four rows so each of the kernel's four compute threads reduces one, and a row
# length of one HVX vector's worth of fp16 lanes.
_ROWS, _LENGTH = 4, 64


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none.

    The settings are process-wide, so they are cleared on the way out; leaving
    them set would retarget every later test in this process.
    """
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")


def _softmax_graph(
    name: str, *, device: DeviceRef, dtype: DType, axis: int = -1
) -> Graph:
    """Builds one `mo.reduce.softmax`, by default over the last axis."""
    with Graph(
        name,
        input_types=[TensorType(dtype, [_ROWS, _LENGTH], device)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        g.output(ops.softmax(g.inputs[0].tensor, axis=axis))
    return g


def _compile_error(exc: BaseException) -> str:
    """Joins a compile failure with its causes.

    `InferenceSession.compile` reports a generic "Failed to compile the model"
    and chains the diagnostic underneath, so the message worth asserting on is
    never the top one.
    """
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def test_npu_softmax_fp16_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """An fp16 `mo.reduce.softmax` on an NPU device compiles and serializes."""
    graph = _softmax_graph(
        "npu_softmax_fp16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_softmax_fp16.mef"
    compiled.export_mef(mef)
    assert b"hexagon_ondevice" in mef.read_bytes()


def test_npu_softmax_fp16_names_the_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """An fp16 graph instantiates the stub, not a built-in softmax.

    The name can only be in the artifact if the compiler instantiated the branch
    that spells it. Paired with the signature it was compiled against, so the
    name is tied to an fp16 op rather than merely present.
    """
    graph = _softmax_graph(
        "npu_softmax_sel_f16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_softmax_sel_f16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"softmax\x00" in blob
    assert b"mo.reduce.softmax : (!mo.tensor<[4, 64], f16, npu:0>" in blob


def test_npu_softmax_refuses_bf16(virtual_npu: None) -> None:
    """bf16 is refused, which is why FLUX.2 has to stage fp16 to reach this."""
    graph = _softmax_graph(
        "npu_softmax_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "fp16 only" in diagnostic


def test_npu_softmax_refuses_a_non_last_axis(virtual_npu: None) -> None:
    """The dtype gate is not the only one, so fp16 alone does not imply a match.

    The selector reduces contiguous rows; an axis-0 reduction over the same
    tensor is a different traversal, and refusing it is the second half of the
    selection proof.
    """
    graph = _softmax_graph(
        "npu_softmax_axis0",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        axis=0,
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "last-axis only" in diagnostic


def test_cpu_softmax_accepts_what_the_npu_refuses(virtual_npu: None) -> None:
    """The refusals above are the NPU's, not the graph's.

    Same shapes, same kernel package, a CPU device: the bf16 graph the NPU stub
    rejects compiles here, so the rejection is kernel selection rather than
    something the graph API would have refused anyway.
    """
    graph = _softmax_graph(
        "cpu_softmax_bf16", device=DeviceRef.CPU(), dtype=DType.bfloat16
    )
    InferenceSession(devices=[CPU()]).compile(graph)
