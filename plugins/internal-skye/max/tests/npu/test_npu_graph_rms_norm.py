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
"""Compiles a bf16 `mo.reduce.rms_norm` onto the on-device NPU.

The sibling of `test_npu_graph_layernorm.py`, and there for the same reason: the
device library's kernel set is fixed when it is signed, so a dtype reaches a
selector compiled for it or none at all. `NPUKernels.def` carries `rms_norm_bf16`
beside `rms_norm` at the same wire shape, which is what lets a bf16 graph reach
this part at all -- FLUX.2-Klein being bf16 throughout.

Which selector the stub named is the thing worth proving, since a built-in rms
norm would also compile a bf16 graph and would say nothing about the NPU. Two
discriminators, in opposite directions: the exported MEF has to carry the bf16
selector's name, and fp32 -- which the built-in accepts -- has to be refused by
this stub's own message.

Unlike the layer norm, this selector applies the affine itself, so `weight` is a
real per-lane tensor here rather than an identity the stub has to refuse.
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

# Four rows so each of the kernel's four compute threads normalizes one, and a
# row length of one HVX vector's worth of bf16 lanes.
_ROWS, _LENGTH = 4, 64

# The epsilon that makes the result exact. See `npu_rms_norm_bf16_exec.c` for the
# construction: with the row pattern used there the mean square is exactly 85, so
# this eps puts it on 256 and the reciprocal rms on 1/16 -- a power of two, which
# keeps every output exactly representable in bf16 rather than something a
# tolerance has to cover.
_EPSILON = 171.0

# Powers of two, so scaling by the weight stays exact for the same reason.
_WEIGHT = [1.0, 2.0, 4.0, 8.0] * (_LENGTH // 4)


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


def _rms_norm_graph(name: str, *, device: DeviceRef, dtype: DType) -> Graph:
    """Builds one `mo.reduce.rms_norm` over the last axis."""
    with Graph(
        name,
        input_types=[TensorType(dtype, [_ROWS, _LENGTH], device)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        x = g.inputs[0].tensor
        weight = ops.constant(_WEIGHT, dtype, device=device)
        g.output(ops.rms_norm(x, weight, epsilon=_EPSILON))
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


def test_npu_rms_norm_bf16_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """A bf16 `mo.reduce.rms_norm` on an NPU device compiles and serializes."""
    graph = _rms_norm_graph(
        "npu_rms_norm_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_rms_norm_bf16.mef"
    compiled.export_mef(mef)
    assert b"hexagon_ondevice" in mef.read_bytes()


def _selector_names(blob: bytes) -> set[bytes]:
    """The selector names a MEF carries, NUL-terminated as they are emitted.

    `loadFunction` resolves a selector by the name string the stub hands it, and
    that string reaches the artifact beside the `registrations.mojo` path and the
    `mo.reduce.rms_norm` signature it was instantiated for. Terminated, because
    `rms_norm` is a prefix of both `rms_norm_bf16` and the `rms_norm_staged` row
    -- an unterminated match would count either as a dispatch of the fp16 one.

    Only presence is meaningful. A MEF carries every kernel compiled in the
    process that exported it, not only the ones its own graph reaches, so a
    second dtype compiled earlier in the same pytest process appears in this set
    too. That makes the absence of a name unassertable and its presence exact.
    """
    return {
        name for name in (b"rms_norm\x00", b"rms_norm_bf16\x00") if name in blob
    }


def test_npu_rms_norm_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 graph instantiates the stub's bf16 branch.

    The name can only be in the artifact if the compiler instantiated the branch
    that spells it: a built-in rms norm names no selector at all, and the fp16
    branch names a different one. Paired with the signature it was compiled
    against, so the name is tied to a bf16 op rather than merely present.
    """
    graph = _rms_norm_graph(
        "npu_rms_norm_sel_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rms_norm_sel_bf16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"rms_norm_bf16\x00" in _selector_names(blob)
    assert b"mo.reduce.rms_norm : (!mo.tensor<[4, 64], bf16, npu:0>" in blob


def test_npu_rms_norm_fp16_still_names_the_fp16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """Widening to bf16 did not repoint the fp16 path at the bf16 selector.

    The complement of the test above, and the reason both are positive claims:
    the fp16 name is emitted only by the branch this dtype must take, so seeing
    it rules out a `comptime if` that fell the wrong way -- which asserting the
    bf16 name's absence cannot do here, the MEF carrying whatever else the
    process compiled.
    """
    graph = _rms_norm_graph(
        "npu_rms_norm_sel_f16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rms_norm_sel_f16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"rms_norm\x00" in _selector_names(blob)
    assert b"mo.reduce.rms_norm : (!mo.tensor<[4, 64], f16, npu:0>" in blob


def test_npu_rms_norm_refuses_fp32(virtual_npu: None) -> None:
    """Widening to bf16 did not widen to everything.

    fp32 is what the built-in accepts, so this failing with this stub's message
    is the negative half of the selection proof.
    """
    graph = _rms_norm_graph(
        "npu_rms_norm_fp32", device=DeviceRef.NPU(), dtype=DType.float32
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "fp16 or bf16 only" in diagnostic


def test_cpu_rms_norm_accepts_what_the_npu_refuses(virtual_npu: None) -> None:
    """The refusal above is the NPU's, not the graph's.

    Same shapes, same kernel package, a CPU device: the fp32 graph the NPU stub
    rejects compiles here, so the rejection is kernel selection rather than
    something the graph API would have refused anyway.
    """
    graph = _rms_norm_graph(
        "cpu_rms_norm_fp32", device=DeviceRef.CPU(), dtype=DType.float32
    )
    InferenceSession(devices=[CPU()]).compile(graph)
