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
"""Records where `mo.cast` actually lowers on the on-device NPU.

`mo.cast` is real elementwise work -- fp16 and bf16 are the same width and
different numbers -- but the signed kernel set carries no cast entry point, so
there is nothing to dispatch to. `Cast` therefore converts on the host, which
device memory being host memory allows, and emits no device kernel at all.

That makes the artifact evidence read backwards from every other registration
here, and the reason is worth pinning: because nothing is dispatched, no
selector name and no kernel-package path reaches the MEF. The absence of
`hexagon_ondevice` is what a correct lowering looks like, not what an escaped
one looks like. Removing the registration does not produce a MEF without that
string -- it produces no MEF, because the compiler falls back to an elementwise
SIMD lambda that is device code this target cannot compile. Measured both ways.

One case still fails, and it is recorded here rather than left to be
rediscovered: a cast whose input is a **view** does not reach this stub at all.
The compiler folds the view into the cast and lowers the pair as an elementwise
lambda, which is the device code this target cannot compile. So the takeover
covers a cast over a materialized operand and not one over an alias.

These tests do not claim every cast in a real model is covered. The census
counts 55 without recording which have materialized operands and which have
views, so the share that still falls in the gap above is unmeasured.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from max.driver import (
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

_ROWS, _LENGTH = 4, 64


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")


@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (DType.bfloat16, DType.float16),
        (DType.float16, DType.bfloat16),
        (DType.bfloat16, DType.float32),
        (DType.float32, DType.float16),
    ],
)
def test_standalone_cast_compiles_without_a_device_kernel(
    virtual_npu: None, tmp_path: Path, src: DType, dst: DType
) -> None:
    """A cast on its own needs nothing from this package.

    Both directions between the two 16-bit types, and both directions against
    fp32. Each compiles, and the artifact carries no reference to the on-device
    kernel package -- the conversion is in an `mgp` sequence the runtime
    performs on the host, not a dispatch.
    """
    name = f"npu_cast_{src.name}_{dst.name}"
    types = [TensorType(src, [_ROWS, _LENGTH], DeviceRef.NPU())]
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(ops.cast(graph.inputs[0].tensor, dst))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"{name}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"mo.cast" in blob
    assert b'{"label":"npu","id":0}' in blob
    # No selector name and no kernel-package path: this registration dispatches
    # nothing, so the conversion is inlined into the host sequence rather than
    # emitted as a device kernel. Asserted so a future reader does not read the
    # absence as the op having escaped this package -- the refusal test below is
    # what ties the lowering to this file.
    assert b"hexagon_ondevice" not in blob


def test_cast_feeding_a_registered_op_compiles(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A cast whose result a dispatching op consumes.

    The adjacency that decides whether the casts inside a real model are free
    the way a standalone one is. Here the cast feeds a layer norm, which is
    registered and dispatches, so the cast cannot be folded into the consumer's
    prologue -- a pre-signed kernel takes no prologue any more than it takes an
    epilogue.

    What this does NOT settle: a cast whose consumer is itself an elementwise
    lambda, where the two could fuse into one lambda that this target cannot
    compile. That case needs the real graph to answer, and the op census counts
    55 casts without recording which kind they are.
    """
    dev = DeviceRef.NPU()
    types = [TensorType(DType.float32, [_ROWS, _LENGTH], dev)]
    with Graph(
        "npu_cast_then_layernorm",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x = ops.cast(graph.inputs[0].tensor, DType.bfloat16)
        gamma = ops.constant([1.0] * _LENGTH, DType.bfloat16, device=dev)
        beta = ops.constant([0.0] * _LENGTH, DType.bfloat16, device=dev)
        graph.output(ops.layer_norm(x, gamma, beta, epsilon=1.0))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_cast_then_layernorm.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    # The consumer dispatches; the cast rides the host path in front of it.
    assert b"layernorm_bf16\x00" in blob
    assert b"hexagon_ondevice" in blob


def test_cast_of_a_spread_operand_compiles(virtual_npu: None) -> None:
    """The gap this stub could not close alone, and what closed it.

    A cast over a materialized operand reaches this package's stub; a cast
    over a VIEW did not -- the compiler folded the broadcast into the cast and
    lowered the pair as an elementwise lambda, which was pinned here as a
    still-failing case. The device-scoped materializer
    (`mogg.call.materialize.npu`, the same host walk that serves `mo.slice`)
    closed it from the other side: the view materializes to a dense buffer
    first, which is exactly the "view materialized before the cast" this
    docstring used to name as the missing piece, and the stub takes the dense
    operand from there. Asserting the compile keeps the pin discriminating --
    this exact graph was refused before the materializer resolved per device.
    """
    dev = DeviceRef.NPU()
    with Graph(
        "npu_cast_spread",
        input_types=[TensorType(DType.float32, [1, _LENGTH], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        spread = ops.broadcast_to(graph.inputs[0].tensor, [_ROWS, _LENGTH])
        graph.output(ops.cast(spread, DType.bfloat16))

    InferenceSession(devices=[NPU()]).compile(graph)
