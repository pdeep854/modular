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
"""Compiles `mo.concat`, `mo.split` and `mo.slice` onto the on-device NPU as
host copies.

Device memory on this transport is host memory, so pure data movement needs no
device kernel and no selector: the registration copies through the operands'
pointers. That is the property `_require_identity_affine` already relies on to
read a weight, and what the hand-written driver already does for the joint
concat and the text drop.

`mo.slice` is the one that is not simply a copy. Its built-in is a **view**
kernel (`@extensibility.view_kernel`, alongside `mo.static.reshape`,
`mo.static.broadcast_to` and `mo.transpose`): it aliases its input's storage, so
a slice costs nothing while the alias survives, and only the aliases the
compiler cannot fold away have to be materialized. Materializing used to run
`view_copy_impl` -- generated device code, which this target cannot load -- so
those refused outright.

None of the view ops is registered here, and none needs to be. What is
registered is the **materializer**, scoped to this device
(`mogg.call.materialize.npu`), so every view op reaches a host walk when it must
and stays a free alias when it need not. The tests below pin both halves: the
whole-tensor slices that dominate the count still fold away for free, the ones
that must materialize now compile, and `mo.transpose` and
`mo.static.broadcast_to` came along without a line of their own.

`mo.cast` is counted with movement and is not: it converts every element between
two dtypes, so there is nothing to copy. It was an elementwise SIMD lambda and a
device-code gap until the `Cast` registration took it over; it now converts on
the host and the graph compiles.

Selection is argued differently here than for a dispatching stub, and one leg of
the usual ladder is simply unavailable: a registration that calls no
`load_function` puts **no** selector name and no package path into the exported
MEF, so "the artifact names our file" -- the positive half of every other proof
in this directory -- says nothing about these. What stands in its place is a
before/after that was measured in both directions: with these registrations
removed, an `mo.concat` on an NPU device does not compile at all, failing inside
MAX's own `nn/concat.mojo`, and a materializing `mo.slice` fails inside
`builtin_primitives/primitives.mojo`; with them, both compile. `mo.pad` below is
the durable half of that, since it is movement of the same kind with no
registration and still fails. For the materializer the proof is sharper than a
before/after: `test_the_npu_materializer_is_the_one_that_runs` reads the
resolved symbol out of the lowered IR, so a graph that compiled via the generic
path would not pass.
"""

from __future__ import annotations

import re
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

# Rank 3 with the join on the middle axis, so a copy would be a per-slab walk
# rather than the single contiguous block an axis-0 join degenerates to -- the
# case a registration would have had to get right.
_OUTER, _INNER = 2, 8
_PARTS = (3, 5)


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


def _npu_concat(name: str, dtype: DType) -> Graph:
    """Joins two tensors on the middle axis of three, on an NPU device."""
    dev = DeviceRef.NPU()
    input_types = [
        TensorType(dtype, [_OUTER, part, _INNER], dev) for part in _PARTS
    ]
    with Graph(
        name,
        input_types=input_types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        g.output(ops.concat([v.tensor for v in g.inputs], axis=1))
    return g


def test_npu_concat_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """An `mo.concat` on an NPU device compiles and serializes.

    Joined on the middle axis of three, so the copy is a per-slab walk rather
    than the single contiguous block an axis-0 join degenerates to.

    The MEF is checked for the op at its NPU-typed signature and *not* for this
    package's name: a registration that reaches no selector emits neither a
    selector string nor a source path, so its absence here is expected rather
    than evidence of anything.
    """
    graph = _npu_concat("npu_concat", DType.float16)
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_concat.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"mo.concat" in blob
    assert b'{"label":"npu","id":0}' in blob


def test_npu_concat_on_the_leading_axis_compiles(
    virtual_npu: None, tmp_path: Path
) -> None:
    """An axis-0 `mo.concat` of graph inputs compiles and serializes.

    Not a repeat of the middle-axis case: an axis-0 join of distinct
    non-constant *op results* is claimed by `ConcatInPlacePattern` and
    lowered to `mgp.buffer.concat` -- the in-place bet -- instead of this
    package's `Concat` dispatch. When the borrow cannot fire (a squeezed
    graph input's buffer is borrowed, not locally allocated, so
    `ConcatInPlacePass` bails), the op is emitted as the buffer-concat
    primitive, and the generic one is generated device code: its non-CPU arm
    compiles a kernel for the bare-metal triple, which no engine build
    carries. That is exactly the shape of the FLUX whole graph's packed-ids
    concat (`concat(txt_ids.squeeze, latent_image_ids.squeeze, 0)`), and it
    is served by this device's `mgp.buffer.concat.npu` host copy.

    The squeezes are load-bearing twice over: raw block arguments are
    refused by the pattern (the plain-inputs form goes to the `Concat`
    dispatch and never minted a buffer concat), and their borrowed buffers
    are what starve the borrow pass. si64 mirrors the id tensor; the buffer
    primitive is dtype-blind either way (it moves bytes).
    """
    dev = DeviceRef.NPU()
    input_types = [
        TensorType(DType.int64, [1, part, _INNER], dev) for part in _PARTS
    ]
    with Graph(
        "npu_concat_axis0",
        input_types=input_types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        squeezed = [ops.reshape(v.tensor, v.tensor.shape[1:]) for v in g.inputs]
        g.output(ops.concat(squeezed, axis=0))

    compiled = InferenceSession(devices=[NPU()]).compile(g)
    mef = tmp_path / "npu_concat_axis0.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b'{"label":"npu","id":0}' in blob


def test_npu_split_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """An `mo.split` on an NPU device compiles and serializes."""
    joined = sum(_PARTS)
    dev = DeviceRef.NPU()
    with Graph(
        "npu_split",
        input_types=[TensorType(DType.float16, [_OUTER, joined, _INNER], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(*ops.split(graph.inputs[0].tensor, list(_PARTS), axis=1))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_split.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"mo.split" in blob
    assert b'{"label":"npu","id":0}' in blob


def test_npu_pad_is_movement_that_still_does_not_compile(
    virtual_npu: None,
) -> None:
    """The durable half of the before/after: the class is not free.

    `mo.pad` is data movement of the same kind, has no registration, and fails
    to compile on the same device in the same process -- so the two above
    compiling is attributable to what was registered rather than to movement ops
    being served somewhere below kernel resolution.
    """
    dev = DeviceRef.NPU()
    with Graph(
        "npu_pad",
        input_types=[TensorType(DType.float32, [_OUTER, _INNER], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(ops.pad(graph.inputs[0].tensor, [0, 0, 1, 1], value=0.0))

    with pytest.raises(Exception):
        InferenceSession(devices=[NPU()]).compile(graph)


@pytest.mark.parametrize(
    "index",
    [
        pytest.param((slice(None), slice(2, 6), slice(None)), id="interior"),
        pytest.param((slice(None), slice(None), slice(0, 4)), id="last_axis"),
        pytest.param((slice(0, 1), slice(2, 6), slice(None)), id="two_axes"),
        pytest.param(
            (slice(None), slice(None), slice(0, _INNER, 2)), id="step_2"
        ),
    ],
)
def test_a_materializing_slice_compiles(virtual_npu: None, index) -> None:
    """The four shapes the compiler will not spare, all of which now compile.

    `isContiguousSliceKernel` (`FuseViews.cpp:82`) spares a slice only when it
    touches a single axis, that axis is the first whose static extent is not 1,
    and its step is 1. Each case here breaks one of those clauses, so each used
    to reach `view_copy_impl` -- generated device code this target cannot load.
    The last-axis case is the one that matters: it is how a fused QKV or gated
    MLP projection is chunked, and it is 122 of the 141 materializing slices in
    a FLUX denoise step.
    """
    dev = DeviceRef.NPU()
    with Graph(
        "npu_slice",
        input_types=[TensorType(DType.float16, [_OUTER, 8, _INNER], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(graph.inputs[0].tensor[index])

    InferenceSession(devices=[NPU()]).compile(graph)


def _slice_kernels_in_lowered_ir(index, dumps: Path, name: str) -> int:
    """Compiles one sliced graph and counts the slice kernels that survive.

    The count is read off `.mogg-pre-mgp.mlir`, after `mogg-fuse-views` has
    folded away every view it could. It carries a fixed baseline of mentions
    from the kernel-declaration section, so only differences between graphs
    compiled the same way are meaningful.
    """
    dev = DeviceRef.NPU()
    with Graph(
        name,
        input_types=[TensorType(DType.float16, [_OUTER, 8, _INNER], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(x if index is None else x[index])

    dumps.mkdir(parents=True, exist_ok=True)
    session = InferenceSession(devices=[NPU()])
    session.debug.ir_output_dir = str(dumps)
    session.compile(graph)
    dumped = list(dumps.glob("*.mogg-pre-mgp.mlir"))
    # A graph the compile cache already holds emits no IR, and its count would
    # read as zero rather than as "not measured".
    assert dumped, f"{name} was served from the compile cache, not measured"
    return sum(f.read_text().count('"mo.slice"') for f in dumped)


def test_a_whole_tensor_slice_needs_no_dispatch(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The registration must not turn the free majority into copies.

    162 of a denoise step's 355 slices select the whole tensor, and 100 of those
    slice a `[3072, 3072]` weight -- copying those would cost far more than the
    gap being closed. They fold before kernel resolution and never reach a
    dispatch.

    Asserted by counting the slice kernels left in the lowered IR, not by the
    compile succeeding: a copy would compile too, so success cannot tell the two
    apart. The last-axis case is the positive control -- the same measurement on
    a slice that does need the dispatch.
    """
    plain = _slice_kernels_in_lowered_ir(
        None, tmp_path / "plain", "npu_slice_none"
    )
    whole = _slice_kernels_in_lowered_ir(
        (slice(None), slice(None), slice(None)),
        tmp_path / "whole",
        "npu_slice_whole",
    )
    chunked = _slice_kernels_in_lowered_ir(
        (slice(None), slice(None), slice(0, 3)),
        tmp_path / "chunked",
        "npu_slice_chunked",
    )

    assert whole == plain, "a whole-tensor slice became a dispatch"
    assert chunked > plain, "a chunked slice did not become a dispatch"


@pytest.mark.parametrize("upstream", ["reshape", "transpose"])
def test_a_slice_whose_input_is_a_view(
    virtual_npu: None, upstream: str
) -> None:
    """A view feeding a slice compiles, whichever view it is.

    This pinned a refusal until the materializer became device-scoped. A slice
    of a transpose forces the transpose to materialize, which used to fall to
    `view_copy_impl` and refuse; both arms now compile, and neither needs a
    registration of its own.
    """
    dev = DeviceRef.NPU()
    with Graph(
        f"npu_slice_of_{upstream}",
        input_types=[TensorType(DType.float16, [_OUTER, 8, _INNER], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x = graph.inputs[0].tensor
        viewed = (
            ops.reshape(x, [_OUTER * 8, _INNER])
            if upstream == "reshape"
            else ops.transpose(x, 0, 2)
        )
        graph.output(viewed[..., 0:1])

    InferenceSession(devices=[NPU()]).compile(graph)


@pytest.mark.parametrize(
    "view",
    [
        pytest.param("transpose", id="transpose"),
        pytest.param("broadcast_to", id="broadcast_to"),
    ],
)
def test_a_materializing_view_compiles_without_its_own_registration(
    virtual_npu: None, view: str
) -> None:
    """The whole view family closes, not one op at a time.

    Neither `mo.transpose` nor `mo.static.broadcast_to` is registered for this
    device and neither needs to be: both are view kernels, so both reach the
    same `mogg.call.materialize`, and scoping that one primitive serves all of
    them. Both refused before it existed.

    `broadcast_to` is the arm that would survive a walk that assumed a dense
    input -- it materializes a stride-0 axis, so a copy reading it as a run
    would repeat the wrong element rather than fail.
    """
    dev = DeviceRef.NPU()
    shape = [_OUTER, _INNER] if view == "transpose" else [1, _INNER]
    with Graph(
        f"npu_{view}",
        input_types=[TensorType(DType.float16, shape, dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x = graph.inputs[0].tensor
        graph.output(
            ops.transpose(x, 0, 1)
            if view == "transpose"
            else ops.broadcast_to(x, [_OUTER, _INNER])
        )

    InferenceSession(devices=[NPU()]).compile(graph)


def _materializer_names(dumps: Path, device: DeviceRef, name: str) -> set[str]:
    """Compiles a graph whose view must materialize, and returns what it calls.

    Returns the symbols the `mogg.call.materialize` ops resolved to, read off
    `.mogg-pre-mgp.mlir`. Matching the call form rather than searching the file
    is what makes this a measurement: every primitive registered in the process
    is *declared* in every module, so both materializers are named in both
    dumps no matter which one the lowering picked.

    A transpose is used rather than a slice because nothing folds it away: it
    reaches `mogg.call.materialize` on any device, which is what makes the same
    graph a fair probe for both. Its shape is deliberately one no other test
    here uses, since a graph the compile cache already holds emits no IR, and
    its dtype is `float32` so that the two arms differ in device and nothing
    else -- `float16` is not supported on `cpu:0`.
    """
    with Graph(
        name,
        input_types=[
            TensorType(DType.float32, [_OUTER + 1, _INNER + 1], device)
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(ops.transpose(graph.inputs[0].tensor, 0, 1))

    dumps.mkdir(parents=True, exist_ok=True)
    session = InferenceSession(
        devices=[NPU() if device.device_type == "npu" else CPU()]
    )
    session.debug.ir_output_dir = str(dumps)
    session.compile(graph)
    dumped = list(dumps.glob("*.mogg-pre-mgp.mlir"))
    assert dumped, f"{name} was served from the compile cache, not measured"
    called = set()
    for f in dumped:
        called.update(
            re.findall(r'mogg\.call\.materialize\["([^"]*)"\]', f.read_text())
        )
    assert called, f"{name} lowered no materialize call to read"
    return called


def test_the_npu_materializer_is_the_one_that_runs(
    virtual_npu: None, tmp_path: Path
) -> None:
    """Which materializer the lowering picked, read off the IR.

    The views above compiling is not evidence that this device's materializer
    served them -- the generic `view_copy_impl` suddenly working would look
    identical. `mogg.call.materialize` carries the resolved primitive's symbol,
    so the lowered IR names one or the other outright.

    The CPU arm is the half that matters most. `crossVerifyKernels` already
    showed that this machinery is device-blind by default, so a lookup that
    captured CPU graphs would be a worse bug than the one being fixed, and it
    would not show up in any NPU test.
    """
    npu = _materializer_names(
        tmp_path / "npu", DeviceRef.NPU(), "npu_materializer"
    )
    cpu = _materializer_names(
        tmp_path / "cpu", DeviceRef.CPU(), "cpu_materializer"
    )

    assert all("npu_view_copy" in s for s in npu), (
        f"an NPU graph did not resolve this device's materializer: {npu}"
    )
    assert all("view_copy_impl" in s for s in cpu), (
        f"a CPU graph did not resolve the generic materializer: {cpu}"
    )


def test_movement_is_dtype_blind_where_a_dispatch_is_not(
    virtual_npu: None,
) -> None:
    """Nothing here is gated on a dtype, unlike every dispatching stub.

    The dispatching registrations reach a selector that had to be compiled for
    the dtype, which is why the norms needed widening and the batched matmul
    could not be. These ops move bytes without interpreting them, so bf16 and
    even a sub-byte 4-bit encoding compile with no twin of anything -- which is
    the clearest statement that no device code is involved.
    """
    session = InferenceSession(devices=[NPU()])
    for dtype in (DType.bfloat16, DType.float32, DType.float4_e2m1fn):
        session.compile(_npu_concat(f"npu_concat_{dtype}", dtype))


def test_a_cast_is_not_movement_and_is_served(virtual_npu: None) -> None:
    """The one op in this class that is genuine device compute.

    `mo.cast` looks like movement and is counted with it, but it converts every
    element between two dtypes, so there is nothing to copy: fp16 and bf16 are
    the same width and different numbers. It used to land in the QAIC
    elementwise hook and fail; the `Cast` registration in
    `hexagon_ondevice/registrations.mojo` now converts on the host, so the
    graph compiles. `test_npu_graph_cast.py` carries the full evidence shape --
    including that dispatching nothing means the ABSENCE of `hexagon_ondevice`
    in the artifact, and the view-input case that still refuses.
    """
    dev = DeviceRef.NPU()
    with Graph(
        "npu_cast",
        input_types=[TensorType(DType.bfloat16, [_OUTER, _INNER], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(graph.inputs[0].tensor.cast(DType.float16))

    InferenceSession(devices=[NPU()]).compile(graph)
