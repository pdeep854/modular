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
"""Lowers `x + gate * y` onto the on-device NPU's signed `gate_residual` rows.

The signed kernel set has carried an entry point for this expression, in fp16
and bf16, since the modulation kernels landed; no graph could reach it, because
a model writes the expression as `mo.mul` and `mo.add` and there is no `mo.*`
operator that means the pair. `GatedResidualPattern` mints
`mo.composite.gated_residual` for it, and the registration here claims that name.

Two things are worth proving, and each has its own test below:

- **The pass ordering.** The pair is not merely unregistered on this target, it
  is actively consumed: the graph compiler folds a mul beside a matmul into that
  matmul's epilogue, and a kernel signed before compilation cannot take one, so
  `flux2.py:406` compiled to a refusal rather than to a missing kernel. The
  pattern is only useful if it claims the operands before the epilogue fusion
  reaches them. `test_flux_gate_after_matmul_compiles` is that measurement: it
  is the graph that was refused, and it now compiles.
- **Which selector the stub named**, on the same two discriminators the layer
  norm's test uses: the exported MEF has to carry the selector name, and a dtype
  the signed set has no row for has to keep its old path rather than reaching a
  wrong row.
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
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# Four rows so each of the kernel's four compute threads gates one, and a row
# length of one HVX vector's worth of 16-bit lanes.
_ROWS, _LENGTH = 4, 64

# The matmul the ordering probe puts in front of the gate, at the shape the
# earlier epilogue probe used: `[M, K] @ [N, K]^T -> [M, N]`.
_M, _N, _K = 4, 8, 64


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


def _model_sequences(blob: bytes) -> list[str]:
    """The exported model's own op plans, `Sequence(...)` per execute step.

    A raw substring check over the whole MEF is not sound for ABSENCE: the
    artifact embeds every kernel the exporting process compiled, so a
    composite another test in this process minted shows up here too
    (measured). The `Sequence` plans are this model's alone, which is what
    makes an op's absence from them assertable.
    """
    return [
        m.decode("utf-8", "replace")
        for m in re.findall(rb"Sequence\([^)]{0,8000}\)", blob)
    ]


def _gated_residual_graph(
    name: str, *, device: DeviceRef, dtype: DType
) -> Graph:
    """Builds `x + gate * y` with a per-channel gate broadcast over the rows.

    The gate is a graph input rather than a constant so that constant folding
    cannot turn the broadcast into something else on its way to the pattern --
    and because in the reference it is a slice of a projection of the timestep
    embedding, which is a value, not a literal.
    """
    types = [
        TensorType(dtype, [_ROWS, _LENGTH], device),
        TensorType(dtype, [_ROWS, _LENGTH], device),
        TensorType(dtype, [1, _LENGTH], device),
    ]
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        x, y, gate = (v.tensor for v in g.inputs)
        g.output(x + gate * y)
    return g


def test_flux_gate_after_matmul_compiles(virtual_npu: None) -> None:
    """`hidden + gate * to_out(attn)` -- `flux2.py:406` -- compiles.

    The measurement the whole approach rested on. This exact graph was refused
    before the pattern existed, and not for want of a kernel: the mul and the
    add were folded into the matmul's epilogue, and the signed matmul refused
    one (`registrations.mojo`: "an epilogue cannot be fused into a pre-signed
    kernel"). Compiling means the pattern claimed the operands first -- which it
    can, `mo-pattern-fusion` running a whole pipeline stage before the MOGG
    epilogue fusion that would otherwise take them.

    Recorded verbatim in `.agentwork/flux-npu/epilogue-check/` as the refusal
    this replaces.
    """
    types = [
        TensorType(DType.float16, [_M, _K], DeviceRef.NPU()),
        TensorType(DType.float16, [_N, _K], DeviceRef.NPU()),
        TensorType(DType.float16, [1, _N], DeviceRef.NPU()),
        TensorType(DType.float16, [_M, _N], DeviceRef.NPU()),
    ]
    with Graph(
        "npu_flux_gate_residual",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        attn, w_to_out, gate, hidden = (v.tensor for v in graph.inputs)
        to_out = attn @ ops.transpose(w_to_out, -1, -2)
        graph.output(hidden + gate * to_out)

    InferenceSession(devices=[NPU()]).compile(graph)


def test_npu_gated_residual_bf16_compiles(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 gated residual on an NPU device compiles and serializes."""
    graph = _gated_residual_graph(
        "npu_gate_residual_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_gate_residual_bf16.mef"
    compiled.export_mef(mef)
    assert b"hexagon_ondevice" in mef.read_bytes()


def _selector_names(blob: bytes) -> set[bytes]:
    """The selector names a MEF carries, NUL-terminated as they are emitted.

    Terminated because `gate_residual` is a prefix of both `gate_residual_bf16`
    and `gate_residual_clamp`, so an unterminated match would count either as an
    fp16 dispatch.

    Only presence is meaningful. A MEF carries every kernel compiled in the
    process that exported it, not only the ones its own graph reaches, so a
    second dtype compiled earlier in the same pytest process appears in this set
    too. That makes the absence of a name unassertable and its presence exact.
    """
    return {
        name
        for name in (
            b"gate_residual\x00",
            b"gate_residual_bf16\x00",
            b"gate_residual_clamp\x00",
        )
        if name in blob
    }


def test_npu_gated_residual_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 graph instantiates the stub's bf16 branch.

    The name can only be in the artifact if the compiler instantiated the branch
    that spells it, which is what `comptime if` buys: a conditional expression
    would put both names in every MEF and leave this unassertable. Paired with
    the composite's own signature, so the name is tied to a bf16 op rather than
    merely present.
    """
    graph = _gated_residual_graph(
        "npu_gate_residual_sel_bf16",
        device=DeviceRef.NPU(),
        dtype=DType.bfloat16,
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_sel_bf16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"gate_residual_bf16\x00" in _selector_names(blob)
    assert (
        b"mo.composite.gated_residual : (!mo.tensor<[4, 64], bf16, npu:0>"
        in blob
    )


def test_npu_gated_residual_fp16_names_the_fp16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The fp16 path reaches the fp16 row, not the bf16 one.

    The complement of the test above, and the reason both are positive claims:
    the fp16 name is emitted only by the branch this dtype must take, so seeing
    it rules out a `comptime if` that fell the wrong way -- which asserting the
    bf16 name's absence cannot do here, the MEF carrying whatever else the
    process compiled.
    """
    graph = _gated_residual_graph(
        "npu_gate_residual_sel_f16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_sel_f16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"gate_residual\x00" in _selector_names(blob)
    assert (
        b"mo.composite.gated_residual : (!mo.tensor<[4, 64], f16, npu:0>"
        in blob
    )


def test_gated_residual_takes_a_symbolic_batch(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The shape class the denoise graph actually stages.

    Every gate in the whole-graph staging is `[batch, 1, D]` with the batch
    symbolic -- deliberately, because a literal-1 batch is squeezed into a
    matmul form the NPU stub refuses (`test_npu_graph_batch_matmul.py`). A gate
    whose outermost extent is the result's own is one row per batch, which the
    registration serves.
    """
    types = [
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.float16, ["batch", 1, _LENGTH], DeviceRef.NPU()),
    ]
    with Graph(
        "npu_gate_residual_symbolic",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, y, gate = (v.tensor for v in graph.inputs)
        graph.output(x + gate * y)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_symbolic.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"gate_residual\x00" in _selector_names(blob)
    assert (
        b"mo.composite.gated_residual : (!mo.tensor<[dyn, 4, 64], f16, npu:0>"
        in blob
    )


def test_clamped_residual_compiles_to_the_clamp_composite(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The fp16-saturated close -- `saturate_to_float16` at `flux2.py:69`.

    25 of the reference's 40 sub-block closes clamp the stream right after the
    residual, and #166's clamp-absence guard kept them refusing whole rather
    than half-claimed. `GatedResidualClampPattern` is the rule that guard
    reserved the anchor for: it consumes the residual and its saturation pair
    as one composite, at the symbolic-batch shape the denoise graph stages.
    The MEF naming the clamp selector ties the compile to the fp16-only
    `gate_residual_clamp` row rather than the plain one.
    """
    types = [
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.float16, ["batch", 1, _LENGTH], DeviceRef.NPU()),
    ]
    with Graph(
        "npu_gate_residual_clamped",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, y, gate = (v.tensor for v in graph.inputs)
        h = x + gate * y
        graph.output(ops.min(ops.max(h, -65504.0), 65504.0))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_clamped.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"gate_residual_clamp\x00" in _selector_names(blob)
    assert (
        b"mo.composite.gated_residual_clamp : (!mo.tensor<[dyn, 4, 64], f16,"
        b" npu:0>" in blob
    )


def test_clamp_with_other_bounds_avoids_the_kernels(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A clamp that is not float16 saturation is a different computation.

    The signed kernel hardcodes +-65504, so the rule holds the graph's
    constants to exactly those. Any other bound used to keep the refusal it
    had; since `registrations.mojo` claimed the bare elementwise tail as host
    walks the chain compiles -- served, not minted -- and the pin moves to
    the model's own op plans: they carry the raw ops and neither gated
    composite.
    """
    types = [
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.float16, ["batch", 1, _LENGTH], DeviceRef.NPU()),
    ]
    with Graph(
        "npu_gate_residual_wrong_bounds",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, y, gate = (v.tensor for v in graph.inputs)
        h = x + gate * y
        graph.output(ops.min(ops.max(h, -1000.0), 1000.0))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_wrong_bounds.mef"
    compiled.export_mef(mef)
    seqs = _model_sequences(mef.read_bytes())
    assert any("mo.add" in s for s in seqs), seqs
    assert not any("mo.composite.gated_residual" in s for s in seqs), seqs


def test_bf16_clamped_residual_avoids_the_kernels(
    virtual_npu: None, tmp_path: Path
) -> None:
    """bf16 has no clamp row, and no bf16 graph should reach one.

    bf16 carries fp32's exponent range, so the reference never stages the
    saturation at bf16 (65504 is not even bf16-representable; this graph
    spells the nearest bound, 65536). The clamp rule is fp16-only, and the
    plain rule's clamp-absence guard is dtype-blind, so nothing is
    half-claimed; the chain compiles on the host walks, and the pin is that
    the model's own op plans reach neither gated composite.
    """
    types = [
        TensorType(DType.bfloat16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.bfloat16, ["batch", _ROWS, _LENGTH], DeviceRef.NPU()),
        TensorType(DType.bfloat16, ["batch", 1, _LENGTH], DeviceRef.NPU()),
    ]
    with Graph(
        "npu_gate_residual_clamped_bf16",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, y, gate = (v.tensor for v in graph.inputs)
        h = x + gate * y
        graph.output(ops.min(ops.max(h, -65536.0), 65536.0))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_clamped_bf16.mef"
    compiled.export_mef(mef)
    seqs = _model_sequences(mef.read_bytes())
    assert any("mo.add" in s for s in seqs), seqs
    assert not any("mo.composite.gated_residual" in s for s in seqs), seqs


def test_npu_gated_residual_leaves_fp32_alone(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A dtype the signed set has no row for keeps off the signed kernel.

    The pattern's dtype guard is what confines it; without one an fp32 graph
    would reach a kernel compiled for neither of its rows. fp32 used to have
    no NPU path at all; it now compiles on the elementwise host walks, and
    what this asserts is that the composite is not what served it.
    """
    graph = _gated_residual_graph(
        "npu_gate_residual_fp32", device=DeviceRef.NPU(), dtype=DType.float32
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_fp32.mef"
    compiled.export_mef(mef)
    seqs = _model_sequences(mef.read_bytes())
    assert any("mo.add" in s for s in seqs), seqs
    assert not any("mo.composite.gated_residual" in s for s in seqs), seqs


def test_cpu_gated_residual_is_untouched(virtual_npu: None) -> None:
    """The pattern is inert off this target.

    Same expression, same kernel package, a CPU device: it compiles, because the
    pattern gates itself on the device API and a CPU never matches. This is the
    blast-radius check -- the rule lives in the shared graph compiler, so every
    other target has to see the graph it saw before.
    """
    graph = _gated_residual_graph(
        "cpu_gate_residual_fp32", device=DeviceRef.CPU(), dtype=DType.float32
    )
    InferenceSession(devices=[CPU()]).compile(graph)
