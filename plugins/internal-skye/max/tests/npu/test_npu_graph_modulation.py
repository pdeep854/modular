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
"""Lowers adaLN modulation, SwiGLU and bare SiLU onto their signed selectors.

The three that follow `mo.composite.gated_residual` onto the on-device NPU's
signed kernel set. Two need a PatternFusion rule because the expression has no
`mo.*` operator; the third needs only a registration, `mo.silu` being an
operator the compiler otherwise lowers to an elementwise lambda that is device
code and does not compile here.

The load-bearing test is `test_modulation_without_the_plus_one_is_refused`. The
signed `ada_modulate` kernel computes `(1 + scale) * x + shift` and the
composite carries the raw scale, so a graph that wrote a plain
`scale * x + shift` must not reach it -- it would be modulated twice, which is a
wrong answer rather than a missed fusion.
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

# Four rows so each of the kernels' four compute threads takes one, and a row
# length of one HVX vector's worth of 16-bit lanes.
_ROWS, _LENGTH = 4, 64


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none."""
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


def _modulate_graph(
    name: str,
    *,
    device: DeviceRef,
    dtype: DType,
    plus_one: bool = True,
    batch: int | str = 1,
) -> Graph:
    """Builds `(1 + scale) * x + shift`, or without the one when asked.

    `scale` and `shift` are graph inputs at the parameter shape
    `Flux2Modulation` produces, so constant folding cannot reshape the
    expression on its way to the pattern. `batch` may be a symbolic name, which
    is how the denoise graph stages it.
    """
    types = [
        TensorType(dtype, [batch, _ROWS, _LENGTH], device),
        TensorType(dtype, [batch, 1, _LENGTH], device),
        TensorType(dtype, [batch, 1, _LENGTH], device),
    ]
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        x, scale, shift = (v.tensor for v in g.inputs)
        g.output(((1 + scale) if plus_one else scale) * x + shift)
    return g


def _swiglu_graph(name: str, *, device: DeviceRef, dtype: DType) -> Graph:
    """Builds `silu(gate) * up` from the two halves of one projection."""
    types = [TensorType(dtype, [_ROWS, 2 * _LENGTH], device)]
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        proj = g.inputs[0].tensor
        gate, up = ops.chunk(proj, 2, axis=-1)
        g.output(ops.silu(gate) * up)
    return g


def _silu_graph(name: str, *, device: DeviceRef, dtype: DType) -> Graph:
    """Builds a bare `silu`, the activation with no sibling to gate."""
    types = [TensorType(dtype, [_ROWS, _LENGTH], device)]
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        g.output(ops.silu(g.inputs[0].tensor))
    return g


def _export(graph: Graph, tmp_path: Path, name: str) -> bytes:
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"{name}.mef"
    compiled.export_mef(mef)
    return mef.read_bytes()


def _names(blob: bytes, *candidates: str) -> set[str]:
    """The selector names a MEF carries, NUL-terminated as they are emitted.

    Terminated because each fp16 name is a prefix of its bf16 twin, so an
    unterminated match would count a bf16 dispatch as an fp16 one.

    Only presence is meaningful. A MEF carries every kernel compiled in the
    process that exported it, not only the ones its own graph reaches, so the
    absence of a name is unassertable and its presence exact.
    """
    return {n for n in candidates if f"{n}\0".encode() in blob}


def test_ada_modulate_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 modulation reaches the bf16 row, through the composite."""
    blob = _export(
        _modulate_graph(
            "npu_ada_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
        ),
        tmp_path,
        "npu_ada_bf16",
    )
    assert "ada_modulate_bf16" in _names(
        blob, "ada_modulate", "ada_modulate_bf16"
    )
    assert (
        b"mo.composite.ada_modulate : (!mo.tensor<[1, 4, 64], bf16, npu:0>"
        in blob
    )


def test_ada_modulate_fp16_names_the_fp16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The complement, so a `comptime if` that fell the wrong way shows up."""
    blob = _export(
        _modulate_graph(
            "npu_ada_f16", device=DeviceRef.NPU(), dtype=DType.float16
        ),
        tmp_path,
        "npu_ada_f16",
    )
    assert "ada_modulate" in _names(blob, "ada_modulate", "ada_modulate_bf16")
    assert (
        b"mo.composite.ada_modulate : (!mo.tensor<[1, 4, 64], f16, npu:0>"
        in blob
    )


def test_modulation_without_the_plus_one_avoids_the_kernel(
    virtual_npu: None, tmp_path: Path
) -> None:
    """`scale * x + shift` must not reach a kernel that adds a one.

    The correctness guard the rule exists around: the signed selector computes
    `(1 + scale) * x + shift`, so a graph that never asked for the one would be
    modulated twice. Nothing about the shapes distinguishes the two -- only the
    `mo.add` against a constant one does, which is why the rule requires it.

    This expression used to have no kernel at all without the composite, and
    the pin was that the compile FAILED without naming `ada_modulate`. Since
    `registrations.mojo` claimed the bare `mo.mul`/`mo.add` as host walks the
    pair compiles -- served, not minted -- so the pin is now on the model's
    own op plans: they carry the raw pair and no composite.
    """
    blob = _export(
        _modulate_graph(
            "npu_ada_no_one",
            device=DeviceRef.NPU(),
            dtype=DType.bfloat16,
            plus_one=False,
        ),
        tmp_path,
        "npu_ada_no_one",
    )
    seqs = _model_sequences(blob)
    assert any("mo.mul" in s for s in seqs), seqs
    assert any("mo.add" in s for s in seqs), seqs
    assert not any("mo.composite" in s for s in seqs), seqs


def test_ada_modulate_takes_a_symbolic_batch(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The shape class the denoise graph actually stages.

    Every modulation parameter in the whole-graph staging is `[batch, 1, D]`
    with the batch symbolic -- deliberately, because a literal-1 batch is
    squeezed into a matmul form the NPU stub refuses
    (`test_npu_graph_batch_matmul.py`). A parameter whose outermost extent is
    the result's own is one row per batch, which the registration serves, so
    the rule accepts it; a symbolic extent that is NOT the result's own stays
    refused.
    """
    blob = _export(
        _modulate_graph(
            "npu_ada_symbolic",
            device=DeviceRef.NPU(),
            dtype=DType.float16,
            batch="batch",
        ),
        tmp_path,
        "npu_ada_symbolic",
    )
    assert "ada_modulate" in _names(blob, "ada_modulate", "ada_modulate_bf16")
    assert (
        b"mo.composite.ada_modulate : (!mo.tensor<[dyn, 4, 64], f16, npu:0>"
        in blob
    )


def test_modulation_without_the_plus_one_avoids_the_kernel_at_symbolic_batch(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The `1 +` guard survives the symbolic-batch relaxation.

    Same correctness stake as the static case above it, held at the shapes the
    relaxation newly admits: `scale * x + shift` at a symbolic batch must
    reach neither the modulation kernel (it would add a one the graph never
    asked for) nor the gated-residual one (it would read the spread shift as
    a dense residual it is not). The pair compiles on the host walks; the pin
    is that the model's own op plans carry no composite.
    """
    blob = _export(
        _modulate_graph(
            "npu_ada_no_one_symbolic",
            device=DeviceRef.NPU(),
            dtype=DType.float16,
            plus_one=False,
            batch="batch",
        ),
        tmp_path,
        "npu_ada_no_one_symbolic",
    )
    seqs = _model_sequences(blob)
    assert any("mo.mul" in s for s in seqs), seqs
    assert any("mo.add" in s for s in seqs), seqs
    assert not any("mo.composite" in s for s in seqs), seqs


def test_swiglu_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 SwiGLU reaches the bf16 row, carrying the packed projection.

    The composite's operand is `[4, 128]` where the result is `[4, 64]`: the
    rule handed the kernel back the row the graph had split, which is what the
    selector reads.
    """
    blob = _export(
        _swiglu_graph(
            "npu_swiglu_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
        ),
        tmp_path,
        "npu_swiglu_bf16",
    )
    assert "swiglu_bf16" in _names(blob, "swiglu", "swiglu_bf16")
    assert (
        b"mo.composite.swiglu : (!mo.tensor<[4, 128], bf16, npu:0>) ->"
        b" !mo.tensor<[4, 64], bf16, npu:0>" in blob
    )


def test_swiglu_fp16_names_the_fp16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The complement, as the modulation's is."""
    blob = _export(
        _swiglu_graph(
            "npu_swiglu_f16", device=DeviceRef.NPU(), dtype=DType.float16
        ),
        tmp_path,
        "npu_swiglu_f16",
    )
    assert "swiglu" in _names(blob, "swiglu", "swiglu_bf16")


def test_bare_silu_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A silu with no sibling to gate compiles, and reaches the silu row.

    Worth little numerically -- the census counts five in a denoise step -- and
    load-bearing structurally: without this registration the compiler lowers
    `mo.silu` to an elementwise lambda that is device code and does not compile
    here, so five ops refuse the whole graph.
    """
    blob = _export(
        _silu_graph(
            "npu_silu_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
        ),
        tmp_path,
        "npu_silu_bf16",
    )
    assert "silu_bf16" in _names(blob, "silu", "silu_bf16")
    assert b"mo.silu : (!mo.tensor<[4, 64], bf16, npu:0>" in blob


def test_bare_silu_fp16_names_the_fp16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The complement for the one op whose fp16 row had no test.

    The other three each had both dtypes covered from the start; this one did
    not, and fp16 is where the model is heading -- the platform has no usable
    bf16 multiply-accumulate, so staging FLUX at fp16 is landing separately.
    """
    blob = _export(
        _silu_graph(
            "npu_silu_f16", device=DeviceRef.NPU(), dtype=DType.float16
        ),
        tmp_path,
        "npu_silu_f16",
    )
    assert "silu" in _names(blob, "silu", "silu_bf16")
    assert b"mo.silu : (!mo.tensor<[4, 64], f16, npu:0>" in blob


def test_gated_silu_does_not_become_two_dispatches(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The gated form reaches `swiglu`, not `silu` followed by a multiply.

    Taking `mo.silu` over is a trade -- one dispatch per activation against a
    graph that will not compile -- and it would be a bad one if it also claimed
    the 30 activations that belong inside SwiGLU. The rule runs first, so the
    composite's own signature is the evidence that it did.
    """
    blob = _export(
        _swiglu_graph(
            "npu_swiglu_not_silu",
            device=DeviceRef.NPU(),
            dtype=DType.bfloat16,
        ),
        tmp_path,
        "npu_swiglu_not_silu",
    )
    assert b"mo.composite.swiglu : (!mo.tensor<[4, 128], bf16, npu:0>" in blob
    assert b"mo.silu : (!mo.tensor<[4, 64], bf16, npu:0>" not in blob


def test_cpu_is_untouched(virtual_npu: None) -> None:
    """The rules are inert off this target.

    Same expressions, same kernel package, a CPU device. The blast-radius check
    -- the rules live in the shared graph compiler, so every other target has to
    see the graph it saw before.
    """
    session = InferenceSession(devices=[CPU()])
    session.compile(
        _modulate_graph("cpu_ada", device=DeviceRef.CPU(), dtype=DType.float32)
    )
    session.compile(
        _swiglu_graph("cpu_swiglu", device=DeviceRef.CPU(), dtype=DType.float32)
    )
    session.compile(
        _silu_graph("cpu_silu", device=DeviceRef.CPU(), dtype=DType.float32)
    )
