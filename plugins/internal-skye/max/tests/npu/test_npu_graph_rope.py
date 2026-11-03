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
"""Compiles FLUX.2's `mo.rope.ragged.with_position_id` onto the on-device NPU.

The op the transformer emits fifty times per denoise step, and the one whose
operands are staged rather than passed: `rope_ragged_with_position_ids`
synthesizes the row offsets and the start position, so a graph that means "one
dense stream" still arrives in ragged clothing.

Which selector the stub named is what these prove. The convention decides it --
`interleaved=True` is what FLUX.2 passes and reaches `rope_interleaved`, and the
`rotate_half` graph reaches `rope_half` -- so a MEF carrying one name and not the
other is evidence the `comptime if` fell the way the parameter says.
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
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType
from max.nn.kernels import rope_ragged_with_position_ids

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# FLUX.2-Klein's own head width, with a token count that divides across the
# kernel's four compute threads and a head count above one so the per-token to
# per-row expansion is actually exercised.
_TOKENS, _HEADS, _HEAD_DIM = 8, 2, 128


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


def _rope_graph(
    name: str,
    *,
    device: DeviceRef,
    dtype: DType,
    interleaved: bool = True,
    head_dim: int = _HEAD_DIM,
    rope_dim: int | None = None,
) -> Graph:
    """Builds one rope over `[tokens, heads, head_dim]`.

    `freqs_cis` and `position_ids` are graph inputs rather than constants so the
    shapes stay independent: `rope_dim` below `head_dim` is the partial rope the
    stub refuses, and it cannot be expressed if the two are tied.
    """
    rope_dim = head_dim if rope_dim is None else rope_dim
    with Graph(
        name,
        input_types=[
            TensorType(dtype, [_TOKENS, _HEADS, head_dim], device),
            TensorType(DType.float32, [_TOKENS, rope_dim], device),
            TensorType(DType.uint32, [1, _TOKENS], device),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        x, freqs_cis, position_ids = (v.tensor for v in g.inputs)
        g.output(
            rope_ragged_with_position_ids(
                x, freqs_cis, position_ids, interleaved=interleaved
            )
        )
    return g


def _compile_error(exc: BaseException) -> str:
    """Joins a compile failure with its causes."""
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def _selector_names(blob: bytes) -> set[bytes]:
    """The rope selector names a MEF carries, NUL-terminated as emitted.

    Terminated because `rope_interleaved` and `rope_half` are both prefixes of
    their bf16 twins, and because the stub's prose mentions neither -- an
    unterminated match would count a docstring as a dispatch.

    Only presence is meaningful: a MEF carries every kernel compiled in the
    process that exported it, so a convention compiled earlier in the same
    pytest process appears here too. Each test below therefore exports from a
    freshly built graph and asserts the name it must contain.
    """
    return {
        name
        for name in (
            b"rope_interleaved\x00",
            b"rope_half\x00",
            b"rope_interleaved_bf16\x00",
            b"rope_half_bf16\x00",
        )
        if name in blob
    }


def test_npu_rope_interleaved_names_the_interleaved_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """FLUX.2's convention compiles and reaches `rope_interleaved`.

    `interleaved=True` is what `_apply_flux2_qk_rope` passes, so this is the
    dispatch a denoise step actually makes. The signature is asserted beside the
    name so the name is tied to a rope op rather than merely present.
    """
    graph = _rope_graph(
        "npu_rope_interleaved", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rope_interleaved.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"hexagon_ondevice" in blob
    assert b"rope_interleaved\x00" in _selector_names(blob)
    assert b"mo.rope.ragged.with_position_id" in blob


def test_npu_rope_rotate_half_names_the_half_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The other convention reaches the other selector.

    The complement of the test above, and the reason both are positive claims:
    each name is emitted only by the branch its parameter selects, so seeing the
    right one rules out a `comptime if` that fell the wrong way -- which
    asserting a name's absence cannot do, the MEF carrying whatever else the
    process compiled.
    """
    graph = _rope_graph(
        "npu_rope_half",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        interleaved=False,
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rope_half.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"rope_half\x00" in _selector_names(blob)


def test_npu_rope_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 graph instantiates the stub's bf16 branch.

    The dtype twins are the other half of the selector proof: the name can only
    be in the artifact if the compiler instantiated the branch that spells it.
    """
    graph = _rope_graph(
        "npu_rope_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rope_bf16.mef"
    compiled.export_mef(mef)

    assert b"rope_interleaved_bf16\x00" in _selector_names(mef.read_bytes())


def test_npu_rope_refuses_fp32(virtual_npu: None) -> None:
    """The dtype gate did not widen to everything.

    fp32 is what the built-in accepts, so this failing with the stub's own
    message is the negative half of the selection proof.
    """
    graph = _rope_graph(
        "npu_rope_fp32", device=DeviceRef.NPU(), dtype=DType.float32
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "fp16 or bf16 only" in diagnostic


def test_cpu_rope_accepts_what_the_npu_refuses(virtual_npu: None) -> None:
    """The refusal above is the NPU's, not the graph's.

    Same shapes, same kernel package, a CPU device: the fp32 graph the NPU stub
    rejects compiles here, so the rejection is kernel selection rather than
    something the graph API would have refused anyway.
    """
    graph = _rope_graph(
        "cpu_rope_fp32", device=DeviceRef.CPU(), dtype=DType.float32
    )
    InferenceSession(devices=[CPU()]).compile(graph)
