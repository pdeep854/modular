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
"""Compiles a bf16 `mo.reduce.group_norm` onto the on-device NPU.

The sibling of `test_npu_graph_layernorm.py` for the second op whose signed
selector already has a bf16 twin: `NPUKernels.def` carries `groupnorm_bf16`
beside `groupnorm` at the same wire shape, so widening this registration's dtype
gate reaches it with no new device code.

Which selector the stub named is the thing worth proving, since a built-in group
norm would also compile a bf16 graph and would say nothing about the NPU. Two
discriminators, in opposite directions: the exported MEF has to carry the bf16
selector's name, and fp32 has to be refused by this stub's own message while the
same graph on a CPU device fails from a different file entirely.
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

# `[N, C, spatial]`, two groups of two channels. See `npu_groupnorm_bf16_exec.c`
# for why these extents make the expected output exact: each group's 64 elements
# are four periods of the 16-value pattern, so the variance lands on 85 and this
# eps puts `var + eps` on 256.
_N, _CHANNELS, _SPATIAL = 1, 4, 32
_GROUPS = 2
_EPSILON = 171.0

# Per-channel affine, powers of two against integer offsets so the affine is
# exercised rather than skipped and the result stays exactly representable. The
# offsets are 16 apart while the scaled values stay within +-7.5, which puts
# each channel's outputs in a range of its own.
_GAMMA = [1.0, 2.0, 4.0, 8.0]
_BETA = [0.0, 16.0, 32.0, 48.0]


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


def _group_norm_graph(name: str, *, device: DeviceRef, dtype: DType) -> Graph:
    """Builds one `mo.reduce.group_norm` over `[N, C, spatial]`."""
    with Graph(
        name,
        input_types=[
            TensorType(dtype, [_N, _CHANNELS, _SPATIAL], device),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        x = g.inputs[0].tensor
        gamma = ops.constant(_GAMMA, dtype, device=device)
        beta = ops.constant(_BETA, dtype, device=device)
        g.output(
            ops.group_norm(x, gamma, beta, num_groups=_GROUPS, epsilon=_EPSILON)
        )
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


def _selector_names(blob: bytes) -> set[bytes]:
    """The selector names a MEF carries, NUL-terminated as they are emitted.

    Terminated because `groupnorm` is a prefix of `groupnorm_bf16`, so an
    unterminated match would count the bf16 dispatch as an fp16 one.

    Only presence is meaningful. A MEF carries every kernel compiled in the
    process that exported it, not only the ones its own graph reaches, so a
    second dtype compiled earlier in the same pytest process appears in this set
    too. That makes the absence of a name unassertable and its presence exact.
    """
    return {
        name
        for name in (b"groupnorm\x00", b"groupnorm_bf16\x00")
        if name in blob
    }


def test_npu_groupnorm_bf16_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """A bf16 `mo.reduce.group_norm` on an NPU device compiles."""
    graph = _group_norm_graph(
        "npu_groupnorm_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_groupnorm_bf16.mef"
    compiled.export_mef(mef)
    assert b"hexagon_ondevice" in mef.read_bytes()


def test_npu_groupnorm_bf16_names_the_bf16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A bf16 graph instantiates the stub's bf16 branch.

    The name can only be in the artifact if the compiler instantiated the branch
    that spells it: a built-in group norm names no selector at all, and the fp16
    branch names a different one. Paired with the signature it was compiled
    against, so the name is tied to a bf16 op rather than merely present.
    """
    graph = _group_norm_graph(
        "npu_groupnorm_sel_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_groupnorm_sel_bf16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"groupnorm_bf16\x00" in _selector_names(blob)
    signature = b"mo.reduce.group_norm : (!mo.tensor<[1, 4, 32], bf16, npu:0>"
    assert signature in blob


def test_npu_groupnorm_fp16_still_names_the_fp16_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """Widening to bf16 did not repoint the fp16 path at the bf16 selector.

    The complement of the test above, and the reason both are positive claims:
    the fp16 name is emitted only by the branch this dtype must take, so seeing
    it rules out a `comptime if` that fell the wrong way -- which asserting the
    bf16 name's absence cannot do here, the MEF carrying whatever else the
    process compiled.
    """
    graph = _group_norm_graph(
        "npu_groupnorm_sel_f16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_groupnorm_sel_f16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"groupnorm\x00" in _selector_names(blob)
    assert b"mo.reduce.group_norm : (!mo.tensor<[1, 4, 32], f16, npu:0>" in blob


def test_npu_groupnorm_refuses_fp32(virtual_npu: None) -> None:
    """Widening to bf16 did not widen to everything.

    Failing with this stub's own message, naming this stub's own file, is the
    negative half of the selection proof; the CPU test below is what rules out a
    refusal the op would have produced anywhere.
    """
    graph = _group_norm_graph(
        "npu_groupnorm_fp32", device=DeviceRef.NPU(), dtype=DType.float32
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "fp16 or bf16 only" in diagnostic


def test_cpu_groupnorm_fails_somewhere_else_entirely(
    virtual_npu: None,
) -> None:
    """The refusal above is this stub's, established by the CPU differential.

    Until the 2026-08 upstream sync, the same graph on a CPU device also
    failed -- from MAX's own `nn/normalization.mojo`, with "group_norm only
    supports GPU targets" -- and the control was that the two devices refused
    from two different files. Upstream has since grown a CPU
    `mo.reduce.group_norm` (the normalization.mojo rewrite in that sync
    window), so the control sharpened: the identical graph now COMPILES on a
    CPU device. A refusal that appears only when the NPU is the target, on a
    graph another device accepts outright, is attributable to this package's
    registration having been selected -- not to anything the op would refuse
    wherever it ran.

    If this test starts raising again, read where from: a CPU-side error
    naming `hexagon_ondevice` means the NPU registration leaked into CPU
    selection, which is the failure this control exists to catch.
    """
    graph = _group_norm_graph(
        "cpu_groupnorm_fp32", device=DeviceRef.CPU(), dtype=DType.float32
    )
    InferenceSession(devices=[CPU()]).compile(graph)
