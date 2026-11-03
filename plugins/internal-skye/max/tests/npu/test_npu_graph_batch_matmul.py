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
"""Compiles a rank-3 matmul onto the on-device NPU as `mo.batch_matmul`.

`mo.matmul` is rank-2 by op contract, so a Linear applied to a
`[batch, seq, dim]` activation legalizes to `mo.batch_matmul` instead. That op
had no NPU registration, so most of a real model's matmuls could not reach the
one selector proven bit-exact on silicon even though it is the kernel they need.

Which kernel the compiler chose is the thing worth proving, because a built-in
batched matmul would also compile and would say nothing about the NPU. The
stub's own refusals are the discriminator: `transpose_b` and fp16 are `comptime
assert`s inside it and the built-in accepts both, so a graph that fails to
compile with the stub's message can only have been lowered onto the stub.
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

# The per-slice shape the `matmul` selector was measured bit-exact against on
# silicon, under the batch of 1 a diffusion graph stages. The selector reads K
# in 64-element vectors; a ragged K is zero-padded to the next multiple of 64
# at dispatch, host-side.
_M, _N, _K = 4, 8, 64

# The leading extent for the rank-3 cases, and deliberately not 1. A *static* 1
# is collapsed away before matmul legalization, so the op becomes `mo.matmul`
# rather than the batched one -- see
# `test_a_static_unit_batch_collapses_to_mo_matmul`. A real diffusion graph
# stages its batch symbolically, which the compiler cannot fold, so a non-unit
# extent is the representative case here.
_BATCH = 2


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


def _batch_matmul_graph(
    name: str,
    *,
    device: DeviceRef,
    dtype: DType,
    transpose_b: bool,
    batch: list[int | str] | None = None,
    k: int = _K,
) -> Graph:
    """Builds `A @ B` over leading dims, as one `mo.batch_matmul`.

    Rank 3 or more is what makes it the batched op rather than `mo.matmul`
    (`LegalizeRMOOps` splits on exactly that). `MatmulTranspBPattern` folds a
    `mo.transpose` of the B operand into the matmul's `transpose_b` attribute,
    which is the only way to reach the attribute from the graph API; B is a
    graph input rather than a constant so the fold cannot be pre-empted by
    constant folding.
    """
    lead: list[int | str] = [_BATCH] if batch is None else list(batch)
    b_tail = [_N, k] if transpose_b else [k, _N]
    input_types = [
        TensorType(dtype, [*lead, _M, k], device),
        TensorType(dtype, [*lead, *b_tail], device),
    ]
    with Graph(
        name,
        input_types=input_types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        a, b = (v.tensor for v in g.inputs)
        g.output(a @ (ops.transpose(b, -1, -2) if transpose_b else b))
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


def test_npu_batch_matmul_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """A rank-3 matmul on an NPU device compiles and serializes."""
    graph = _batch_matmul_graph(
        "npu_batch_matmul",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_batch_matmul.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    # The registration package's own name, and the selector this stub dispatches
    # paired with the batched op's signature -- so the name is tied to a rank-3
    # matmul rather than merely present. NUL-terminated because the stub's
    # refusal messages mention the selector in prose.
    assert b"hexagon_ondevice" in blob
    assert b"matmul\x00" in blob
    assert b"mo.batch_matmul : (!mo.tensor<[2, 4, 64], f16, npu:0>" in blob


def test_npu_batch_matmul_takes_a_ragged_k(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A rank-3 matmul whose K is not a multiple of 64 compiles.

    The stub zero-pads K to 64 at dispatch (execute-side, proven host-side in
    `kernels/test/hexagon_ondevice/test_matmul_k_pad.mojo`); what a compile
    can pin is that the ragged shape stages onto the NPU registration.
    """
    graph = _batch_matmul_graph(
        "npu_batch_matmul_k32",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
        k=32,
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_batch_matmul_k32.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b"mo.batch_matmul : (!mo.tensor<[2, 4, 32], f16, npu:0>" in blob


def test_npu_batch_matmul_selects_the_npu_registration(
    virtual_npu: None,
) -> None:
    """The stub is the kernel chosen, not a built-in that also accepts fp16.

    Its `transpose_b` assert fires only if the compiler instantiated it, and the
    message is unique to `hexagon_ondevice/registrations.mojo`.
    """
    graph = _batch_matmul_graph(
        "npu_batch_matmul_no_transpose",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=False,
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "mo.batch_matmul on the on-device NPU" in diagnostic
    assert "transpose_b must be set" in diagnostic


def test_npu_batch_matmul_refuses_fp32(virtual_npu: None) -> None:
    """A second refusal from the same stub, on a different constraint.

    fp16-only rather than the two dtypes the norms now take, because
    `NPUKernels.def` carries no `matmul_bf16`: a bf16 batched matmul needs
    device code, not a second selector name.
    """
    graph = _batch_matmul_graph(
        "npu_batch_matmul_fp32",
        device=DeviceRef.NPU(),
        dtype=DType.float32,
        transpose_b=True,
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "mo.batch_matmul on the on-device NPU: fp16 only" in diagnostic


def test_npu_batch_matmul_refuses_bf16(virtual_npu: None) -> None:
    """The dtype the two norms now accept is still refused here.

    Worth its own case because the norms were widened in the same series of
    changes and the manifest does carry bf16 twins of *those* selectors; reading
    across from them to this one is the mistake this pins.
    """
    graph = _batch_matmul_graph(
        "npu_batch_matmul_bf16",
        device=DeviceRef.NPU(),
        dtype=DType.bfloat16,
        transpose_b=True,
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "mo.batch_matmul on the on-device NPU: fp16 only" in diagnostic


def test_npu_batch_matmul_compiles_over_several_leading_dims(
    virtual_npu: None, tmp_path: Path
) -> None:
    """Rank 4 with a real batch, so the slice loop is not always one iteration.

    The attention-shaped case: `[batch, heads, seq, dim]`, where the stub walks
    the two leading dims as `batch * heads` dispatches of the signed selector.
    """
    graph = _batch_matmul_graph(
        "npu_batch_matmul_rank4",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
        batch=[2, 3],
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_batch_matmul_rank4.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"matmul\x00" in blob
    assert b"mo.batch_matmul : (!mo.tensor<[2, 3, 4, 64], f16, npu:0>" in blob


def test_npu_batch_matmul_takes_a_symbolic_batch(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The shape class a real diffusion graph actually stages.

    Its batch is symbolic, not a literal, which is what makes this the case that
    decides the item's magnitude: a symbolic extent cannot be folded to 1, so it
    stays rank 3 through `LegalizeRMOOps`' split and becomes `mo.batch_matmul`
    -- measured here rather than derived from the static cases above, which
    behave differently in both directions (a literal 1 collapses, a literal 2
    does not).
    """
    graph = _batch_matmul_graph(
        "npu_batch_matmul_symbolic",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
        batch=["batch"],
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_batch_matmul_symbolic.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"matmul\x00" in blob
    # A symbolic extent is rendered `dyn`, and it is still there at rank 3: the
    # squeeze that catches a literal 1 cannot fire on a dimension whose value
    # the compiler does not know.
    assert b"mo.batch_matmul : (!mo.tensor<[dyn, 4, 64], f16, npu:0>" in blob


def test_a_static_unit_batch_collapses_to_mo_matmul(virtual_npu: None) -> None:
    """A statically-1 leading dim does not reach this registration at all.

    Measured, and it cuts against reading `LegalizeRMOOps`' `fullRank == 2`
    split as the whole story: the same rank-3 graph with a leading extent of a
    literal 1 is squeezed before that split, so it becomes `mo.matmul` -- the
    refusal below is `Matmul`'s message, not `BatchMatmul`'s. It also arrives
    with `transpose_b` unset, because the reshape the squeeze inserts sits
    between the transpose and the matmul where `MatmulTranspBPattern` can no
    longer fold it.

    So a graph that pins its batch to a literal 1 is refused today even though
    the equivalent computation at a symbolic or non-unit batch compiles. This is
    pinned rather than fixed because the fix belongs in the fold, not here, and
    because a model that stages its batch symbolically -- which is what the
    diffusion graph does -- never takes this path.
    """
    graph = _batch_matmul_graph(
        "npu_unit_batch",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
        batch=[1],
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "mo.matmul on the on-device NPU" in diagnostic
    assert "mo.batch_matmul" not in diagnostic
    assert "transpose_b must be set" in diagnostic


def test_cpu_batch_matmul_accepts_what_the_npu_refuses(
    virtual_npu: None,
) -> None:
    """The refusals above are the NPU's, not the graph's.

    Same shapes, same kernel package, a CPU device: both configurations the NPU
    stub rejects compile here, so the rejection is kernel selection rather than
    something the graph API would have refused anyway.
    """
    session = InferenceSession(devices=[CPU()])
    for name, dtype, transpose_b in (
        ("cpu_batch_matmul_no_transpose", DType.float32, False),
        ("cpu_batch_matmul_fp32", DType.float32, True),
    ):
        graph = _batch_matmul_graph(
            name,
            device=DeviceRef.CPU(),
            dtype=dtype,
            transpose_b=transpose_b,
        )
        session.compile(graph)
