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
"""Compiles a chain of on-device NPU dispatches as one graph.

Every other test here proves one op. This one asks whether a *chain* survives
the graph compiler, which is not deducible from the single-op tests: between
two ops the compiler is free to fuse, to canonicalize, and to turn a
materializing view into generated device code that this target cannot compile.

The chain is FLUX.2-Klein-4B's attention sub-block at its real widths (24 heads
x 128 = 3072), in fp16 -- the one dtype the whole selector set exists for --
built as the reference writes it, with a separate projection per operand:

    x [S, 3072]
      -> mo.reduce.layer_norm                     `layernorm`
      -> mo.matmul x3 with Wq, Wk, Wv [3072,3072] `matmul` x3
      -> reshape each to [1, S, 24, 128]          view
      -> mo.reduce.rms_norm on q and k            `rms_norm` x2
      -> mo.mha.no_cache                          `mha_no_cache`
      -> reshape back to [S, 3072]                view
      -> mo.matmul with Wo [3072,3072]            `matmul`

Eight dispatches over four signed selectors, built in named prefixes so a
failure names the op the chain dies at rather than just failing.

Two omissions are deliberate. The weights are graph *inputs* rather than
`mo.constant.external`, because a weight has never crossed on the graph path
and folding that unknown in would make a failure unattributable. And the chain
stops before the gated residual `x + gate * attn`, whose `+` the compiler folds
into the matmul's epilogue and which the stub then refuses by design
(`registrations.mojo:166`); `test_npu_unregistered_op.py` already records that
fold, and reproducing it here would only mask the question being asked.

`npu_attention_chain_exec.c` executes what the Windows sibling of this file
exports, so the widths and the epsilon here are the ones its reference is built
against.
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
from max.graph import (
    DevicePlacementPolicy,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# Klein-4B's attention block, at its real widths. The sequence length is the
# only dimension the proof does not depend on: `mha_no_cache` fixes the head dim
# at 128 and `matmul` needs K % 64 == 0, which 3072 satisfies.
SEQ = 16
MODEL = 3072
HEADS = 24
HEAD_DIM = 128

# Both norms are fed rows whose variance and mean square are exactly 1, so
# `var + eps` lands on 4 and the reciprocal square root on exactly 1/2 -- which
# keeps every intermediate an exactly representable fp16 value and lets the
# execution harness compare with no epsilon at all. The construction is in
# `npu_attention_chain_exec.c`; the epsilon lives here because it is baked into
# the graph that harness runs.
EPSILON = 3.0

# The winning attention score beats every other by 32 before scaling, so this
# puts the losers 16384 below the row max. `exp` of that is exactly zero in
# fp32, which makes attention an exact selection of one V row rather than a
# weighted sum a tolerance would have to cover.
SCALE = 512.0

# The chain in the order its ops are added. Every prefix is a compilable graph,
# which is how a failure gets attributed to an op rather than to "the chain".
STAGES = ("layer_norm", "project", "rms_norm", "mha", "matmul_out")


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


def _attention(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    *,
    device: DeviceRef,
    dtype: DType,
    seq: int,
) -> TensorValue:
    """`mo.mha.no_cache` with the null mask, spelled as `nn.kernels` spells it.

    Reached through `ops.custom` rather than through `max.nn`, which would pull
    the whole pipeline package in for one op. `mask_str` and the scale operand's
    CPU device are what that helper does; the stub reads both.
    """
    return ops.custom(
        "mo.mha.no_cache",
        values=[
            q,
            k,
            v,
            ops.constant(SCALE, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[TensorType(dtype, [1, seq, HEADS, HEAD_DIM], device)],
        parameters={"mask_str": "null", "local_window_size": -1},
        device=device,
    )[0].tensor


def attention_chain_graph(
    name: str,
    *,
    device: DeviceRef,
    dtype: DType,
    seq: int = SEQ,
    stop_after: str = "matmul_out",
) -> Graph:
    """Builds the chain above, up to and including `stop_after`.

    `transpose_b` is reachable only by folding: `MatmulTranspBPattern` rewrites
    a `mo.transpose` of the B operand into the attribute, and the weights are
    inputs rather than constants so the fold cannot be pre-empted by constant
    folding -- the construction `test_npu_graph_matmul.py` uses.
    """
    if stop_after not in STAGES:
        raise ValueError(f"unknown stage {stop_after!r}")

    weight = TensorType(dtype, [MODEL, MODEL], device)
    with Graph(
        name,
        input_types=[TensorType(dtype, [seq, MODEL], device)] + [weight] * 4,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, w_q, w_k, w_v, w_out = (value.tensor for value in graph.inputs)

        # The identity affine the `layernorm` selector applies, spelled as real
        # tensors because that is the shape a model meaning "no affine" arrives
        # in. The stub reads their values and refuses anything else.
        gamma = ops.constant([1.0] * MODEL, dtype, device=device)
        beta = ops.constant([0.0] * MODEL, dtype, device=device)
        normed = ops.layer_norm(x, gamma, beta, epsilon=EPSILON)
        if stop_after == "layer_norm":
            graph.output(normed)
            return graph

        bshd = [1, seq, HEADS, HEAD_DIM]
        q, k, v = (
            ops.reshape(normed @ ops.transpose(w, -1, -2), bshd)
            for w in (w_q, w_k, w_v)
        )
        if stop_after == "project":
            graph.output(q, k, v)
            return graph

        rms_weight = ops.constant([1.0] * HEAD_DIM, dtype, device=device)
        q = ops.rms_norm(q, rms_weight, epsilon=EPSILON)
        k = ops.rms_norm(k, rms_weight, epsilon=EPSILON)
        if stop_after == "rms_norm":
            graph.output(q, k)
            return graph

        attention = _attention(q, k, v, device=device, dtype=dtype, seq=seq)
        if stop_after == "mha":
            graph.output(attention)
            return graph

        merged = ops.reshape(attention, [seq, MODEL])
        graph.output(merged @ ops.transpose(w_out, -1, -2))
    return graph


def _fused_qkv_graph(
    name: str, *, device: DeviceRef, dtype: DType, materialize: bool
) -> Graph:
    """The same block with one fused QKV projection and an `mo.split`.

    The shape the chain was first written in, kept because what the compiler
    does with it is the finding recorded below. `materialize` adds the three
    split results to the graph's outputs, which changes nothing about the
    computation and everything about how the split is lowered.
    """
    with Graph(
        name,
        input_types=[
            TensorType(dtype, [SEQ, MODEL], device),
            TensorType(dtype, [3 * MODEL, MODEL], device),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, w_qkv = (value.tensor for value in graph.inputs)
        gamma = ops.constant([1.0] * MODEL, dtype, device=device)
        beta = ops.constant([0.0] * MODEL, dtype, device=device)
        normed = ops.layer_norm(x, gamma, beta, epsilon=EPSILON)
        parts = ops.split(
            normed @ ops.transpose(w_qkv, -1, -2), [MODEL] * 3, axis=1
        )
        bshd = [1, SEQ, HEADS, HEAD_DIM]
        q, k, v = (ops.reshape(part, bshd) for part in parts)
        rms_weight = ops.constant([1.0] * HEAD_DIM, dtype, device=device)
        attention = _attention(
            ops.rms_norm(q, rms_weight, epsilon=EPSILON),
            ops.rms_norm(k, rms_weight, epsilon=EPSILON),
            v,
            device=device,
            dtype=dtype,
            seq=SEQ,
        )
        if materialize:
            graph.output(attention, q, k, v)
        else:
            graph.output(attention)
    return graph


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


@pytest.mark.parametrize("stage", STAGES)
def test_npu_attention_chain_prefix_compiles(
    virtual_npu: None, stage: str
) -> None:
    """Every prefix of the chain compiles for the NPU.

    Parametrized rather than written as one test so a failure names the op:
    every prefix up to the failing one passes and the rest fail together. The
    diagnostic is reported rather than swallowed, because which refusal fired
    is the whole content of a failure here.
    """
    graph = attention_chain_graph(
        f"npu_chain_{stage}",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        stop_after=stage,
    )
    try:
        InferenceSession(devices=[NPU()]).compile(graph)
    except Exception as exc:
        pytest.fail(_compile_error(exc))


def test_npu_attention_chain_names_every_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """Every dispatch in the chain reached a signed selector.

    This is what separates "the chain compiled" from "the chain compiled onto
    the NPU": a name can only be in the artifact if the compiler instantiated
    the branch that spells it, and a built-in lowering names no selector at
    all. Presence only -- a MEF carries every kernel compiled in the process
    that exported it, which makes an absence unassertable and a presence exact.

    `layernorm` is matched NUL-terminated because it is a prefix of
    `layernorm_bf16`, which another test in the same process may have compiled.
    """
    graph = attention_chain_graph(
        "npu_attention_chain_sel", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_attention_chain_sel.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    for selector in (
        b"layernorm\x00",
        b"matmul\x00",
        b"rms_norm\x00",
        b"mha_no_cache\x00",
    ):
        assert selector in blob, selector

    # Paired with the signatures they were instantiated for, so each name is
    # tied to one of this chain's ops rather than merely present.
    for signature in (
        b"mo.reduce.layer_norm : (!mo.tensor<[16, 3072], f16, npu:0>",
        b"mo.matmul : (!mo.tensor<[16, 3072], f16, npu:0>",
        b"mo.reduce.rms_norm : (!mo.tensor<[1, 16, 24, 128], f16, npu:0>",
        b"mo.mha.no_cache : (!mo.tensor<[1, 16, 24, 128], f16, npu:0>",
    ):
        assert signature in blob, signature


def test_npu_fused_qkv_split_compiles(
    virtual_npu: None, tmp_path: Path
) -> None:
    """A fused QKV projection whose `mo.split` feeds only kernels compiles.

    It did not until the materializer resolved per device, which is the
    finding: an axis-1 split lowers as *views*, a consumer that wants a dense
    operand forces one to materialize, and materializing used to fall to
    generated device code (`elementwise` at rank 4, refused by
    `compile_function[parametric indexed-closure] is NPU-only`). The
    device-scoped `npu_view_copy` puts a host walk on that path instead.

    That the views are `mo.slice` kernels forced to materialize is what ties
    the split's fate to the view machinery at all, so it is counted rather
    than inferred from the compile succeeding.

    This retires the advice to project Q, K and V separately: the chain above
    still does, because three dispatches of a third of the work cost the same
    and the split's copy is avoided, but it is now a choice rather than the only
    graph that compiles.
    """
    graph = _fused_qkv_graph(
        "npu_fused_qkv",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        materialize=False,
    )
    session = InferenceSession(devices=[NPU()])
    session.debug.ir_output_dir = str(tmp_path)
    session.compile(graph)

    lowered = list(tmp_path.glob("*.mogg-pre-mgp.mlir"))
    assert lowered, "the graph was served from the compile cache, not measured"
    assert any('"mo.slice"' in f.read_text() for f in lowered)


def test_npu_fused_qkv_split_compiles_when_materialized(
    virtual_npu: None,
) -> None:
    """Naming the split's results as outputs is enough to reach the host copy.

    The complement of the test above, and what makes that one a statement about
    how the split is *lowered* rather than about the ops around it: the
    computation is identical and only the output list differs. So the refusal
    was the view-materialization path, not `mo.split` itself, which the
    registration handles and `test_npu_graph_movement.py` proves it handles.
    """
    graph = _fused_qkv_graph(
        "npu_fused_qkv_materialized",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        materialize=True,
    )
    InferenceSession(devices=[NPU()]).compile(graph)
