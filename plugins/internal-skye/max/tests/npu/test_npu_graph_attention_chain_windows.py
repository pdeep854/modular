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
"""Cross-compiles the attention-chain NPU graph for the Windows ARM64 host.

The sibling of `test_npu_graph_layernorm_windows.py` for the first graph that
chains dispatches rather than making one: the NPU's host half runs on Windows
ARM64 while the graph compiler runs on Linux, so the MEF that reaches the
device names two targets at once. `npu_attention_chain_exec.c` is the host that
runs what this exports, and the chain it builds is documented in
`test_npu_graph_attention_chain.py`.

One artifact per prefix, because a whole-chain comparison on silicon can only
say that the numbers are wrong. The harness takes the stage name and knows what
each prefix should produce, so running the set names the op the chain first
diverges at -- the device half of the same staging the compile test uses.

Its own process, as the other cross-compiles are: `set_virtual_cpu_target` is
global and retargets every later compile in the process. The graph is restated
here rather than imported for the same reason -- each of these files is one
`py_test` with one source, so a sibling module is not on the path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from max.driver import (
    NPU,
    set_virtual_cpu_target,
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

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page. Same string the matmul MEF was built
# with.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_SEQ, _MODEL, _HEADS, _HEAD_DIM = 16, 3072, 24, 128
_EPSILON = 3.0
_SCALE = 512.0

# The selectors each prefix is expected to carry, cumulative down the chain.
_SELECTORS = {
    "layer_norm": (b"layernorm\x00",),
    "project": (b"layernorm\x00", b"matmul\x00"),
    "rms_norm": (b"layernorm\x00", b"matmul\x00", b"rms_norm\x00"),
    "mha": (
        b"layernorm\x00",
        b"matmul\x00",
        b"rms_norm\x00",
        b"mha_no_cache\x00",
    ),
    "matmul_out": (
        b"layernorm\x00",
        b"matmul\x00",
        b"rms_norm\x00",
        b"mha_no_cache\x00",
    ),
}


def test_npu_rms_norm_pair_in_reversed_order_cross_compiles(
    tmp_path: Path,
) -> None:
    """The `rms_norm` prefix with K's norm emitted before Q's.

    Discriminates position from operand for the asymmetry where the first of
    two RMS-norm dispatches came back wrong and the second exact. Outputs stay
    in (q, k) order so the harness compares against the same references; only
    the order the two ops are added to the graph changes. An error that stays
    on Q is about the operand; one that moves to K is about position, which is
    what a missing barrier would look like.
    """
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dtype = DType.float16
    device = DeviceRef.NPU()
    weight = TensorType(dtype, [_MODEL, _MODEL], device)
    bshd = [1, _SEQ, _HEADS, _HEAD_DIM]
    with Graph(
        "npu_rms_norm_rev",
        input_types=[TensorType(dtype, [_SEQ, _MODEL], device)] + [weight] * 4,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, w_q, w_k, w_v, w_out = (value.tensor for value in graph.inputs)
        gamma = ops.constant([1.0] * _MODEL, dtype, device=device)
        beta = ops.constant([0.0] * _MODEL, dtype, device=device)
        normed = ops.layer_norm(x, gamma, beta, epsilon=_EPSILON)
        q, k, v = (
            ops.reshape(normed @ ops.transpose(w, -1, -2), bshd)
            for w in (w_q, w_k, w_v)
        )
        rms = ops.constant([1.0] * _HEAD_DIM, dtype, device=device)
        # K first, Q second -- the whole point of this variant.
        k_normed = ops.rms_norm(k, rms, epsilon=_EPSILON)
        q_normed = ops.rms_norm(q, rms, epsilon=_EPSILON)
        graph.output(q_normed, k_normed)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rms_norm_rev.mef"
    compiled.export_mef(mef)
    assert b"rms_norm\x00" in mef.read_bytes()
    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)


def test_npu_rms_norm_over_a_plain_rank4_input_cross_compiles(
    tmp_path: Path,
) -> None:
    """One `mo.reduce.rms_norm` over a rank-4 graph input, no reshape anywhere.

    The discriminator for why the chain's `rms_norm` writes nothing. Every op
    proven on silicon reads a non-view operand, while the chain's `rms_norm`
    reads a `mo.static.reshape` view of the projection: in the chain the reshape
    sits on the *input* side, where it feeds a kernel's `InputTensor`, and where
    the chain reshapes a projection it is on the *output* side, materialized for
    a graph output. That difference has never been tested on device -- a rank-4
    reshape has only been proven to compile.

    So this graph removes the view and changes nothing else: same op, same
    widths, same epsilon, MAX's own tensor pointers. It passing puts the cause
    on the view; it failing says MAX tensor buffers are not reachable by the DSP
    for this op at all, and moves the question to why `matmul` and `layernorm`
    are.
    """
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dtype = DType.float16
    device = DeviceRef.NPU()
    bshd = [1, _SEQ, _HEADS, _HEAD_DIM]
    with Graph(
        "npu_rms_direct",
        input_types=[TensorType(dtype, bshd, device)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        rms = ops.constant([1.0] * _HEAD_DIM, dtype, device=device)
        graph.output(
            ops.rms_norm(graph.inputs[0].tensor, rms, epsilon=_EPSILON)
        )

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_rms_direct.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"rms_norm\x00" in blob
    assert (
        b"mo.reduce.rms_norm : (!mo.tensor<[1, 16, 24, 128], f16, npu:0>"
        in (blob)
    )
    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)


@pytest.mark.parametrize("stage", tuple(_SELECTORS))
def test_npu_attention_chain_cross_compiles_for_windows_arm64(
    tmp_path: Path, stage: str
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dtype = DType.float16
    device = DeviceRef.NPU()
    weight = TensorType(dtype, [_MODEL, _MODEL], device)
    bshd = [1, _SEQ, _HEADS, _HEAD_DIM]
    with Graph(
        f"npu_attention_chain_win_{stage}",
        input_types=[TensorType(dtype, [_SEQ, _MODEL], device)] + [weight] * 4,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, w_q, w_k, w_v, w_out = (value.tensor for value in graph.inputs)
        gamma = ops.constant([1.0] * _MODEL, dtype, device=device)
        beta = ops.constant([0.0] * _MODEL, dtype, device=device)
        normed = ops.layer_norm(x, gamma, beta, epsilon=_EPSILON)
        if stage == "layer_norm":
            graph.output(normed)
        else:
            q, k, v = (
                ops.reshape(normed @ ops.transpose(w, -1, -2), bshd)
                for w in (w_q, w_k, w_v)
            )
            if stage == "project":
                graph.output(q, k, v)
            else:
                rms = ops.constant([1.0] * _HEAD_DIM, dtype, device=device)
                q = ops.rms_norm(q, rms, epsilon=_EPSILON)
                k = ops.rms_norm(k, rms, epsilon=_EPSILON)
                if stage == "rms_norm":
                    graph.output(q, k)
                else:
                    attention = ops.custom(
                        "mo.mha.no_cache",
                        values=[
                            q,
                            k,
                            v,
                            ops.constant(
                                _SCALE,
                                dtype=DType.float32,
                                device=DeviceRef.CPU(),
                            ),
                        ],
                        out_types=[TensorType(dtype, bshd, device)],
                        parameters={
                            "mask_str": "null",
                            "local_window_size": -1,
                        },
                        device=device,
                    )[0].tensor
                    if stage == "mha":
                        graph.output(attention)
                    else:
                        merged = ops.reshape(attention, [_SEQ, _MODEL])
                        graph.output(merged @ ops.transpose(w_out, -1, -2))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"npu_attention_chain_win_{stage}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    # The selectors this prefix dispatches, so the artifact is a chain of NPU
    # dispatches rather than a chain that lowered somewhere else. Presence
    # only: every stage compiles in the same process and a MEF carries whatever
    # else that process compiled, so an absence would say nothing.
    for selector in _SELECTORS[stage]:
        assert selector in blob, selector

    # The three entry points the stubs' bodies reach, plus the stack probe that
    # is only emitted for a Windows target.
    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
        b"AsyncRT_DeviceContext_createBuffer_async",
        b"__chkstk",
    ):
        assert symbol in blob, symbol

    # The Windows host half is built and run out of band, so leave the artifact
    # where a run can pick it up rather than making it regenerate one.
    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
