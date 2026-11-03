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
"""Compiles an fp16 `mo.mha.no_cache` onto the on-device NPU.

The registration has been in the tree since the `mha_no_cache` selector was
signed, but no graph had ever been built against it, so nothing established that
FLUX.2's unmasked attention reaches it rather than a built-in.

The op is staged through `ops.custom` rather than `max.nn`'s
`flash_attention_gpu` helper, which builds exactly this op: the parameters the
stub gates on -- `mask_str` and `local_window_size` -- are then written where a
reader can see them, and the test needs no dependency on the attention package
to say what it is asserting.

The stub's `comptime assert`s are the discriminators. bf16 and a causal mask are
both accepted by the built-in attention kernels and refused here, so a graph
that fails with these messages can only have been lowered onto the stub.
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

# BSHD, at the batch of 1 and the head dim of 128 the flash selector is
# written for. The batch bound is enforced by the stub from tensor extents at
# execution rather than at compile time, so a host with no device cannot
# exercise that refusal; a single head of any other dim is not refused but
# decomposed onto the signed matmul/softmax/matmul selectors (the wide
# single-head test below).
_BATCH, _SEQ_LEN, _HEADS, _HEAD_DIM = 1, 16, 4, 128

# `MaskName.NULL`. FLUX.2's denoiser attends over an unmasked sequence, which is
# the only variant this selector implements.
_NULL_MASK = "null"

_SCALE = 0.088388


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


def _mha_graph(
    name: str,
    *,
    device: DeviceRef,
    dtype: DType,
    mask_str: str = _NULL_MASK,
    local_window_size: int = -1,
    shape: tuple[int | str, ...] = (_BATCH, _SEQ_LEN, _HEADS, _HEAD_DIM),
) -> Graph:
    """Builds one `mo.mha.no_cache` over BSHD Q, K and V of one shape."""
    qkv = TensorType(dtype, list(shape), device)
    with Graph(
        name,
        input_types=[qkv, qkv, qkv],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        q, k, v = (value.tensor for value in g.inputs)
        scale = ops.constant(_SCALE, DType.float32, device=DeviceRef.CPU())
        g.output(
            ops.custom(
                "mo.mha.no_cache",
                device=device,
                values=[q, k, v, scale],
                out_types=[TensorType(dtype, q.shape, device)],
                parameters={
                    "mask_str": mask_str,
                    "local_window_size": local_window_size,
                },
            )[0].tensor
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


def test_npu_mha_no_cache_fp16_compiles(
    virtual_npu: None, tmp_path: Path
) -> None:
    """An fp16 unmasked `mo.mha.no_cache` compiles and serializes."""
    graph = _mha_graph(
        "npu_mha_no_cache_fp16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)

    mef = tmp_path / "npu_mha_no_cache_fp16.mef"
    compiled.export_mef(mef)
    assert b"hexagon_ondevice" in mef.read_bytes()


def test_npu_mha_no_cache_fp16_names_the_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """An fp16 graph instantiates the stub, not a built-in attention kernel.

    The name can only be in the artifact if the compiler instantiated the branch
    that spells it. Paired with the signature it was compiled against, so the
    name is tied to an fp16 BSHD op rather than merely present.
    """
    graph = _mha_graph(
        "npu_mha_no_cache_sel_f16", device=DeviceRef.NPU(), dtype=DType.float16
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_mha_no_cache_sel_f16.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"mha_no_cache\x00" in blob
    assert b"mo.mha.no_cache : (!mo.tensor<[1, 16, 4, 128], f16, npu:0>" in blob


def test_npu_mha_no_cache_takes_a_wide_single_head(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The VAE mid-block shape: ONE head of dim 512, batch symbolic.

    A head dim off the flash kernel's 128 no longer refuses at first execute:
    the registration decomposes it onto the signed matmul/softmax/matmul
    selectors, the hand driver's certified sequence. The decomposition is
    execute-side and proven host-side in
    `kernels/test/hexagon_ondevice/test_mha_decompose.mojo`; what a compile
    pins is that the shape stages onto this registration, that the artifact
    carries the decomposition's selector names, and that the head-dim raise
    the decomposition replaced is out of the artifact -- the registration
    body is compiled INTO the MEF, so the string's absence is the fix's
    byte-level receipt.
    """
    graph = _mha_graph(
        "npu_mha_no_cache_wide_single_head",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        shape=("batch", 4096, 1, 512),
    )
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_mha_no_cache_wide.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    assert b"hexagon_ondevice" in blob
    assert (
        b"mo.mha.no_cache : (!mo.tensor<[dyn, 4096, 1, 512], f16, npu:0>"
        in blob
    )
    assert b"matmul\x00" in blob
    assert b"softmax\x00" in blob
    assert b"head dim must be" not in blob
    # The crouton staging gate rides the same receipt: the env name is read
    # by the decompose's B staging (order 307), so its presence in the blob
    # is the fix's engagement proof -- pre-fix MEFs carry zero occurrences.
    assert b"MODULAR_NPU_WEIGHT_NATIVE" in blob


def test_npu_mha_no_cache_refuses_bf16(virtual_npu: None) -> None:
    """bf16 is refused, which is why FLUX.2 has to stage fp16 to reach this."""
    graph = _mha_graph(
        "npu_mha_no_cache_bf16", device=DeviceRef.NPU(), dtype=DType.bfloat16
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "fp16 only" in diagnostic


def test_npu_mha_no_cache_refuses_a_mask(virtual_npu: None) -> None:
    """The dtype gate is not the only one, so fp16 alone does not imply a match.

    `mha_no_cache` applies no mask, and the separately signed `mha_masked`
    selector is not reached by any registration yet, so a causal graph has to
    fail rather than silently attend over the whole sequence.
    """
    graph = _mha_graph(
        "npu_mha_no_cache_causal",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        mask_str="causal",
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "mha_masked" in diagnostic


def test_npu_mha_no_cache_refuses_a_sliding_window(virtual_npu: None) -> None:
    """A window would restrict which keys each query sees; the kernel has none."""
    graph = _mha_graph(
        "npu_mha_no_cache_window",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        local_window_size=8,
    )
    with pytest.raises(Exception) as excinfo:
        InferenceSession(devices=[NPU()]).compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "no sliding window" in diagnostic
