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
"""Cross-compiles all four composites at **fp16** for the Windows ARM64 host.

The bf16 sibling of this proved the four selectors on silicon at the dtype the
model was staged in when they were written. Staging FLUX at fp16 is landing
separately -- this part has no usable bf16 multiply-accumulate, so bf16 costs a
widening penalty the platform does not have to pay -- which makes "do these fire
at fp16?" a question the tree should answer rather than assume.

The signed manifest carries an fp16 row for each of the four beside its `_bf16`
twin, and neither the PatternFusion rules nor the registrations gate on bf16.
`npu_modulation_fp16_exec.c` is what turns that from a reading of the code into
a run on the device.

Its own process: `set_virtual_cpu_target` is global and retargets every later
compile in the process.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

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

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_ROWS, _LENGTH = 4, 64


def test_npu_modulation_fp16_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    types = [
        # adaLN: x, scale, shift.
        TensorType(DType.float16, [1, _ROWS, _LENGTH], dev),
        TensorType(DType.float16, [1, 1, _LENGTH], dev),
        TensorType(DType.float16, [1, 1, _LENGTH], dev),
        # SwiGLU: one packed projection.
        TensorType(DType.float16, [_ROWS, 2 * _LENGTH], dev),
        # Bare SiLU.
        TensorType(DType.float16, [_ROWS, _LENGTH], dev),
        # Gated residual: the branch output and its per-channel gate.
        TensorType(DType.float16, [1, _ROWS, _LENGTH], dev),
        TensorType(DType.float16, [1, 1, _LENGTH], dev),
    ]
    with Graph(
        "npu_modulation_fp16_win",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, scale, shift, proj, z, y, gate = (v.tensor for v in graph.inputs)
        gate_half, up = ops.chunk(proj, 2, axis=-1)
        graph.output(
            (1 + scale) * x + shift,
            ops.silu(gate_half) * up,
            ops.silu(z),
            x + gate * y,
        )

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_modulation_fp16_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob

    # The four fp16 rows by name, each NUL-terminated: every one is a prefix of
    # its bf16 twin, so an unterminated match would count a bf16 dispatch as an
    # fp16 one and this test would pass without proving anything.
    for selector in (
        b"ada_modulate\x00",
        b"swiglu\x00",
        b"silu\x00",
        b"gate_residual\x00",
    ):
        assert selector in blob, selector

    # Paired with the ops' own signatures, so each name is tied to an fp16
    # operation rather than merely present in an artifact.
    for signature in (
        b"mo.composite.ada_modulate : (!mo.tensor<[1, 4, 64], f16, npu:0>",
        b"mo.composite.swiglu : (!mo.tensor<[4, 128], f16, npu:0>",
        b"mo.composite.gated_residual : (!mo.tensor<[1, 4, 64], f16, npu:0>",
        b"mo.silu : (!mo.tensor<[4, 64], f16, npu:0>",
    ):
        assert signature in blob, signature

    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
        b"AsyncRT_DeviceContext_createBuffer_async",
        b"__chkstk",
    ):
        assert symbol in blob, symbol

    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
