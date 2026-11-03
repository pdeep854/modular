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
"""Cross-compiles the bf16 modulation NPU graph for the Windows ARM64 host.

One graph with three outputs, so a single device run exercises all three of the
selectors this branch reached: `ada_modulate_bf16`, `swiglu_bf16` and
`silu_bf16`. `npu_modulation_bf16_exec.c` is the host that runs what this
exports.

Its own process, as the layer norm's cross-compile is: `set_virtual_cpu_target`
is global and retargets every later compile in the process.
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

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_ROWS, _LENGTH = 4, 64


def test_npu_modulation_bf16_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    types = [
        # adaLN: x, scale, shift.
        TensorType(DType.bfloat16, [1, _ROWS, _LENGTH], dev),
        TensorType(DType.bfloat16, [1, 1, _LENGTH], dev),
        TensorType(DType.bfloat16, [1, 1, _LENGTH], dev),
        # SwiGLU: one packed projection.
        TensorType(DType.bfloat16, [_ROWS, 2 * _LENGTH], dev),
        # Bare SiLU.
        TensorType(DType.bfloat16, [_ROWS, _LENGTH], dev),
    ]
    with Graph(
        "npu_modulation_bf16_win",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, scale, shift, proj, z = (v.tensor for v in graph.inputs)
        gate, up = ops.chunk(proj, 2, axis=-1)
        graph.output(
            (1 + scale) * x + shift,
            ops.silu(gate) * up,
            ops.silu(z),
        )

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_modulation_bf16_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob

    # The three selectors by name: this is what makes the artifact three bf16
    # dispatches rather than a bf16 graph that lowered somewhere else.
    for selector in (b"ada_modulate_bf16", b"swiglu_bf16", b"silu_bf16"):
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
