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
"""Cross-compiles the bf16 gated-residual NPU graph for the Windows ARM64 host.

The sibling of `test_npu_graph_layernorm_windows.py` for the second op with a
signed bf16 selector: the NPU's host half runs on Windows ARM64 while the graph
compiler runs on Linux, so the MEF that reaches the device names two targets at
once. `npu_gated_residual_bf16_exec.c` is the host that runs what this exports.

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
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page. Same string the layer norm MEF was
# built with.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_ROWS, _LENGTH = 4, 64


def test_npu_gated_residual_bf16_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    types = [
        TensorType(DType.bfloat16, [_ROWS, _LENGTH], dev),
        TensorType(DType.bfloat16, [_ROWS, _LENGTH], dev),
        TensorType(DType.bfloat16, [1, _LENGTH], dev),
    ]
    with Graph(
        "npu_gate_residual_bf16_win",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, y, gate = (v.tensor for v in graph.inputs)
        graph.output(x + gate * y)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_gate_residual_bf16_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    # The bf16 selector by name: this is what makes the artifact a bf16 dispatch
    # rather than a bf16 graph that lowered somewhere else.
    assert b"gate_residual_bf16" in blob

    # The three entry points the stub's body reaches, plus the stack probe that
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
