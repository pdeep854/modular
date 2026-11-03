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
"""Cross-compiles the on-device NPU matmul graph for the Windows ARM64 host.

The NPU's host half runs on Windows ARM64 while the graph compiler runs on
Linux, so the MEF that would reach the device has to name two targets at once:
the NPU for the kernel, and the Windows host triple for the dispatch stub the
model setup JIT-links. This checks the pair is expressible.

Its own process: `set_virtual_cpu_target` is global and retargets every later
compile in the process, which is why the sibling in-process suite lives in
another target.
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

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page. Same string the host-only smoke MEF
# was built with.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

# The shape the `matmul` selector was measured bit-exact against on silicon.
_M, _N, _K = 4, 8, 64


def test_npu_matmul_cross_compiles_for_windows_arm64(tmp_path: Path) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    input_types = [
        TensorType(DType.float16, [_M, _K], dev),
        TensorType(DType.float16, [_N, _K], dev),
    ]
    with Graph(
        "npu_matmul_win",
        input_types=input_types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        a, b = (v.tensor for v in graph.inputs)
        graph.output(a @ ops.transpose(b, -1, -2))

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_matmul_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob

    # The three entry points the stub's body reaches, plus the stack probe that
    # is only emitted for a Windows target. Asserting on the kernel object's own
    # imports is what separates "a MEF came out" from "a MEF that dispatches on
    # the NPU came out, built for the right ABI".
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
