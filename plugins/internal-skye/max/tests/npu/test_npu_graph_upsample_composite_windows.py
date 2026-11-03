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
"""Cross-compiles the folded upsample-composite NPU graph for Windows ARM64.

`test_npu_graph_gated_residual_windows.py`'s sibling for the
`npu_upsample_conv` custom: staging the composite is proven device-free by
`test_npu_graph_upsample_composite.py`; this file proves the registration
ELABORATES for the production target and the exported MEF carries the
selector-45 dispatch. Its own process because `set_virtual_cpu_target` is
global and retargets every later compile.
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
from max.pipelines.architectures.autoencoders.vae_flux2 import Upsample2D

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"


def test_npu_upsample_composite_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    os.environ["MODULAR_NPU_UPSAMPLE_FOLD"] = "1"
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    up = Upsample2D(
        channels=64,
        use_conv=True,
        out_channels=64,
        kernel_size=3,
        padding=1,
        bias=True,
        interpolate=True,
        device=dev,
        dtype=DType.float16,
    )
    with Graph(
        "npu_upsample_composite_win",
        input_types=[TensorType(DType.float16, [1, 64, 16, 16], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(up(graph.inputs[0].tensor))

    # The composite must have staged, or this file certifies the wrong path.
    assert 'symbol = "npu_upsample_conv"' in str(graph)
    assert "rmo.mul" not in str(graph)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_upsample_composite_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    # The custom's registration reached elaboration and its dispatch names
    # the ep selector: this is what makes the artifact a selector-45 fold
    # rather than a graph that lowered somewhere else.
    assert b"npu_upsample_conv" in blob
    assert b"conv_im2col_hmx_ep" in blob

    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
        b"AsyncRT_DeviceContext_createBuffer_async",
        b"__chkstk",
    ):
        assert symbol in blob, symbol

    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
