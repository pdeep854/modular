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
"""Cross-compiles ONE real decoder conv as its own MEF: the activation probe.

The one-conv falsification instrument scoped by
`.agentwork/flux-npu/upstreaming/VAE_CONV_LAYOUT_STUDY.md` §5: the whole-graph
VAE decode renders a flat-gray field whose signature matches activation-path
corruption (H8/H8b), not any weight-side hypothesis, and this MEF isolates a
single conv dispatch — input bind, the compiler's NCHW/NHWC permutes, the
`mo.conv` registration's FCRS->(r,s,c) filter staging, one `conv_im2col`
enqueue, bias add, output read — from the rest of the graph. Executed on
device against a host fp32 reference of the same weights and input, it splits
the 298-clean world in one execute: within the fp16 band means the activation
path is clean for this conv class; garbage means the corruption is localized
to a single conv dispatch.

The layer is `decoder.mid_block.resnets.0.conv1` — the smallest-activation
real 3x3 in the decoder (512 channels at 64x64, M=4096, K=4608), served by
the FUSED arm (`C_in % 64 == 0`) exactly as 32 of the 33 whole-graph convs
are. The seam is built with the very same `max.nn.Conv2d(permute=True)` the
decoder uses (`vae_flux2.ResnetBlock2D`), so the MEF contains the identical
permute/conv/bias op sequence, not a lookalike.

Weights stay external (`mo.constant.external`) under their checkpoint FQNs,
so the device client binds the same registry rows the whole-graph MEF binds.

Its own process, as every cross-compile here is: `set_virtual_cpu_target` is
global and retargets every later compile in the process.
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
)
from max.nn.conv import Conv2d

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_LAYER = "decoder.mid_block.resnets.0.conv1"
_CHANNELS = 512
_SPATIAL = 64


def test_npu_conv_mid_fp16_cross_compile_for_windows_arm64(
    tmp_path: Path,
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    conv = Conv2d(
        kernel_size=3,
        in_channels=_CHANNELS,
        out_channels=_CHANNELS,
        dtype=DType.float16,
        stride=1,
        padding=1,
        dilation=1,
        num_groups=1,
        has_bias=True,
        device=dev,
        permute=True,
        name=_LAYER,
    )

    with Graph(
        "npu_conv_mid_fp16_win",
        input_types=[
            TensorType(
                DType.float16, [1, _CHANNELS, _SPATIAL, _SPATIAL], dev
            ),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(conv(graph.inputs[0].tensor))

    # The weights must ride as external constants under the checkpoint FQNs:
    # that is what lets the device client bind the SAME registry rows the
    # whole-graph MEF binds, and a graph that constant-folded them would probe
    # nothing.
    graph_str = str(graph)
    assert "mo.constant.external" in graph_str
    assert _LAYER + ".weight" in graph_str

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_conv_mid_fp16_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    assert (_LAYER + ".weight").encode() in blob
    assert (_LAYER + ".bias").encode() in blob

    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
        b"AsyncRT_DeviceContext_createBuffer_async",
        b"__chkstk",
    ):
        assert symbol in blob, symbol

    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
