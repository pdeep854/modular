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
"""Cross-compiles a bf16 group norm whose gamma and beta are external weights.

The sibling of `test_npu_graph_groupnorm_windows.py`, differing in one thing
that has never been exercised on this device: gamma and beta arrive as
`mo.constant.external` rather than as inline constants, so the runtime has to
look them up in the weights registry `M_initModel` is given, allocate a device
buffer for each, and copy the host bytes across. Every NPU proof to date passes
a null registry.

`group_norm` is the only op that can carry the probe. All 41 of the denoise
graph's layer norms are identity-affine and the NPU's `layer_norm` registration
demands exactly that, so the one op lowering for the model today takes no weight
at all -- a weighted dispatch has to be built deliberately.

The dispatch that results mixes both kinds of allocation: output, input and the
witness are buffers the runtime creates, while gamma and beta are weights. That
mixture is the case the registry actually has to survive.

Its own process, as the other cross-compiles are: `set_virtual_cpu_target` is
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
    Weight,
    ops,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_N, _CHANNELS, _SPATIAL = 1, 4, 32
_GROUPS = 2
_EPSILON = 171.0

# Fully qualified the way a checkpointed build names a weight, so the harness's
# registry keys are the shape a real model would present.
_GAMMA = "group_norm.weight"
_BETA = "group_norm.bias"


def test_npu_groupnorm_bf16_weights_cross_compile_for_windows_arm64(
    tmp_path: Path,
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    with Graph(
        "npu_groupnorm_bf16_weights_win",
        input_types=[
            TensorType(DType.bfloat16, [_N, _CHANNELS, _SPATIAL], dev),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x = graph.inputs[0].tensor
        # `add_weight` is the route a real model takes, and the one that has a
        # lowering here: a `constant.external` declared straight onto the device
        # lowers to `deviceBufferConstantExternal`, which takes the registry's
        # pointer to already be a device address. Declared on the host it
        # becomes an allocate-and-copy, which is the crossing this probe is for.
        gamma = graph.add_weight(
            Weight(_GAMMA, DType.bfloat16, [_CHANNELS], device=dev)
        )
        beta = graph.add_weight(
            Weight(_BETA, DType.bfloat16, [_CHANNELS], device=dev)
        )
        graph.output(
            ops.group_norm(x, gamma, beta, num_groups=_GROUPS, epsilon=_EPSILON)
        )

    # The weights are external, so they are unresolved at compile time and the
    # graph must still compile: this is the assertion that the NPU path does not
    # need a value to plan an external constant.
    assert "mo.constant.external" in str(graph)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_groupnorm_bf16_weights_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    assert b"groupnorm_bf16" in blob

    # The names the host has to key its registry on. A MEF that lost them would
    # take a null registry silently and read whatever the allocation held.
    assert _GAMMA.encode() in blob
    assert _BETA.encode() in blob

    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
        b"AsyncRT_DeviceContext_createBuffer_async",
        b"__chkstk",
    ):
        assert symbol in blob, symbol

    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
