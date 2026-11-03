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
"""Cross-compiles bare-`mo.add` NPU graphs for the Windows ARM64 host.

The `Add` registration's device arm picks its route at serve time
(`MODULAR_NPU_ADD_DEVICE`), so a compile test cannot watch a dispatch; what a
MEF can prove is which selector names the compiled stub reaches, because a
selector string lands in the artifact only through the stub's own
`load_function` call. The whole-graph mints established that names never leak
from the embedded device package: the encoder MEF greps zero for
`rope_table`/`sinusoid_embed` although the package carries both kernels
(`MACH_NPU_STAGE_MINTS.md`, byte receipts).

The two graphs here are each other's can-fail controls. The fp16 add must
carry `gate_residual` (the ones-gate arm) and no `vadd` -- that arm is
comptime-pruned for a non-fp32 dtype -- and the fp32 add must carry `vadd`
and no `gate_residual`. If selector names ever did leak into every MEF, both
negative assertions would fail loudly rather than let the positives pass
vacuously.

The counts are NUL-bracketed (`\\0name\\0`) because a bare substring is not
an instrument: the MEF embeds the kernel package's source path, and any
checkout directory carrying a selector name in its path -- a worktree named
after this very change did -- scores a false hit. A string-table constant is
NUL-terminated on both sides; a path byte never is.

Its own process, as every windows sibling is: `set_virtual_cpu_target` is
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
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page. Same string the layer norm MEF was
# built with.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

# A VAE-shaped residual close: leading axes fold to the row count, the
# trailing axis is the ones gate's length.
_SHAPE = [1, 4, 8, 64]


def _compile_add_mef(name: str, dtype: DType, tmp_path: Path) -> bytes:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    types = [
        TensorType(dtype, _SHAPE, dev),
        TensorType(dtype, _SHAPE, dev),
    ]
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(x + y)

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"{name}.mef"
    compiled.export_mef(mef)

    # The Windows host half is built and run out of band, so leave the
    # artifact where a run can pick it up rather than making it regenerate
    # one.
    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
    return mef.read_bytes()


def test_npu_add_fp16_cross_compiles_with_gate_residual_route(
    tmp_path: Path,
) -> None:
    blob = _compile_add_mef("npu_add_f16_win", DType.float16, tmp_path)

    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    assert b"__chkstk" in blob

    # The fp16 route by name, and only it: the fp16 helper instantiation
    # compiles exactly one selector constant, the bf16 twin's name is not it,
    # and the fp32 arm is pruned.
    assert blob.count(b"\x00gate_residual\x00") == 1
    assert blob.count(b"\x00gate_residual_bf16\x00") == 0
    assert blob.count(b"\x00vadd\x00") == 0

    # The entry points the device arm's body reaches: the dispatch pair and
    # the ones gate's staging buffer.
    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
        b"AsyncRT_DeviceContext_createBuffer_async",
    ):
        assert symbol in blob, symbol


def test_npu_add_fp32_cross_compiles_with_vadd_route(tmp_path: Path) -> None:
    blob = _compile_add_mef("npu_add_f32_win", DType.float32, tmp_path)

    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob
    assert b"__chkstk" in blob

    # The fp32 route by name, and only it: both half-precision selector
    # strings belong to an arm this dtype prunes.
    assert blob.count(b"\x00vadd\x00") == 1
    assert blob.count(b"\x00gate_residual\x00") == 0
    assert blob.count(b"\x00gate_residual_bf16\x00") == 0

    for symbol in (
        b"AsyncRT_DeviceContext_loadFunction",
        b"AsyncRT_DeviceContext_enqueueFunctionDirect",
    ):
        assert symbol in blob, symbol
