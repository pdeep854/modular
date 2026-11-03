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
"""Cross-compiles the symbolic-batch modulation NPU graph for Windows ARM64.

One graph, two outputs, every extent the whole-graph staging uses: the batch is
SYMBOLIC, which is the shape class every modulation and gated residual in the
FLUX.2 denoise graph stages and the one the matchers declined before the
axis-wise relaxation. `npu_symbatch_fp16_exec.c` is the host that runs what
this exports, binding the batch to 1 (the single-dispatch path) and to 2 (the
per-batch slice loop, with different parameter rows per batch).

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
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_ROWS, _LENGTH = 4, 64


def test_npu_modulation_symbolic_batch_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    dev = DeviceRef.NPU()
    types = [
        # adaLN: x, scale, shift -- parameters one row per batch.
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], dev),
        TensorType(DType.float16, ["batch", 1, _LENGTH], dev),
        TensorType(DType.float16, ["batch", 1, _LENGTH], dev),
        # Gated residual: the branch output and its per-channel gate.
        TensorType(DType.float16, ["batch", _ROWS, _LENGTH], dev),
        TensorType(DType.float16, ["batch", 1, _LENGTH], dev),
    ]
    with Graph(
        "npu_symbatch_fp16_win",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, scale, shift, y, gate = (v.tensor for v in graph.inputs)
        graph.output(
            (1 + scale) * x + shift,
            x + gate * y,
        )

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_symbatch_fp16_win.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert b'{"label":"npu","id":0}' in blob

    # The two fp16 rows by name, NUL-terminated: each is a prefix of its bf16
    # twin, so an unterminated match would count a bf16 dispatch as an fp16
    # one.
    for selector in (b"ada_modulate\x00", b"gate_residual\x00"):
        assert selector in blob, selector

    # Paired with the ops' own signatures. `dyn` is the symbolic batch, still
    # symbolic in the artifact -- the compile did not pin it, so the host
    # binds it at execute time.
    for signature in (
        b"mo.composite.ada_modulate : (!mo.tensor<[dyn, 4, 64], f16, npu:0>",
        b"mo.composite.gated_residual : (!mo.tensor<[dyn, 4, 64], f16, npu:0>",
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
