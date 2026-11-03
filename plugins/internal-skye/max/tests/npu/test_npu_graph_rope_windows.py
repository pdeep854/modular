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
"""Cross-compiles the rope NPU graphs for the Windows ARM64 host.

The sibling of `test_npu_graph_layernorm_windows.py`: the NPU's host half runs on
Windows ARM64 while the graph compiler runs on Linux, so the MEF that reaches the
device names two targets at once. `npu_rope_exec.c` is the host that runs what
this exports.

Both conventions are exported, from one process and to distinct files. They are
the A/B the device run needs: same shapes, same buffers, same launch, and only
the pairing differs -- so the pair measures what vectorizing the interleaved
selector is worth, on top of proving each dispatches.

Its own process, as the matmul's cross-compile is: `set_virtual_cpu_target` is
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
from max.nn.kernels import rope_ragged_with_position_ids

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page. Same string the matmul MEF was built
# with.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

# FLUX.2-Klein's own widths: 24 attention heads of 128, over a 1024-token image
# stream. A real width rather than a token one, because the same artifact is what
# the A/B is timed on and a sixteen-row dispatch would price only the launch.
# `npu_rope_exec.c` pins the same three numbers.
_TOKENS, _HEADS, _HEAD_DIM = 1024, 24, 128


def _export(tmp_path: Path, *, interleaved: bool) -> bytes:
    name = "npu_rope_interleaved_win" if interleaved else "npu_rope_half_win"
    dev = DeviceRef.NPU()
    # The token count is symbolic, and that is load-bearing rather than
    # generality for its own sake. `rope_ragged_with_position_ids` synthesizes
    # its row offsets as `stack(constant(0), cast(shape_to_tensor(x.shape)[0]))`,
    # which over a static extent is a foldable subgraph -- and constant folding
    # JIT-compiles that subgraph for the *current CPU target* and then runs it,
    # here on Linux, where a Windows-target object wants `__chkstk` and the JIT
    # session fails to materialize it. A symbolic extent makes the offsets a
    # runtime value, so there is no subgraph to fold and nothing to run on the
    # wrong host. The stub reads every extent from the operands at execute time,
    # so the dispatch is unaffected.
    tokens = "tokens"
    with Graph(
        name,
        input_types=[
            TensorType(DType.float16, [tokens, _HEADS, _HEAD_DIM], dev),
            TensorType(DType.float32, [tokens, _HEAD_DIM], dev),
            TensorType(DType.uint32, [1, tokens], dev),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, freqs_cis, position_ids = (v.tensor for v in graph.inputs)
        graph.output(
            rope_ragged_with_position_ids(
                x, freqs_cis, position_ids, interleaved=interleaved
            )
        )

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"{name}.mef"
    compiled.export_mef(mef)

    # The Windows host half is built and run out of band, so leave the artifact
    # where a run can pick it up rather than making it regenerate one.
    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
    return mef.read_bytes()


def test_npu_rope_cross_compiles_for_windows_arm64(tmp_path: Path) -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    interleaved = _export(tmp_path, interleaved=True)
    half = _export(tmp_path, interleaved=False)

    for blob in (interleaved, half):
        assert b"hexagon_ondevice" in blob
        assert b'{"label":"npu","id":0}' in blob
        # The entry points the stub's body reaches, plus the stack probe that is
        # only emitted for a Windows target. `createBuffer_async` is load-bearing
        # here rather than incidental: the stub allocates the expanded positions
        # through it, not only the witness.
        for symbol in (
            b"AsyncRT_DeviceContext_loadFunction",
            b"AsyncRT_DeviceContext_enqueueFunctionDirect",
            b"AsyncRT_DeviceContext_createBuffer_async",
            b"__chkstk",
        ):
            assert symbol in blob, symbol

    # Exported second, so the interleaved artifact cannot carry the half name;
    # the reverse is not assertable, a MEF carrying whatever the process
    # compiled before it.
    assert b"rope_interleaved\x00" in interleaved
    assert b"rope_half\x00" not in interleaved
    assert b"rope_half\x00" in half
