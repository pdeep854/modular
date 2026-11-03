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
"""Cross-compiles graphs whose fold surface is non-empty for Windows ARM64.

The constant-folding pass compiles its foldable subgraphs to a sub-model and
executes it in the compiling process. Before the fix this file pins, that
sub-compilation inherited `set_virtual_cpu_target`, so on a Linux or macOS
host the fold sub-model was a Windows-ABI object whose `__chkstk` could not be
materialized and the export refused with `[ConstantFold] setup for model
<name>_constant_subgraphs ... Failed to materialize symbols`. The fold now
compiles for the machine that runs it; only the exported kernels carry the
virtual target.

Two fold surfaces, the two measured instances of the wall:

- the row offsets `rope_ragged_with_position_ids` synthesizes over a STATIC
  token extent (`test_npu_graph_rope_windows.py` dodges this with a symbolic
  extent, and says why);
- a preamble-shaped table chain (`range -> mul -> exp -> cast`), the class the
  whole-denoise graph's in-graph sinusoid/rope tables fold as.

Its own process, as every windows sibling is: `set_virtual_cpu_target` is
global and retargets every later compile in the process.
"""

from __future__ import annotations

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
from max.nn.kernels import rope_ragged_with_position_ids

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# "Windows ABI, ELF container": the pinned LLVM's JITLink has no COFF_aarch64
# backend, and a plain Linux triple would drop the `__chkstk` stack probe that
# Windows requires on frames over a page.
_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_TOKENS, _HEADS, _HEAD_DIM = 1024, 24, 128


def _set_windows_target() -> None:
    set_virtual_cpu_target(_WINDOWS_ARM64_ELF)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)


def _export(graph: Graph, mef: Path) -> bytes:
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    compiled.export_mef(mef)
    return mef.read_bytes()


def _assert_windows_npu_mef(blob: bytes) -> None:
    assert b'{"label":"npu","id":0}' in blob
    # The stack probe only a Windows target emits: the fix must retarget the
    # fold sub-model alone, so the exported kernels still carry the Windows
    # ABI.
    assert b"__chkstk" in blob


def test_npu_static_extent_rope_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    _set_windows_target()

    dev = DeviceRef.NPU()
    with Graph(
        "npu_rope_static_win",
        strict_device_placement=DevicePlacementPolicy.Error,
        input_types=[
            TensorType(DType.float16, [_TOKENS, _HEADS, _HEAD_DIM], dev),
            TensorType(DType.float32, [_TOKENS, _HEAD_DIM], dev),
            TensorType(DType.uint32, [1, _TOKENS], dev),
        ],
        custom_extensions=[_KERNELS],
    ) as graph:
        x, freqs_cis, position_ids = (v.tensor for v in graph.inputs)
        # The static extent makes the synthesized row offsets
        # `stack(constant(0), cast(shape_to_tensor(x.shape)[0]))` a foldable
        # subgraph -- the fold surface this test exists to carry.
        graph.output(
            rope_ragged_with_position_ids(
                x, freqs_cis, position_ids, interleaved=True
            )
        )

    blob = _export(graph, tmp_path / "npu_rope_static_win.mef")
    _assert_windows_npu_mef(blob)
    # A real device dispatch survives here, so the on-device kernel package
    # and its selector must be aboard.
    assert b"hexagon_ondevice" in blob
    assert b"rope_interleaved\x00" in blob


def test_npu_preamble_table_chain_cross_compiles_for_windows_arm64(
    tmp_path: Path,
) -> None:
    _set_windows_target()

    dev = DeviceRef.NPU()
    with Graph(
        "npu_fold_table_win",
        strict_device_placement=DevicePlacementPolicy.Error,
        input_types=[TensorType(DType.float16, [4, 16], dev)],
        custom_extensions=[_KERNELS],
    ) as graph:
        (x,) = (v.tensor for v in graph.inputs)
        # All-constant band/frequency chain, the shape the whole-graph
        # preamble's sinusoid/rope tables fold as. `exp` keeps it out of reach
        # of canonicalization, so only the constant-folding pass can retire it.
        # The table is an output rather than an elementwise operand: a folded
        # constant needs no kernel dispatch, while a bare `x * table` fuses
        # into an unsigned elementwise region the closed NPU kernel set
        # refuses -- a registration matter, not this test's subject.
        bands = ops.range(0, 16, 1, dtype=DType.float32, device=dev)
        table = ops.cast(ops.exp(bands * -0.25), DType.float16)
        graph.output(x, table)

    blob = _export(graph, tmp_path / "npu_fold_table_win.mef")
    # No `hexagon_ondevice` assert: every device op folded away, so the MEF
    # legitimately carries no on-device kernel package -- only the host half
    # (whose stubs the `__chkstk` assert pins to the Windows ABI).
    _assert_windows_npu_mef(blob)
    # The folded table itself, as fp16 bytes: 1.0 (0x3C00) then exp(-0.25)
    # (0x3A3B). Proof the fold executed and its result -- not the chain --
    # shipped. exp(-0.25) sits ~480 fp32 ulps from the nearest fp16 rounding
    # boundary, so any correct host libm/kernel folds to these exact bytes.
    assert b"\x00<;:" in blob
