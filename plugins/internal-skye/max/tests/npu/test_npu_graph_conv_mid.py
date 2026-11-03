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
"""The one-conv probe's linux twin: shared-lowering leg of DRAFT-301.

Two tests, two different questions about the same conv seam
(`max.nn.Conv2d(permute=True)`, the decoder's own — see
`test_npu_graph_conv_mid_windows.py` for the device-bound windows twin):

1. **CPU numeric execute.** The whole-graph VAE MEF has never decoded
   correctly on ANY target — its first numeric exercise anywhere was the
   device run that produced the flat-gray field. An NPU-placed graph cannot
   execute off-device (virtual-device mode hands back a mock,
   `test_npu_graph_matmul.py`), but the SHARED part of the lowering — the
   permute/conv/bias construction and the target-independent compiler passes
   — executes fine on CPU. Seeded fp16 operands, fp32 im2col reference in
   this file, the DRAFT-301 band (`rel_rms <= 2e-3`, `max|d| <= 1.0`). A
   failure here localizes the flat-gray defect in shared lowering with zero
   device time; a pass moves the whole suspicion to the NPU-specific half.

2. **Virtual-NPU compile + linux MEF export.** The same graph the windows
   twin compiles, minted for linux so both targets' artifacts exist
   side-by-side; asserts the weights ride external under checkpoint FQNs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from max.driver import (
    CPU,
    NPU,
    Buffer,
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

_LAYER = "decoder.mid_block.resnets.0.conv1"
_CHANNELS = 512
_SPATIAL = 64


def _conv_graph(
    name: str, dev: DeviceRef, dtype: DType = DType.float16
) -> tuple[Graph, Conv2d]:
    conv = Conv2d(
        kernel_size=3,
        in_channels=_CHANNELS,
        out_channels=_CHANNELS,
        dtype=dtype,
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
        name,
        input_types=[
            TensorType(dtype, [1, _CHANNELS, _SPATIAL, _SPATIAL], dev),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(conv(graph.inputs[0].tensor))
    return graph, conv


def _reference_conv(x, w, b):
    """fp32 accumulate over fp16 operands: pad-1 3x3 as nine [M,C]@[C,O]."""
    xf = x.astype(np.float32)[0].transpose(1, 2, 0)  # NHWC [H,W,C]
    wf = w.astype(np.float32)  # FCRS [O,C,3,3]
    h, wd, c = xf.shape
    xp = np.zeros((h + 2, wd + 2, c), dtype=np.float32)
    xp[1 : 1 + h, 1 : 1 + wd] = xf
    out = np.zeros((h * wd, wf.shape[0]), dtype=np.float32)
    for rr in range(3):
        for ss in range(3):
            out += xp[rr : rr + h, ss : ss + wd].reshape(-1, c) @ wf[
                :, :, rr, ss
            ].T
    out += b.astype(np.float32)
    return out.reshape(h, wd, -1).transpose(2, 0, 1)[None]  # NCHW


def test_conv_mid_cpu_numeric_matches_fp32_reference() -> None:
    """The shared-lowering numeric leg: CPU execute vs the fp32 reference."""
    # fp16 compute is not supported on cpu:0 (the compile refuses it), so
    # the CPU leg runs float32 over fp16-SEEDED values: it discharges the
    # construction/permute/lowering axis; the fp16 dtype axis is device-only.
    dev = DeviceRef.CPU()
    graph, _ = _conv_graph("npu_conv_mid_f32_cpu", dev, DType.float32)

    rng = np.random.default_rng(301)
    x16 = rng.standard_normal(
        (1, _CHANNELS, _SPATIAL, _SPATIAL), dtype=np.float32
    ).astype(np.float16)
    w16 = (
        rng.standard_normal((_CHANNELS, _CHANNELS, 3, 3), dtype=np.float32)
        * 0.02
    ).astype(np.float16)
    b16 = (rng.standard_normal(_CHANNELS, dtype=np.float32) * 0.1).astype(
        np.float16
    )

    session = InferenceSession(devices=[CPU()])
    runtime = session.init(
        session.compile(graph),
        weights_registry={
            _LAYER + ".weight": w16.astype(np.float32),
            _LAYER + ".bias": b16.astype(np.float32),
        },
    )
    got = runtime.execute(Buffer.from_numpy(x16.astype(np.float32)))[0].to_numpy()
    assert got.shape == (1, _CHANNELS, _SPATIAL, _SPATIAL)

    ref = _reference_conv(x16, w16, b16)
    d = got.astype(np.float32) - ref
    rel_rms = float(np.sqrt((d**2).mean()) / np.sqrt((ref**2).mean()))
    max_abs = float(np.abs(d).max())
    corr = float(
        np.corrcoef(got.astype(np.float32).ravel(), ref.ravel())[0, 1]
    )
    print(
        f"cpu-numeric: rel_rms={rel_rms:.2e} max_abs={max_abs:.3f}"
        f" corr={corr:.6f}"
    )
    # The DRAFT-301 band. A failure here is the shared-lowering conviction.
    assert rel_rms <= 2e-3, (rel_rms, max_abs, corr)
    assert max_abs <= 1.0, (rel_rms, max_abs, corr)


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none."""
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")


def test_npu_conv_mid_fp16_compile_linux(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The linux MEF artifact: same graph, NPU placement, no retarget."""
    import os
    import shutil

    graph, _ = _conv_graph("npu_conv_mid_fp16_linux", DeviceRef.NPU())
    graph_str = str(graph)
    assert "mo.constant.external" in graph_str
    assert _LAYER + ".weight" in graph_str

    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / "npu_conv_mid_fp16_linux.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"hexagon_ondevice" in blob
    assert (_LAYER + ".weight").encode() in blob
    assert (_LAYER + ".bias").encode() in blob

    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        shutil.copy(mef, Path(outputs) / mef.name)
