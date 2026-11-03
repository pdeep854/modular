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
"""Compiles the diffusion bundle's fourth stage -- denoise_predict -- for the
on-device NPU.

The other three whole-graph tests cover the stages the certified render
already runs (encoder, denoise, VAE). This one is the stage the NPU world
does not have: the compiled Euler step ``latents + dt * noise_pred``, which
the splinter host loop performs in numpy. The serving engine's bundle
interpreter structurally requires it -- the last step-stage's output IS the
next step's latents, so with denoise_compute alone the *velocity* output
would be fed back as latents -- and its boot recipe additionally requires
one weight-free MEF, which no existing NPU stage is (encoder, denoise and
VAE all carry registries).

Staged through ``EulerStepModule`` -- the same class the upstream exporter's
``DenoisePredict`` component wraps, under the same graph name -- so the MEF
this test proves is the graph the bundle would carry, not a lookalike. The
usual staging choices apply: batch symbolic (the literal-1 refusals of
``test_npu_graph_batch_matmul.py``, kept here for uniformity with the other
stage MEFs even though this graph has no matmul), sequence dims pinned to
the run package's 512x512 geometry (seq 1024, channels 128), float16 in and
out with the module's own f32 arithmetic between.

What makes this an empirical question rather than a formality: the whole
graph is one elementwise chain (cast, broadcast-mul, add, cast), and
``test_npu_graph_cast.py`` records exactly this case as unsettled -- a cast
whose consumer is itself elementwise can fuse into one lambda that this
target cannot compile, where a cast over a materialized operand converts on
the host for free. Whether the chain fuses into an unclaimed
``mogg.call.foreach``, gets minted as a composite, or lowers as host walks
is the compiler's answer, and this test records it either way: the census
and full diagnostic on refusal, the MEF's size, sha256 and markers on
success.

``FLUX_MEF_TARGET=windows`` retargets the host side to Windows ARM64 (ELF
container), the form the device driver executes -- in its own process, as
every cross-compile here is, since ``set_virtual_cpu_target`` is global and
has no reset. Compile-only either way: no device, no execute.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest
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

# The run package's geometry (512x512 render: a 32x32 token grid), the shape
# the bundle map fixes for every stage's latents: [batch, 1024, 128].
# `FLUX_IMAGE_SEQ` re-pins the image sequence length, the same knob and
# default as the denoise sibling, so a bundler can mint every stage of a
# smaller twin bundle with one variable.
_IMAGE_SEQ = int(os.environ.get("FLUX_IMAGE_SEQ", "1024"))
_PINNED = {"seq": _IMAGE_SEQ, "channels": 128}

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_DEF = re.compile(r"^\s+(?:%\S+(?:, %\S+)*) = ([a-z][A-Za-z0-9_.]*)")


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


def _chain(exc: BaseException) -> str:
    """Joins a compile failure with its causes.

    `InferenceSession.compile` reports a generic "Failed to compile the model"
    and chains the diagnostic underneath, so the message worth reading is
    never the top one.
    """
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return "\n---\n".join(parts)


def _census(mlir: str) -> dict[str, int]:
    kinds: Counter[str] = Counter()
    for line in mlir.splitlines():
        if m := _DEF.match(line):
            kinds[m.group(1)] += 1
    return dict(kinds.most_common())


def _emit(record: dict[str, object], tag: str = "JSON") -> None:
    print(f"###{tag}###" + json.dumps(record), flush=True)
    if out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        (Path(out) / "denoise_predict_record.json").write_text(
            json.dumps(record, indent=2)
        )


def test_denoise_predict_compiles_and_exports_for_the_npu(
    virtual_npu: None, tmp_path: Path
) -> None:
    from max.pipelines.architectures.flux2.components.denoise_predict import (
        EulerStepModule,
    )

    target = os.environ.get("FLUX_MEF_TARGET", "linux")
    if target == "windows":
        set_virtual_cpu_target(_WINDOWS_ARM64_ELF)

    device = DeviceRef.NPU()
    euler = EulerStepModule(dtype=DType.float16, device=device)
    pinned = [
        TensorType(t.dtype, [_PINNED.get(str(d), d) for d in t.shape], device)
        for t in euler.input_types()
    ]

    record: dict[str, object] = {
        "stage": "denoise_predict",
        "dtype": "float16",
        "target": target,
        "image_seq_len": _IMAGE_SEQ,
        "input_types": [str(t) for t in pinned],
    }

    started = time.monotonic()
    with Graph(
        "denoise_predict",
        input_types=pinned,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(euler(*(v.tensor for v in graph.inputs)))
    record["stage_seconds"] = round(time.monotonic() - started, 1)
    record["op_census"] = _census(str(graph))
    _emit(record, tag="STAGED")

    started = time.monotonic()
    try:
        compiled = InferenceSession(devices=[NPU()]).compile(graph)
    except BaseException as exc:
        record["outcome"] = "REFUSED"
        record["compile_seconds"] = round(time.monotonic() - started, 1)
        # Untruncated for the whole-graph tests' reason: the diagnostic is the
        # ordered list of everything that refused, and a cap shortens it into
        # a smaller-looking problem.
        record["diagnostic"] = _chain(exc)
        _emit(record)
        pytest.fail(
            "denoise_predict refused on the NPU -- the bundle's fourth stage "
            f"and its weight-free boot shim have no MEF.\n{record['diagnostic']}"
        )
    record["compile_seconds"] = round(time.monotonic() - started, 1)

    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tmp_path))
    mef = out / f"denoise_predict_float16_{target}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()

    # The compile is the pin, and the MEF is its artifact. The npu placement
    # marker is asserted; the kernel-package path is recorded but NOT asserted,
    # because for an undispatched graph its absence is what a correct host
    # lowering looks like (`test_npu_graph_cast.py`), and which of the two this
    # chain gets is exactly what the record exists to say. `__chkstk` rides
    # along on the windows leg for the same recorded-not-assumed reason.
    marker_names = ['{"label":"npu","id":0}', "hexagon_ondevice"]
    if target == "windows":
        marker_names.append("__chkstk")
    markers = {marker: marker.encode() in blob for marker in marker_names}
    record["outcome"] = "COMPILED"
    record["mef_bytes"] = len(blob)
    record["mef_magic"] = blob[:4].hex()
    record["mef_sha256"] = hashlib.sha256(blob).hexdigest()
    record["mef_markers"] = markers
    _emit(record)

    assert blob, "empty MEF"
    assert markers['{"label":"npu","id":0}'], markers
