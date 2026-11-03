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
"""Compiles the two preamble table composites onto the on-device NPU.

`mo.composite.sinusoid_embed` and `mo.composite.rope_table` are the two
signed kernels the whole-graph census found missing: the timestep sinusoidal
embedding and the per-axis rope (cos, sin) table. No PatternFusion rule mints
them yet, so these tests stage the composites directly through `ops.custom`,
which is also what pins the operand contract a future rule must emit.

Which selector the stub named is what these prove, the way the rope tests
prove theirs: a MEF carrying the selector's NUL-terminated name is evidence
`loadFunction` will resolve the signed row.

`ops.custom` verifies the op against the kernel library as it is added, when
the graph knows only the device's label -- and these registrations are
matchable only by device API, which exists once `InferenceSession.compile`
holds the real (here virtual) device. An op the model author writes has
built-in CPU/GPU rows that satisfy the add-time check; a composite scoped to
one accelerator API has none. So the add-time check is disabled per test and
the compile does the resolving, which is also where an unresolved kernel
fails loudly -- the whole-graph census is built on exactly that behavior.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from max.driver import (
    NPU,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops
from max.graph.graph import KernelLibrary

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# FLUX.2-Klein's own geometry: H=128 timestep bands, a [tokens, 4] packed id
# tensor read by column, and 16 frequencies per rope axis. The token count is
# small; the extents are runtime arguments of the selectors either way.
_ROWS = 4
_HALF_DIM = 128
_TOKENS = 64
_AXES = 4
_FREQS = 16


@pytest.fixture
def virtual_npu(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none, and
    defers custom-op verification to the compile (see the module docstring)."""
    monkeypatch.setattr(
        KernelLibrary, "verify_custom_op", lambda self, custom_op: None
    )
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")


def _sinusoid_graph(name: str, *, device: DeviceRef, flip: int) -> Graph:
    """Builds one `mo.composite.sinusoid_embed` over `[_ROWS]` timesteps.

    The scale and band coefficient ride as CPU scalar constants, the folded
    graph constants the future pattern will capture (`1000.0` and
    `-ln(10000)`).
    """
    with Graph(
        name,
        input_types=[TensorType(DType.float32, [_ROWS], device)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        t = g.inputs[0].tensor
        time_scale = ops.constant(1000.0, DType.float32, device=DeviceRef.CPU())
        neg_ln_max_period = ops.constant(
            -9.210340371976184, DType.float32, device=DeviceRef.CPU()
        )
        g.output(
            ops.custom(
                "mo.composite.sinusoid_embed",
                device=device,
                values=[t, time_scale, neg_ln_max_period],
                out_types=[
                    TensorType(DType.float16, [_ROWS, 2 * _HALF_DIM], device)
                ],
                parameters={"flip": flip},
            )[0].tensor
        )
    return g


def _rope_table_graph(name: str, *, device: DeviceRef, column: int) -> Graph:
    """Builds one `mo.composite.rope_table` from a column of the packed ids.

    The id column is sliced out of the `[tokens, 4]` si64 tensor the FLUX
    staging concatenates, which is exactly the operand shape the composite
    exists to take without a materialized copy: the selector reads the column
    through a stride.
    """
    with Graph(
        name,
        input_types=[
            TensorType(DType.int64, [_TOKENS, _AXES], device),
            TensorType(DType.float32, [_FREQS], device),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        ids, freqs = (v.tensor for v in g.inputs)
        positions = ids[:, column]
        g.output(
            ops.custom(
                "mo.composite.rope_table",
                device=device,
                values=[positions, freqs],
                out_types=[
                    TensorType(DType.float32, [_TOKENS, 2 * _FREQS], device)
                ],
            )[0].tensor
        )
    return g


def _export(graph: Graph, tmp_path: Path, name: str) -> bytes:
    compiled = InferenceSession(devices=[NPU()]).compile(graph)
    mef = tmp_path / f"{name}.mef"
    compiled.export_mef(mef)
    return mef.read_bytes()


def test_npu_sinusoid_embed_names_the_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The sinusoid composite compiles and the MEF names `sinusoid_embed`."""
    blob = _export(
        _sinusoid_graph("npu_sinusoid_embed", device=DeviceRef.NPU(), flip=1),
        tmp_path,
        "npu_sinusoid_embed",
    )
    assert b"sinusoid_embed\x00" in blob


def test_npu_sinusoid_embed_compiles_unflipped(
    virtual_npu: None, tmp_path: Path
) -> None:
    """`flip=0` (the reference's sin-first order) instantiates too."""
    blob = _export(
        _sinusoid_graph(
            "npu_sinusoid_embed_sin_first", device=DeviceRef.NPU(), flip=0
        ),
        tmp_path,
        "npu_sinusoid_embed_sin_first",
    )
    assert b"sinusoid_embed\x00" in blob


def test_npu_rope_table_names_the_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The rope-table composite, fed a sliced id column, compiles and the MEF
    names `rope_table` -- the whole point of the stride operand is that this
    slice needs no materialized copy at dispatch time."""
    blob = _export(
        _rope_table_graph("npu_rope_table", device=DeviceRef.NPU(), column=2),
        tmp_path,
        "npu_rope_table",
    )
    assert b"rope_table\x00" in blob
