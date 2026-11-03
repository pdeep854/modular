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
"""The graph-grain upsample composite: fires, declines, and stays honest.

`_maybe_fold_npu_upsample_conv` (vae_flux2) elides the
`interpolate_2d_nearest` materialization ops -- the reshape/broadcast-ones/mul
chain that executes as host walks on the on-device NPU -- and rebinds the
following conv to the pre-upsample map as the `npu_upsample_conv` custom op
(signed selector 45, `upsample_log2 = 1`). This file pins the whole contract
device-free:

- FIRES: with `MODULAR_NPU_UPSAMPLE_FOLD=1` on an NPU fp16 graph, the staged
  ops lose exactly the idiom (2 reshapes, the mul, the broadcast, the ones
  constant, the conv) and gain exactly the composite; the composite's input
  is the NCHW->NHWC pin of the PRE-upsample tensor and its parameters carry
  the conv's own paddings.
- DECLINES: a second consumer of the upsampled tensor anywhere, an
  array-ineligible conv, or the knob off each keep today's ops exactly --
  census-identical to an authoring that never heard of the fold.
- NUMERICS: the composite's staged binding (operands + paddings) rebuilt as
  today's ops computes bit-identically to the materialized path on the plain
  CPU executor -- the graph-grain half of the correctness claim. The kernel
  half (selector 45's gather equals the interpolate semantics on silicon) is
  the signed kernel's own certification, deliberately not re-claimed here.
- MUTATION: the eligibility harness is proven able to fail -- with the
  admission constants sabotaged, the ineligible-conv falsifier's own
  assertions flip.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from max import _core
from max.driver import (
    CPU,
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
    ops,
)
from max.graph.type import FilterLayout
from max.pipelines.architectures.autoencoders import vae_flux2
from max.pipelines.architectures.autoencoders.vae_flux2 import (
    Upsample2D,
    _maybe_fold_npu_upsample_conv,
    interpolate_2d_nearest,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

_FOLD_ENV = "MODULAR_NPU_UPSAMPLE_FOLD"

_DEF = re.compile(r"^\s+(?:%\S+(?:, %\S+)*) = ([a-z][A-Za-z0-9_.]*)")

# One staged kind per walk-executed op class of the idiom
# (`HOST_WALK_CENSUS.md`'s class-D rows): the two reshapes, the broadcast,
# and the mul are what the fold must remove; the conv is what it replaces.
_IDIOM_KINDS = ("rmo.reshape", "rmo.broadcast_to", "rmo.mul", "rmo.conv")


def _kinds(graph: Graph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in str(graph).splitlines():
        if m := _DEF.match(line):
            counts[m.group(1)] += 1
    return counts


def _find_custom(graph: Graph, symbol: str) -> _core.Operation | None:
    """The one `mo.custom` op with this symbol, in the typed `_core` world."""
    for up_op in graph._body:
        core_op = _core.Operation._from_cmlir(up_op)
        if type(core_op).__name__ == "CustomOp" and core_op.symbol == symbol:
            return core_op
    return None


def _operand_value(op: _core.Operation, index: int) -> _core.Value:
    """Operand ``index`` as a value.

    `mo.custom`'s variadic accessor hands back values directly where the
    fixed-arity generated classes hand back ``OpOperand``s.
    """
    operand = op.operands[index]
    return getattr(operand, "value", operand)


@pytest.fixture
def virtual_npu():  # noqa: ANN201
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


def _upsample_module(channels: int, device: DeviceRef) -> Upsample2D:
    return Upsample2D(
        channels=channels,
        use_conv=True,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        bias=True,
        interpolate=True,
        device=device,
        dtype=DType.float16,
    )


def _stage_module_graph(
    name: str, channels: int, device: DeviceRef, *, fold_call: bool = True
) -> Graph:
    """One Upsample2D over a [1, C, 16, 16] fp16 input.

    ``fold_call=False`` authors the identical ops without going through
    `Upsample2D.__call__`'s fold seam -- the pre-change control.
    """
    up = _upsample_module(channels, device)
    with Graph(
        name,
        input_types=[TensorType(DType.float16, [1, channels, 16, 16], device)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x = graph.inputs[0].tensor
        if fold_call:
            y = up(x)
        else:
            assert up.conv is not None
            y = interpolate_2d_nearest(x, scale_factor=2)
            y = up.conv(y)
        graph.output(y)
    return graph


def test_composite_fires_and_elides_the_walk_ops(
    virtual_npu, monkeypatch
) -> None:
    monkeypatch.setenv(_FOLD_ENV, "1")
    graph = _stage_module_graph("upsample_fold_fires", 64, DeviceRef.NPU())
    text = str(graph)
    counts = _kinds(graph)

    # The composite staged...
    assert counts["mo.custom"] == 1, text
    assert 'symbol = "npu_upsample_conv"' in text, text
    # ...and every materialization op of the idiom is gone.
    for kind in _IDIOM_KINDS:
        assert counts[kind] == 0, (kind, counts, text)

    # The small-map binding: the custom's input is the NCHW->NHWC pin of the
    # PRE-upsample tensor (the graph input), at pre-upsample extents.
    custom = _find_custom(graph, "npu_upsample_conv")
    assert custom is not None, text
    pin = _operand_value(custom, 0).owner
    assert type(pin).__name__ == "MoTransposeOp", text
    assert not isinstance(pin.operands[0].value.owner, _core.Operation), (
        "the composite must read the graph input, not a materialized value"
    )
    assert "[1, 16, 16, 64]" in str(_operand_value(custom, 0).type), text
    # The paddings ride the op; the conv's [1, 1, 1, 1] row, by name.
    for param in ("pad_top", "pad_bottom", "pad_left", "pad_right"):
        assert re.search(rf"{param} = 1 : index", text), (param, text)

    # Downstream is untouched: bias add and the NHWC->NCHW unpermute still
    # stage, now consuming the composite's full-size output. Two transposes
    # remain by design -- the pin survives at pre-upsample extents (4x
    # smaller walk) plus the unpermute; only the materialization ops die.
    assert counts["rmo.add"] == 1, counts
    assert counts["rmo.mo.transpose"] == 2, counts
    assert "[1, 32, 32, 64]" in str(custom.results[0].type), text


def test_composite_fires_with_symbolic_batch(virtual_npu, monkeypatch) -> None:  # noqa: ANN001
    """The serve world's one symbolic dim is the batch; the fold must thread
    it through rather than decline on it (the dispatch loops batch from
    runtime extents)."""
    monkeypatch.setenv(_FOLD_ENV, "1")
    dev = DeviceRef.NPU()
    up = _upsample_module(64, dev)
    with Graph(
        "upsample_symbolic_batch",
        input_types=[TensorType(DType.float16, ["batch", 64, 16, 16], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        graph.output(up(graph.inputs[0].tensor))
    counts = _kinds(graph)
    assert counts["mo.custom"] == 1, counts
    assert counts["rmo.mul"] == 0, counts
    assert counts["rmo.conv"] == 0, counts


def test_knob_off_stages_exactly_todays_ops(virtual_npu, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv(_FOLD_ENV, raising=False)
    folded_path = _stage_module_graph("upsample_same", 64, DeviceRef.NPU())
    control = _stage_module_graph(
        "upsample_same", 64, DeviceRef.NPU(), fold_call=False
    )
    # Census-identical is the floor; with the knob off the seam must author
    # byte-identical MLIR, not merely the same op multiset.
    assert str(folded_path) == str(control)


def test_falsifier_second_consumer_keeps_the_walks(
    virtual_npu, monkeypatch
) -> None:
    """The elision safety rule's falsifier: an upsampled tensor with TWO
    consumers must decline, keeping every materialization op staged."""
    monkeypatch.setenv(_FOLD_ENV, "1")
    dev = DeviceRef.NPU()
    up = _upsample_module(64, dev)
    with Graph(
        "upsample_shared",
        input_types=[TensorType(DType.float16, [1, 64, 16, 16], dev)],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        assert up.conv is not None
        x_up = interpolate_2d_nearest(graph.inputs[0].tensor, scale_factor=2)
        # The second consumer: the upsampled map escapes to the output too.
        second = ops.reshape(x_up, [1, 64 * 32 * 32])
        y = up.conv(x_up)
        folded = _maybe_fold_npu_upsample_conv(y)
        assert folded is y, "a shared upsample must decline the fold"
        graph.output(folded, second)

    counts = _kinds(graph)
    assert counts["mo.custom"] == 0, counts
    assert counts["rmo.mul"] == 1, counts
    assert counts["rmo.broadcast_to"] == 1, counts
    assert counts["rmo.conv"] == 1, counts
    # Both reshapes of the idiom survive, plus the second consumer's own.
    assert counts["rmo.reshape"] == 3, counts


def test_ineligible_conv_declines(virtual_npu, monkeypatch) -> None:  # noqa: ANN001
    """C_in % 64 != 0 fails the selector-45 admission; the fold must stage
    the fallback even with the knob on."""
    monkeypatch.setenv(_FOLD_ENV, "1")
    graph = _stage_module_graph("upsample_narrow", 48, DeviceRef.NPU())
    counts = _kinds(graph)
    assert counts["mo.custom"] == 0, counts
    for kind in _IDIOM_KINDS:
        assert counts[kind] >= 1, (kind, counts)


def test_mutation_eligibility_harness_can_fail(
    virtual_npu, monkeypatch
) -> None:
    """Proves `test_ineligible_conv_declines` has teeth: with the admission
    constants sabotaged the same graph folds, flipping that test's own
    assertions. A guard a mutation cannot flip is not a guard."""
    monkeypatch.setenv(_FOLD_ENV, "1")
    monkeypatch.setattr(vae_flux2, "_NPU_MM_W", 1)
    monkeypatch.setattr(vae_flux2, "_NPU_HMX_TILE", 1)
    graph = _stage_module_graph("upsample_narrow_mutant", 48, DeviceRef.NPU())
    counts = _kinds(graph)
    assert counts["mo.custom"] == 1, counts
    assert counts["rmo.mul"] == 0, counts


def _author_materialized(
    name: str,
    device: DeviceRef,
    pads: tuple[int, int, int, int],
    *,
    fold: bool = False,
    dtype: DType = DType.float16,
    layout: FilterLayout = FilterLayout.FCRS,
) -> Graph:
    """Today's ops, filter and bias as graph inputs so the graph executes.

    The NPU staging keeps FCRS (what `Conv2d(permute=True)` authors and the
    recognizer matches); the CPU twins author RSCF because the builtin CPU
    conv does not elaborate the FCRS arrangement.
    """
    filter_shape = (
        [64, 64, 3, 3] if layout is FilterLayout.FCRS else [3, 3, 64, 64]
    )
    with Graph(
        name,
        input_types=[
            TensorType(dtype, [1, 64, 16, 16], device),
            TensorType(dtype, filter_shape, device),
            TensorType(dtype, [64], device),
        ],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        x, filt, bias = (v.tensor for v in graph.inputs)
        y = interpolate_2d_nearest(x, scale_factor=2)
        y = ops.permute(y, [0, 2, 3, 1])
        y = ops.conv2d(
            y,
            filt,
            stride=(1, 1),
            dilation=(1, 1),
            padding=pads,
            groups=1,
            bias=bias,
            filter_layout=layout,
        )
        y = ops.permute(y, [0, 3, 1, 2])
        if fold:
            y = _maybe_fold_npu_upsample_conv(y)
        graph.output(y)
    return graph


def _run_cpu(graph: Graph, feeds: list[np.ndarray]) -> np.ndarray:
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    outputs = model.execute(
        *(
            Buffer.from_numpy(np.ascontiguousarray(a)).to(
                model.input_devices[0]
            )
            for a in feeds
        )
    )
    result = outputs[0]
    assert isinstance(result, Buffer)
    return result.to_numpy()


def test_numerics_composite_binding_bit_exact_on_cpu(monkeypatch) -> None:  # noqa: ANN001
    """The staged composite's binding, decomposed back into today's ops, is
    bit-identical to the materialized path on the CPU executor.

    What this pins is the graph half: the fold bound the right tensors and
    the right geometry (a wrong pad, filter or spatial extent breaks the
    bit-compare). The kernel half of the claim -- selector 45's in-gather
    equals the interpolate semantics -- is certified on silicon by the
    signed kernel's own ceremony and is deliberately not re-claimed by a
    host test (`npu-device-measurement`'s rule: host proof is not device
    proof).

    The virtual NPU is scoped by hand: it must be gone again before the CPU
    executions below, or the driver serves mocks.
    """
    # Stage the composite on the virtual NPU and read its binding back.
    monkeypatch.setenv(_FOLD_ENV, "1")
    authored_pads = (1, 1, 1, 1)
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        npu_graph = _author_materialized(
            "upsample_bind", DeviceRef.NPU(), authored_pads, fold=True
        )
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")
    text = str(npu_graph)
    assert 'symbol = "npu_upsample_conv"' in text, text

    custom = _find_custom(npu_graph, "npu_upsample_conv")
    assert custom is not None, text
    asm = custom.asm(skip_regions=True)
    staged_pads = tuple(
        int(m.group(1))
        for name in ("pad_top", "pad_bottom", "pad_left", "pad_right")
        if (m := re.search(rf"{name} = (\d+) : index", asm))
    )
    assert len(staged_pads) == 4, asm
    # The composite reads the pin of graph input 0 and the filter input.
    pin = _operand_value(custom, 0).owner
    assert type(pin).__name__ == "MoTransposeOp", asm
    assert not isinstance(pin.operands[0].value.owner, _core.Operation), asm

    # Execute both decompositions on the plain CPU with identical bytes.
    # float32 because this host's CPU executor refuses f16; the binding
    # claim (right tensors, right geometry) is dtype-independent, and the
    # values are fp16-rounded first so both graphs read the same lattice.
    rng = np.random.default_rng(0x45)
    x_arr = rng.standard_normal((1, 64, 16, 16)).astype(np.float16)
    filt_fcrs = rng.standard_normal((64, 64, 3, 3)).astype(np.float16)
    bias_arr = rng.standard_normal(64).astype(np.float16)
    feeds = [
        x_arr.astype(np.float32),
        np.ascontiguousarray(filt_fcrs.transpose(2, 3, 1, 0)).astype(
            np.float32
        ),
        bias_arr.astype(np.float32),
    ]
    cpu = DeviceRef.CPU()
    out_materialized = _run_cpu(
        _author_materialized(
            "upsample_mat",
            cpu,
            authored_pads,
            dtype=DType.float32,
            layout=FilterLayout.RSCF,
        ),
        feeds,
    )
    out_rebuilt = _run_cpu(
        _author_materialized(
            "upsample_twin",
            cpu,
            staged_pads,
            dtype=DType.float32,
            layout=FilterLayout.RSCF,
        ),
        feeds,
    )
    assert np.array_equal(
        out_materialized.view(np.uint32), out_rebuilt.view(np.uint32)
    ), "the composite's staged binding does not describe the materialized path"
