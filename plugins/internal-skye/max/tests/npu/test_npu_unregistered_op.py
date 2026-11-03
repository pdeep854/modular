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
"""Pins the compile-time refusal of ops with no on-device NPU registration.

A graph may only name kernels already carried by the signed device library, so
an op outside `hexagon_ondevice/registrations.mojo` that reaches the device has
to be refused somewhere. These tests pin *where*, because one of the two paths
gets there by accident and a plausible-looking cleanup would disarm it.

"Reaches the device" is doing real work in that sentence, and this file used to
be written as though it were free. `test_npu_op_disposition` carries the full
ladder; the two rungs these tests do *not* cover compile clean and compute on
the host.

The ceiling is the static manifest, not an inability to produce device code:
compiling a kernel inside a live host process, cataloguing and signing it
unprivileged, and loading it through FastRPC's own privileged loader all work.
What a generated kernel lacks is a `NPUKernels.def` row for `loadFunction` to
resolve its name against, and a library filename reserved before the session's
first open attempt. Saying "no JIT" overstates it, and conflates the absent
in-memory `dlopenbuf` loader with code generation as such.

An elementwise op does not reach the `unsupported_target_error` at the bottom
of `elementwise`. The on-device NPU target declares `stdlib_plugin = "qaic"`
and `accelerator_arch = "npu"`, so `elementwise` takes its plugin branch and
hands the op to `QAICBackend` -- which both Hexagon parts share, and only one
of which can be served by generating a kernel. The hook therefore has an arm
each, and the on-device arm refuses: a body minted at graph-compile time has no
row in the signed library's manifest for `loadFunction` to resolve, so
producing the code would not make it loadable.

That arm is what this test pins, and it replaced an accident. The refusal used
to come from `_is_npu_target` one frame further in, which is exact equality
against the card's `hexagon-unknown-linux-musl` triple while this target is
`hexagon-unknown-unknown-elf` -- a triple mismatch standing in for a decision,
reporting "is NPU-only" while refusing an NPU. Teaching `_is_npu_target` the
bare-metal triple looked like a fix and would have routed on-device NPU graphs
into a JIT that cannot serve them. It no longer can, because the arm decides
before that predicate is consulted -- but the predicate is still the card's
alone and still must not be widened. It gates all five of the card's
`compile_function` overloads, so widening it hands generated device code to
every other stdlib site that asks for a compiled kernel, and flips
`enqueue_function`'s compile-and-launch overload onto the card's FAM-slot
packing. The dispatch stubs in `registrations.mojo` are unaffected either way:
they reach the device through `load_function`, whose
`DeviceExternalFunction` launch consults no such predicate.

An op whose built-in is *not* written over `elementwise` takes the honest path:
it resolves to an unscoped built-in, codegen is asked for the bare-metal Hexagon
triple, and the backend refuses by naming it. That split is by implementation
rather than by op category -- `mo.gather` is not elementwise and still lands in
the hook above.

Neither path is exhaustive, and this file does not claim it is. `mo.cumsum` was
the standing counter-example -- it compiled clean with no registration and no
refusal -- and the reason is now measured: there is no `mo.cumsum` on an NPU
device to refuse. `ops.cumsum` reads its input's device while staging and, for
anything but the CPU, rewrites the graph into
`input -> mo.transfer -> mo.cumsum(CPU) -> mo.transfer -> output`
(`ops/cumsum.py:67-83`). Kernel resolution never sees the op, so no refusal has
anything to fire on. The transfer happens under every placement policy; the
policy only chooses between a `UserWarning` and a `ValueError`, and it defaults
to the warning. `test_npu_op_disposition` pins both halves of that.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from max.driver import (
    CPU,
    NPU,
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

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# The shape the `matmul` selector was measured bit-exact against on silicon.
_M, _N, _K = 4, 8, 64


@pytest.fixture
def virtual_npu() -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none.

    The settings are process-wide, so they are cleared on the way out;
    leaving them set would retarget every later test in this process.
    """
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")


def _compile_error(exc: BaseException) -> str:
    """Joins a compile failure with its causes.

    `InferenceSession.compile` reports a generic "Failed to compile the model"
    and chains the diagnostic underneath, so the message worth asserting on is
    never the top one.
    """
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def _div_graph(name: str, *, device: DeviceRef, dtype: DType) -> Graph:
    """Builds a lone `mo.div`, the cheapest op with no NPU registration.

    `mo.add` held this role until `registrations.mojo` claimed the bare
    binary arithmetic as host walks; `mo.div` is the neighbouring lambda-class
    op the walks deliberately left unclaimed, so it still names the refusal
    path this test pins.
    """
    types = [TensorType(dtype, [_M, _K], device)] * 2
    with Graph(
        name,
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        a, b = (v.tensor for v in g.inputs)
        g.output(ops.div(a, b))
    return g


def test_unregistered_elementwise_op_is_refused(virtual_npu: None) -> None:
    """A lone `mo.div` fails to compile, in the on-device NPU's own arm.

    Two assertions, and the second is the one that discriminates. Naming the
    plugin's `elementwise.mojo` proves the op really was routed into the shared
    hook; naming the manifest proves the *on-device* arm is what stopped it,
    rather than the card's `compile_function` constraint one frame further in.
    A run that reported only the first would be satisfied by either.

    If this test starts passing compilation, an on-device NPU graph is being
    handed to a JIT that does not exist on this part.
    """
    session = InferenceSession(devices=[NPU()])
    with pytest.raises(Exception) as excinfo:
        session.compile(
            _div_graph("npu_div", device=DeviceRef.NPU(), dtype=DType.float16)
        )
    diagnostic = _compile_error(excinfo.value)
    assert "_plugin/qaic/elementwise.mojo" in diagnostic
    assert "elementwise on the on-device NPU" in diagnostic
    assert "loads kernels by name from a static manifest" in diagnostic
    assert (
        "compile_function[parametric indexed-closure] is NPU-only"
        not in diagnostic
    ), (
        "the card's constraint fired, so the on-device arm did not decide "
        "first -- widening `_is_npu_target` would route this graph into the "
        "card's JIT"
    )


def test_unregistered_non_elementwise_op_is_refused(virtual_npu: None) -> None:
    """A `mo.top_k` is refused by codegen, which names the triple.

    It resolves to an unscoped built-in whose implementation is not written over
    `algorithm.functional.elementwise`, so it asks for device code directly and
    the backend refuses by naming the target it was asked for.

    A rank-3 matmul was the example here until `mo.batch_matmul` gained a
    registration, and `mo.conv` after it until the host-im2col registration
    claimed conv for the VAE decoder -- any op named here has to be one no
    registration claims, so this needs revisiting whenever the set grows.
    `mo.gather` does *not* take this path -- its built-in is written over
    `elementwise` (and it now has a host walk of its own) -- which means the
    split is not "elementwise op versus not" but "built-in implemented over
    `elementwise` or not", and the honest-codegen path is the narrower of the
    two.
    """
    types = [TensorType(DType.float32, [4, 64], DeviceRef.NPU())]
    with Graph(
        "npu_top_k",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        values, indices = ops.top_k(graph.inputs[0].tensor, 4)
        graph.output(values, indices)

    session = InferenceSession(devices=[NPU()])
    with pytest.raises(Exception) as excinfo:
        session.compile(graph)
    assert (
        "target 'hexagon-unknown-unknown-elf' is not supported by this build"
        in _compile_error(excinfo.value)
    )


def test_a_div_beside_a_matmul_becomes_an_epilogue(virtual_npu: None) -> None:
    """An unclaimed elementwise op next to a matmul is an epilogue, not a gap.

    MOGG folds it into the matmul's epilogue, so what fails is the registered
    stub refusing `has_epilogue_fusion` -- a pre-signed kernel cannot carry a
    generated epilogue. An `mo.add` carried this pin until the ungated rules
    and the host walks began serving adds before fusion could take them --
    deliberately, that is how the encoder's residuals compile -- so the op
    that still demonstrates the fold is `mo.div`, which nothing claims.
    """
    types = [
        TensorType(DType.float16, [_M, _K], DeviceRef.NPU()),
        TensorType(DType.float16, [_N, _K], DeviceRef.NPU()),
        TensorType(DType.float16, [_M, _N], DeviceRef.NPU()),
    ]
    with Graph(
        "npu_matmul_div",
        input_types=types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as graph:
        a, b, other = (v.tensor for v in graph.inputs)
        graph.output(ops.div(a @ ops.transpose(b, -1, -2), other))

    session = InferenceSession(devices=[NPU()])
    with pytest.raises(Exception) as excinfo:
        session.compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "an epilogue cannot be fused into a pre-signed kernel" in diagnostic


def test_the_same_div_compiles_on_cpu(virtual_npu: None) -> None:
    """The refusals above are the NPU's, not something `mo.div` always hits.

    fp32 rather than fp16 because fp16 is not a CPU dtype at all, which would
    make this pass for the wrong reason.
    """
    session = InferenceSession(devices=[CPU()])
    session.compile(
        _div_graph("cpu_div", device=DeviceRef.CPU(), dtype=DType.float32)
    )
