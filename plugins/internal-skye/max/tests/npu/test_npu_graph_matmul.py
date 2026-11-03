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
"""Compiles a graph whose single `mo.matmul` lowers onto the on-device NPU.

A registration in `max/kernels/src/hexagon_ondevice` is a host dispatch stub
for a kernel that is already signed into the device library, so a compile is
all a host with no NPU can check: virtual-device mode enumerates the device,
and `InferenceSession.init_all` hands back a mock there, so these tests compile
and export a MEF rather than execute one.

Which kernel the compiler chose is the thing worth proving, because a built-in
matmul would also compile and would say nothing about the NPU. The stub's own
refusals are the discriminator: `transpose_b` and fp16 are `comptime assert`s
inside it, and the built-in accepts both, so a graph that fails to compile with
the stub's message can only have been lowered onto the stub.
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
# The selector reads K in 64-element vectors; a ragged K is zero-padded to
# the next multiple of 64 at dispatch, host-side.
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


def _matmul_graph(
    name: str,
    *,
    device: DeviceRef,
    dtype: DType,
    transpose_b: bool,
    m: int | str = _M,
    n: int = _N,
    k: int = _K,
) -> Graph:
    """Builds `A @ B` as one `mo.matmul`, with or without `transpose_b`.

    `MatmulTranspBPattern` folds a `mo.transpose` of the B operand into the
    matmul's `transpose_b` attribute, which is the only way to reach the
    attribute from the graph API. B is a graph input rather than a constant so
    the fold cannot be pre-empted by constant folding.
    """
    b_shape = [n, k] if transpose_b else [k, n]
    input_types = [
        TensorType(dtype, [m, k], device),
        TensorType(dtype, b_shape, device),
    ]
    with Graph(
        name,
        input_types=input_types,
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        a, b = (v.tensor for v in g.inputs)
        g.output(a @ (ops.transpose(b, -1, -2) if transpose_b else b))
    return g


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


def test_npu_matmul_compiles(virtual_npu: None, tmp_path: Path) -> None:
    """An `mo.matmul` on an NPU device compiles and serializes."""
    graph = _matmul_graph(
        "npu_matmul",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
    )
    session = InferenceSession(devices=[NPU()])
    compiled = session.compile(graph)

    mef = tmp_path / "npu_matmul.mef"
    compiled.export_mef(mef)
    # The serialized artifact carries the registration package's own name,
    # which is the positive half of the selection proof below.
    assert b"hexagon_ondevice" in mef.read_bytes()


def test_npu_matmul_takes_a_ragged_k(virtual_npu: None, tmp_path: Path) -> None:
    """K = 32: the VAE's `post_quant_conv` after `ConvToMatmulPattern`.

    The pattern rewrites the 1x1 conv to `[M, 32] @ [32, 32]^T` with a
    symbolic M, so the stub must take a K that is not a multiple of 64; it
    zero-pads K to 64 at dispatch. The pad itself is execute-side and proven
    host-side in `kernels/test/hexagon_ondevice/test_matmul_k_pad.mojo`;
    what a compile can pin is that the shape stages onto the NPU
    registration rather than refusing or falling to a built-in.
    """
    graph = _matmul_graph(
        "npu_matmul_k32",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=True,
        m="m",
        n=32,
        k=32,
    )
    session = InferenceSession(devices=[NPU()])
    compiled = session.compile(graph)

    mef = tmp_path / "npu_matmul_k32.mef"
    compiled.export_mef(mef)
    assert b"hexagon_ondevice" in mef.read_bytes()


def test_npu_matmul_selects_the_npu_registration(virtual_npu: None) -> None:
    """The stub is the kernel chosen, not a built-in that also accepts fp16.

    Its `transpose_b` assert fires only if the compiler instantiated it, and
    the message is unique to `hexagon_ondevice/registrations.mojo`.
    """
    graph = _matmul_graph(
        "npu_matmul_no_transpose",
        device=DeviceRef.NPU(),
        dtype=DType.float16,
        transpose_b=False,
    )
    session = InferenceSession(devices=[NPU()])
    with pytest.raises(Exception) as excinfo:
        session.compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "transpose_b must be set" in diagnostic


def test_npu_matmul_refuses_fp32(virtual_npu: None) -> None:
    """A second refusal from the same stub, on a different constraint."""
    graph = _matmul_graph(
        "npu_matmul_fp32",
        device=DeviceRef.NPU(),
        dtype=DType.float32,
        transpose_b=True,
    )
    session = InferenceSession(devices=[NPU()])
    with pytest.raises(Exception) as excinfo:
        session.compile(graph)
    diagnostic = _compile_error(excinfo.value)
    assert "hexagon_ondevice/registrations.mojo" in diagnostic
    assert "fp16 only" in diagnostic


def test_cpu_matmul_accepts_what_the_npu_refuses(virtual_npu: None) -> None:
    """The refusals above are the NPU's, not the graph's.

    Same shapes, same kernel package, a CPU device: both configurations the
    NPU stub rejects compile here, so the rejection is kernel selection rather
    than something the graph API would have refused anyway.
    """
    session = InferenceSession(devices=[CPU()])
    for name, dtype, transpose_b in (
        ("cpu_matmul_no_transpose", DType.float32, False),
        ("cpu_matmul_fp32", DType.float32, True),
    ):
        graph = _matmul_graph(
            name,
            device=DeviceRef.CPU(),
            dtype=dtype,
            transpose_b=transpose_b,
        )
        session.compile(graph)
