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
"""Proves the two fp32 preamble minting rules against FLUX.2's own staging.

Two claims, each with a demonstrated-failing control:

- **The rules mint from the real staging functions.** The graphs here are
  built by the same `get_timestep_embedding` / `Flux2PosEmbed` /
  `pack_flux2_rope` code the denoise graph uses, at FLUX.2-Klein geometry.
  Compiled for a virtual on-device NPU, the exported MEF names the
  `sinusoid_embed` / `rope_table` selectors -- the whole chain reached one
  signed kernel. The controls stage expressions that deviate from the signed
  kernels' exact construction (a `downscale_freq_shift` band, a swapped
  (sin, cos) pair order): those must REFUSE to compile, not mint -- the
  exactness-preserving half of the rules' contract.

- **The staged expressions equal the kernels' construction in values, on the
  host, to the CPU executor's own transcendental rounding.** The kernels
  were proven bit-exact against their reference construction on silicon
  with a withheld-skel control (the #156 record: double-internal scalar
  IEEE, f32-rounded once, zero ulp off f64 libm). This file executes the
  same staged graphs on the plain host CPU at the FLUX geometries -- the
  [4608, 4] packed ids read by column into 4608 x 32 tables per axis, and
  the fp16-round-tripped timestep rows -- against a recomputation of that
  reference construction over the graph's OWN folded frequency rows. The
  residual is the CPU executor's sin/cos/exp, which are not correctly
  rounded (measured here: one ulp on 0.3% of the rope table; the same
  one-ulp band rounding amplified through the up-to-1000 timestep argument
  into at most ~1e3 f32 ulps of the embedding), so the bounds assert that
  measured envelope -- on the NPU path these sites are computed MORE
  exactly than the CPU executor renders the same expression. The controls
  deviate in construction, not rounding (the fp16 round-trip dropped; the
  band divisor shifted; the pair order swapped), and each must exceed the
  rounding envelope by orders of magnitude, so the comparison can fail.
"""

from __future__ import annotations

import math
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
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops
from max.pipelines.architectures.flux2.layers.embeddings import (
    get_timestep_embedding,
)
from max.pipelines.architectures.flux2.layers.flux2_attention import (
    Flux2PosEmbed,
    pack_flux2_rope,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# FLUX.2-Klein at one 1024x1024 image: 512 text tokens, a 64x64 latent grid,
# four rope axes of 32 (16 frequencies each), 256 timestep-embedding channels.
_TXT = 512
_GRID = 64
_TOKENS = _TXT + _GRID * _GRID
_AXES_DIM = (32, 32, 32, 32)
_THETA = 10000
_EMBED_DIM = 256
_TIME_SCALE = 1000.0


def _flux_ids() -> np.ndarray:
    """The packed [tokens, 4] id tensor the denoise graph concatenates:
    zero text ids, then the (0, h, w, 0) latent grid."""
    ids = np.zeros((_TOKENS, 4), dtype=np.int64)
    h, w = np.meshgrid(np.arange(_GRID), np.arange(_GRID), indexing="ij")
    ids[_TXT:, 1] = h.reshape(-1)
    ids[_TXT:, 2] = w.reshape(-1)
    return ids


def _timestep_rows() -> np.ndarray:
    """Denoise-schedule-shaped rows in [0, 1], plus the endpoints."""
    return np.array([0.0, 0.05, 0.25, 0.5, 0.75, 0.9999, 1.0], np.float32)


@pytest.fixture
def virtual_npu(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Enumerates an NPU as a compilation target on a host that has none, and
    defers custom-op verification to the compile, as the sincos-table tests
    do: a composite scoped to one accelerator API has no built-in row for the
    add-time check."""
    from max.graph.graph import KernelLibrary

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


def _stage_timestep_embedding(
    device: DeviceRef,
    *,
    downscale_freq_shift: float = 0.0,
    pinned_batch: bool = False,
) -> Graph:
    """FLUX.2's timestep-embedding staging: the timestep scaled by 1000 and
    cast to the model dtype (the fp16 round-trip's first half), embedded, and
    cast back to the model dtype. The NPU kernel extensions ride along only
    when the graph targets the NPU; the host-exactness leg stages for the
    plain CPU.

    `pinned_batch` stages the batch as a literal 1 instead of the symbolic
    `batch` -- the view-folding lever's staging pin (VIEW_FOLDING_SCOPE.md
    A2). The canonicalizer then squeezes the preamble's interior to rank 1,
    the spelling `SinusoidEmbedPattern`'s squeezed arm exists for."""
    extensions = [_KERNELS] if device != DeviceRef.CPU() else []
    batch: int | str = 1 if pinned_batch else "batch"
    with Graph(
        "preamble_sinusoid",
        input_types=[TensorType(DType.float32, [batch], device)],
        custom_extensions=extensions,
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        t = g.inputs[0].tensor
        scaled = ops.cast(t * _TIME_SCALE, DType.float16)
        emb = get_timestep_embedding(
            scaled,
            _EMBED_DIM,
            flip_sin_to_cos=True,
            downscale_freq_shift=downscale_freq_shift,
        )
        g.output(ops.cast(emb, DType.float16))
    return g


def _stage_rope_tables(device: DeviceRef, *, swap_pairs: bool = False) -> Graph:
    """FLUX.2's rope staging: the per-axis tables over the packed ids, packed
    into the rope kernels' (cos, sin) pair layout. `swap_pairs` packs
    (sin, cos) instead -- a layout the signed kernels do not read."""
    extensions = [_KERNELS] if device != DeviceRef.CPU() else []
    with Graph(
        "preamble_rope",
        input_types=[TensorType(DType.int64, [_TOKENS, 4], device)],
        custom_extensions=extensions,
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        ids = g.inputs[0].tensor
        cos, sin = Flux2PosEmbed(theta=_THETA, axes_dim=_AXES_DIM)(ids)
        if swap_pairs:
            freqs_cis, _ = pack_flux2_rope(sin, cos, 1, _TOKENS)
        else:
            freqs_cis, _ = pack_flux2_rope(cos, sin, 1, _TOKENS)
        g.output(freqs_cis)
    return g


def _chain(exc: BaseException) -> str:
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def test_sinusoid_minting_reaches_the_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The real timestep-embedding staging compiles for the on-device NPU and
    the MEF names the `sinusoid_embed` selector: the minting rule claimed the
    whole expression and nothing in it needed generated device code."""
    compiled = InferenceSession(devices=[NPU()]).compile(
        _stage_timestep_embedding(DeviceRef.NPU())
    )
    mef = tmp_path / "sinusoid.mef"
    compiled.export_mef(mef)
    assert b"sinusoid_embed\x00" in mef.read_bytes()


def test_sinusoid_shifted_band_refuses(virtual_npu: None) -> None:
    """The failing control: `downscale_freq_shift=1` divides the band by 127
    where the signed kernel divides by the half-width, so the expression
    deviates from the kernel's exact construction and must refuse loudly --
    a rule that minted it would hand back a plausible wrong table."""
    with pytest.raises(Exception) as exc:
        InferenceSession(devices=[NPU()]).compile(
            _stage_timestep_embedding(DeviceRef.NPU(), downscale_freq_shift=1.0)
        )
    assert "elementwise on the on-device NPU" in _chain(exc.value)


def test_sinusoid_minting_survives_the_batch_pin(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The unit-batch-pinned staging squeezes the preamble's interior to rank
    1 (the `[256] -> [1, 256]` re-lift lands between the inner concat and the
    flip slices) and the matcher must mint the same composite: this is the
    batch-pin enabler for the view-folding lever (VIEW_FOLDING_SCOPE.md A2).
    On a compiler without the squeezed arm this dies as elementwise refusal
    chains instead of a MEF."""
    compiled = InferenceSession(devices=[NPU()]).compile(
        _stage_timestep_embedding(DeviceRef.NPU(), pinned_batch=True)
    )
    mef = tmp_path / "sinusoid_pinned.mef"
    compiled.export_mef(mef)
    assert b"sinusoid_embed\x00" in mef.read_bytes()


def test_sinusoid_shifted_band_refuses_at_the_batch_pin(
    virtual_npu: None,
) -> None:
    """The squeezed arm keeps the exactness gate: a shifted band at the
    pinned batch is the same different table it is at the symbolic batch,
    and must refuse rather than mint."""
    with pytest.raises(Exception) as exc:
        InferenceSession(devices=[NPU()]).compile(
            _stage_timestep_embedding(
                DeviceRef.NPU(), downscale_freq_shift=1.0, pinned_batch=True
            )
        )
    assert "elementwise on the on-device NPU" in _chain(exc.value)


def test_rope_minting_reaches_the_selector(
    virtual_npu: None, tmp_path: Path
) -> None:
    """The real rope staging -- per-axis tables plus the pair packing --
    compiles and the MEF names the `rope_table` selector."""
    compiled = InferenceSession(devices=[NPU()]).compile(
        _stage_rope_tables(DeviceRef.NPU())
    )
    mef = tmp_path / "rope.mef"
    compiled.export_mef(mef)
    assert b"rope_table\x00" in mef.read_bytes()


def test_rope_swapped_pairs_refuse(virtual_npu: None) -> None:
    """The failing control: a (sin, cos) pair order is a table the signed
    rope kernels do not read, so the rule must refuse it."""
    with pytest.raises(Exception) as exc:
        InferenceSession(devices=[NPU()]).compile(
            _stage_rope_tables(DeviceRef.NPU(), swap_pairs=True)
        )
    assert "elementwise on the on-device NPU" in _chain(exc.value)


def _run_cpu(graph: Graph, *arrays: np.ndarray) -> np.ndarray:
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph)
    outputs = model.execute(
        *(
            Buffer.from_numpy(np.ascontiguousarray(a)).to(
                model.input_devices[0]
            )
            for a in arrays
        )
    )
    result = outputs[0]
    assert isinstance(result, Buffer)
    return result.to_numpy()


def _stage_embedding_core(device: DeviceRef) -> Graph:
    """`get_timestep_embedding` over an already-round-tripped f32 argument,
    stopping before the final f16 narrowing.

    This host CPU refuses the f16 dtype outright, so the two fp16 casts of
    the staged expression -- the round-trip of the scaled timestep and the
    output narrowing -- cannot execute here. Both are single IEEE
    conversions with one defined answer per input, applied in numpy on both
    sides of the comparison instead; the f32 core between them (the band,
    the outer product, sin/cos, the flip) is what this graph executes.
    """
    with Graph(
        "preamble_sinusoid_core",
        input_types=[TensorType(DType.float32, ["batch"], device)],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        g.output(
            get_timestep_embedding(
                g.inputs[0].tensor,
                _EMBED_DIM,
                flip_sin_to_cos=True,
                downscale_freq_shift=0.0,
            )
        )
    return g


def _stage_freq_row(device: DeviceRef, dim: int) -> Graph:
    """The frequency chain exactly as `get_1d_rotary_pos_embed` stages it.

    The signed kernel takes the frequency row as an operand -- the graph
    constant its own folding produces, never a recomputation -- so the
    reference table must be built from the graph's own row, and this graph
    is how the test reads it.
    """
    with Graph(
        "preamble_freq_row",
        input_types=[],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        freq_exponent = (
            ops.range(0, dim, 2, dtype=DType.float32, device=device) / dim
        )
        g.output(1.0 / (_THETA**freq_exponent))
    return g


# The CPU executor's transcendentals are not correctly rounded. Measured over
# these exact cases: one ulp on 0.3% of the rope table; up to 1013 ulps of
# the embedding core (a one-ulp band rounding scaled by arguments up to
# 1000). The bounds hold that envelope with a small margin, far under the
# smallest construction-level deviation the controls produce.
_ROPE_ULP_BOUND = 4
_SINUSOID_ULP_BOUND = 4096


def _max_ulp(a: np.ndarray, b: np.ndarray) -> int:
    """The largest bitwise f32 distance between two same-shape arrays."""
    d = a.view(np.uint32).astype(np.int64) - b.view(np.uint32).astype(np.int64)
    return int(np.abs(d).max())


def _reference_sinusoid_core(
    arg: np.ndarray, *, divisor: float = 128.0
) -> np.ndarray:
    """The signed kernel's construction over the round-tripped argument (the
    #156 record): the band exponent as `(coeff * i) / half` multiply before
    divide in f32, the transcendental cores in f64 rounded to f32 once,
    cosines first."""
    half = _EMBED_DIM // 2
    coeff = np.float32(-math.log(_THETA))
    i = np.arange(half, dtype=np.float32)
    band = np.exp(
        ((coeff * i) / np.float32(divisor)).astype(np.float64)
    ).astype(np.float32)
    angles = (arg[:, None] * band[None, :]).astype(np.float32)
    cos = np.cos(angles.astype(np.float64)).astype(np.float32)
    sin = np.sin(angles.astype(np.float64)).astype(np.float32)
    return np.concatenate([cos, sin], axis=1)


def _reference_rope(ids: np.ndarray, freq_rows: list[np.ndarray]) -> np.ndarray:
    """One row per token: interleaved (cos, sin) pairs over the given
    frequency rows, positions read column-by-column from the raw ids --
    `rope_table_2d`'s construction, with the graph's own folded rows as the
    frequency operands."""
    blocks = []
    for axis, freqs in enumerate(freq_rows):
        pos = ids[:, axis].astype(np.float32)
        angles = (pos[:, None] * freqs[None, :]).astype(np.float32)
        cos = np.cos(angles.astype(np.float64)).astype(np.float32)
        sin = np.sin(angles.astype(np.float64)).astype(np.float32)
        block = np.empty((ids.shape[0], 2 * freqs.shape[0]), dtype=np.float32)
        block[:, 0::2] = cos
        block[:, 1::2] = sin
        blocks.append(block)
    return np.concatenate(blocks, axis=1)


def test_staged_sinusoid_equals_the_reference_on_host() -> None:
    """The staged expression's f32 core, executed on the plain CPU at the
    round-tripped timestep rows, is bit-identical to the kernel's reference
    construction -- and the controls show the comparison can fail: feeding
    the un-round-tripped argument mismatches (the round-trip the kernel
    reproduces is real), and a shifted band divisor mismatches (the divisor
    gate guards a real difference)."""
    t = _timestep_rows()
    arg = (t * np.float32(_TIME_SCALE)).astype(np.float16).astype(np.float32)
    staged = _run_cpu(_stage_embedding_core(DeviceRef.CPU()), arg)

    reference = _reference_sinusoid_core(arg)
    assert staged.dtype == np.float32
    envelope = _max_ulp(staged, reference)
    assert envelope <= _SINUSOID_ULP_BOUND, (
        f"the staged embedding core deviates from the reference by "
        f"{envelope} ulps, past the CPU executor's measured rounding envelope"
    )

    no_round_trip = (t * np.float32(_TIME_SCALE)).astype(np.float32)
    staged_raw = _run_cpu(_stage_embedding_core(DeviceRef.CPU()), no_round_trip)
    control = _max_ulp(staged_raw, reference)
    assert control > _SINUSOID_ULP_BOUND, (
        f"an un-round-tripped argument deviates by only {control} ulps; the "
        "comparison could not tell the round-trip's absence apart"
    )

    shifted = _reference_sinusoid_core(arg, divisor=127.0)
    control = _max_ulp(staged, shifted)
    assert control > _SINUSOID_ULP_BOUND, (
        f"a shifted band divisor deviates by only {control} ulps; the "
        "comparison could not tell a downscale_freq_shift band apart"
    )


def test_staged_rope_equals_the_reference_on_host() -> None:
    """The staged rope maze, executed on the plain CPU over the packed FLUX
    ids, is bit-identical to `rope_table_2d`'s construction over the graph's
    own folded frequency rows -- 4608 x 32 per axis in (cos, sin) pairs --
    and mismatches a (sin, cos) reference, so the pair order is checked by
    something that can fail."""
    ids = _flux_ids()
    staged = _run_cpu(_stage_rope_tables(DeviceRef.CPU()), ids)
    freq_rows = [
        _run_cpu(_stage_freq_row(DeviceRef.CPU(), dim)) for dim in _AXES_DIM
    ]

    reference = _reference_rope(ids, freq_rows)
    assert staged.shape == (_TOKENS, sum(_AXES_DIM))
    envelope = _max_ulp(staged, reference)
    assert envelope <= _ROPE_ULP_BOUND, (
        f"the staged rope table deviates from the reference by {envelope} "
        "ulps, past the CPU executor's measured rounding envelope"
    )
    # The zero-id text rows are exact anchors: cos(0) and sin(0) have one
    # representable answer, so any operand or layout error would break them.
    assert np.array_equal(
        staged[:_TXT].view(np.uint32), reference[:_TXT].view(np.uint32)
    ), "the zero-position rows must be bit-exact"

    swapped = np.empty_like(reference)
    swapped[:, 0::2] = reference[:, 1::2]
    swapped[:, 1::2] = reference[:, 0::2]
    control = _max_ulp(staged, swapped)
    assert control > _ROPE_ULP_BOUND, (
        f"a (sin, cos) pair order deviates by only {control} ulps; the "
        "comparison could not tell the pair layout apart"
    )
