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
"""Produces a MEF from a FLUX.2 denoise graph for the on-device NPU.

`test_flux_denoise_whole_graph.py` pins what the unmodified graph refuses;
this file is the probe past that wall. Nothing has ever been compiled past
KGEN elaboration for this graph, so the point is to reach MEF serialization
and device execution with the smallest graph that can get there, and to
record what breaks.

The measured refusing surface at 1+0 float16 is layered, and serving one
layer exposes the next -- kernel resolution stops a chain at its first
refusal, so the whole-graph test's pinned set is the first layer, not the
set. Two censuses of the emitted module (2026-08-05, npu-main 5fc6da9abb9):

- Layer 1, the unmodified graph: eighteen generated-elementwise stubs from
  exactly three model-source families -- the six adaLN modulation sites
  (five token-shaped ``scale*x+shift`` regions plus three parameter-shaped
  ``1+scale`` residues), the rope tables (four per-axis frequency products,
  two ``mo.sin``, two ``mo.cos``), and the timestep sinusoid (the band
  product and the ``t*1000`` scaling).
- Layer 2, revealed once layer 1 is served: four matmuls carrying the
  gated-residual (one of them plus the fp16 saturation clamp) as an
  epilogue, which the pre-signed matmul refuses by design. The
  ``GatedResidualPattern`` that should claim the pair before epilogue
  fusion does not fire on this graph's shapes.

Everything else -- attention, norms, SwiGLU, silu, casts, concats, slices,
the plain and affine matmuls -- already lowers. So this probe substitutes
exactly those families and nothing else:

- modulation goes through the model's ``ada_modulate`` seam, patched here
  to emit ``mo.composite.ada_modulate``, whose signed selector is proven on
  silicon; the ``1+`` lives in the kernel, and the raw scale is what the
  seam carries.
- the gated residual goes through the model's ``gated_residual`` seam,
  patched to emit ``mo.composite.gated_residual`` (also signed and proven);
  emitting the composite directly means the mul/add pair never exists for
  epilogue fusion to claim.
- the fp16 saturation clamp is elided through the ``saturate_to_float16``
  seam: no signed kernel carries it yet (`gate_residual_clamp` has manifest
  rows but nothing claims a standalone clamp), and the probe's inputs are
  scaled so saturation cannot fire. This is the one place the probe graph
  computes something other than the model.
- the rope operands arrive as two graph inputs -- the packed `freqs_cis`
  table and the position ids -- and only the rope application stays
  in-graph. The first device run kept the packing in-graph (tables as raw
  cos/sin) and the rope kernel refused with `position_ids[0] is
  2147483648`: the `mo.range`-produced intermediate reached the kernel as
  an unwritten buffer, so host-materialized intermediates are their own
  unproven transport, and the probe feeds the packed operands directly.
- the timestep sinusoid arrives as one graph input; the timestep MLP and
  its weights stay in-graph.

The substitution costs three extra inputs (8 -> 11) and leaves the
``timestep``, ``guidance`` and the three id inputs unread by the probe graph
-- the ids exist only to build the tables this graph now takes as inputs.
That is the honest price of not writing the three PatternFusion rules first,
and it is deliberate: the rules are enumerable work, while everything past
elaboration is not.

`FLUX_MEF_TARGET=windows` retargets the host side to Windows ARM64 (ELF
container) and exports the MEF the device driver executes. Its own process,
as every cross-compile here is: `set_virtual_cpu_target` is global.

`FLUX_MEF_DUMP=dir` additionally writes the weights registry (real
checkpoint weights through the model's own `adapt_weights` +
`load_state_dict` route) and one set of small deterministic inputs, for the
device driver to load. `FLUX_WEIGHTS` names the safetensors file.

`FLUX_MODEL_EXACT=1` drops the three seam substitutions (modulate, gated
residual, clamp elision) and keeps only the table inputs. The seams predate
the PatternFusion rules: when this probe was written nothing claimed those
regions, and `#166`/`#174` have since landed matchers that mint
`mo.composite.ada_modulate`, `mo.composite.gated_residual` and
`mo.composite.gated_residual_clamp` from the raw ops -- the whole-graph
compile at this base proves all of them fire, and the clamp elision in
particular is no longer an acceptable price: the recorded render ran WITH
the clamp, so a run judged against it must carry it. Default off,
preserving the pinned probe behavior. The table substitution stays either
way: it is what keeps the windows export's constant-fold surface empty
(known-limitations/constant-folding-runs-on-the-cross-target-and-breaks-a-
windows-export) and what keeps the `mo.range` builtin-fill transport --
device finding D1's unproven class -- out of the executed graph.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from flux_rope_fused_staging import rope_fused_staging
from max.driver import (
    CPU,
    NPU,
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.graph.graph import KernelLibrary

# Captured at import, BEFORE any fixture patches it: the census test
# re-installs the real add-time verifier for its production-stager legs,
# because a census that runs under `virtual_npu`'s suppression reads green
# over a stager that cannot stage (window 285).
_REAL_VERIFY_CUSTOM_OP = KernelLibrary.verify_custom_op

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# FLUX.2-Klein-4B's transformer config at the locked revision; the depth is
# the probe's parameter. See `test_flux_denoise_whole_graph.py`.
_KLEIN_4B = {
    "num_layers": 5,
    "num_single_layers": 20,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "joint_attention_dim": 7680,
    "guidance_embeds": False,
}

# Geometry defaults to the recorded render (512x512: a 32x32 token grid),
# so the recorded stage tensors bind directly; FLUX_IMAGE_SEQ reaches other
# rungs, 4096 being the 1024x1024 the whole-graph test pins.
_IMAGE_SEQ = int(os.environ.get("FLUX_IMAGE_SEQ", "1024"))
_PINNED = {"image_seq_len": _IMAGE_SEQ, "img_seq": 0, "text_seq_len": 512}
_TOKENS = _PINNED["text_seq_len"] + _PINNED["image_seq_len"]
_HEAD_DIM = 128
_SIN_CHANNELS = 256
_AXES_DIM = (32, 32, 32, 32)
_ROPE_THETA = 2000.0
_STAGES = os.environ.get(
    "FLUX_STAGES", "/home/fabio/agentwork-backup/render-artifacts/stages"
)

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"


class _UnusedAccelerator:
    """Stands in for the `Accelerator` every block's `Allreduce` opens."""

    def __init__(self, id: int) -> None:
        self.id = id

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"single-device staging reached Allreduce.{name}")


def _composite_modulate(x, scale, shift):  # noqa: ANN001, ANN202
    """Emits the composite the modulation pattern would mint.

    The signed `ada_modulate` selector computes ``(1 + scale) * x + shift``,
    so the raw scale crosses unchanged -- adding the one here would modulate
    twice.
    """
    return ops.custom(
        "mo.composite.ada_modulate",
        device=x.device,
        values=[x, scale, shift],
        out_types=[TensorType(x.dtype, x.shape, x.device)],
    )[0].tensor


def _composite_gated_residual(x, gate, y):  # noqa: ANN001, ANN202
    """Emits the composite `GatedResidualPattern` would mint: x + gate * y.

    The registration binds (x, y, gate) positionally.
    """
    return ops.custom(
        "mo.composite.gated_residual",
        device=x.device,
        values=[x, y, gate],
        out_types=[TensorType(x.dtype, x.shape, x.device)],
    )[0].tensor


def _stage(
    layers: int, single: int, dtype: str, device: DeviceRef
) -> tuple[Graph, object]:
    from max.nn.comm import allreduce
    from max.pipelines.architectures.flux2 import flux2 as flux2_mod
    from max.pipelines.architectures.flux2.components.denoise_compute import (
        DenoiseComputeStep,
    )
    from max.pipelines.architectures.flux2.layers import (
        embeddings as embeddings_mod,
    )
    from max.pipelines.architectures.flux2.layers import (
        flux2_attention as attn_mod,
    )
    from max.pipelines.architectures.flux2.layers import (
        normalizations as norms_mod,
    )
    from max.pipelines.architectures.flux2.model_config import Flux2Config

    allreduce.Accelerator = _UnusedAccelerator  # type: ignore[misc,assignment]

    shape = dict(_KLEIN_4B, num_layers=layers, num_single_layers=single)
    driver = NPU() if device != DeviceRef.CPU() else CPU()
    config = Flux2Config.initialize_from_config(shape, dtype, [driver])
    step = DenoiseComputeStep(
        transformer=flux2_mod.Flux2Transformer2DModel(config),
        dtype=config.dtype,
        device=device,
    )
    for name, weight, _layer in step._iter_named_weights():
        weight.name = name

    pinned = [
        TensorType(t.dtype, [_PINNED.get(str(d), d) for d in t.shape], device)
        for t in step.input_types()
    ]
    substituted = [
        # timesteps_proj: what `Timesteps` computes from the scaled timestep.
        TensorType(DType.float32, ["batch", _SIN_CHANNELS], device),
        # rope operands: what `pack_flux2_rope` computes from the tables.
        TensorType(DType.float32, [_TOKENS, _HEAD_DIM], device),
        TensorType(DType.uint32, ["batch", _TOKENS], device),
    ]

    fed: dict[str, object] = {}
    saved = (
        embeddings_mod.Timesteps.__call__,
        attn_mod.Flux2PosEmbed.__call__,
        flux2_mod.pack_flux2_rope,
        norms_mod.ada_modulate,
        flux2_mod.ada_modulate,
        flux2_mod.gated_residual,
        flux2_mod.saturate_to_float16,
    )
    embeddings_mod.Timesteps.__call__ = lambda self, t: fed["tproj"]
    # The table computation must not be STAGED at all -- dead ops are not
    # reliably eliminated before kernel elaboration -- so `pos_embed` is
    # stubbed even though the patched pack ignores its result.
    attn_mod.Flux2PosEmbed.__call__ = lambda self, ids: (
        fed["freqs_cis"],
        fed["freqs_cis"],
    )
    flux2_mod.pack_flux2_rope = lambda cos, sin, batch, seq: (
        fed["freqs_cis"],
        ops.reshape(fed["position_ids"], [1, batch * seq]),
    )
    if os.environ.get("FLUX_MODEL_EXACT", "") != "1":
        norms_mod.ada_modulate = _composite_modulate
        flux2_mod.ada_modulate = _composite_modulate
        flux2_mod.gated_residual = _composite_gated_residual
        flux2_mod.saturate_to_float16 = lambda x: x

    try:
        # The fused-qk staging rewrite is SHARED with the whole-graph stager
        # (flux_rope_fused_staging), so the production export and this probe
        # cannot diverge -- window 281's finding.
        with (
            rope_fused_staging(step),
            Graph(
                "flux_denoise_probe",
                input_types=pinned + substituted,
                custom_extensions=[_KERNELS],
            ) as graph,
        ):
            fed["tproj"] = graph.inputs[8].tensor
            fed["freqs_cis"] = graph.inputs[9].tensor
            fed["position_ids"] = graph.inputs[10].tensor
            graph.output(
                step(*(v.tensor for v in graph.inputs[:8]), signal_buffers=[])
            )
    finally:
        (
            embeddings_mod.Timesteps.__call__,
            attn_mod.Flux2PosEmbed.__call__,
            flux2_mod.pack_flux2_rope,
            norms_mod.ada_modulate,
            flux2_mod.ada_modulate,
            flux2_mod.gated_residual,
            flux2_mod.saturate_to_float16,
        ) = saved
    return graph, step


def _chain(exc: BaseException) -> str:
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return "\n---\n".join(parts)


def _emit(record: dict[str, object], tag: str = "JSON") -> None:
    print(f"###{tag}###" + json.dumps(record), flush=True)
    if out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        name = (
            f"mef_{record['layers']}_{record['single']}_{record['target']}.json"
        )
        (Path(out) / name).write_text(json.dumps(record, indent=2))


def _dump_weights_and_inputs(step: object, out_dir: Path) -> dict[str, int]:
    """Writes the registry and one input set for the device driver.

    Weights go through the model's own route -- `load_weights`,
    `adapt_weights`, `load_state_dict` -- so the registry keys and bytes are
    the ones a real build would hand `M_initModel`, filtered to the weights
    this depth actually stages.
    """
    _load_real_weights(step)

    wdir = out_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    total = 0
    for name, value in step.state_dict().items():
        arr = np.ascontiguousarray(_host_array(value))
        fname = name.replace("/", "_") + ".bin"
        arr.tofile(wdir / fname)
        manifest.append(f"{name}\t{fname}\t{arr.nbytes}")
        total += arr.nbytes
    (wdir / "MANIFEST.tsv").write_text("\n".join(manifest) + "\n")

    idir = out_dir / "inputs"
    idir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, (name, arr) in enumerate(_build_inputs().items()):
        np.ascontiguousarray(arr).tofile(idir / f"input{i}.bin")
        lines.append(
            f"input{i}\t{name}\tinput{i}.bin\t{arr.dtype}\t{list(arr.shape)}"
        )
    (idir / "MANIFEST.tsv").write_text("\n".join(lines) + "\n")
    return {"weights": len(manifest), "weight_bytes": total}


def _build_inputs() -> dict[str, np.ndarray]:
    """One input set, in graph positional order.

    At the recorded render's geometry the real recorded stage tensors are
    the inputs -- `denoise.latent_init` and `encoder.out`, with the recorded
    text ids -- and the step-0 timestep is the flow-match sigma, exactly 1.0.
    At any other geometry the tensors are small deterministic stand-ins.
    The substituted tables are the exact float64 mirrors of `Timesteps` and
    `Flux2PosEmbed` over the same ids the graph would have used.
    """
    img, txt = _PINNED["image_seq_len"], _PINNED["text_seq_len"]
    in_ch = 128
    joint = _KLEIN_4B["joint_attention_dim"]
    side = int(np.sqrt(img))
    r = np.arange(img, dtype=np.int64)
    latent_image_ids = np.stack(
        [np.zeros_like(r), r // side, r % side, np.zeros_like(r)], axis=-1
    )[None]
    recorded = Path(_STAGES) / "denoise.latent_init.npy"
    if img == 1024 and recorded.exists():
        latents = np.load(recorded).astype(np.float16)
        enc = np.load(Path(_STAGES) / "encoder.out.npy").astype(np.float16)
        txt_ids = np.load(Path(_STAGES) / "encoder.text_ids.npy")
        assert latents.shape == (1, img, in_ch), latents.shape
        assert enc.shape == (1, txt, joint), enc.shape
    else:
        rng = np.random.default_rng(20260805)
        latents = (rng.standard_normal((1, img, in_ch)) * 0.05).astype(
            np.float16
        )
        enc = (rng.standard_normal((1, txt, joint)) * 0.05).astype(np.float16)
        t = np.arange(txt, dtype=np.int64)
        txt_ids = np.stack(
            [np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), t], axis=-1
        )[None]
    ids = np.concatenate([txt_ids[0], latent_image_ids[0]], axis=0)
    cos, sin = _rope_tables(ids.astype(np.float64))
    # `pack_flux2_rope`'s pairing: even columns carry the unique cosines,
    # odd columns the matching sines.
    freqs_cis = np.empty((_TOKENS, _HEAD_DIM), np.float64)
    freqs_cis[:, 0::2] = cos[:, 0::2]
    freqs_cis[:, 1::2] = sin[:, 0::2]
    return {
        "latents": latents,
        "image_latents": np.zeros((1, 0, in_ch), np.float16),
        "encoder_hidden_states": enc,
        "timestep": np.full((1,), 1.0, np.float32),
        "guidance": np.full((1,), 3.5, np.float32),
        "latent_image_ids": latent_image_ids,
        "image_latent_ids": np.zeros((1, 0, 4), np.int64),
        "txt_ids": txt_ids,
        "timesteps_proj": _timestep_sinusoid(
            np.array([1000.0], np.float64)
        ).astype(np.float32),
        "freqs_cis": freqs_cis.astype(np.float32),
        "position_ids": np.arange(_TOKENS, dtype=np.uint32)[None, :].copy(),
    }


def _host_array(value: object) -> np.ndarray:
    """Materializes a WeightData or driver buffer as a numpy array.

    bfloat16 has no numpy dtype, so a bf16 buffer crosses as its uint16 bit
    pattern and is widened to float32 by shifting into the high half --
    exact, because bf16 is float32's top sixteen bits.
    """
    data = getattr(value, "data", value)
    if isinstance(data, np.ndarray):
        return data
    declared = getattr(value, "dtype", None) or getattr(data, "dtype", None)
    shape = [
        int(d)
        for d in (getattr(value, "shape", None) or getattr(data, "shape", []))
    ]
    if isinstance(data, memoryview):
        if declared == DType.bfloat16:
            bits = np.frombuffer(data, dtype=np.uint16).reshape(shape)
            return (bits.astype(np.uint32) << 16).view(np.float32)
        np_dtype = (
            np.dtype(declared.to_numpy())
            if isinstance(declared, DType)
            else np.dtype(declared)
        )
        return np.frombuffer(data, dtype=np_dtype).reshape(shape)
    if getattr(data, "dtype", None) == DType.bfloat16:
        bits = data.view(DType.uint16, tuple(data.shape)).to_numpy()
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return data.to_numpy()


def _timestep_sinusoid(t: np.ndarray) -> np.ndarray:
    """Mirrors `Timesteps` (flip_sin_to_cos=True, shift 0) in float64.

    The flip places the cos half first. ``t`` is the already-scaled
    timestep; at sigma 1.0 the scaling to 1000.0 is exact in float16, so
    the host mirror and the graph agree bitwise on the input.
    """
    half = _SIN_CHANNELS // 2
    exponent = -np.log(10000.0) * np.arange(half, dtype=np.float64) / half
    emb = t[:, None] * np.exp(exponent)[None, :]
    return np.concatenate([np.cos(emb), np.sin(emb)], axis=-1)


def _rope_tables(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirrors `Flux2PosEmbed` (theta 2000, repeat-interleaved) in float64."""
    cos_out, sin_out = [], []
    for axis, dim in enumerate(_AXES_DIM):
        freq = 1.0 / _ROPE_THETA ** (np.arange(0, dim, 2, np.float64) / dim)
        freqs = ids[:, axis][:, None] * freq[None, :]
        cos_out.append(np.repeat(np.cos(freqs), 2, axis=1))
        sin_out.append(np.repeat(np.sin(freqs), 2, axis=1))
    return (
        np.concatenate(cos_out, axis=-1),
        np.concatenate(sin_out, axis=-1),
    )


def _unfused_attention(q, k, v, mask_variant, scale, **_kwargs):  # noqa: ANN001, ANN202
    """Spells attention in ops the CPU target has.

    `flash_attention_gpu` bottoms out in a builtin that is `only valid on
    GPUs` (measured here), so the CPU reference computes softmax(sQK^T)V
    unfused. Layout matches the kernel: [B, S, H, D] in and out.
    """
    qt = ops.permute(q, [0, 2, 1, 3])
    kt = ops.permute(k, [0, 2, 1, 3])
    vt = ops.permute(v, [0, 2, 1, 3])
    scores = ops.matmul(qt, ops.transpose(kt, 2, 3)) * scale
    return ops.permute(ops.matmul(ops.softmax(scores), vt), [0, 2, 1, 3])


def _cpu_reference(
    layers: int, single: int, out_dir: Path
) -> dict[str, object]:
    """Runs the same function on the host CPU and records the comparison.

    The CPU twin stages the same substituted graph -- tables as inputs,
    saturation elided -- but keeps the modulation and gated residual as raw
    ops, since the composites are NPU-scoped; the two spellings compute the
    same expression. It runs at float32 because the CPU target refuses f16
    outright ("The f16 data type is not supported on device 'cpu:0'",
    measured here), over the same input VALUES the device sees -- the fp16
    inputs widened bit-exactly -- so the comparison carries fp16-kernel
    rounding but no input skew. The recorded step-0 velocity says how far a
    1+0 truncation sits from the real 5+20 model.
    """
    from max.pipelines.architectures.flux2 import flux2 as flux2_mod
    from max.pipelines.architectures.flux2.components.denoise_compute import (
        DenoiseComputeStep,
    )
    from max.pipelines.architectures.flux2.layers import (
        embeddings as embeddings_mod,
    )
    from max.pipelines.architectures.flux2.layers import (
        flux2_attention as attn_mod,
    )
    from max.pipelines.architectures.flux2.model_config import Flux2Config

    shape = dict(_KLEIN_4B, num_layers=layers, num_single_layers=single)
    config = Flux2Config.initialize_from_config(shape, "float32", [CPU()])
    step = DenoiseComputeStep(
        transformer=flux2_mod.Flux2Transformer2DModel(config),
        dtype=config.dtype,
        device=DeviceRef.CPU(),
    )
    for name, weight, _layer in step._iter_named_weights():
        weight.name = name

    device = DeviceRef.CPU()
    pinned = [
        TensorType(t.dtype, [_PINNED.get(str(d), d) for d in t.shape], device)
        for t in step.input_types()
    ]
    substituted = [
        TensorType(DType.float32, ["batch", _SIN_CHANNELS], device),
        TensorType(DType.float32, [_TOKENS, _HEAD_DIM], device),
        TensorType(DType.uint32, ["batch", _TOKENS], device),
    ]
    fed: dict[str, object] = {}
    saved = (
        embeddings_mod.Timesteps.__call__,
        attn_mod.Flux2PosEmbed.__call__,
        flux2_mod.pack_flux2_rope,
        flux2_mod.saturate_to_float16,
        attn_mod.flash_attention_gpu,
    )
    embeddings_mod.Timesteps.__call__ = lambda self, t: fed["tproj"]
    attn_mod.Flux2PosEmbed.__call__ = lambda self, ids: (
        fed["freqs_cis"],
        fed["freqs_cis"],
    )
    flux2_mod.pack_flux2_rope = lambda cos, sin, batch, seq: (
        fed["freqs_cis"],
        ops.reshape(fed["position_ids"], [1, batch * seq]),
    )
    flux2_mod.saturate_to_float16 = lambda x: x
    attn_mod.flash_attention_gpu = _unfused_attention
    try:
        with Graph(
            "flux_denoise_probe_cpu", input_types=pinned + substituted
        ) as graph:
            fed["tproj"] = graph.inputs[8].tensor
            fed["freqs_cis"] = graph.inputs[9].tensor
            fed["position_ids"] = graph.inputs[10].tensor
            graph.output(
                step(*(v.tensor for v in graph.inputs[:8]), signal_buffers=[])
            )
    finally:
        (
            embeddings_mod.Timesteps.__call__,
            attn_mod.Flux2PosEmbed.__call__,
            flux2_mod.pack_flux2_rope,
            flux2_mod.saturate_to_float16,
            attn_mod.flash_attention_gpu,
        ) = saved

    _load_real_weights(step)
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=step.state_dict())
    widened = [
        a.astype(np.float32) if a.dtype == np.float16 else a
        for a in _build_inputs().values()
    ]
    outputs = model.execute(*widened)
    result = np.from_dlpack(outputs[0])
    np.save(out_dir / "reference_output.npy", result)

    record: dict[str, object] = {
        "reference_dtype": "float32",
        "reference_shape": list(result.shape),
        "reference_nonfinite": int(np.count_nonzero(~np.isfinite(result))),
        "reference_max_abs": float(np.abs(result).max()),
    }
    velocity = Path(_STAGES) / "denoise.step_00.velocity.npy"
    if _IMAGE_SEQ == 1024 and velocity.exists():
        v = np.load(velocity)
        d = result.astype(np.float64) - v.astype(np.float64)
        record["vs_recorded_step0_velocity"] = {
            "max_abs_diff": float(np.abs(d).max()),
            "mean_abs_diff": float(np.abs(d).mean()),
            "recorded_mean_abs": float(np.abs(v).mean()),
        }
    return record


def _load_real_weights(step: object) -> None:
    """Loads checkpoint weights and casts each to its staged dtype, for real.

    `adapt_weights` is given no dtype on purpose: its cast rides
    `WeightData.astype`, which in virtual-device mode changes only the dtype
    METADATA and leaves the bytes alone -- sound for the warm-cache compile
    it was written for, and silently wrong for anything that then uses the
    bytes, which a registry dump does. So the cast happens here in numpy,
    against the dtype each staged weight declares, trusting the buffer's own
    dtype rather than the metadata.
    """
    from max.graph.weights import load_weights
    from max.pipelines.architectures.flux2.weight_adapters import adapt_weights

    weights_file = os.environ.get(
        "FLUX_WEIGHTS",
        "/home/fabio/work/flux2-klein-4b/transformer/"
        "diffusion_pytorch_model.safetensors",
    )
    staged = {
        name: weight.dtype for name, weight, _l in step._iter_named_weights()
    }
    raw = {
        key: value.data()
        for key, value in load_weights([Path(weights_file)]).items()
    }
    adapted = adapt_weights(raw, None, None)
    state_dict: dict[str, np.ndarray] = {}
    for key, value in adapted.items():
        name = f"transformer.{key}"
        if name not in staged:
            continue
        arr = _host_array(value)
        want = np.dtype(staged[name].to_numpy())
        if arr.dtype != want:
            arr = arr.astype(np.float32).astype(want)
        state_dict[name] = np.ascontiguousarray(arr)
    missing = set(staged) - set(state_dict)
    if missing:
        raise AssertionError(f"checkpoint lacks staged weights: {missing}")
    step.load_state_dict(state_dict, weight_alignment=1)


@pytest.fixture
def virtual_npu(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    """Enumerates an NPU target and defers custom-op verification.

    The composite is matchable only by device API, which exists once the
    compile holds the (virtual) device -- `ops.custom`'s add-time check
    cannot see it, exactly as `test_npu_graph_sincos_tables.py` records.
    """
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


def test_flux_denoise_probe_produces_a_mef(virtual_npu: None) -> None:
    layers = int(os.environ.get("FLUX_LAYERS", "1"))
    single = int(os.environ.get("FLUX_SINGLE", "0"))
    target = os.environ.get("FLUX_MEF_TARGET", "linux")
    dump = os.environ.get("FLUX_MEF_DUMP", "")

    # Same stage-dump affordance as the whole-graph sibling: this module is
    # the compiles-whole control the terminal over there gets diffed against,
    # so it has to be dumpable with the same switch.
    if os.environ.get("FLUX_IR_DUMP", "") == "1" and (
        out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    ):
        ir_dir = Path(out) / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODULAR_DEBUG"] = f"ir-output-dir={ir_dir}"

    record: dict[str, object] = {
        "layers": layers,
        "single": single,
        "dtype": "float16",
        "target": target,
    }
    if target == "windows":
        set_virtual_cpu_target(_WINDOWS_ARM64_ELF)

    started = time.monotonic()
    try:
        graph, step = _stage(layers, single, "float16", DeviceRef.NPU())
    except BaseException as exc:
        record["outcome"] = "STAGING_ERROR"
        record["error"] = _chain(exc)[:8000]
        _emit(record)
        raise
    record["stage_seconds"] = round(time.monotonic() - started, 1)
    text = str(graph)
    record["composites"] = {
        "ada_modulate": text.count("mo.composite.ada_modulate"),
        "gated_residual": text.count("mo.composite.gated_residual"),
        "rms_norm_rope": text.count("fused_rms_norm_rope"),
    }
    record["rope_fused_norm"] = (
        os.environ.get("FLUX_ROPE_FUSED_NORM", "") == "1"
    )
    record["attention_family"] = {
        "rms_norm": text.count("mo.reduce.rms_norm"),
        "rope_ragged": text.count("mo.rope.ragged.with_position_id"),
    }
    record["external_constants"] = text.count("mo.constant.external")

    started = time.monotonic()
    try:
        compiled = InferenceSession(devices=[NPU()]).compile(graph)
    except BaseException as exc:
        record["outcome"] = "REFUSED"
        record["compile_seconds"] = round(time.monotonic() - started, 1)
        record["diagnostic"] = _chain(exc)
        _emit(record)
        pytest.fail(
            "the probe graph refused; the substitution set moved or is "
            "incomplete. See the record."
        )
    record["outcome"] = "COMPILED"
    record["compile_seconds"] = round(time.monotonic() - started, 1)

    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
    mef = out / f"flux_denoise_{layers}_{single}_{target}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    record["mef_bytes"] = len(blob)
    record["mef_markers"] = {
        marker: marker.encode() in blob
        for marker in (
            "hexagon_ondevice",
            '{"label":"npu","id":0}',
            "ada_modulate",
            "rms_norm_rope",
            "transformer.transformer_blocks.0.attn.to_q.weight",
            "__chkstk",
        )
    }

    if dump:
        # Two process-global modes gate this leg, both measured here. The
        # windows retarget cannot be undone, and a host session under it
        # fails kernel materialization ("Failed to materialize symbols:
        # __kernel_*") -- so the dump and the reference ride the linux
        # invocation only. Virtual-device mode stubs real execution
        # (`model.execute` returns a Mock), so it is dropped first; the
        # fixture's teardown repeats these resets.
        assert target != "windows", (
            "FLUX_MEF_DUMP needs a process the windows retarget has not"
            " poisoned; run the dump with FLUX_MEF_TARGET=linux"
        )
        set_virtual_device_count(0)
        set_virtual_device_api("")
        set_virtual_device_target_arch("")
        record["dump"] = _dump_weights_and_inputs(step, Path(dump))
        record["reference"] = _cpu_reference(layers, single, Path(dump))

    _emit(record)

    assert blob, "empty MEF"
    assert b"hexagon_ondevice" in blob
    if target == "windows":
        assert b"__chkstk" in blob


def test_flux_rope_fused_norm_census(
    virtual_npu: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ROPE_FUSED_NORM engagement census, device-free.

    The fourth real-correct-unreachable-arm instance (#487) passed every
    step-0 gate while nothing on the production path asked for the fused arm.
    This is the check that class bought: at 1 double + 1 single block, the
    flag-on staging must mint exactly the single-stream pairs as
    `fused_rms_norm_rope` (and no others), the composite must survive
    graph compile against the registration and appear in the exported MEF's
    bytes, and the flag-off staging must contain none of it. The production
    stager's legs then run at production depth with the REAL add-time
    verifier re-installed (window 285: the suppressed verifier let a green
    census hide a production export that refused to stage), behind a
    can-fail control that reproduces the refusal on an absent kernel name.
    """
    monkeypatch.setenv("FLUX_ROPE_FUSED_NORM", "1")
    graph_on, _ = _stage(1, 1, "float16", DeviceRef.NPU())
    on = str(graph_on)
    monkeypatch.setenv("FLUX_ROPE_FUSED_NORM", "0")
    off = str(_stage(1, 1, "float16", DeviceRef.NPU())[0])

    assert on.count("fused_rms_norm_rope") == 2, (
        "one fused op per single-stream qk rope (q and k of 1 single block)"
    )
    assert off.count("fused_rms_norm_rope") == 0, (
        "flag-off staging must be inert"
    )
    assert off.count("mo.rope.ragged.with_position_id") == 4
    assert on.count("mo.rope.ragged.with_position_id") == 2, (
        "the dual-stream block's norm-concat-rope chain stays unfused"
    )
    assert (
        off.count("mo.reduce.rms_norm") - on.count("mo.reduce.rms_norm") == 2
    ), "exactly the two fused qk-norms leave the unfused op census"

    compiled = InferenceSession(devices=[NPU()]).compile(graph_on)
    mef = tmp_path / "flux_rope_fused_census.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    assert b"rms_norm_rope" in blob, (
        "the fused composite's launcher must reach the exported MEF -- this"
        " byte census is the window's step-0 engagement receipt"
    )

    # The PRODUCTION stager: both banked npu.mef recipes export through
    # test_flux_denoise_whole_graph._stage, and window 281 caught the rewrite
    # living only in this module's stager (the fifth unreachable-arm
    # instance). The shared helper closes that; this leg proves it AT
    # PRODUCTION DEPTH on the production module. Deltas rather than absolute
    # unfused counts, because the whole8 contract stages the real preamble
    # and modulation paths this module substitutes away.
    #
    # WITH THE ADD-TIME KERNEL CHECK LIVE (window 285): the production
    # stager's fixture never suppresses `verify_custom_op`, and this module's
    # `virtual_npu` fixture does -- which is exactly how a green census hid a
    # production export that refused to stage. These legs re-install the real
    # verifier, prove it can fail (an absent kernel name must refuse with the
    # very error the 285 seat observed), and only then stage the production
    # module through it.
    import test_flux_denoise_whole_graph as whole_graph

    monkeypatch.setattr(
        KernelLibrary, "verify_custom_op", _REAL_VERIFY_CUSTOM_OP
    )
    # The control mints `mo.composite.ada_modulate`, whose registration pins
    # `api="npu"` -- the exact device-info-mismatch refusal the 285 seat
    # observed on the production export. It refusing here proves the
    # verifier below is genuinely live, not the fixture's no-op.
    with pytest.raises(Exception, match="no kernel matching device info"):
        with Graph(
            "verify_live_control",
            input_types=[TensorType(DType.float16, [4, 4], DeviceRef.NPU())],
            custom_extensions=[_KERNELS],
        ) as control:
            x = control.inputs[0].tensor
            ops.custom(
                "mo.composite.ada_modulate",
                device=DeviceRef.NPU(),
                values=[x, x, x],
                out_types=[TensorType(DType.float16, [4, 4], DeviceRef.NPU())],
            )

    monkeypatch.setenv("FLUX_ROPE_FUSED_NORM", "1")
    wg_on = str(whole_graph._stage(5, 20, "float16", DeviceRef.NPU())[0])
    monkeypatch.setenv("FLUX_ROPE_FUSED_NORM", "0")
    wg_off = str(whole_graph._stage(5, 20, "float16", DeviceRef.NPU())[0])
    assert wg_on.count("fused_rms_norm_rope") == 40, (
        "production depth: 2 fused ops per single block x 20 single blocks"
    )
    assert wg_off.count("fused_rms_norm_rope") == 0
    assert (
        wg_off.count("mo.rope.ragged.with_position_id")
        - wg_on.count("mo.rope.ragged.with_position_id")
        == 40
    ), "the 40 single-block ropes leave the unfused census"
    assert (
        wg_off.count("mo.reduce.rms_norm") - wg_on.count("mo.reduce.rms_norm")
        == 40
    ), "the 40 fused qk-norms leave the unfused census"
    print(
        "whole-graph census (5,20) under LIVE verify_custom_op: fused=40,"
        f" rope {wg_off.count('mo.rope.ragged.with_position_id')}->"
        f"{wg_on.count('mo.rope.ragged.with_position_id')},"
        f" rms {wg_off.count('mo.reduce.rms_norm')}->"
        f"{wg_on.count('mo.reduce.rms_norm')}"
    )
