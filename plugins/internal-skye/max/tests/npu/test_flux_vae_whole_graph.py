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
"""Compiles the FLUX.2-Klein VAE decode graph for the on-device NPU.

The last of the three render legs, staged from the model code the Klein
pipeline runs: `PostprocessAndDecode` (`flux2/components/vae_decoder.py`),
which fuses the BN denormalize, the 2x2 unpatchify, the `autoencoders`
`Decoder` conv stack and the `[-1,1] -> uint8` range map into the one
`vae_decode` graph `VaeDecoder` compiles. The boundary is the recorded
render's: `denoise.out` packed latents ``[B, S, 128]`` with two spatial
carrier inputs, to `vae.rgb` ``[B, 16*h, 16*w, 3]`` uint8 on the host.

The decoder topology is the checkpoint's (`vae/config.json` at the locked
revision): stages [128, 256, 512, 512], two ResNets per block (staged as
three per up-block), GroupNorm(32)/SiLU throughout, a mid-block with the
one-head full-channel self-attention, `use_post_quant_conv` on, every conv
stride-1. All of it is env-parameterized because the spatial grid, not the
op set, is what costs: the pinned rung stages the real channel/topology at a
small 8x8 packed grid, and `FLUX_VAE_PACKED=32` reaches the recorded render's
geometry (512x512 pixels) for the run package.

The port design keeps conv off the device's new-kernel list -- host im2col
feeding the existing b-transposed matmul, exactly what the silicon-verified
hand driver (`npu_vae_decoder_dispatch.cpp`) runs -- and the graph path now
does the same in two parts, measured from the emitted MOGG module
(FLUX_IR_DUMP=1, `mogg.call` census): the three 1x1 convs (post_quant_conv
and both stage-narrowing shortcuts) are rewritten to `mo.matmul` by
`ConvToMatmulPattern` (Algebraic.cpp) and bind the SIGNED NPU matmul --
the design's "1x1 conv is a matmul, no im2col" clause was already the graph
path's behavior -- while the thirty-three 3x3 convs go through the NPU
`mo.conv` registration (`registrations.mojo`), a host im2col gather in the
filter's own FCRS contraction order feeding the signed matmul selector.
Everything else the decoder stages is claimed: group_norm x30, silu x29,
matmul x7, `mo.mha.no_cache` for the mid-block attention, and the movement
-- measured, 137 `npu_view_copy` host-walk materializations carry the
graph's views.

Both denoise staging lessons carry over: the batch stays symbolic (a
literal-1 batch is squeezed into `mo.matmul` and refused;
`test_npu_graph_batch_matmul.py`), and the dtype is float16
(`FLUX_DTYPE=bfloat16` stages anyway; its dtype refusals keep `_ABSENT`
honest). `FLUX_MEF_TARGET=windows` retargets the host side in its own
process; `FLUX_STAGE_ONLY=1` stops after staging and the inventory.

`FLUX_MEF_DUMP=dir` additionally writes the full weights registry -- every
external constant the graph declares, real checkpoint bytes from
`FLUX_VAE_WEIGHTS` -- with the same manifest format as the denoise sibling's
dump leg. The VAE registry is geometry-independent (a conv stack), so a dump
at any packed grid is THE registry: 142 names, the decoder side plus
`post_quant_conv` and the unsuffixed `decoder_bn_mean`/`decoder_bn_var`
batch-norm stats. Linux invocation only: the windows retarget poisons the
process for anything that follows it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from flux_stager_inventory import weight_consumers
from max.driver import (
    CPU,
    NPU,
    set_virtual_cpu_target,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# FLUX.2-Klein-4B's VAE at the locked revision (`vae/config.json`, HF SHA
# e7b7dc2, `hf-repo-lock.tsv`): the diffusers-standard FLUX shape the P5
# scoping predicted and the checkpoint confirms.
_KLEIN_VAE = {
    "latent_channels": 32,
    "out_channels": 3,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "mid_block_add_attention": True,
    "use_post_quant_conv": True,
    "batch_norm_eps": 1e-4,
}

# The packed token grid `denoise.out` carries: latents are [B, h*w, 128], the
# unpatchify doubles the grid to the 32-channel latent, and the decoder's
# three upsamples plus the 8x VAE scale make one packed token 16x16 pixels.
# 8 pins a 128x128-pixel decode; the recorded render's 512x512 is 32.
_PACKED = int(os.environ.get("FLUX_VAE_PACKED", "8"))

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_DEF = re.compile(r"^\s+(?:%\S+(?:, %\S+)*) = ([a-z][A-Za-z0-9_.]*)")
_SYMBOL = re.compile(r'symbol\s*=\s*"([^"]+)"')


def _blocks() -> tuple[int, ...]:
    if raw := os.environ.get("FLUX_VAE_BLOCKS", ""):
        return tuple(int(part) for part in raw.split(","))
    return _KLEIN_VAE["block_out_channels"]


def _stage(
    blocks: tuple[int, ...],
    layers_per_block: int,
    attn: bool,
    dtype: str,
    device: DeviceRef,
) -> tuple[Graph, object]:
    from max.dtype import DType
    from max.graph import TensorType
    from max.pipelines.architectures.autoencoders.vae_flux2 import Decoder
    from max.pipelines.architectures.flux2.components.vae_decoder import (
        PostprocessAndDecode,
    )

    decoder = Decoder(
        in_channels=_KLEIN_VAE["latent_channels"],
        out_channels=_KLEIN_VAE["out_channels"],
        up_block_types=("UpDecoderBlock2D",) * len(blocks),
        block_out_channels=blocks,
        layers_per_block=layers_per_block,
        norm_num_groups=_KLEIN_VAE["norm_num_groups"],
        act_fn="silu",
        norm_type="group",
        mid_block_add_attention=attn,
        use_post_quant_conv=_KLEIN_VAE["use_post_quant_conv"],
        device=device,
        dtype=DType[dtype],
    )
    fused = PostprocessAndDecode(
        decoder=decoder,
        batch_norm_eps=_KLEIN_VAE["batch_norm_eps"],
        num_channels=_KLEIN_VAE["latent_channels"] * 4,
        device=device,
        dtype=DType[dtype],
    )
    # What `load_state_dict` does to a weight besides giving it data; without
    # it the external constants carry no fully-qualified name.
    for name, weight, _layer in fused._iter_named_weights():
        weight.name = name

    # Batch stays symbolic (the denoise staging lesson); the packed grid and
    # with it the `seq` axis are pinned, as the whole-graph tests pin every
    # spatial extent. The carriers convey h/w as their lengths.
    pinned_dims = {
        "seq": _PACKED * _PACKED,
        "latent_h": _PACKED,
        "latent_w": _PACKED,
    }
    pinned = [
        TensorType(
            t.dtype,
            [pinned_dims.get(str(d), d) for d in t.shape],
            t.device,
        )
        for t in fused.input_types()
    ]
    with Graph(
        "vae_decode", input_types=pinned, custom_extensions=[_KERNELS]
    ) as graph:
        graph.output(fused(*(v.tensor for v in graph.inputs)))
    return graph, fused


def _inventory(mlir: str) -> dict[str, object]:
    """Counts op kinds and records where each first appears."""
    kinds: Counter[str] = Counter()
    first: dict[str, int] = {}
    customs: Counter[str] = Counter()
    for index, line in enumerate(mlir.splitlines()):
        if not (m := _DEF.match(line)):
            continue
        kind = m.group(1)
        kinds[kind] += 1
        first.setdefault(kind, index + 1)
        if kind.endswith("custom") and (sym := _SYMBOL.search(line)):
            customs[sym.group(1)] += 1
    return {
        "total_ops": sum(kinds.values()),
        "kinds": [
            {"op": k, "count": c, "first_line": first[k]}
            for k, c in sorted(kinds.items(), key=lambda kv: first[kv[0]])
        ],
        "custom_symbols": dict(customs.most_common()),
    }


def _chain(exc: BaseException) -> str:
    """Joins a compile failure with its causes; the top message is generic."""
    parts: list[str] = []
    cur: BaseException | None = exc
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    return "\n---\n".join(parts)


def _load_real_weights(fused: object) -> None:
    """Loads checkpoint weights and casts each to its staged dtype, for real.

    The denoise sibling's route (`test_flux_denoise_mef._load_real_weights`),
    with the VAE's own name mapping -- the four rules
    `VaeDecoder._adapt_state_dict` documents (`decoder.*` verbatim,
    `post_quant_conv.*` under `decoder.`, `bn.running_mean/var` to the
    unsuffixed `decoder_bn_*`). The mapping is applied here rather than by
    calling that method because its dtype cast rides `WeightData.astype`,
    which under virtual-device mode changes only the metadata; the cast
    happens in numpy instead, against each staged weight's declared dtype.
    """
    from max.graph.weights import load_weights
    from test_flux_denoise_mef import _host_array

    spec = os.environ.get("FLUX_VAE_WEIGHTS", "")
    assert spec, (
        "FLUX_MEF_DUMP needs FLUX_VAE_WEIGHTS: the vae checkpoint safetensors"
    )
    path = Path(spec)
    assert path.is_file(), f"FLUX_VAE_WEIGHTS does not exist: {path}"
    staged = {
        name: weight.dtype for name, weight, _l in fused._iter_named_weights()
    }
    state_dict: dict[str, np.ndarray] = {}
    for key, value in load_weights([path]).items():
        if key.startswith("decoder."):
            name = key
        elif key.startswith("post_quant_conv."):
            name = f"decoder.{key}"
        elif key == "bn.running_mean":
            name = "decoder_bn_mean"
        elif key == "bn.running_var":
            name = "decoder_bn_var"
        else:
            continue
        if name not in staged:
            continue
        arr = np.ascontiguousarray(_host_array(value.data()))
        want = np.dtype(staged[name].to_numpy())
        if arr.dtype != want:
            arr = arr.astype(np.float32).astype(want)
        state_dict[name] = np.ascontiguousarray(arr)
    missing = set(staged) - set(state_dict)
    if missing:
        raise AssertionError(f"checkpoint lacks staged weights: {missing}")
    # All-zero BN stats collapse the denormalization to near-zero and serve
    # washed-out images -- the pipeline's own loader refuses them
    # (`VaeDecoder.load_model`), so the registry dump does too.
    for bn_key in ("decoder_bn_mean", "decoder_bn_var"):
        assert not np.all(state_dict[bn_key] == 0), f"{bn_key} is all zeros"
    fused.load_state_dict(state_dict, weight_alignment=1)


# The rung the census below describes: the real Klein topology at the 8x8
# packed grid, fp16. The grid scales extents, not op kinds or instantiation
# classes, so the census is the same one the recorded render's 32 would pin
# at four times the compile cost.
_PINNED_RUNG = ((128, 256, 512, 512), 2, True, "float16")

# What the compile reports at that rung: nothing -- THE RUNG COMPILES, and
# `_EXPECTED` holds every class it refused in on the way there at zero, so a
# reappearing one is named rather than counted. The censused route, in the
# order the layers surfaced:
#
# - Ten elementwise fused-region instantiations -- the decoder is where the
#   model's bare elementwise surface lives. The residual adds now mint
#   `mo.composite.gated_residual` with a unit gate and the projection bias
#   adds mint `mo.composite.ada_modulate` with a zero scale
#   (`UngatedResidualPattern` / `BiasAddPattern`, Fusion.cpp), both onto
#   signed kernels; the tail (the BN denormalize's sqrt and NCHW affine, the
#   fp32 output range map) runs as elementwise host walks
#   (`registrations.mojo`), the refusal text's own fallback route.
# - Two epilogue-fused `mo.matmul` instantiations, gone with the adds that
#   fed them.
# - The bare-metal offload wall ("target 'hexagon-unknown-unknown-elf' is
#   not supported by this build"), surfaced only at census-zero: the
#   thirty-three 3x3 convs' generic `builtin_kernels.conv.Conv` binding is
#   generated device code. Served since the `mo.conv` registration runs the
#   design's host im2col into the signed matmul (K zero-padded to the
#   selector's 64-element granularity where a narrow conv needs it).
_EXPECTED = {
    "elementwise on the on-device NPU": 0,
    "an epilogue cannot be fused into a pre-signed kernel": 0,
    "target 'hexagon-unknown-unknown-elf' is not supported by this build": 0,
}

# Refusals that must not appear at float16. The matmul gate DOES appear when
# the same graph is staged at bfloat16 (measured: two instantiations), which
# is what makes its absence here a real check. The mha gate is listed even
# though bfloat16 does NOT surface it from this rung (measured -- the
# attention's resolution sits behind the elementwise layer), and the last
# two have never been observed at either dtype: all three are here because a
# stub silently starting to refuse a shape it used to take is the failure
# this file would otherwise miss.
_ABSENT = (
    "mo.matmul on the on-device NPU: fp16 only",
    "mo.mha.no_cache on the on-device NPU: fp16 only",
    "is not densely row-major",
    "K must be a multiple of",
    "head dim must be",
)


def _assert_expected_refusals(
    record: dict[str, object],
    rung: tuple[tuple[int, ...], int, bool, str],
) -> None:
    """Names what broke when the pinned rung refuses, then fails it.

    The rung compiles; reaching this function at all is the regression. The
    zeroed `_EXPECTED` classes exist so a reappearing refusal names WHICH
    registration or rule broke before the unconditional count check below --
    a count is actionable where a wall of diagnostic is not.
    """
    diagnostic = str(record["diagnostic"])
    if rung != _PINNED_RUNG or _PACKED != 8:
        pytest.skip(
            f"{rung} at packed grid {_PACKED} refused; only {_PINNED_RUNG} "
            "at grid 8 carries assertions. See the printed record."
        )

    chains = diagnostic.count("error: function instantiation failed")
    for needle, count in _EXPECTED.items():
        assert diagnostic.count(needle) == count, (
            f"{needle!r} appears {diagnostic.count(needle)} times, expected "
            f"{count}. The set of things the NPU refuses in the VAE decode "
            "graph moved; say which registration or rule did it."
            f"\n{diagnostic}"
        )
    assert chains == sum(_EXPECTED.values()), (
        f"{chains} failing instantiations against {sum(_EXPECTED.values())} "
        f"pinned: a class this test does not name appeared.\n{diagnostic}"
    )
    for needle in _ABSENT:
        assert needle not in diagnostic, (
            f"{needle!r} refused at float16. Every gate in this list is one "
            "the bfloat16 staging hits and float16 is supposed to clear, so "
            f"this is a regression in the fp16 path.\n{diagnostic}"
        )


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


def test_flux_vae_decode_compiles_for_the_npu(virtual_npu: None) -> None:
    blocks = _blocks()
    layers_per_block = int(os.environ.get("FLUX_VAE_LAYERS", "2"))
    attn = os.environ.get("FLUX_VAE_ATTN", "1") == "1"
    dtype = os.environ.get("FLUX_DTYPE", "float16")
    stage_only = os.environ.get("FLUX_STAGE_ONLY", "") == "1"
    on_npu = os.environ.get("FLUX_DEVICE", "npu") == "npu"
    target = os.environ.get("FLUX_MEF_TARGET", "linux")
    dump = os.environ.get("FLUX_MEF_DUMP", "")

    if target == "windows":
        # Process-global and irreversible; its own bazel target, as every
        # cross-compile here is.
        set_virtual_cpu_target(_WINDOWS_ARM64_ELF)

    if os.environ.get("FLUX_IR_DUMP", "") == "1" and (
        out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    ):
        ir_dir = Path(out) / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODULAR_DEBUG"] = f"ir-output-dir={ir_dir}"

    device = DeviceRef.NPU() if on_npu else DeviceRef.CPU()
    record: dict[str, object] = {
        "blocks": list(blocks),
        "layers_per_block": layers_per_block,
        "mid_block_attention": attn,
        "dtype": dtype,
        "device": "npu" if on_npu else "cpu",
        "target": target,
        "packed_grid": _PACKED,
    }

    started = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph, fused = _stage(blocks, layers_per_block, attn, dtype, device)
        record["staging_warnings"] = sorted(
            {str(w.message)[:300] for w in caught}
        )
    except BaseException as exc:
        record["outcome"] = "STAGING_ERROR"
        record["error"] = _chain(exc)[:4000]
        _emit(record)
        raise
    record["stage_seconds"] = round(time.monotonic() - started, 1)
    graph_text = str(graph)
    record["inventory"] = _inventory(graph_text)
    # The per-weight consumer table (dispatch M/N/K per consuming op): what
    # the crouton selection and the bundler's pack-for-the-arm refusal are
    # computed from. Order 315: the six M=1 temb rows shipped crouton under
    # an op-type census; only consumer shapes can see that class.
    record["inventory"]["weights"] = weight_consumers(graph_text)
    _emit(record, tag="STAGED")

    if stage_only:
        record["outcome"] = "STAGED"
        _emit(record)
        return

    started = time.monotonic()
    try:
        compiled = InferenceSession(
            devices=[NPU() if on_npu else CPU()]
        ).compile(graph)
    except BaseException as exc:
        record["outcome"] = "REFUSED"
        record["compile_seconds"] = round(time.monotonic() - started, 1)
        # Untruncated: the compiler reports every failing instantiation in one
        # elaboration, so the diagnostic IS the census this test pins.
        record["diagnostic"] = _chain(exc)
        _emit(record)
        _assert_expected_refusals(
            record, (blocks, layers_per_block, attn, dtype)
        )
        return
    record["outcome"] = "COMPILED"
    record["compile_seconds"] = round(time.monotonic() - started, 1)

    if _EXPECTED and any(_EXPECTED.values()):
        _emit(record)
        pytest.fail(
            "the pinned rung compiled while `_EXPECTED` still pins refusals: "
            "the census was served -- move the pins to zero and keep the MEF "
            "assertions below as the rung's contract."
        )

    # The compile is the pin, and the MEF is its artifact: the markers say the
    # serialized model really carries the on-device kernel package, the npu
    # placement and this model's weights, which "compiled" alone does not.
    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
    mef = out / f"flux_vae_{_PACKED}_{dtype}_{target}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    marker_names = [
        "hexagon_ondevice",
        '{"label":"npu","id":0}',
        "decoder_bn_mean",
        "decoder.conv_in.weight",
    ]
    if target == "windows":
        marker_names.append("__chkstk")
    markers = {marker: marker.encode() in blob for marker in marker_names}
    record["mef_bytes"] = len(blob)
    record["mef_magic"] = blob[:4].hex()
    record["mef_markers"] = markers
    _emit(record)
    assert blob, "empty MEF"
    if on_npu:
        assert all(markers.values()), markers

    if dump:
        # The registry the device driver hands `M_initModel`: every external
        # constant of THIS graph, real checkpoint bytes, in the denoise
        # sibling's dump format (raw .bin per weight + MANIFEST.tsv with
        # per-weight sha256 for independent re-verification).
        assert target != "windows", (
            "FLUX_MEF_DUMP needs a process the windows retarget has not"
            " poisoned; run the dump with FLUX_MEF_TARGET=linux"
        )
        from test_flux_denoise_mef import _host_array

        started = time.monotonic()
        _load_real_weights(fused)
        wdir = Path(dump) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        total = 0
        for name, value in fused.state_dict().items():
            arr = np.ascontiguousarray(_host_array(value))
            fname = name.replace("/", "_") + ".bin"
            arr.tofile(wdir / fname)
            digest = hashlib.sha256(arr.tobytes()).hexdigest()
            rows.append(
                f"{name}\t{fname}\t{arr.nbytes}\t{arr.dtype}\t"
                f"{list(arr.shape)}\t{digest}"
            )
            total += arr.nbytes
        (wdir / "MANIFEST.tsv").write_text("\n".join(rows) + "\n")
        record["dump"] = {
            "weights": len(rows),
            "weight_bytes": total,
            "dump_seconds": round(time.monotonic() - started, 1),
        }
        _emit(record)


def _emit(record: dict[str, object], tag: str = "JSON") -> None:
    text = json.dumps(record, indent=2)
    print(f"###{tag}###" + json.dumps(record), flush=True)
    if out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        name = f"vae_{record['packed_grid']}_{record['dtype']}.json"
        (Path(out) / name).write_text(text)
