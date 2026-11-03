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
"""Compiles the FLUX.2-Klein Qwen3 text encoder for the on-device NPU.

`test_flux_denoise_whole_graph.py` answered the whole-graph question for the
denoise transformer; this file asks it for the encoder leg, staged from the
same model code the Klein pipeline runs: `Qwen3TextEncoderTransformer`, the
module `Qwen3TextEncoderKleinModel` compiles (`qwen3/text_encoder/model.py`),
at the klein-4B checkpoint's dimensions (hidden 2560, 32:8 GQA heads of 128,
SwiGLU 9728, theta 1e6 half-rotated rope, vocab 151936 -- read from
`text_encoder/config.json` at the locked revision). The boundary is the
recorded render's: `tokenize.*` inputs (tokens ``[S]`` int64, additive
causal+padding bias ``[1,1,S,S]`` float32) to `encoder.out`
``[1, S, len(select) * hidden]``.

Depth and the hidden-state selection are parameters because Klein's contract
is layered: 36 checkpoint blocks, 27 run (early-exit at the highest selected
state), and `encoder.out` is the states after blocks 9/18/27 concatenated.
The pinned rung stages one block with the embedding and its output selected,
which carries every op kind the full stack stages -- the embedding gather, the
block (norms, projections, rope application, masked attention, SwiGLU,
residual adds) and the select/stack/merge close -- without the 27x compile
cost. `FLUX_ENC_LAYERS=27` reaches the real depth and derives Klein's 9/18/27
selection.

THE RUNG COMPILES AND EXPORTS A MEF. It did not when this file landed: the
kernel map's "every encoder op kind already served" missed four layered
classes, each surfaced only once the previous was closed -- the attention
(`mo.composite.masked_flash_attention_gpu`, claimed only by the GPU-only
builtin until the `mha_masked` selector got its registration), a SwiGLU
product folded into a matmul epilogue, the raw rope-application elementwise
chains (`max.nn.RotaryEmbedding.__call__`'s half-rotated spelling, host
walks today; the signed `rope_half` selector is the named phase-2 upgrade),
and the embedding gather (a host row-copy, as the P1 design specifies).
`_EXPECTED` holds every once-refusing class at zero so a regression is
named, and the MEF's markers say the artifact really carries the on-device
package.

Staging choices, each mirrored from the denoise siblings:

- **Sequence length is pinned, 512 by default** -- the padded static S the
  Klein tokenizer emits (`FLUX2_TEXT_SEQ_LEN`). The encoder has no batch
  axis: tokens are rank-1 and every matmul runs ``[S, K]``-shaped, so the
  literal-1-batch matmul squeeze the denoise staging dodges cannot arise.
- **float16, not the checkpoint's bf16** -- the platform is fp16-native
  (hb#294); `FLUX_DTYPE=bfloat16` stages anyway and its dtype refusals are
  what keep `_ABSENT` a real check.
- **The rope table stays in-graph.** `RotaryEmbedding.freqs_cis` stages its
  range/outer/cos/sin preamble on the host CPU, a constant-fold surface the
  windows retarget used to poison; since the fold sub-model compiles for the
  executing host (#185, pinned by `test_npu_graph_constfold_windows.py`) the
  table folds under `FLUX_MEF_TARGET=windows` too, so nothing is substituted.
  `FLUX_ENC_MAX_SEQ` bounds the folded table's row count: the checkpoint's
  40960 stages a [81920, 64] fp32 table, real but pointless to fold on every
  gate run against a 512-token sequence, so the default pins 1024 (>= 2*S)
  and the run package can carry the checkpoint value.

`FLUX_MEF_TARGET=windows` retargets the host side to Windows ARM64 in its own
process, exactly as every cross-compile here is; `FLUX_STAGE_ONLY=1` stops
after staging and the inventory.

`FLUX_MEF_DUMP=dir` additionally writes the full weights registry -- every
external constant the graph declares, real checkpoint bytes from
`FLUX_ENC_WEIGHTS` (a safetensors file or a directory of shards) -- with the
same manifest format as the denoise sibling's dump leg: each weight's dtype,
shape and sha256. Linux invocation only: the windows retarget poisons the
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

# The Qwen3 side of FLUX.2-Klein-4B at the locked revision
# (`text_encoder/config.json`, HF SHA e7b7dc2, `hf-repo-lock.tsv`). The config
# class defaults are Qwen3-8B's and would stage a different model: klein's
# hidden is 2560 (so `encoder.out` is [1, S, 3*2560] = the DiT's
# `joint_attention_dim` 7680) and its MLP is 9728 wide.
_KLEIN_QWEN3 = {
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "vocab_size": 151936,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
}

# The padded static sequence the Klein tokenizer emits (`FLUX2_TEXT_SEQ_LEN`).
_TEXT_SEQ = int(os.environ.get("FLUX_TEXT_SEQ", "512"))

# Bounds the folded rope table ([2*max_seq, head_dim//2] fp32), not the model:
# the table is sliced to [0:S] before use, so any value >= _TEXT_SEQ stages
# the same consumed values. The checkpoint's 40960 is reachable for a run
# package that wants the exported table byte-identical to the real model's.
_MAX_SEQ = int(os.environ.get("FLUX_ENC_MAX_SEQ", "1024"))

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_DEF = re.compile(r"^\s+(?:%\S+(?:, %\S+)*) = ([a-z][A-Za-z0-9_.]*)")
_SYMBOL = re.compile(r'symbol\s*=\s*"([^"]+)"')


def _select(layers: int) -> list[int]:
    """Derives the hidden-state selection Klein's contract implies at a depth.

    At the real 27 this is Klein's pinned (9, 18, 27) -- thirds, in the HF
    convention where state ``i`` is the output of block ``i-1``. Below three
    blocks there is no third to take, so the selection is every state
    including the embedding (state 0), which keeps the multi-state
    stack/merge close staged at every rung.
    """
    if raw := os.environ.get("FLUX_ENC_SELECT", ""):
        return [int(part) for part in raw.split(",")]
    if layers >= 3:
        return [(i * layers) // 3 for i in (1, 2, 3)]
    return list(range(layers + 1))


def _stage(
    layers: int, select: list[int], dtype: str, device: DeviceRef
) -> tuple[Graph, object]:
    from max.dtype import DType
    from max.graph import TensorType
    from max.pipelines.architectures.qwen3.text_encoder.model_config import (
        Qwen3TextEncoderConfig,
    )
    from max.pipelines.architectures.qwen3.text_encoder.qwen3 import (
        Qwen3TextEncoderTransformer,
    )

    config = Qwen3TextEncoderConfig(
        **_KLEIN_QWEN3,
        num_hidden_layers=layers,
        max_seq_len=_MAX_SEQ,
        dtype=DType[dtype],
        device=device,
        hidden_state_layers=select,
    )
    model = Qwen3TextEncoderTransformer(config)
    # What `load_state_dict` does to a weight besides giving it data; without
    # it the external constants carry no fully-qualified name.
    for name, weight, _layer in model._iter_named_weights():
        weight.name = name

    pinned = [
        TensorType(
            t.dtype,
            [_TEXT_SEQ if str(d) == "total_seq_len" else d for d in t.shape],
            device,
        )
        for t in model.input_types()
    ]
    with Graph(
        "qwen3_text_encoder", input_types=pinned, custom_extensions=[_KERNELS]
    ) as graph:
        graph.output(*model(*(v.tensor for v in graph.inputs)))
    return graph, model


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


def _weight_paths(spec: str) -> list[Path]:
    """Expands `FLUX_ENC_WEIGHTS` to safetensors paths: a file names itself, a
    directory names its (sorted) shards -- the text encoder ships two."""
    path = Path(spec)
    if path.is_dir():
        shards = sorted(path.glob("*.safetensors"))
        assert shards, f"no *.safetensors under {path}"
        return shards
    assert path.is_file(), f"FLUX_ENC_WEIGHTS does not exist: {path}"
    return [path]


def _load_real_weights(model: object) -> None:
    """Loads checkpoint weights and casts each to its staged dtype, for real.

    The denoise sibling's route (`test_flux_denoise_mef._load_real_weights`),
    with the encoder's own name mapping: the checkpoint-to-module renames the
    pipeline applies (`Qwen3TextEncoderKleinModel._state_dict`), then the
    staged-name filter that drops the 9 checkpoint blocks past Klein's
    early-exit depth and the unused final norm / lm_head. The dtype cast
    happens in numpy against each staged weight's declared dtype, because
    `WeightData.astype` under virtual-device mode changes only the metadata.
    """
    from max.graph.weights import load_weights
    from max.pipelines.architectures.qwen3.text_encoder.weight_adapters import (
        QWEN3_TEXT_ENCODER_SAFETENSOR_MAP,
    )
    from test_flux_denoise_mef import _host_array

    spec = os.environ.get("FLUX_ENC_WEIGHTS", "")
    assert spec, (
        "FLUX_MEF_DUMP needs FLUX_ENC_WEIGHTS: the text_encoder checkpoint"
        " (a safetensors file or a directory of shards)"
    )
    staged = {
        name: weight.dtype for name, weight, _l in model._iter_named_weights()
    }
    state_dict: dict[str, np.ndarray] = {}
    for key, value in load_weights(_weight_paths(spec)).items():
        name = key
        for before, after in QWEN3_TEXT_ENCODER_SAFETENSOR_MAP.items():
            name = name.replace(before, after)
        name = name.removeprefix("language_model.")
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
    model.load_state_dict(state_dict, weight_alignment=1)


# The rung the census below describes: one block, the embedding and the
# block's output selected, fp16, S=512. Deeper rungs are reachable through the
# environment; the classes were measured stable from here (an encoder block is
# one repeated shape, so depth multiplies sites, not instantiations).
_PINNED_RUNG = (1, "float16")

# What the compile reports at that rung: nothing -- THE RUNG COMPILES, and
# `_EXPECTED` holds every class it refused in on the way there at zero, so a
# reappearing one is named rather than counted. The censused route, in the
# order the layers surfaced (each close exposed the next -- the mef sibling's
# layering lesson, measured again here):
#
# - "only valid on GPUs": `masked_flash_attention_gpu` stages
#   `mo.composite.masked_flash_attention_gpu`, which only the GPU-only
#   builtin claimed; served since `registrations.mojo`'s
#   `MaskedFlashAttention` dispatches the signed `mha_masked` selector with
#   the encoder's own rank-3 fp32 bias.
# - The matmul epilogue: the SwiGLU product (`silu(gate) * up` over two
#   separate projections, a spelling `SwiGLUPattern`'s packed-halves matcher
#   does not take) folded into the up-projection's epilogue. Served since
#   the `mo.mul` host walk claims the product before fusion; the residual
#   adds that would fold the same way mint `mo.composite.gated_residual`
#   with a unit gate (`UngatedResidualPattern`, Fusion.cpp) onto the signed
#   kernel instead.
# - The elementwise arm: the raw rope-application mul/sub chains
#   (`RotaryEmbedding.__call__`'s half-rotated spelling), served by the
#   elementwise host walks. Moving them onto the signed `rope_half` selector
#   is the named phase-2 upgrade, not a correctness gap.
# - The embedding gather, surfaced only once the elementwise layer was
#   served: a host row-copy, exactly as the P1 design names it.
_EXPECTED = {
    "only valid on GPUs": 0,
    "an epilogue cannot be fused into a pre-signed kernel": 0,
    "elementwise on the on-device NPU": 0,
}

# Refusals that must not appear at float16. The matmul gate DOES appear when
# the same graph is staged at bfloat16 (measured: it replaces the epilogue
# chain, whose dtype assert precedes the epilogue one), which is what makes
# its absence here a real check; the last two have never been observed and
# are here because a stub silently starting to refuse a shape it used to
# take is the failure this file would otherwise miss.
_ABSENT = (
    "mo.matmul on the on-device NPU: fp16 only",
    "is not densely row-major",
    "K must be a multiple of",
    "head dim must be",
)


def _assert_expected_refusals(
    record: dict[str, object], layers: int, dtype: str
) -> None:
    """Names what broke when the pinned rung refuses, then fails it.

    The rung compiles; reaching this function at all is the regression. The
    zeroed `_EXPECTED` classes exist so a reappearing refusal names WHICH
    registration or rule broke before the unconditional count check below --
    a count is actionable where a wall of diagnostic is not.
    """
    diagnostic = str(record["diagnostic"])
    if (layers, dtype) != _PINNED_RUNG:
        pytest.skip(
            f"{layers} layers at {dtype} refused; only {_PINNED_RUNG} carries "
            "assertions. See the printed record."
        )

    chains = diagnostic.count("error: function instantiation failed")
    for needle, count in _EXPECTED.items():
        assert diagnostic.count(needle) == count, (
            f"{needle!r} appears {diagnostic.count(needle)} times, expected "
            f"{count}. The set of things the NPU refuses in the Qwen3 encoder "
            "moved; say which registration or rule did it."
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


def test_flux_encoder_compiles_for_the_npu(virtual_npu: None) -> None:
    layers = int(os.environ.get("FLUX_ENC_LAYERS", "1"))
    select = _select(layers)
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
        "layers": layers,
        "select": select,
        "dtype": dtype,
        "device": "npu" if on_npu else "cpu",
        "target": target,
        "text_seq_len": _TEXT_SEQ,
        "max_seq_len": _MAX_SEQ,
    }

    started = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph, model = _stage(layers, select, dtype, device)
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
        _assert_expected_refusals(record, layers, dtype)
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
    mef = out / f"flux_encoder_{layers}_{_TEXT_SEQ}_{dtype}_{target}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    marker_names = [
        "hexagon_ondevice",
        '{"label":"npu","id":0}',
        "embed_tokens.weight",
        "layers.0.self_attn.q_proj.weight",
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
        _load_real_weights(model)
        wdir = Path(dump) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        total = 0
        for name, value in model.state_dict().items():
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
        name = f"encoder_{record['layers']}_{record['dtype']}.json"
        (Path(out) / name).write_text(text)
