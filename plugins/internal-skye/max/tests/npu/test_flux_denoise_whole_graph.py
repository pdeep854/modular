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
"""Compiles a FLUX.2-Klein-4b denoise graph for the on-device NPU, at a depth.

The whole-graph question every other test in this directory approaches one op
at a time, and it turns out to be a *cheap* question rather than the expensive
one the record assumed: the full 5 + 20 graph, 2873 ops, reaches KGEN
elaboration in 46 seconds at 6.2 GiB. It refused there for the project's whole
life -- first on kernel coverage (the compiler reports every failing
instantiation in one elaboration, so a single run yields the whole ordered
list), then, with every kernel served, on the packed-ids buffer concat asking
for the bare-metal triple. Both surfaces are served, THE GRAPH NOW COMPILES
AND EXPORTS A MEF, and what this test pins is that compile: `_EXPECTED` holds
every once-refusing class at zero so a regression is named, and the MEF's
markers say the artifact really carries the on-device package.

Depth is a parameter because it discriminates. Eight of the twenty refusals
this file spent its life pinning were already present at one transformer
block; the `mo.matmul` epilogue refusal appeared only from 2 + 8 upward, so a
one-block attempt would have missed a class.

Two staging choices, each measured elsewhere and each load-bearing:

- **Batch stays symbolic, sequence lengths are pinned.** A literal-1 batch is
  squeezed into `mo.matmul` with `transpose_b` unset and the NPU stub refuses
  it, while the symbolic form compiles (`test_npu_graph_batch_matmul.py`).
  Re-measured for the view-folding lever (VIEW_FOLDING_SCOPE.md A2): at this
  graph's shapes the matmul hazard did not fire (its class is a rank-3 B
  input; a Linear's rank-2 weight keeps its transpose fold), but the pin
  squeezes unit dims through the preamble and the sinusoid-embed chains fall
  out of `SinusoidEmbedPattern`'s rank-2 anchor (mints 2 -> 0, four
  elementwise refusal chains) -- pinning `batch` needs that matcher taught
  the squeezed form first.
- **float16, not the model's bf16.** Both matmul stubs and `mo.mha.no_cache`
  are fp16-only by `comptime assert` and there is no `matmul_bf16` selector
  row to widen to, so bf16 cannot reach a matmul or the attention at all.
  `FLUX_DTYPE=bfloat16` runs it anyway, and the dtype refusals it produces are
  what make `_ABSENT` a real check rather than one that never fires.

The default device-placement policy is deliberate, where the rest of this
directory sets `DevicePlacementPolicy.Error`: this one wants to *observe* a
relocation rather than stop at the first, so it records the warnings. There
are none at any depth.

The record carried "over 17 GiB and still climbing" for this compile, and that
figure was never a measurement: both earlier attempts were killed by a
watchdog reading `MemAvailable`, which does not count this box's 48 GiB of
swap, so 11.1 and 17.7 GiB are lower bounds observed before a voluntary kill.
Run to completion, it is 6.2 GiB.

`FLUX_MEM_CAP_GIB` bounds the process's own address space for a local run. It
is off by default and should stay off: `RLIMIT_AS` covers the compiler's
thread-stack reservations too, so a value chosen against the graph's working
set dies as `LLVM ERROR: pthread_create failed` before any graph work happens.

Three knobs exist for the whole-graph run package, all defaulting to the
pinned behavior above so the gate's rung is untouched:

- `FLUX_IMAGE_SEQ` re-pins the image sequence length. The default stays 4096
  (1024x1024); the run package builds at 1024 (512x512, a 32x32 token grid)
  because that is the recorded render's geometry, and the recorded stage
  tensors are the only oracle the first device run has.
- `FLUX_MEF_TARGET=windows` retargets the host side to Windows ARM64 (ELF
  container), which is what the device driver executes. Its own process, as
  every cross-compile here is: `set_virtual_cpu_target` is global and has no
  reset.
- `FLUX_MEF_DUMP=dir` additionally writes the full weights registry -- every
  external constant the graph declares, real checkpoint bytes through the
  sibling probe's `_load_real_weights` route (which carries the
  `WeightData.astype` virtual-device workaround) -- with a manifest carrying
  each weight's dtype, shape and sha256. Linux invocation only: the windows
  retarget poisons the process for anything that follows it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import resource
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from flux_rope_fused_staging import rope_fused_requested, rope_fused_staging
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
from max.graph import DeviceRef, Graph, TensorType

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# FLUX.2-Klein-4B's transformer config at the locked revision, as
# `KLEIN_SIZE_VERIFY.md` records it. `model_config.py`'s defaults are the
# FLUX.2-dev ones and would stage a different model.
_KLEIN_4B = {
    "num_layers": 5,
    "num_single_layers": 20,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "joint_attention_dim": 7680,
    "guidance_embeds": False,
}

# One 1024x1024 image: the pipeline packs 16 pixels per token on each axis, so
# a side of 1024 is 64 tokens. `img_seq` is the reference-image stream, empty
# for text-to-image. Text is Qwen3's 512.
# The run package re-pins to the recorded render's 1024 (512x512).
#
# HELD 346 EXPERIMENT (this branch only): `batch` is pinned to 1, the A2 flip
# (VIEW_FOLDING_SCOPE.md; documented at this dict). Serve already runs batch 1
# (`flux_stager_inventory.SERVE_DIM_BINDINGS`: the bundle manifest pins
# batch_size = 1), so the pin adds compile-time knowledge and changes no
# runtime shape. Mainline keeps `batch` symbolic until the 346 A/B rules.
_IMAGE_SEQ = int(os.environ.get("FLUX_IMAGE_SEQ", str(64 * 64)))
_PINNED = {
    "batch": 1,
    "image_seq_len": _IMAGE_SEQ,
    "img_seq": 0,
    "text_seq_len": 512,
}

_WINDOWS_ARM64_ELF = "triple=aarch64-w64-windows-gnu-elf;cpu=generic"

_DEF = re.compile(r"^\s+(?:%\S+(?:, %\S+)*) = ([a-z][A-Za-z0-9_.]*)")
_SYMBOL = re.compile(r'symbol\s*=\s*"([^"]+)"')


class _UnusedAccelerator:
    """Stands in for the `Accelerator` every block's `Allreduce` opens.

    `Allreduce.__init__` opens one whether or not the model is sharded, so a
    host without a GPU cannot build the module at all. Every call site in
    `flux2.py` is guarded by `len(self.devices) > 1`, so on one device it is
    built and never used; `__getattr__` fails loudly rather than letting a call
    through silently.
    """

    def __init__(self, id: int) -> None:
        self.id = id

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"single-device staging reached Allreduce.{name}")


def _stage(
    layers: int, single: int, dtype: str, device: DeviceRef
) -> tuple[Graph, object]:
    from max.nn.comm import allreduce
    from max.pipelines.architectures.flux2.components.denoise_compute import (
        DenoiseComputeStep,
    )
    from max.pipelines.architectures.flux2.flux2 import Flux2Transformer2DModel
    from max.pipelines.architectures.flux2.model_config import Flux2Config

    allreduce.Accelerator = _UnusedAccelerator  # type: ignore[misc,assignment]

    shape = dict(_KLEIN_4B, num_layers=layers, num_single_layers=single)
    driver = NPU() if device != DeviceRef.CPU() else CPU()
    config = Flux2Config.initialize_from_config(shape, dtype, [driver])
    step = DenoiseComputeStep(
        transformer=Flux2Transformer2DModel(config),
        dtype=config.dtype,
        device=device,
    )
    # What `load_state_dict` does to a weight besides giving it data; without
    # it the external constants carry no fully-qualified name.
    for name, weight, _layer in step._iter_named_weights():
        weight.name = name

    pinned = [
        TensorType(t.dtype, [_PINNED.get(str(d), d) for d in t.shape], device)
        for t in step.input_types()
    ]
    # The fused-qk staging rewrite is the SAME object the mef stager applies
    # (flux_rope_fused_staging): this stager mints the production npu.mef,
    # and window 281 caught the rewrite living only in the probe module --
    # the fifth real-correct-unreachable-arm instance, one layer up.
    with (
        rope_fused_staging(step),
        Graph(
            "flux_denoise", input_types=pinned, custom_extensions=[_KERNELS]
        ) as graph,
    ):
        graph.output(step(*(v.tensor for v in graph.inputs), signal_buffers=[]))
    return graph, step


def _inventory(mlir: str) -> dict[str, object]:
    """Counts op kinds and records where each first appears.

    First appearance is the axis a refusal question needs: kernel resolution
    walks the graph, so if the compile reports one error it is the first kind
    that refuses, and everything after it stays invisible until that one is
    served.
    """
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


def _peak_rss_mib() -> float:
    """Peak resident set of this process and of any child it reaped.

    Children are added because a compile that forked a tool would otherwise
    report only the driver's own footprint, and the first honest peak for this
    compile is a result in its own right -- both earlier attempts were killed
    by a watchdog before one existed.
    """
    return (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    ) / 1024.0


def _cap_address_space(gib: float) -> None:
    """Bounds the process rather than the machine, when asked to.

    `RLIMIT_AS` caps *address space*, not resident memory, and the compiler's
    thread pool reserves a stack per thread out of the same budget -- a cap
    that looks generous against the 11.1 GiB this compile was measured at
    still dies as `pthread_create failed` before any graph work happens.
    So the cap has to clear the reservations, and zero disables it, which is
    the right setting when the action runs on a remote executor and this
    machine is not the thing at risk.
    """
    if gib <= 0:
        return
    limit = int(gib * (1 << 30))
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


# The rung the assertions below describe, and the cheapest one that carries
# both generated-code classes. The `mo.matmul` epilogue refusal used to first
# appear at 2+8; since the composites began claiming regions it appears here
# too (see below). The full graph is 5+20; deeper rungs are reachable through
# the environment and were measured, but none is worth a minute on every
# invocation.
_PINNED_RUNG = (1, 1, "float16")

# What the compile reports at that rung, as substrings of the diagnostic and
# the count of frame chains that end in each. The compiler reports EVERY
# failing instantiation in one elaboration rather than stopping at the first,
# so this is the whole refusal set rather than its first entry -- which is
# what makes pinning it meaningful.
#
# One cause, reached from a fused elementwise region (`mogg.call.foreach`) and
# `mo.range`: the elementwise hook both Hexagon parts share has no signed
# kernel to dispatch for a region nothing claimed, and this part cannot be
# handed a generated one. A materialized view (`mogg.call.materialize`) was a
# third route until the materializer resolved per device: `npu_view_copy` in
# `hexagon_ondevice/registrations.mojo` serves it as a host strided walk, so
# that chain compiles and the census dropped from 9 to 8 -- the other eight
# origins were measured unchanged when it did.
#
# The census then moved 8 -> 6 + 1 when `registrations.mojo` took over
# `mo.silu` (a signed-selector dispatch) and `mo.cast` (a host conversion),
# and PatternFusion began minting `mo.composite.*` from the mul/add pairs.
# Measured per chain, against the recorded 8-chain baseline (7 x
# `mogg.call.foreach` + 1 x `mogg.elemwise_for_each`, the #136 replication
# record): the two foreach chains anchored by the bare silu and the cast are
# gone -- served, not silenced; the disposition ladder places silu_bf16 on
# SIGNED_SELECTOR and the cast on the no-device-kernel rung -- five foreach
# chains and the elemwise_for_each survive unchanged, and ONE region now
# terminates at `Matmul`'s own epilogue refusal instead of the elementwise
# arm. That class is not new (it first appeared at 2+8 before the composites
# existed); the composites claiming a region's core is what leaves a lone
# neighbour op small enough for MOGG to try as a matmul epilogue at this
# rung, which the pre-signed kernel refuses by name.
#
# 6 + 1 then became 5 + 1 when the composite matchers began accepting the
# symbolic batch (and the shared parameter spreads) this staging carries.
# Measured from the 5+20 emitted module on both sides of that change, since
# the diagnostic caps at 20: the modulate and unclamped gated-residual
# classes are SERVED (`registrations.mojo`'s AdaModulate at 40 sites,
# GatedResidual at 15, their producing matmuls compiling bare), the split-off
# `1 + scale` chains died with their consumers, and no new class appeared --
# the plain gated rule declines a clamp-fed anchor, so the 25 clamped closes
# keep refusing as the SAME two epilogue-fused matmul classes as before (at
# this rung both share one instantiation, hence the count of 1). What remains
# on the elementwise arm is norm_out's modulate and its `1 + scale` (a rank-2
# parameter reshaped between the sum and the spread, which the rule does not
# yet see through) and the fp32 preamble families (sinusoid embedding, rope
# tables).
#
# The second entry is what stops this pinning a coincidence. The refusal used
# to come from the discrete card's `compile_function` three frames deeper, on
# an exact-equality triple comparison that happened to exclude this target; it
# now comes from the on-device arm. Asserting the card's message is *absent*
# is what would catch a change that reverted the decision to that accident,
# which the count alone cannot see.
# #166 (symbolic-batch matcher relaxation) served the modulate and gated
# chains, so two elementwise refusals left this rung and one more matmul
# surfaced its epilogue refusal from behind them.
#
# The epilogue class then went 2 -> 0 when `GatedResidualClampPattern` began
# minting `mo.composite.gated_residual_clamp` from the fp16-saturated closes
# the #166 guard had deliberately kept whole: at this rung both epilogue
# chains were those clamped closes (the single block's to_out and the double
# block's txt ff), so serving them compiles their matmuls bare and the class
# leaves the rung.
#
# The elementwise class went 3 -> 0 when PatternFusion began minting the two
# fp32 preamble table composites (`SinusoidEmbedPattern` and
# `RopeTablePattern` in GraphCompiler/lib/MODialect/Patterns/Fusion.cpp,
# dispatched by #156's `sinusoid_embed` / `rope_table` registrations). One
# attribution above is measured false at that base: norm_out's modulate was
# already minting (the base pre-mogg dump carries 7
# `mo.composite.ada_modulate` ops, norm_out's among them), so the three
# chains were the two preamble families' foreach classes plus the band's
# `mo.range` arm (.agentwork/flux-npu/preamble-minting/CENSUS.txt).
#
# With both landed the KERNEL-refusal set at this rung is EMPTY -- zero
# failing instantiations; `_EXPECTED` holds every known class to zero so a
# reappearing one is named rather than counted -- and the compile advanced
# to a stage no FLUX graph had reached, where it stopped on one exposed
# terminal: elaborating the fully-served module asked for the bare-metal
# device triple, which no engine build carries. The requester was the
# packed-ids join (`concat(txt_ids, latent_image_ids, 0)`), the one op the
# substituted sibling never stages: `ConcatInPlacePattern` claims an axis-0
# concat of squeezed inputs as `mgp.buffer.concat`, the borrow cannot fire
# on borrowed input buffers, and the surviving op's generic primitive is
# generated device code (measured by native stack trace: `compileOffloads`
# -> `isGPUTriple` -> the target-traits lookup). Served since the
# `mgp.buffer.concat.npu` host-copy registration, which
# `test_npu_graph_movement.py` pins in isolation -- and with it THE RUNG
# COMPILES, so the pin below is the compile itself plus the exported MEF's
# byte sanity. Any refusal at this rung is now a regression: `_EXPECTED`
# names the known classes first, and an unknown one fails the rung by count.
_EXPECTED = {
    "elementwise on the on-device NPU": 0,
    "an epilogue cannot be fused into a pre-signed kernel": 0,
    "compile_function[parametric indexed-closure] is NPU-only": 0,
    "target 'hexagon-unknown-unknown-elf' is not supported by this build": 0,
}

# Refusals that must not appear at float16. The three dtype gates DO appear
# when the same graph is staged at bfloat16, which is what makes their absence
# here a measured pass rather than a check that never fires; the last two have
# never been observed at either dtype and are here because a stub silently
# starting to refuse a shape it used to take is the failure this file would
# otherwise miss.
_ABSENT = (
    "mo.matmul on the on-device NPU: fp16 only",
    "mo.reduce.rms_norm on the on-device NPU: fp16 only",
    "mo.mha.no_cache on the on-device NPU: fp16 only",
    "is not densely row-major",
    "K must be a multiple of",
    "head dim must be",
)


def _assert_expected_refusals(
    record: dict[str, object], layers: int, single: int, dtype: str
) -> None:
    """Names what broke when the pinned rung refuses, then fails it.

    The rung compiles; reaching this function at all is the regression. The
    zeroed `_EXPECTED` classes exist so a reappearing refusal names WHICH
    registration or rule broke before the unconditional failure below --
    a count is actionable where a wall of diagnostic is not.
    """
    diagnostic = str(record["diagnostic"])
    if (layers, single, dtype) != _PINNED_RUNG:
        pytest.skip(
            f"{layers}+{single} at {dtype} refused; only "
            f"{_PINNED_RUNG} carries assertions. See the printed record."
        )

    chains = diagnostic.count("error: function instantiation failed")
    for needle, count in _EXPECTED.items():
        assert diagnostic.count(needle) == count, (
            f"{needle!r} appears {diagnostic.count(needle)} times, expected "
            f"{count}. The set of things the NPU refuses in a whole FLUX "
            "denoise graph moved; say which registration or rule did it."
            f"\n{diagnostic}"
        )
    assert chains == 0, (
        f"{chains} failing instantiations in classes this test does not "
        f"name appeared.\n{diagnostic}"
    )
    for needle in _ABSENT:
        assert needle not in diagnostic, (
            f"{needle!r} refused at float16. Every gate in this list is one "
            "the bfloat16 staging hits and float16 is supposed to clear, so "
            f"this is a regression in the fp16 path.\n{diagnostic}"
        )
    pytest.fail(
        "the pinned rung refused in a class none of the pins name -- it has "
        f"compiled since the buffer-concat registration.\n{diagnostic}"
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


def test_flux_denoise_compiles_for_the_npu(virtual_npu: None) -> None:
    layers = int(os.environ.get("FLUX_LAYERS", "1"))
    single = int(os.environ.get("FLUX_SINGLE", "1"))
    dtype = os.environ.get("FLUX_DTYPE", "float16")
    cap_gib = float(os.environ.get("FLUX_MEM_CAP_GIB", "0"))
    stage_only = os.environ.get("FLUX_STAGE_ONLY", "") == "1"
    on_npu = os.environ.get("FLUX_DEVICE", "npu") == "npu"
    target = os.environ.get("FLUX_MEF_TARGET", "linux")
    dump = os.environ.get("FLUX_MEF_DUMP", "")

    if target == "windows":
        # Process-global and irreversible, exactly as the sibling probe
        # records; the dump leg below refuses to share a process with it.
        set_virtual_cpu_target(_WINDOWS_ARM64_ELF)

    # The refusal chains name generated functions only by line number in a
    # module nobody can read (`<unknown>:13428`), so enumerating what a fused
    # region actually computes needs the compiler's own stage dumps. Dumping
    # into TEST_UNDECLARED_OUTPUTS_DIR is what lets a remote execution bring
    # them home.
    if os.environ.get("FLUX_IR_DUMP", "") == "1" and (
        out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    ):
        ir_dir = Path(out) / "ir"
        ir_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODULAR_DEBUG"] = f"ir-output-dir={ir_dir}"

    device = DeviceRef.NPU() if on_npu else DeviceRef.CPU()
    record: dict[str, object] = {
        "layers": layers,
        "single": single,
        "dtype": dtype,
        "device": "npu" if on_npu else "cpu",
        "target": target,
        "image_seq_len": _IMAGE_SEQ,
        "address_space_cap_gib": cap_gib,
    }

    _cap_address_space(cap_gib)

    started = time.monotonic()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph, step = _stage(layers, single, dtype, device)
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
    record["peak_rss_mib_after_stage"] = round(_peak_rss_mib(), 1)
    _emit(record, tag="STAGED")

    # The temb-projection fusion's staged census, pinned where the compile
    # rung is pinned: ONE fused silu+matmul (`Flux2TembProjections`, the
    # [52224, 3072] row at full width) replaces the four temb-consumer
    # projections and their four silus, so this rung stages 20 matmuls and
    # 5 silus where it staged 23 and 8. A drift here is a model-staging
    # change and must arrive with a new census, not ride in silently.
    if (layers, single, dtype) == _PINNED_RUNG:
        kinds: Counter[str] = Counter()
        for line in graph_text.splitlines():
            if m := _DEF.match(line):
                kinds[m.group(1)] += 1
        assert kinds["rmo.matmul"] == 20, kinds["rmo.matmul"]
        assert kinds["rmo.mo.silu"] == 5, kinds["rmo.mo.silu"]
        assert "transformer.temb_proj.linear.weight" in graph_text
        assert "double_stream_modulation" not in graph_text, (
            "a split temb projection came back"
        )

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
        # elaboration rather than stopping at the first, so the diagnostic *is*
        # the ordered list this test exists to produce. A cap on it silently
        # shortens that list, which reads as a smaller problem -- measured: a
        # 20,000-character cap turned twenty refusals into ten.
        record["diagnostic"] = _chain(exc)
        record["peak_rss_mib"] = round(_peak_rss_mib(), 1)
        _emit(record)
        _assert_expected_refusals(record, layers, single, dtype)
        return
    record["outcome"] = "COMPILED"
    record["compile_seconds"] = round(time.monotonic() - started, 1)
    record["peak_rss_mib"] = round(_peak_rss_mib(), 1)

    # The compile is the pin, and the MEF is its artifact: the markers say
    # the serialized model really carries the on-device kernel package and
    # the npu placement, which "compiled" alone does not.
    out = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "/tmp"))
    mef = out / f"flux_denoise_{layers}_{single}_{dtype}_{target}.mef"
    compiled.export_mef(mef)
    blob = mef.read_bytes()
    marker_names = [
        "hexagon_ondevice",
        '{"label":"npu","id":0}',
        "sinusoid_embed",
        "rope_table",
        "transformer.transformer_blocks.0.attn.to_q.weight",
        "transformer.temb_proj.linear.weight",
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
        # constant of THIS graph, real checkpoint bytes. The sibling probe's
        # loader carries the two hazards already measured there: the
        # `WeightData.astype` metadata-only cast under virtual-device mode
        # (dodged by casting in numpy against each staged dtype), and the
        # windows retarget poisoning the process (refused above). The
        # per-weight sha256 is what lets the registry be re-verified against
        # the checkpoint by an independent reader.
        assert target != "windows", (
            "FLUX_MEF_DUMP needs a process the windows retarget has not"
            " poisoned; run the dump with FLUX_MEF_TARGET=linux"
        )
        from test_flux_denoise_mef import _host_array, _load_real_weights

        started = time.monotonic()
        _load_real_weights(step)
        wdir = Path(dump) / "weights"
        wdir.mkdir(parents=True, exist_ok=True)
        rows: list[str] = []
        total = 0
        for name, value in step.state_dict().items():
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
            "peak_rss_mib_after_dump": round(_peak_rss_mib(), 1),
        }
        _emit(record)


# The depth-1 fuse-views census at the pinned rung, measured from the
# `.mogg-pre-mgp.mlir` dump (post `mogg-fuse-views`): 71 materializes are
# emitted at mo-to-mogg, 32 survive folding, and 3 slices raise to zero-copy
# `mogg.slice` (the symbolic-batch [0:1] slices). Each surviving materialize
# is an `npu_view_copy` host walk plus a matmul-chain flush, so this census is
# the view-folding lever's own falsifier (VIEW_FOLDING_SCOPE.md section 5):
# a drop is a win to re-pin, a rise is a regression to name. 33 -> 32 is the
# A0 pitched-view crossing: the single-block MLP-half slice
# ([b,4608,27648] -> [b,4608,18432]) crosses into the opted-in swiglu as a
# view instead of an npu_view_copy. 32/3 -> 11/0 is the batch pin
# (_PINNED["batch"]=1, order 348): with batch literal the modulation and
# sequence view chains fold outright and the three symbolic-batch [0:1]
# slices vanish with the symbol.
_DEPTH1_MATERIALIZED = 11
_DEPTH1_ZERO_COPY = 0


def test_depth1_fuse_views_census_is_pinned(
    virtual_npu: None, tmp_path: Path
) -> None:
    """Counts the depth-1 views that survive fuse-views, exactly.

    IR, not compile success, is the instrument (see
    `npu_slice_materialization_probe`): the count surviving into
    `.mogg-pre-mgp.mlir` is what this graph will host-copy per step.
    """
    if _IMAGE_SEQ != 4096:
        pytest.skip("census pinned at the default 4096 geometry")
    if rope_fused_requested():
        pytest.skip("FLUX_ROPE_FUSED_NORM restages the norm-rope chains")
    if os.environ.get("FLUX_MEF_TARGET", "linux") != "linux":
        pytest.skip("the windows retarget poisons this process")

    graph, _ = _stage(1, 1, "float16", DeviceRef.NPU())
    session = InferenceSession(devices=[NPU()])
    session.debug.ir_output_dir = str(tmp_path)
    session.compile(graph)

    dumps = list(tmp_path.glob("*.mogg-pre-mgp.mlir"))
    assert dumps, "the compile wrote no fuse-views dump"
    text = "".join(d.read_text() for d in dumps)
    materialized = text.count("mogg.call.materialize")
    zero_copy = text.count("mogg.slice")
    assert (materialized, zero_copy) == (
        _DEPTH1_MATERIALIZED,
        _DEPTH1_ZERO_COPY,
    ), (
        f"the depth-1 view census moved: {materialized} materialize /"
        f" {zero_copy} mogg.slice, pinned"
        f" {_DEPTH1_MATERIALIZED}/{_DEPTH1_ZERO_COPY}. A drop is a win --"
        " re-pin it with the change that bought it; a rise re-buys"
        " npu_view_copy host walks and chain flushes."
    )


def _emit(record: dict[str, object], tag: str = "JSON") -> None:
    text = json.dumps(record, indent=2)
    print(f"###{tag}###" + json.dumps(record), flush=True)
    if out := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        name = (
            f"flux_{record['layers']}_{record['single']}_{record['dtype']}.json"
        )
        (Path(out) / name).write_text(text)
