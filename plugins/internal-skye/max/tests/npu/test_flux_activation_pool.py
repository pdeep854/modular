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
"""Sizes the FLUX.2-Klein-4b denoise graph's single activation allocation.

MAX's memory planner collapses every device activation buffer in the execute
block into ONE allocation sized to the whole peak, each activation a bulk-slice
offset into it (`mgp-dynamic-memory-planning`). That allocation is the only
thing in a FLUX graph that could plausibly cross the NPU transport's 2 GiB
per-allocation cap (`QuRTDeviceContext.cpp` `chunkAlloc`), because weights are
hoisted into the init block before planning runs and each sits alone in an
allocation whose largest is 306 MiB (the fused temb projection).

**This stages the graph and never compiles it, because a compile cannot yield
the pool number on any machine available to us.** A compile stops where its
device target stops it. For the NPU it reaches KGEN elaboration in 46 s at
6.3 GiB and fails on the twenty pinned refusals -- that compile is
`test_flux_denoise_whole_graph`'s job. For the CPU it dies at elaboration on
attention's GPU-only constraint. And for a virtual GPU -- the one target where
nothing refuses, so the only compile that could run the planner for real -- it
has never completed anywhere: eight remote attempts across three depths were
OOM-killed between 48.7 and 59.5 GiB (a 1+1-block graph costs what the full
5+20 does), including under a memory declaration the scheduler accepted. The
planner's own high watermark is not a compile-time constant anyway --
`mgp.buffer.plan` yields it as an SSA value and the greedy allocator runs at
execution time -- so the answer is reconstructed here from the staged graph's
tensor types and def-use order, running the same greedy algorithm the
primitive runs
(`max/kernels/src/graph_compiler/builtin_primitives/buffer_plan.mojo`).

Three numbers per variant, which is what makes this a bracket rather than a
guess. `max_live` is every buffer alive at once at the busiest point, so no
allocator can beat it and it is a hard lower bound. `greedy` is MAX's own
algorithm over those buffers. `requested` lays every buffer end to end, the
pool's value if no two lifetimes were ever disjoint.

WHERE THIS IS AND IS NOT FAITHFUL. It reads the graph before lowering, so the
buffers are the ones the graph asks for rather than the ones a backend
allocates. Fusion removes temporaries, which makes this an over-estimate; a
kernel's private workspace is invisible, which makes it an under-estimate for
that kernel. Views are excluded by op name rather than by the lowering's
judgement, so the second variant counts them as allocating too and brackets
that choice. Weights and inline constants are excluded because
`ExecInvariantCodeMotionPass` hoists both into the init block before planning,
and the planner scans the execute block only.

The staging is validated by a quantity it did not set out to measure: the
external constants it recovers are 166 weights totalling 7,751,089,152 bytes
with a largest of 320,864,256 (the [52224, 3072] fused temb projection; the
byte total is unchanged from the 169-row census two earlier and unrelated
paths agreed on -- the temb fusion moves rows, not bytes).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType
from max.nn.comm import allreduce
from max.pipelines.architectures.flux2.components.denoise_compute import (
    DenoiseComputeStep,
)
from max.pipelines.architectures.flux2.flux2 import Flux2Transformer2DModel
from max.pipelines.architectures.flux2.model_config import Flux2Config

# The transport's per-allocation cap: `chunkAlloc` refuses a byteSize above it,
# and the dispatch path narrows the same value into a 32-bit length field.
_CAP = 0x7FFFFFFF

# FLUX.2-Klein-4B's transformer config, as `KLEIN_SIZE_VERIFY.md` records it
# against the locked revision e7b7dc27f91deacad38e78976d1f2b499d76a294. The
# non-Klein defaults in `model_config.py` are the FLUX.2-dev ones and would
# stage a different model.
_KLEIN_4B = {
    "num_layers": 5,
    "num_single_layers": 20,
    "num_attention_heads": 24,
    "attention_head_dim": 128,
    "joint_attention_dim": 7680,
    "guidance_embeds": False,
}

# One 1024x1024 image. The pipeline packs 16 pixels per token on each axis (VAE
# stride 8 times the 2x2 latent pack), so a side of 1024 is 64 tokens and the
# image carries 4096. `img_seq` is the reference-image stream, empty for
# text-to-image. Text is Qwen3's 512.
_EXTENTS = {
    "batch": 1,
    "image_seq_len": 64 * 64,
    "img_seq": 0,
    "text_seq_len": 512,
}

# The one extent the printed types leave symbolic: the final output slice takes
# its length from shape tensors the graph builds rather than from its static
# type. It slices the 4096-token image stream, so 4096 bounds it, and
# substituting the bound keeps every number here an over-estimate.
_SYMBOLIC_BOUND = 4096

_DTYPE_BYTES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "f64": 8,
    "si8": 1,
    "ui8": 1,
    "si16": 2,
    "ui16": 2,
    "si32": 4,
    "ui32": 4,
    "si64": 8,
    "ui64": 8,
    "i1": 1,
    "bool": 1,
}

# Ops whose result is a view or pure metadata rather than a fresh buffer, from
# the census's own "no kernel" and "movement" columns. The two constant kinds
# are excluded for a different reason: both are execution-invariant and get
# hoisted out of the block the planner scans.
_HOISTED = {"mo.constant", "mo.constant.external"}
_VIEWS = {
    "rmo.reshape",
    "mo.static.reshape",
    "rmo.mo.transpose",
    "rmo.broadcast_to",
    "rmo.shape_to_tensor",
    "rmo.mo.squeeze",
    "rmo.mo.unsqueeze",
}

_DEF = re.compile(r"^\s+(%\S+(?:, %\S+)*) = ([a-z][A-Za-z0-9_.]*)")
_SSA = re.compile(r"%[A-Za-z0-9_#]+")
_TENSOR = re.compile(r"!mo\.tensor<\[([^\]]*)\], *([A-Za-z0-9_]+)")


class _UnusedAccelerator:
    """Stands in for the `Accelerator` every block's `Allreduce` constructs.

    `Allreduce.__init__` opens a real accelerator whether or not the model is
    sharded, so a GPU-less host cannot build the module at all. Every call site
    in `flux2.py` is guarded by `len(self.devices) > 1`, so on one device the
    object is built and never used; `__getattr__` fails loudly rather than
    letting a call through silently.
    """

    def __init__(self, id: int) -> None:
        self.id = id

    def __getattr__(self, name: str) -> object:
        raise AssertionError(
            f"single-device staging reached Allreduce.{name}; the plan would "
            "no longer be the unsharded graph's"
        )


def _tensor_bytes(dims: str, dtype: str, symbolic: dict[str, int]) -> int:
    width = _DTYPE_BYTES[dtype]
    count = 1
    for d in (d.strip() for d in dims.split(",") if d.strip()):
        if d.lstrip("-").isdigit():
            count *= int(d)
        else:
            symbolic[d] = symbolic.get(d, 0) + 1
            count *= _SYMBOLIC_BOUND
    return count * width


def _result_bytes(tail: str, symbolic: dict[str, int]) -> list[int]:
    return [_tensor_bytes(d, t, symbolic) for d, t in _TENSOR.findall(tail)]


def _greedy_pool(sizes: list[int], live: list[tuple[int, int]]) -> int:
    """Replays `buffer_plan.mojo`'s greedy allocator.

    Blocks sit at one common alignment, as `allocate_new_block` lays them; an
    allocation takes the smallest existing block that is large enough and that
    every current member's lifetime is disjoint from, else a fresh block at the
    end.
    """
    blocks: list[tuple[int, set[int]]] = []  # size, members
    pool = 0

    def disjoint(a: int, b: int) -> bool:
        return live[a][1] < live[b][0] or live[b][1] < live[a][0]

    for i, size in enumerate(sizes):
        best, best_size = -1, 0
        for b, (bsize, members) in enumerate(blocks):
            if size > bsize or (best >= 0 and bsize >= best_size):
                continue
            if all(disjoint(i, m) for m in members):
                best, best_size = b, bsize
        if best >= 0:
            blocks[best][1].add(i)
            continue
        offset = (pool + 63) // 64 * 64
        blocks.append((size, {i}))
        pool = offset + size
    return pool


def _max_live(sizes: list[int], live: list[tuple[int, int]]) -> tuple[int, int]:
    events: list[tuple[int, int]] = []
    for i, size in enumerate(sizes):
        events.append((live[i][0], size))
        events.append((live[i][1] + 1, -size))
    events.sort()
    current = peak = at = 0
    for index, delta in events:
        current += delta
        if current > peak:
            peak, at = current, index
    return peak, at


def _stage(layers: int, single: int) -> str:
    allreduce.Accelerator = _UnusedAccelerator  # type: ignore[misc,assignment]
    # `ops.custom` adds the op and only then verifies it against the kernel
    # library. Skipping the verifier leaves the op, its symbol and its result
    # types exactly as the caller built them, which is all this reads -- but it
    # does mean the graph is unverified, so this sizes buffers and proves
    # nothing about whether the kernels resolve.
    graph_module = sys.modules["max.graph.graph"]
    graph_module.KernelLibrary.verify_custom_op = (  # type: ignore[method-assign]
        lambda self, custom_op: None
    )

    shape = dict(_KLEIN_4B, num_layers=layers, num_single_layers=single)
    config = Flux2Config.initialize_from_config(shape, "bfloat16", [CPU()])
    assert config.dtype == DType.bfloat16, config.dtype
    ref = DeviceRef.CPU()
    step = DenoiseComputeStep(
        transformer=Flux2Transformer2DModel(config),
        dtype=config.dtype,
        device=ref,
    )
    # What `load_state_dict` does to a weight besides giving it data
    # (`layer.py:506`); without it the fully-qualified names are absent and the
    # external-constant census would not be comparable.
    for name, weight, _layer in step._iter_named_weights():
        weight.name = name

    pinned = [
        TensorType(t.dtype, [_EXTENTS.get(str(d), d) for d in t.shape], ref)
        for t in step.input_types()
    ]
    with Graph("flux_denoise", input_types=pinned) as graph:
        graph.output(step(*(v.tensor for v in graph.inputs), signal_buffers=[]))
    return str(graph)


def _analyze(mlir: str) -> dict[str, object]:
    defs: dict[str, tuple[int, str, int]] = {}
    last_use: dict[str, int] = {}
    order: list[str] = []
    symbolic: dict[str, int] = {}

    lines = mlir.splitlines()
    for index, line in enumerate(lines):
        m = _DEF.match(line)
        head = m.group(1) if m else ""
        # A multi-result op prints `%214:6 = ...` and is used as `%214#0`, so
        # each result is split out here. Treating the group as one value leaves
        # every use matching nothing, which holds the whole set live to the end
        # of the graph and inflates the peak by all of it.
        names: list[str] = []
        if m:
            base, _, count = head.partition(":")
            names = (
                [f"{base}#{i}" for i in range(int(count))] if count else [base]
            )
        for ssa in _SSA.findall(line[len(head) :] if head else line):
            if ssa not in names:
                last_use[ssa] = index
        if m is None:
            continue
        kind = m.group(2)
        tail = line.rsplit("->", 1)[-1] if "->" in line else line
        sizes = _result_bytes(tail, symbolic)
        if len(names) > 1:
            assert len(sizes) == len(names), (
                f"{kind} has {len(names)} results but {len(sizes)} types"
            )
        for ssa, size in zip(names, sizes or [0], strict=False):
            defs[ssa] = (index, kind, size)
            order.append(ssa)

    variants: dict[str, object] = {}
    for label, drop in (
        ("materializing", _HOISTED | _VIEWS),
        ("materializing_plus_views", _HOISTED),
    ):
        sizes, live, tags = [], [], []
        for ssa in order:
            index, kind, size = defs[ssa]
            if kind in drop or size == 0:
                continue
            sizes.append(size)
            live.append((index, last_use.get(ssa, len(lines))))
            tags.append(f"{ssa} {kind}")
        peak, at = _max_live(sizes, live)
        # What is alive at the peak, largest first. A headline number nobody
        # can audit is not a measurement, and this is what caught the first
        # version's multi-result liveness bug.
        held = sorted(
            (
                (sizes[i], tags[i])
                for i in range(len(sizes))
                if live[i][0] <= at <= live[i][1]
            ),
            reverse=True,
        )
        variants[label] = {
            "buffers": len(sizes),
            "requested_bytes": sum(sizes),
            "largest_buffer_bytes": max(sizes, default=0),
            "max_live_bytes": peak,
            "max_live_at_line": at + 1,
            "max_live_top": [{"bytes": b, "value": t} for b, t in held[:8]],
            "greedy_pool_bytes": _greedy_pool(sizes, live),
        }

    weights = [s for _, k, s in defs.values() if k == "mo.constant.external"]
    return {
        "variants": variants,
        "weights": {
            "count": len(weights),
            "bytes": sum(weights),
            "largest_bytes": max(weights, default=0),
        },
        "symbolic_extents_bounded_at": {
            "bound": _SYMBOLIC_BOUND,
            "occurrences": symbolic,
        },
    }


# Depths, shallowest first. The transformer's blocks run in sequence, so a
# block's temporaries die before the next block's are live and the peak should
# be set by one block plus the residual stream rather than by the depth.
# Measuring the curve is what turns that from an argument into evidence.
_DEPTHS = [(1, 1), (2, 8), (5, 20)]


@pytest.mark.parametrize(("layers", "single"), _DEPTHS)
def test_flux_denoise_activation_pool_fits_the_transport_cap(
    layers: int, single: int
) -> None:
    report = _analyze(_stage(layers, single))
    report["num_layers"] = layers
    report["num_single_layers"] = single
    text = json.dumps(report, indent=2)
    print(text)
    if outputs := os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR"):
        (Path(outputs) / f"pool_{layers}_{single}.json").write_text(text)

    if (layers, single) == (5, 20):
        # The staging reproduces a quantity it was not built to measure, which
        # is what makes the buffer sizes trustworthy: 166 external constants
        # totalling the SAME 7,751,089,152 bytes as the pre-fusion 169-row
        # census -- the temb fusion concatenates rows without moving a byte,
        # and its [52224, 3072] row is now the largest.
        assert report["weights"] == {
            "count": 166,
            "bytes": 7751089152,
            "largest_bytes": 320864256,
        }, report["weights"]

    for label, v in report["variants"].items():  # type: ignore[attr-defined]
        assert v["greedy_pool_bytes"] < _CAP, (
            f"{label} at {layers}+{single} blocks plans a "
            f"{v['greedy_pool_bytes']}-byte activation pool, past the "
            f"transport's {_CAP}-byte per-allocation cap: the denoise graph "
            "can no longer run as one allocation on this device"
        )
