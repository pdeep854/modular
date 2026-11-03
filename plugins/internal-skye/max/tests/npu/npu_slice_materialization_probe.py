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
"""Measures which `mo.slice` shapes the graph compiler forces to materialize.

A slice is a view kernel, so it costs nothing while it stays an alias. Only
materializing it runs `view_copy_impl`, which is generated device code and does
not compile on this target -- so the size of the `mo.slice` gap is not the op
count, it is the count of slices the compiler materializes.

Compile success is the wrong instrument on its own: several consumers here fail
for reasons of their own, so a refusal downstream of a slice is ambiguous. The
IR is unambiguous. `mo-to-mogg` emits one `mogg.call.materialize` per view
*unconditionally* (`MOToMOGGLowering.cpp:390`), and `mogg-fuse-views` erases the
ones it can fold into a consumer, so the count surviving into `.mogg-pre-mgp.mlir`
is exactly what this graph materializes -- readable whether or not the Mojo
elaboration that follows succeeds.

Every slice form is paired with a `none` control that feeds the same consumer
directly, because a consumer contributes materializes of its own.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

from max.driver import (
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
    Dim,
    Graph,
    TensorType,
    TensorValue,
    ops,
)

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

# Batch 2 so that a slice on an interior axis is genuinely non-contiguous: at
# batch 1 every single-axis slice degenerates to one contiguous run and the
# interesting half of the question cannot be asked.
_B, _S, _W = 2, 4, 128
_DEV = DeviceRef.NPU()
_SYM = Dim("seq")


def virtual_npu() -> Iterator[None]:
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


def compile_error(exc: BaseException) -> str:
    """Joins a compile failure with its causes.

    `InferenceSession.compile` reports a generic "Failed to compile the model"
    and chains the diagnostic underneath, so the message worth reading is never
    the top one.
    """
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(str(cur))
        cur = cur.__cause__ or cur.__context__
    return "\n".join(parts)


def blame(diagnostic: str) -> str:
    """Names the refusing file, most specific cause first.

    Ordered so that a consumer's own refusal is not mistaken for the slice's:
    an elementwise op fails through the QAIC hook and its diagnostic still
    mentions the primitives file further up the chain.
    """
    for token, name in (
        ("algorithm_backend/elementwise.mojo", "QAIC elementwise hook"),
        ("on the on-device NPU:", "our own stub"),
        ("hexagon-unknown-unknown-elf", "codegen refuses the triple"),
        ("builtin_primitives/primitives.mojo", "view_copy_impl"),
    ):
        if token in diagnostic:
            return name
    return "unclassified" if diagnostic else ""


# --- slice forms -------------------------------------------------------------


def _f_none(x: TensorValue) -> TensorValue:
    return x


def _f_whole(x: TensorValue) -> TensorValue:
    return x[:, :, :]


def _f_outer_partial(x: TensorValue) -> TensorValue:
    """Partial on the outermost axis, full inside: one contiguous run."""
    return x[0:1, :, :]


def _f_mid_partial(x: TensorValue) -> TensorValue:
    """Partial on an interior axis with a non-singleton axis outside it."""
    return x[:, 1:3, :]


def _f_last_partial(x: TensorValue) -> TensorValue:
    """Partial on the last axis: the halved fused projection FLUX stages."""
    return x[:, :, 0 : _W // 2]


def _f_two_axes(x: TensorValue) -> TensorValue:
    return x[0:1, 1:3, :]


def _f_step2(x: TensorValue) -> TensorValue:
    return x[:, :, 0:_W:2]


SLICES: dict[str, Callable[[TensorValue], TensorValue]] = {
    "none": _f_none,
    "whole": _f_whole,
    "outer_partial": _f_outer_partial,
    "mid_partial": _f_mid_partial,
    "last_partial": _f_last_partial,
    "two_axes": _f_two_axes,
    "step2": _f_step2,
}

# Width the sliced operand carries, for a consumer that must be built to match.
WIDTH_AFTER = dict.fromkeys(SLICES, _W)
WIDTH_AFTER["last_partial"] = _W // 2
WIDTH_AFTER["step2"] = _W // 2


# --- consumers ---------------------------------------------------------------


def _c_output(x: TensorValue, width: int) -> TensorValue:
    return x


def _c_layer_norm(x: TensorValue, width: int) -> TensorValue:
    ones = ops.constant(1.0, x.dtype, DeviceRef.CPU()).broadcast_to([width])
    zeros = ops.constant(0.0, x.dtype, DeviceRef.CPU()).broadcast_to([width])
    return ops.layer_norm(
        x, gamma=ones.to(_DEV), beta=zeros.to(_DEV), epsilon=1e-6
    )


def _c_matmul(x: TensorValue, width: int) -> TensorValue:
    w = ops.constant(0.0, x.dtype, DeviceRef.CPU()).broadcast_to([width, width])
    return x @ ops.transpose(w.to(_DEV), -1, -2)


def _c_concat(x: TensorValue, width: int) -> TensorValue:
    return ops.concat([x, x], axis=-1)


CONSUMERS: dict[str, Callable[[TensorValue, int], TensorValue]] = {
    "graph_output": _c_output,
    "layer_norm": _c_layer_norm,
    "matmul": _c_matmul,
    "concat": _c_concat,
}

# The registrations are dtype-gated, so a case must be staged at a dtype its
# consumer accepts or the stub's own assert masks whatever the slice did.
DTYPES = {
    "graph_output": DType.bfloat16,
    "layer_norm": DType.bfloat16,
    "matmul": DType.float16,
    "concat": DType.bfloat16,
}


def build(kernels, consumer: str, slice_form: str, *, symbolic: bool) -> Graph:
    dtype = DTYPES[consumer]
    shape = [_B, _SYM if symbolic else _S, _W]
    name = f"slice_{slice_form}_into_{consumer}{'_sym' if symbolic else ''}"
    with Graph(
        name,
        input_types=[TensorType(dtype, shape, _DEV)],
        custom_extensions=[kernels],
    ) as g:
        x = g.inputs[0].tensor
        sliced = SLICES[slice_form](x)
        g.output(CONSUMERS[consumer](sliced, WIDTH_AFTER[slice_form]))
    return g


def _count(directory: Path, suffix: str, token: str) -> int:
    return sum(f.read_text().count(token) for f in directory.glob(f"*{suffix}"))


def probe(kernels, dumps: Path, *, symbolic: bool = False) -> list[dict]:
    """Compiles every (consumer, slice form) pair and counts its materializes."""
    records: list[dict] = []
    for consumer in CONSUMERS:
        for slice_form in SLICES:
            rec: dict = {
                "consumer": consumer,
                "slice": slice_form,
                "symbolic": symbolic,
            }
            case_dir = dumps / f"{consumer}__{slice_form}"
            case_dir.mkdir(parents=True, exist_ok=True)
            try:
                graph = build(kernels, consumer, slice_form, symbolic=symbolic)
            except Exception as exc:  # staging, not compilation
                rec["outcome"] = "STAGING_ERROR"
                rec["blame"] = type(exc).__name__
                rec["diagnostic"] = str(exc)[:600]
                records.append(rec)
                continue
            session = InferenceSession(devices=[NPU()])
            session.debug.ir_output_dir = str(case_dir)
            try:
                session.compile(graph)
            except Exception as exc:
                diagnostic = compile_error(exc)
                rec["outcome"] = "REFUSED"
                rec["blame"] = blame(diagnostic)
                rec["diagnostic"] = diagnostic[-700:]
            else:
                rec["outcome"] = "COMPILES"
                rec["blame"] = ""
                rec["diagnostic"] = ""
            rec["views_total"] = _count(
                case_dir, ".mogg.mlir", "mogg.call.materialize"
            )
            rec["views_materialized"] = _count(
                case_dir, ".mogg-pre-mgp.mlir", "mogg.call.materialize"
            )
            rec["views_zero_copy"] = _count(
                case_dir, ".mogg-pre-mgp.mlir", "mogg.slice"
            )
            # Distinguishes "the slice was folded away" from "the slice became a
            # dispatch": a materialize count alone cannot, since an `execute`
            # kernel emits no `mogg.call.materialize` either.
            rec["slice_ops_pre"] = _count(case_dir, ".mogg.mlir", '"mo.slice"')
            rec["slice_ops_post"] = _count(
                case_dir, ".mogg-pre-mgp.mlir", '"mo.slice"'
            )
            rec["dumped_ir"] = bool(list(case_dir.glob("*.mogg.mlir")))
            records.append(rec)
    return records


def render(records: list[dict]) -> str:
    """Prints `materialized/total` views per cell, against the `none` control."""
    forms = list(SLICES)
    width = max(len(c) for c in CONSUMERS) + 2
    head = "consumer".ljust(width) + "".join(f.ljust(22) for f in forms)
    rows = [head, "-" * len(head)]
    for consumer in CONSUMERS:
        by_form = {r["slice"]: r for r in records if r["consumer"] == consumer}
        base = by_form.get("none", {}).get("views_materialized", 0)
        cells = []
        for form in forms:
            r = by_form[form]
            if "views_total" not in r or not r["dumped_ir"]:
                cells.append(f"{r['outcome']}(no IR)".ljust(22))
                continue
            delta = r["views_materialized"] - base
            mark = "ok" if r["outcome"] == "COMPILES" else (r["blame"] or "?")
            cells.append(f"{delta:+d}m {r['slice_ops_post']}k {mark}".ljust(22))
        rows.append(consumer.ljust(width) + "".join(cells))
    return "\n".join(rows)


# --- contiguity verdict cases -------------------------------------------------

# The two positive forms are the FLUX classes a static-1 batch is meant to
# fold (VIEW_FOLDING_SCOPE.md: the [b,1,18432]->[b,1,3072] modulation chunks
# and the [b,4608,3072] sequence splits); the strided controls are copies at
# any batch. The chain rows pin the slice-of-slice collapse the temb-fused
# modulation chunks take (a chunk of a chunk fuses view-into-view before any
# consumer sees it): a stack of unit-step trailing chunks composes back to
# one byte-contiguous chunk and folds, while a non-unit step at any level or
# a second sliced axis keeps the copy. Per case: the input shape, the slice,
# the already-sliced input shape of its `none` control (the rule above: a
# consumer contributes materializes of its own, so only the delta against
# feeding the consumer directly attributes anything to the slice), the
# verdict, and the staged `mo.slice` floor -- 2 for the chains, so a
# staging-level composition cannot quietly turn a chain case into a
# single-slice one and leave its verdict vacuous.
VERDICT_CASES: list[
    tuple[
        str,
        list[int],
        Callable[[TensorValue], TensorValue],
        list[int],
        str,
        int,
    ]
] = [
    (
        "unit_batch_chunk",
        [1, 1, 256],
        lambda x: x[:, :, 0:128],
        [1, 1, 128],
        "fold",
        1,
    ),
    (
        "unit_batch_seq_split",
        [1, 8, 64],
        lambda x: x[:, 0:6, :],
        [1, 6, 64],
        "fold",
        1,
    ),
    (
        "unit_batch_step2",
        [1, 4, 128],
        lambda x: x[:, :, 0:128:2],
        [1, 4, 64],
        "copy",
        1,
    ),
    (
        "unit_batch_two_axes",
        [1, 4, 128],
        lambda x: x[:, 1:3, 0:64],
        [1, 2, 64],
        "copy",
        1,
    ),
    (
        "batch2_step2",
        [2, 4, 128],
        lambda x: x[:, :, 0:128:2],
        [2, 4, 64],
        "copy",
        1,
    ),
    (
        "batch2_two_axes",
        [2, 4, 128],
        lambda x: x[0:1, 1:3, :],
        [1, 2, 128],
        "copy",
        1,
    ),
    (
        "unit_batch_chunk_chain",
        [1, 1, 256],
        lambda x: x[:, :, 32:224][:, :, 0:96],
        [1, 1, 96],
        "fold",
        2,
    ),
    (
        "unit_batch_chunk_chain_step2",
        [1, 1, 256],
        lambda x: x[:, :, 32:224][:, :, 0:192:2],
        [1, 1, 96],
        "copy",
        2,
    ),
    (
        "unit_batch_chain_two_axes",
        [1, 4, 128],
        lambda x: x[:, 1:3, :][:, :, 0:64],
        [1, 2, 64],
        "copy",
        2,
    ),
]


def _fuse_views_ir(
    name: str,
    shape: list[int],
    slice_fn: Callable[[TensorValue], TensorValue],
    width: int,
    dumps: Path,
) -> tuple[str, str, str]:
    """Compiles slice-into-layer_norm; returns (pre-fuse, post-fuse, error).

    A compile failure is recorded, never swallowed: the verdict is read from
    the fuse-views IR, which is written whether or not the elaboration that
    follows succeeds (module docstring), so a consumer's own refusal must not
    mask it -- but when the dump is missing, the diagnostic is the only thing
    that says why.
    """
    dumps.mkdir(parents=True, exist_ok=True)
    with Graph(
        name,
        input_types=[TensorType(DType.bfloat16, shape, DeviceRef.NPU())],
        custom_extensions=[_KERNELS],
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as g:
        g.output(_c_layer_norm(slice_fn(g.inputs[0].tensor), width))
    session = InferenceSession(devices=[NPU()])
    session.debug.ir_output_dir = str(dumps)
    error = ""
    try:
        session.compile(g)
    except Exception as exc:
        error = compile_error(exc)
    pre = "".join(p.read_text() for p in dumps.glob("*.mogg.mlir"))
    post = "".join(p.read_text() for p in dumps.glob("*.mogg-pre-mgp.mlir"))
    return pre, post, error


def run_verdict_case(case: str, dumps: Path) -> dict:
    """Runs one verdict case's control and case compiles and returns counts.

    The asserts live in the pytest driver; this record carries only what the
    two compiles produced, plus their diagnostics, so a missing dump fails
    over there with its cause attached instead of as a bare absence.
    """
    row = next((r for r in VERDICT_CASES if r[0] == case), None)
    if row is None:
        return {"case": case, "outcome": "NO_SUCH_CASE"}
    name, shape, slice_fn, control_shape, want, _min_slice_ops = row
    width = control_shape[-1]
    _, control_post, control_error = _fuse_views_ir(
        f"{name}_none",
        control_shape,
        lambda x: x,
        width,
        dumps / f"{name}_none",
    )
    case_pre, case_post, case_error = _fuse_views_ir(
        name, shape, slice_fn, width, dumps / name
    )
    return {
        "case": name,
        "want": want,
        "control_materialize": control_post.count("mogg.call.materialize"),
        "case_materialize": case_post.count("mogg.call.materialize"),
        "case_pre_slice_ops": case_pre.count('"mo.slice"'),
        "control_dumped": bool(control_post),
        "case_dumped": bool(case_post),
        "control_error": control_error[-700:],
        "case_error": case_error[-700:],
    }


def main() -> int:
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)
    case = sys.argv[1]
    try:
        with TemporaryDirectory() as dumps:
            record = run_verdict_case(case, Path(dumps))
    except BaseException as exc:
        record = {
            "case": case,
            "outcome": "PROBE_CRASH",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-2000:],
        }
    print("###JSON###" + json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
