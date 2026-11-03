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
"""Per-weight consumer extraction from a staged graph's MLIR text.

The whole-graph stagers record, for every external weight constant, the ops
that consume it and — for matmul-class consumers — the dispatch extents
``(M, N, K)`` the kernel will be handed. That table is what makes the crouton
selection MECHANICAL: the skel's matmul selector picks its consumption arm
per dispatch by shape alone (``matmul_bt_hmx_accepts``,
``kernels/src/hexagon_ondevice/matmul_kernel.mojo``), so a weight row must be
packed for the arm that reads it — crouton iff EVERY consuming dispatch is
array-accepted, row-major otherwise, because the refused dispatches run the
HVX vector nest, which reads B row-major unconditionally
(``npu_kernel.mojo``, the post-accept fallback). The six M=1 temb-path rows
that shipped crouton under the fix307 op-type census (order 315) are exactly
the class an op count cannot see and this table can.

The walk is deliberately conservative. A weight's value is followed forward
only through ops that provably preserve its bytes-as-a-2D-read: device
transfers, full 2-D transposes (the staged ``W^T`` the ``matmul_bt`` lowering
folds back into the row-major registry read), same-shape slices and shape
rebinds, and pure reshapes. Anything else terminates the walk and is recorded
as a consumer; an op this module does not know is therefore a REFUSAL
downstream, never a silent pass — row-major is always correct, crouton is the
optimization.

One lowering is modelled rather than observed, because it happens after
staging: ``ConvToMatmulPattern`` rewrites every pointwise convolution
(1x1 kernel, stride 1, no padding, no dilation, one group) into a matmul
whose B is the ``[F, C]`` view of the FCRS filter. Such consumers carry an
``as_matmul`` extent record; every other convolution stays a plain ``conv``
consumer and keeps its weight row-major FCRS (``VAE_CONV_LAYOUT_STUDY.md``).
"""

from __future__ import annotations

import re
from collections import defaultdict

#: The serve world's symbolic-dimension values. The bundle manifest pins
#: batch_size = 1; sequence extents are pinned to literals at staging. A
#: symbol not listed here resolves to None and poisons the dispatch extents,
#: which downstream reads as "not provably array-accepted" — the safe verdict.
SERVE_DIM_BINDINGS = {"batch": 1}

_TENSOR = re.compile(r"!mo\.tensor<\[([^\]]*)\]")
_NAME_ATTR = re.compile(r'name\s*=\s*"([^"]+)"')
_SSA_ID = re.compile(r"%[A-Za-z0-9_]+")
_OP_KIND = re.compile(r"([a-z][A-Za-z0-9_.]*)")

#: Multi-result data movers whose FIRST result is the data (the second is the
#: chain token): `%result, %outChain = rmo.mo.transfer[%chain] %data : ...`.
_FIRST_RESULT_OPS = frozenset({"rmo.mo.transfer", "mo.transfer"})

#: Ops whose result is the operand-0 tensor under a different view; operand 1
#: (a permutation or shape) is metadata and does not propagate weight bytes.
_VIEW_OPS = frozenset(
    {"rmo.mo.transpose", "mo.transpose", "rmo.reshape", "mo.reshape"}
)

#: View ops that must ALSO preserve the shape to be transparent: a
#: full-extent slice and a layout/shape rebind are identity reads, a partial
#: slice is a real subselection of the row and terminates the walk (croutons
#: of the whole row would be sliced mid-tile).
_SAME_SHAPE_VIEW_OPS = frozenset(
    {
        "rmo.slice",
        "mo.slice",
        "rmo.rebind_tensor_shape",
        "mo.rebind_tensor_shape",
    }
)

_MATMUL_OPS = frozenset({"rmo.matmul", "mo.matmul"})
_CONV_OPS = frozenset({"rmo.conv", "mo.conv"})

_STRIDE_ONE = "#mosh<ape[1, 1]>"
_PAD_ZERO = "#mosh<ape[0, 0, 0, 0]>"


def _parse_ops(mlir: str) -> list[dict[str, object]]:
    """One record per SSA-defining line: kind, results, operands, shapes."""
    ops: list[dict[str, object]] = []
    for lineno, raw in enumerate(mlir.splitlines(), start=1):
        line = raw.strip()
        if not line.startswith("%") or " = " not in line:
            continue
        lhs, rhs = line.split(" = ", 1)
        kind_match = _OP_KIND.match(rhs)
        if not kind_match:
            continue
        operand_text = rhs[kind_match.end() :].split(" : ", 1)[0]
        name_match = _NAME_ATTR.search(line)
        ops.append(
            {
                "lineno": lineno,
                "line": line,
                "kind": kind_match.group(1),
                "results": [r.strip() for r in lhs.split(",")],
                "operands": _SSA_ID.findall(operand_text),
                "shapes": [
                    [d.strip() for d in dims.split(",")] if dims.strip() else []
                    for dims in _TENSOR.findall(line)
                ],
                "name": name_match.group(1) if name_match else None,
            }
        )
    return ops


def _resolve(dim: str, bindings: dict[str, int]) -> int | None:
    try:
        return int(dim)
    except ValueError:
        return bindings.get(dim)


def _rows(dims: list[int | None]) -> int | None:
    """The 2-D row extent of a rank-N operand: every dim but the last."""
    rows = 1
    for dim in dims[:-1]:
        if dim is None:
            return None
        rows *= dim
    return rows


def _matmul_consumer(
    op: dict[str, object], value: str, bindings: dict[str, int]
) -> dict[str, object]:
    operands = op["operands"]
    shapes = op["shapes"]
    entry: dict[str, object] = {
        "op": op["kind"],
        "line": op["lineno"],
        "role": "b" if operands and operands[0] != value else "a",
    }
    # (A, B) -> C on one line: B is staged [K, N] (the folded W^T), C's
    # trailing dim is N and its leading dims are the dispatch rows M.
    if len(shapes) == 3:
        b_dims = [_resolve(d, bindings) for d in shapes[1]]
        c_dims = [_resolve(d, bindings) for d in shapes[2]]
        entry["m"] = _rows(c_dims)
        entry["n"] = c_dims[-1] if c_dims else None
        entry["k"] = b_dims[0] if b_dims else None
    return entry


def _conv_consumer(
    op: dict[str, object], value: str, bindings: dict[str, int]
) -> dict[str, object]:
    operands = op["operands"]
    shapes = op["shapes"]
    entry: dict[str, object] = {
        "op": op["kind"],
        "line": op["lineno"],
        "role": (
            "filter" if len(operands) > 1 and operands[1] == value else "input"
        ),
    }
    line = op["line"]
    if (
        entry["role"] == "filter"
        and len(shapes) == 3
        and len(shapes[1]) == 4
        and shapes[1][2] == "1"
        and shapes[1][3] == "1"
        and f"strides = {_STRIDE_ONE}" in line
        and f"dilations = {_STRIDE_ONE}" in line
        and f"paddings = {_PAD_ZERO}" in line
        and "num_groups = 1 " in line
    ):
        out_dims = [_resolve(d, bindings) for d in shapes[2]]
        entry["as_matmul"] = {
            "m": _rows(out_dims),
            "n": out_dims[-1] if out_dims else None,
            "k": _resolve(shapes[1][1], bindings),
        }
    return entry


def weight_consumers(
    mlir: str, bindings: dict[str, int] | None = None
) -> list[dict[str, object]]:
    """Builds the per-weight consumer table from staged MLIR text.

    Args:
        mlir: The staged graph as text (``str(graph)``).
        bindings: Symbolic-dimension values for the serve world; defaults to
            ``SERVE_DIM_BINDINGS``.

    Returns:
        One entry per ``mo.constant.external`` with a ``name`` attribute:
        ``{"name", "shape", "consumers": [...]}``, in declaration order.
        Matmul consumers carry ``role`` (``a``/``b``) and the dispatch
        ``m``/``n``/``k`` (None where a symbol did not resolve); pointwise
        convolutions carry ``as_matmul`` with the post-lowering extents; every
        other terminal op is recorded by kind alone.
    """
    resolved_bindings = (
        SERVE_DIM_BINDINGS if bindings is None else dict(bindings)
    )
    ops = _parse_ops(mlir)
    uses: dict[str, list[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        for value in op["operands"]:
            uses[value].append(index)

    table: list[dict[str, object]] = []
    for op in ops:
        if op["kind"] != "mo.constant.external" or not op["name"]:
            continue
        consumers: list[dict[str, object]] = []
        frontier = [op["results"][0]]
        seen = set(frontier)

        def follow(value: str) -> None:
            if value not in seen:
                seen.add(value)
                frontier.append(value)

        while frontier:
            value = frontier.pop()
            for use_index in uses[value]:
                use = ops[use_index]
                kind = use["kind"]
                if kind in _FIRST_RESULT_OPS:
                    follow(use["results"][0])
                elif kind in _VIEW_OPS:
                    if use["operands"] and use["operands"][0] == value:
                        follow(use["results"][0])
                    # else: the value is the permutation/shape operand of
                    # some other tensor's view — metadata, not a data flow.
                elif kind in _SAME_SHAPE_VIEW_OPS:
                    shapes = use["shapes"]
                    if (
                        use["operands"]
                        and use["operands"][0] == value
                        and len(shapes) == 2
                        and shapes[0] == shapes[1]
                    ):
                        follow(use["results"][0])
                    else:
                        consumers.append(
                            {"op": f"{kind}:partial", "line": use["lineno"]}
                        )
                elif kind in _MATMUL_OPS:
                    consumers.append(
                        _matmul_consumer(use, value, resolved_bindings)
                    )
                elif kind in _CONV_OPS:
                    consumers.append(
                        _conv_consumer(use, value, resolved_bindings)
                    )
                else:
                    consumers.append({"op": kind, "line": use["lineno"]})
        table.append(
            {
                "name": op["name"],
                "shape": [
                    _resolve(d, resolved_bindings) for d in op["shapes"][0]
                ],
                "consumers": consumers,
            }
        )
    return table
