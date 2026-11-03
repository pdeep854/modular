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
"""Compiles one op at a time against the virtual NPU and reports its fate.

Run as a script with one case name to probe that case; the JSON it prints is
the record. One process per case, because a MEF carries every kernel the
exporting process compiled, so the absence of a marker only means anything in a
process that compiled one graph.
"""

from __future__ import annotations

import json
import re
import sys
import traceback
import warnings
from collections.abc import Callable
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
from max.graph import DeviceRef, Graph, TensorType, ops

_KERNELS = (
    Path(__file__).resolve().parents[2] / "kernels" / "src" / "hexagon_ondevice"
)

_BF16 = DType.bfloat16
_F16 = DType.float16
_F32 = DType.float32
_I32 = DType.int32

# Markers looked for in the exported MEF. A stub's selector name reaches the
# artifact beside the source path of the registration that spells it, so the
# path says which package served the op and the name says which signed kernel.
_SELECTORS = (
    "matmul",
    "rms_norm",
    "layernorm",
    "layernorm_bf16",
    "groupnorm",
    "groupnorm_bf16",
    "softmax",
    "mha_no_cache",
    # Reachable since the `mo.silu` takeover in
    # `hexagon_ondevice/registrations.mojo`: a bare silu dispatches the signed
    # row its dtype picks. The menu names what CAN be dispatched, so a selector
    # missing here misreads a genuine dispatch as SILENT_HOST_CPU -- which is
    # exactly how silu_bf16 was misfiled before these two rows were added.
    "silu",
    "silu_bf16",
    # Reachable since `Add` grew its device arm (`MODULAR_NPU_ADD_DEVICE`):
    # a bare add names the selector its dtype picks -- vadd for fp32,
    # a ones-gated gate_residual row for the half precisions.
    "vadd",
    "gate_residual",
    "gate_residual_bf16",
)


def _t(dtype: DType, shape: list[int], device: DeviceRef) -> TensorType:
    return TensorType(dtype, shape, device)


def _graph(name: str, types: list[TensorType], body: Callable) -> Graph:
    """Stages one op, deliberately under the *default* placement policy.

    Every landed test in this directory sets
    `strict_device_placement=DevicePlacementPolicy.Error` and a structural
    check enforces it. This file is not one of them: its job is to *observe*
    which ops relocate themselves to the CPU, and the Error policy would turn
    the observation into an exception before the record could be made.
    """
    with Graph(name, input_types=types, custom_extensions=[_KERNELS]) as g:
        g.output(body(*(v.tensor for v in g.inputs)))
    return g


# Each case builds a graph carrying exactly one interesting op. The dtype is
# the one FLUX stages (bf16) wherever the model uses it, so the answer is about
# this model rather than about a dtype it never emits.
def _cases(d: DeviceRef) -> dict[str, Callable[[], Graph]]:
    def unary(fn, dtype=_BF16, shape=None):  # noqa: ANN001, ANN202
        shape = shape or [4, 64]
        return lambda: _graph(f"g_{fn.__name__}", [_t(dtype, shape, d)], fn)

    def binary(fn, dtype=_BF16, shape=None):  # noqa: ANN001, ANN202
        shape = shape or [4, 64]
        return lambda: _graph(f"g_{fn.__name__}", [_t(dtype, shape, d)] * 2, fn)

    def broadcast_binary(fn, dtype=_BF16):  # noqa: ANN001, ANN202
        """The shape FLUX's modulation and residual arithmetic actually is."""
        return lambda: _graph(
            f"gb_{fn.__name__}",
            [_t(dtype, [1, 4, 64], d), _t(dtype, [1, 1, 64], d)],
            fn,
        )

    cases: dict[str, Callable[[], Graph]] = {}

    # FLUX compute: matmul family.
    cases["matmul_bf16"] = lambda: _graph(
        "g_matmul_bf16",
        [_t(_BF16, [4, 64], d), _t(_BF16, [8, 64], d)],
        lambda a, b: a @ ops.transpose(b, -1, -2),
    )
    cases["matmul_f16"] = lambda: _graph(
        "g_matmul_f16",
        [_t(_F16, [4, 64], d), _t(_F16, [8, 64], d)],
        lambda a, b: a @ ops.transpose(b, -1, -2),
    )
    cases["batch_matmul_bf16"] = lambda: _graph(
        "g_bmm_bf16",
        [_t(_BF16, [2, 4, 64], d), _t(_BF16, [2, 8, 64], d)],
        lambda a, b: a @ ops.transpose(b, -1, -2),
    )
    cases["batch_matmul_f16"] = lambda: _graph(
        "g_bmm_f16",
        [_t(_F16, [2, 4, 64], d), _t(_F16, [2, 8, 64], d)],
        lambda a, b: a @ ops.transpose(b, -1, -2),
    )

    # FLUX compute: broadcast arithmetic (241 ops).
    cases["add_bf16"] = broadcast_binary(ops.add)
    cases["mul_bf16"] = broadcast_binary(ops.mul)
    cases["sub_bf16"] = broadcast_binary(ops.sub)
    cases["div_bf16"] = broadcast_binary(ops.div)
    cases["pow_bf16"] = broadcast_binary(ops.pow)
    cases["max_bf16"] = broadcast_binary(ops.max)
    cases["min_bf16"] = broadcast_binary(ops.min)

    # FLUX compute: elementwise unary.
    cases["silu_bf16"] = unary(ops.silu)
    cases["gelu_bf16"] = unary(ops.gelu)
    cases["sigmoid_bf16"] = unary(ops.sigmoid)
    cases["tanh_bf16"] = unary(ops.tanh)
    cases["relu_bf16"] = unary(ops.relu)
    cases["exp_bf16"] = unary(ops.exp)
    cases["log_bf16"] = unary(ops.log)
    cases["sqrt_bf16"] = unary(ops.sqrt)
    cases["rsqrt_bf16"] = unary(ops.rsqrt)
    cases["sin_bf16"] = unary(ops.sin)
    cases["cos_bf16"] = unary(ops.cos)
    cases["abs_bf16"] = unary(ops.abs)
    cases["negate_bf16"] = unary(ops.negate)
    cases["floor_bf16"] = unary(ops.floor)
    cases["round_bf16"] = unary(ops.round)
    cases["erf_bf16"] = unary(ops.erf)
    cases["logical_not_bool"] = unary(ops.logical_not, dtype=DType.bool)

    # FLUX compute: norms and attention.
    def _ln(x):  # noqa: ANN001, ANN202
        g = ops.constant([1.0] * 64, _BF16, device=d)
        b = ops.constant([0.0] * 64, _BF16, device=d)
        return ops.layer_norm(x, g, b, epsilon=1e-5)

    cases["layer_norm_bf16"] = lambda: _graph(
        "g_ln_bf16", [_t(_BF16, [4, 64], d)], _ln
    )
    cases["rms_norm_bf16"] = lambda: _graph(
        "g_rms_bf16",
        [_t(_BF16, [4, 64], d)],
        lambda x: ops.rms_norm(
            x, ops.constant([1.0] * 64, _BF16, device=d), 1e-5
        ),
    )
    cases["softmax_bf16"] = unary(ops.softmax)
    cases["group_norm_bf16"] = lambda: _graph(
        "g_gn_bf16",
        [_t(_BF16, [1, 8, 16], d)],
        lambda x: ops.group_norm(
            x,
            ops.constant([1.0] * 8, _BF16, device=d),
            ops.constant([0.0] * 8, _BF16, device=d),
            num_groups=2,
            epsilon=1e-5,
        ),
    )

    # FLUX movement.
    cases["cast_bf16_to_f32"] = lambda: _graph(
        "g_cast", [_t(_BF16, [4, 64], d)], lambda x: ops.cast(x, _F32)
    )
    cases["concat_bf16"] = lambda: _graph(
        "g_concat",
        [_t(_BF16, [4, 64], d)] * 2,
        lambda a, b: ops.concat([a, b], axis=0),
    )
    cases["split_bf16"] = lambda: _graph(
        "g_split",
        [_t(_BF16, [8, 64], d)],
        lambda x: ops.split(x, [4, 4], axis=0)[0],
    )
    cases["slice_bf16"] = lambda: _graph(
        "g_slice", [_t(_BF16, [8, 64], d)], lambda x: x[0:4, :]
    )
    cases["transpose_bf16"] = lambda: _graph(
        "g_transpose",
        [_t(_BF16, [4, 64], d)],
        lambda x: ops.transpose(x, -1, -2),
    )
    cases["reshape_bf16"] = lambda: _graph(
        "g_reshape", [_t(_BF16, [4, 64], d)], lambda x: ops.reshape(x, [256])
    )
    cases["broadcast_to_bf16"] = lambda: _graph(
        "g_bcast",
        [_t(_BF16, [1, 64], d)],
        lambda x: ops.broadcast_to(x, [4, 64]),
    )
    cases["permute_bf16"] = lambda: _graph(
        "g_permute",
        [_t(_BF16, [1, 4, 64], d)],
        lambda x: ops.permute(x, [0, 2, 1]),
    )
    cases["pad_bf16"] = lambda: _graph(
        "g_pad",
        [_t(_BF16, [4, 64], d)],
        lambda x: ops.pad(x, [0, 0, 1, 1], value=0.0),
    )
    cases["stack_bf16"] = lambda: _graph(
        "g_stack",
        [_t(_BF16, [4, 64], d)] * 2,
        lambda a, b: ops.stack([a, b], axis=0),
    )
    cases["squeeze_bf16"] = lambda: _graph(
        "g_squeeze", [_t(_BF16, [1, 4, 64], d)], lambda x: ops.squeeze(x, 0)
    )
    cases["flatten_bf16"] = lambda: _graph(
        "g_flatten", [_t(_BF16, [1, 4, 64], d)], lambda x: ops.flatten(x)
    )
    cases["chunk_bf16"] = lambda: _graph(
        "g_chunk", [_t(_BF16, [8, 64], d)], lambda x: ops.chunk(x, 2, axis=0)[0]
    )
    cases["repeat_interleave_bf16"] = lambda: _graph(
        "g_repeat",
        [_t(_BF16, [4, 64], d)],
        lambda x: ops.repeat_interleave(x, 2, axis=0),
    )

    # The mechanism-1 set (`_check_device_placement` callers).
    cases["cumsum_bf16"] = lambda: _graph(
        "g_cumsum", [_t(_BF16, [4, 64], d)], lambda x: ops.cumsum(x, axis=-1)
    )
    cases["cumsum_f32"] = lambda: _graph(
        "g_cumsum32", [_t(_F32, [4, 64], d)], lambda x: ops.cumsum(x, axis=-1)
    )
    cases["nonzero_bf16"] = lambda: _graph(
        "g_nonzero", [_t(_BF16, [4, 64], d)], lambda x: ops.nonzero(x, "nz")
    )
    cases["tile_bf16"] = lambda: _graph(
        "g_tile", [_t(_BF16, [4, 64], d)], lambda x: ops.tile(x, [2, 1])
    )
    cases["scatter_bf16"] = lambda: _graph(
        "g_scatter",
        [_t(_BF16, [4, 64], d), _t(_BF16, [2, 64], d), _t(_I32, [2], d)],
        lambda x, u, i: ops.scatter(x, u, i, axis=0),
    )
    cases["scatter_add_bf16"] = lambda: _graph(
        "g_scatter_add",
        [_t(_BF16, [4, 64], d), _t(_BF16, [2, 64], d), _t(_I32, [2], d)],
        lambda x, u, i: ops.scatter_add(x, u, i, axis=0),
    )

    # Reductions, indexing, and the rest of the surface.
    cases["reduce_sum_bf16"] = unary(lambda x: ops.sum(x, axis=-1))
    cases["reduce_mean_bf16"] = unary(lambda x: ops.mean(x, axis=-1))
    cases["reduce_max_bf16"] = unary(lambda x: ops.max(x, axis=-1))
    cases["reduce_prod_bf16"] = unary(lambda x: ops.prod(x, axis=-1))
    cases["argmax_bf16"] = unary(lambda x: ops.argmax(x, axis=-1))
    cases["argsort_f32"] = lambda: _graph(
        "g_argsort", [_t(_F32, [64], d)], lambda x: ops.argsort(x)
    )
    cases["top_k_bf16"] = lambda: _graph(
        "g_topk", [_t(_BF16, [4, 64], d)], lambda x: ops.top_k(x, 4, axis=-1)[0]
    )
    cases["gather_bf16"] = lambda: _graph(
        "g_gather",
        [_t(_BF16, [8, 64], d), _t(_I32, [4], d)],
        lambda x, i: ops.gather(x, i, axis=0),
    )
    cases["gather_nd_bf16"] = lambda: _graph(
        "g_gather_nd",
        [_t(_BF16, [8, 64], d), _t(_I32, [4, 1], d)],
        lambda x, i: ops.gather_nd(x, i),
    )
    cases["where_bf16"] = lambda: _graph(
        "g_where",
        [_t(DType.bool, [4, 64], d), _t(_BF16, [4, 64], d)] * 1
        + [_t(_BF16, [4, 64], d)],
        lambda c, a, b: ops.where(c, a, b),
    )
    cases["equal_bf16"] = binary(ops.equal)
    cases["greater_bf16"] = binary(ops.greater)
    cases["outer_bf16"] = lambda: _graph(
        "g_outer",
        [_t(_BF16, [4], d), _t(_BF16, [8], d)],
        lambda a, b: ops.outer(a, b),
    )
    cases["band_part_bf16"] = lambda: _graph(
        "g_band",
        [_t(_BF16, [8, 8], d)],
        lambda x: ops.band_part(x, exclude=False),
    )

    # VAE decoder.
    cases["conv2d_f32"] = lambda: _graph(
        "g_conv",
        [_t(_F32, [1, 8, 8, 4], d), _t(_F32, [3, 3, 4, 4], d)],
        lambda x, f: ops.conv2d(x, f),
    )
    cases["resize_f32"] = lambda: _graph(
        "g_resize",
        [_t(_F32, [1, 4, 8, 8], d)],
        lambda x: ops.resize_nearest(x, [1, 4, 16, 16]),
    )
    cases["avg_pool2d_f32"] = lambda: _graph(
        "g_avgpool",
        [_t(_F32, [1, 8, 8, 4], d)],
        lambda x: ops.avg_pool2d(x, (2, 2)),
    )
    cases["max_pool2d_f32"] = lambda: _graph(
        "g_maxpool",
        [_t(_F32, [1, 8, 8, 4], d)],
        lambda x: ops.max_pool2d(x, (2, 2)),
    )

    # RoPE, the shape a diffusion graph stages it in.
    cases["rope_ragged_bf16"] = lambda: _graph(
        "g_rope",
        [
            _t(_BF16, [8, 4, 64], d),
            _t(DType.uint32, [2], d),
            _t(DType.uint32, [1], d),
            _t(_F32, [8, 64], d),
        ],
        lambda x, row_offsets, start_pos, freqs_cis: ops.custom(
            "mo.composite.rope.ragged",
            device=d,
            values=[x, row_offsets, start_pos, freqs_cis],
            out_types=[_t(_BF16, [8, 4, 64], d)],
            parameters={"interleaved": True, "rope_first": True},
        )[0].tensor,
    )

    # Attention.
    cases["mha_no_cache_f16"] = lambda: _graph(
        "g_mha",
        [_t(_F16, [1, 16, 2, 128], d)] * 3,
        lambda q, k, v: ops.custom(
            "mo.mha.no_cache",
            device=d,
            values=[q, k, v, ops.constant(0.088, _F32, device=DeviceRef.CPU())],
            out_types=[_t(_F16, [1, 16, 2, 128], d)],
        )[0].tensor,
    )

    return cases


def _selector_hits(blob: bytes) -> list[str]:
    return [s for s in _SELECTORS if (s.encode() + b"\x00") in blob]


# The sentence `_check_device_placement` builds. Its presence in a staging
# warning is the only signal that an op rewrote the graph around itself.
_RELOCATION = "does not support GPU execution and will implicitly transfer"


def classify(record: dict) -> str:
    """Names the rung of the disposition ladder a probe record landed on.

    Only meaningful for a record produced by a process that compiled one graph:
    a MEF carries every kernel its exporting process compiled, so every rung
    below `SIGNED_SELECTOR` rests on a marker being *absent*.
    """
    if any(_RELOCATION in w for w in record.get("staging_warnings", [])):
        return "RELOCATED_TO_CPU"
    if record.get("outcome") != "CLEAN":
        return str(record.get("outcome"))
    if record.get("selectors") and record.get("names_hexagon_ondevice"):
        return "SIGNED_SELECTOR"
    if not any(d.startswith("npu") for d in record.get("device_tokens", [])):
        return "LEFT_THE_NPU"
    if record.get("kernel_sources"):
        return "SILENT_HOST_CPU"
    return "NO_DEVICE_KERNEL"


_SIG_RE = re.compile(rb"mo\.[a-z_.0-9]+ : \([^\x00\n]{0,120}")
_DEV_RE = re.compile(rb"(?:cpu|npu|gpu):\d+")
_SRC_RE = re.compile(rb"[A-Za-z0-9_./-]{4,90}\.mojo")


def _mef_forensics(blob: bytes) -> dict:
    sigs = sorted({m.decode("utf-8", "replace") for m in _SIG_RE.findall(blob)})
    devs = sorted({m.decode() for m in _DEV_RE.findall(blob)})
    srcs = sorted({m.decode() for m in _SRC_RE.findall(blob)})
    return {
        "op_signatures": sigs[:40],
        "device_tokens": devs,
        "kernel_sources": [
            s
            for s in srcs
            if "builtin_kernels" in s
            or "hexagon_ondevice" in s
            or "/nn/" in s
            or "/linalg/" in s
            or "_plugin/qaic" in s
        ][:40],
    }


def probe(case: str) -> dict:
    """Compiles one case on the NPU device and reports what happened."""
    set_virtual_device_api("npu")
    set_virtual_device_target_arch("hexagonv81")
    set_virtual_device_count(1)

    record: dict = {"case": case}
    device = DeviceRef.NPU()
    try:
        builder = _cases(device)[case]
    except KeyError:
        return {"case": case, "outcome": "NO_SUCH_CASE"}

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            graph = builder()
        record["staging_warnings"] = [str(w.message) for w in caught]
    except Exception as exc:
        record["outcome"] = "STAGING_ERROR"
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    # Printed before the compile, because a graph the compiler aborts on takes
    # the process with it and the staging verdict would be lost with it.
    print("###STAGED###" + json.dumps(record), flush=True)

    try:
        compiled = InferenceSession(devices=[NPU()]).compile(graph)
    except Exception as exc:
        record["outcome"] = "REFUSED"
        parts: list[str] = []
        cur: BaseException | None = exc
        while cur is not None:
            parts.append(str(cur))
            cur = cur.__cause__ or cur.__context__
        record["diagnostic"] = "\n".join(parts)[:4000]
        return record

    record["outcome"] = "CLEAN"
    with TemporaryDirectory() as tmp:
        mef = Path(tmp) / f"{case}.mef"
        try:
            compiled.export_mef(mef)
            blob = mef.read_bytes()
        except Exception as exc:
            record["mef_error"] = f"{type(exc).__name__}: {exc}"
            return record
    record["mef_bytes"] = len(blob)
    record["selectors"] = _selector_hits(blob)
    record["names_hexagon_ondevice"] = b"hexagon_ondevice" in blob
    record["names_builtin_kernels"] = b"builtin_kernels" in blob
    record.update(_mef_forensics(blob))
    return record


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        print(json.dumps(sorted(_cases(DeviceRef.NPU()))))
        return 0
    case = sys.argv[1]
    try:
        record = probe(case)
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
