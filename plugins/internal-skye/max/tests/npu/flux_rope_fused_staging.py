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
"""The FLUX_ROPE_FUSED_NORM staging rewrite, shared by BOTH denoise stagers.

One module because window 281 caught the fifth real-correct-unreachable-arm
instance one layer above the launcher fix: the rewrite lived only in
`test_flux_denoise_mef.py`'s `_stage` while both banked production-MEF
recipes export through `test_flux_denoise_whole_graph.py`'s `_stage`, so the
frozen production `npu.mef` censused zero fused ops. The two stagers build
the same model; the rewrite they apply must be the same object, not two
copies that can drift.

The rewrite (FUSION_S2 §8): a qk RMSNorm returns its input tagged instead of
staging a norm; the tag follows reshapes; the interleaved rope mints
`fused_rms_norm_rope` from the PRE-norm value, and any concat materializes
the norm on the spot — so the dual-stream blocks' norm-concat-rope chains
keep today's ops exactly. A tag no patched consumer claimed means an
unpatched op computed on pre-norm values, which the exit check turns into a
loud refusal rather than a wrong graph. With the flag off this module
patches NOTHING, so a flag-off export is the pre-change export by
construction.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from max.dtype import DType
from max.graph import DeviceRef, TensorType, ops


def rope_fused_requested() -> bool:
    """Reads the staging flag exactly one way for every caller."""
    return os.environ.get("FLUX_ROPE_FUSED_NORM", "") == "1"


def _qk_norm_ids(step: object) -> set[int]:
    """The identity set of the model's qk RMSNorm modules, shards included.

    Collected from the built transformer rather than by patching the RMSNorm
    class wholesale: the model also carries embedding-width RMSNorms whose
    consumers are neither a rope nor a concat, and tagging one of those would
    trip the dangling refusal on a correct graph.
    """
    ids: set[int] = set()
    transformer = step.transformer  # type: ignore[attr-defined]
    blocks = list(transformer.transformer_blocks) + list(
        transformer.single_transformer_blocks
    )
    for blk in blocks:
        for owner in (blk.attn, *getattr(blk, "attn_shards", ())):
            for attr in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
                mod = getattr(owner, attr, None)
                if mod is not None:
                    ids.add(id(mod))
    return ids


@contextmanager
def rope_fused_staging(step: object) -> Iterator[None]:
    """Applies the fused-qk-rope staging rewrite around one graph build.

    A no-op unless ``FLUX_ROPE_FUSED_NORM=1``. Patches are applied on entry
    and restored on exit; a tag left unconsumed after a successful build
    raises rather than let an unpatched consumer read pre-norm values.

    Args:
        step: The ``DenoiseComputeStep`` whose transformer's qk norms are the
            rewrite's targets, built before the graph context opens.
    """
    if not rope_fused_requested():
        yield
        return

    from max.nn.norm import RMSNorm
    from max.pipelines.architectures.flux2.layers import (
        flux2_attention as attn_mod,
    )

    qk_norm_ids = _qk_norm_ids(step)
    pending: dict[int, tuple[object, object]] = {}
    orig_rms_call = RMSNorm.__call__
    orig_rope = attn_mod.rope_ragged_with_position_ids
    orig_reshape = ops.reshape
    orig_concat = ops.concat

    def _take_tag(v):  # noqa: ANN001, ANN202
        tag = pending.get(id(v))
        if tag is not None and tag[1] is v:
            return pending.pop(id(v))
        return None

    def _materialize(v):  # noqa: ANN001, ANN202
        tag = _take_tag(v)
        return orig_rms_call(tag[0], v) if tag is not None else v

    def _tagging_rms_call(self, x):  # noqa: ANN001, ANN202
        if (
            id(self) not in qk_norm_ids
            or not self.multiply_before_cast
            or self.weight_offset != 0.0
        ):
            return orig_rms_call(self, x)
        pending[id(x)] = (self, x)
        return x

    def _tag_following_reshape(x, *args, **kwargs):  # noqa: ANN001, ANN202
        out = orig_reshape(x, *args, **kwargs)
        tag = _take_tag(x)
        if tag is not None:
            pending[id(out)] = (tag[0], out)
        return out

    def _flushing_concat(vals, *args, **kwargs):  # noqa: ANN001, ANN202
        return orig_concat([_materialize(v) for v in vals], *args, **kwargs)

    def _fused_rope(x, freqs_cis, position_ids, **kwargs):  # noqa: ANN001, ANN202
        tag = _take_tag(x)
        if tag is None:
            return orig_rope(x, freqs_cis, position_ids, **kwargs)
        if not kwargs.get("interleaved", True) or kwargs.get("mrope_section"):
            return orig_rope(
                orig_rms_call(tag[0], x), freqs_cis, position_ids, **kwargs
            )
        norm = tag[0]
        weight = norm.weight.cast(x.dtype)
        if x.device:
            weight = weight.to(x.device)
        eps = ops.constant(
            float(norm.eps), DType.float32, device=DeviceRef.CPU()
        )
        return ops.custom(
            "fused_rms_norm_rope",
            device=x.device,
            values=[x, weight, freqs_cis, position_ids, eps],
            out_types=[TensorType(x.dtype, x.shape, x.device)],
        )[0].tensor

    RMSNorm.__call__ = _tagging_rms_call
    attn_mod.rope_ragged_with_position_ids = _fused_rope
    ops.reshape = _tag_following_reshape
    ops.concat = _flushing_concat
    try:
        yield
        if pending:
            raise AssertionError(
                f"FLUX_ROPE_FUSED_NORM staging left {len(pending)} qk-norm"
                " tag(s) unconsumed -- an unpatched consumer read pre-norm"
                " values; the fused staging refuses rather than export a"
                " wrong graph"
            )
    finally:
        RMSNorm.__call__ = orig_rms_call
        attn_mod.rope_ragged_with_position_ids = orig_rope
        ops.reshape = orig_reshape
        ops.concat = orig_concat
