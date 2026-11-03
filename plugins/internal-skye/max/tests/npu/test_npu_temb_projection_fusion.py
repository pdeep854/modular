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
"""Temb-projection fusion: adapter contract and staged-math parity, device-free.

FLUX.2's four temb consumers (the img/txt/single modulation heads and
norm_out's conditioning projection) all read the same ``silu(temb)``
activation, so the model stages them as ONE fused Linear
(``Flux2TembProjections``) whose output row splits back into the members.
Two contracts hold that together, and each gets a test at its own grain:

- The ADAPTER contract: ``weight_adapters._fuse_temb_projections``
  concatenates the four checkpoint weights along axis 0 in exactly the
  member order ``Flux2TembProjections.out_dims`` splits them back out.
  Byte-level round-trip checks, including the bf16 uint16-view path and
  key prefixes, because a member-order swap would load cleanly and corrupt
  every modulation parameter downstream.
- The MATH contract: the fused staging (real ``Flux2TembProjections`` +
  ``_modulation_param_sets`` + ``AdaLayerNormContinuous.apply_projected``,
  weights supplied through the real adapter) computes bit-identically to
  the four separate ``linear(silu(temb))`` projections it replaced, staged
  here from the same member bytes. fp32 on CPU: each fused output column is
  the same K-reduction as its member's, so any drift is a wiring bug, not
  arithmetic.

The mechanic that gates the wider horizontal-GEMM family -- a fused
output's chunks surviving ``mogg-fuse-views`` as per-chunk
``mogg.call.materialize`` copies -- is measured by the whole-graph IR A/B
(``test_flux_denoise_whole_graph`` under ``FLUX_IR_DUMP``), not here; this
group's splits measured materialize-clean while the QKV groups' did not.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    DevicePlacementPolicy,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.graph.shape import Shape
from max.graph.weights import WeightData
from max.pipelines.architectures.flux2.flux2 import (
    Flux2TembProjections,
    _modulation_param_sets,
)
from max.pipelines.architectures.flux2.layers.normalizations import (
    AdaLayerNormContinuous,
    ada_modulate,
)
from max.pipelines.architectures.flux2.weight_adapters import (
    _fuse_temb_projections,
    _split_stacked_qkv,
    adapt_weights,
)

_DIM = 64
_BATCH = 2
_SEQ = 8
# Member out-dims in Flux2TembProjections order: img 2-set, txt 2-set,
# single 1-set modulations, then norm_out's (scale, shift).
_MEMBERS = (
    ("double_stream_modulation_img.linear", 6 * _DIM),
    ("double_stream_modulation_txt.linear", 6 * _DIM),
    ("single_stream_modulation.linear", 3 * _DIM),
    ("norm_out.linear", 2 * _DIM),
)


def _member_arrays(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        name: rng.standard_normal((out_dim, _DIM)).astype(np.float32)
        for name, out_dim in _MEMBERS
    }


def _fp32_weight(name: str, arr: np.ndarray) -> WeightData:
    return WeightData(
        Buffer.from_numpy(np.ascontiguousarray(arr)),
        name,
        DType.float32,
        Shape(arr.shape),
    )


def _bf16_weight(name: str, arr_fp32: np.ndarray) -> WeightData:
    u16 = (arr_fp32.view(np.uint32) >> 16).astype(np.uint16)
    buf = Buffer.from_numpy(np.ascontiguousarray(u16)).view(
        DType.bfloat16, arr_fp32.shape
    )
    return WeightData(buf, name, DType.bfloat16, Shape(arr_fp32.shape))


def _as_u16(value: WeightData) -> np.ndarray:
    return value.to_buffer().view(DType.uint16).to_numpy()


def test_fuse_temb_projections_member_order_and_bytes() -> None:
    """The fused bf16 row is the members concatenated in declared order."""
    rng = np.random.default_rng(7)
    members = _member_arrays(rng)
    state = {
        f"{name}.weight": _bf16_weight(f"{name}.weight", arr)
        for name, arr in members.items()
    }
    state["proj_out.weight"] = _bf16_weight(
        "proj_out.weight", rng.standard_normal((8, _DIM)).astype(np.float32)
    )

    out = _fuse_temb_projections(state)

    assert "temb_proj.linear.weight" in out
    assert "proj_out.weight" in out
    for name, _ in _MEMBERS:
        assert f"{name}.weight" not in out
    fused = out["temb_proj.linear.weight"]
    assert fused.dtype == DType.bfloat16
    assert tuple(int(d) for d in fused.shape) == (17 * _DIM, _DIM)
    np.testing.assert_array_equal(
        _as_u16(fused),
        np.concatenate(
            [
                (members[name].view(np.uint32) >> 16).astype(np.uint16)
                for name, _ in _MEMBERS
            ],
            axis=0,
        ),
    )


def test_fuse_temb_projections_handles_prefix_and_bias() -> None:
    """Prefixed keys fuse under the same prefix; biases fuse alongside."""
    rng = np.random.default_rng(11)
    members = _member_arrays(rng)
    state: dict[str, WeightData] = {}
    for name, out_dim in _MEMBERS:
        key = f"transformer.{name}"
        state[f"{key}.weight"] = _fp32_weight(f"{key}.weight", members[name])
        state[f"{key}.bias"] = _fp32_weight(
            f"{key}.bias", rng.standard_normal(out_dim).astype(np.float32)
        )

    out = _fuse_temb_projections(state)

    assert set(out) == {
        "transformer.temb_proj.linear.weight",
        "transformer.temb_proj.linear.bias",
    }
    assert tuple(
        int(d) for d in out["transformer.temb_proj.linear.bias"].shape
    ) == (17 * _DIM,)


def test_fuse_temb_projections_incomplete_group_raises() -> None:
    """A checkpoint carrying some members but not all is named, not loaded."""
    rng = np.random.default_rng(13)
    members = _member_arrays(rng)
    state = {
        f"{name}.weight": _fp32_weight(f"{name}.weight", arr)
        for name, arr in members.items()
    }
    del state["norm_out.linear.weight"]
    with pytest.raises(ValueError, match="norm_out.linear.weight"):
        _fuse_temb_projections(state)


def test_fuse_temb_projections_without_group_passes_through() -> None:
    """A state dict without the members (e.g. the VAE's) is untouched."""
    state = {
        "encoder.conv_in.weight": _fp32_weight(
            "encoder.conv_in.weight", np.ones((4, 4), dtype=np.float32)
        )
    }
    assert _fuse_temb_projections(dict(state)).keys() == state.keys()


def test_adapt_weights_fuses_temb_and_still_splits_stacked_qkv() -> None:
    """The unquantized path emits split Q/K/V and the fused temb row.

    The QKV split staying split is load-bearing: the fused-QKV variant was
    refuted by the materialize A/B, so the model keeps three attention
    projections while temb fuses -- the adapter must do exactly both.
    """
    rng = np.random.default_rng(17)
    members = _member_arrays(rng)
    state = {
        f"{name}.weight": _fp32_weight(f"{name}.weight", arr)
        for name, arr in members.items()
    }
    stacked = rng.standard_normal((3 * _DIM, _DIM)).astype(np.float32)
    key = "transformer_blocks.0.attn.qkv_proj.weight"
    state[key] = _fp32_weight(key, stacked)

    out = adapt_weights(state, None, None)

    assert "temb_proj.linear.weight" in out
    for member in ("to_q", "to_k", "to_v"):
        assert f"transformer_blocks.0.attn.{member}.weight" in out, sorted(out)
    assert key not in out


def test_split_then_fuse_temb_round_trip_is_byte_stable() -> None:
    """Fusing is deterministic over a re-split of its own output shapes."""
    rng = np.random.default_rng(19)
    members = _member_arrays(rng)
    state = {
        f"{name}.weight": _bf16_weight(f"{name}.weight", arr)
        for name, arr in members.items()
    }
    once = _fuse_temb_projections(dict(state))["temb_proj.linear.weight"]
    again = _fuse_temb_projections(
        {
            f"{name}.weight": _bf16_weight(f"{name}.weight", members[name])
            for name, _ in _MEMBERS
        }
    )["temb_proj.linear.weight"]
    np.testing.assert_array_equal(_as_u16(once), _as_u16(again))
    # _split_stacked_qkv shares the concat orientation (axis 0 of [N, K]);
    # a three-way even split of a synthetic stacked weight round-trips, so
    # the two helpers agree on the layout they assume.
    qkv = _fp32_weight(
        "a.attn.qkv_proj.weight",
        np.arange(3 * _DIM * _DIM, dtype=np.float32).reshape(3 * _DIM, _DIM),
    )
    split = _split_stacked_qkv({"a.attn.qkv_proj.weight": qkv})
    np.testing.assert_array_equal(
        np.concatenate(
            [
                split[f"a.attn.{m}.weight"].to_buffer().to_numpy()
                for m in ("to_q", "to_k", "to_v")
            ],
            axis=0,
        ),
        qkv.to_buffer().to_numpy(),
    )


def _execute(graph: Graph, registry: dict, inputs: list[np.ndarray]):
    session = InferenceSession(devices=[CPU()])
    model = session.load(graph, weights_registry=registry)
    return [np.from_dlpack(t) for t in model.execute(*inputs)]


def test_staged_parity_fused_vs_member_projections() -> None:
    """The fused staging is bit-identical to the four member projections.

    Reference arm: the pre-fusion math, staged from raw ops against the
    member weights as constants -- one ``silu(temb) @ W_i^T`` per member,
    the old ``Flux2Modulation`` split, and norm_out's chunk + modulate.
    Fused arm: the shipping code path end to end, its single weight coming
    from the real adapter. All fifteen modulation parameters and the
    norm_out application must match exactly.
    """
    rng = np.random.default_rng(23)
    members = _member_arrays(rng)
    temb_np = rng.standard_normal((_BATCH, _DIM)).astype(np.float32)
    x_np = rng.standard_normal((_BATCH, _SEQ, _DIM)).astype(np.float32)

    input_types = [
        TensorType(DType.float32, [_BATCH, _DIM], DeviceRef.CPU()),
        TensorType(DType.float32, [_BATCH, _SEQ, _DIM], DeviceRef.CPU()),
    ]

    def _norm(x: TensorValue) -> TensorValue:
        ones = ops.constant(1.0, x.dtype, DeviceRef.CPU()).broadcast_to([_DIM])
        zeros = ops.constant(0.0, x.dtype, DeviceRef.CPU()).broadcast_to([_DIM])
        return ops.layer_norm(x, gamma=ones, beta=zeros, epsilon=1e-6)

    with Graph(
        "temb_reference",
        input_types=input_types,
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as ref_graph:
        temb, x = (v.tensor for v in ref_graph.inputs)
        outs: list[TensorValue] = []
        for name, sets in (
            ("double_stream_modulation_img.linear", 2),
            ("double_stream_modulation_txt.linear", 2),
            ("single_stream_modulation.linear", 1),
        ):
            w = ops.constant(members[name], DType.float32, DeviceRef.CPU())
            mod = ops.silu(temb) @ ops.transpose(w, -1, -2)
            mod = ops.unsqueeze(mod, 1)
            parts = ops.split(mod, [_DIM] * (3 * sets), axis=-1)
            outs.extend(parts)
        w4 = ops.constant(
            members["norm_out.linear"], DType.float32, DeviceRef.CPU()
        )
        emb = ops.silu(ops.cast(temb, x.dtype)) @ ops.transpose(w4, -1, -2)
        scale, shift = ops.chunk(emb, chunks=2, axis=1)
        outs.append(
            ada_modulate(_norm(x), scale[:, None, :], shift[:, None, :])
        )
        ref_graph.output(*outs)

    fused_weight = _fuse_temb_projections(
        {
            f"{name}.weight": _fp32_weight(f"{name}.weight", arr)
            for name, arr in members.items()
        }
    )["temb_proj.linear.weight"]

    temb_proj = Flux2TembProjections(
        _DIM, dtype=DType.float32, device=DeviceRef.CPU()
    )
    norm_out = AdaLayerNormContinuous(
        embedding_dim=_DIM,
        conditioning_embedding_dim=_DIM,
        elementwise_affine=False,
        dtype=DType.float32,
        device=DeviceRef.CPU(),
        eps=1e-6,
        bias=False,
        external_projection=True,
    )
    for name, weight, _layer in temb_proj._iter_named_weights():
        weight.name = f"temb_proj.{name}"

    with Graph(
        "temb_fused",
        input_types=input_types,
        strict_device_placement=DevicePlacementPolicy.Error,
    ) as fused_graph:
        temb, x = (v.tensor for v in fused_graph.inputs)
        img_mod, txt_mod, single_mod, norm_out_emb = temb_proj(temb)
        outs = []
        for mod, sets in ((img_mod, 2), (txt_mod, 2), (single_mod, 1)):
            for triple in _modulation_param_sets(mod, _DIM, sets):
                outs.extend(triple)
        outs.append(norm_out.apply_projected(x, norm_out_emb))
        fused_graph.output(*outs)

    ref_out = _execute(ref_graph, {}, [temb_np, x_np])
    fused_out = _execute(
        fused_graph,
        {"temb_proj.linear.weight": fused_weight.to_buffer().to_numpy()},
        [temb_np, x_np],
    )

    assert len(ref_out) == len(fused_out) == 16
    for i, (r, f) in enumerate(zip(ref_out, fused_out, strict=True)):
        np.testing.assert_array_equal(
            r, f, err_msg=f"output {i} diverged (fp32 exact)"
        )
