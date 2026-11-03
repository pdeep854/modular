"""The adaLN modulation crossing, and the chunk order it stands on.

The 17 vectors the block stack reads are four ``nn.Linear`` outputs chunked
along the feature axis, and nothing about a chunk says which parameter it is.
Get the split wrong and every shape still matches: the wire is 17 x 3072
either way, the driver runs, an image comes out, and it is the wrong image.
So the order is not documented here, it is *proven* here, three ways:

* against diffusers itself -- the same ``Flux2Modulation`` /
  ``AdaLayerNormContinuous`` modules the checkpoint was written by, element
  for element (:func:`test_the_derivation_matches_diffusers`). This is the
  only test that can establish the order rather than restate it, so it is the
  one that skips when torch is missing rather than being written around.
* as load-bearing -- a permuted wire has to move the driver's velocity, or the
  order was never being read
  (:func:`test_the_chunk_order_is_load_bearing`).
* as delivered -- the driver reports ``modulation_source: request`` and its
  answer moves with the modulation, so the host's numbers really are the ones
  in play (:func:`test_real_modulation_reaches_the_driver`).

The driver comes from bazel, like ``test_device_bridge.py``::

    ./bazelw build --config=plugin \\
      @plugin//examples/hexagon-ondevice:npu_block_stack_reference
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from flux_klein_host.config import KLEIN_4B, KleinConfig
from flux_klein_host.device_bridge import (
    DeviceBlockStack,
    DeviceBridge,
    pack_modulation,
)
from flux_klein_host.interfaces import StepConditioning, StepModulation
from flux_klein_host.timestep_embed import (
    MODULATION_HEADS,
    MODULATION_VECTORS,
    ModulationWeights,
    derive_modulation,
)
from test_device_bridge import _driver, _mini_config, _request

pytestmark = pytest.mark.skipif(
    not _driver().is_file(),
    reason=(
        "no block-stack driver; build "
        "@plugin//examples/hexagon-ondevice:npu_block_stack_reference or set "
        "NPU_BLOCK_STACK_DRIVER"
    ),
)


def _modulated(
    config: KleinConfig, *, seed: int = 0, mod_seed: int = 7
) -> object:
    """One denoise step's request, carrying real derived modulation."""
    base = _request(config, seed=seed)
    weights = ModulationWeights.random(config, seed=mod_seed)
    return replace(
        base,
        conditioning=StepConditioning(
            step_index=base.conditioning.step_index,
            timestep=base.conditioning.timestep,
            temb=base.conditioning.temb,
            modulation=derive_modulation(base.conditioning.temb, weights),
        ),
    )


def _backend(
    tmp_path: Path, config: KleinConfig, **kwargs: object
) -> DeviceBlockStack:
    return DeviceBlockStack(
        _driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_the_derivation_matches_diffusers() -> None:
    """The chunk order is the checkpoint's, not one this scaffold invented.

    Builds the four heads as ``transformer_flux2`` builds them, unpacks them
    with diffusers' own ``Flux2Modulation.split`` and ``AdaLayerNormContinuous``
    chunk, and requires every one of the 17 vectors to agree. The last two are
    the ones worth staring at: ``norm_out`` reads ``scale`` *before* ``shift``,
    the opposite of every block head, and a plausible-looking wire that had
    them the other way round would be wrong everywhere the final norm runs.
    """
    torch = pytest.importorskip("torch")
    flux2 = pytest.importorskip(
        "diffusers.models.transformers.transformer_flux2"
    )
    normalization = pytest.importorskip("diffusers.models.normalization")

    torch.manual_seed(0)
    dim = 64
    # The mirror runs in float16 like the pipeline (and like the host chain
    # since the D3 re-record patch); weights widen to float32 losslessly.
    img = flux2.Flux2Modulation(dim, mod_param_sets=2, bias=False).half()
    txt = flux2.Flux2Modulation(dim, mod_param_sets=2, bias=False).half()
    single = flux2.Flux2Modulation(dim, mod_param_sets=1, bias=False).half()
    norm_out = normalization.AdaLayerNormContinuous(
        dim, dim, elementwise_affine=False, eps=1e-6, bias=False
    ).half()
    temb = torch.randn(1, dim).half()

    def rows(module: object) -> npt.NDArray[np.float32]:
        return (
            module.linear.weight.detach().numpy().astype(np.float32)  # type: ignore[attr-defined]
        )

    got = derive_modulation(
        temb.numpy().astype(np.float32),
        ModulationWeights(
            double_img=rows(img),
            double_txt=rows(txt),
            single=rows(single),
            norm_out=rows(norm_out),
        ),
    )

    split = flux2.Flux2Modulation.split
    (i_attn, i_mlp) = split(img(temb), 2)
    (t_attn, t_mlp) = split(txt(temb), 2)
    (s_set,) = split(single(temb), 1)
    out_scale, out_shift = torch.chunk(
        norm_out.linear(norm_out.silu(temb)), 2, dim=1
    )

    want = [
        *i_attn,
        *i_mlp,
        *t_attn,
        *t_mlp,
        *s_set,
        out_scale,
        out_shift,
    ]
    mine = [
        got.double_img[0].shift,
        got.double_img[0].scale,
        got.double_img[0].gate,
        got.double_img[1].shift,
        got.double_img[1].scale,
        got.double_img[1].gate,
        got.double_txt[0].shift,
        got.double_txt[0].scale,
        got.double_txt[0].gate,
        got.double_txt[1].shift,
        got.double_txt[1].scale,
        got.double_txt[1].gate,
        got.single.shift,
        got.single.scale,
        got.single.gate,
        got.out_scale,
        got.out_shift,
    ]
    assert len(want) == MODULATION_VECTORS == len(mine)
    for index, (produced, expected) in enumerate(zip(mine, want, strict=False)):
        # `Flux2Modulation.split` unsqueezes a token axis for broadcasting;
        # the host keeps its vectors `(B, dim)`, so compare the values.
        # The tolerance covers accumulation-order rounding between torch's
        # fp16 linears and the host's (both store fp16, both accumulate
        # wide): a couple of fp16 ulps at these O(1) magnitudes. A swapped
        # or miswired vector differs at O(0.3).
        np.testing.assert_allclose(
            produced.reshape(dim),
            expected.detach().numpy().astype(np.float32).reshape(dim),
            rtol=0,
            atol=2e-3,
            err_msg=f"modulation vector {index} disagrees with diffusers",
        )

    # The control on the control: the two norm_out chunks are not equal, so
    # the assertion above would have caught the swapped order rather than
    # passing on a coincidence.
    assert not np.allclose(got.out_scale, got.out_shift)


def test_the_heads_are_the_checkpoints_shapes() -> None:
    """Each head's row count is its chunk count times the DiT width."""
    weights = ModulationWeights.random(KLEIN_4B)
    for (name, chunks), head in zip(
        MODULATION_HEADS, weights.heads, strict=False
    ):
        assert head.shape == (chunks * KLEIN_4B.dim, KLEIN_4B.dim), name
    assert MODULATION_VECTORS == 17


def test_a_short_modulation_vector_is_refused() -> None:
    """A wire the driver would misread has to fail on the host side."""
    config = _mini_config()
    weights = ModulationWeights.random(config)
    good = derive_modulation(
        np.zeros((1, config.dim), dtype=np.float32), weights
    )
    truncated = StepModulation(
        double_img=good.double_img,
        double_txt=good.double_txt,
        single=good.single,
        out_scale=good.out_scale[:, :-1],
        out_shift=good.out_shift,
    )
    with pytest.raises(ValueError, match="must be"):
        pack_modulation(truncated, config.dim)


def test_real_modulation_reaches_the_driver(tmp_path: Path) -> None:
    """The host's 17 vectors cross, and the driver says they are the ones used.

    Without this the driver would fall back to its own synthetic modulation and
    still return a perfectly plausible velocity -- the failure that looks like
    success right up to the image.
    """
    config = _mini_config()
    stack = _backend(tmp_path, config, keep_scratch=True)

    velocity = stack(_modulated(config))

    response = stack.bridge.last_response
    assert response is not None
    assert response.modulation_source == "request"
    assert response.output_source == "host_reference"
    assert np.isfinite(velocity).all()

    scratch = stack.bridge.last_scratch
    assert scratch is not None
    manifest = json.loads((scratch / "request.json").read_text())
    assert manifest["inputs"]["modulation"]["shape"] == [
        MODULATION_VECTORS,
        config.dim,
    ]

    # Absent modulation must still work, and must be reported differently.
    plain = _backend(tmp_path, config)
    plain(_request(config))
    plain_response = plain.bridge.last_response
    assert plain_response is not None
    assert plain_response.modulation_source == "synthetic"


def test_the_velocity_moves_with_the_modulation(tmp_path: Path) -> None:
    """Two modulations, two answers: the vectors are not being ignored."""
    config = _mini_config()
    stack = _backend(tmp_path, config)

    first = stack(_modulated(config, mod_seed=7))
    same = stack(_modulated(config, mod_seed=7))
    other = stack(_modulated(config, mod_seed=8))

    np.testing.assert_array_equal(first, same)
    assert not np.allclose(first, other)


def test_the_chunk_order_is_load_bearing(tmp_path: Path) -> None:
    """A permuted wire must produce a different velocity.

    This is what makes the order a claim with teeth. Each permutation below
    keeps the 17 vectors and their values and changes only which slot each one
    lands in -- exactly the mistake a wrong reading of the chunk order would
    make -- and every one of them has to move the driver's answer. Were the
    driver indifferent to the order, the confirmation from diffusers would be
    describing something nothing downstream actually reads.
    """
    config = _mini_config()
    request = _modulated(config)
    modulation = request.conditioning.modulation
    assert modulation is not None
    wire = pack_modulation(modulation, config.dim)

    geometry = {
        "batch": 1,
        "heads": config.num_attention_heads,
        "head_dim": config.attention_head_dim,
        "text_seq": config.text_seq_len,
        "image_seq": request.geometry.image_seq_len,
        "in_channels": config.in_channels,
        "joint_dim": config.joint_attention_dim,
        "mlp_ratio": int(config.mlp_ratio),
        "num_double": config.num_layers,
        "num_single": config.num_single_layers,
    }
    shape = (request.geometry.image_seq_len, config.in_channels)
    base_inputs: dict[str, npt.NDArray[object]] = {
        "latent": np.ascontiguousarray(
            request.latent.reshape(-1, config.in_channels), dtype=np.float32
        ),
        "prompt_embeds": np.ascontiguousarray(
            request.text.prompt_embeds.reshape(-1, config.joint_attention_dim),
            dtype=np.float32,
        ),
        "temb": np.ascontiguousarray(
            request.conditioning.temb, dtype=np.float32
        ),
        "image_ids": np.ascontiguousarray(request.image_ids[0], dtype=np.int64),
        "text_ids": np.ascontiguousarray(
            request.text.text_ids[0], dtype=np.int64
        ),
    }

    def velocity_for(vectors: npt.NDArray[np.float32]) -> npt.NDArray[object]:
        bridge = DeviceBridge(
            executable=_driver(), scratch_root=tmp_path, interop=False
        )
        values, response = bridge.invoke(
            "dit_block_stack",
            geometry,
            {**base_inputs, "modulation": vectors},
            {"velocity": (shape, "float32")},
        )
        assert response.modulation_source == "request"
        return values["velocity"]

    reference = velocity_for(wire)

    permutations = {
        # (shift, scale) read as (scale, shift) in the first double triple.
        "double_img_attn shift<->scale": (0, 1),
        # The image head's attention set read as its MLP set.
        "double_img attn<->mlp gate": (2, 5),
        # The two streams' heads swapped.
        "double img<->txt attn shift": (0, 6),
        # norm_out's (scale, shift), the one place the convention flips.
        "norm_out scale<->shift": (15, 16),
    }
    for label, (left, right) in permutations.items():
        swapped = wire.copy()
        swapped[[left, right]] = swapped[[right, left]]
        # The permutation has to actually change the wire, or the assertion
        # below would pass on two identical vectors rather than on the order.
        assert not np.array_equal(swapped, wire), label
        assert not np.allclose(velocity_for(swapped), reference), (
            f"permuting {label} did not move the velocity, so the driver is "
            "not reading the chunk order the host packs"
        )
