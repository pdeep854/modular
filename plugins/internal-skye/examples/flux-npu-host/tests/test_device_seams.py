"""The encoder and VAE seams on the device-dispatch bridge, with no NPU.

``test_device_bridge.py`` proves the crossing for the DiT block stack. These
prove the other two seams reuse it: the same ``DeviceBridge``, the same
manifest, the same ``response.json``, against the host-only builds of the real
drivers -- so the marshalling under test is the one the Windows ``.exe`` runs.

Each seam is checked at both levels, because they falsify different things:

* ``--plan-only`` proves the round trip is WIRED: files land, the driver
  returns 0, an output of the right shape comes back, and with ``--weights``
  every tensor the seam needs is resolved out of the converted artifact. Its
  output is a placeholder, so it cannot tell a delivered input from an ignored
  one.
* the host-reference run proves the round trip is CONNECTED: the output is a
  function of the marshalled input, so moving one on the host side has to move
  it. That is the failure a shape-only check cannot catch, and the one that
  would survive all the way to a wrong image.

The drivers come from bazel::

    ./bazelw build --config=plugin \\
      @plugin//examples/hexagon-ondevice:npu_qwen3_encoder_stack_reference \\
      @plugin//examples/hexagon-ondevice:npu_vae_decoder_reference

Point ``NPU_ENCODER_STACK_DRIVER`` / ``NPU_VAE_DECODER_DRIVER`` at them (or let
the tests find them under ``bazel-bin``); without them they skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from flux_klein_host.config import KleinConfig, resolve_geometry
from flux_klein_host.device_bridge import (
    BridgeError,
    DeviceTextEncoder,
    DeviceVaeDecoder,
)
from flux_klein_host.interfaces import VaeDecodeInput
from flux_klein_host.stages import PromptTokens, TextEncodeRequest
from test_device_bridge import _repo_root

_BAZEL = Path("bazel-bin/external/plugin+/examples/hexagon-ondevice")

#: The real converted Klein artifact, when this checkout has one. The
#: weights-resolution tests need it and skip without it -- they are the only
#: ones here that say anything about the real model's tensor names.
KLEIN_MANIFEST = Path(
    "/home/fabio/work/flux2-klein-4b-npu/klein-4b-npu-fp16.json"
)


def _driver(env: str, name: str) -> Path:
    override = os.environ.get(env)
    return Path(override) if override else _repo_root() / _BAZEL / name


def _encoder_driver() -> Path:
    return _driver(
        "NPU_ENCODER_STACK_DRIVER", "npu_qwen3_encoder_stack_reference"
    )


def _vae_driver() -> Path:
    return _driver("NPU_VAE_DECODER_DRIVER", "npu_vae_decoder_reference")


def _mini_encoder_config() -> KleinConfig:
    """Klein's shape, shrunk until a host reference costs seconds.

    ``head_dim`` stays 128 -- the masked-attention kernel bakes it, and so does
    Qwen3. Every other width shrinks, keeping each matmul ``K`` a multiple of
    64 (the HVX fp16 vector), which is what the driver's plan check asserts.
    """
    return KleinConfig(
        num_attention_heads=4,
        attention_head_dim=128,
        text_seq_len=32,
        text_hidden_size=512,
        joint_attention_dim=3 * 512,
        text_hidden_state_layers=(1, 2, 3),
        text_num_attention_heads=4,
        text_num_key_value_heads=2,
        text_head_dim=128,
        text_intermediate_size=1536,
        num_layers=1,
        num_single_layers=1,
    )


def _mini_vae_config() -> KleinConfig:
    """The decoder's four stages in miniature, so the whole walk still runs."""
    return KleinConfig(
        vae_block_out_channels=(32, 32, 64, 64),
        vae_layers_per_block=1,
    )


def _tokens(config: KleinConfig, *, seed: int = 0) -> PromptTokens:
    """One tokenized prompt, right-padded, with its causal + padding bias."""
    seq = config.text_seq_len
    valid = seq - 4
    rng = np.random.RandomState(seed)
    ids = rng.randint(1, 1000, size=(seq,)).astype(np.int32)
    ids[valid:] = 0
    mask = np.zeros((seq,), dtype=np.uint8)
    mask[:valid] = 1
    bias = np.zeros((1, 1, seq, seq), dtype=np.float32)
    for query in range(seq):
        bias[0, 0, query, query + 1 :] = -10000.0
        bias[0, 0, query, valid:] = -10000.0
    return PromptTokens(
        tokens=ids,
        attention_mask=mask,
        attention_bias=bias,
        num_valid_tokens=valid,
        rendered="a red bicycle",
    )


def _latent(
    config: KleinConfig, *, seed: int = 0, scale: float = 1.0
) -> VaeDecodeInput:
    """One decoder-facing latent: already denormalized, already unpacked."""
    geometry = resolve_geometry(config, 64, 64)
    rng = np.random.RandomState(seed)
    return VaeDecodeInput(
        latent=(
            rng.standard_normal((1, *geometry.unpacked_shape)) * scale
        ).astype(np.float32),
        geometry=geometry,
    )


encoder = pytest.mark.skipif(
    not _encoder_driver().is_file(),
    reason=(
        "no encoder driver; build "
        "@plugin//examples/hexagon-ondevice:npu_qwen3_encoder_stack_reference "
        "or set NPU_ENCODER_STACK_DRIVER"
    ),
)
vae = pytest.mark.skipif(
    not _vae_driver().is_file(),
    reason=(
        "no VAE driver; build "
        "@plugin//examples/hexagon-ondevice:npu_vae_decoder_reference or set "
        "NPU_VAE_DECODER_DRIVER"
    ),
)
artifact = pytest.mark.skipif(
    not KLEIN_MANIFEST.is_file(),
    reason=f"no converted Klein artifact at {KLEIN_MANIFEST}",
)


# --------------------------------------------------------------------------
# The encoder seam
# --------------------------------------------------------------------------


@encoder
def test_encoder_plan_only_round_trip(tmp_path: Path) -> None:
    """Marshal out, exec, read back: the crossing runs and returns a shape."""
    config = _mini_encoder_config()
    stage = DeviceTextEncoder(
        _encoder_driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
        keep_scratch=True,
    )

    text = stage(TextEncodeRequest(tokens=_tokens(config)))

    text.validate(config)
    assert text.prompt_embeds.shape == (
        1,
        config.text_seq_len,
        config.joint_attention_dim,
    )
    # A plan-only output is a placeholder, and the driver says so rather than
    # leaving the caller to guess from the values.
    assert not text.prompt_embeds.any()
    response = stage.bridge.last_response
    assert response is not None
    assert response.status == "ok"
    assert response.plan_only
    assert response.output_source == "none"
    assert not response.computed


@encoder
def test_encoder_marshals_the_whole_seam(tmp_path: Path) -> None:
    """The scratch directory holds the request, self-described, and echoed."""
    config = _mini_encoder_config()
    stage = DeviceTextEncoder(
        _encoder_driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
        keep_scratch=True,
    )
    stage(TextEncodeRequest(tokens=_tokens(config)))

    scratch = stage.bridge.last_scratch
    assert scratch is not None
    manifest = json.loads((scratch / "request.json").read_text())
    assert manifest["format"] == "skye-npu-io"
    assert manifest["seam"] == "qwen3_encoder"
    assert set(manifest["inputs"]) == {"tokens", "attention_bias"}
    for spec in manifest["inputs"].values():
        payload = scratch / spec["path"]
        itemsize = np.dtype(spec["dtype"]).itemsize
        assert payload.stat().st_size == int(np.prod(spec["shape"])) * itemsize

    # The geometry the host asked for is the geometry the driver reports back,
    # so a driver silently running its own default shape would be caught. The
    # selection crosses as `num_select` + `select_i`, and has to survive.
    response = json.loads((scratch / "response.json").read_text())
    assert response["geometry"] == manifest["geometry"]
    assert response["geometry"]["num_select"] == 3
    assert [response["geometry"][f"select_{i}"] for i in range(3)] == [1, 2, 3]


@encoder
def test_encoder_placeholder_is_not_handed_back_as_a_result(
    tmp_path: Path,
) -> None:
    """A plan-only response must not satisfy a caller that wanted numbers."""
    config = _mini_encoder_config()
    # `plan_only` on the driver but not on the backend: the crossing returns a
    # placeholder and nobody asked for one.
    stage = DeviceTextEncoder(
        _encoder_driver(), config, scratch_root=tmp_path, interop=False
    )
    stage.bridge.plan_only = True
    with pytest.raises(BridgeError, match="placeholders"):
        stage(TextEncodeRequest(tokens=_tokens(config)))


@encoder
def test_encoder_output_depends_on_the_tokens(tmp_path: Path) -> None:
    """The crossing DELIVERS: change the prompt, the answer has to move.

    A shape-only round trip passes just as happily when the driver ignores
    everything the host sent. Here the driver's fp32 reference computes the
    hidden states from the marshalled request, so two different token
    sequences cannot produce one answer.
    """
    config = _mini_encoder_config()
    stage = DeviceTextEncoder(
        _encoder_driver(), config, scratch_root=tmp_path, interop=False
    )

    first = stage(TextEncodeRequest(tokens=_tokens(config, seed=0)))
    same = stage(TextEncodeRequest(tokens=_tokens(config, seed=0)))
    other = stage(TextEncodeRequest(tokens=_tokens(config, seed=1)))

    response = stage.bridge.last_response
    assert response is not None
    assert response.output_source == "host_reference"
    assert np.isfinite(first.prompt_embeds).all()

    np.testing.assert_array_equal(first.prompt_embeds, same.prompt_embeds)
    assert not np.allclose(first.prompt_embeds, other.prompt_embeds)
    # Without an artifact there is no embedding table, so the driver's stand-in
    # is seeded per token rather than looked up. It says so, and that is what
    # keeps the assertion above honest: the tokens really did reach the
    # arithmetic, they just did not index a real vocabulary.
    assert response.raw["embedding_source"] == "synthetic"


@encoder
def test_encoder_output_depends_on_the_attention_bias(tmp_path: Path) -> None:
    """The padding mask crosses too, and it changes the answer."""
    config = _mini_encoder_config()
    stage = DeviceTextEncoder(
        _encoder_driver(), config, scratch_root=tmp_path, interop=False
    )

    base = _tokens(config)
    shorter = PromptTokens(
        tokens=base.tokens,
        attention_mask=base.attention_mask,
        # Half the prompt masked off: the same shape, a different mask.
        attention_bias=np.minimum(
            base.attention_bias,
            np.where(
                np.arange(config.text_seq_len) < config.text_seq_len // 2,
                np.float32(0.0),
                np.float32(-10000.0),
            ).astype(np.float32),
        ),
        num_valid_tokens=base.num_valid_tokens,
        rendered=base.rendered,
    )
    first = stage(TextEncodeRequest(tokens=base)).prompt_embeds
    masked = stage(TextEncodeRequest(tokens=shorter)).prompt_embeds
    assert not np.allclose(first, masked)


@encoder
@artifact
def test_encoder_resolves_the_real_klein_weights(tmp_path: Path) -> None:
    """Every Qwen3 tensor the stack needs is in the converted artifact.

    Full Klein geometry -- 28 layers, hidden 2560, 32:8 GQA -- and no NPU: the
    manifest is read and every tensor shape-checked against what the geometry
    implies, which is what a plan-only run can prove about a 16 GiB artifact
    without reading it. The embedding gather is real here, so
    ``embedding_source`` is ``artifact`` rather than ``synthetic``.
    """
    config = KleinConfig()
    stage = DeviceTextEncoder(
        _encoder_driver(),
        config,
        weights=KLEIN_MANIFEST,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
    )

    text = stage(TextEncodeRequest(tokens=_tokens(config)))

    assert text.prompt_embeds.shape == (1, 512, 7680)
    response = stage.bridge.last_response
    assert response is not None
    assert response.weight_source == "artifact"
    assert response.raw["embedding_source"] == "artifact"
    assert response.raw["geometry"]["layers"] == 28


# --------------------------------------------------------------------------
# The VAE seam
# --------------------------------------------------------------------------


@vae
def test_vae_plan_only_round_trip(tmp_path: Path) -> None:
    """Marshal out, exec, read back: the crossing runs and returns a shape."""
    config = _mini_vae_config()
    stage = DeviceVaeDecoder(
        _vae_driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
        keep_scratch=True,
    )
    request = _latent(config)

    rgb = stage(request)

    scale = config.vae_scale_factor
    assert rgb.shape == (
        1,
        3,
        request.geometry.latent_height * scale,
        request.geometry.latent_width * scale,
    )
    assert not rgb.any()
    response = stage.bridge.last_response
    assert response is not None
    assert response.plan_only
    assert response.output_source == "none"
    assert not response.computed


@vae
def test_vae_marshals_the_whole_seam(tmp_path: Path) -> None:
    """The scratch directory holds the request, self-described, and echoed."""
    config = _mini_vae_config()
    stage = DeviceVaeDecoder(
        _vae_driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
        keep_scratch=True,
    )
    stage(_latent(config))

    scratch = stage.bridge.last_scratch
    assert scratch is not None
    manifest = json.loads((scratch / "request.json").read_text())
    assert manifest["seam"] == "vae_decoder"
    assert set(manifest["inputs"]) == {"latent"}
    spec = manifest["inputs"]["latent"]
    assert (scratch / spec["path"]).stat().st_size == int(
        np.prod(spec["shape"])
    ) * 4

    response = json.loads((scratch / "response.json").read_text())
    assert response["geometry"] == manifest["geometry"]
    # `block_out` crosses in ENCODER order, as `vae/config.json` writes it; the
    # driver walks it reversed and must not hand back its own order.
    assert [
        response["geometry"][f"block_out_{i}"]
        for i in range(response["geometry"]["stages"])
    ] == list(config.vae_block_out_channels)


@vae
def test_vae_rgb_depends_on_the_latent(tmp_path: Path) -> None:
    """The crossing DELIVERS: change the latent, the image has to move."""
    config = _mini_vae_config()
    stage = DeviceVaeDecoder(
        _vae_driver(), config, scratch_root=tmp_path, interop=False
    )

    first = stage(_latent(config, seed=0))
    same = stage(_latent(config, seed=0))
    other = stage(_latent(config, seed=1))

    response = stage.bridge.last_response
    assert response is not None
    assert response.output_source == "host_reference"
    assert np.isfinite(first).all()

    np.testing.assert_array_equal(first, same)
    assert not np.allclose(first, other)


@vae
def test_vae_a_shape_disagreement_fails_at_the_seam(tmp_path: Path) -> None:
    """A manifest that lies about the latent is caught, not reinterpreted.

    Driven through a raw :class:`DeviceBridge` rather than the backend, because
    the backend's own ``validate`` would reject this first -- and the thing
    under test is that the DRIVER checks too. A host and a driver that only
    agree because one of them never looked is the failure this rules out.
    """
    config = _mini_vae_config()
    stage = DeviceVaeDecoder(
        _vae_driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
    )
    request = _latent(config)
    channels, height, width = request.geometry.unpacked_shape
    short: npt.NDArray[np.float32] = np.zeros(
        (channels, height - 1, width), dtype=np.float32
    )
    with pytest.raises(BridgeError, match="the geometry needs"):
        stage.bridge.invoke(
            "vae_decoder",
            stage.geometry(request.geometry),
            {"latent": short},
            {"rgb": ((1, 3, height * 8, width * 8), "float32")},
            require_computed=False,
        )


@vae
@artifact
def test_vae_resolves_the_real_klein_weights(tmp_path: Path) -> None:
    """Every decoder tensor the walk needs is in the converted artifact.

    Full Klein geometry -- a 64x64 latent to a 512x512 image, four stages with
    the narrowing shortcuts and the mid-block attention -- resolved and
    shape-checked with no NPU and without reading the 16 GiB ``.bin``.
    """
    config = KleinConfig()
    stage = DeviceVaeDecoder(
        _vae_driver(),
        config,
        weights=KLEIN_MANIFEST,
        scratch_root=tmp_path,
        interop=False,
        plan_only=True,
    )

    rgb = stage(_latent_at(config, 512))

    assert rgb.shape == (1, 3, 512, 512)
    response = stage.bridge.last_response
    assert response is not None
    assert response.weight_source == "artifact"
    assert response.raw["geometry"]["latent_height"] == 64


def _latent_at(config: KleinConfig, size: int) -> VaeDecodeInput:
    """A decoder-facing latent for a square image of ``size`` pixels."""
    geometry = resolve_geometry(config, size, size)
    return VaeDecodeInput(
        latent=np.zeros((1, *geometry.unpacked_shape), dtype=np.float32),
        geometry=geometry,
    )
