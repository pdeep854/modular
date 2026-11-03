"""Validates the timestep embedding against diffusers and a closed form."""

from __future__ import annotations

import math

import numpy as np
import pytest
from flux_klein_host import (
    KLEIN_4B,
    TIMESTEP_SCALE,
    FlowMatchEulerScheduler,
    TimestepEmbedderWeights,
    embed_timestep,
    sinusoidal_embedding,
)
from references import diffusers_available

requires_diffusers = pytest.mark.skipif(
    not diffusers_available(), reason="torch/diffusers reference not installed"
)

# Klein's settings, from transformer/config.json plus the Timesteps
# construction in MAX's Flux2TimestepGuidanceEmbeddings.
EMBED_DIM = KLEIN_4B.timestep_guidance_channels  # 256
FLIP_SIN_TO_COS = True
DOWNSCALE_FREQ_SHIFT = 0.0


def _klein_timesteps() -> np.ndarray:
    """The 4 timesteps a real 512^2 request embeds, in training units."""
    scheduler = FlowMatchEulerScheduler()
    sigmas = scheduler.sigmas(1024, 4)
    return (scheduler.timesteps(sigmas) * TIMESTEP_SCALE).astype(np.float32)


def test_shape_and_dtype() -> None:
    emb = sinusoidal_embedding(_klein_timesteps(), EMBED_DIM)
    assert emb.shape == (4, EMBED_DIM)
    assert emb.dtype == np.float32


# Agreement with torch is bounded by torch's own float32 trig, not by the
# formula. At the top of the range the pre-trig argument reaches 1000, where
# torch's float32 range reduction carries ~3e-5 of error while numpy's carries
# ~5e-8 (both measured against float64 evaluation of the same arguments). So
# the cross-check runs at a libm-scale tolerance, and
# ``test_sinusoidal_matches_float64_ground_truth`` is what actually pins our
# accuracy.
LIBM_ATOL = 1e-4


@requires_diffusers
def test_sinusoidal_matches_diffusers_on_the_klein_schedule() -> None:
    import torch
    from diffusers.models.embeddings import get_timestep_embedding

    timesteps = _klein_timesteps()
    reference = get_timestep_embedding(
        torch.from_numpy(timesteps),
        EMBED_DIM,
        flip_sin_to_cos=FLIP_SIN_TO_COS,
        downscale_freq_shift=DOWNSCALE_FREQ_SHIFT,
        scale=1.0,
        max_period=10000,
    )
    np.testing.assert_allclose(
        sinusoidal_embedding(timesteps, EMBED_DIM),
        reference.cpu().numpy(),
        rtol=1e-3,
        atol=LIBM_ATOL,
    )


def test_sinusoidal_matches_float64_ground_truth() -> None:
    """Pins our accuracy independently of any framework's float32 libm.

    Evaluates the same formula in float64 and requires float32-rounding
    agreement. torch's ``get_timestep_embedding`` does not pass at this
    tolerance, which is why the diffusers cross-check above is looser.
    """
    timesteps = np.linspace(0.0, 1000.0, 33, dtype=np.float32)
    half = EMBED_DIM // 2

    exponent = -math.log(10000) * np.arange(half, dtype=np.float32)
    freqs = np.exp((exponent / np.float32(half)).astype(np.float32))
    args = (timesteps[:, None] * freqs[None, :]).astype(np.float32)
    exact = np.concatenate(
        [
            np.cos(args.astype(np.float64)),
            np.sin(args.astype(np.float64)),
        ],
        axis=-1,
    )

    np.testing.assert_allclose(
        sinusoidal_embedding(timesteps, EMBED_DIM), exact, rtol=0, atol=1e-6
    )


@requires_diffusers
@pytest.mark.parametrize(
    "timesteps",
    [
        np.array([0.0], dtype=np.float32),
        np.array([1000.0], dtype=np.float32),
        np.linspace(0.0, 1000.0, 17, dtype=np.float32),
    ],
    ids=["t0", "t1000", "sweep"],
)
def test_sinusoidal_matches_diffusers_across_the_range(
    timesteps: np.ndarray,
) -> None:
    """Endpoints included: t=0 is where a sign or shift error hides."""
    import torch
    from diffusers.models.embeddings import get_timestep_embedding

    reference = get_timestep_embedding(
        torch.from_numpy(timesteps),
        EMBED_DIM,
        flip_sin_to_cos=FLIP_SIN_TO_COS,
        downscale_freq_shift=DOWNSCALE_FREQ_SHIFT,
    )
    np.testing.assert_allclose(
        sinusoidal_embedding(timesteps, EMBED_DIM),
        reference.cpu().numpy(),
        rtol=1e-3,
        atol=LIBM_ATOL,
    )


def test_sinusoidal_closed_form_at_t_zero() -> None:
    """At t=0 every argument is 0, so the flipped layout is [1...1, 0...0]."""
    emb = sinusoidal_embedding(np.zeros(1, dtype=np.float32), EMBED_DIM)
    half = EMBED_DIM // 2
    np.testing.assert_allclose(emb[0, :half], 1.0, atol=1e-7)
    np.testing.assert_allclose(emb[0, half:], 0.0, atol=1e-7)


def test_flip_sin_to_cos_swaps_the_halves() -> None:
    timesteps = np.array([137.0], dtype=np.float32)
    half = EMBED_DIM // 2
    flipped = sinusoidal_embedding(timesteps, EMBED_DIM, flip_sin_to_cos=True)
    plain = sinusoidal_embedding(timesteps, EMBED_DIM, flip_sin_to_cos=False)
    np.testing.assert_allclose(flipped[:, :half], plain[:, half:], atol=1e-7)
    np.testing.assert_allclose(flipped[:, half:], plain[:, :half], atol=1e-7)


def test_frequency_denominator_uses_the_zero_shift() -> None:
    """Klein sets downscale_freq_shift=0.0; the default 1.0 would rescale
    every frequency, which is invisible in shapes but wrong in values."""
    timesteps = np.array([500.0], dtype=np.float32)
    half = EMBED_DIM // 2
    emb = sinusoidal_embedding(timesteps, EMBED_DIM, flip_sin_to_cos=False)
    # Last sin channel corresponds to the highest-index frequency band.
    expected = math.sin(500.0 * math.exp(-math.log(10000) * (half - 1) / half))
    assert emb[0, half - 1] == pytest.approx(expected, abs=1e-6)


def test_mlp_produces_the_dit_width() -> None:
    weights = TimestepEmbedderWeights.random(KLEIN_4B, seed=3)
    assert weights.in_channels == KLEIN_4B.timestep_guidance_channels
    assert weights.out_dim == KLEIN_4B.dim == 3072

    scheduler = FlowMatchEulerScheduler()
    timesteps = scheduler.timesteps(scheduler.sigmas(1024, 4))
    temb = embed_timestep(timesteps, weights)
    assert temb.shape == (4, KLEIN_4B.dim)
    assert temb.dtype == np.float32
    assert np.isfinite(temb).all()


def test_embed_applies_the_training_unit_rescale() -> None:
    """The DiT was trained on [0, 1000]; the scheduler emits [0, 1]."""
    weights = TimestepEmbedderWeights.random(KLEIN_4B, seed=3)
    normalized = np.array([0.25], dtype=np.float32)
    np.testing.assert_allclose(
        embed_timestep(normalized, weights),
        embed_timestep(normalized * TIMESTEP_SCALE, weights, rescale=False),
        rtol=1e-6,
        atol=1e-6,
    )


def test_distinct_timesteps_give_distinct_embeddings() -> None:
    """A collapsed embedding would make every step identical."""
    weights = TimestepEmbedderWeights.random(KLEIN_4B, seed=3)
    scheduler = FlowMatchEulerScheduler()
    temb = embed_timestep(
        scheduler.timesteps(scheduler.sigmas(1024, 4)), weights
    )
    for i in range(len(temb)):
        for j in range(i + 1, len(temb)):
            assert not np.allclose(temb[i], temb[j], atol=1e-4)


def test_mismatched_projections_rejected() -> None:
    with pytest.raises(ValueError):
        TimestepEmbedderWeights(
            linear_1=np.zeros((3072, 256), dtype=np.float32),
            linear_2=np.zeros((3072, 1024), dtype=np.float32),
        )
