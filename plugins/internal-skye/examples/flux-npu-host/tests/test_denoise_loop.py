"""Validates the loop skeleton and the three interface seams.

The block stack here is a **test double**, not an implementation: it satisfies
the :class:`DiTBlockStack` protocol with host arithmetic so the loop's
bookkeeping -- step count, timestep routing, Euler telescoping, shape
contracts -- can be checked before any device exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest
from flux_klein_host import (
    KLEIN_4B,
    DiTBlockStack,
    DiTBlockStackRequest,
    FlowMatchEulerScheduler,
    LatentBatchNormStats,
    StepRecord,
    TextConditioning,
    TimestepEmbedderWeights,
    denoise,
    init_latent,
    latent_image_ids,
    pack,
    precompute_schedule,
    resolve_geometry,
    text_conditioning_from_hidden_states,
    text_ids,
    unpack,
    vae_decode_input,
)

# A small resolution keeps the double's arithmetic cheap; the geometry logic is
# resolution-independent and is covered at 512^2 in test_latents.
GEOMETRY = resolve_geometry(KLEIN_4B, 128, 128)


@dataclass
class ConstantVelocity:
    """Returns a fixed velocity and records how it was called."""

    value: float = 1.0
    calls: list[DiTBlockStackRequest] = field(default_factory=list)

    def __call__(self, request: DiTBlockStackRequest) -> np.ndarray:
        self.calls.append(request)
        return np.full_like(request.latent, self.value)


@dataclass
class TimestepEchoVelocity:
    """Velocity proportional to the timestep, to prove routing per step."""

    def __call__(self, request: DiTBlockStackRequest) -> np.ndarray:
        return np.full_like(request.latent, request.conditioning.timestep)


def make_text(batch_size: int = 1) -> TextConditioning:
    """Builds conditioning shaped like a real Qwen3 3-layer stack."""
    seq = KLEIN_4B.text_seq_len
    hidden = KLEIN_4B.text_hidden_size
    rng = np.random.RandomState(0)
    states = [
        rng.standard_normal((batch_size, seq, hidden)).astype(np.float32)
        for _ in KLEIN_4B.text_hidden_state_layers
    ]
    return text_conditioning_from_hidden_states(
        states, text_ids(seq, batch_size)
    )


@pytest.fixture(scope="module")
def weights() -> TimestepEmbedderWeights:
    return TimestepEmbedderWeights.random(KLEIN_4B, seed=5)


def test_encoder_seam_widths_match_joint_attention_dim() -> None:
    text = make_text()
    assert text.prompt_embeds.shape == (1, 512, KLEIN_4B.joint_attention_dim)
    assert text.prompt_embeds.shape[-1] == 7680
    text.validate(KLEIN_4B)


def test_encoder_seam_rejects_ragged_hidden_states() -> None:
    good = np.zeros((1, 512, 2560), dtype=np.float32)
    bad = np.zeros((1, 512, 1280), dtype=np.float32)
    with pytest.raises(ValueError):
        text_conditioning_from_hidden_states([good, bad], text_ids(512))


def test_encoder_seam_rejects_wrong_width() -> None:
    text = TextConditioning(
        prompt_embeds=np.zeros((1, 512, 1024), dtype=np.float32),
        text_ids=text_ids(512),
    )
    with pytest.raises(ValueError):
        text.validate(KLEIN_4B)


@pytest.mark.parametrize("num_steps", [1, 2, 4, 8])
def test_loop_calls_the_block_stack_once_per_step(
    weights: TimestepEmbedderWeights, num_steps: int
) -> None:
    stack = ConstantVelocity()
    result = denoise(
        stack,
        make_text(),
        GEOMETRY,
        weights,
        num_steps=num_steps,
        seed=42,
    )
    assert len(stack.calls) == num_steps
    assert [c.conditioning.step_index for c in stack.calls] == list(
        range(num_steps)
    )
    np.testing.assert_allclose(
        [c.conditioning.timestep for c in stack.calls],
        result.schedule.timesteps,
        rtol=1e-6,
    )


@pytest.mark.parametrize("num_steps", [1, 4])
def test_constant_velocity_telescopes_to_the_full_sigma_interval(
    weights: TimestepEmbedderWeights, num_steps: int
) -> None:
    """Sum of dt is sigma_N - sigma_0 = -1, whatever the shift or step count.

    This is the sharpest cheap check on the schedule: it fails if the ladder
    does not start at 1.0, does not end at 0.0, or skips an interval.
    """
    stack = ConstantVelocity(value=1.0)
    initial = init_latent(GEOMETRY, seed=3)
    result = denoise(
        stack,
        make_text(),
        GEOMETRY,
        weights,
        num_steps=num_steps,
        initial_latent=initial,
    )
    np.testing.assert_allclose(
        result.latent, pack(initial) - 1.0, rtol=1e-5, atol=1e-5
    )


def test_loop_matches_a_hand_rolled_euler_chain(
    weights: TimestepEmbedderWeights,
) -> None:
    stack = TimestepEchoVelocity()
    initial = init_latent(GEOMETRY, seed=9)
    result = denoise(
        stack,
        make_text(),
        GEOMETRY,
        weights,
        num_steps=4,
        initial_latent=initial,
    )

    scheduler = FlowMatchEulerScheduler()
    sigmas = scheduler.sigmas(GEOMETRY.image_seq_len, 4)
    expected = pack(initial)
    for i in range(4):
        velocity = np.full_like(expected, sigmas[i])
        expected = scheduler.step(velocity, expected, sigmas, i)

    np.testing.assert_allclose(result.latent, expected, rtol=1e-6, atol=1e-6)


def test_step_hook_observes_every_step(
    weights: TimestepEmbedderWeights,
) -> None:
    """The P6 per-stage oracle hangs off this hook."""
    seen: list[StepRecord] = []
    denoise(
        ConstantVelocity(),
        make_text(),
        GEOMETRY,
        weights,
        num_steps=4,
        seed=1,
        on_step=seen.append,
    )
    assert [r.step_index for r in seen] == [0, 1, 2, 3]
    assert seen[0].sigma == pytest.approx(1.0, abs=1e-6)
    assert seen[-1].next_sigma == 0.0
    for record in seen:
        assert record.dt < 0.0
        np.testing.assert_allclose(
            record.latent_out,
            record.latent_in + record.dt * record.velocity,
            rtol=1e-6,
            atol=1e-6,
        )


def test_loop_validates_the_block_stack_output_shape(
    weights: TimestepEmbedderWeights,
) -> None:
    def wrong_shape(request: DiTBlockStackRequest) -> np.ndarray:
        return np.zeros((1, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="block stack returned"):
        denoise(wrong_shape, make_text(), GEOMETRY, weights, num_steps=1)


def test_loop_rejects_mismatched_initial_latent(
    weights: TimestepEmbedderWeights,
) -> None:
    with pytest.raises(ValueError):
        denoise(
            ConstantVelocity(),
            make_text(),
            GEOMETRY,
            weights,
            initial_latent=np.zeros((1, 32, 8, 8), dtype=np.float32),
        )


def test_block_stack_request_is_fully_validated(
    weights: TimestepEmbedderWeights,
) -> None:
    """Every field the driver indexes must be checked before dispatch."""
    schedule = precompute_schedule(GEOMETRY, weights, num_steps=4)
    request = DiTBlockStackRequest(
        latent=np.zeros((1, GEOMETRY.image_seq_len, 128), dtype=np.float32),
        image_ids=latent_image_ids(GEOMETRY),
        text=make_text(),
        conditioning=type(
            "C",
            (),
            {
                "step_index": 0,
                "timestep": 1.0,
                "temb": np.zeros((1, 64), dtype=np.float32),
                "modulation": None,
            },
        )(),
        geometry=GEOMETRY,
    )
    with pytest.raises(ValueError, match="temb"):
        request.validate(KLEIN_4B)
    assert schedule.temb.shape == (4, 1, KLEIN_4B.dim)


def test_double_typing_satisfies_the_protocol() -> None:
    assert isinstance(ConstantVelocity(), DiTBlockStack)


def test_precomputed_schedule_is_latent_independent(
    weights: TimestepEmbedderWeights,
) -> None:
    """Everything in the bundle must be stageable before the first dispatch."""
    schedule = precompute_schedule(GEOMETRY, weights, num_steps=4)
    assert schedule.num_steps == 4
    assert schedule.sigmas.shape == (5,)
    assert schedule.timesteps.shape == (4,)
    assert schedule.dts.shape == (4,)
    assert schedule.temb.shape == (4, 1, KLEIN_4B.dim)
    np.testing.assert_allclose(
        schedule.dts, np.diff(schedule.sigmas), rtol=1e-6
    )


def test_vae_seam_denormalizes_then_unpacks() -> None:
    """Order matters: BN stats are per packed channel, so unpacking first
    would apply them to the wrong channels."""
    rng = np.random.RandomState(2)
    packed = rng.standard_normal((1, GEOMETRY.image_seq_len, 128)).astype(
        np.float32
    )
    stats = LatentBatchNormStats(
        mean=rng.standard_normal(128).astype(np.float32),
        var=np.abs(rng.standard_normal(128)).astype(np.float32) + 0.5,
        eps=KLEIN_4B.vae_batch_norm_eps,
    )

    decode_input = vae_decode_input(packed, GEOMETRY, stats)
    decode_input.validate(KLEIN_4B)
    assert decode_input.latent.shape == (1, *GEOMETRY.unpacked_shape)

    std = np.sqrt(stats.var + stats.eps)
    expected = unpack(packed * std + stats.mean, GEOMETRY)
    np.testing.assert_allclose(decode_input.latent, expected, rtol=1e-6)


def test_vae_seam_identity_stats_are_a_pure_unpack() -> None:
    packed = (
        np.random.RandomState(4)
        .standard_normal((1, GEOMETRY.image_seq_len, 128))
        .astype(np.float32)
    )
    stats = LatentBatchNormStats.identity(128, eps=KLEIN_4B.vae_batch_norm_eps)
    decode_input = vae_decode_input(packed, GEOMETRY, stats)
    np.testing.assert_allclose(
        decode_input.latent, unpack(packed, GEOMETRY), rtol=1e-6, atol=1e-6
    )


def test_end_to_end_seam_chain(weights: TimestepEmbedderWeights) -> None:
    """Seam 1 -> loop -> seam 3, with the device call stubbed at seam 2."""
    text = make_text()
    result = denoise(
        ConstantVelocity(value=0.1),
        text,
        GEOMETRY,
        weights,
        num_steps=4,
        seed=17,
    )
    decode_input = vae_decode_input(
        result.latent,
        GEOMETRY,
        LatentBatchNormStats.identity(128, KLEIN_4B.vae_batch_norm_eps),
    )
    decode_input.validate(KLEIN_4B)
    assert decode_input.latent.shape == (1, 32, 16, 16)
    assert np.isfinite(decode_input.latent).all()
