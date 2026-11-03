"""Validates the sigma schedule and the Euler update against two references."""

from __future__ import annotations

import numpy as np
import pytest
from flux_klein_host import (
    KLEIN_4B,
    KLEIN_4B_SCHEDULER,
    FlowMatchEulerScheduler,
    empirical_mu,
    linear_mu,
    resolve_geometry,
)
from references import (
    diffusers_available,
    load_max_scheduler_module,
    make_diffusers_scheduler,
)

# 4 is Klein's distilled target; 1 is the degenerate edge (a single jump from
# sigma=1 to sigma=0); 28 is the stock FLUX default, kept as a non-distilled
# control.
STEP_COUNTS = [1, 2, 4, 8, 28]

SEQ_LENS = [
    resolve_geometry(KLEIN_4B, 512, 512).image_seq_len,  # 1024
    resolve_geometry(KLEIN_4B, 1024, 1024).image_seq_len,  # 4096
]

requires_diffusers = pytest.mark.skipif(
    not diffusers_available(), reason="torch/diffusers reference not installed"
)


@pytest.fixture(scope="module")
def scheduler() -> FlowMatchEulerScheduler:
    return FlowMatchEulerScheduler()


def test_512_geometry_matches_the_design_target() -> None:
    geometry = resolve_geometry(KLEIN_4B, 512, 512)
    assert geometry.unpacked_shape == (32, 64, 64)
    assert geometry.image_seq_len == 1024
    assert geometry.packed_channels == 128
    # M = 1536 is what the block driver's matmuls are sized for.
    assert geometry.joint_seq_len == 1536


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
@pytest.mark.parametrize("image_seq_len", SEQ_LENS)
def test_sigma_ladder_shape_and_endpoints(
    scheduler: FlowMatchEulerScheduler, num_steps: int, image_seq_len: int
) -> None:
    sigmas = scheduler.sigmas(image_seq_len, num_steps)
    assert sigmas.shape == (num_steps + 1,)
    assert sigmas.dtype == np.float32
    assert sigmas[0] == pytest.approx(1.0, abs=1e-6)
    assert sigmas[-1] == 0.0
    # Strictly descending: a flat pair would mean a wasted forward pass.
    assert np.all(np.diff(sigmas) < 0.0)


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
@pytest.mark.parametrize("image_seq_len", SEQ_LENS)
def test_sigmas_match_max_reference(
    scheduler: FlowMatchEulerScheduler, num_steps: int, image_seq_len: int
) -> None:
    """Our schedule must be substitutable for MAX's host-side prep."""
    max_module = load_max_scheduler_module()
    max_scheduler = max_module.FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=KLEIN_4B_SCHEDULER.base_image_seq_len,
        max_image_seq_len=KLEIN_4B_SCHEDULER.max_image_seq_len,
        base_shift=KLEIN_4B_SCHEDULER.base_shift,
        max_shift=KLEIN_4B_SCHEDULER.max_shift,
        use_flow_sigmas=False,
        use_dynamic_shifting=KLEIN_4B_SCHEDULER.use_dynamic_shifting,
        use_empirical_mu=KLEIN_4B_SCHEDULER.use_empirical_mu,
        shift_terminal=KLEIN_4B_SCHEDULER.shift_terminal,
    )
    ref_timesteps, ref_sigmas = max_scheduler.retrieve_timesteps_and_sigmas(
        image_seq_len, num_steps, sigma_min=None
    )

    sigmas = scheduler.sigmas(image_seq_len, num_steps)
    np.testing.assert_allclose(sigmas, ref_sigmas, rtol=0, atol=0)
    # MAX's executor derives per-step timesteps as sigmas[:-1]; ours must be
    # the same array it would feed the transformer.
    np.testing.assert_allclose(
        scheduler.timesteps(sigmas), ref_timesteps, rtol=0, atol=1e-7
    )


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
@pytest.mark.parametrize("image_seq_len", SEQ_LENS)
@requires_diffusers
def test_sigmas_match_diffusers(
    scheduler: FlowMatchEulerScheduler, num_steps: int, image_seq_len: int
) -> None:
    """Independent check that the shift math is upstream-correct.

    ``mu`` is supplied by us on both sides: upstream treats ``mu`` as a
    pipeline-computed input, so this isolates the base ladder, the exponential
    shift and the terminal-zero append from the ``mu`` policy (which
    :func:`test_empirical_mu_differs_from_linear_mu` covers separately).
    """
    reference = make_diffusers_scheduler(KLEIN_4B_SCHEDULER)
    mu = scheduler.compute_mu(image_seq_len, num_steps)
    base = np.linspace(1.0, 1.0 / num_steps, num_steps)
    reference.set_timesteps(sigmas=base, mu=mu)

    ref_sigmas = reference.sigmas.cpu().numpy().astype(np.float32)
    sigmas = scheduler.sigmas(image_seq_len, num_steps)

    assert ref_sigmas.shape == sigmas.shape
    np.testing.assert_allclose(sigmas, ref_sigmas, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("num_steps", STEP_COUNTS)
@requires_diffusers
def test_step_update_matches_diffusers(
    scheduler: FlowMatchEulerScheduler, num_steps: int
) -> None:
    """The update must match upstream at every step, not just the first."""
    image_seq_len = 1024
    reference = make_diffusers_scheduler(KLEIN_4B_SCHEDULER)
    mu = scheduler.compute_mu(image_seq_len, num_steps)
    reference.set_timesteps(
        sigmas=np.linspace(1.0, 1.0 / num_steps, num_steps), mu=mu
    )

    import torch

    sigmas = scheduler.sigmas(image_seq_len, num_steps)
    rng = np.random.RandomState(0)
    latent = rng.standard_normal((1, 8, 128)).astype(np.float32)

    ours = latent.copy()
    theirs = torch.from_numpy(latent.copy())

    for i in range(num_steps):
        velocity = rng.standard_normal(latent.shape).astype(np.float32)
        ours = scheduler.step(velocity, ours, sigmas, i)
        theirs = reference.step(
            model_output=torch.from_numpy(velocity),
            timestep=reference.timesteps[i],
            sample=theirs,
            return_dict=False,
        )[0]

    np.testing.assert_allclose(ours, theirs.cpu().numpy(), rtol=1e-6, atol=1e-6)


def test_single_step_is_a_direct_jump_to_zero(
    scheduler: FlowMatchEulerScheduler,
) -> None:
    """N=1 edge: one step must consume the whole sigma interval."""
    sigmas = scheduler.sigmas(1024, 1)
    assert sigmas.shape == (2,)
    assert sigmas[0] == pytest.approx(1.0, abs=1e-6)
    assert sigmas[-1] == 0.0

    latent = np.ones((1, 4, 128), dtype=np.float32)
    velocity = np.full_like(latent, 0.25)
    stepped = scheduler.step(velocity, latent, sigmas, 0)
    np.testing.assert_allclose(stepped, latent - 0.25, rtol=0, atol=1e-6)


def test_step_rejects_out_of_range_index(
    scheduler: FlowMatchEulerScheduler,
) -> None:
    sigmas = scheduler.sigmas(1024, 4)
    latent = np.zeros((1, 4, 128), dtype=np.float32)
    with pytest.raises(IndexError):
        scheduler.step(latent, latent, sigmas, 4)


def test_zero_steps_rejected(scheduler: FlowMatchEulerScheduler) -> None:
    with pytest.raises(ValueError):
        scheduler.sigmas(1024, 0)


def test_empirical_mu_differs_from_linear_mu() -> None:
    """Guards the one config choice with no counterpart in the public JSON.

    MAX injects ``use_empirical_mu=True`` for the FLUX.2 family. If that were
    silently dropped the schedule would still look plausible, so pin the gap.
    """
    image_seq_len = 1024
    assert empirical_mu(image_seq_len, 4) != pytest.approx(
        linear_mu(image_seq_len), abs=1e-3
    )
    # The empirical fit is step-count aware; the linear one is not.
    assert empirical_mu(image_seq_len, 4) != pytest.approx(
        empirical_mu(image_seq_len, 28), abs=1e-3
    )


def test_empirical_mu_saturates_above_the_fit_knee() -> None:
    """Past 4300 tokens the fit drops its step-count term."""
    assert empirical_mu(4400, 4) == pytest.approx(empirical_mu(4400, 28))
