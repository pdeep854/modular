"""The host-side denoise loop.

This is the skeleton the P6 design calls for: pure host math around a single
stubbed device call. It owns the schedule, the timestep conditioning, the Euler
update and the per-step checkpoint hook; it owns none of the tensor compute.

The loop is deliberately thin. Everything that can be computed before the first
dispatch -- sigmas, timesteps, every step's ``temb`` -- is computed by
:func:`precompute_schedule` and handed over as one bundle, because the driver
stages those as device buffers once rather than marshalling per step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .config import (
    DEFAULT_NUM_STEPS,
    KLEIN_4B,
    KleinConfig,
    LatentGeometry,
)
from .interfaces import (
    DiTBlockStack,
    DiTBlockStackRequest,
    StepConditioning,
    StepModulation,
    TextConditioning,
)
from .latents import init_latent, latent_image_ids, pack
from .scheduler import FlowMatchEulerScheduler
from .timestep_embed import (
    ModulationWeights,
    TimestepEmbedderWeights,
    derive_modulation,
    embed_timestep,
)

StepHook = Callable[["StepRecord"], None]
"""Called after each step. The P6 per-stage oracle uses this to diff every
intermediate latent against MAX-in-WSL fp32 without the loop knowing about it.
"""


@dataclass(frozen=True)
class StepRecord:
    """Observation of one completed denoise step."""

    step_index: int
    sigma: float
    next_sigma: float
    timestep: float
    latent_in: npt.NDArray[np.float32]
    velocity: npt.NDArray[np.float32]
    latent_out: npt.NDArray[np.float32]

    @property
    def dt(self) -> float:
        """The Euler coefficient applied, ``next_sigma - sigma`` (negative)."""
        return self.next_sigma - self.sigma


@dataclass(frozen=True)
class PrecomputedSchedule:
    """Latent-independent per-step conditioning for a whole request."""

    sigmas: npt.NDArray[np.float32]
    """``(N + 1,)`` descending, ending at 0.0."""

    timesteps: npt.NDArray[np.float32]
    """``(N,)`` normalized timesteps, ``sigmas[:-1]``."""

    dts: npt.NDArray[np.float32]
    """``(N,)`` Euler coefficients, ``sigmas[1:] - sigmas[:-1]``."""

    temb: npt.NDArray[np.float32]
    """``(N, B, dim)`` timestep embeddings, one per step."""

    mu: float
    """The resolution/step-count shift actually used."""

    modulation: tuple[StepModulation | None, ...] = ()
    """``(N,)`` per-step adaLN parameters, or empty when no modulation weights
    were supplied and the callee is left to derive them."""

    @property
    def num_steps(self) -> int:
        """Number of denoise steps."""
        return len(self.timesteps)


def precompute_schedule(
    geometry: LatentGeometry,
    timestep_weights: TimestepEmbedderWeights,
    *,
    num_steps: int = DEFAULT_NUM_STEPS,
    scheduler: FlowMatchEulerScheduler | None = None,
    batch_size: int = 1,
    modulation_weights: ModulationWeights | None = None,
) -> PrecomputedSchedule:
    """Computes everything the loop needs that does not depend on the latent.

    Args:
        geometry: Resolved latent geometry; supplies ``image_seq_len``, which
            drives the resolution-dependent shift.
        timestep_weights: Timestep MLP projections.
        num_steps: Denoising steps.
        scheduler: Scheduler; defaults to Klein's public config.
        batch_size: Number of images.
        modulation_weights: The four shared adaLN heads. Given them, every
            step's 17 modulation vectors are derived here; without them
            ``modulation`` stays empty and each step's is left to the callee.

    Returns:
        The precomputed bundle.
    """
    scheduler = scheduler or FlowMatchEulerScheduler()
    sigmas = scheduler.sigmas(geometry.image_seq_len, num_steps)
    timesteps = scheduler.timesteps(sigmas)
    mu = scheduler.compute_mu(geometry.image_seq_len, num_steps)

    temb = np.stack(
        [
            embed_timestep(
                np.full((batch_size,), t, dtype=np.float32), timestep_weights
            )
            for t in timesteps
        ]
    )
    return PrecomputedSchedule(
        sigmas=sigmas,
        timesteps=timesteps,
        dts=scheduler.dts(sigmas),
        temb=temb.astype(np.float32),
        mu=mu,
        modulation=()
        if modulation_weights is None
        else tuple(
            derive_modulation(step, modulation_weights) for step in temb
        ),
    )


@dataclass
class DenoiseResult:
    """Outcome of a full denoise loop."""

    latent: npt.NDArray[np.float32]
    """``(B, image_seq_len, 128)`` final packed latent, ready for seam 3."""

    schedule: PrecomputedSchedule
    initial_latent: npt.NDArray[np.float32]
    records: list[StepRecord] = field(default_factory=list)


def denoise(
    block_stack: DiTBlockStack,
    text: TextConditioning,
    geometry: LatentGeometry,
    timestep_weights: TimestepEmbedderWeights,
    *,
    config: KleinConfig = KLEIN_4B,
    scheduler: FlowMatchEulerScheduler | None = None,
    num_steps: int = DEFAULT_NUM_STEPS,
    seed: int | None = None,
    batch_size: int = 1,
    initial_latent: npt.NDArray[np.float32] | None = None,
    modulation_weights: ModulationWeights | None = None,
    on_step: StepHook | None = None,
    keep_records: bool = False,
) -> DenoiseResult:
    """Runs the flow-matching denoise loop.

    The body is the four lines the design specifies -- embed, call the block
    stack, Euler-update, repeat -- with the embedding hoisted out of the loop
    since it does not depend on the latent.

    Args:
        block_stack: The device call (seam 2). Not implemented by this package.
        text: Prompt conditioning (seam 1).
        geometry: Resolved latent geometry.
        timestep_weights: Timestep MLP projections.
        config: Model geometry.
        scheduler: Scheduler; defaults to Klein's public config.
        num_steps: Denoising steps.
        seed: Noise seed, used only when ``initial_latent`` is ``None``.
        batch_size: Number of images.
        initial_latent: Optional unpacked ``(B, 32, H, W)`` starting noise, to
            share exact noise with a reference run.
        modulation_weights: The four shared adaLN heads. Given them, every
            step carries real modulation; without them the block stack derives
            its own.
        on_step: Per-step observer; see :data:`StepHook`.
        keep_records: Retain every :class:`StepRecord` on the result. Off by
            default -- at 512^2 each record holds three latents.

    Returns:
        The :class:`DenoiseResult`.

    Raises:
        ValueError: If ``initial_latent`` does not match ``geometry``.
    """
    text.validate(config)

    if initial_latent is None:
        initial_latent = init_latent(geometry, seed=seed, batch_size=batch_size)
    else:
        expected = (batch_size, *geometry.unpacked_shape)
        if initial_latent.shape != expected:
            raise ValueError(
                f"initial_latent must be {expected}, got {initial_latent.shape}"
            )

    schedule = precompute_schedule(
        geometry,
        timestep_weights,
        num_steps=num_steps,
        scheduler=scheduler,
        batch_size=batch_size,
        modulation_weights=modulation_weights,
    )
    scheduler = scheduler or FlowMatchEulerScheduler()
    image_ids = latent_image_ids(geometry, batch_size=batch_size)

    latent = pack(initial_latent)
    result = DenoiseResult(
        latent=latent, schedule=schedule, initial_latent=initial_latent
    )

    for i in range(schedule.num_steps):
        conditioning = StepConditioning(
            step_index=i,
            timestep=float(schedule.timesteps[i]),
            temb=schedule.temb[i],
            modulation=(
                schedule.modulation[i] if schedule.modulation else None
            ),
        )
        request = DiTBlockStackRequest(
            latent=latent,
            image_ids=image_ids,
            text=text,
            conditioning=conditioning,
            geometry=geometry,
        )
        request.validate(config)

        velocity = block_stack(request)
        if velocity.shape != latent.shape:
            raise ValueError(
                f"block stack returned {velocity.shape}, expected "
                f"{latent.shape}"
            )

        next_latent = scheduler.step(velocity, latent, schedule.sigmas, i)

        if on_step is not None or keep_records:
            record = StepRecord(
                step_index=i,
                sigma=float(schedule.sigmas[i]),
                next_sigma=float(schedule.sigmas[i + 1]),
                timestep=float(schedule.timesteps[i]),
                latent_in=latent,
                velocity=velocity,
                latent_out=next_latent,
            )
            if on_step is not None:
                on_step(record)
            if keep_records:
                result.records.append(record)

        latent = next_latent

    result.latent = latent
    return result


__all__ = [
    "DenoiseResult",
    "PrecomputedSchedule",
    "StepHook",
    "StepRecord",
    "denoise",
    "precompute_schedule",
]
