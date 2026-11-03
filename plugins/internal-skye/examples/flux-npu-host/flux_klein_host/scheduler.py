"""Flow-matching Euler scheduler for FLUX.2-Klein.

Host-side, numpy-only, and device-independent: the whole schedule is a function
of ``(image_seq_len, num_steps)``, so it is computed once before the denoise
loop starts and handed to the driver as a plain buffer.

Conventions worth pinning down, because three codebases spell them differently:

* **Sigma array length is ``N + 1``.** ``sigmas[0] == 1.0`` (pure noise) and
  ``sigmas[N] == 0.0`` (clean latent). The base ladder is
  ``linspace(1.0, 1/N, N)`` with ``0.0`` appended, which is identical to
  black-forest-labs' ``linspace(1, 0, N + 1)`` and to what diffusers'
  ``FluxPipeline`` passes into ``set_timesteps(sigmas=...)``. It is *not* the
  bare diffusers scheduler default (``linspace(1, 1/1000, N)``).
* **Timesteps are normalized to ``[0, 1]``**, equal to ``sigmas[:-1]``. The
  ``x1000`` rescale into training units happens inside the timestep embedding
  (see :mod:`flux_klein_host.timestep_embed`), matching where MAX applies it.
* **The update is plain Euler on the velocity field**, so the step size is a
  negative ``dt``; see :meth:`FlowMatchEulerScheduler.step`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import KLEIN_4B_SCHEDULER, SchedulerConfig


def time_shift_exponential(
    mu: float, sigma_param: float, t: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Applies the resolution-dependent exponential timestep shift.

    ``t -> exp(mu) / (exp(mu) + (1/t - 1) ** sigma_param)``, evaluated in
    float64 and clipped away from the ``t -> 0`` pole before narrowing back to
    float32.

    Args:
        mu: Shift parameter; larger ``mu`` pushes sigmas toward 1 (more time
            spent at high noise).
        sigma_param: Exponent on the odds ratio; 1.0 for every FLUX config.
        t: Base sigma ladder in ``(0, 1]``.

    Returns:
        The shifted ladder, same shape as ``t``.
    """
    t_safe = np.clip(t.astype(np.float64), 1e-7, 1.0)
    out = np.exp(mu) / (np.exp(mu) + (1.0 / t_safe - 1.0) ** sigma_param)
    return out.astype(np.float32)


def linear_mu(
    image_seq_len: int,
    *,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Interpolates ``mu`` linearly in sequence length.

    This is diffusers' ``calculate_shift`` and the FLUX.1 policy. Klein does
    not use it by default -- see :func:`empirical_mu` -- but it is kept so the
    two policies can be diffed.
    """
    slope = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    intercept = base_shift - slope * base_image_seq_len
    return float(slope * image_seq_len + intercept)


def empirical_mu(image_seq_len: int, num_inference_steps: int) -> float:
    """Computes the FLUX.2 step-count-aware ``mu``.

    Piecewise fit published in black-forest-labs/flux2 ``sampling.py`` and
    ported in MAX's ``FlowMatchEulerDiscreteScheduler._compute_empirical_mu``.
    Unlike :func:`linear_mu` it depends on the step count, which matters a lot
    at Klein's 4-step target.

    Args:
        image_seq_len: Number of image tokens (``H//2 * W//2`` after packing).
        num_inference_steps: Denoising steps.

    Returns:
        The shift parameter ``mu``.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_inference_steps + b)


class FlowMatchEulerScheduler:
    """Flow-matching Euler discrete scheduler over a precomputed sigma ladder.

    Stateless with respect to the loop: :meth:`step` takes the step index, so
    the driver can replay, checkpoint, or resume without hidden counters.
    """

    def __init__(self, config: SchedulerConfig | None = None) -> None:
        """Initializes the scheduler.

        Args:
            config: Scheduler configuration; defaults to Klein-4B's public
                ``scheduler_config.json`` plus MAX's ``use_empirical_mu``
                override.

        Raises:
            NotImplementedError: If the config asks for a shift type other than
                ``"exponential"``.
        """
        self.config = config or KLEIN_4B_SCHEDULER
        if self.config.time_shift_type != "exponential":
            raise NotImplementedError(
                f"time_shift_type={self.config.time_shift_type!r} is not "
                "implemented; Klein uses 'exponential'."
            )

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
        """Returns ``mu`` under the configured policy."""
        if self.config.use_empirical_mu:
            return empirical_mu(image_seq_len, num_inference_steps)
        return linear_mu(
            image_seq_len,
            base_image_seq_len=self.config.base_image_seq_len,
            max_image_seq_len=self.config.max_image_seq_len,
            base_shift=self.config.base_shift,
            max_shift=self.config.max_shift,
        )

    def sigmas(
        self,
        image_seq_len: int,
        num_inference_steps: int,
        *,
        sigma_min: float | None = None,
    ) -> npt.NDArray[np.float32]:
        """Builds the ``N + 1`` sigma ladder for one request.

        Args:
            image_seq_len: Number of image tokens.
            num_inference_steps: Denoising steps ``N``.
            sigma_min: Terminal sigma of the *base* ladder before shifting.
                Defaults to ``1 / N``, which makes the appended ``0.0`` land on
                an evenly spaced grid.

        Returns:
            Float32 array of shape ``(N + 1,)``, descending, ending at ``0.0``.

        Raises:
            ValueError: If ``num_inference_steps`` is not positive.
        """
        if num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {num_inference_steps}"
            )

        min_sigma = (
            float(sigma_min)
            if sigma_min is not None
            else 1.0 / num_inference_steps
        )
        sigmas = np.linspace(
            1.0, min_sigma, num_inference_steps, dtype=np.float32
        )

        if self.config.use_dynamic_shifting:
            mu = self.compute_mu(image_seq_len, num_inference_steps)
            sigmas = time_shift_exponential(mu, 1.0, sigmas)

        terminal = self.config.shift_terminal
        if terminal is not None and terminal > 0:
            one_minus_z = 1.0 - sigmas
            scale_factor = one_minus_z[-1] / (1.0 - terminal)
            sigmas = (1.0 - (one_minus_z / scale_factor)).astype(np.float32)

        return np.append(sigmas, np.float32(0.0))

    def timesteps(
        self, sigmas: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Returns the per-step timesteps, normalized to ``[0, 1]``."""
        return np.ascontiguousarray(sigmas[:-1])

    def dts(self, sigmas: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Returns the per-step ``sigma`` deltas (negative)."""
        return np.ascontiguousarray(sigmas[1:] - sigmas[:-1])

    def step(
        self,
        velocity: npt.NDArray[np.float32],
        latent: npt.NDArray[np.float32],
        sigmas: npt.NDArray[np.float32],
        step_index: int,
    ) -> npt.NDArray[np.float32]:
        """Advances the latent one Euler step.

        ``x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v``. The DiT predicts the
        velocity toward the *noise*, so the coefficient is negative and the
        latent walks from noise (``sigma = 1``) to clean (``sigma = 0``).

        Args:
            velocity: DiT output ``v``, broadcastable to ``latent``.
            latent: Current latent.
            sigmas: The ladder from :meth:`sigmas`.
            step_index: Index ``i`` in ``[0, N)``.

        Returns:
            The updated latent, float32.

        Raises:
            IndexError: If ``step_index`` is outside ``[0, N)``.
        """
        num_steps = len(sigmas) - 1
        if not 0 <= step_index < num_steps:
            raise IndexError(
                f"step_index {step_index} outside [0, {num_steps})"
            )
        dt = np.float32(sigmas[step_index + 1] - sigmas[step_index])
        return (latent + dt * velocity).astype(np.float32)


__all__ = [
    "FlowMatchEulerScheduler",
    "empirical_mu",
    "linear_mu",
    "time_shift_exponential",
]
