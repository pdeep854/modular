"""Timestep conditioning for the Klein DiT: sinusoidal projection plus MLP.

Klein is guidance-distilled (``guidance_embeds: false``), so there is no
guidance embedder and no classifier-free guidance: one forward per step, and
``temb`` depends only on the timestep. That makes the entire conditioning path
latent-independent, so the host precomputes ``temb`` -- and the per-block adaLN
shift/scale/gate derived from it -- once per step and copies them to the device
as plain buffers, exactly as the block driver already consumes them.

The sinusoidal half follows DDPM/diffusers with Klein's settings:
``flip_sin_to_cos=True`` (cosine block first) and ``downscale_freq_shift=0.0``.
The MLP half is two bias-free linears with SiLU between, computed the way an
fp16 pipeline computes them: float16 storage between ops, wide accumulation
inside each (see :func:`embed_timestep`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .config import KleinConfig
from .interfaces import ModulationTriple, StepModulation

#: Training-unit rescale applied before the sinusoidal projection. The
#: scheduler works in normalized ``[0, 1]`` sigmas; the DiT was trained on
#: ``[0, 1000]``. MAX applies it in ``Flux2Transformer2DModel``
#: rather than in the scheduler, and this scaffold keeps it in the same place.
TIMESTEP_SCALE = 1000.0


def sinusoidal_embedding(
    timesteps: npt.NDArray[np.float32],
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> npt.NDArray[np.float32]:
    """Projects scalar timesteps onto a sinusoidal basis.

    Args:
        timesteps: Shape ``(B,)``, already in training units (see
            :data:`TIMESTEP_SCALE`).
        embedding_dim: Output width; 256 for Klein.
        flip_sin_to_cos: Emit ``[cos, sin]`` instead of ``[sin, cos]``.
        downscale_freq_shift: Subtracted from ``half_dim`` in the frequency
            denominator; 0.0 for Klein.
        scale: Multiplier on the pre-trig argument.
        max_period: Longest sinusoid period.

    Returns:
        Float32 array of shape ``(B, embedding_dim)``.

    Raises:
        ValueError: If ``timesteps`` is not 1-D.
    """
    if timesteps.ndim != 1:
        raise ValueError(f"expected shape (B,), got {timesteps.shape}")

    # Held in float32 throughout, matching both diffusers and MAX. float64
    # would be more accurate but would drift from the reference the on-device
    # result is checked against, which is the wrong trade here.
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * np.arange(half_dim, dtype=np.float32)
    exponent = (exponent / np.float32(half_dim - downscale_freq_shift)).astype(
        np.float32
    )
    freqs = np.exp(exponent)

    args = (
        np.float32(scale)
        * timesteps.astype(np.float32)[:, None]
        * freqs[None, :]
    ).astype(np.float32)
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if flip_sin_to_cos:
        emb = np.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)))
    return emb.astype(np.float32)


def _f16(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Rounds through float16: the storage an fp16 pipeline keeps between
    ops. Matmuls still accumulate wide; only the stored result rounds,
    matching fp16 ``nn.Linear`` semantics."""
    return x.astype(np.float16).astype(np.float32)


def silu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """SiLU, ``x * sigmoid(x)``.

    Written through ``tanh`` rather than ``exp(-x)`` so large-magnitude inputs
    saturate instead of overflowing.
    """
    x32 = x.astype(np.float32)
    half = np.float32(0.5)
    sigmoid = half * (np.float32(1.0) + np.tanh(half * x32))
    return (x32 * sigmoid).astype(np.float32)


@dataclass(frozen=True)
class TimestepEmbedderWeights:
    """The two bias-free projections of the timestep MLP.

    Shapes follow the checkpoint convention ``(out_features, in_features)``, so
    the forward pass is ``x @ W.T``. Corresponding checkpoint tensors are
    ``time_guidance_embed.timestep_embedder.linear_{1,2}.weight``.
    """

    linear_1: npt.NDArray[np.float32]
    linear_2: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.linear_1.ndim != 2 or self.linear_2.ndim != 2:
            raise ValueError("both projections must be 2-D")
        if self.linear_1.shape[0] != self.linear_2.shape[1]:
            raise ValueError(
                f"linear_1 out ({self.linear_1.shape[0]}) must match linear_2 "
                f"in ({self.linear_2.shape[1]})"
            )

    @property
    def in_channels(self) -> int:
        """Sinusoidal width the MLP consumes."""
        return self.linear_1.shape[1]

    @property
    def out_dim(self) -> int:
        """Embedding width the MLP produces."""
        return self.linear_2.shape[0]

    @classmethod
    def random(
        cls, config: KleinConfig, seed: int = 0
    ) -> TimestepEmbedderWeights:
        """Draws scaled-normal weights for weight-free scaffold testing.

        Real weights arrive from the V1 conversion; this exists so the loop
        skeleton and its shape contracts can be exercised without a checkpoint.
        """
        rng = np.random.RandomState(seed)
        in_c, out_d = config.timestep_guidance_channels, config.dim
        return cls(
            linear_1=(
                rng.standard_normal((out_d, in_c)) / math.sqrt(in_c)
            ).astype(np.float32),
            linear_2=(
                rng.standard_normal((out_d, out_d)) / math.sqrt(out_d)
            ).astype(np.float32),
        )


def scaled_timestep_argument(
    timesteps: npt.NDArray[np.float32],
    *,
    rescale: bool = True,
) -> npt.NDArray[np.float32]:
    """The timestep exactly as the DiT's ``Timesteps`` block receives it.

    Rescales to training units and rounds through float16, matching
    ``Flux2Transformer2DModel``: ``ops.cast(timestep * 1000.0,
    hidden_states.dtype)`` (flux2.py) with the model in float16. The
    2026-08-01 recording skipped that cast, which is device finding D3: any
    timestep whose scaled value is not fp16-exact (958.085... -> 958.0)
    diverges the faithful graph arms from the reference. One function owns
    the semantics so the host ``temb`` path and a staged ``timesteps_proj``
    table cannot disagree about it.

    Args:
        timesteps: Shape ``(B,)`` in the scheduler's normalized ``[0, 1]``
            units (or already scaled, with ``rescale=False``).
        rescale: Multiply by :data:`TIMESTEP_SCALE` first.

    Returns:
        Float32 array of shape ``(B,)``, fp16-rounded.
    """
    t = timesteps.astype(np.float32)
    if rescale:
        t = (t * np.float32(TIMESTEP_SCALE)).astype(np.float32)
    return _f16(t)


def embed_timestep(
    timesteps: npt.NDArray[np.float32],
    weights: TimestepEmbedderWeights,
    *,
    rescale: bool = True,
) -> npt.NDArray[np.float32]:
    """Embeds normalized timesteps into the DiT conditioning vector.

    Computes what an fp16 pipeline computes, which the 2026-08-01
    recording's pure-float32 chain did not (device finding D3, both counts
    measured on silicon). First the scaled timestep rounds through float16
    -- :func:`scaled_timestep_argument`, the one owner of the argument
    semantics, so a staged ``timesteps_proj`` table cannot disagree with
    this path. Then the projection and every value after it store float16
    between ops, as fp16 ``nn.Linear`` modules do
    (``timesteps_proj.to(timestep.dtype)`` feeding ``TimestepEmbedding``):
    matmuls accumulate wide and round their outputs; the trig itself stays
    float32.

    Args:
        timesteps: Shape ``(B,)`` in the scheduler's normalized ``[0, 1]``
            units.
        weights: MLP projections.
        rescale: Multiply by :data:`TIMESTEP_SCALE` first. Leave ``True``
            unless the caller already scaled.

    Returns:
        ``temb`` of shape ``(B, out_dim)``, float32.
    """
    t = scaled_timestep_argument(timesteps, rescale=rescale)
    proj = _f16(sinusoidal_embedding(t, weights.in_channels))
    hidden = _f16(silu(_f16(proj @ weights.linear_1.T)))
    return _f16((hidden @ weights.linear_2.T).astype(np.float32))


#: The four adaLN heads, in the order :class:`ModulationWeights` stores them,
#: with the number of chunks each one's linear emits. Straight out of
#: ``Flux2Transformer2DModel.__init__``: two ``mod_param_sets=2`` heads for the
#: double stream, one ``mod_param_sets=1`` head for the single stream, and
#: ``norm_out``'s ``AdaLayerNormContinuous`` head.
MODULATION_HEADS: tuple[tuple[str, int], ...] = (
    ("double_stream_modulation_img.linear.weight", 6),
    ("double_stream_modulation_txt.linear.weight", 6),
    ("single_stream_modulation.linear.weight", 3),
    ("norm_out.linear.weight", 2),
)

#: Vectors the four heads produce together: ``6 + 6 + 3 + 2``.
MODULATION_VECTORS = sum(chunks for _, chunks in MODULATION_HEADS)


@dataclass(frozen=True)
class ModulationWeights:
    """The four bias-free adaLN projections, in :data:`MODULATION_HEADS` order.

    Each is ``(chunks * dim, dim)`` following the checkpoint's
    ``(out_features, in_features)``, so one head's forward pass is
    ``silu(temb) @ W.T`` split into ``chunks`` equal pieces along the last
    axis. ``bias=False`` throughout, which is why there are no biases here.
    """

    double_img: npt.NDArray[np.float32]
    double_txt: npt.NDArray[np.float32]
    single: npt.NDArray[np.float32]
    norm_out: npt.NDArray[np.float32]

    @property
    def heads(self) -> tuple[npt.NDArray[np.float32], ...]:
        """The four projections in wire order."""
        return (self.double_img, self.double_txt, self.single, self.norm_out)

    @property
    def dim(self) -> int:
        """DiT hidden width, the length of every modulation vector."""
        return self.double_img.shape[1]

    def __post_init__(self) -> None:
        dim = self.dim
        for (name, chunks), weight in zip(
            MODULATION_HEADS, self.heads, strict=False
        ):
            if weight.shape != (chunks * dim, dim):
                raise ValueError(
                    f"{name} is {weight.shape}, expected "
                    f"{(chunks * dim, dim)} -- {chunks} chunks of {dim}"
                )

    @classmethod
    def random(cls, config: KleinConfig, seed: int = 0) -> ModulationWeights:
        """Draws scaled-normal heads, for exercising the path weight-free."""
        rng = np.random.RandomState(seed)
        dim = config.dim

        def draw(chunks: int) -> npt.NDArray[np.float32]:
            return (
                rng.standard_normal((chunks * dim, dim)) / math.sqrt(dim)
            ).astype(np.float32)

        return cls(*(draw(chunks) for _, chunks in MODULATION_HEADS))


def modulation_chunks(
    temb: npt.NDArray[np.float32], weights: ModulationWeights
) -> list[npt.NDArray[np.float32]]:
    """Evaluates the four heads and returns their 17 chunks in wire order.

    Args:
        temb: ``(B, dim)`` timestep embedding.
        weights: The four projections.

    Returns:
        Seventeen ``(B, dim)`` arrays: the image head's six, the text head's
        six, the single head's three, then ``norm_out``'s scale and shift.
        Inputs and outputs of each head store float16 between ops, as the
        fp16 pipeline's ``nn.Linear`` heads do (see :func:`embed_timestep`).

    Raises:
        ValueError: If ``temb`` is not ``(B, dim)`` for the heads' ``dim``.
    """
    if temb.ndim != 2 or temb.shape[1] != weights.dim:
        raise ValueError(f"temb must be (B, {weights.dim}), got {temb.shape}")
    # SiLU lives *inside* each head in diffusers (`Flux2Modulation.act_fn`,
    # `AdaLayerNormContinuous.silu`), so it is applied per head rather than
    # once to `temb` -- same value here, but the same value for a reason worth
    # keeping visible.
    chunks: list[npt.NDArray[np.float32]] = []
    for (_, count), weight in zip(
        MODULATION_HEADS, weights.heads, strict=False
    ):
        emitted = _f16(_f16(silu(temb)) @ weight.T)
        chunks += [
            np.ascontiguousarray(piece)
            for piece in np.split(emitted.astype(np.float32), count, axis=-1)
        ]
    return chunks


def derive_modulation(
    temb: npt.NDArray[np.float32], weights: ModulationWeights
) -> StepModulation:
    """Builds one step's shared adaLN parameters from its ``temb``.

    Args:
        temb: ``(B, dim)`` timestep embedding.
        weights: The four projections.

    Returns:
        The :class:`~flux_klein_host.interfaces.StepModulation` for the step.
    """
    c = modulation_chunks(temb, weights)

    def triple(at: int) -> ModulationTriple:
        return ModulationTriple(shift=c[at], scale=c[at + 1], gate=c[at + 2])

    return StepModulation(
        double_img=(triple(0), triple(3)),
        double_txt=(triple(6), triple(9)),
        single=triple(12),
        out_scale=c[15],
        out_shift=c[16],
    )


__all__ = [
    "MODULATION_HEADS",
    "MODULATION_VECTORS",
    "TIMESTEP_SCALE",
    "ModulationWeights",
    "TimestepEmbedderWeights",
    "derive_modulation",
    "embed_timestep",
    "modulation_chunks",
    "scaled_timestep_argument",
    "silu",
    "sinusoidal_embedding",
]
