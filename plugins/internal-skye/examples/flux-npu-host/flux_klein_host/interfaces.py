"""The three seams between the host sampling loop and the rest of the pipeline.

This module is the contract, not an implementation. It exists so the pieces
landing separately -- the Qwen3 encoder driver, the DiT block driver plus its
batched matmul, and the VAE decoder driver -- can be written against a fixed
shape/dtype agreement instead of against whatever the loop happened to pass.

The seams, in pipeline order:

1. :class:`TextConditioning` -- **encoder output in.** Built once per prompt
   from the Qwen3 encoder's stacked hidden states.
2. :class:`DiTBlockStack` -- **the device call.** Invoked once per denoise
   step. Deliberately *unimplemented here*: the block driver satisfies it.
3. :class:`VaeDecodeInput` -- **decoder input out.** Built once from the final
   latent.

Everything crossing a seam is a host-side numpy array in float32. The driver
casts to fp16 and applies the ``[N, K]`` weight layout on its own side; that is
a device concern and is deliberately not encoded in these shapes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .config import KleinConfig, LatentGeometry
from .latents import batch_norm_denormalize, unpack

# --------------------------------------------------------------------------
# Seam 1: encoder output -> text conditioning
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextConditioning:
    """Prompt conditioning, constant across the whole denoise loop.

    The encoder runs once; these buffers stay device-resident for every step.
    """

    prompt_embeds: npt.NDArray[np.float32]
    """``(B, text_seq_len, joint_attention_dim)`` -- 512 x 7680 for Klein.

    This is the *pre*-``context_embedder`` width. The 7680 is three stacked
    Qwen3 hidden states (layers 9/18/27, hidden 2560), and projecting it to the
    DiT's 3072 is the block stack's job, not the host's.
    """

    text_ids: npt.NDArray[np.int64]
    """``(B, text_seq_len, 4)`` RoPE position ids, ``(0, 0, 0, position)``."""

    attention_bias: npt.NDArray[np.float32] | None = None
    """Optional ``(B, 1, S, S)`` additive padding bias from the tokenizer.

    ``None`` means no padding to mask. Only the encoder needs this; the joint
    DiT attention over text+image is dense (``NULL_MASK``).
    """

    def validate(self, config: KleinConfig) -> None:
        """Checks the shapes against the model config.

        Raises:
            ValueError: On any shape or dtype disagreement.
        """
        if self.prompt_embeds.ndim != 3:
            raise ValueError(
                f"prompt_embeds must be (B, S, D), got "
                f"{self.prompt_embeds.shape}"
            )
        batch, seq, width = self.prompt_embeds.shape
        if width != config.joint_attention_dim:
            raise ValueError(
                f"prompt_embeds width {width} != joint_attention_dim "
                f"{config.joint_attention_dim}"
            )
        if self.text_ids.shape != (batch, seq, 4):
            raise ValueError(
                f"text_ids must be {(batch, seq, 4)}, got {self.text_ids.shape}"
            )
        if self.attention_bias is not None and self.attention_bias.shape != (
            batch,
            1,
            seq,
            seq,
        ):
            raise ValueError(
                f"attention_bias must be {(batch, 1, seq, seq)}, got "
                f"{self.attention_bias.shape}"
            )


def text_conditioning_from_hidden_states(
    hidden_states: Sequence[npt.NDArray[np.float32]],
    text_ids: npt.NDArray[np.int64],
    *,
    attention_bias: npt.NDArray[np.float32] | None = None,
) -> TextConditioning:
    """Stacks selected encoder hidden states into prompt embeddings.

    Klein early-exits the Qwen3 encoder at layer 27 and concatenates the
    hidden states from layers 9, 18 and 27 along the feature axis, giving
    ``3 * 2560 == 7680``. Concatenation order is the layer order given.

    Args:
        hidden_states: One ``(B, S, hidden)`` array per selected layer.
        text_ids: ``(B, S, 4)`` position ids.
        attention_bias: Optional additive padding bias.

    Returns:
        The assembled :class:`TextConditioning`.

    Raises:
        ValueError: If no states are given or their shapes disagree.
    """
    if not hidden_states:
        raise ValueError("need at least one hidden state to stack")
    reference = hidden_states[0].shape
    for i, state in enumerate(hidden_states):
        if state.shape != reference:
            raise ValueError(
                f"hidden state {i} has shape {state.shape}, expected "
                f"{reference}"
            )
    stacked = np.concatenate(
        [state.astype(np.float32) for state in hidden_states], axis=-1
    )
    return TextConditioning(
        prompt_embeds=np.ascontiguousarray(stacked),
        text_ids=np.ascontiguousarray(text_ids),
        attention_bias=attention_bias,
    )


# --------------------------------------------------------------------------
# Seam 2: the DiT block-stack device call
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModulationTriple:
    """One adaLN parameter set, each ``(B, 1, dim)``."""

    shift: npt.NDArray[np.float32]
    scale: npt.NDArray[np.float32]
    gate: npt.NDArray[np.float32]


@dataclass(frozen=True)
class StepModulation:
    """The four shared adaLN heads, evaluated once per step.

    FLUX.2 computes modulation *once* per forward and shares it across all
    blocks rather than per block, so these are small, latent-independent host
    buffers -- the driver copies them in at step start and every one of the 25
    blocks reads the same ones.

    Seventeen ``(dim,)`` vectors in all, and the order is not a convention this
    scaffold chose: it is the chunk order of the four ``nn.Linear`` heads in
    ``diffusers.models.transformers.transformer_flux2``, which is what the
    checkpoint's weight rows are laid out in.

    * ``double_stream_modulation_img.linear`` and ``..._txt.linear`` each emit
      ``3 * 2`` chunks, split ``(shift, scale, gate)`` per set and attention
      before MLP (``Flux2Modulation.split``).
    * ``single_stream_modulation.linear`` emits one such triple.
    * ``norm_out.linear`` emits **two** chunks, and stock
      ``AdaLayerNormContinuous`` reads them ``scale`` *then* ``shift`` -- the
      opposite of the blocks, and the one place the convention flips.

    Every scale is applied as ``x * (1 + scale) + shift``; every gate is a
    plain multiply on its residual branch.
    """

    double_img: tuple[ModulationTriple, ModulationTriple]
    """Image stream of the 5 double blocks: (attention, MLP)."""

    double_txt: tuple[ModulationTriple, ModulationTriple]
    """Text stream of the 5 double blocks: (attention, MLP)."""

    single: ModulationTriple
    """The 20 single blocks' parallel attention+MLP set."""

    out_scale: npt.NDArray[np.float32]
    """``norm_out``'s per-channel scale -- chunk 0 of its linear."""

    out_shift: npt.NDArray[np.float32]
    """``norm_out``'s per-channel shift -- chunk 1 of its linear."""


@dataclass(frozen=True)
class StepConditioning:
    """Per-step conditioning that does not depend on the latent.

    Everything here is computable before the loop runs, which is why the P6
    design precomputes the whole set in WSL and stages it as device buffers.
    """

    step_index: int
    timestep: float
    """Normalized timestep in ``[0, 1]`` -- ``sigmas[step_index]``."""

    temb: npt.NDArray[np.float32]
    """``(B, dim)`` timestep embedding."""

    modulation: StepModulation | None = None
    """Precomputed adaLN parameters, or ``None`` to let the callee derive them.

    ``None`` is the honest default while the modulation weights are not yet
    converted; a driver that receives ``None`` must apply the shared
    modulation heads itself.
    """


@dataclass(frozen=True)
class DiTBlockStackRequest:
    """Full argument bundle for one block-stack invocation."""

    latent: npt.NDArray[np.float32]
    """``(B, image_seq_len, 128)`` packed noisy latent for this step."""

    image_ids: npt.NDArray[np.int64]
    """``(B, image_seq_len, 4)`` RoPE position ids for image tokens."""

    text: TextConditioning
    conditioning: StepConditioning
    geometry: LatentGeometry

    def validate(self, config: KleinConfig) -> None:
        """Checks every shape the block driver will rely on.

        Raises:
            ValueError: On any disagreement with ``config``/``geometry``.
        """
        self.text.validate(config)
        expected = (
            self.latent.shape[0],
            self.geometry.image_seq_len,
            config.in_channels,
        )
        if self.latent.shape != expected:
            raise ValueError(
                f"latent must be {expected}, got {self.latent.shape}"
            )
        if self.image_ids.shape != (expected[0], expected[1], 4):
            raise ValueError(
                f"image_ids must be {(expected[0], expected[1], 4)}, got "
                f"{self.image_ids.shape}"
            )
        if self.conditioning.temb.shape != (expected[0], config.dim):
            raise ValueError(
                f"temb must be {(expected[0], config.dim)}, got "
                f"{self.conditioning.temb.shape}"
            )


@runtime_checkable
class DiTBlockStack(Protocol):
    """The on-device DiT forward, called once per denoise step.

    **This slice defines the interface and does not implement it.** The
    implementation is the hand-orchestrated block driver: it chains the 5
    double and 20 single blocks over host-mapped shared buffers, joint
    sequence ``M = image_seq_len + text_seq_len`` (1536 at 512^2).

    Contract:

    * Pure with respect to the host -- the same request yields the same
      velocity. Device buffers may be reused across calls, but the returned
      array must not alias the input ``latent``.
    * Returns the velocity ``v``, shape and dtype identical to
      ``request.latent`` (``(B, image_seq_len, 128)``, float32 on the host
      side).
    * Text tokens are consumed but not returned: the double blocks update the
      text stream internally and the loop only ever advances the image latent.
    * One forward per step. Klein is guidance-distilled, so there is no
      negative-prompt pass to combine.
    """

    def __call__(
        self, request: DiTBlockStackRequest
    ) -> npt.NDArray[np.float32]: ...


# --------------------------------------------------------------------------
# Seam 3: final latent -> VAE decode input
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LatentBatchNormStats:
    """Per-packed-channel latent batch-norm statistics from the checkpoint.

    Checkpoint tensors are the VAE decoder's running ``mean``/``var``; both are
    ``(128,)`` for Klein, i.e. per *packed* channel.
    """

    mean: npt.NDArray[np.float32]
    var: npt.NDArray[np.float32]
    eps: float = 1e-4

    @classmethod
    def identity(cls, channels: int, eps: float = 1e-4) -> LatentBatchNormStats:
        """Returns a no-op denormalization, for weight-free scaffold runs."""
        return cls(
            mean=np.zeros(channels, dtype=np.float32),
            var=np.ones(channels, dtype=np.float32) - np.float32(eps),
            eps=eps,
        )


@dataclass(frozen=True)
class VaeDecodeInput:
    """What the VAE decoder driver consumes.

    Already denormalized and unpacked, so the decoder starts at its own
    ``post_quant_conv`` with nothing left to reshape.
    """

    latent: npt.NDArray[np.float32]
    """``(B, 32, latent_height, latent_width)`` -- 64x64 at 512^2."""

    geometry: LatentGeometry

    def validate(self, config: KleinConfig) -> None:
        """Checks the decoder-facing shape.

        Raises:
            ValueError: If the latent does not match ``geometry``.
        """
        expected = (self.latent.shape[0], *self.geometry.unpacked_shape)
        if self.latent.shape != expected:
            raise ValueError(
                f"latent must be {expected}, got {self.latent.shape}"
            )
        if self.geometry.latent_channels != config.vae_latent_channels:
            raise ValueError(
                f"geometry latent channels {self.geometry.latent_channels} != "
                f"config {config.vae_latent_channels}"
            )


def vae_decode_input(
    final_latent: npt.NDArray[np.float32],
    geometry: LatentGeometry,
    bn_stats: LatentBatchNormStats,
) -> VaeDecodeInput:
    """Converts the loop's final packed latent into decoder input.

    Order matters: batch-norm denormalization runs on the *packed* latent, then
    the 2x2 unpack. Swapping them permutes the per-channel statistics against
    the channels they belong to.

    Args:
        final_latent: ``(B, image_seq_len, 128)`` output of the last step.
        geometry: Resolved geometry.
        bn_stats: Latent batch-norm statistics.

    Returns:
        The decoder-facing :class:`VaeDecodeInput`.
    """
    denormalized = batch_norm_denormalize(
        final_latent, bn_stats.mean, bn_stats.var, bn_stats.eps
    )
    return VaeDecodeInput(
        latent=unpack(denormalized, geometry), geometry=geometry
    )


__all__ = [
    "DiTBlockStack",
    "DiTBlockStackRequest",
    "LatentBatchNormStats",
    "ModulationTriple",
    "StepConditioning",
    "StepModulation",
    "TextConditioning",
    "VaeDecodeInput",
    "text_conditioning_from_hidden_states",
    "vae_decode_input",
]
