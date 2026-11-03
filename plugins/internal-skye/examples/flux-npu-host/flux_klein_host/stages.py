"""Every stage the pipeline calls through a seam, and the stubs standing in.

The scaffold's :mod:`flux_klein_host.interfaces` fixes the *data* crossing the
three device seams. This module fixes the *calls*: one ``Protocol`` per stage,
so swapping a stub for the real NPU driver is a single constructor change in
:func:`flux_klein_host.pipeline.default_stages` and nothing else moves.

Four stages, in pipeline order:

1. :class:`PromptTokenizer` -- host. Prompt to the encoder's padded token and
   mask buffers. The V2 slice (``klein_tokenize.py``) implements it.
2. :class:`TextEncoder` -- **device.** Tokens to :class:`TextConditioning`.
   The Qwen3 encoder driver implements it.
3. :class:`~flux_klein_host.interfaces.DiTBlockStack` -- **device**, defined by
   the scaffold, called once per denoise step. The DiT block driver plus the
   batched matmul implement it.
4. :class:`VaeDecoder` -- **device.** :class:`VaeDecodeInput` to RGB in
   ``[-1, 1]``. The VAE decoder driver implements it.

The stubs here are *synthetic*: cheap numpy stand-ins that produce the right
shapes and dtypes and depend on every one of their inputs, so a run proves the
orchestration and the data flow without proving any model math. They are
deliberately not approximations of the real stages -- an approximation would
invite reading the resulting image as a result. What each one *does* guarantee
is that no input can be dropped on the floor without the output moving, which
is what the end-to-end test checks by perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .config import KLEIN_4B, KleinConfig, LatentGeometry
from .interfaces import (
    DiTBlockStackRequest,
    TextConditioning,
    VaeDecodeInput,
    text_conditioning_from_hidden_states,
)
from .latents import text_ids

# --------------------------------------------------------------------------
# Stage 1: prompt -> tokens (host)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptTokens:
    """The tokenizer's output, as the encoder driver consumes it.

    Mirrors what the V2 slice writes to ``tokens.bin`` / ``attention_mask.bin``
    / ``attention_bias.bin``, one prompt at a time and unbatched: the batch
    axis is added by the encoder, not the tokenizer.
    """

    tokens: npt.NDArray[np.int32]
    """``(text_seq_len,)`` right-padded token ids."""

    attention_mask: npt.NDArray[np.uint8]
    """``(text_seq_len,)`` 1 for a real token, 0 for padding."""

    attention_bias: npt.NDArray[np.float32]
    """``(1, 1, S, S)`` additive causal + padding bias."""

    num_valid_tokens: int
    rendered: str
    """The prompt after the chat template, for provenance."""

    def validate(self, config: KleinConfig) -> None:
        """Checks the shapes against the model config.

        Raises:
            ValueError: On any shape disagreement.
        """
        seq = config.text_seq_len
        if self.tokens.shape != (seq,):
            raise ValueError(
                f"tokens must be {(seq,)}, got {self.tokens.shape}"
            )
        if self.attention_mask.shape != (seq,):
            raise ValueError(
                f"attention_mask must be {(seq,)}, got "
                f"{self.attention_mask.shape}"
            )
        if self.attention_bias.shape != (1, 1, seq, seq):
            raise ValueError(
                f"attention_bias must be {(1, 1, seq, seq)}, got "
                f"{self.attention_bias.shape}"
            )


@runtime_checkable
class PromptTokenizer(Protocol):
    """Turns one prompt into the encoder's padded input buffers."""

    def __call__(self, prompt: str) -> PromptTokens: ...


# --------------------------------------------------------------------------
# Stage 2: tokens -> text conditioning (device)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TextEncodeRequest:
    """Full argument bundle for one text-encoder invocation."""

    tokens: PromptTokens
    batch_size: int = 1


@runtime_checkable
class TextEncoder(Protocol):
    """The on-device Qwen3 forward, called once per prompt.

    Contract:

    * Returns a :class:`TextConditioning` whose ``prompt_embeds`` is
      ``(B, text_seq_len, joint_attention_dim)`` -- the layer 9/18/27 hidden
      states concatenated, *before* the DiT's ``context_embedder``.
    * ``text_ids`` are the ``(0, 0, 0, position)`` RoPE ids; the encoder owns
      them because it owns the padding.
    * The tokenizer's additive bias is carried through unchanged, so the joint
      DiT attention can see which text tokens were padding.
    """

    def __call__(self, request: TextEncodeRequest) -> TextConditioning: ...


# --------------------------------------------------------------------------
# Stage 4: VAE decode (device)
# --------------------------------------------------------------------------


@runtime_checkable
class VaeDecoder(Protocol):
    """The on-device VAE decoder, called once per image.

    Consumes the already-denormalized, already-unpacked
    :class:`VaeDecodeInput` and returns ``(B, 3, height, width)`` float32 RGB
    in ``[-1, 1]``. Quantizing to uint8 is the host's job
    (:mod:`flux_klein_host.image`), not the decoder's.
    """

    def __call__(self, request: VaeDecodeInput) -> npt.NDArray[np.float32]: ...


# --------------------------------------------------------------------------
# Synthetic stubs
# --------------------------------------------------------------------------


def _projection(
    rng: np.random.RandomState, shape: tuple[int, ...]
) -> npt.NDArray[np.float32]:
    """A fixed random projection scaled to keep activations order-1."""
    return (rng.standard_normal(shape) / np.sqrt(shape[0])).astype(np.float32)


class SyntheticTokenizer:
    """Deterministic stand-in for the V2 tokenizer, with no vocabulary.

    Hashes the prompt into plausible token ids so the pipeline runs with no
    network and no ``tokenizers`` install. Token *values* are meaningless; what
    is real is the padded shape, the mask, and the additive bias, which is
    built the same way the V2 slice builds it.
    """

    def __init__(
        self,
        config: KleinConfig | None = None,
        *,
        vocab_size: int = 151936,
        pad_token_id: int = 151643,
        mask_fill_value: float = -10000.0,
    ) -> None:
        """Initializes the tokenizer.

        Args:
            config: Model geometry, for the padded sequence length.
            vocab_size: Range the hashed ids are drawn from.
            pad_token_id: Right-padding filler, matching Qwen3's.
            mask_fill_value: The masked-position sentinel, matching MAX's
                ``causal_attention_mask.py``.
        """
        self.config = config or KLEIN_4B
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.mask_fill_value = mask_fill_value

    def __call__(self, prompt: str) -> PromptTokens:
        """Encodes a prompt to the padded token contract.

        Args:
            prompt: The raw user prompt.

        Returns:
            The synthetic :class:`PromptTokens`.
        """
        seq = self.config.text_seq_len
        # One id per word, floored at 1 and capped at the sequence length, so
        # a longer prompt really does produce a longer valid span.
        words = prompt.split() or [prompt]
        num_valid = max(1, min(len(words) + 2, seq))

        tokens = np.full(seq, self.pad_token_id, dtype=np.int32)
        for i in range(num_valid):
            digest = 0
            for byte in words[i % len(words)].encode("utf-8"):
                digest = (digest * 131 + byte) & 0xFFFF_FFFF
            # Mixed in as an integer, not appended as a byte: the position can
            # reach `text_seq_len`, which does not fit in a byte.
            digest = (digest * 131 + i) & 0xFFFF_FFFF
            tokens[i] = np.int32(digest % (self.vocab_size - 1) + 1)

        mask = np.zeros(seq, dtype=np.uint8)
        mask[:num_valid] = 1
        visible = np.tril(np.ones((seq, seq), dtype=np.bool_))
        visible &= mask.astype(np.bool_)[np.newaxis, :]
        bias = np.where(
            visible, np.float32(0.0), np.float32(self.mask_fill_value)
        ).reshape(1, 1, seq, seq)

        return PromptTokens(
            tokens=tokens,
            attention_mask=mask,
            attention_bias=bias.astype(np.float32),
            num_valid_tokens=num_valid,
            rendered=prompt,
        )


class SyntheticTextEncoder:
    """Stands in for the on-device Qwen3 encoder driver.

    Gathers three per-layer codebooks by token id and hands them to the
    scaffold's own :func:`text_conditioning_from_hidden_states`, so the
    ``3 x 2560 -> 7680`` concatenation the seam specifies is exercised for
    real. The codebook is indexed by ``token_id % codebook_size`` rather than
    being a full ``151936 x 2560`` embedding table, which would cost 4.6 GiB
    per layer for no additional truth.
    """

    def __init__(
        self,
        config: KleinConfig,
        *,
        seed: int = 0,
        codebook_size: int = 512,
    ) -> None:
        """Initializes the encoder stub.

        Args:
            config: Model geometry.
            seed: Codebook seed.
            codebook_size: Rows in each layer's codebook.
        """
        self.config = config
        self.codebook_size = codebook_size
        rng = np.random.RandomState(seed)
        self.codebooks = [
            (
                rng.standard_normal((codebook_size, config.text_hidden_size))
                * np.float32(0.5)
            ).astype(np.float32)
            for _ in config.text_hidden_state_layers
        ]

    def __call__(self, request: TextEncodeRequest) -> TextConditioning:
        """Encodes tokens into prompt conditioning.

        Args:
            request: The tokenized prompt and the batch size.

        Returns:
            The assembled :class:`TextConditioning`.
        """
        request.tokens.validate(self.config)
        rows = (
            request.tokens.tokens.astype(np.int64) % self.codebook_size
        ).astype(np.intp)

        # Each layer is offset along the codebook so the three selected states
        # differ, the way real layer 9 and layer 27 outputs differ.
        hidden_states = [
            np.tile(
                book[(rows + offset) % self.codebook_size][np.newaxis, :, :],
                (request.batch_size, 1, 1),
            )
            for offset, book in enumerate(self.codebooks)
        ]

        seq = request.tokens.tokens.shape[0]
        bias = np.tile(
            request.tokens.attention_bias, (request.batch_size, 1, 1, 1)
        )
        return text_conditioning_from_hidden_states(
            hidden_states,
            text_ids(seq, batch_size=request.batch_size),
            attention_bias=bias.astype(np.float32),
        )


class ReferenceTextEncoder:
    """Replays a stage dump from the fp32 reference oracle as seam 1.

    The oracle (``flux_oracle.py encode``) writes ``encoder.out`` at exactly
    the seam's ``(1, S, 3 * hidden)`` shape, so a dump can be substituted for
    the encoder without running torch inside the pipeline. This is the route
    that makes a denoise run reproducible against the numerical oracle rather
    than merely self-consistent.
    """

    def __init__(
        self,
        prompt_embeds: npt.NDArray[np.float32],
        config: KleinConfig,
    ) -> None:
        """Initializes the replay encoder.

        Args:
            prompt_embeds: ``(1, S, joint_attention_dim)`` reference output.
            config: Model geometry, checked against the dump.

        Raises:
            ValueError: If the dump does not match ``config``.
        """
        if prompt_embeds.ndim != 3:
            raise ValueError(
                f"prompt_embeds must be (1, S, D), got {prompt_embeds.shape}"
            )
        if prompt_embeds.shape[2] != config.joint_attention_dim:
            raise ValueError(
                f"reference width {prompt_embeds.shape[2]} != "
                f"joint_attention_dim {config.joint_attention_dim}; the dump "
                "was produced at different dims than the pipeline runs at"
            )
        self.prompt_embeds = np.ascontiguousarray(
            prompt_embeds, dtype=np.float32
        )
        self.config = config

    @classmethod
    def from_stage_dump(
        cls,
        reference_dir: Path | str,
        config: KleinConfig,
        stage: str = "encoder.out",
    ) -> ReferenceTextEncoder:
        """Loads ``<reference_dir>/<stage>.npy`` written by the oracle.

        Args:
            reference_dir: Directory holding the oracle's stage dumps.
            config: Model geometry.
            stage: Stage name to replay.

        Returns:
            The replay encoder.

        Raises:
            FileNotFoundError: If the stage dump is absent.
        """
        path = Path(reference_dir) / f"{stage}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"no {stage} dump under {reference_dir}")
        return cls(np.load(path), config)

    def __call__(self, request: TextEncodeRequest) -> TextConditioning:
        """Returns the replayed conditioning, broadcast to the batch size.

        Args:
            request: The tokenized prompt and the batch size.

        Returns:
            The replayed :class:`TextConditioning`.
        """
        request.tokens.validate(self.config)
        seq = self.prompt_embeds.shape[1]
        embeds = np.tile(self.prompt_embeds, (request.batch_size, 1, 1))
        bias = np.tile(
            request.tokens.attention_bias, (request.batch_size, 1, 1, 1)
        )
        return TextConditioning(
            prompt_embeds=np.ascontiguousarray(embeds),
            text_ids=text_ids(seq, batch_size=request.batch_size),
            attention_bias=bias.astype(np.float32),
        )


class SyntheticBlockStack:
    """Stands in for the on-device DiT block driver (seam 2).

    Returns a velocity built from *every* input the real stack consumes -- the
    latent, the timestep embedding, the prompt embeddings in token order, and
    the image position ids -- so the end-to-end test can prove each one reaches
    the device call by perturbing it and watching the image move.

    The position term is what gives a synthetic run a spatially structured
    image rather than filtered noise, which incidentally makes a broken 2x2
    pack visible by eye as well as by assertion.
    """

    def __init__(
        self,
        config: KleinConfig,
        *,
        seed: int = 1,
        latent_gain: float = 1.0,
        cond_gain: float = 0.1,
        spatial_gain: float = 0.5,
    ) -> None:
        """Initializes the block-stack stub.

        Args:
            config: Model geometry.
            seed: Projection seed.
            latent_gain: Coefficient on the latent itself. At 1.0 the Euler
                trajectory decays by ``prod(1 + dt)``, which is well
                conditioned for the 4-step ladder.
            cond_gain: Coefficient on the text + timestep term.
            spatial_gain: Coefficient on the position term.
        """
        self.config = config
        rng = np.random.RandomState(seed)
        channels = config.in_channels
        self.text_proj = _projection(
            rng, (config.joint_attention_dim, channels)
        )
        self.temb_proj = _projection(rng, (config.dim, channels))
        self.position_proj = _projection(rng, (2, channels))
        self.latent_gain = np.float32(latent_gain)
        self.cond_gain = np.float32(cond_gain)
        self.spatial_gain = np.float32(spatial_gain)
        self._cached_text: (
            tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]] | None
        ) = None

    def _text_summary(
        self, prompt_embeds: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """Order-sensitive pooling of the prompt, cached across steps.

        The cache holds the array itself, not its ``id()``. A bare ``id()``
        key is a correctness bug here rather than a style point: once the
        previous prompt's array is collected, numpy hands the same address to
        the next same-shape allocation, and the second prompt of a run that
        reuses one :class:`PipelineStages` would silently get the first
        prompt's conditioning. Keeping the reference makes the address
        unreusable while it is still the key.
        """
        if self._cached_text is not None and self._cached_text[0] is (
            prompt_embeds
        ):
            return self._cached_text[1]
        seq = prompt_embeds.shape[1]
        ramp = np.linspace(1.0, -1.0, seq, dtype=np.float32)
        pooled = np.einsum("bsd,s->bd", prompt_embeds, ramp) / np.float32(seq)
        summary = (pooled @ self.text_proj).astype(np.float32)
        self._cached_text = (prompt_embeds, summary)
        return summary

    def __call__(
        self, request: DiTBlockStackRequest
    ) -> npt.NDArray[np.float32]:
        """Returns the velocity for one denoise step.

        Args:
            request: The full block-stack argument bundle.

        Returns:
            ``(B, image_seq_len, 128)`` velocity, not aliasing the input.
        """
        request.validate(self.config)
        geometry = request.geometry

        conditioning = (
            self._text_summary(request.text.prompt_embeds)
            + request.conditioning.temb @ self.temb_proj
        )

        rows = request.image_ids[..., 1].astype(np.float32)
        cols = request.image_ids[..., 2].astype(np.float32)
        normalized = np.stack(
            [
                rows / max(geometry.token_rows - 1, 1) - np.float32(0.5),
                cols / max(geometry.token_cols - 1, 1) - np.float32(0.5),
            ],
            axis=-1,
        )
        # einsum, not `@`: numpy 2.5's matmul raises spurious FPE warnings on
        # a contraction this narrow (K = 2) even though the result is finite.
        spatial = np.einsum("bsk,kc->bsc", normalized, self.position_proj)

        velocity = (
            request.latent * self.latent_gain
            + conditioning[:, np.newaxis, :] * self.cond_gain
            + spatial * self.spatial_gain
        )
        return np.ascontiguousarray(velocity, dtype=np.float32)


class SyntheticVaeDecoder:
    """Stands in for the on-device VAE decoder driver (seam 3).

    Mixes the 32 latent channels down to RGB and nearest-upsamples by the VAE's
    own ``vae_scale_factor``, which is the decoder's spatial contract without
    any of its conv stack. ``tanh`` bounds the output to the ``[-1, 1]`` the
    real decoder emits, so the host quantization downstream is exercised on a
    correctly ranged tensor.
    """

    def __init__(self, config: KleinConfig, *, seed: int = 2) -> None:
        """Initializes the decoder stub.

        Args:
            config: Model geometry; supplies the channel count and the stride.
            seed: Channel-mixing seed.
        """
        self.config = config
        rng = np.random.RandomState(seed)
        self.channel_mix = _projection(rng, (config.vae_latent_channels, 3))

    def __call__(self, request: VaeDecodeInput) -> npt.NDArray[np.float32]:
        """Decodes a latent to RGB.

        Args:
            request: The denormalized, unpacked latent.

        Returns:
            ``(B, 3, height, width)`` float32 RGB in ``[-1, 1]``.
        """
        request.validate(self.config)
        mixed = np.einsum("bchw,cr->brhw", request.latent, self.channel_mix)
        stride = self.config.vae_scale_factor
        upsampled = np.repeat(np.repeat(mixed, stride, axis=2), stride, axis=3)
        return np.tanh(upsampled).astype(np.float32)


def synthetic_stages(
    config: KleinConfig,
    geometry: LatentGeometry,
    *,
    seed: int = 0,
) -> tuple[
    SyntheticTokenizer,
    SyntheticTextEncoder,
    SyntheticBlockStack,
    SyntheticVaeDecoder,
]:
    """Builds one synthetic instance of each stage.

    Args:
        config: Model geometry.
        geometry: Resolved latent geometry. Unused by the stubs today; taken so
            a stage that needs the resolution can have it without a signature
            change.
        seed: Base seed; each stage offsets it so their projections differ.

    Returns:
        The four stubs in pipeline order.
    """
    del geometry
    return (
        SyntheticTokenizer(config),
        SyntheticTextEncoder(config, seed=seed),
        SyntheticBlockStack(config, seed=seed + 1),
        SyntheticVaeDecoder(config, seed=seed + 2),
    )


__all__ = [
    "PromptTokenizer",
    "PromptTokens",
    "ReferenceTextEncoder",
    "SyntheticBlockStack",
    "SyntheticTextEncoder",
    "SyntheticTokenizer",
    "SyntheticVaeDecoder",
    "TextEncodeRequest",
    "TextEncoder",
    "VaeDecoder",
    "synthetic_stages",
]
