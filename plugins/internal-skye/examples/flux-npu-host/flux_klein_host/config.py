"""FLUX.2-Klein-4B model geometry, from the model's public component configs.

Every value here is transcribed from a public (non weight-gated) ``config.json``
on ``black-forest-labs/FLUX.2-klein-4B`` at revision ``e7b7dc27``, which is the
revision pinned in ``max/tests/integration/hf-repo-lock.tsv``. Nothing in this
module reads a checkpoint, so the whole host scaffold runs without weights.

The one derived quantity worth stating up front is the latent packing, because
three separate "patch" notions collide in the FLUX.2 configs:

* ``transformer/config.json`` has ``patch_size: 1`` -- the DiT does *not*
  patchify again on top of the pack below.
* ``vae/config.json`` has ``patch_size: [2, 2]`` -- this is the 2x2 latent pack
  that turns VAE latent channels into DiT input channels, *not* an extra
  pixel-space unshuffle in the VAE itself.
* ``block_out_channels`` has 4 entries, so the VAE conv stack strides by
  ``2 ** (4 - 1) == 8``.

Those compose to ``in_channels == latent_channels * 4`` (``128 == 32 * 4``),
which the configs satisfy exactly, and to a total pixels-per-token factor of
``vae_scale_factor * pack == 16``. See ``LatentGeometry`` for the resolved
shapes and ``docs`` in the package README for the cross-check against MAX.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KleinConfig:
    """Static FLUX.2-Klein-4B geometry.

    Field values mirror the public component configs; see the module docstring
    for provenance. Defaults are the 4B variant.
    """

    # --- transformer/config.json ---
    in_channels: int = 128
    out_channels: int | None = None
    num_attention_heads: int = 24
    attention_head_dim: int = 128
    num_layers: int = 5
    num_single_layers: int = 20
    joint_attention_dim: int = 7680
    mlp_ratio: float = 3.0
    timestep_guidance_channels: int = 256
    guidance_embeds: bool = False
    axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32)
    rope_theta: int = 2000
    dit_patch_size: int = 1
    eps: float = 1e-6

    # --- vae/config.json ---
    vae_latent_channels: int = 32
    vae_block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    vae_batch_norm_eps: float = 1e-4
    vae_layers_per_block: int = 2
    vae_mid_block_add_attention: bool = True

    # --- text_encoder/config.json (Qwen3-4B) ---
    text_hidden_size: int = 2560
    text_seq_len: int = 512
    text_hidden_state_layers: tuple[int, ...] = (9, 18, 27)
    text_num_attention_heads: int = 32
    text_num_key_value_heads: int = 8
    """GQA: 32 query heads share 8 key/value heads."""

    text_head_dim: int = 128
    """``head_dim`` is explicit in the Qwen3 config, so ``q_dim`` is
    ``32 * 128 == 4096`` rather than ``text_hidden_size``."""

    text_intermediate_size: int = 9728

    def __post_init__(self) -> None:
        if self.in_channels != self.vae_latent_channels * self.pack_factor**2:
            raise ValueError(
                f"in_channels ({self.in_channels}) must equal "
                f"vae_latent_channels ({self.vae_latent_channels}) * "
                f"{self.pack_factor}^2 -- the configs disagree about the "
                "latent packing."
            )
        if sum(self.axes_dims_rope) != self.attention_head_dim:
            raise ValueError(
                f"axes_dims_rope {self.axes_dims_rope} must sum to "
                f"attention_head_dim ({self.attention_head_dim})."
            )
        if (
            len(self.text_hidden_state_layers) * self.text_hidden_size
            != self.joint_attention_dim
        ):
            raise ValueError(
                "The encoder hidden-state stack must be as wide as "
                f"joint_attention_dim ({self.joint_attention_dim}); got "
                f"{len(self.text_hidden_state_layers)} x "
                f"{self.text_hidden_size}."
            )

    @property
    def dim(self) -> int:
        """DiT hidden width (3072 for Klein-4B)."""
        return self.num_attention_heads * self.attention_head_dim

    @property
    def dit_out_channels(self) -> int:
        """Packed channel count the DiT emits, matching its input."""
        return self.out_channels or self.in_channels

    @property
    def pack_factor(self) -> int:
        """Side of the square latent pack (2 -- a 2x2 patch per token)."""
        return 2

    @property
    def vae_scale_factor(self) -> int:
        """Pixel:latent spatial stride of the VAE conv stack.

        Mirrors ``2 ** (len(block_out_channels) - 1)``, the formula MAX uses in
        ``max/python/max/pipelines/lib/pixel_tokenizer.py``.
        """
        return 2 ** (len(self.vae_block_out_channels) - 1)

    @property
    def pixels_per_token_side(self) -> int:
        """Pixels along one side collapsed into a single DiT token (16)."""
        return self.vae_scale_factor * self.pack_factor

    @property
    def num_blocks(self) -> int:
        """Total transformer blocks the block driver must chain (25)."""
        return self.num_layers + self.num_single_layers

    @property
    def text_num_layers(self) -> int:
        """Encoder blocks a run must execute (28).

        One more than the deepest selected state, because transformers applies
        the final norm to the last entry of ``hidden_states`` -- so the deepest
        selection has to stay interior, and therefore un-normed. MAX early-exits
        one block sooner; the extra block's output is unused either way.
        """
        return max(self.text_hidden_state_layers) + 1


@dataclass(frozen=True)
class LatentGeometry:
    """Resolved latent and token shapes for one target image size."""

    height: int
    width: int
    latent_height: int
    latent_width: int
    token_rows: int
    token_cols: int
    latent_channels: int
    packed_channels: int
    text_seq_len: int

    @property
    def image_seq_len(self) -> int:
        """Number of image tokens the DiT sees (1024 at 512^2)."""
        return self.token_rows * self.token_cols

    @property
    def joint_seq_len(self) -> int:
        """Image + text tokens -- the ``M`` of the block driver's matmuls."""
        return self.image_seq_len + self.text_seq_len

    @property
    def unpacked_shape(self) -> tuple[int, int, int]:
        """``(C, H, W)`` of the VAE-facing latent."""
        return (self.latent_channels, self.latent_height, self.latent_width)

    @property
    def packed_shape(self) -> tuple[int, int]:
        """``(tokens, packed_channels)`` of the DiT-facing latent."""
        return (self.image_seq_len, self.packed_channels)


def resolve_geometry(
    config: KleinConfig,
    height: int = 512,
    width: int = 512,
) -> LatentGeometry:
    """Resolves a target image size to latent and token shapes.

    Reproduces MAX's rounding (``pixel_tokenizer.py``): the latent side is
    snapped *down* to an even number of latent pixels, so the 2x2 pack always
    divides it. A 512x512 request therefore yields a 64x64x32 latent, a 32x32
    token grid (1024 image tokens), and -- with the 512 padded text tokens --
    the ``M = 1536`` joint sequence the block driver is sized for.

    Args:
        config: Model geometry.
        height: Target image height in pixels.
        width: Target image width in pixels.

    Returns:
        The resolved :class:`LatentGeometry`.

    Raises:
        ValueError: If either side is too small to yield a single token.
    """
    step = config.pixels_per_token_side
    latent_height = config.pack_factor * (int(height) // step)
    latent_width = config.pack_factor * (int(width) // step)
    if latent_height < config.pack_factor or latent_width < config.pack_factor:
        raise ValueError(
            f"{height}x{width} is below the {step}x{step} pixel floor for one "
            "DiT token."
        )
    return LatentGeometry(
        height=int(height),
        width=int(width),
        latent_height=latent_height,
        latent_width=latent_width,
        token_rows=latent_height // config.pack_factor,
        token_cols=latent_width // config.pack_factor,
        latent_channels=config.vae_latent_channels,
        packed_channels=config.in_channels,
        text_seq_len=config.text_seq_len,
    )


@dataclass(frozen=True)
class SchedulerConfig:
    """Public ``scheduler/scheduler_config.json`` for FLUX.2-Klein-4B.

    ``use_empirical_mu`` is the one field with no counterpart in the JSON: MAX
    injects it for the whole FLUX.2 family in ``Flux2Tokenizer.__init__``
    (``scheduler_config_overrides={"use_empirical_mu": True}``), which switches
    the resolution-dependent shift from the linear ``base_shift``/``max_shift``
    interpolation to the piecewise fit published in black-forest-labs/flux2
    ``sampling.py``. It defaults to True here so the scaffold matches MAX's
    Klein path rather than the bare JSON.
    """

    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    base_shift: float = 0.5
    max_shift: float = 1.15
    shift: float = 3.0
    shift_terminal: float | None = None
    use_dynamic_shifting: bool = True
    time_shift_type: str = "exponential"
    num_train_timesteps: int = 1000
    use_empirical_mu: bool = True


KLEIN_4B = KleinConfig()
KLEIN_4B_SCHEDULER = SchedulerConfig()

#: Steps the distilled Klein checkpoint is trained for. MAX's
#: ``_DISTILLED_KLEIN_NUM_STEPS`` and the P6 demo config both use 4.
DEFAULT_NUM_STEPS = 4

__all__ = [
    "DEFAULT_NUM_STEPS",
    "KLEIN_4B",
    "KLEIN_4B_SCHEDULER",
    "KleinConfig",
    "LatentGeometry",
    "SchedulerConfig",
    "resolve_geometry",
]
