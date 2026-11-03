"""Latent initialization and the 2x2 pack/unpack the DiT works in.

Two latent layouts exist and the pipeline moves between them exactly twice:

``unpacked`` ``(B, 32, H, W)``
    The VAE's layout. This is what the decoder consumes and what noise is
    sampled in.
``packed`` ``(B, H//2 * W//2, 128)``
    The DiT's layout: each 2x2 spatial patch becomes one token whose 128
    channels are the 4 patch positions of the 32 latent channels.

:func:`pack` and :func:`unpack` are exact inverses and reproduce MAX's
``Flux2KleinExecutor._patchify_and_pack`` and the inverse permutation in
``flux2/components/vae_decoder.py``. The channel order within a token is
``c * 4 + dy * 2 + dx``; getting that wrong is silent (shapes still match) and
shows up only as a scrambled image, so the round-trip is asserted in tests.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import LatentGeometry


def init_latent(
    geometry: LatentGeometry,
    seed: int | None = None,
    batch_size: int = 1,
) -> npt.NDArray[np.float32]:
    """Samples the initial unpacked noise latent.

    Uses ``np.random.RandomState`` rather than the modern ``Generator`` so the
    stream matches MAX's ``PixelGenerationTokenizer._randn_tensor`` bit for
    bit -- the reference image and the on-device image must start from the
    same noise for the P6 two-tier comparison to mean anything.

    Args:
        geometry: Resolved latent geometry.
        seed: Seed for the noise; ``None`` draws from OS entropy.
        batch_size: Number of images.

    Returns:
        Float32 array of shape ``(batch_size, 32, H, W)``.
    """
    shape = (batch_size, *geometry.unpacked_shape)
    rng = np.random.RandomState(seed)
    return rng.standard_normal(shape).astype(np.float32)


def pack(latent: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Packs ``(B, C, H, W)`` into DiT tokens ``(B, H//2 * W//2, C*4)``.

    Args:
        latent: Unpacked latent with even spatial dims.

    Returns:
        Contiguous packed latent.

    Raises:
        ValueError: If the rank is wrong or a spatial dim is odd.
    """
    if latent.ndim != 4:
        raise ValueError(f"expected (B, C, H, W), got shape {latent.shape}")
    b, c, h, w = latent.shape
    if h % 2 or w % 2:
        raise ValueError(f"spatial dims must be even, got {h}x{w}")

    h2, w2 = h // 2, w // 2
    arr = latent.reshape(b, c, h2, 2, w2, 2)
    arr = arr.transpose(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h2, w2)
    arr = arr.reshape(b, c * 4, h2 * w2).transpose(0, 2, 1)
    return np.ascontiguousarray(arr)


def unpack(
    packed: npt.NDArray[np.float32], geometry: LatentGeometry
) -> npt.NDArray[np.float32]:
    """Unpacks DiT tokens ``(B, T, C*4)`` back to ``(B, C, H, W)``.

    Args:
        packed: Packed latent.
        geometry: Geometry supplying the token grid shape.

    Returns:
        Contiguous unpacked latent.

    Raises:
        ValueError: If the token count or channel count disagrees with
            ``geometry``.
    """
    if packed.ndim != 3:
        raise ValueError(f"expected (B, T, C*4), got shape {packed.shape}")
    b, tokens, packed_c = packed.shape
    if tokens != geometry.image_seq_len:
        raise ValueError(
            f"got {tokens} tokens, geometry expects {geometry.image_seq_len}"
        )
    if packed_c != geometry.packed_channels:
        raise ValueError(
            f"got {packed_c} packed channels, geometry expects "
            f"{geometry.packed_channels}"
        )

    h2, w2 = geometry.token_rows, geometry.token_cols
    c = packed_c // 4
    arr = packed.reshape(b, h2, w2, packed_c).transpose(0, 3, 1, 2)
    arr = arr.reshape(b, c, 2, 2, h2, w2)
    arr = arr.transpose(0, 1, 4, 2, 5, 3)
    return np.ascontiguousarray(arr.reshape(b, c, h2 * 2, w2 * 2))


def latent_image_ids(
    geometry: LatentGeometry, batch_size: int = 1
) -> npt.NDArray[np.int64]:
    """Builds the 4-axis RoPE position ids for image tokens.

    FLUX.2 uses four position axes ``(T, H, W, L)`` matching
    ``axes_dims_rope = (32, 32, 32, 32)``. Image tokens vary only in ``H`` and
    ``W``, over the *token* grid rather than the latent grid.

    Args:
        geometry: Resolved geometry.
        batch_size: Number of images.

    Returns:
        Int64 array of shape ``(batch_size, image_seq_len, 4)``.
    """
    rows = np.arange(geometry.token_rows, dtype=np.int64)
    cols = np.arange(geometry.token_cols, dtype=np.int64)
    h_coords, w_coords = np.meshgrid(rows, cols, indexing="ij")
    zeros = np.zeros_like(h_coords)
    ids = np.stack([zeros, h_coords, w_coords, zeros], axis=-1).reshape(-1, 4)
    return np.ascontiguousarray(
        np.tile(ids[np.newaxis, :, :], (batch_size, 1, 1))
    )


def text_ids(seq_len: int, batch_size: int = 1) -> npt.NDArray[np.int64]:
    """Builds the 4-axis RoPE position ids for text tokens.

    Text tokens sit at ``(0, 0, 0, position)``, so they occupy a distinct
    subspace from image tokens in the joint attention.

    Args:
        seq_len: Padded text length (512 for Klein).
        batch_size: Number of images.

    Returns:
        Int64 array of shape ``(batch_size, seq_len, 4)``.
    """
    zeros = np.zeros(seq_len, dtype=np.int64)
    coords = np.stack(
        [zeros, zeros, zeros, np.arange(seq_len, dtype=np.int64)], axis=-1
    )
    return np.ascontiguousarray(
        np.tile(coords[np.newaxis, :, :], (batch_size, 1, 1))
    )


def batch_norm_denormalize(
    packed: npt.NDArray[np.float32],
    bn_mean: npt.NDArray[np.float32],
    bn_var: npt.NDArray[np.float32],
    eps: float,
) -> npt.NDArray[np.float32]:
    """Undoes the VAE's latent batch-norm: ``x * sqrt(var + eps) + mean``.

    FLUX.2's VAE normalizes latents with per-channel batch-norm statistics
    instead of the single scale/shift pair older FLUX VAEs use. The statistics
    are per *packed* channel and the denorm runs on the packed latent, before
    :func:`unpack` -- that ordering is load-bearing and matches
    ``vae_decoder.py``.

    Args:
        packed: Packed latent ``(B, T, C*4)``.
        bn_mean: Per-channel mean, shape ``(C*4,)``.
        bn_var: Per-channel variance, shape ``(C*4,)``.
        eps: ``batch_norm_eps`` from the VAE config.

    Returns:
        Denormalized packed latent.

    Raises:
        ValueError: If the statistics do not match the channel count.
    """
    channels = packed.shape[-1]
    if bn_mean.shape != (channels,) or bn_var.shape != (channels,):
        raise ValueError(
            f"bn_mean/bn_var must have shape ({channels},), got "
            f"{bn_mean.shape}/{bn_var.shape}"
        )
    std = np.sqrt(bn_var.astype(np.float32) + np.float32(eps))
    return (packed * std + bn_mean.astype(np.float32)).astype(np.float32)


__all__ = [
    "batch_norm_denormalize",
    "init_latent",
    "latent_image_ids",
    "pack",
    "text_ids",
    "unpack",
]
