"""Validates latent init, the 2x2 pack, and the position ids."""

from __future__ import annotations

import numpy as np
import pytest
from flux_klein_host import (
    KLEIN_4B,
    KleinConfig,
    init_latent,
    latent_image_ids,
    pack,
    resolve_geometry,
    text_ids,
    unpack,
)


@pytest.mark.parametrize(
    ("size", "latent_side", "tokens"),
    [
        (512, 64, 1024),
        (1024, 128, 4096),
        (256, 32, 256),
    ],
)
def test_geometry_resolution(size: int, latent_side: int, tokens: int) -> None:
    geometry = resolve_geometry(KLEIN_4B, size, size)
    assert geometry.latent_height == latent_side
    assert geometry.latent_width == latent_side
    assert geometry.image_seq_len == tokens
    assert geometry.packed_channels == KLEIN_4B.in_channels


def test_non_multiple_sizes_snap_down() -> None:
    """MAX floors to whole tokens; 520 must not become 65 latent rows."""
    geometry = resolve_geometry(KLEIN_4B, 520, 500)
    assert geometry.latent_height == 64
    assert geometry.latent_width == 62
    assert geometry.latent_height % 2 == 0
    assert geometry.latent_width % 2 == 0


def test_tiny_sizes_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_geometry(KLEIN_4B, 8, 8)


def test_config_rejects_inconsistent_packing() -> None:
    """The in_channels == latent_channels * 4 identity is load-bearing."""
    with pytest.raises(ValueError):
        KleinConfig(in_channels=64)


def test_latent_shape_and_seed_reproducibility() -> None:
    geometry = resolve_geometry(KLEIN_4B, 512, 512)
    first = init_latent(geometry, seed=1234)
    second = init_latent(geometry, seed=1234)
    assert first.shape == (1, 32, 64, 64)
    assert first.dtype == np.float32
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, init_latent(geometry, seed=1235))


def test_latent_matches_max_noise_stream() -> None:
    """MAX seeds with RandomState; a Generator would give different noise and
    quietly break the reference-image comparison.

    Falsifier for KB entry
    known-limitations/validating-a-seeded-render-must-be-rng-free: while this
    holds, our noise stream is not a torch Generator stream, so a whole-image
    comparison against a torch-seeded reference cannot pass on correctness.
    """
    geometry = resolve_geometry(KLEIN_4B, 512, 512)
    expected = (
        np.random.RandomState(7)
        .standard_normal((1, 32, 64, 64))
        .astype(np.float32)
    )
    np.testing.assert_array_equal(init_latent(geometry, seed=7), expected)


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_pack_unpack_round_trip(size: int) -> None:
    geometry = resolve_geometry(KLEIN_4B, size, size)
    latent = init_latent(geometry, seed=11)
    packed = pack(latent)
    assert packed.shape == (1, geometry.image_seq_len, 128)
    np.testing.assert_array_equal(unpack(packed, geometry), latent)


def test_pack_channel_order() -> None:
    """Token channels are ``c * 4 + dy * 2 + dx``.

    Built from a latent whose value encodes its own coordinates, so a
    transposed pack shows up as a specific wrong number rather than as noise.
    """
    geometry = resolve_geometry(KLEIN_4B, 32, 32)
    c, h, w = geometry.unpacked_shape
    coords = np.arange(c * h * w, dtype=np.float32).reshape(1, c, h, w)
    packed = pack(coords)

    for channel in (0, 5, 31):
        for dy in (0, 1):
            for dx in (0, 1):
                token = 0  # top-left 2x2 patch
                got = packed[0, token, channel * 4 + dy * 2 + dx]
                assert got == coords[0, channel, dy, dx]


def test_pack_rejects_odd_spatial_dims() -> None:
    with pytest.raises(ValueError):
        pack(np.zeros((1, 32, 7, 8), dtype=np.float32))


def test_unpack_rejects_wrong_token_count() -> None:
    geometry = resolve_geometry(KLEIN_4B, 512, 512)
    with pytest.raises(ValueError):
        unpack(np.zeros((1, 16, 128), dtype=np.float32), geometry)


def test_image_ids_cover_the_token_grid() -> None:
    geometry = resolve_geometry(KLEIN_4B, 512, 512)
    ids = latent_image_ids(geometry)
    assert ids.shape == (1, 1024, 4)
    assert ids.dtype == np.int64
    # Axis 0 (time) and axis 3 (text position) are unused by image tokens.
    assert not ids[..., 0].any()
    assert not ids[..., 3].any()
    # Row-major over the 32x32 token grid.
    np.testing.assert_array_equal(ids[0, :4, 2], [0, 1, 2, 3])
    np.testing.assert_array_equal(ids[0, ::32, 1], np.arange(32))


def test_text_ids_occupy_a_separate_axis() -> None:
    """Text varies only on axis 3 and image only on axes 1-2, so RoPE
    keeps them apart."""
    ids = text_ids(KLEIN_4B.text_seq_len)
    assert ids.shape == (1, 512, 4)
    assert not ids[..., :3].any()
    np.testing.assert_array_equal(ids[0, :, 3], np.arange(512))


def test_rope_axes_partition_the_head_dim() -> None:
    assert sum(KLEIN_4B.axes_dims_rope) == KLEIN_4B.attention_head_dim
    assert len(KLEIN_4B.axes_dims_rope) == 4
