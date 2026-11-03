"""The packed rope-table mirror, pinned bit-for-bit to its previous home.

``packed_freqs_cis`` moved from ``whole_graph_prep.py`` (where it was inlined
as ``_packed_freqs_cis``) into the package so the thin runner can import the
numbers instead of copying them. The move is only safe if it changed nothing:
the table feeds the denoise MEF's ``input9``, whose provenance chain runs back
through ``test_flux_denoise_mef.py::_rope_tables`` to the graph-verified
float64 mirror. The reference below is the pre-move expression, kept verbatim,
so a drift in the shared implementation fails here byte-for-byte.
"""

from __future__ import annotations

import numpy as np
from flux_klein_host.config import KLEIN_4B, resolve_geometry
from flux_klein_host.latents import latent_image_ids, text_ids
from flux_klein_host.rope import packed_freqs_cis

_AXES_DIM = (32, 32, 32, 32)
_ROPE_THETA = 2000.0
_HEAD_DIM = 128


def _reference(ids: np.ndarray) -> np.ndarray:
    """`whole_graph_prep._packed_freqs_cis` as it was before the move."""
    cos_out, sin_out = [], []
    for axis, dim in enumerate(_AXES_DIM):
        freq = 1.0 / _ROPE_THETA ** (np.arange(0, dim, 2, np.float64) / dim)
        freqs = ids[:, axis][:, None] * freq[None, :]
        cos_out.append(np.repeat(np.cos(freqs), 2, axis=1))
        sin_out.append(np.repeat(np.sin(freqs), 2, axis=1))
    cos = np.concatenate(cos_out, axis=-1)
    sin = np.concatenate(sin_out, axis=-1)
    packed = np.empty((ids.shape[0], _HEAD_DIM), np.float64)
    packed[:, 0::2] = cos[:, 0::2]
    packed[:, 1::2] = sin[:, 0::2]
    return packed


def _joint_ids() -> np.ndarray:
    """Text-then-image ids at the recorded 512^2 geometry, as prep builds them."""
    geometry = resolve_geometry(KLEIN_4B, 512, 512)
    return np.concatenate(
        [
            text_ids(KLEIN_4B.text_seq_len)[0].astype(np.float64),
            latent_image_ids(geometry)[0].astype(np.float64),
        ],
        axis=0,
    )


def test_matches_the_premove_expression_bitwise() -> None:
    ids = _joint_ids()
    moved = packed_freqs_cis(
        ids, axes_dim=KLEIN_4B.axes_dims_rope, theta=KLEIN_4B.rope_theta
    )
    assert moved.dtype == np.float64
    assert moved.shape == (1536, 128)
    assert moved.tobytes() == _reference(ids).tobytes()


def test_defaults_are_kleins() -> None:
    """Calling with no keywords must mean Klein, the only model staged."""
    ids = _joint_ids()[:8]
    assert packed_freqs_cis(ids).tobytes() == _reference(ids).tobytes()


def test_refuses_mismatched_ids() -> None:
    ids = np.zeros((4, 3), np.float64)
    try:
        packed_freqs_cis(ids)
    except ValueError as error:
        assert "4" in str(error) or "axes" in str(error)
    else:
        raise AssertionError("a 3-axis id table must be refused")
