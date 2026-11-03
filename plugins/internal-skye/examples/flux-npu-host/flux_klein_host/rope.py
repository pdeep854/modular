"""The packed rope table the denoise MEF consumes as ``input9``.

Float64 mirror of ``Flux2PosEmbed`` plus ``pack_flux2_rope``'s pairing: per
position-axis, the unique cosines land in the even columns and the matching
sines in the odd ones. The provenance chain is
``test_flux_denoise_mef.py::_rope_tables`` (graph-verified against MAX's own
tables) -> ``whole_graph_prep._packed_freqs_cis`` (which staged the first
device run's ``input9``) -> here, moved so every consumer imports one
implementation. ``tests/test_rope.py`` pins the move bit-for-bit against the
pre-move expression.

The recording's tables were fp32-native; this mirror is float64 and callers
narrow at the staging boundary, exactly as the prep script did. That skew is
adjudicated in the run package's RUNBOOK ("plausible table-provenance skew").
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .config import KLEIN_4B


def packed_freqs_cis(
    ids: npt.NDArray[np.float64],
    *,
    axes_dim: tuple[int, ...] = KLEIN_4B.axes_dims_rope,
    theta: float = float(KLEIN_4B.rope_theta),
) -> npt.NDArray[np.float64]:
    """Builds the packed cos/sin table for a joint id sequence.

    Args:
        ids: ``(tokens, len(axes_dim))`` position ids -- text ids then image
            ids for the denoise graph's joint sequence.
        axes_dim: Rotary width per position axis; sums to the head dim.
        theta: Rope base.

    Returns:
        ``(tokens, sum(axes_dim))`` float64, even columns the unique cosines,
        odd columns the matching sines.

    Raises:
        ValueError: If ``ids`` does not carry one column per axis.
    """
    if ids.ndim != 2 or ids.shape[1] != len(axes_dim):
        raise ValueError(
            f"ids must be (tokens, {len(axes_dim)}) to match axes_dim "
            f"{axes_dim}, got {ids.shape}"
        )
    cos_out, sin_out = [], []
    for axis, dim in enumerate(axes_dim):
        freq = 1.0 / theta ** (np.arange(0, dim, 2, np.float64) / dim)
        freqs = ids[:, axis][:, None] * freq[None, :]
        cos_out.append(np.repeat(np.cos(freqs), 2, axis=1))
        sin_out.append(np.repeat(np.sin(freqs), 2, axis=1))
    cos = np.concatenate(cos_out, axis=-1)
    sin = np.concatenate(sin_out, axis=-1)
    packed = np.empty((ids.shape[0], sum(axes_dim)), np.float64)
    packed[:, 0::2] = cos[:, 0::2]
    packed[:, 1::2] = sin[:, 0::2]
    return packed


__all__ = ["packed_freqs_cis"]
