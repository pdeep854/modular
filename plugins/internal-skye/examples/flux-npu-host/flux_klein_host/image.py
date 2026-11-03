"""The last host step: decoder RGB to a PNG on disk.

Slice V6 of the P6 plan, minus the parts that belong elsewhere -- the
batch-norm denormalization and the 2x2 unpack live in
:func:`~flux_klein_host.interfaces.vae_decode_input`, upstream of the decoder,
because the VAE's statistics are per *packed* channel.

What is left is the quantization and the file. The PNG encoder is written
against the spec here rather than taken from Pillow: it is a dozen lines for
8-bit RGB, and the whole host scaffold otherwise needs nothing but numpy.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt


def to_rgb_uint8(rgb: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Quantizes decoder output to 8-bit interleaved RGB.

    Applies the diffusers convention, ``(x / 2 + 0.5)`` clamped to ``[0, 1]``
    and scaled by 255, and moves the channel axis last.

    Args:
        rgb: ``(B, 3, H, W)`` float32 in ``[-1, 1]``.

    Returns:
        ``(B, H, W, 3)`` uint8.

    Raises:
        ValueError: If the rank or channel count is wrong.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"expected (B, 3, H, W), got shape {rgb.shape}")
    scaled = np.asarray(rgb, dtype=np.float32) * np.float32(0.5) + np.float32(
        0.5
    )
    quantized = np.rint(np.clip(scaled, 0.0, 1.0) * np.float32(255.0))
    return np.ascontiguousarray(
        quantized.astype(np.uint8).transpose(0, 2, 3, 1)
    )


def _chunk(tag: bytes, data: bytes) -> bytes:
    """One length-tag-data-CRC PNG chunk."""
    payload = tag + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFF_FFFF)
    )


def encode_png(image: npt.NDArray[np.uint8]) -> bytes:
    """Encodes one ``(H, W, 3)`` uint8 image as a PNG.

    Args:
        image: Interleaved 8-bit RGB.

    Returns:
        The PNG bytes.

    Raises:
        ValueError: If the shape is not ``(H, W, 3)``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got shape {image.shape}")
    height, width, _ = image.shape
    rows = np.ascontiguousarray(image, dtype=np.uint8)
    # Filter type 0 (None) in front of every scanline. Prepending a zero
    # column and flattening is the whole of it for unfiltered RGB.
    raw = np.zeros((height, width * 3 + 1), dtype=np.uint8)
    raw[:, 1:] = rows.reshape(height, width * 3)

    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _chunk(
                b"IHDR",
                struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0),
            ),
            _chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)),
            _chunk(b"IEND", b""),
        )
    )


def save_png(path: Path | str, image: npt.NDArray[np.uint8]) -> Path:
    """Writes one image to a PNG file.

    Args:
        path: Destination path; parent directories are created.
        image: ``(H, W, 3)`` or ``(1, H, W, 3)`` uint8.

    Returns:
        The path written.

    Raises:
        ValueError: If a batched image holds more than one entry.
    """
    array = np.asarray(image)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(
                f"save_png writes one image, got a batch of {array.shape[0]}; "
                "index it first"
            )
        array = array[0]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encode_png(array))
    return destination


__all__ = ["encode_png", "save_png", "to_rgb_uint8"]
