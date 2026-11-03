"""Per-stage tensor dumps for the P6 numerical oracle.

The P6 design's load-bearing correctness gate is per-stage numeric agreement
against an fp32 reference, not the final image: an image can look plausible
while a kernel is subtly wrong. This module is the sink for that -- the
pipeline hands it every stage boundary, and it writes each one as a float32
``.npy`` plus a manifest in the oracle's own ``skye-npu-reference`` format.

The format is what makes the dumps useful in both directions:

* as a **candidate**, ``flux_oracle.py compare --ref <oracle> --got <dump>``
  reads ``<name>.npy`` straight out of the directory this writes, and
* as a **reference**, the manifest carries the per-stage tolerance, so a later
  on-device run can be compared against a host run of the same pipeline.

Stage names are chosen so the encoder dump collides *deliberately* with the
oracle's: this writes ``encoder.out``, which is the name the oracle's encoder
reference already uses, so the two line up with no renaming.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .denoise import StepRecord

#: The oracle's manifest identity (``flux_oracle.MANIFEST_FORMAT``). Written
#: literally rather than imported, so dumping never drags torch in.
MANIFEST_FORMAT = "skye-npu-reference"
MANIFEST_VERSION = 1
MANIFEST_NAME = "stages.json"

#: The P6 §7 per-op band, the same defaults the oracle applies.
DEFAULT_ABS_TOL = 3e-2
DEFAULT_REL_TOL = 5e-2


@dataclass
class StageCheckpoints:
    """Collects per-stage tensors and writes them in the oracle's format."""

    out_dir: Path
    abs_tol: float = DEFAULT_ABS_TOL
    rel_tol: float = DEFAULT_REL_TOL
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        name: str,
        array: npt.NDArray[Any],
        *,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
    ) -> Path:
        """Writes one stage tensor.

        Integer stages -- token ids, the attention mask -- keep their dtype
        rather than being widened to float32, because a token id silently
        becoming 151643.0 is exactly the kind of thing a checkpoint exists to
        catch.

        Args:
            name: Stage name; becomes ``<name>.npy``.
            array: The tensor.
            abs_tol: Per-stage absolute tolerance override.
            rel_tol: Per-stage relative tolerance override.

        Returns:
            The path written.
        """
        values = np.asarray(array)
        if values.dtype.kind == "f":
            values = np.ascontiguousarray(values, dtype=np.float32)
        else:
            values = np.ascontiguousarray(values)
        path = self.out_dir / f"{name}.npy"
        np.save(path, values)
        self.entries[name] = {
            "path": path.name,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "abs_tol": self.abs_tol if abs_tol is None else abs_tol,
            "rel_tol": self.rel_tol if rel_tol is None else rel_tol,
        }
        return path

    def on_step(self, record: StepRecord) -> None:
        """The scaffold's :data:`~flux_klein_host.denoise.StepHook`.

        Dumps the velocity the block stack returned and the latent the Euler
        update produced. ``latent_in`` is not dumped: it is the previous step's
        ``latent`` (and, for step 0, ``denoise.latent_init``), so writing it
        would double the dump for no extra evidence.
        """
        step = f"denoise.step_{record.step_index:02d}"
        self.record(f"{step}.velocity", record.velocity)
        self.record(f"{step}.latent", record.latent_out)

    def write_manifest(self, meta: dict[str, Any] | None = None) -> Path:
        """Writes ``stages.json`` describing everything recorded.

        Args:
            meta: Extra provenance merged into the manifest.

        Returns:
            The manifest path.
        """
        manifest = {
            "format": MANIFEST_FORMAT,
            "version": MANIFEST_VERSION,
            **(meta or {}),
            "stages": self.entries,
        }
        path = self.out_dir / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        return path

    @property
    def names(self) -> list[str]:
        """Every stage recorded so far, in insertion order."""
        return list(self.entries)


@contextmanager
def maybe_checkpoints(
    out_dir: Path | str | None,
    **kwargs: Any,
) -> Iterator[StageCheckpoints | None]:
    """Yields a writer for ``out_dir``, or ``None`` when it is ``None``.

    Keeps the caller free of ``if checkpoints is not None`` around the manifest
    write, which is the one step easy to forget and silent when missed.

    Args:
        out_dir: Where to dump, or ``None`` to dump nothing.
        **kwargs: Forwarded to :class:`StageCheckpoints`.

    Yields:
        The writer, or ``None``.
    """
    if out_dir is None:
        yield None
        return
    writer = StageCheckpoints(Path(out_dir), **kwargs)
    try:
        yield writer
    finally:
        # Written even when the pipeline raised: a partial dump is how a
        # failing stage gets diagnosed, and the oracle skips stages it has no
        # candidate for anyway.
        writer.write_manifest()


__all__ = [
    "DEFAULT_ABS_TOL",
    "DEFAULT_REL_TOL",
    "MANIFEST_FORMAT",
    "MANIFEST_NAME",
    "MANIFEST_VERSION",
    "StageCheckpoints",
    "maybe_checkpoints",
]
