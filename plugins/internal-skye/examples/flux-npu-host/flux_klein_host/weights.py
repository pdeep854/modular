"""Reading the host's share of the V1 weight artifact.

The converted artifact (``flux_weights.py``, ``skye-npu-weights`` v1) is laid
out for the *device*: fp16, ``[N, K]``, K zero-padded to a multiple of 64,
128-byte aligned. Almost all of it is the drivers' business. Three things in it
are the host's, because the host computes them before any dispatch:

* the timestep MLP, which turns a scheduler sigma into ``temb``,
* the VAE's latent batch-norm statistics, which denormalize the final latent,
  and
* the four shared adaLN heads, which turn ``temb`` into the 17 modulation
  vectors the block stack needs. FLUX.2 evaluates them once per forward, not
  per block, so they are host work by the same argument as ``temb``.

This module pulls exactly those out and leaves the open artifact on the result
so a driver can reach the rest without opening the ``.bin`` twice. The adaLN
heads are ~640 MiB widened, and only a device run needs them, so they load
behind :meth:`PipelineWeights.load_modulation` rather than eagerly.

The fp16 values are widened to float32 here. The host arithmetic around them --
the sinusoidal projection, the SiLU, the denormalization -- is float32 in MAX
and in the scaffold, so widening at load keeps one dtype through the whole host
path and confines fp16 to where it belongs, which is the device.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import KLEIN_4B, KleinConfig
from .interfaces import LatentBatchNormStats
from .timestep_embed import (
    MODULATION_HEADS,
    ModulationWeights,
    TimestepEmbedderWeights,
)

#: Where ``flux_weights.py`` lives relative to the monorepo root. Imported by
#: path rather than as a package: the hexagon examples are scripts, not an
#: installable distribution, and the reference oracle already reaches MAX's own
#: helpers the same way.
_CONVERTER_RELPATH = (
    "plugins/internal-skye/examples/hexagon-ondevice/flux_weights.py"
)

#: Checkpoint names of the two host-side tensor groups, without the component
#: prefix the converter adds when it walks a diffusers snapshot.
#:
#: FLUX.2 renamed the embedder FLUX.1 called ``time_text_embed``: Klein's
#: checkpoint spells it ``time_guidance_embed``, and holds only the timestep
#: half of it (``guidance_embeds: false``, so there is no guidance embedder and
#: no text-pooling projection to share the module with).
TIMESTEP_LINEAR_1 = "time_guidance_embed.timestep_embedder.linear_1.weight"
TIMESTEP_LINEAR_2 = "time_guidance_embed.timestep_embedder.linear_2.weight"
BN_RUNNING_MEAN = "bn.running_mean"
BN_RUNNING_VAR = "bn.running_var"


def _repo_root(start: Path) -> Path | None:
    """Finds the monorepo root by looking for the converter."""
    for candidate in [start, *start.parents]:
        if (candidate / _CONVERTER_RELPATH).is_file():
            return candidate
    return None


def load_converter_module() -> Any:
    """Imports ``flux_weights`` from the sibling hexagon-ondevice example.

    Returns:
        The imported module.

    Raises:
        ImportError: If this file is not inside the monorepo.
    """
    cached = sys.modules.get("_skye_flux_weights")
    if cached is not None:
        return cached
    root = _repo_root(Path(__file__).resolve())
    if root is None:
        raise ImportError(
            "cannot locate the monorepo root; the V1 weight artifact is read "
            f"through {_CONVERTER_RELPATH} and this file must live in-tree"
        )
    spec = importlib.util.spec_from_file_location(
        "_skye_flux_weights", root / _CONVERTER_RELPATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_skye_flux_weights"] = module
    spec.loader.exec_module(module)
    return module


def _find(artifact: Any, suffix: str) -> str:
    """Resolves a bare checkpoint name to the artifact's prefixed one.

    Raises:
        KeyError: If no tensor matches, or more than one does.
    """
    matches = [
        name
        for name in artifact.names
        if name == suffix or name.endswith("." + suffix)
    ]
    if not matches:
        raise KeyError(f"no tensor named {suffix!r} in the artifact")
    if len(matches) > 1:
        raise KeyError(f"{suffix!r} is ambiguous: {matches}")
    return matches[0]


def _depadded(artifact: Any, name: str) -> npt.NDArray[np.float32]:
    """Reads a matmul weight back as float32 with its K padding trimmed."""
    record = artifact.record(name)
    values = np.asarray(artifact.tensor(name), dtype=np.float32)
    k_real = record.get("k_real")
    if k_real and values.ndim == 2 and int(k_real) < values.shape[1]:
        values = values[:, : int(k_real)]
    return np.ascontiguousarray(values)


def _load_modulation(artifact: Any, config: KleinConfig) -> ModulationWeights:
    """Reads the four adaLN heads out of an open artifact.

    Raises:
        ValueError: If a head is not ``(chunks * dim, dim)``.
    """
    heads: list[npt.NDArray[np.float32]] = []
    for name, chunks in MODULATION_HEADS:
        values = _depadded(artifact, _find(artifact, name))
        if values.shape != (chunks * config.dim, config.dim):
            raise ValueError(
                f"{name} is {values.shape}, config expects "
                f"{(chunks * config.dim, config.dim)}"
            )
        heads.append(values)
    return ModulationWeights(*heads)


@dataclass
class PipelineWeights:
    """The host-side weights the pipeline itself evaluates.

    Everything else in the artifact belongs to a device driver, which reaches
    it through :attr:`artifact`.
    """

    timestep: TimestepEmbedderWeights
    bn_stats: LatentBatchNormStats
    source: str
    """``"synthetic"`` or the manifest path the weights came from."""

    artifact: Any = None
    """The open ``flux_weights.Artifact``, or ``None`` for synthetic weights."""

    modulation: ModulationWeights | None = None
    """The four adaLN heads, once :meth:`load_modulation` has read them."""

    def load_modulation(
        self, config: KleinConfig = KLEIN_4B, *, seed: int = 0
    ) -> ModulationWeights:
        """Reads (and caches) the four shared adaLN heads.

        Separate from the constructor because they are ~640 MiB widened and
        only a run that marshals modulation to a device driver needs them.
        Without an artifact the heads are random, which exercises the crossing
        without claiming anything about the model -- the driver reports
        ``weight_source`` alongside ``modulation_source`` so the two cannot be
        confused.

        Args:
            config: Model geometry, checked against the loaded shapes.
            seed: Seed for the synthetic heads, when there is no artifact.

        Returns:
            The four projections.

        Raises:
            ValueError: If a loaded head disagrees with ``config``.
        """
        if self.modulation is None:
            self.modulation = (
                ModulationWeights.random(config, seed=seed)
                if self.artifact is None
                else _load_modulation(self.artifact, config)
            )
        return self.modulation

    def close(self) -> None:
        """Releases the memory-mapped artifact, if any."""
        if self.artifact is not None:
            self.artifact.close()
            self.artifact = None

    def __enter__(self) -> PipelineWeights:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @classmethod
    def synthetic(
        cls, config: KleinConfig = KLEIN_4B, *, seed: int = 0
    ) -> PipelineWeights:
        """Builds weight-free stand-ins for a synthetic run.

        The batch-norm statistics are the identity, so the denormalization is
        a no-op and the packed/unpacked latent comparison in the end-to-end
        test is not confounded by a scale the stub decoder cannot know about.

        Args:
            config: Model geometry.
            seed: Timestep-MLP seed.

        Returns:
            Synthetic :class:`PipelineWeights`.
        """
        return cls(
            timestep=TimestepEmbedderWeights.random(config, seed=seed),
            bn_stats=LatentBatchNormStats.identity(
                config.in_channels, eps=config.vae_batch_norm_eps
            ),
            source="synthetic",
        )

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path | str,
        config: KleinConfig = KLEIN_4B,
    ) -> PipelineWeights:
        """Loads the host-side weights from a converted V1 artifact.

        Args:
            manifest_path: The artifact's ``.json`` manifest.
            config: Model geometry, checked against the loaded shapes.

        Returns:
            The loaded :class:`PipelineWeights`, holding the open artifact.

        Raises:
            ValueError: If a loaded tensor disagrees with ``config``.
        """
        converter = load_converter_module()
        artifact = converter.Artifact(Path(manifest_path))
        try:
            linear_1 = _depadded(artifact, _find(artifact, TIMESTEP_LINEAR_1))
            linear_2 = _depadded(artifact, _find(artifact, TIMESTEP_LINEAR_2))
            if linear_1.shape != (
                config.dim,
                config.timestep_guidance_channels,
            ):
                raise ValueError(
                    f"timestep linear_1 is {linear_1.shape}, config expects "
                    f"{(config.dim, config.timestep_guidance_channels)}"
                )
            if linear_2.shape != (config.dim, config.dim):
                raise ValueError(
                    f"timestep linear_2 is {linear_2.shape}, config expects "
                    f"{(config.dim, config.dim)}"
                )

            mean = np.asarray(
                artifact.tensor(_find(artifact, BN_RUNNING_MEAN)),
                dtype=np.float32,
            )
            var = np.asarray(
                artifact.tensor(_find(artifact, BN_RUNNING_VAR)),
                dtype=np.float32,
            )
            if mean.shape != (config.in_channels,):
                raise ValueError(
                    f"latent batch-norm is {mean.shape}, config expects "
                    f"{(config.in_channels,)} -- the statistics are per "
                    "*packed* channel, so a decoder-width vector here means "
                    "the wrong tensor was picked up"
                )
            # A running variance is non-negative by construction. A negative
            # one means the wrong tensor was matched, and the only symptom
            # downstream is a NaN latent well after the cause.
            if not np.all(var >= 0.0):
                raise ValueError(
                    f"latent batch-norm variance has "
                    f"{int(np.count_nonzero(var < 0.0))} negative entries; "
                    "the tensor matched is not a running variance"
                )
        except Exception:
            artifact.close()
            raise

        return cls(
            timestep=TimestepEmbedderWeights(
                linear_1=linear_1, linear_2=linear_2
            ),
            bn_stats=LatentBatchNormStats(
                mean=np.ascontiguousarray(mean),
                var=np.ascontiguousarray(var),
                eps=config.vae_batch_norm_eps,
            ),
            source=str(manifest_path),
            artifact=artifact,
        )


__all__ = [
    "BN_RUNNING_MEAN",
    "BN_RUNNING_VAR",
    "TIMESTEP_LINEAR_1",
    "TIMESTEP_LINEAR_2",
    "PipelineWeights",
    "load_converter_module",
]
