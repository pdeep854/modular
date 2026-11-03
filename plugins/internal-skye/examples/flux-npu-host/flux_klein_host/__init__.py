"""Host-side pipeline for FLUX.2-Klein-4B on the on-device NPU.

Device-independent, numpy-only. This package owns the parts of the pipeline
that are pure host math -- the noise latent, the flow-matching schedule, the
timestep conditioning, the denoise loop, the latent-to-image tail -- and it
owns the orchestration that chains them. It defines, but does not implement,
the seams where the device drivers plug in: :mod:`flux_klein_host.interfaces`
fixes the data crossing them, :mod:`flux_klein_host.stages` fixes the calls.

:func:`flux_klein_host.pipeline.run_pipeline` is the top-level driver. Every
device stage it reaches is a synthetic stub today; each is replaced by passing
a different :class:`~flux_klein_host.pipeline.PipelineStages`.
"""

from __future__ import annotations

from .checkpoints import StageCheckpoints, maybe_checkpoints
from .config import (
    DEFAULT_NUM_STEPS,
    KLEIN_4B,
    KLEIN_4B_SCHEDULER,
    KleinConfig,
    LatentGeometry,
    SchedulerConfig,
    resolve_geometry,
)
from .denoise import (
    DenoiseResult,
    PrecomputedSchedule,
    StepHook,
    StepRecord,
    denoise,
    precompute_schedule,
)
from .image import encode_png, save_png, to_rgb_uint8
from .interfaces import (
    DiTBlockStack,
    DiTBlockStackRequest,
    LatentBatchNormStats,
    ModulationTriple,
    StepConditioning,
    StepModulation,
    TextConditioning,
    VaeDecodeInput,
    text_conditioning_from_hidden_states,
    vae_decode_input,
)
from .latents import (
    batch_norm_denormalize,
    init_latent,
    latent_image_ids,
    pack,
    text_ids,
    unpack,
)
from .pipeline import (
    PipelineRequest,
    PipelineResult,
    PipelineStages,
    default_stages,
    run_pipeline,
)
from .scheduler import (
    FlowMatchEulerScheduler,
    empirical_mu,
    linear_mu,
    time_shift_exponential,
)
from .stages import (
    PromptTokenizer,
    PromptTokens,
    ReferenceTextEncoder,
    SyntheticBlockStack,
    SyntheticTextEncoder,
    SyntheticTokenizer,
    SyntheticVaeDecoder,
    TextEncoder,
    TextEncodeRequest,
    VaeDecoder,
    synthetic_stages,
)
from .timestep_embed import (
    TIMESTEP_SCALE,
    TimestepEmbedderWeights,
    embed_timestep,
    sinusoidal_embedding,
)
from .weights import PipelineWeights

__all__ = [
    "DEFAULT_NUM_STEPS",
    "KLEIN_4B",
    "KLEIN_4B_SCHEDULER",
    "TIMESTEP_SCALE",
    "DenoiseResult",
    "DiTBlockStack",
    "DiTBlockStackRequest",
    "FlowMatchEulerScheduler",
    "KleinConfig",
    "LatentBatchNormStats",
    "LatentGeometry",
    "ModulationTriple",
    "PipelineRequest",
    "PipelineResult",
    "PipelineStages",
    "PipelineWeights",
    "PrecomputedSchedule",
    "PromptTokenizer",
    "PromptTokens",
    "ReferenceTextEncoder",
    "SchedulerConfig",
    "StageCheckpoints",
    "StepConditioning",
    "StepHook",
    "StepModulation",
    "StepRecord",
    "SyntheticBlockStack",
    "SyntheticTextEncoder",
    "SyntheticTokenizer",
    "SyntheticVaeDecoder",
    "TextConditioning",
    "TextEncodeRequest",
    "TextEncoder",
    "TimestepEmbedderWeights",
    "VaeDecodeInput",
    "VaeDecoder",
    "batch_norm_denormalize",
    "default_stages",
    "denoise",
    "embed_timestep",
    "empirical_mu",
    "encode_png",
    "init_latent",
    "latent_image_ids",
    "linear_mu",
    "maybe_checkpoints",
    "pack",
    "precompute_schedule",
    "resolve_geometry",
    "run_pipeline",
    "save_png",
    "sinusoidal_embedding",
    "synthetic_stages",
    "text_conditioning_from_hidden_states",
    "text_ids",
    "time_shift_exponential",
    "to_rgb_uint8",
    "unpack",
    "vae_decode_input",
]
