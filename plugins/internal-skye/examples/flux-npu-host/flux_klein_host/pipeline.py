"""The FLUX.2-Klein-4B pipeline driver -- P7 slice V8, host side.

One function wires the whole flow:

.. code-block:: text

    prompt
      -> tokenize                     (V2, host)
      -> text encoder                 (V3, device)  -> TextConditioning
      -> denoise loop, N steps        (V4, device)  -> packed latent
      -> BN-denorm then 2x2 unpack    (host)        -> VaeDecodeInput
      -> VAE decode                   (V5, device)  -> RGB in [-1, 1]
      -> quantize + PNG               (V6, host)

Every device stage is reached through a seam in :mod:`flux_klein_host.stages`,
so a real driver replaces a stub by being passed to :func:`run_pipeline` in a
different :class:`PipelineStages` -- nothing inside this module knows which
implementation it called. The stubs that ship today are synthetic: they make
the orchestration and the data flow testable on a machine with no NPU, and
they make no claim about the numbers.

The per-stage checkpoint hook is wired through the same call. Passing a
:class:`~flux_klein_host.checkpoints.StageCheckpoints` dumps every stage
boundary in the reference oracle's format, which is the P6 §7 correctness gate
and the reason the loop exposes ``on_step`` at all.

Run it:

.. code-block:: bash

    python -m flux_klein_host.pipeline --prompt "a red bicycle" \\
        --out-dir /tmp/klein-run --checkpoints /tmp/klein-run/stages
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .checkpoints import StageCheckpoints, maybe_checkpoints
from .config import (
    DEFAULT_NUM_STEPS,
    KLEIN_4B,
    KleinConfig,
    LatentGeometry,
    resolve_geometry,
)
from .denoise import DenoiseResult, denoise
from .device_bridge import (
    DEFAULT_STAGE_DIR,
    DeviceBlockStack,
    DeviceTextEncoder,
    DeviceVaeDecoder,
)
from .image import save_png, to_rgb_uint8
from .interfaces import (
    DiTBlockStack,
    TextConditioning,
    VaeDecodeInput,
    vae_decode_input,
)
from .latents import pack
from .scheduler import FlowMatchEulerScheduler
from .stages import (
    PromptTokenizer,
    PromptTokens,
    TextEncoder,
    TextEncodeRequest,
    VaeDecoder,
    synthetic_stages,
)
from .tokenizer import ArtifactTokenizer, KleinTokenizer
from .weights import PipelineWeights


@dataclass(frozen=True)
class PipelineRequest:
    """One image-generation request."""

    prompt: str
    height: int = 512
    width: int = 512
    num_steps: int = DEFAULT_NUM_STEPS
    seed: int | None = 0
    batch_size: int = 1


@dataclass(frozen=True)
class PipelineStages:
    """The four swappable stages. Replacing a stub happens here and nowhere else.

    Today every field holds a synthetic stub from
    :func:`~flux_klein_host.stages.synthetic_stages`. Each one is replaced by
    the driver named in its docstring once that driver is silicon-confirmed.
    """

    tokenizer: PromptTokenizer
    """Host. V2 -- ``klein_tokenize.py``."""

    text_encoder: TextEncoder
    """Device. V3 -- the Qwen3 encoder driver (``hvx-encoder-layer``)."""

    block_stack: DiTBlockStack
    """Device. V4 -- the DiT block driver (``hvx-block-driver``) over the
    batched matmul (``hvx-batched-matmul``)."""

    vae_decoder: VaeDecoder
    """Device. V5 -- the VAE decoder driver (``hvx-vae-block``)."""


@dataclass
class PipelineResult:
    """Everything one run produced, stage by stage."""

    request: PipelineRequest
    geometry: LatentGeometry
    tokens: PromptTokens
    text: TextConditioning
    denoise: DenoiseResult
    vae_input: VaeDecodeInput
    rgb: npt.NDArray[np.float32]
    """``(B, 3, height, width)`` decoder output in ``[-1, 1]``."""

    image: npt.NDArray[np.uint8]
    """``(B, height, width, 3)`` quantized image."""

    stage_seconds: dict[str, float] = field(default_factory=dict)
    png_paths: list[Path] = field(default_factory=list)

    @property
    def latent(self) -> npt.NDArray[np.float32]:
        """The final packed latent, ``(B, image_seq_len, 128)``."""
        return self.denoise.latent

    def stage_shapes(self) -> dict[str, tuple[int, ...]]:
        """The shape at every stage boundary, in pipeline order.

        This is the thing an integration run is really asserting: that the
        tokenizer, the encoder, the loop, the decoder and the image agree
        about what they are handing each other. Both sides of each of the two
        layout changes appear -- ``denoise.noise`` to ``denoise.latent_init``
        is the 2x2 pack, ``denoise.out`` to ``vae.input`` is the unpack.
        """
        return {
            "tokenize.tokens": self.tokens.tokens.shape,
            "tokenize.attention_bias": self.tokens.attention_bias.shape,
            "encoder.out": self.text.prompt_embeds.shape,
            "encoder.text_ids": self.text.text_ids.shape,
            "denoise.noise": self.denoise.initial_latent.shape,
            "denoise.latent_init": (
                *self.denoise.initial_latent.shape[:1],
                *self.geometry.packed_shape,
            ),
            "denoise.out": self.denoise.latent.shape,
            "vae.input": self.vae_input.latent.shape,
            "vae.rgb": self.rgb.shape,
            "image.rgb_u8": self.image.shape,
        }


def default_stages(
    config: KleinConfig = KLEIN_4B,
    geometry: LatentGeometry | None = None,
    *,
    seed: int = 0,
) -> PipelineStages:
    """Builds the all-synthetic stage set.

    Args:
        config: Model geometry.
        geometry: Resolved latent geometry; defaults to 512^2.
        seed: Base seed for the stubs' fixed projections.

    Returns:
        A :class:`PipelineStages` holding one stub per stage.
    """
    geometry = geometry or resolve_geometry(config)
    tokenizer, encoder, block_stack, decoder = synthetic_stages(
        config, geometry, seed=seed
    )
    return PipelineStages(
        tokenizer=tokenizer,
        text_encoder=encoder,
        block_stack=block_stack,
        vae_decoder=decoder,
    )


def run_pipeline(
    request: PipelineRequest,
    stages: PipelineStages,
    weights: PipelineWeights,
    *,
    config: KleinConfig = KLEIN_4B,
    scheduler: FlowMatchEulerScheduler | None = None,
    checkpoints: StageCheckpoints | None = None,
) -> PipelineResult:
    """Runs one prompt end to end.

    Args:
        request: The prompt and its sampling parameters.
        stages: The four stage implementations.
        weights: Host-side weights -- the timestep MLP and the latent
            batch-norm statistics.
        config: Model geometry.
        scheduler: Scheduler; defaults to Klein's public config with MAX's
            ``use_empirical_mu`` override.
        checkpoints: Per-stage dump sink for the numerical oracle, or ``None``.

    Returns:
        The :class:`PipelineResult`.

    Raises:
        ValueError: If any stage returns a shape the next one cannot consume.
    """
    geometry = resolve_geometry(config, request.height, request.width)
    timings: dict[str, float] = {}

    def _dump(name: str, array: npt.NDArray[Any]) -> None:
        if checkpoints is not None:
            checkpoints.record(name, array)

    start = time.perf_counter()
    tokens = stages.tokenizer(request.prompt)
    tokens.validate(config)
    timings["tokenize"] = time.perf_counter() - start
    _dump("tokenize.tokens", tokens.tokens)
    _dump("tokenize.attention_mask", tokens.attention_mask)
    _dump("tokenize.attention_bias", tokens.attention_bias)

    start = time.perf_counter()
    text = stages.text_encoder(
        TextEncodeRequest(tokens=tokens, batch_size=request.batch_size)
    )
    text.validate(config)
    # `validate` only checks the conditioning against itself. A driver that
    # ignored the requested batch would otherwise broadcast its way through
    # the loop and surface as a `temb` shape error three frames deeper.
    if text.prompt_embeds.shape[0] != request.batch_size:
        raise ValueError(
            f"text encoder returned batch {text.prompt_embeds.shape[0]}, "
            f"expected {request.batch_size}"
        )
    timings["text_encoder"] = time.perf_counter() - start
    _dump("encoder.out", text.prompt_embeds)
    _dump("encoder.text_ids", text.text_ids)

    start = time.perf_counter()
    result = denoise(
        stages.block_stack,
        text,
        geometry,
        weights.timestep,
        config=config,
        scheduler=scheduler,
        num_steps=request.num_steps,
        seed=request.seed,
        batch_size=request.batch_size,
        modulation_weights=weights.modulation,
        on_step=None if checkpoints is None else checkpoints.on_step,
    )
    timings["denoise"] = time.perf_counter() - start
    # The noise is dumped unpacked and again packed: the pack is the one host
    # transform between them, so a dump of only one side cannot localize a
    # channel-order bug to the pack rather than to the noise.
    _dump("denoise.noise", result.initial_latent)
    _dump("denoise.latent_init", pack(result.initial_latent))
    _dump("denoise.out", result.latent)

    vae_input = vae_decode_input(result.latent, geometry, weights.bn_stats)
    vae_input.validate(config)
    _dump("vae.input", vae_input.latent)

    start = time.perf_counter()
    rgb = stages.vae_decoder(vae_input)
    timings["vae_decode"] = time.perf_counter() - start
    expected_rgb = (
        request.batch_size,
        3,
        geometry.latent_height * config.vae_scale_factor,
        geometry.latent_width * config.vae_scale_factor,
    )
    if rgb.shape != expected_rgb:
        raise ValueError(
            f"VAE decoder returned {rgb.shape}, expected {expected_rgb}"
        )
    _dump("vae.rgb", rgb)

    image = to_rgb_uint8(rgb)
    _dump("image.rgb_u8", image)

    return PipelineResult(
        request=request,
        geometry=geometry,
        tokens=tokens,
        text=text,
        denoise=result,
        vae_input=vae_input,
        rgb=rgb,
        image=image,
        stage_seconds=timings,
    )


def _build_tokenizer(
    spec: str | None, artifact: Path | None, config: KleinConfig
) -> PromptTokenizer | None:
    """Resolves the requested tokenizer, or ``None`` for the synthetic stub."""
    if artifact is not None:
        return ArtifactTokenizer(artifact, config)
    if spec is not None:
        return KleinTokenizer(config, spec=spec)
    return None


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the FLUX.2-Klein-4B host pipeline end to end. Device stages "
            "are synthetic stubs unless a driver is wired in."
        )
    )
    parser.add_argument(
        "--prompt", default="a red bicycle leaning on a white wall"
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("klein-run"),
        help="Where the PNG is written.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help=(
            "Dump every stage boundary here in the reference oracle's format, "
            "for `flux_oracle.py compare --got`."
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "A converted V1 manifest (`<name>.json`). Without it the timestep "
            "MLP is random and the latent batch-norm is the identity."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Tokenizer spec for the real V2 path: a local tokenizer.json, a "
            "directory, or a HuggingFace repo id."
        ),
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="A directory the V2 CLI wrote, read the way the driver reads it.",
    )
    parser.add_argument(
        "--backend",
        choices=("synthetic", "device"),
        default="synthetic",
        help=(
            "Which DiT block stack runs each denoise step. `device` marshals "
            "the step to the Windows NPU driver over the file bridge."
        ),
    )
    parser.add_argument(
        "--driver",
        type=Path,
        default=DEFAULT_STAGE_DIR / "npu_block_stack_dispatch.exe",
        help="The block-stack driver `--backend device` execs.",
    )
    parser.add_argument(
        "--encoder-driver",
        type=Path,
        default=None,
        help=(
            "The text-encoder driver. Omitted leaves the encoder synthetic, so "
            "one seam can be moved onto the device at a time."
        ),
    )
    parser.add_argument(
        "--vae-driver",
        type=Path,
        default=None,
        help=("The VAE decoder driver. Omitted leaves the decoder synthetic."),
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_STAGE_DIR,
        help=(
            "Where the bridge stages each step. Must be Windows-visible, so a "
            "`/mnt/...` path, not a Linux-side directory."
        ),
    )
    parser.add_argument(
        "--driver-weights",
        type=Path,
        default=None,
        help=(
            "The manifest the DEVICE driver stages from; defaults to "
            "`--weights`. Separate because the host reads its two tensors "
            "wherever the artifact sits, while the driver stages gigabytes and "
            "wants a copy on the Windows drive rather than over the 9P share."
        ),
    )
    parser.add_argument(
        "--no-modulation",
        action="store_true",
        help=(
            "Do not marshal the adaLN modulation; let the driver supply its "
            "own synthetic set (`modulation_source=synthetic`)."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Bridge mechanism test: the driver returns before it opens the "
            "device, so every step's velocity is a ZERO PLACEHOLDER and the "
            "image is not a render. Needs no NPU."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    args = _parse_args(argv)
    config = KLEIN_4B
    geometry = resolve_geometry(config, args.height, args.width)

    stages = default_stages(config, geometry, seed=args.seed or 0)
    tokenizer = _build_tokenizer(args.tokenizer, args.tokens, config)
    if tokenizer is not None:
        stages = replace(stages, tokenizer=tokenizer)
    if args.backend == "device":
        # Each seam is switched on independently: a driver that is not built
        # yet leaves its stage synthetic rather than failing the whole run, so
        # the pipeline can be moved onto the device one seam at a time.
        shared = {
            "weights": args.driver_weights or args.weights,
            "scratch_root": args.scratch_root,
            "plan_only": args.plan_only,
        }
        stages = replace(
            stages,
            block_stack=DeviceBlockStack(args.driver, config, **shared),
        )
        if args.encoder_driver is not None:
            stages = replace(
                stages,
                text_encoder=DeviceTextEncoder(
                    args.encoder_driver, config, **shared
                ),
            )
        if args.vae_driver is not None:
            stages = replace(
                stages,
                vae_decoder=DeviceVaeDecoder(args.vae_driver, config, **shared),
            )
    elif args.plan_only:
        raise SystemExit(
            "--plan-only only means something for --backend device"
        )

    weights = (
        PipelineWeights.from_manifest(args.weights, config)
        if args.weights is not None
        else PipelineWeights.synthetic(config)
    )
    # Only a device run marshals modulation, and the heads are ~640 MiB
    # widened, so a synthetic run does not pay to read them.
    if args.backend == "device" and not args.no_modulation:
        weights.load_modulation(config, seed=args.seed or 0)

    request = PipelineRequest(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        seed=args.seed,
    )

    with weights, maybe_checkpoints(args.checkpoints) as checkpoints:
        result = run_pipeline(
            request, stages, weights, config=config, checkpoints=checkpoints
        )
        recorded = [] if checkpoints is None else checkpoints.names

    png = save_png(Path(args.out_dir) / "image.png", result.image)
    result.png_paths.append(png)

    print(f"prompt        {request.prompt!r}")
    print(f"geometry      {args.height}x{args.width}, {args.steps} steps")
    print(f"weights       {weights.source}")
    print(f"backend       {args.backend}")
    seams = [
        ("encoder", stages.text_encoder),
        ("blocks", stages.block_stack),
        ("vae", stages.vae_decoder),
    ]
    device_seams = [
        (name, seam.bridge) for name, seam in seams if hasattr(seam, "bridge")
    ]
    for name, bridge in device_seams:
        response = bridge.last_response
        # `modulation_source` only exists on the seam that has modulation, so
        # printing it unconditionally would invent a field for the other two.
        extra = (
            ""
            if response is None or response.modulation_source == "unknown"
            else f", modulation={response.modulation_source}"
        )
        print(
            f"{name:<14}{bridge.calls} crossings, "
            f"{bridge.last_seconds:.2f}s each, "
            f"{bridge.executable.name}, source="
            f"{'?' if response is None else response.output_source}{extra}"
        )
    if device_seams and args.plan_only:
        print(
            "              PLAN-ONLY: the image is a placeholder, not a render"
        )
    print(f"mu            {result.denoise.schedule.mu:.6f}")
    print(f"sigmas        {np.round(result.denoise.schedule.sigmas, 6)}")
    print("stage shapes")
    for name, shape in result.stage_shapes().items():
        print(f"  {name:<26} {shape}")
    print("stage seconds")
    for name, seconds in result.stage_seconds.items():
        print(f"  {name:<26} {seconds:.3f}")
    if recorded:
        print(f"checkpoints   {len(recorded)} stages -> {args.checkpoints}")
    print(f"image         {png}")
    return 0


__all__ = [
    "PipelineRequest",
    "PipelineResult",
    "PipelineStages",
    "default_stages",
    "main",
    "run_pipeline",
]


if __name__ == "__main__":
    raise SystemExit(main())
