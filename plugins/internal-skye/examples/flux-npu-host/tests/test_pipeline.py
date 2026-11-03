"""End-to-end validation of the pipeline driver on synthetic weights.

What these tests prove is the **orchestration**, not the model: that the
tokenizer, the encoder, the denoise loop, the decoder and the image tail agree
about the shapes they hand each other, that every stage's input actually
reaches the stage that consumes it, and that the per-stage checkpoints land in
the format the reference oracle reads.

They cannot prove numerics, because the device stages are synthetic stubs.
That is the point of the split: when a real driver replaces a stub, these tests
keep passing unchanged and the *numeric* gate is the oracle comparison, which
these tests only prove is wired.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zlib
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import numpy.typing as npt
import pytest
from flux_klein_host.checkpoints import StageCheckpoints
from flux_klein_host.config import KLEIN_4B, KleinConfig, resolve_geometry
from flux_klein_host.image import encode_png, save_png, to_rgb_uint8
from flux_klein_host.interfaces import (
    DiTBlockStackRequest,
    LatentBatchNormStats,
    StepConditioning,
    TextConditioning,
    VaeDecodeInput,
    vae_decode_input,
)
from flux_klein_host.latents import (
    batch_norm_denormalize,
    latent_image_ids,
    pack,
    unpack,
)
from flux_klein_host.pipeline import (
    PipelineRequest,
    PipelineResult,
    default_stages,
    main,
    run_pipeline,
)
from flux_klein_host.stages import (
    PromptTokens,
    ReferenceTextEncoder,
    SyntheticBlockStack,
    SyntheticTextEncoder,
    SyntheticTokenizer,
    TextEncodeRequest,
)
from flux_klein_host.tokenizer import (
    ArtifactTokenizer,
    KleinTokenizer,
    load_tokenizer_module,
)
from flux_klein_host.weights import (
    PipelineWeights,
    _repo_root,
    load_converter_module,
)

PROMPT = "a red bicycle leaning on a white wall"

#: MINI dims that keep every relation `KleinConfig` asserts -- the RoPE axes
#: sum to the head dim, `in_channels == 4 * latent channels`, and the three
#: stacked encoder states are exactly `joint_attention_dim` wide -- at a width
#: the fp32 reference oracle can run on CPU in seconds. The 96-wide timestep
#: projection is deliberately *not* a multiple of 64, so the V1 artifact
#: K-pads it and the loader has to trim the padding back off.
MINI = KleinConfig(
    num_attention_heads=2,
    joint_attention_dim=384,
    text_hidden_size=128,
    text_seq_len=16,
    text_hidden_state_layers=(1, 2, 3),
    timestep_guidance_channels=96,
)


def run_default(
    prompt: str = PROMPT,
    *,
    seed: int | None = 0,
    steps: int = 4,
    height: int = 512,
    width: int = 512,
    config: KleinConfig = KLEIN_4B,
    checkpoints: StageCheckpoints | None = None,
) -> PipelineResult:
    """Runs the all-synthetic pipeline once."""
    geometry = resolve_geometry(config, height, width)
    return run_pipeline(
        PipelineRequest(
            prompt=prompt,
            height=height,
            width=width,
            num_steps=steps,
            seed=seed,
        ),
        default_stages(config, geometry),
        PipelineWeights.synthetic(config),
        config=config,
        checkpoints=checkpoints,
    )


# --------------------------------------------------------------------------
# Stage-boundary shapes -- the contract the real drivers inherit
# --------------------------------------------------------------------------


def test_end_to_end_stage_shapes_at_512() -> None:
    result = run_default()
    assert result.stage_shapes() == {
        "tokenize.tokens": (512,),
        "tokenize.attention_bias": (1, 1, 512, 512),
        "encoder.out": (1, 512, 7680),
        "encoder.text_ids": (1, 512, 4),
        "denoise.noise": (1, 32, 64, 64),
        "denoise.latent_init": (1, 1024, 128),
        "denoise.out": (1, 1024, 128),
        "vae.input": (1, 32, 64, 64),
        "vae.rgb": (1, 3, 512, 512),
        "image.rgb_u8": (1, 512, 512, 3),
    }


def test_stage_dtypes() -> None:
    result = run_default()
    assert result.tokens.tokens.dtype == np.int32
    assert result.tokens.attention_mask.dtype == np.uint8
    assert result.tokens.attention_bias.dtype == np.float32
    assert result.text.prompt_embeds.dtype == np.float32
    assert result.text.text_ids.dtype == np.int64
    assert result.latent.dtype == np.float32
    assert result.vae_input.latent.dtype == np.float32
    assert result.rgb.dtype == np.float32
    assert result.image.dtype == np.uint8


def test_joint_sequence_is_the_block_drivers_target() -> None:
    result = run_default()
    assert result.geometry.image_seq_len == 1024
    assert result.geometry.joint_seq_len == 1536


def test_every_stage_output_is_finite() -> None:
    result = run_default()
    for name, array in {
        "encoder.out": result.text.prompt_embeds,
        "denoise.out": result.latent,
        "vae.input": result.vae_input.latent,
        "vae.rgb": result.rgb,
    }.items():
        assert np.isfinite(array).all(), f"{name} is not finite"


def test_pipeline_runs_at_other_resolutions_and_step_counts() -> None:
    for height, width, steps, tokens in (
        (256, 256, 1, 256),
        (512, 256, 8, 512),
    ):
        result = run_default(height=height, width=width, steps=steps)
        assert result.geometry.image_seq_len == tokens
        assert result.latent.shape == (1, tokens, 128)
        assert result.rgb.shape == (1, 3, height, width)
        assert len(result.denoise.schedule.timesteps) == steps


def test_denoise_uses_the_empirical_mu_ladder() -> None:
    schedule = run_default().denoise.schedule
    assert schedule.mu == pytest.approx(2.030690, abs=1e-6)
    np.testing.assert_allclose(
        schedule.sigmas,
        [1.0, 0.958085, 0.883982, 0.717497, 0.0],
        atol=1e-6,
    )


def test_a_decoder_returning_the_wrong_shape_is_rejected() -> None:
    config = KLEIN_4B
    geometry = resolve_geometry(config)
    stages = default_stages(config, geometry)

    class WrongShapeDecoder:
        def __call__(self, request: VaeDecodeInput) -> npt.NDArray[np.float32]:
            return np.zeros((1, 3, 64, 64), dtype=np.float32)

    with pytest.raises(ValueError, match="VAE decoder returned"):
        run_pipeline(
            PipelineRequest(prompt=PROMPT),
            replace(stages, vae_decoder=WrongShapeDecoder()),
            PipelineWeights.synthetic(config),
            config=config,
        )


def test_an_encoder_ignoring_the_batch_size_is_rejected() -> None:
    """An encoder pinned to batch 1 must not broadcast into a batch-2 run."""
    config = KLEIN_4B
    inner = default_stages(config, resolve_geometry(config))

    class SingleBatchEncoder:
        def __call__(self, request: TextEncodeRequest) -> TextConditioning:
            return inner.text_encoder(replace(request, batch_size=1))

    with pytest.raises(ValueError, match="text encoder returned batch"):
        run_pipeline(
            PipelineRequest(prompt=PROMPT, batch_size=2),
            replace(inner, text_encoder=SingleBatchEncoder()),
            PipelineWeights.synthetic(config),
            config=config,
        )


def test_a_batch_produces_one_distinct_image_per_entry() -> None:
    config = KLEIN_4B
    geometry = resolve_geometry(config, 128, 128)
    result = run_pipeline(
        PipelineRequest(
            prompt=PROMPT, height=128, width=128, num_steps=2, batch_size=2
        ),
        default_stages(config, geometry),
        PipelineWeights.synthetic(config),
        config=config,
    )
    assert result.image.shape == (2, 128, 128, 3)
    assert result.text.prompt_embeds.shape == (2, 512, 7680)
    # Same prompt, different noise -- the batch axis has to carry independent
    # samples, not a broadcast of one.
    assert not np.array_equal(result.image[0], result.image[1])


# --------------------------------------------------------------------------
# The image
# --------------------------------------------------------------------------


def test_image_is_image_shaped_and_not_degenerate() -> None:
    image = run_default().image
    assert image.shape == (1, 512, 512, 3)
    assert image.dtype == np.uint8
    # A stuck stage -- a decoder returning zeros, a latent that collapsed --
    # shows up here as a flat image long before it shows up in a shape.
    assert len(np.unique(image)) > 32
    assert image.std() > 4.0


def test_quantization_maps_the_decoder_range_onto_0_255() -> None:
    ramp = np.linspace(-1.0, 1.0, 12, dtype=np.float32)
    rgb = np.broadcast_to(ramp[None, None, None, :], (1, 3, 1, 12))
    quantized = to_rgb_uint8(np.ascontiguousarray(rgb))
    assert quantized[0, 0, 0, 0] == 0
    assert quantized[0, 0, -1, 0] == 255
    # Values outside [-1, 1] clamp rather than wrapping around.
    out_of_range = np.full((1, 3, 1, 1), 4.0, dtype=np.float32)
    assert to_rgb_uint8(out_of_range).max() == 255
    assert to_rgb_uint8(-out_of_range).min() == 0


def test_png_is_a_well_formed_file(tmp_path: Path) -> None:
    image = run_default().image
    path = save_png(tmp_path / "image.png", image)
    blob = path.read_bytes()

    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    offset, chunks = 8, []
    while offset < len(blob):
        length = struct.unpack(">I", blob[offset : offset + 4])[0]
        tag = blob[offset + 4 : offset + 8]
        body = blob[offset + 8 : offset + 8 + length]
        crc = struct.unpack(
            ">I", blob[offset + 8 + length : offset + 12 + length]
        )[0]
        assert crc == zlib.crc32(tag + body) & 0xFFFF_FFFF, (
            f"bad CRC on {tag!r}"
        )
        chunks.append(tag)
        offset += 12 + length
    assert chunks == [b"IHDR", b"IDAT", b"IEND"]

    width, height, depth, colour = struct.unpack(">IIBB", blob[16:26])
    assert (width, height, depth, colour) == (512, 512, 8, 2)


def test_png_decodes_back_to_the_same_pixels(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")
    image = run_default().image
    path = save_png(tmp_path / "image.png", image)
    with pillow.open(path) as decoded:
        np.testing.assert_array_equal(np.asarray(decoded), image[0])


def test_save_png_refuses_a_real_batch(tmp_path: Path) -> None:
    batch = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="one image"):
        save_png(tmp_path / "image.png", batch)


def test_encode_png_rejects_a_non_rgb_array() -> None:
    with pytest.raises(ValueError, match=r"\(H, W, 3\)"):
        encode_png(np.zeros((4, 4), dtype=np.uint8))


# --------------------------------------------------------------------------
# Conditioning actually reaches the device call
# --------------------------------------------------------------------------


def test_a_different_seed_gives_a_different_image() -> None:
    first, second = run_default(seed=0), run_default(seed=1)
    assert first.image.shape == second.image.shape
    assert not np.array_equal(first.image, second.image)
    assert len(np.unique(second.image)) > 32


def test_the_same_seed_and_prompt_reproduce_exactly() -> None:
    np.testing.assert_array_equal(run_default().image, run_default().image)


def test_a_different_prompt_gives_a_different_image() -> None:
    other = run_default("a blue tractor in a wheat field")
    assert not np.array_equal(run_default().image, other.image)


def test_reusing_one_stage_set_across_prompts_does_not_bleed() -> None:
    """One `PipelineStages`, many prompts -- the design's intended usage.

    The block stack caches its pooled text summary across the four steps of a
    run. Keyed on `id(prompt_embeds)` that cache aliased: once the previous
    prompt's array was collected, numpy handed the same address to the next
    one and the second prompt silently rendered the first prompt's image, with
    no error. Discarding each result before the next call is what frees the
    array, so the loop below is the shape that reproduces it.
    """
    config = KLEIN_4B
    stages = default_stages(config, resolve_geometry(config, 128, 128))
    prompts = [
        "a red bicycle leaning on a white wall",
        "a blue tractor in a wheat field",
        "a lighthouse at dusk under heavy cloud",
    ]

    digests = []
    for prompt in prompts:
        result = run_pipeline(
            PipelineRequest(prompt=prompt, height=128, width=128, num_steps=2),
            stages,
            PipelineWeights.synthetic(config),
            config=config,
        )
        digests.append(zlib.crc32(result.image.tobytes()))
        del result

    assert len(set(digests)) == len(prompts), (
        "two prompts rendered the same image through a shared stage set"
    )


def test_the_synthetic_tokenizer_handles_a_prompt_longer_than_255_words() -> (
    None
):
    """The valid span runs to `text_seq_len`, which does not fit in a byte."""
    tokenizer = SyntheticTokenizer(KLEIN_4B)

    encoded = tokenizer(" ".join(f"word{i}" for i in range(400)))
    assert encoded.num_valid_tokens == 402
    # Positions past 255 are still distinct, not folded onto earlier ones.
    assert len(np.unique(encoded.tokens[250:300])) == 50

    capped = tokenizer(" ".join(f"word{i}" for i in range(600)))
    assert capped.num_valid_tokens == KLEIN_4B.text_seq_len
    assert int(capped.attention_mask.sum()) == KLEIN_4B.text_seq_len
    assert (capped.tokens > 0).all()


def test_the_block_stack_consumes_every_input_it_is_given() -> None:
    """Perturbing any one input has to move the velocity.

    A stub that quietly ignored ``temb`` or ``image_ids`` would still produce
    a plausible image and still pass every shape assertion, so the wiring has
    to be checked by perturbation rather than by inspection.
    """
    config, geometry = KLEIN_4B, resolve_geometry(KLEIN_4B)
    stages = default_stages(config, geometry)
    result = run_default()

    latent = pack(result.denoise.initial_latent)

    base_request = DiTBlockStackRequest(
        latent=latent,
        image_ids=latent_image_ids(geometry),
        text=result.text,
        conditioning=StepConditioning(
            step_index=0,
            timestep=1.0,
            temb=result.denoise.schedule.temb[0],
        ),
        geometry=geometry,
    )
    baseline = stages.block_stack(base_request)
    assert not np.shares_memory(baseline, latent)

    perturbations = {
        "latent": replace(base_request, latent=latent + np.float32(1.0)),
        "temb": replace(
            base_request,
            conditioning=StepConditioning(
                step_index=0,
                timestep=1.0,
                temb=result.denoise.schedule.temb[1],
            ),
        ),
        "prompt_embeds": replace(
            base_request,
            text=replace(
                result.text,
                prompt_embeds=result.text.prompt_embeds
                + np.float32(1.0).astype(np.float32),
            ),
        ),
        "image_ids": replace(
            base_request,
            image_ids=base_request.image_ids[:, ::-1, :].copy(),
        ),
    }
    for name, request in perturbations.items():
        moved = stages.block_stack(request)
        assert not np.array_equal(baseline, moved), (
            f"the block stack ignores {name}"
        )


def test_the_text_encoder_carries_the_padding_bias_through() -> None:
    result = run_default()
    assert result.text.attention_bias is not None
    np.testing.assert_array_equal(
        result.text.attention_bias[0], result.tokens.attention_bias[0]
    )
    valid = result.tokens.num_valid_tokens
    # Row 0 may attend to key 0 only; nothing may attend past the valid span.
    assert result.text.attention_bias[0, 0, 0, 0] == 0.0
    assert (result.text.attention_bias[0, 0, :, valid:] < 0).all()


def test_the_encoder_stacks_three_hidden_states_in_layer_order() -> None:
    result = run_default()
    hidden = KLEIN_4B.text_hidden_size
    embeds = result.text.prompt_embeds
    assert embeds.shape[-1] == 3 * hidden
    slices = [embeds[..., i * hidden : (i + 1) * hidden] for i in range(3)]
    for i, first in enumerate(slices):
        for j, second in enumerate(slices[i + 1 :], start=i + 1):
            assert not np.array_equal(first, second), (
                f"selected layers {i} and {j} are identical, so the "
                "concatenation is not carrying three distinct states"
            )


# --------------------------------------------------------------------------
# The VAE seam's load-bearing ordering
# --------------------------------------------------------------------------


def test_denorm_runs_on_the_packed_latent_before_the_unpack() -> None:
    """Swapping the order permutes statistics against their channels.

    The scaffold's seam already tests this in isolation; repeating it at the
    pipeline level is what catches a driver that "helpfully" unpacks first.
    """
    config = KLEIN_4B
    geometry = resolve_geometry(config)
    rng = np.random.RandomState(7)
    stats = LatentBatchNormStats(
        mean=rng.standard_normal(config.in_channels).astype(np.float32),
        var=(rng.random(config.in_channels) + 0.5).astype(np.float32),
        eps=config.vae_batch_norm_eps,
    )
    packed = rng.standard_normal((1, geometry.image_seq_len, 128)).astype(
        np.float32
    )

    correct = vae_decode_input(packed, geometry, stats).latent
    swapped = unpack(packed, geometry) * np.sqrt(
        stats.var.reshape(32, 4).mean(axis=1) + stats.eps
    ).astype(np.float32).reshape(1, 32, 1, 1)
    assert not np.allclose(correct, swapped)
    np.testing.assert_allclose(
        correct,
        unpack(
            batch_norm_denormalize(packed, stats.mean, stats.var, stats.eps),
            geometry,
        ),
    )


# --------------------------------------------------------------------------
# Per-stage checkpoints for the numerical oracle
# --------------------------------------------------------------------------

EXPECTED_STAGES = (
    "tokenize.tokens",
    "tokenize.attention_mask",
    "tokenize.attention_bias",
    "encoder.out",
    "encoder.text_ids",
    "denoise.noise",
    "denoise.latent_init",
    "denoise.out",
    "vae.input",
    "vae.rgb",
    "image.rgb_u8",
    *(
        f"denoise.step_{i:02d}.{part}"
        for i in range(4)
        for part in ("velocity", "latent")
    ),
)


def test_every_stage_boundary_is_checkpointed(tmp_path: Path) -> None:
    checkpoints = StageCheckpoints(tmp_path)
    result = run_default(checkpoints=checkpoints)
    checkpoints.write_manifest({"prompt": result.request.prompt})

    assert set(checkpoints.names) == set(EXPECTED_STAGES)
    for name in EXPECTED_STAGES:
        assert (tmp_path / f"{name}.npy").is_file()


def test_the_checkpoint_manifest_is_in_the_oracles_format(
    tmp_path: Path,
) -> None:
    checkpoints = StageCheckpoints(tmp_path)
    run_default(checkpoints=checkpoints)
    manifest = json.loads(
        (checkpoints.write_manifest()).read_text(encoding="utf-8")
    )

    assert manifest["format"] == "skye-npu-reference"
    assert manifest["version"] == 1
    for entry in manifest["stages"].values():
        stored = np.load(tmp_path / entry["path"])
        assert list(stored.shape) == entry["shape"]
        assert str(stored.dtype) == entry["dtype"]
        assert entry["abs_tol"] > 0 and entry["rel_tol"] > 0


def test_checkpointed_token_ids_keep_their_integer_dtype(
    tmp_path: Path,
) -> None:
    checkpoints = StageCheckpoints(tmp_path)
    run_default(checkpoints=checkpoints)
    assert np.load(tmp_path / "tokenize.tokens.npy").dtype == np.int32
    assert np.load(tmp_path / "image.rgb_u8.npy").dtype == np.uint8
    assert np.load(tmp_path / "encoder.out.npy").dtype == np.float32


def test_the_step_hook_records_the_loop_it_observed(tmp_path: Path) -> None:
    checkpoints = StageCheckpoints(tmp_path)
    result = run_default(checkpoints=checkpoints)

    latent = np.load(tmp_path / "denoise.latent_init.npy")
    np.testing.assert_array_equal(latent, pack(result.denoise.initial_latent))
    for i, dt in enumerate(result.denoise.schedule.dts):
        velocity = np.load(tmp_path / f"denoise.step_{i:02d}.velocity.npy")
        stepped = np.load(tmp_path / f"denoise.step_{i:02d}.latent.npy")
        np.testing.assert_allclose(
            stepped, latent + np.float32(dt) * velocity, rtol=1e-6, atol=1e-6
        )
        latent = stepped
    np.testing.assert_array_equal(latent, result.latent)


def test_the_dumps_compare_clean_against_the_oracles_comparator(
    tmp_path: Path,
) -> None:
    """The dump directory is readable by ``flux_oracle.py compare``.

    Compared against itself every stage must pass; corrupting one stage must
    make exactly that stage fail. Without the second half the first proves
    only that the comparator was pointed somewhere.
    """
    oracle = _load_oracle()
    reference = tmp_path / "ref"
    checkpoints = StageCheckpoints(reference)
    run_default(checkpoints=checkpoints)
    checkpoints.write_manifest()

    float_stages = {
        name
        for name, entry in json.loads((reference / "stages.json").read_text())[
            "stages"
        ].items()
        if entry["dtype"] == "float32"
    }

    reports = oracle.compare_dirs(reference, reference, only=float_stages)
    assert len(reports) == len(float_stages)
    assert all(report.passed for report in reports), [
        report.line() for report in reports if not report.passed
    ]

    corrupted = tmp_path / "got"
    corrupted.mkdir()
    for name in float_stages:
        values = np.load(reference / f"{name}.npy")
        if name == "encoder.out":
            values = values + np.float32(10.0)
        np.save(corrupted / f"{name}.npy", values)

    after = {
        report.name: report.passed
        for report in oracle.compare_dirs(
            reference, corrupted, only=float_stages
        )
    }
    assert after["encoder.out"] is False
    assert all(
        passed for name, passed in after.items() if name != "encoder.out"
    )


# --------------------------------------------------------------------------
# The V1 weight artifact
# --------------------------------------------------------------------------


def _bf16(
    shape: tuple[int, ...], seed: int, low: float = -1.0, high: float = 1.0
) -> np.ndarray:
    """bf16 values as the u16 bit pattern, truncated so widening is exact."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(low, high, size=shape).astype(np.float32)
    return (values.view(np.uint32) >> 16).astype(np.uint16)


def _write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {}
    blobs: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        raw = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for blob in blobs:
            out.write(blob)


def _widen(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _mini_checkpoint(root: Path) -> dict[str, np.ndarray]:
    """Writes the host-side slice of a klein snapshot at MINI dims."""
    dit = {
        "time_guidance_embed.timestep_embedder.linear_1.weight": _bf16(
            (MINI.dim, MINI.timestep_guidance_channels), 1
        ),
        "time_guidance_embed.timestep_embedder.linear_2.weight": _bf16(
            (MINI.dim, MINI.dim), 2
        ),
    }
    vae = {
        "bn.running_mean": _bf16((MINI.in_channels,), 3),
        "bn.running_var": _bf16((MINI.in_channels,), 4, low=0.25, high=4.0),
    }
    _write_safetensors(
        root / "transformer" / "diffusion_pytorch_model.safetensors", dit
    )
    _write_safetensors(
        root / "vae" / "diffusion_pytorch_model.safetensors", vae
    )
    return {
        **{f"transformer.{k}": v for k, v in dit.items()},
        **{f"vae.{k}": v for k, v in vae.items()},
    }


def _load_converter() -> ModuleType:
    return load_converter_module()


def _load_oracle() -> ModuleType:
    """Imports ``flux_oracle`` from the sibling hexagon-ondevice example."""
    root = _repo_root(Path(__file__).resolve())
    assert root is not None
    path = (
        root / "plugins/internal-skye/examples/hexagon-ondevice/flux_oracle.py"
    )
    if not path.is_file():
        pytest.skip("the reference oracle is not in this checkout")
    spec = importlib.util.spec_from_file_location("_test_flux_oracle", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_test_flux_oracle"] = module
    spec.loader.exec_module(module)
    return module


def test_host_weights_load_from_a_converted_v1_artifact(tmp_path: Path) -> None:
    converter = _load_converter()
    source = _mini_checkpoint(tmp_path / "checkpoint")
    manifest_path = converter.write_artifact(
        tmp_path / "checkpoint", tmp_path / "npu", name="mini"
    )[1]

    with PipelineWeights.from_manifest(manifest_path, MINI) as weights:
        assert weights.source == str(manifest_path)
        assert weights.timestep.in_channels == (MINI.timestep_guidance_channels)
        assert weights.timestep.out_dim == MINI.dim
        # The artifact K-pads 96 up to 128; the loader has to trim it back or
        # the sinusoidal projection is silently built at the wrong width.
        record = weights.artifact.record(
            "transformer.time_guidance_embed.timestep_embedder.linear_1.weight"
        )
        assert record["shape"] == [MINI.dim, 128]
        assert record["k_real"] == 96

        np.testing.assert_allclose(
            weights.timestep.linear_1,
            _widen(
                source[
                    "transformer.time_guidance_embed.timestep_embedder"
                    ".linear_1.weight"
                ]
            ).astype(np.float16),
            rtol=0,
            atol=0,
        )
        np.testing.assert_allclose(
            weights.bn_stats.mean,
            _widen(source["vae.bn.running_mean"]).astype(np.float16),
            rtol=0,
            atol=0,
        )
        assert weights.bn_stats.eps == MINI.vae_batch_norm_eps


def test_the_loaded_weights_drive_the_pipeline(tmp_path: Path) -> None:
    converter = _load_converter()
    _mini_checkpoint(tmp_path / "checkpoint")
    manifest_path = converter.write_artifact(
        tmp_path / "checkpoint", tmp_path / "npu", name="mini"
    )[1]

    geometry = resolve_geometry(MINI, 64, 64)
    with PipelineWeights.from_manifest(manifest_path, MINI) as weights:
        result = run_pipeline(
            PipelineRequest(prompt=PROMPT, height=64, width=64, num_steps=4),
            default_stages(MINI, geometry),
            weights,
            config=MINI,
        )
    assert result.image.shape == (1, 64, 64, 3)
    assert np.isfinite(result.rgb).all()
    # A real batch-norm is not the identity, so the denormalization has to
    # move the latent -- otherwise the artifact's statistics went unread.
    assert not np.allclose(
        result.vae_input.latent, unpack(result.latent, geometry)
    )


def test_a_negative_running_variance_is_refused(tmp_path: Path) -> None:
    converter = _load_converter()
    _mini_checkpoint(tmp_path / "checkpoint")
    _write_safetensors(
        tmp_path / "checkpoint" / "vae" / "diffusion_pytorch_model.safetensors",
        {
            "bn.running_mean": _bf16((MINI.in_channels,), 7),
            "bn.running_var": _bf16(
                (MINI.in_channels,), 8, low=-4.0, high=-1.0
            ),
        },
    )
    manifest_path = converter.write_artifact(
        tmp_path / "checkpoint", tmp_path / "npu", name="mini"
    )[1]
    with pytest.raises(ValueError, match="not a running variance"):
        PipelineWeights.from_manifest(manifest_path, MINI)


def test_a_wrong_width_batch_norm_is_refused(tmp_path: Path) -> None:
    converter = _load_converter()
    _mini_checkpoint(tmp_path / "checkpoint")
    # The decoder's own width, not the packed-channel width -- the mistake a
    # driver makes when it reaches for the first `bn.*` tensor it sees.
    _write_safetensors(
        tmp_path / "checkpoint" / "vae" / "diffusion_pytorch_model.safetensors",
        {
            "bn.running_mean": _bf16((64,), 5),
            "bn.running_var": _bf16((64,), 6),
        },
    )
    manifest_path = converter.write_artifact(
        tmp_path / "checkpoint", tmp_path / "npu", name="mini"
    )[1]
    with pytest.raises(ValueError, match="per \\*packed\\* channel"):
        PipelineWeights.from_manifest(manifest_path, MINI)


def test_synthetic_batch_norm_is_a_pure_unpack() -> None:
    weights = PipelineWeights.synthetic(KLEIN_4B)
    geometry = resolve_geometry(KLEIN_4B)
    packed = (
        np.random.RandomState(0)
        .standard_normal((1, geometry.image_seq_len, 128))
        .astype(np.float32)
    )

    np.testing.assert_allclose(
        vae_decode_input(packed, geometry, weights.bn_stats).latent,
        unpack(packed, geometry),
        rtol=1e-6,
        atol=1e-6,
    )


# --------------------------------------------------------------------------
# The V2 tokenizer, in both the in-process and the artifact route
# --------------------------------------------------------------------------


def _klein_tokenizer() -> KleinTokenizer:
    pytest.importorskip("tokenizers")
    module = load_tokenizer_module()
    cache = (
        Path.home()
        / ".cache/modular-npu/tokenizers"
        / module.KLEIN_REPO.replace("/", "--")
        / module.KLEIN_REVISION
        / module.KLEIN_SUBFOLDER
    )
    if not (cache / "tokenizer.json").is_file():
        pytest.skip("the klein tokenizer is not cached and the network is out")
    return KleinTokenizer(KLEIN_4B, spec=str(cache))


def test_the_real_tokenizer_drives_the_pipeline() -> None:
    tokenizer = _klein_tokenizer()
    config, geometry = KLEIN_4B, resolve_geometry(KLEIN_4B)
    result = run_pipeline(
        PipelineRequest(prompt=PROMPT),
        replace(default_stages(config, geometry), tokenizer=tokenizer),
        PipelineWeights.synthetic(config),
        config=config,
    )
    assert result.tokens.tokens.shape == (512,)
    assert 0 < result.tokens.num_valid_tokens < 512
    assert PROMPT in result.tokens.rendered
    assert result.image.shape == (1, 512, 512, 3)
    # Padding really is padding: every id past the valid span is the same.
    tail = result.tokens.tokens[result.tokens.num_valid_tokens :]
    assert len(np.unique(tail)) == 1


def test_the_artifact_route_matches_the_in_process_route(
    tmp_path: Path,
) -> None:
    tokenizer = _klein_tokenizer()
    module = load_tokenizer_module()
    encoded = module.tokenize(tokenizer.tokenizer, PROMPT)
    module.write_artifacts(tmp_path, encoded, tokenizer.source)

    direct = tokenizer(PROMPT)
    replayed = ArtifactTokenizer(tmp_path, KLEIN_4B)(PROMPT)
    np.testing.assert_array_equal(direct.tokens, replayed.tokens)
    np.testing.assert_array_equal(
        direct.attention_mask, replayed.attention_mask
    )
    np.testing.assert_array_equal(
        direct.attention_bias, replayed.attention_bias
    )
    assert direct.num_valid_tokens == replayed.num_valid_tokens


def test_an_artifact_refuses_a_prompt_it_does_not_hold(tmp_path: Path) -> None:
    tokenizer = _klein_tokenizer()
    module = load_tokenizer_module()
    module.write_artifacts(
        tmp_path, module.tokenize(tokenizer.tokenizer, PROMPT), tokenizer.source
    )
    with pytest.raises(ValueError, match="tokenizes"):
        ArtifactTokenizer(tmp_path, KLEIN_4B)("a different prompt entirely")


# --------------------------------------------------------------------------
# Replaying an oracle dump through seam 1
# --------------------------------------------------------------------------


def test_the_pipeline_runs_on_a_reference_encoder_dump(tmp_path: Path) -> None:
    """The one route where the encoder stage is not synthetic.

    Runs the fp32 oracle at MINI dims, dumps ``encoder.out``, and drives the
    whole pipeline off that dump. This is what makes a denoise run reproducible
    against the numerical oracle instead of merely self-consistent, and it is
    the shape of the swap a real encoder driver makes.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    oracle = _load_oracle()

    encoder_config = oracle.EncoderConfig(
        hidden_size=MINI.text_hidden_size,
        intermediate_size=2 * MINI.text_hidden_size,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=64,
        seq_len=MINI.text_seq_len,
        hidden_state_layers=MINI.text_hidden_state_layers,
    )
    tokens = oracle.default_tokens(encoder_config, num_valid=6)
    stages_dump = oracle.run_encoder_stages(
        encoder_config,
        tokens,
        oracle.build_attention_bias(6, encoder_config.seq_len),
    )
    oracle.dump_stages(stages_dump, tmp_path, {"preset": "mini"})
    assert stages_dump["encoder.out"].shape == (
        1,
        MINI.text_seq_len,
        MINI.joint_attention_dim,
    )

    geometry = resolve_geometry(MINI, 64, 64)
    result = run_pipeline(
        PipelineRequest(prompt=PROMPT, height=64, width=64, num_steps=4),
        replace(
            default_stages(MINI, geometry),
            text_encoder=ReferenceTextEncoder.from_stage_dump(tmp_path, MINI),
        ),
        PipelineWeights.synthetic(MINI),
        config=MINI,
    )
    np.testing.assert_array_equal(
        result.text.prompt_embeds, stages_dump["encoder.out"]
    )
    assert result.latent.shape == (1, 16, 128)
    assert result.image.shape == (1, 64, 64, 3)


def test_a_reference_dump_at_the_wrong_width_is_refused() -> None:
    with pytest.raises(ValueError, match="joint_attention_dim"):
        ReferenceTextEncoder(
            np.zeros((1, 512, 1024), dtype=np.float32), KLEIN_4B
        )


# --------------------------------------------------------------------------
# The command-line driver
# --------------------------------------------------------------------------


def test_main_writes_a_png_and_a_checkpoint_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--prompt",
            PROMPT,
            "--steps",
            "4",
            "--out-dir",
            str(tmp_path / "run"),
            "--checkpoints",
            str(tmp_path / "run" / "stages"),
        ]
    )
    assert status == 0
    assert (tmp_path / "run" / "image.png").is_file()
    assert (tmp_path / "run" / "stages" / "stages.json").is_file()

    out = capsys.readouterr().out
    assert "encoder.out                (1, 512, 7680)" in out
    assert "image.rgb_u8               (1, 512, 512, 3)" in out
    assert "19 stages" in out


def test_main_runs_without_checkpoints(tmp_path: Path) -> None:
    assert main(["--steps", "1", "--out-dir", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "image.png").is_file()
    assert not (tmp_path / "run" / "stages").exists()


def test_a_synthetic_block_stack_rejects_a_mismatched_request() -> None:
    config, geometry = KLEIN_4B, resolve_geometry(KLEIN_4B)
    stack = SyntheticBlockStack(config)

    with pytest.raises(ValueError, match="latent must be"):
        stack(
            DiTBlockStackRequest(
                latent=np.zeros((1, 16, 128), dtype=np.float32),
                image_ids=latent_image_ids(geometry),
                text=run_default().text,
                conditioning=StepConditioning(
                    step_index=0,
                    timestep=1.0,
                    temb=np.zeros((1, config.dim), dtype=np.float32),
                ),
                geometry=geometry,
            )
        )


def test_the_encoder_rejects_tokens_of_the_wrong_length() -> None:
    encoder = SyntheticTextEncoder(KLEIN_4B)
    short = PromptTokens(
        tokens=np.zeros(8, dtype=np.int32),
        attention_mask=np.ones(8, dtype=np.uint8),
        attention_bias=np.zeros((1, 1, 8, 8), dtype=np.float32),
        num_valid_tokens=8,
        rendered="",
    )
    with pytest.raises(ValueError, match="tokens must be"):
        encoder(TextEncodeRequest(tokens=short))
