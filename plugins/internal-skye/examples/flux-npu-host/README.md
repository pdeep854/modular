# FLUX.2-Klein host pipeline

Host side of running FLUX.2-Klein-4B on the on-device Hexagon NPU: the P6/P7
host slice, pure numpy, device-independent, no MAX or bazel dependency.

It owns the sampling math — the noise latent, the flow-matching schedule, the
timestep conditioning, the Euler update, the latent-to-image tail — and the
orchestration that chains all of it. It **defines but does not implement** the
seams where the device drivers plug in: the on-device text encoder, DiT block
stack, and VAE decoder are stubs here, and the drivers land separately and
satisfy them.

## Layout

```text
flux_klein_host/
  config.py           Klein-4B geometry from the public component configs
  latents.py          noise init, the 2x2 pack/unpack, RoPE position ids
  scheduler.py        flow-match Euler: sigmas, mu, the step update
  timestep_embed.py   sinusoidal projection + the bias-free MLP
  interfaces.py       what crosses the three device seams (the data contract)
  denoise.py          the loop skeleton
  stages.py           what calls across each seam, plus the synthetic stubs
  weights.py          the host's share of the V1 weight artifact
  tokenizer.py        the V2 tokenizer, in process or from its artifact
  checkpoints.py      per-stage dumps in the reference oracle's format
  image.py            quantization and a dependency-free PNG writer
  pipeline.py         the driver: prompt in, PNG out
tests/                validation against MAX, diffusers, and the oracle
```

## Running it

Needs numpy only. The diffusers cross-checks additionally need torch and
diffusers, the real-tokenizer tests need `tokenizers`, and all of them skip
cleanly without those.

```bash
cd plugins/internal-skye/examples/flux-npu-host
python -m pytest tests -q
```

## The pipeline driver

`python -m flux_klein_host` takes a prompt to a PNG. With no flags every device
stage is a synthetic stub and the weights are random, which is enough to
exercise the whole orchestration on a machine with no NPU:

```bash
python -m flux_klein_host --prompt "a red bicycle leaning on a white wall" \
    --out-dir /tmp/klein-run --checkpoints /tmp/klein-run/stages
```

Each real input is opted into separately, so a stage can be made real without
waiting for the others:

| Flag                        | Route it switches on                                      |
|-----------------------------|-----------------------------------------------------------|
| `--tokenizer <spec>`        | the real V2 tokenizer, in process                         |
| `--tokens <dir>`            | a V2 artifact directory, read the way the driver reads it |
| `--weights <manifest.json>` | the converted V1 artifact                                 |
| `--checkpoints <dir>`       | per-stage dumps for the numerical oracle                  |
| `--backend device`          | the real NPU block-stack driver, over the file bridge     |

### The device bridge

`--backend device` replaces the DiT stub with the real Windows ARM64 driver.
Nothing links this process to that one, so each denoise step is marshalled into
a scratch directory, the `.exe` is exec'd at it over Windows interop, and the
velocity is read back. `device_bridge.py` is that crossing, and it is
seam-agnostic — the encoder and VAE drivers plug into the same `DeviceBridge`.
The wire contract, the reuse recipe and the interop gotchas are in
[`../hexagon-ondevice/PIPELINE_IO.md`](../hexagon-ondevice/PIPELINE_IO.md).

`--plan-only` proves the crossing with **no NPU**: the driver returns before it
opens the device, so every step's velocity is a zero placeholder and the image
is not a render.

```bash
python -m flux_klein_host --backend device --plan-only --steps 2
```

`--checkpoints` writes the 19 stage boundaries below plus a `stages.json` in
the oracle's own `skye-npu-reference` format, so the comparison is the oracle's
existing tool with no glue:

```bash
python3 ../hexagon-ondevice/flux_oracle.py compare \
    --ref /tmp/klein-reference --got /tmp/klein-run/stages
```

### Stage boundaries at 512², 4 steps

| Stage                      | Shape                    | Owner                        |
|----------------------------|--------------------------|------------------------------|
| `tokenize.tokens`          | `(512,)` int32           | host, V2                     |
| `tokenize.attention_mask`  | `(512,)` uint8           | host, V2                     |
| `tokenize.attention_bias`  | `(1, 1, 512, 512)`       | host, V2                     |
| `encoder.out`              | `(1, 512, 7680)`         | **device**, V3               |
| `encoder.text_ids`         | `(1, 512, 4)` int64      | host                         |
| `denoise.noise`            | `(1, 32, 64, 64)`        | host                         |
| `denoise.latent_init`      | `(1, 1024, 128)`         | host (the 2x2 pack)          |
| `denoise.step_NN.velocity` | `(1, 1024, 128)`         | **device**, V4, one per step |
| `denoise.step_NN.latent`   | `(1, 1024, 128)`         | host (Euler), one per step   |
| `denoise.out`              | `(1, 1024, 128)`         | host (Euler)                 |
| `vae.input`                | `(1, 32, 64, 64)`        | host (BN-denorm then unpack) |
| `vae.rgb`                  | `(1, 3, 512, 512)`       | **device**, V5               |
| `image.rgb_u8`             | `(1, 512, 512, 3)` uint8 | host, V6                     |

At 4 steps that is 11 one-off stages plus 2 per step, so 19 files. The set is
pinned by `EXPECTED_STAGES` in `tests/test_pipeline.py`, so the table and the
dump cannot drift apart silently.

### Which driver replaces which stub

Swapping one in is a single field on `PipelineStages`; nothing inside
`run_pipeline` knows which implementation it called.

| Seam              | Stub today             | Replaced by                                  | Branch                                   |
|-------------------|------------------------|----------------------------------------------|------------------------------------------|
| `PromptTokenizer` | `SyntheticTokenizer`   | `klein_tokenize.py`                          | `hvx-tokenizer`                          |
| `TextEncoder`     | `SyntheticTextEncoder` | `npu_qwen3_encoder_dispatch.cpp`             | `hvx-encoder-layer`                      |
| `DiTBlockStack`   | `SyntheticBlockStack`  | `npu_block_dispatch.cpp` + `mojo_matmul_f16` | `hvx-block-driver`, `hvx-batched-matmul` |
| `VaeDecoder`      | `SyntheticVaeDecoder`  | `npu_vae_block_dispatch.cpp`                 | `hvx-vae-block`                          |

Where each of those stands, as of this branch: the tokenizer is landed and
host-only; the block driver is silicon-verified at MINI dims and is the one
device stage already proven; the batched matmul it needs at full dims is
host-green with no device run and no dispatcher yet; the encoder and VAE
drivers are host-green with their device runs queued.

`ReferenceTextEncoder` is the third route for the encoder seam: it replays an
`encoder.out` dump from the fp32 reference oracle, which is what makes a
denoise run reproducible against the oracle rather than merely self-consistent.

The synthetic stubs are deliberately not approximations of the real stages —
an approximation would invite reading the resulting image as a result. What
they do guarantee is that every input they are handed changes their output, so
a dropped `temb` or an ignored position grid fails a test instead of producing
a plausible picture.

## The three seams

Everything crossing a seam is a host-side float32 numpy array. Casting to fp16
and the `[N, K]` weight layout are the driver's concern and are deliberately
not encoded in these shapes.

### 1. Encoder output in — `TextConditioning`

| Field            | Shape                        | Notes                             |
|------------------|------------------------------|-----------------------------------|
| `prompt_embeds`  | `(B, 512, 7680)`             | Qwen3 layers 9/18/27 concatenated |
| `text_ids`       | `(B, 512, 4)` int64          | `(0, 0, 0, position)`             |
| `attention_bias` | `(B, 1, 512, 512)` or `None` | encoder padding mask only         |

The 7680 is the pre-`context_embedder` width — three stacked Qwen3 hidden
states at 2560 each, which is exactly `joint_attention_dim`. Projecting to the
DiT's 3072 is the block stack's job. Build one with
`text_conditioning_from_hidden_states`.

### 2. The DiT block-stack device call — `DiTBlockStack`

**Stubbed here.** A `Protocol` taking a `DiTBlockStackRequest` and returning the
velocity:

```python
def __call__(self, request: DiTBlockStackRequest) -> NDArray[float32]: ...
```

`DiTBlockStackRequest` carries `latent` `(B, 1024, 128)`, `image_ids`
`(B, 1024, 4)`, the `TextConditioning`, a `StepConditioning`, and the
`LatentGeometry`. The contract:

- returns the velocity with shape and dtype identical to `request.latent`, not
  aliasing it;
- text tokens are consumed but not returned — the double blocks update the text
  stream internally and the loop only advances the image latent;
- one forward per step. Klein is guidance-distilled, so there is no
  negative-prompt pass to combine.

`StepConditioning` carries the step index, the normalized timestep, `temb`
`(B, 3072)`, and an optional `StepModulation`. Modulation is *shared across all
25 blocks* in FLUX.2 rather than per block, so it is three small
latent-independent buffers (`double_img` and `double_txt` with 2 parameter sets
each, `single` with 1) that the host can precompute and stage once per step.
`None` means the callee derives them itself, which is the honest default until
the modulation weights are converted.

### 3. VAE decode input out — `VaeDecodeInput`

`vae_decode_input(final_latent, geometry, bn_stats)` produces
`(B, 32, 64, 64)` at 512², ready for the decoder's `post_quant_conv`.

Order is load-bearing: batch-norm denormalization runs on the **packed**
latent, then the 2x2 unpack. FLUX.2's VAE uses per-channel BN statistics
(`(128,)`, per packed channel) instead of the single scale/shift pair older
FLUX VAEs use, so unpacking first would apply them to the wrong channels.

## Validation

`python -m pytest tests -q` → 133 tests.

### The sampling math

Two independent references:

- **MAX's `FlowMatchEulerDiscreteScheduler`** — pure numpy, loaded from the
  monorepo by path, no build. This is the P6 oracle: matching it is what makes
  the scaffold substitutable for MAX-in-WSL's host prep. **Bit-exact** for
  every step count and both resolutions tested.
- **diffusers `FlowMatchEulerDiscreteScheduler` and
  `get_timestep_embedding`** — the genuinely independent check, since MAX's
  scheduler is itself a port and matching only MAX would not catch a shared
  porting error. Sigmas agree to 6e-8; the step update to 1e-6.

The timestep embedding agrees with diffusers only to ~3e-5, and that gap is
torch's, not ours: at the top of the range the pre-trig argument reaches 1000,
where torch's float32 range reduction carries ~3e-5 of error against float64
evaluation of the same arguments while numpy's carries ~5e-8. So the diffusers
comparison runs at a libm-scale tolerance and
`test_sinusoidal_matches_float64_ground_truth` is what actually pins accuracy.

### The orchestration

`tests/test_pipeline.py` checks what the driver is responsible for, which is
not numerics — the device stages are stubs, so there are no numerics to check.
It checks that the stages fit together and that nothing is silently dropped:

- every stage-boundary shape and dtype at 512²/4 steps, and again at 256²,
  512x256, and 1, 4 and 8 steps;
- **perturbation, not inspection** — changing the latent, `temb`,
  `prompt_embeds`, or `image_ids` each has to move the block stack's velocity,
  because a stub that ignored one of them would still produce a plausible
  image and still pass every shape assertion;
- a seed change and a prompt change each move the image, and the same
  seed and prompt reproduce it exactly (the P6 §7 sanity gates);
- the PNG is structurally valid — magic, chunk order, per-chunk CRC, IHDR
  dims — and decodes back to the same pixels under Pillow;
- the checkpoint dump replays the Euler chain it observed, and
  `flux_oracle.compare_dirs` reads the directory clean against itself while a
  single corrupted stage fails **that stage only** (the falsifiable control);
- the V1 loader trims K-padding off the timestep projection, and refuses a
  batch-norm at the decoder's width or with a negative variance — both of
  which otherwise surface as a NaN latent far downstream;
- the in-process and artifact tokenizer routes agree buffer for buffer.

The end-to-end runs cover the all-synthetic path, the real V2 tokenizer, a
converted V1 artifact, and an `encoder.out` dump from the fp32 oracle replayed
through seam 1 at MINI dims.

### Resolved 512² geometry

| Quantity           | Value                          |
|--------------------|--------------------------------|
| VAE latent         | `(32, 64, 64)`                 |
| DiT tokens         | `1024` (a 32x32 grid)          |
| Packed channels    | `128`                          |
| Joint sequence `M` | `1536` (1024 image + 512 text) |

### The 4-step schedule

`mu = 2.030690`, and the ladder is strongly back-loaded — the last step
consumes 72% of the interval, which is the empirical `mu` doing its job for
distilled few-step sampling:

```text
sigmas = [1.0, 0.958085, 0.883982, 0.717497, 0.0]
dts    = [-0.041915, -0.074104, -0.166485, -0.717497]
```

At `N = 1` the ladder degenerates to `[1.0, 0.0]` — a single jump — which is
covered as an edge case.

## Config provenance and the one judgement call

Every number in `config.py` is transcribed from a public, non-weight-gated
`config.json` on `black-forest-labs/FLUX.2-klein-4B` at revision `e7b7dc27`
(the revision pinned in `max/tests/integration/hf-repo-lock.tsv`).

**The latent packing is asserted, not assumed.** Three separate "patch" notions
collide in the FLUX.2 configs:

- `transformer/config.json` → `patch_size: 1` — the DiT does not patchify again
  on top of the pack.
- `vae/config.json` → `patch_size: [2, 2]` — read here as the 2x2 latent pack
  that turns VAE latent channels into DiT input channels, **not** an extra
  pixel-space unshuffle inside the VAE.
- `vae/config.json` → `block_out_channels` has 4 entries, so the conv stack
  strides by `2 ** 3 = 8`.

That reading is forced by two independent constraints, both checked in
`KleinConfig.__post_init__`: `in_channels == latent_channels * 4`
(`128 == 32 * 4`), and the resulting `M = 1536` at 512², which is the value the
block driver is sized for. It also matches MAX, which computes
`vae_scale_factor = 8` from `block_out_channels` and ignores the VAE
`patch_size` field entirely. The alternative reading — treating the VAE
`patch_size` as a real extra 2x downsample — would give a total stride of 16, a
`32x32x32` latent, 256 tokens and `M = 768`, contradicting both constraints.

**`use_empirical_mu` has no counterpart in the JSON.** MAX injects it for the
whole FLUX.2 family in `Flux2Tokenizer.__init__`, switching the
resolution-dependent shift from the linear `base_shift`/`max_shift`
interpolation to the step-count-aware piecewise fit from
black-forest-labs/flux2 `sampling.py`. It defaults to `True` here so the
scaffold matches MAX's Klein path rather than the bare JSON; the difference is
large at 4 steps, and `test_empirical_mu_differs_from_linear_mu` pins it so it
cannot be silently dropped.

## What is not here

Any device compute. The three on-device stages — the Qwen3 encoder driver, the
DiT block driver over the batched/tiled HVX matmul that makes the dispatch
count tractable, the VAE decoder driver — land on their own branches and are
called here only through the seams. Weight conversion (V1) and the tokenizer
(V2) are likewise consumed, not produced, and the reference oracle's fp32
forward is consumed as dumps rather than run in the loop.

So the image a synthetic run produces is not a render. It is proof that the
prompt reaches the encoder, the conditioning reaches the block stack, the
latent survives a pack and an unpack, and the pixels reach a file — with every
stage boundary dumped for the numerical gate that the real drivers will have to
pass.
