"""The thin runner: one command in WSL, an image out, MEFs on the NPU (P1.3).

.. code-block:: bash

    python3 flux_runner.py --prompt "a red bicycle" --out image.png \\
        --tokenizer /path/to/tokenizer.json [--steps 4]

The flow is the directive's: tokenize on the host, encoder stage, the denoise
loop with the scheduler host-side and ONE Windows process per executed step
(finding W), VAE stage, PNG. Orchestration is
:func:`flux_klein_host.pipeline.run_pipeline` -- this module only chooses the
stage backends, claims the device lane, and wires verification and the poison
path around them.

Stage backends are pluggable per stage. The denoise stage is always the MEF
(it is the leg proven on device); the encoder and the VAE each run ``mef`` or
``hand``:

* ``hand`` -- the dispatch-loop drivers that produced the 2026-08-01 render,
  through :mod:`flux_klein_host.device_bridge`. The default while P1.1/P1.2
  are open, so the runner renders TODAY.
* ``mef`` -- the whole-graph drivers of :mod:`flux_klein_host.mef_stages`.
  Selected automatically the moment a ``--encoder-stage``/``--vae-stage``
  staging directory is provided, which is the whole P1.1/P1.2 flip: no
  runner rework, one flag.

``--step-process per-render`` (the default since the order-071 A/B passed:
bit-identical latents, 26.0 s off the render wall) serves every denoise step
from ONE persistent process (EXECUTOR_DESIGN.md serve mode); ``per-step`` is
the permanent kill-switch restoring one-process-per-execute exactly as
shipped.

Lane discipline: the runner claims ``DEVICE_LANE.lock`` per ``DEVICE_LANE.md``
before the first device stage, refuses politely (naming the holder) when the
lane is held or poisoned, releases on every exit path, and on a launch
timeout marks the lock POISONED for the fleet instead of releasing it.

``--dry-run`` walks the identical orchestration -- lock included -- with
fabricated stage executors, no device and no Windows. Its image is a
placeholder, and the runner says so.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .checkpoints import StageCheckpoints
from .config import DEFAULT_NUM_STEPS, KLEIN_4B, KleinConfig, resolve_geometry
from .device_bridge import (
    DEFAULT_STAGE_DIR,
    BridgeError,
    DeviceTextEncoder,
    DeviceVaeDecoder,
)
from .image import save_png
from .interfaces import LatentBatchNormStats
from .lane import LaneBusyError, LaneLock, LanePoisonedError
from .mef_stages import (
    DEFAULT_STAGE_ROOT,
    CompletedLaunch,
    LaunchError,
    LaunchTimeoutError,
    MefDenoiseBlockStack,
    MefStageSpec,
    MefTextEncoder,
    MefVaeDecoder,
    PersistentStageSession,
    StageLauncher,
)
from .pipeline import PipelineRequest, PipelineStages, run_pipeline
from .stages import SyntheticTokenizer
from .tokenizer import ArtifactTokenizer, KleinTokenizer
from .weights import PipelineWeights

#: The hand-path drivers that produced the validated render, as staged.
_HAND_ENCODER = "npu_qwen3_encoder_stack_dispatch.exe"
_HAND_VAE = "npu_vae_decoder_dispatch.exe"

#: ``verifier(stage_names, got_dir)`` -- raises to STOP the run at that
#: boundary. The default shells out to ``stage_verify.py``; tests inject.
StageVerifier = Callable[[Sequence[str], Path], None]


class VerificationStop(RuntimeError):
    """A stage boundary refused verification; the run must not continue."""


class _Refusal(Exception):
    """A configuration that must not run; becomes a return code of 2."""


@dataclass
class VerifyingCheckpoints(StageCheckpoints):
    """Stage dumps that are judged as they land.

    Every recorded boundary is handed to the verifier the moment it is
    written, so the denoise loop stops at the FIRST refused step instead of
    rendering garbage three more times -- the run script's per-step
    ``stage_verify`` discipline, expressed through the checkpoint seam the
    pipeline already has.
    """

    verifier: StageVerifier | None = None

    def record(self, name, array, **kwargs):  # noqa: ANN001, ANN201
        path = super().record(name, array, **kwargs)
        if self.verifier is not None:
            try:
                self.verifier([name], self.out_dir)
            except Exception as error:
                raise VerificationStop(
                    f"stage_verify refused {name!r}: {error}"
                ) from error
        return path


def subprocess_verifier(ref_dir: Path) -> StageVerifier:
    """The real verifier: ``stage_verify.py`` against a recorded bundle.

    Stages absent from the reference bundle are skipped rather than failed --
    "every stage boundary stage-verified" means every boundary the recording
    knows; a bundle without ``image.rgb_u8`` must not stop a run at it.
    """
    script = (
        Path(__file__).resolve().parents[2]
        / "hexagon-ondevice"
        / "stage_verify.py"
    )

    def verify(stages: Sequence[str], got_dir: Path) -> None:
        present = [s for s in stages if (ref_dir / f"{s}.npy").is_file()]
        if not present:
            return
        command = [
            sys.executable,
            str(script),
            "verify",
            "--ref",
            str(ref_dir),
            "--got",
            str(got_dir),
        ]
        for stage in present:
            command += ["--stage", stage]
        done = subprocess.run(command, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(
                f"exit {done.returncode}\n{done.stdout}\n{done.stderr}"
            )

    return verify


def force_identity_bn_stats(
    weights: PipelineWeights,
    config: KleinConfig,
    *,
    echo: Callable[[str], None] = print,
) -> None:
    """Makes the host-side latent denormalization a no-op.

    The VAE MEF (``PostprocessAndDecode``) owns the batch-norm
    denormalization in-graph -- the real statistics are the graph's own
    weights. Applying a manifest's statistics on the host too would denorm
    twice, the silent plausible-wrong-image class, so a ``mef`` VAE forces
    the host stats to the identity.
    """
    weights.bn_stats = LatentBatchNormStats.identity(
        config.in_channels, eps=config.vae_batch_norm_eps
    )
    echo(
        "vae=mef: host batch-norm denormalization forced to identity "
        "(the graph owns the real statistics)"
    )


def _dry_run_process(
    argv: list[str], *, cwd: Path, env: dict[str, str], timeout: float
) -> CompletedLaunch:
    """Fabricates each stage's output, standing in for the device.

    The velocity depends on the staged ``input0`` so the loop integrates
    something step-varying; the encoder and VAE outputs are seeded noise.
    Nothing here is a model -- the resulting image is a placeholder.
    """
    del env, timeout
    exe, tag = Path(argv[0]).name, argv[3]
    outputs = Path(cwd) / "outputs"
    outputs.mkdir(exist_ok=True)
    if "encoder" in exe:
        np.random.RandomState(101).standard_normal((1, 512, 7680)).astype(
            np.float16
        ).tofile(outputs / "encoder.out.bin")
    elif "vae" in exe:
        np.random.RandomState(102).randint(
            0, 256, size=(1, 512, 512, 3), dtype=np.uint8
        ).tofile(outputs / "vae.rgb.bin")
    else:
        latent = np.fromfile(
            Path(cwd) / "inputs" / f"input0.step_{tag}.bin", dtype=np.float16
        )
        (latent * np.float16(-0.5)).tofile(outputs / f"step_{tag}.velocity.bin")
    return CompletedLaunch(returncode=0, output="ok: dry-run fabricated\n")


class _DryRunServeChild:
    """A fabricated serve-mode exe, driven through the liveness poll.

    Mirrors the real coupling exactly: the session only ever observes files
    and ``poll()``, so each poll gives this fake one turn to answer a ``go``
    sentinel the way ``npu_flux_whole_graph_exec.c`` would -- velocity
    derived from the staged ``input0``, then the nonce-carrying ``done``.
    """

    def __init__(self, argv: list[str], cwd: Path, log_path: Path) -> None:
        self.stage_dir = Path(cwd)
        self.log_path = log_path
        self.nonce = argv[5].split(":", 1)[1]
        self.returncode: int | None = None
        (self.stage_dir / "outputs").mkdir(exist_ok=True)
        self._log(f"ok: serve mode {self.nonce}")
        (self.stage_dir / "outputs" / f"serve.{self.nonce}.ready").touch()
        self._log("ok: serve ready")

    def _log(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        inputs = self.stage_dir / "inputs"
        finish = inputs / f"serve.{self.nonce}.finish"
        if finish.exists():
            finish.unlink()
            self._log("ok: serve finish")
            self.returncode = 0
            return self.returncode
        for go in sorted(inputs.glob(f"step_*.{self.nonce}.go")):
            tag = go.name[len("step_") : -len(f".{self.nonce}.go")]
            go.unlink()
            latent = np.fromfile(
                inputs / f"input0.step_{tag}.bin", dtype=np.float16
            )
            outputs = self.stage_dir / "outputs"
            (latent * np.float16(-0.5)).tofile(
                outputs / f"step_{tag}.velocity.bin"
            )
            (outputs / f"step_{tag}.{self.nonce}.done").touch()
            self._log(f"ok: serve step {tag} done")
        return None

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9


def _dry_run_serve_popen(
    argv: list[str], *, cwd: str, env: dict[str, str], stdout, stderr
) -> _DryRunServeChild:  # noqa: ANN001
    """The serve-mode counterpart of :func:`_dry_run_process`."""
    del env, stderr
    return _DryRunServeChild(argv, Path(cwd), Path(stdout.name))


def _default_lane_lock() -> Path:
    """The fleet's one lock file, resolved the way the run script does.

    Worktrees share the MAIN checkout's ``.agentwork``, so the path is
    derived from git's common dir, not from this file's checkout.
    """
    here = Path(__file__).resolve().parent
    try:
        common = subprocess.run(
            [
                "git",
                "-C",
                str(here),
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        root = Path(common).parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        root = here.parents[3]
    return root / ".agentwork" / "flux-npu" / "DEVICE_LANE.lock"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flux_runner.py",
        description=(
            "One command: prompt to image with every compute stage a MAX "
            "MEF on the NPU (P1.3). Denoise runs the staged whole-graph "
            "MEF; encoder and VAE run 'hand' until their staging "
            "directories exist, then flip to 'mef'."
        ),
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="Where the PNG lands."
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=3.5)

    parser.add_argument(
        "--tokenizer",
        default=None,
        help="tokenizer.json / directory / HF repo id (the real V2 path).",
    )
    parser.add_argument(
        "--tokens",
        type=Path,
        default=None,
        help="A directory the V2 tokenize CLI wrote, replayed as staged.",
    )

    for stage in ("encoder", "vae"):
        parser.add_argument(
            f"--{stage}",
            choices=("auto", "mef", "hand"),
            default="auto",
            help=(
                f"{stage} backend. auto = mef when --{stage}-stage is "
                "given, hand otherwise."
            ),
        )
        parser.add_argument(
            f"--{stage}-stage",
            type=Path,
            default=None,
            help=(
                f"Staging directory for the {stage} MEF (exe, MEF, "
                "weights/). Providing it is the P1.1/P1.2 flip."
            ),
        )
        parser.add_argument(
            f"--{stage}-driver",
            type=Path,
            default=None,
            help=f"The hand-path {stage} dispatch driver .exe.",
        )
    parser.add_argument(
        "--driver-weights",
        type=Path,
        default=None,
        help="skye-npu-weights manifest the HAND drivers stage from.",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=DEFAULT_STAGE_DIR,
        help="Where the hand-path bridge stages each crossing.",
    )

    parser.add_argument(
        "--stage",
        type=Path,
        default=None,
        help=(
            "The denoise staging directory (default the run package's "
            f"{DEFAULT_STAGE_ROOT})."
        ),
    )
    parser.add_argument(
        "--denoise-exe",
        default=None,
        help="Override the denoise driver exe name.",
    )
    parser.add_argument(
        "--denoise-mef", default=None, help="Override the denoise MEF name."
    )
    parser.add_argument(
        "--contract",
        default="probe11",
        choices=("probe11", "whole8"),
        help=(
            "The denoise MEF's input table. whole8 becomes available when "
            "the windows export of the whole graph compiles."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Seconds per device process; expiry POISONS.",
    )
    parser.add_argument(
        "--step-process",
        choices=("per-step", "per-render"),
        default="per-render",
        help=(
            "Denoise process granularity. per-render (default) serves "
            "every step from ONE persistent process -- the Phase-1 device "
            "A/B passed with bit-identical latents and 26.0 s off the "
            "render wall (order 071, PERF_PLAN §8). per-step restores "
            "one-process-per-execute exactly as shipped: the permanent "
            "kill-switch."
        ),
    )
    parser.add_argument(
        "--registry-mmap",
        action="store_true",
        help=(
            "Stage NPU_REGISTRY_MMAP=1 (#188): map weight files read-only "
            "so per-step processes share page-cache pages. Off by default."
        ),
    )
    parser.add_argument(
        "--teacher-force",
        type=Path,
        default=None,
        help=(
            "Bind each step's input0 from this recorded bundle instead of "
            "the free-running chain (the P0.2 motion). Off by default."
        ),
    )

    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Converted V1 manifest for the HOST-side tensors.",
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="Dump every stage boundary here (oracle format).",
    )
    parser.add_argument(
        "--verify-ref",
        type=Path,
        default=None,
        help=(
            "Recorded render bundle: every dumped boundary is stage-verified "
            "against it AS THE RUN GOES, stopping at the first refusal."
        ),
    )
    parser.add_argument(
        "--lane-lock",
        type=Path,
        default=None,
        help="Override the DEVICE_LANE.lock path (tests; default the repo's).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Walk the whole orchestration -- lock included -- with "
            "fabricated stage executors. No device, no Windows; the image "
            "is a placeholder."
        ),
    )
    return parser.parse_args(argv)


def _refuse(message: str) -> int:
    print(f"REFUSED: {message}", file=sys.stderr)
    return 2


def main(
    argv: Sequence[str] | None = None,
    *,
    run_process: Callable[..., CompletedLaunch] | None = None,
    popen: Callable[..., object] | None = None,
    verifier: StageVerifier | None = None,
) -> int:
    """Command-line entry point.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
        run_process: Exec seam override for every stage launcher. Tests use
            it to script the device; ``--dry-run`` defaults it to the
            fabricator.
        popen: Process seam override for the ``--step-process per-render``
            serve session. Tests inject a fake serve child; ``--dry-run``
            defaults it to the serve fabricator.
        verifier: Verification seam override; defaults to ``stage_verify.py``
            against ``--verify-ref``.

    Returns:
        0 on success; 1 on a stage failure or verification stop; 2 on a
        refused configuration; 3 when the lane is held or poisoned; 4 when
        this run poisoned the lane (timeout).
    """
    args = _parse_args(argv)
    config = KLEIN_4B
    geometry = resolve_geometry(config, args.height, args.width)
    if geometry.image_seq_len != 1024:
        return _refuse(
            f"the staged denoise MEF is compiled at 512x512 (image_seq "
            f"1024); {args.height}x{args.width} gives "
            f"{geometry.image_seq_len}. Re-export the MEF for this geometry."
        )

    if args.dry_run and run_process is None:
        run_process = _dry_run_process
    if args.dry_run and popen is None:
        popen = _dry_run_serve_popen

    # Every pure-configuration refusal fires before anything is constructed
    # or claimed. A device render with hashed synthetic tokens is the
    # coherent-but-wrong trap, so the tokenizer requirement is one of them.
    if args.tokens is None and args.tokenizer is None and not args.dry_run:
        return _refuse(
            "a device render needs the real tokenizer (--tokenizer or "
            "--tokens); synthetic hashed tokens would produce a coherent "
            "but wrong image."
        )

    out_dir = args.out.parent

    def _mef_launcher(stage_dir: Path, spec: MefStageSpec) -> StageLauncher:
        return StageLauncher(
            stage_dir=stage_dir,
            spec=spec,
            registry_mmap=args.registry_mmap,
            timeout_seconds=args.timeout,
            **({} if run_process is None else {"run_process": run_process}),
        )

    def _stage_mode(name: str) -> tuple[str, Path | None]:
        mode = getattr(args, name)
        stage_dir = getattr(args, f"{name}_stage")
        if mode == "auto":
            mode = "mef" if stage_dir is not None or args.dry_run else "hand"
        if mode == "hand" and args.dry_run:
            # The hand drivers ignore the run_process seam, so an explicit
            # `--encoder hand --dry-run` used to occupy the device for a
            # real encode labeled "dry run" (order-071 harness defect 2).
            # Dry run PROMISES no device and no Windows; coerce to the stub.
            print(
                f"dry-run: {name}=hand coerced to the fabricated stub "
                "(dry run never touches the device)"
            )
            mode = "mef"
        if mode == "mef" and stage_dir is None:
            if not args.dry_run:
                raise _Refusal(f"--{name}=mef needs --{name}-stage")
            stage_dir = out_dir / f"dry-run-stage-{name}"
            stage_dir.mkdir(parents=True, exist_ok=True)
        return mode, stage_dir

    try:
        encoder_mode, encoder_stage = _stage_mode("encoder")
        vae_mode, vae_stage = _stage_mode("vae")
    except _Refusal as refusal:
        return _refuse(str(refusal))

    if args.tokens is not None:
        tokenizer = ArtifactTokenizer(args.tokens, config)
    elif args.tokenizer is not None:
        tokenizer = KleinTokenizer(config, spec=args.tokenizer)
    else:
        tokenizer = SyntheticTokenizer(config)

    if encoder_mode == "mef":
        spec = MefStageSpec.encoder()
        text_encoder = MefTextEncoder(
            _mef_launcher(encoder_stage, spec), config
        )
    else:
        text_encoder = DeviceTextEncoder(
            args.encoder_driver or args.scratch_root / _HAND_ENCODER,
            config,
            weights=args.driver_weights,
            scratch_root=args.scratch_root,
            timeout_seconds=args.timeout,
        )
    if vae_mode == "mef":
        vae_decoder = MefVaeDecoder(
            _mef_launcher(vae_stage, MefStageSpec.vae()), config
        )
    else:
        vae_decoder = DeviceVaeDecoder(
            args.vae_driver or args.scratch_root / _HAND_VAE,
            config,
            weights=args.driver_weights,
            scratch_root=args.scratch_root,
            timeout_seconds=args.timeout,
        )

    denoise_stage = args.stage
    if denoise_stage is None:
        denoise_stage = (
            out_dir / "dry-run-stage-denoise"
            if args.dry_run
            else DEFAULT_STAGE_ROOT
        )
    if args.dry_run:
        denoise_stage.mkdir(parents=True, exist_ok=True)
    denoise_spec = MefStageSpec.denoise()
    if args.denoise_exe or args.denoise_mef or args.contract != "probe11":
        denoise_spec = MefStageSpec(
            exe=args.denoise_exe or denoise_spec.exe,
            mef=args.denoise_mef or denoise_spec.mef,
            contract=args.contract,
        )
    serve_session: PersistentStageSession | None = None
    if args.step_process == "per-render":
        serve_session = PersistentStageSession(
            stage_dir=denoise_stage,
            spec=denoise_spec,
            registry_mmap=args.registry_mmap,
            timeout_seconds=args.timeout,
            **({} if popen is None else {"popen": popen}),
        )
    block_stack = MefDenoiseBlockStack(
        serve_session or _mef_launcher(denoise_stage, denoise_spec),
        config,
        guidance=args.guidance,
        teacher_force=args.teacher_force,
    )
    if serve_session is not None:
        # Order 071 measured the session holding ~15.2 GB for 152.7 s past
        # the last step, overlapping the VAE. The runner knows the step
        # count, so the session closes the moment the last velocity is read
        # back -- the denoise footprint is freed before the VAE starts, and
        # the exe's teardown perf marks stay visible to the next A/B.
        mef_stack, session = block_stack, serve_session
        final_step = args.steps - 1

        def close_after_last_step(request):  # noqa: ANN001, ANN201
            velocity = mef_stack(request)
            if request.conditioning.step_index == final_step:
                session.close()
            return velocity

        block_stack = close_after_last_step

    stages = PipelineStages(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        block_stack=block_stack,
        vae_decoder=vae_decoder,
    )

    weights = (
        PipelineWeights.from_manifest(args.weights, config)
        if args.weights is not None
        else PipelineWeights.synthetic(config)
    )
    if vae_mode == "mef":
        force_identity_bn_stats(weights, config)

    checkpoints: StageCheckpoints | None = None
    if args.verify_ref is not None or args.checkpoints is not None:
        checkpoint_dir = args.checkpoints or out_dir / "stages"
        active_verifier = verifier
        if active_verifier is None and args.verify_ref is not None:
            active_verifier = subprocess_verifier(args.verify_ref)
        checkpoints = VerifyingCheckpoints(
            out_dir=checkpoint_dir, verifier=active_verifier
        )

    request = PipelineRequest(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.steps,
        seed=args.seed,
    )

    lock = LaneLock(
        args.lane_lock or _default_lane_lock(),
        holder="flux_runner (P1.3 thin runner)",
        running=(
            f"{args.steps}-step render, encoder={encoder_mode} "
            f"vae={vae_mode} contract={denoise_spec.contract}"
            f"{' DRY RUN' if args.dry_run else ''}, prompt={args.prompt[:48]!r}"
        ),
        expect=(
            "dry run, seconds"
            if args.dry_run
            else f"~{12 * args.steps + 25} min ({args.steps} denoise legs "
            "at ~11-12 min measured, plus the encoder and VAE legs)"
        ),
        yield_at="between steps on request",
    )

    try:
        with lock:
            try:
                with weights:
                    result = run_pipeline(
                        request,
                        stages,
                        weights,
                        config=config,
                        checkpoints=checkpoints,
                    )
            except LaunchTimeoutError as error:
                lock.poison(f"flux_runner: {error}")
                print(
                    f"POISONED: {error}\nThe lane lock now carries the "
                    "poison marker; recovery is a full Windows reboot "
                    "(human-gated). Do not rerun.",
                    file=sys.stderr,
                )
                return 4
            except VerificationStop as error:
                print(f"STOP: {error}", file=sys.stderr)
                return 1
            except (BridgeError, LaunchError) as error:
                # The stage refused or its output was unusable. The lock
                # releases on this path; a TIMEOUT never lands here.
                print(f"STOP: {error}", file=sys.stderr)
                return 1
            finally:
                # The serve session ends while the lane is still held: its
                # teardown is device work like any other. No-op on the
                # per-step path and after a timeout (the child is dead).
                if serve_session is not None:
                    serve_session.close()
    except LanePoisonedError as error:
        print(f"{error}", file=sys.stderr)
        return 3
    except LaneBusyError as error:
        print(f"{error}", file=sys.stderr)
        return 3

    if checkpoints is not None:
        checkpoints.write_manifest(
            {"prompt": args.prompt, "steps": args.steps, "seed": args.seed}
        )
    png = save_png(args.out, result.image)

    print(f"prompt       {args.prompt!r}")
    print(
        f"backends     encoder={encoder_mode} denoise=mef"
        f"[{denoise_spec.contract}] vae={vae_mode} "
        f"step-process={args.step_process}"
    )
    print(
        f"schedule     mu={result.denoise.schedule.mu:.6f} "
        f"sigmas={np.round(result.denoise.schedule.sigmas, 6)}"
    )
    print("stage seconds")
    for name, seconds in result.stage_seconds.items():
        print(f"  {name:<14} {seconds:.3f}")
    if args.verify_ref is not None:
        print(f"verified     every recorded boundary vs {args.verify_ref}")
    print(f"image        {png}")
    if args.dry_run:
        print("DRY RUN: the image is a placeholder, not a render.")
    return 0


__all__ = [
    "VerificationStop",
    "VerifyingCheckpoints",
    "force_identity_bn_stats",
    "main",
    "subprocess_verifier",
]
