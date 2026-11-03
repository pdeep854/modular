"""The thin runner, proven device-free (P1.3).

Everything here runs with stubbed stage executors and a temporary lane lock:
what is asserted is the ORCHESTRATION -- stage sequencing, the one-process-
per-execute discipline, knob plumbing across the WSL boundary, lane lock
acquire/release on every exit path, and the vsock-flake retry -- never any
model math. The numerics stay owned by the modules the runner imports.

The stub seam sits BELOW the retry/knob logic: a fake ``run_process`` stands
in for the Windows interop exec, so the launcher's own retry loop and WSLENV
composition are what actually execute here, against fabricated driver
behavior (flakes, timeouts, written or withheld outputs).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from flux_klein_host.config import KLEIN_4B, resolve_geometry
from flux_klein_host.denoise import denoise
from flux_klein_host.device_bridge import BridgeError
from flux_klein_host.image import to_rgb_uint8
from flux_klein_host.interfaces import (
    LatentBatchNormStats,
    TextConditioning,
    vae_decode_input,
)
from flux_klein_host.lane import LaneBusyError, LaneLock, LanePoisonedError
from flux_klein_host.latents import init_latent, pack, text_ids
from flux_klein_host.mef_stages import (
    CompletedLaunch,
    LaunchError,
    LaunchTimeoutError,
    MefDenoiseBlockStack,
    MefStageSpec,
    MefTextEncoder,
    MefVaeDecoder,
    StageLauncher,
)
from flux_klein_host.rope import packed_freqs_cis
from flux_klein_host.runner import (
    StageVerifier,
    _DryRunServeChild,
    force_identity_bn_stats,
    main,
)
from flux_klein_host.scheduler import FlowMatchEulerScheduler
from flux_klein_host.stages import SyntheticTokenizer, TextEncodeRequest
from flux_klein_host.timestep_embed import (
    TIMESTEP_SCALE,
    TimestepEmbedderWeights,
    sinusoidal_embedding,
)
from flux_klein_host.weights import PipelineWeights

_GEOMETRY = resolve_geometry(KLEIN_4B, 512, 512)

# --------------------------------------------------------------------------
# The device-lane lock (DEVICE_LANE.md discipline)
# --------------------------------------------------------------------------


def _lock(tmp_path: Path, **kwargs: object) -> LaneLock:
    defaults: dict[str, object] = {
        "path": tmp_path / "DEVICE_LANE.lock",
        "holder": "flux_runner (mac/thin-runner)",
        "running": "device-free lock test",
        "expect": "seconds -- a unit test",
        "yield_at": "none",
        "busy_check": lambda: None,
    }
    defaults.update(kwargs)
    return LaneLock(**defaults)  # type: ignore[arg-type]


def test_claim_writes_the_five_fields_and_release_removes(
    tmp_path: Path,
) -> None:
    lock = _lock(tmp_path)
    with lock:
        text = lock.path.read_text(encoding="utf-8")
        for field in ("holder:", "running:", "claimed:", "expect:", "yield:"):
            assert field in text, f"missing {field!r} in the claim"
    assert not lock.path.exists()


def test_release_happens_on_the_failure_path_too(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    with pytest.raises(RuntimeError, match="mid-run"):
        with lock:
            raise RuntimeError("mid-run failure")
    assert not lock.path.exists()


def test_a_held_lane_is_refused_politely_naming_the_holder(
    tmp_path: Path,
) -> None:
    path = tmp_path / "DEVICE_LANE.lock"
    path.write_text(
        "holder: someone else\nclaimed: 2026-08-06T00:00:00Z\n",
        encoding="utf-8",
    )
    with pytest.raises(LaneBusyError, match="someone else"):
        with _lock(tmp_path):
            pass
    assert path.read_text(encoding="utf-8").startswith("holder: someone else")


def test_a_poisoned_lane_is_refused_with_the_recovery_truth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "DEVICE_LANE.lock"
    path.write_text("holder: POISONED -- do NOT claim\n", encoding="utf-8")
    with pytest.raises(LanePoisonedError, match="reboot"):
        with _lock(tmp_path):
            pass
    assert path.exists(), "a poison marker must never be deleted"


def test_a_physically_busy_lane_is_refused_before_claiming(
    tmp_path: Path,
) -> None:
    def busy() -> str:
        return "npu_flux_whole_graph_exec.exe"

    with pytest.raises(LaneBusyError, match=r"npu_flux_whole_graph_exec\.exe"):
        with _lock(tmp_path, busy_check=busy):
            pass
    assert not (tmp_path / "DEVICE_LANE.lock").exists()


def test_poison_rewrites_the_lock_and_release_keeps_it(tmp_path: Path) -> None:
    lock = _lock(tmp_path)
    with lock:
        lock.poison("step 02 timed out after 1800s")
    text = lock.path.read_text(encoding="utf-8")
    assert "POISONED" in text
    assert "step 02 timed out" in text
    assert "reboot" in text.lower()


def test_two_claims_of_one_path_race_to_a_single_winner(
    tmp_path: Path,
) -> None:
    first = _lock(tmp_path)
    second = _lock(tmp_path, holder="the other agent")
    with first:
        with pytest.raises(LaneBusyError, match="flux_runner"):
            second.claim()
    assert not first.path.exists()


# --------------------------------------------------------------------------
# The per-stage Windows launch: knobs, retry, timeout
# --------------------------------------------------------------------------


_FLAKE_LOG = "<3>WSL (1234) ERROR: UtilAcceptVsock:250: accept4 failed 110\n"


class FakeDriver:
    """Stands in for the Windows interop exec, scripted per attempt.

    ``behaviors`` is consumed one entry per call: ``"flake"`` reproduces the
    vsock-110 interop failure (the exe never ran, so no ``ok:`` line),
    ``"timeout"`` raises the way the real seam does when the process outlives
    its budget, ``"fail"`` is a driver that ran and refused, and ``"ok"``
    runs ``on_launch`` (the output fabricator) and reports success.
    """

    def __init__(
        self,
        behaviors: Sequence[str] | None = None,
        on_launch: Callable[[list[str], Path], None] | None = None,
    ) -> None:
        self.behaviors = list(behaviors or [])
        self.on_launch = on_launch
        self.calls: list[tuple[list[str], Path, dict[str, str], float]] = []

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> CompletedLaunch:
        self.calls.append((list(argv), Path(cwd), dict(env), timeout))
        behavior = self.behaviors.pop(0) if self.behaviors else "ok"
        if behavior == "flake":
            return CompletedLaunch(returncode=1, output=_FLAKE_LOG)
        if behavior == "timeout":
            raise LaunchTimeoutError(
                f"{argv[0]} did not finish within {timeout}s"
            )
        if behavior == "fail":
            return CompletedLaunch(
                returncode=2, output="ok: M_initModel\nFAIL: execute refused\n"
            )
        if self.on_launch is not None:
            self.on_launch(list(argv), Path(cwd))
        return CompletedLaunch(returncode=0, output="ok: M_executeModelSync\n")


def _launcher(stage_dir: Path, driver: FakeDriver, **kwargs) -> StageLauncher:
    sleeps: list[float] = []
    launcher = StageLauncher(
        stage_dir=stage_dir,
        base_env={"WSLENV": "PRE_EXISTING/p"},
        run_process=driver,
        sleep=sleeps.append,
        echo=lambda line: None,
        **kwargs,
    )
    launcher.test_sleeps = sleeps  # type: ignore[attr-defined]
    return launcher


def test_registry_mmap_knob_crosses_the_interop_boundary(
    tmp_path: Path,
) -> None:
    driver = FakeDriver()
    _launcher(tmp_path, driver, registry_mmap=True).launch("00")
    _, _, env, _ = driver.calls[0]
    assert env["NPU_REGISTRY_MMAP"] == "1"
    # One composed WSLENV: the pre-existing entry must survive alongside the
    # knob, or one of the two silently stops crossing to Windows.
    assert env["WSLENV"] == "PRE_EXISTING/p:NPU_REGISTRY_MMAP"


def test_the_mmap_knob_is_off_by_default(tmp_path: Path) -> None:
    driver = FakeDriver()
    _launcher(tmp_path, driver).launch("00")
    _, _, env, _ = driver.calls[0]
    assert "NPU_REGISTRY_MMAP" not in env
    assert env["WSLENV"] == "PRE_EXISTING/p"


def test_modular_npu_knobs_cross_wslenv_automatically(tmp_path: Path) -> None:
    """Order-071 harness defect 1: a MODULAR_NPU_* knob set but not named
    in WSLENV never reaches the Windows exe, so a device leg can run
    flag-off while reporting success. The composition now names every such
    knob automatically, absolute-path values with /p for translation."""
    driver = FakeDriver()
    launcher = StageLauncher(
        stage_dir=tmp_path,
        base_env={
            "WSLENV": "PRE_EXISTING/p",
            "MODULAR_NPU_PERF_LOG": "/mnt/c/Users/Public/perf",
            "MODULAR_NPU_PERF_DEVICE": "1",
        },
        run_process=driver,
        sleep=lambda seconds: None,
        echo=lambda line: None,
    )
    launcher.launch("00")
    _, _, env, _ = driver.calls[0]
    assert env["WSLENV"] == (
        "PRE_EXISTING/p:MODULAR_NPU_PERF_DEVICE:MODULAR_NPU_PERF_LOG/p"
    )


def test_wslenv_names_already_present_are_not_duplicated(
    tmp_path: Path,
) -> None:
    driver = FakeDriver()
    launcher = StageLauncher(
        stage_dir=tmp_path,
        base_env={
            "WSLENV": "MODULAR_NPU_PERF_DEVICE",
            "MODULAR_NPU_PERF_DEVICE": "1",
        },
        run_process=driver,
        sleep=lambda seconds: None,
        echo=lambda line: None,
    )
    launcher.launch("00")
    _, _, env, _ = driver.calls[0]
    assert env["WSLENV"] == "MODULAR_NPU_PERF_DEVICE"


def test_a_vsock_flake_is_retried_and_then_succeeds(tmp_path: Path) -> None:
    driver = FakeDriver(behaviors=["flake", "ok"])
    launcher = _launcher(tmp_path, driver)
    launcher.launch("00")
    assert len(driver.calls) == 2
    assert launcher.test_sleeps == [2.0]  # type: ignore[attr-defined]


def test_a_persistent_vsock_flake_fails_clearly_after_three(
    tmp_path: Path,
) -> None:
    driver = FakeDriver(behaviors=["flake", "flake", "flake"])
    with pytest.raises(LaunchError, match=r"(?i)vsock"):
        _launcher(tmp_path, driver).launch("00")
    assert len(driver.calls) == 3


def test_a_timeout_propagates_without_retry(tmp_path: Path) -> None:
    driver = FakeDriver(behaviors=["timeout"])
    with pytest.raises(LaunchTimeoutError):
        _launcher(tmp_path, driver).launch("00")
    assert len(driver.calls) == 1, "a timeout poisons; it must not be retried"


def test_a_driver_refusal_carries_the_log_tail(tmp_path: Path) -> None:
    driver = FakeDriver(behaviors=["fail"])
    with pytest.raises(LaunchError, match="execute refused"):
        _launcher(tmp_path, driver).launch("00")


def test_the_argv_is_the_proven_whole_graph_contract(tmp_path: Path) -> None:
    driver = FakeDriver()
    _launcher(tmp_path, driver).launch("02")
    argv, cwd, _, timeout = driver.calls[0]
    assert argv == [
        "./npu_flux_whole_graph_exec.exe",
        "npu.mef",
        ".",
        "02",
        "probe11",
    ]
    assert cwd == tmp_path
    assert timeout == 1800.0


# --------------------------------------------------------------------------
# The denoise stage: one process per step, prep-exact staging
# --------------------------------------------------------------------------


def _text_conditioning(seed: int = 3) -> TextConditioning:
    rng = np.random.RandomState(seed)
    return TextConditioning(
        prompt_embeds=rng.standard_normal((1, 512, 7680)).astype(np.float32),
        text_ids=text_ids(512),
    )


def _velocity_fabricator(argv: list[str], cwd: Path) -> None:
    """What a healthy driver does: read input0, write a dependent velocity."""
    tag = argv[3]
    latent = np.fromfile(
        cwd / "inputs" / f"input0.step_{tag}.bin", dtype=np.float16
    )
    (cwd / "outputs").mkdir(exist_ok=True)
    (latent * np.float16(-0.5)).tofile(
        cwd / "outputs" / f"step_{tag}.velocity.bin"
    )


def _denoise_stack(
    tmp_path: Path, driver: FakeDriver, **kwargs
) -> MefDenoiseBlockStack:
    echoes: list[str] = []
    stack = MefDenoiseBlockStack(
        _launcher(tmp_path, driver),
        echo=echoes.append,
        **kwargs,
    )
    stack.test_echoes = echoes  # type: ignore[attr-defined]
    return stack


def test_the_denoise_loop_is_one_process_per_step(tmp_path: Path) -> None:
    driver = FakeDriver(on_launch=_velocity_fabricator)
    stack = _denoise_stack(tmp_path, driver)
    text = _text_conditioning()
    weights = PipelineWeights.synthetic(KLEIN_4B)

    result = denoise(
        stack,
        text,
        _GEOMETRY,
        weights.timestep,
        num_steps=2,
        seed=0,
        keep_records=True,
    )

    assert [call[0][3] for call in driver.calls] == ["00", "01"]

    inputs = tmp_path / "inputs"
    step0_latent = np.fromfile(
        inputs / "input0.step_00.bin", dtype=np.float16
    ).reshape(1, 1024, 128)
    assert step0_latent.tobytes() == (
        pack(result.initial_latent).astype(np.float16).tobytes()
    )

    sigmas = FlowMatchEulerScheduler().sigmas(_GEOMETRY.image_seq_len, 2)
    for step in range(2):
        staged_t = np.fromfile(
            inputs / f"input3.step_{step:02d}.bin", dtype=np.float32
        )
        assert (
            staged_t.tobytes()
            == np.full((1,), sigmas[step], np.float32).tobytes()
        )
        # D3 semantics (#196): the graph casts the scaled timestep to fp16
        # before `Timesteps` (flux2.py), so the staged projection carries the
        # fp16-rounded argument -- 958.085... staged as 958.0.
        scaled = (
            (
                np.full((1,), sigmas[step], np.float32)
                * np.float32(TIMESTEP_SCALE)
            )
            .astype(np.float16)
            .astype(np.float32)
        )
        assert (
            np.fromfile(
                inputs / f"input8.step_{step:02d}.bin", dtype=np.float32
            ).tobytes()
            == sinusoidal_embedding(
                scaled, KLEIN_4B.timestep_guidance_channels
            ).tobytes()
        )

    assert (
        np.fromfile(inputs / "input2.bin", dtype=np.float16).tobytes()
        == text.prompt_embeds.astype(np.float16).tobytes()
    )
    assert np.fromfile(inputs / "input4.bin", dtype=np.float32)[0] == 3.5
    assert (inputs / "input1.bin").stat().st_size == 0
    assert (inputs / "input6.bin").stat().st_size == 0
    assert (
        np.fromfile(inputs / "input7.bin", dtype=np.int64)
        .reshape(1, 512, 4)
        .tobytes()
        == text.text_ids.tobytes()
    )

    all_ids = np.concatenate(
        [
            text.text_ids[0].astype(np.float64),
            np.fromfile(inputs / "input5.bin", dtype=np.int64)
            .reshape(1, 1024, 4)[0]
            .astype(np.float64),
        ],
        axis=0,
    )
    assert (
        np.fromfile(inputs / "input9.bin", dtype=np.float32).tobytes()
        == packed_freqs_cis(all_ids).astype(np.float32).tobytes()
    )
    assert (
        np.fromfile(inputs / "input10.bin", dtype=np.uint32).tobytes()
        == np.arange(1536, dtype=np.uint32)[None, :].tobytes()
    )

    # The tripwire from the plan: every per-step binding names its sha256.
    echoes = "\n".join(stack.test_echoes)  # type: ignore[attr-defined]
    for name in ("input0.step_01", "input3.step_01", "input8.step_01"):
        staged = (inputs / f"{name}.bin").read_bytes()
        assert hashlib.sha256(staged).hexdigest() in echoes, name

    # The loop consumed the fabricated velocities, not placeholders.
    assert (
        result.records[0].velocity.tobytes()
        == (step0_latent.astype(np.float16) * np.float16(-0.5))
        .astype(np.float32)
        .tobytes()
    )


def test_a_stale_velocity_cannot_be_read_back(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    stale = np.full((1, 1024, 128), np.float16(7.0))
    stale.tofile(tmp_path / "outputs" / "step_00.velocity.bin")

    driver = FakeDriver()  # "ok", but writes nothing
    stack = _denoise_stack(tmp_path, driver)
    with pytest.raises(BridgeError, match=r"(?i)wrote no|missing|not.*writ"):
        denoise(
            stack,
            _text_conditioning(),
            _GEOMETRY,
            PipelineWeights.synthetic(KLEIN_4B).timestep,
            num_steps=1,
            seed=0,
        )


def test_a_non_finite_velocity_is_refused(tmp_path: Path) -> None:
    def poisoned(argv: list[str], cwd: Path) -> None:
        (cwd / "outputs").mkdir(exist_ok=True)
        np.full((1, 1024, 128), np.float16(np.nan)).tofile(
            cwd / "outputs" / f"step_{argv[3]}.velocity.bin"
        )

    stack = _denoise_stack(tmp_path, FakeDriver(on_launch=poisoned))
    with pytest.raises(BridgeError, match="non-finite"):
        denoise(
            stack,
            _text_conditioning(),
            _GEOMETRY,
            PipelineWeights.synthetic(KLEIN_4B).timestep,
            num_steps=1,
            seed=0,
        )


def test_teacher_forcing_binds_the_recorded_latents(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    recording.mkdir()
    rng = np.random.RandomState(11)
    recorded_init = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    recorded_step0 = rng.standard_normal((1, 1024, 128)).astype(np.float32)
    np.save(recording / "denoise.latent_init.npy", recorded_init)
    np.save(recording / "denoise.step_00.latent.npy", recorded_step0)

    stage = tmp_path / "stage"
    stage.mkdir()
    driver = FakeDriver(on_launch=_velocity_fabricator)
    stack = _denoise_stack(stage, driver, teacher_force=recording)
    denoise(
        stack,
        _text_conditioning(),
        _GEOMETRY,
        PipelineWeights.synthetic(KLEIN_4B).timestep,
        num_steps=2,
        seed=0,
    )
    inputs = stage / "inputs"
    assert (
        np.fromfile(inputs / "input0.step_00.bin", dtype=np.float16).tobytes()
        == recorded_init.astype(np.float16).tobytes()
    )
    # Step 1 binds the RECORDING's step-0 latent, not the free-run chain.
    assert (
        np.fromfile(inputs / "input0.step_01.bin", dtype=np.float16).tobytes()
        == recorded_step0.astype(np.float16).tobytes()
    )


def test_a_partial_bundle_forces_step_0_and_free_runs_the_rest(
    tmp_path: Path,
) -> None:
    # 293's seam: a fresh seed's bundle carries ONLY latent_init, because
    # steps 1..N-1 do not exist until the device produces them. Step 0 is
    # forced from the recording; later steps ride the free-running chain.
    recording = tmp_path / "recording"
    recording.mkdir()
    recorded_init = (
        np.random.RandomState(11)
        .standard_normal((1, 1024, 128))
        .astype(np.float32)
    )
    np.save(recording / "denoise.latent_init.npy", recorded_init)

    stage = tmp_path / "stage"
    stage.mkdir()
    stack = _denoise_stack(
        stage,
        FakeDriver(on_launch=_velocity_fabricator),
        teacher_force=recording,
    )
    denoise(
        stack,
        _text_conditioning(),
        _GEOMETRY,
        PipelineWeights.synthetic(KLEIN_4B).timestep,
        num_steps=2,
        seed=0,
    )

    inputs = stage / "inputs"
    assert (
        np.fromfile(inputs / "input0.step_00.bin", dtype=np.float16).tobytes()
        == recorded_init.astype(np.float16).tobytes()
    )
    # Step 1 must be the Euler update of the free-running chain (numpy seed
    # 0) under the velocity the device computed on the FORCED step-0 input,
    # reproduced here to the byte -- a re-raise and a wrong fallback source
    # both fail this equality.
    scheduler = FlowMatchEulerScheduler()
    sigmas = scheduler.sigmas(_GEOMETRY.image_seq_len, 2)
    chain0 = pack(init_latent(_GEOMETRY, seed=0))
    velocity0 = (
        recorded_init.astype(np.float16) * np.float16(-0.5)
    ).astype(np.float32)
    expected = scheduler.step(velocity0, chain0, sigmas, 0)
    assert (
        np.fromfile(inputs / "input0.step_01.bin", dtype=np.float16).tobytes()
        == expected.astype(np.float16).tobytes()
    )
    echoes = "\n".join(stack.test_echoes)  # type: ignore[attr-defined]
    assert "denoise.step_00.latent.npy absent" in echoes


def test_a_wrong_shape_forced_latent_still_refuses(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    recording.mkdir()
    np.save(
        recording / "denoise.latent_init.npy",
        np.zeros((1, 512, 128), np.float32),
    )
    stage = tmp_path / "stage"
    stage.mkdir()
    stack = _denoise_stack(
        stage,
        FakeDriver(on_launch=_velocity_fabricator),
        teacher_force=recording,
    )
    with pytest.raises(BridgeError, match="shape"):
        denoise(
            stack,
            _text_conditioning(),
            _GEOMETRY,
            PipelineWeights.synthetic(KLEIN_4B).timestep,
            num_steps=1,
            seed=0,
        )


# --------------------------------------------------------------------------
# The encoder and VAE MEF seams (P1.1 / P1.2 flip targets)
# --------------------------------------------------------------------------


def test_the_encoder_mef_seam_stages_the_graph_contract(
    tmp_path: Path,
) -> None:
    hidden = np.random.RandomState(5).standard_normal((1, 512, 7680))

    def fabricate(argv: list[str], cwd: Path) -> None:
        (cwd / "outputs").mkdir(exist_ok=True)
        hidden.astype(np.float16).tofile(cwd / "outputs" / "encoder.out.bin")

    driver = FakeDriver(on_launch=fabricate)
    encoder = MefTextEncoder(
        _launcher(tmp_path, driver, spec=MefStageSpec.encoder())
    )
    tokens = SyntheticTokenizer(KLEIN_4B)("a red bicycle")
    conditioning = encoder(TextEncodeRequest(tokens=tokens))

    inputs = tmp_path / "inputs"
    # Qwen3TextEncoderTransformer.input_types: int64 [S] tokens, then the
    # float32 [1, 1, S, S] additive bias.
    assert (
        np.fromfile(inputs / "input0.bin", dtype=np.int64).tobytes()
        == tokens.tokens.astype(np.int64).tobytes()
    )
    assert (
        np.fromfile(inputs / "input1.bin", dtype=np.float32).tobytes()
        == tokens.attention_bias.tobytes()
    )
    argv = driver.calls[0][0]
    assert argv[0] == "./npu_flux_encoder_exec.exe"
    assert argv[1] == "encoder.mef"

    conditioning.validate(KLEIN_4B)
    assert conditioning.prompt_embeds.tobytes() == (
        hidden.astype(np.float16).astype(np.float32).tobytes()
    )


# --------------------------------------------------------------------------
# The runner CLI: sequencing, lock lifecycle, knobs, verification, poison
# --------------------------------------------------------------------------


class RecordingRig:
    """A scripted device standing behind the whole runner.

    One ``run_process`` serves every stage launcher; it dispatches on the
    exe name, fabricates each stage's output, and records the order of
    launches plus the lane-lock state observed DURING each launch -- which is
    how the tests prove the lock is held exactly when a device process runs.
    """

    def __init__(self, lock_path: Path) -> None:
        self.lock_path = lock_path
        self.launches: list[str] = []
        self.envs: list[dict[str, str]] = []
        self.lock_seen: list[bool] = []
        self.fail_denoise_with: Exception | None = None

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout: float,
    ) -> CompletedLaunch:
        del timeout
        exe, tag = Path(argv[0]).name, argv[3]
        self.launches.append(f"{exe}:{tag}")
        self.envs.append(dict(env))
        self.lock_seen.append(self.lock_path.exists())
        inputs, outputs = Path(cwd) / "inputs", Path(cwd) / "outputs"
        outputs.mkdir(exist_ok=True)
        if "encoder" in exe:
            assert (inputs / "input0.bin").is_file(), "tokens staged first"
            np.random.RandomState(5).standard_normal((1, 512, 7680)).astype(
                np.float16
            ).tofile(outputs / "encoder.out.bin")
        elif "vae" in exe:
            np.random.RandomState(9).randint(
                0, 256, size=(1, 512, 512, 3), dtype=np.uint8
            ).tofile(outputs / "vae.rgb.bin")
        else:
            if self.fail_denoise_with is not None:
                raise self.fail_denoise_with
            _velocity_fabricator(list(argv), Path(cwd))
        return CompletedLaunch(returncode=0, output="ok: fabricated\n")


def _run_main(
    tmp_path: Path,
    *extra: str,
    rig: RecordingRig | None = None,
    verifier: StageVerifier | None = None,
) -> tuple[int, RecordingRig]:
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = rig or RecordingRig(lock)
    argv = [
        "--dry-run",
        "--prompt",
        "a red bicycle",
        "--out",
        str(tmp_path / "render" / "image.png"),
        "--steps",
        "2",
        "--lane-lock",
        str(lock),
        *extra,
    ]
    return main(argv, run_process=rig, verifier=verifier), rig


def test_one_command_walks_every_stage_in_order(tmp_path: Path) -> None:
    # Explicit per-step: the kill-switch path keeps its canonical
    # stage-ordering pin now that the default is the serve session.
    status, rig = _run_main(tmp_path, "--step-process", "per-step")
    assert status == 0
    assert rig.launches == [
        "npu_flux_encoder_exec.exe:00",
        "npu_flux_whole_graph_exec.exe:00",
        "npu_flux_whole_graph_exec.exe:01",
        "npu_flux_vae_exec.exe:00",
    ], "encoder, then one denoise process per step, then the VAE"
    assert (tmp_path / "render" / "image.png").is_file()
    # The lane was held for every device process and released at exit.
    assert all(rig.lock_seen)
    assert not (tmp_path / "DEVICE_LANE.lock").exists()


def test_dry_run_says_the_image_is_a_placeholder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, _ = _run_main(tmp_path)
    assert status == 0
    assert "placeholder, not a render" in capsys.readouterr().out


def test_a_held_lane_refuses_and_launches_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "DEVICE_LANE.lock").write_text(
        "holder: the lane seat\n", encoding="utf-8"
    )
    status, rig = _run_main(tmp_path)
    assert status == 3
    assert rig.launches == []
    assert "the lane seat" in capsys.readouterr().err
    assert (
        (tmp_path / "DEVICE_LANE.lock")
        .read_text(encoding="utf-8")
        .startswith("holder: the lane seat")
    )


def test_a_denoise_timeout_poisons_the_lane_for_the_fleet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = RecordingRig(lock)
    rig.fail_denoise_with = LaunchTimeoutError(
        "npu_flux_whole_graph_exec.exe did not finish within 1800s"
    )
    status, _ = _run_main(tmp_path, "--step-process", "per-step", rig=rig)
    assert status == 4
    text = lock.read_text(encoding="utf-8")
    assert "POISONED" in text
    assert "reboot" in text.lower()
    assert "reboot" in capsys.readouterr().err.lower()


def test_the_registry_mmap_flag_reaches_every_launch(tmp_path: Path) -> None:
    status, rig = _run_main(
        tmp_path, "--registry-mmap", "--step-process", "per-step"
    )
    assert status == 0
    for env in rig.envs:
        assert env.get("NPU_REGISTRY_MMAP") == "1"
        assert "NPU_REGISTRY_MMAP" in env.get("WSLENV", "")


def test_the_knob_stays_home_without_the_flag(tmp_path: Path) -> None:
    status, rig = _run_main(tmp_path, "--step-process", "per-step")
    assert status == 0
    assert all("NPU_REGISTRY_MMAP" not in env for env in rig.envs)


def test_a_device_run_requires_a_real_tokenizer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--prompt",
            "a red bicycle",
            "--out",
            str(tmp_path / "image.png"),
            "--lane-lock",
            str(tmp_path / "DEVICE_LANE.lock"),
        ]
    )
    assert status == 2
    assert "tokenizer" in capsys.readouterr().err.lower()
    assert not (tmp_path / "DEVICE_LANE.lock").exists(), (
        "a refused run must not have touched the lane"
    )


def test_verification_stops_the_loop_at_the_first_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    verified: list[str] = []

    def verifier(stages: Sequence[str], got_dir: Path) -> None:
        verified.extend(stages)
        if "denoise.step_00.velocity" in stages:
            raise RuntimeError("outside the band")

    status, rig = _run_main(
        tmp_path,
        "--verify-ref",
        str(tmp_path / "recording"),
        "--step-process",
        "per-step",
        verifier=verifier,
    )
    assert status == 1
    assert verified[0] == "tokenize.tokens"
    assert "encoder.out" in verified
    # The refusal fired after step 0; step 1 must never have launched.
    assert "npu_flux_whole_graph_exec.exe:01" not in rig.launches
    assert not (tmp_path / "DEVICE_LANE.lock").exists()
    assert "stop" in capsys.readouterr().err.lower()


def test_explicit_mef_without_a_stage_dir_returns_a_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(
        [
            "--prompt",
            "x",
            "--out",
            str(tmp_path / "image.png"),
            "--tokenizer",
            "unused",
            "--encoder",
            "mef",
            "--lane-lock",
            str(tmp_path / "DEVICE_LANE.lock"),
        ]
    )
    assert status == 2, "a refusal is a return code, never a raise"
    assert "--encoder-stage" in capsys.readouterr().err


def test_a_driver_refusal_stops_with_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = RecordingRig(lock)
    rig.fail_denoise_with = LaunchError("tag 00 exited 2:\nFAIL: refused")
    status, _ = _run_main(tmp_path, "--step-process", "per-step", rig=rig)
    assert status == 1
    assert "FAIL: refused" in capsys.readouterr().err
    assert not lock.exists(), "released on the failure path"


class _ServeRecorder:
    """The popen seam for ``--step-process per-render`` runner tests.

    Wraps the runner's own dry-run serve fabricator so the child walks the
    real sentinel protocol, and records every session launched.
    """

    def __init__(self, child_class: type | None = None) -> None:
        self.children: list[object] = []
        self.child_class = child_class or _DryRunServeChild

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdout,  # noqa: ANN001
        stderr,  # noqa: ANN001
    ) -> object:
        del env, stderr
        child = self.child_class(list(argv), Path(cwd), Path(stdout.name))
        self.children.append(child)
        return child


class _InertServeChild(_DryRunServeChild):
    """Starts up cleanly, then never answers any ``go``."""

    def poll(self) -> int | None:
        return self.returncode


def test_per_render_serves_the_denoise_loop_from_one_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    serve = _ServeRecorder()
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = RecordingRig(lock)
    status = main(
        [
            "--dry-run",
            "--prompt",
            "a red bicycle",
            "--out",
            str(tmp_path / "render" / "image.png"),
            "--steps",
            "2",
            "--lane-lock",
            str(lock),
            "--step-process",
            "per-render",
        ],
        run_process=rig,
        popen=serve,
    )
    assert status == 0
    # The denoise steps never became processes; encoder and VAE still did.
    assert [name for name in rig.launches if "whole_graph" in name] == []
    assert len(serve.children) == 1, "one serve session for the whole loop"
    child = serve.children[0]
    log = Path(child.log_path).read_text(encoding="utf-8")
    assert "ok: serve step 00 done" in log
    assert "ok: serve step 01 done" in log
    assert child.returncode == 0, "the finish sentinel was honoured"
    assert (tmp_path / "render" / "image.png").is_file()
    assert not lock.exists()
    assert "step-process=per-render" in capsys.readouterr().out


def test_per_render_is_the_default_since_the_071_pass(
    tmp_path: Path,
) -> None:
    """The order-071 A/B passed (bit-identical, 26.0 s off the wall), so
    the default flipped: no --step-process flag means ONE serve session."""
    serve = _ServeRecorder()
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = RecordingRig(lock)
    status = main(
        [
            "--dry-run",
            "--prompt",
            "a red bicycle",
            "--out",
            str(tmp_path / "render" / "image.png"),
            "--steps",
            "2",
            "--lane-lock",
            str(lock),
        ],
        run_process=rig,
        popen=serve,
    )
    assert status == 0
    assert len(serve.children) == 1, "the default is the serve session"
    assert [name for name in rig.launches if "whole_graph" in name] == []


def test_the_per_step_kill_switch_spawns_one_process_per_step(
    tmp_path: Path,
) -> None:
    """--step-process per-step restores the shipped path exactly: one
    process per executed step, no session anywhere near the device."""
    serve = _ServeRecorder()
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = RecordingRig(lock)
    status = main(
        [
            "--dry-run",
            "--prompt",
            "a red bicycle",
            "--out",
            str(tmp_path / "render" / "image.png"),
            "--steps",
            "2",
            "--lane-lock",
            str(lock),
            "--step-process",
            "per-step",
        ],
        run_process=rig,
        popen=serve,
    )
    assert status == 0
    assert serve.children == [], "the kill-switch spawns no session"
    assert [name for name in rig.launches if "whole_graph" in name] == [
        "npu_flux_whole_graph_exec.exe:00",
        "npu_flux_whole_graph_exec.exe:01",
    ]


def test_the_session_closes_right_after_the_last_step(tmp_path: Path) -> None:
    """Order 071 measured the session holding ~15.2 GB for 152.7 s past
    the last step, overlapping the VAE. The runner must have ended the
    session BEFORE the VAE process launches."""
    lock = tmp_path / "DEVICE_LANE.lock"
    serve = _ServeRecorder()
    session_state_at_vae: list[int | None] = []

    class VaeObservingRig(RecordingRig):
        def __call__(self, argv, *, cwd, env, timeout):  # noqa: ANN001, ANN201
            if "vae" in Path(argv[0]).name and serve.children:
                session_state_at_vae.append(serve.children[0].returncode)
            return super().__call__(argv, cwd=cwd, env=env, timeout=timeout)

    rig = VaeObservingRig(lock)
    status = main(
        [
            "--dry-run",
            "--prompt",
            "a red bicycle",
            "--out",
            str(tmp_path / "render" / "image.png"),
            "--steps",
            "2",
            "--lane-lock",
            str(lock),
        ],
        run_process=rig,
        popen=serve,
    )
    assert status == 0
    assert session_state_at_vae == [0], (
        "the serve session must have exited (finish honoured) before the "
        "VAE launched; None here means it was still holding the footprint"
    )


def test_dry_run_never_constructs_a_hand_driver(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Order-071 harness defect 2: the hand drivers ignore the run_process
    seam, so `--encoder hand --dry-run` occupied the device for a real
    ~272 s encode labeled dry run. Dry run coerces hand stages to stubs."""
    status, rig = _run_main(tmp_path, "--encoder", "hand", "--vae", "hand")
    assert status == 0
    assert any("encoder_exec" in name for name in rig.launches), (
        "the encoder ran through the fabricated mef stub, not the device"
    )
    assert "coerced to the fabricated stub" in capsys.readouterr().out


def test_a_per_render_step_timeout_poisons_the_lane(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    serve = _ServeRecorder(child_class=_InertServeChild)
    lock = tmp_path / "DEVICE_LANE.lock"
    rig = RecordingRig(lock)
    status = main(
        [
            "--dry-run",
            "--prompt",
            "a red bicycle",
            "--out",
            str(tmp_path / "render" / "image.png"),
            "--steps",
            "2",
            "--lane-lock",
            str(lock),
            "--step-process",
            "per-render",
            "--timeout",
            "0.2",
        ],
        run_process=rig,
        popen=serve,
    )
    assert status == 4
    text = lock.read_text(encoding="utf-8")
    assert "POISONED" in text
    assert "reboot" in capsys.readouterr().err.lower()


def test_mef_vae_forces_identity_batch_norm_stats() -> None:
    weights = PipelineWeights(
        timestep=TimestepEmbedderWeights.random(KLEIN_4B),
        bn_stats=LatentBatchNormStats(
            mean=np.full(128, 0.25, np.float32),
            var=np.full(128, 2.0, np.float32),
        ),
        source="a real manifest",
    )
    force_identity_bn_stats(weights, KLEIN_4B, echo=lambda line: None)
    identity = LatentBatchNormStats.identity(
        KLEIN_4B.in_channels, eps=KLEIN_4B.vae_batch_norm_eps
    )
    assert weights.bn_stats.mean.tobytes() == identity.mean.tobytes()
    assert weights.bn_stats.var.tobytes() == identity.var.tobytes()


def test_the_vae_mef_seam_repacks_and_reads_uint8(tmp_path: Path) -> None:
    final_latent = (
        np.random.RandomState(7)
        .standard_normal((1, 1024, 128))
        .astype(np.float32)
    )
    rgb_u8 = np.random.RandomState(8).randint(
        0, 256, size=(1, 512, 512, 3), dtype=np.uint8
    )

    def fabricate(argv: list[str], cwd: Path) -> None:
        (cwd / "outputs").mkdir(exist_ok=True)
        rgb_u8.tofile(cwd / "outputs" / "vae.rgb.bin")

    driver = FakeDriver(on_launch=fabricate)
    decoder = MefVaeDecoder(
        _launcher(tmp_path, driver, spec=MefStageSpec.vae())
    )
    request = vae_decode_input(
        final_latent,
        _GEOMETRY,
        LatentBatchNormStats.identity(KLEIN_4B.in_channels),
    )
    rgb = decoder(request)

    inputs = tmp_path / "inputs"
    # The graph owns the BN denorm, so the seam re-packs the identity-stats
    # unpacked latent and must recover the raw packed latent exactly.
    assert (
        np.fromfile(inputs / "input0.bin", dtype=np.float16).tobytes()
        == final_latent.astype(np.float16).tobytes()
    )
    # Shape carriers: the lengths are the payload.
    assert np.fromfile(inputs / "input1.bin", dtype=np.float32).shape == (32,)
    assert np.fromfile(inputs / "input2.bin", dtype=np.float32).shape == (32,)

    assert rgb.shape == (1, 3, 512, 512)
    # The device quantized in-graph; the host quantization must reproduce
    # its exact bytes from the float form the seam hands back.
    assert to_rgb_uint8(rgb).tobytes() == rgb_u8.tobytes()
