"""The serve-mode session protocol, proven device-free (EXECUTOR_DESIGN §2.2).

Everything here drives :class:`PersistentStageSession` against a scripted
serve child behind the popen seam. What is asserted is the PROTOCOL -- the
five review-hardened requirements R1-R5, each of which exists because a
specific failure would otherwise read as something else:

* R1 (nonce + sweep): a prior render's sentinel in the reused staging dir
  would satisfy a bare wait instantly and hand back a stale velocity.
* R2 (liveness polling): without it, a dead child -- including a STALE
  per-step exe refusing the serve argv -- degrades into a silent full
  per-step timeout indistinguishable from a wedged device.
* R3 (stdout to a file): a persistent child writing into an undrained pipe
  blocks at ~64 KiB mid-render.
* R4 (go after close): a ``go`` racing its own step's staging binds a
  truncated input.
* R5 (per-step marks) is the C side plus ``npu_perf_summary.py``; its host
  test lives in ``npu_perf_summary_test.py``.

The fake child is interrogated only through ``poll()``, mirroring the real
coupling: the exe acts, the runner only ever observes files and liveness.
"""

from __future__ import annotations

import itertools
import subprocess
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from flux_klein_host.config import KLEIN_4B, resolve_geometry
from flux_klein_host.denoise import denoise
from flux_klein_host.interfaces import TextConditioning
from flux_klein_host.latents import text_ids
from flux_klein_host.mef_stages import (
    LaunchError,
    LaunchTimeoutError,
    MefDenoiseBlockStack,
    PersistentStageSession,
)
from flux_klein_host.weights import PipelineWeights

_GEOMETRY = resolve_geometry(KLEIN_4B, 512, 512)

_USAGE_LINE = (
    "usage: ./npu_flux_whole_graph_exec.exe <flux.mef> <artifact-dir> "
    "<step-tag> whole8|probe11"
)
_FLAKE_LINE = "<3>WSL (1234) ERROR: UtilAcceptVsock:250: accept4 failed 110"


class FakeServeExe:
    """A scripted serve-mode exe behind the popen seam.

    ``behavior`` selects the script:

    * ``"serve"`` -- the healthy protocol: announce, ready, then one ``go``
      consumed per ``poll()`` (velocity derived from the staged ``input0``,
      done touched after), finish honoured with exit 0.
    * ``"stale-exe"`` -- an OLD per-step build: prints its usage line and
      exits 2 without ever serving (the argc refusal).
    * ``"flake"`` -- the vsock interop flake: the exe never ran.
    * ``"inert"`` -- starts up, then never answers any ``go``.
    * ``"die-mid-step"`` -- starts up, then dies the moment a ``go`` lands.
    * ``"foreign-done"`` -- starts up, then answers every ``go`` with a
      nonce-LESS and a foreign-nonce ``done`` but never the real one: the
      stale-writer race only the sentinel names themselves can defeat (R1).
    """

    def __init__(
        self, argv: list[str], cwd: Path, log_path: Path, behavior: str
    ) -> None:
        self.argv = argv
        self.stage_dir = Path(cwd)
        self.log_path = log_path
        self.behavior = behavior
        self.returncode: int | None = None
        self.killed = False
        self.consumed_gos: list[str] = []
        self.nonce = (
            argv[5].split(":", 1)[1]
            if len(argv) > 5 and argv[5].startswith("serve:")
            else None
        )
        if behavior == "stale-exe":
            self._log(_USAGE_LINE)
            self.returncode = 2
            return
        if behavior == "flake":
            self._log(_FLAKE_LINE)
            self.returncode = 1
            return
        (self.stage_dir / "outputs").mkdir(exist_ok=True)
        self._log(f"ok: serve mode {self.nonce}")
        (self.stage_dir / "outputs" / f"serve.{self.nonce}.ready").touch()
        self._log("ok: serve ready")

    def _log(self, line: str) -> None:
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    def poll(self) -> int | None:
        if self.returncode is not None or self.behavior == "inert":
            return self.returncode
        inputs = self.stage_dir / "inputs"
        finish = inputs / f"serve.{self.nonce}.finish"
        if finish.exists():
            finish.unlink()
            self._log("ok: serve finish")
            self.returncode = 0
            return self.returncode
        for go in sorted(inputs.glob(f"step_*.{self.nonce}.go")):
            if self.behavior == "die-mid-step":
                self._log("FAIL: the NSP faulted mid-execute")
                self.returncode = 139
                return self.returncode
            if self.behavior == "foreign-done":
                tag = go.name[len("step_") : -len(f".{self.nonce}.go")]
                outputs = self.stage_dir / "outputs"
                (outputs / f"step_{tag}.done").touch()
                (outputs / f"step_{tag}.deadbeef.done").touch()
                continue
            tag = go.name[len("step_") : -len(f".{self.nonce}.go")]
            go.unlink()
            self.consumed_gos.append(go.name)
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
        self.killed = True
        if self.returncode is None:
            self.returncode = -9


class ServeRig:
    """The popen seam: records every exec, scripts one child per attempt."""

    def __init__(self, behaviors: tuple[str, ...] = ("serve",)) -> None:
        self.behaviors = list(behaviors)
        self.children: list[FakeServeExe] = []
        self.calls: list[dict] = []

    def __call__(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: Mapping[str, str],
        stdout,  # noqa: ANN001
        stderr,  # noqa: ANN001
    ) -> FakeServeExe:
        self.calls.append(
            {
                "argv": list(argv),
                "cwd": Path(cwd),
                "env": dict(env),
                "stdout": stdout,
                "stderr": stderr,
            }
        )
        behavior = self.behaviors.pop(0) if self.behaviors else "serve"
        child = FakeServeExe(argv, Path(cwd), Path(stdout.name), behavior)
        self.children.append(child)
        return child


def _session(
    tmp_path: Path, rig: ServeRig, **kwargs: object
) -> PersistentStageSession:
    """A session with no real sleeping and a per-call ticking clock."""
    ticks = itertools.count()
    clock_reads: list[float] = []
    echoes: list[str] = []

    def clock() -> float:
        clock_reads.append(next(ticks) * 0.01)
        return clock_reads[-1]

    defaults: dict[str, object] = {
        "stage_dir": tmp_path,
        "base_env": {"WSLENV": "PRE_EXISTING/p"},
        "popen": rig,
        "sleep": lambda seconds: None,
        "clock": clock,
        "echo": echoes.append,
    }
    defaults.update(kwargs)
    session = PersistentStageSession(**defaults)  # type: ignore[arg-type]
    session.test_echoes = echoes  # type: ignore[attr-defined]
    session.test_clock_reads = clock_reads  # type: ignore[attr-defined]
    return session


def _stage_step_zero(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    np.zeros((1, 1024, 128), np.float16).tofile(
        inputs / "input0.step_00.bin"
    )


# --------------------------------------------------------------------------
# The serve argv and the knob crossing
# --------------------------------------------------------------------------


def test_the_serve_argv_carries_the_mode_and_the_nonce(tmp_path: Path) -> None:
    rig = ServeRig()
    session = _session(tmp_path, rig, registry_mmap=True)
    session.start()
    call = rig.calls[0]
    assert call["argv"] == [
        "./npu_flux_whole_graph_exec.exe",
        "npu.mef",
        ".",
        "serve",
        "probe11",
        f"serve:{session.nonce}",
    ]
    # The WSLENV composition discipline holds in serve mode too.
    assert call["env"]["NPU_REGISTRY_MMAP"] == "1"
    assert call["env"]["WSLENV"] == "PRE_EXISTING/p:NPU_REGISTRY_MMAP"


def test_the_megastep_flag_crosses_into_wslenv(tmp_path: Path) -> None:
    """MODULAR_NPU_MEGASTEP reaches the serve exe, named in WSLENV.

    The megastep's table builder lives in the exe's dispatch path; the serve
    session's only wiring obligation is the order-071 crossing discipline --
    a knob set but not named in WSLENV never reaches the Windows exe, and the
    device leg then runs flag-off while reporting success. Pinned for this
    flag by name so the device order can quote the test.
    """
    rig = ServeRig()
    session = _session(
        tmp_path,
        rig,
        base_env={
            "WSLENV": "PRE_EXISTING/p",
            "MODULAR_NPU_MEGASTEP": "1",
        },
    )
    session.start()
    call = rig.calls[0]
    assert call["env"]["MODULAR_NPU_MEGASTEP"] == "1"
    assert "MODULAR_NPU_MEGASTEP" in call["env"]["WSLENV"].split(":")


# --------------------------------------------------------------------------
# R1 -- stale-sentinel immunity
# --------------------------------------------------------------------------


def test_the_pre_launch_sweep_removes_stale_sentinels(tmp_path: Path) -> None:
    inputs, outputs = tmp_path / "inputs", tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    stale = [
        inputs / "step_00.deadbeef.go",
        inputs / "serve.deadbeef.finish",
        outputs / "step_00.deadbeef.done",
        outputs / "step_03.done",
        outputs / "serve.deadbeef.log",
    ]
    for path in stale:
        path.touch()
    survivor = outputs / "step_00.velocity.bin"
    survivor.write_bytes(b"\x00\x00")

    session = _session(tmp_path, ServeRig())
    session.start()

    for path in stale:
        assert not path.exists(), f"{path.name} survived the sweep"
    assert survivor.exists(), "the sweep must only eat sentinels"


def test_a_foreign_nonce_done_sentinel_is_ignored(tmp_path: Path) -> None:
    # The foreign dones land DURING the wait, after launch()'s own stale-
    # landing unlink -- so neither the sweep nor the unlink can be what
    # saves us: only the nonce in the wait's own filename can.
    rig = ServeRig(behaviors=("foreign-done",))
    session = _session(tmp_path, rig, timeout_seconds=1.0)

    with pytest.raises(LaunchTimeoutError):
        session.launch("00")
    assert (tmp_path / "outputs" / "step_00.done").exists(), (
        "the foreign writer did fire; the wait just refused it"
    )
    assert rig.children[0].killed


def test_every_sentinel_the_session_touches_carries_the_nonce(
    tmp_path: Path,
) -> None:
    rig = ServeRig()
    session = _session(tmp_path, rig)
    _stage_step_zero(tmp_path)
    session.launch("00")
    session.close()
    child = rig.children[0]
    assert child.consumed_gos == [f"step_00.{session.nonce}.go"]
    # The done sentinel was consumed by the wait; the finish by the child.
    assert not (tmp_path / "outputs" / f"step_00.{session.nonce}.done").exists()
    assert child.returncode == 0


# --------------------------------------------------------------------------
# R2 -- crash detection by process, not timeout
# --------------------------------------------------------------------------


def test_a_dead_child_fails_the_step_fast_with_code_and_tail(
    tmp_path: Path,
) -> None:
    rig = ServeRig(behaviors=("die-mid-step",))
    session = _session(tmp_path, rig, timeout_seconds=1800.0)
    _stage_step_zero(tmp_path)
    with pytest.raises(LaunchError, match="exit 139") as caught:
        session.launch("00")
    assert "NSP faulted" in str(caught.value), "the log tail travels"
    # The fake clock ticks 0.01 per read; sitting out the 1800 s budget
    # would take 180,000 reads. A handful means the death was SEEN (R2),
    # not timed out.
    reads = session.test_clock_reads  # type: ignore[attr-defined]
    assert len(reads) < 50, f"{len(reads)} clock reads -- the wait sat there"


def test_a_stale_exe_refuses_the_serve_argv_loudly(tmp_path: Path) -> None:
    """The treatment-delivery marker: an old exe usage-refuses argc 6."""
    rig = ServeRig(behaviors=("stale-exe", "stale-exe", "stale-exe"))
    session = _session(tmp_path, rig)
    with pytest.raises(LaunchError, match="stale per-step exe") as caught:
        session.start()
    assert "usage:" in str(caught.value), "the refusal's own words travel"
    assert "exit 2" in str(caught.value)
    # A refusal is not a flake: no retries burned.
    assert len(rig.calls) == 1


def test_a_vsock_flake_at_session_start_is_retried(tmp_path: Path) -> None:
    rig = ServeRig(behaviors=("flake", "serve"))
    session = _session(tmp_path, rig)
    session.start()
    assert len(rig.calls) == 2
    assert rig.children[1].nonce == session.nonce


def test_a_persistent_vsock_flake_fails_after_three(tmp_path: Path) -> None:
    rig = ServeRig(behaviors=("flake", "flake", "flake"))
    with pytest.raises(LaunchError, match="exit 1"):
        _session(tmp_path, rig).start()
    assert len(rig.calls) == 3


# --------------------------------------------------------------------------
# R3 -- stdout to a file, tailed for the self-report lines
# --------------------------------------------------------------------------


def test_child_output_goes_to_a_log_file_never_a_pipe(tmp_path: Path) -> None:
    rig = ServeRig()
    session = _session(tmp_path, rig)
    session.start()
    call = rig.calls[0]
    assert call["stdout"] is not subprocess.PIPE
    assert Path(call["stdout"].name) == session.log_path
    assert session.log_path.parent == tmp_path / "outputs"
    assert call["stderr"] == subprocess.STDOUT


def test_the_runner_tails_the_serve_self_report_lines(tmp_path: Path) -> None:
    rig = ServeRig()
    session = _session(tmp_path, rig)
    session.start()
    echoes = "\n".join(session.test_echoes)  # type: ignore[attr-defined]
    assert f"ok: serve mode {session.nonce}" in echoes
    assert "ok: serve ready" in echoes


# --------------------------------------------------------------------------
# R4 -- go only after the staged inputs are closed, and the loop composes
# --------------------------------------------------------------------------


def test_one_serve_process_serves_every_step_of_the_loop(
    tmp_path: Path,
) -> None:
    """The whole point, end to end: the denoise loop -- scheduler, Euler
    update, checkpoints all host-side -- rides ONE process, and each step's
    velocity is a function of the input0 staged for THAT step, which is
    only possible if every ``go`` followed its own staging (R4)."""
    rig = ServeRig()
    session = _session(tmp_path, rig)
    stack = MefDenoiseBlockStack(session, echo=lambda line: None)
    rng = np.random.RandomState(3)
    text = TextConditioning(
        prompt_embeds=rng.standard_normal((1, 512, 7680)).astype(np.float32),
        text_ids=text_ids(512),
    )

    result = denoise(
        stack,
        text,
        _GEOMETRY,
        PipelineWeights.synthetic(KLEIN_4B).timestep,
        num_steps=2,
        seed=0,
        keep_records=True,
    )
    session.close()

    assert len(rig.children) == 1, "one process for the whole loop"
    child = rig.children[0]
    assert child.consumed_gos == [
        f"step_00.{session.nonce}.go",
        f"step_01.{session.nonce}.go",
    ]
    assert child.returncode == 0, "finish honoured at close"

    # Each velocity is derived from that step's staged latent: the loop
    # consumed step-dependent answers, not placeholders or stale bytes.
    for step, record in enumerate(result.records):
        staged = np.fromfile(
            tmp_path / "inputs" / f"input0.step_{step:02d}.bin",
            dtype=np.float16,
        ).reshape(1, 1024, 128)
        assert (
            record.velocity.tobytes()
            == (staged * np.float16(-0.5)).astype(np.float32).tobytes()
        )


def test_a_timeout_kills_the_serve_child_and_propagates(
    tmp_path: Path,
) -> None:
    rig = ServeRig(behaviors=("inert",))
    session = _session(tmp_path, rig, timeout_seconds=0.5)
    _stage_step_zero(tmp_path)
    with pytest.raises(LaunchTimeoutError, match="wedged"):
        session.launch("00")
    assert rig.children[0].killed
    # A poisoning error must not be followed by more device traffic.
    session.close()
    assert len(rig.calls) == 1


def test_close_is_safe_and_final_even_before_start(tmp_path: Path) -> None:
    rig = ServeRig()
    session = _session(tmp_path, rig)
    session.close()
    assert rig.calls == []
    with pytest.raises(LaunchError, match="already ended"):
        session.launch("00")
    assert rig.calls == [], "a closed session launches nothing"


def test_a_dead_session_refuses_relaunch_and_keeps_the_log(
    tmp_path: Path,
) -> None:
    """A retrying caller must not respawn under the SAME nonce: start()
    sweeps and re-creates ``serve.<nonce>.log``, which would destroy the
    treatment self-report the device order quotes. Dead session = dead
    nonce; a fresh session is a fresh nonce."""
    rig = ServeRig(behaviors=("die-mid-step",))
    session = _session(tmp_path, rig)
    _stage_step_zero(tmp_path)
    with pytest.raises(LaunchError, match="exit 139"):
        session.launch("00")

    with pytest.raises(LaunchError, match="already ended"):
        session.launch("01")
    assert len(rig.calls) == 1, "no silent respawn"
    log = session.log_path.read_text(encoding="utf-8")
    assert f"ok: serve mode {session.nonce}" in log, (
        "the dead child's self-report survives for the record"
    )
    session.close()
    assert not (
        tmp_path / "inputs" / f"serve.{session.nonce}.finish"
    ).exists(), "no finish sentinel for a child already dead"


def test_a_closed_session_refuses_relaunch_fresh_session_fresh_nonce(
    tmp_path: Path,
) -> None:
    rig = ServeRig()
    session = _session(tmp_path, rig)
    _stage_step_zero(tmp_path)
    session.launch("00")
    session.close()

    with pytest.raises(LaunchError, match="already ended"):
        session.launch("01")

    replacement = _session(tmp_path, rig)
    assert replacement.nonce != session.nonce
    replacement.start()
    assert len(rig.calls) == 2
