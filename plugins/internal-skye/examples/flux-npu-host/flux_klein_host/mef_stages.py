"""MEF-backed stage executors: MAX graphs on the NPU, one process per execute.

These are the graph-path counterparts of :mod:`flux_klein_host.device_bridge`'s
hand-orchestrated drivers. Where the bridge marshals a seam into a per-call
scratch directory for a dispatch-loop driver, a MEF stage stages positional
``inputs/inputN[.step_TT].bin`` files into the run package's staging directory
and launches ``npu_flux_whole_graph_exec.exe`` (or a per-stage clone of it)
exactly once per execute -- finding W wedged the CDSP domain on the second
in-process ``M_executeModelSync`` at model scale, so the process boundary IS
the correctness mechanism, not a convenience.

Finding W does not reproduce on current builds (order 037), so
:class:`PersistentStageSession` adds the serve-mode alternative of
``EXECUTOR_DESIGN.md`` §2.2: one process initializes once and serves every
step over nonce-carrying file sentinels. It sits BESIDE
:class:`StageLauncher`, never replacing it -- per-step processes remain the
default and the permanent kill-switch.

The launch mechanics reproduce ``run_whole_graph_denoise.sh``'s proven loop:

* the WSL vsock interop flake (``UtilAcceptVsock ... accept4 failed 110``
  before the exe runs) is detected by its signature -- the flake log without
  any of the driver's own ``ok:`` lines -- and retried, bounded;
* every knob that must reach the Windows process crosses via ONE composed
  ``WSLENV`` (two ``WSLENV`` entries in an environment keep only the last,
  silently dropping the other knob's crossing);
* a timeout is not retried and not absorbed: it poisons the device, so it
  propagates as its own error class for the caller to mark the lane.

Numerics are imported, never re-implemented: the staged bytes come from the
same functions ``whole_graph_prep.py`` stages the recording's inputs with
(:mod:`~flux_klein_host.rope`, :mod:`~flux_klein_host.timestep_embed`,
:mod:`~flux_klein_host.latents`), and the read-back guards are the device
bridge's own.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import KLEIN_4B, KleinConfig
from .device_bridge import BridgeError, _reject_unusable_payload
from .interfaces import (
    DiTBlockStackRequest,
    TextConditioning,
    VaeDecodeInput,
)
from .latents import pack, text_ids
from .rope import packed_freqs_cis
from .stages import TextEncodeRequest
from .timestep_embed import scaled_timestep_argument, sinusoidal_embedding

#: The run package's staging directory, as WSL sees it (RUNBOOK.md).
DEFAULT_STAGE_ROOT = Path("/mnt/c/Users/Public/mojonpu-wholegraph")

#: The guidance value staged for the bind guard's domain check; Klein has
#: ``guidance_embeds=False`` so the model never reads it (whole_graph_prep).
DEFAULT_GUIDANCE = np.float32(3.5)

_VSOCK_ATTEMPTS = 3
_VSOCK_RETRY_SECONDS = 2.0

#: How often a serve-mode wait loop looks at the sentinel files and the
#: child's liveness. Immaterial against a >=180 s step; short enough that a
#: dead child fails the step in well under a second (R2).
_SERVE_POLL_SECONDS = 0.05

#: How long ``close()`` waits for the exe to honour the finish sentinel
#: before killing it. Teardown frees a 7.2 GiB registry; give it time.
_SERVE_FINISH_GRACE_SECONDS = 120.0


class LaunchError(RuntimeError):
    """A stage launch failed: the driver refused, or interop kept flaking."""


class LaunchTimeoutError(RuntimeError):
    """The driver outlived its budget. A timeout POISONS the device: the
    caller must mark the lane and stop, never rerun."""


@dataclass(frozen=True)
class CompletedLaunch:
    """One finished process: its exit status and combined output."""

    returncode: int
    output: str


def _run_windows_process(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: float,
) -> CompletedLaunch:
    """Execs one Windows process through WSL interop and waits.

    The exe is invoked by relative path from the staging directory, exactly
    as the proven run script does -- binfmt interop launches the ``.exe``
    directly, and every path in its argv stays relative to the stage.

    Raises:
        LaunchTimeoutError: If the process outlives ``timeout``.
    """
    try:
        done = subprocess.run(
            argv,
            cwd=str(cwd),
            env=dict(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise LaunchTimeoutError(
            f"{argv[0]} did not finish within {timeout:.0f}s -- the device is "
            "likely wedged. Recovery is a full Windows reboot (human-gated); "
            "do not rerun."
        ) from error
    return CompletedLaunch(returncode=done.returncode, output=done.stdout)


@dataclass(frozen=True)
class MefStageSpec:
    """Which executor and MEF one stage launches, in the proven argv shape.

    ``argv = [./exe, mef, ., <tag>, contract]`` -- the whole-graph driver's
    contract (``npu_flux_whole_graph_exec.c``). The encoder and VAE entries
    name the drivers P1.1/P1.2 will stage beside the denoise one; until those
    land, constructing their backends fails at launch with the exe named,
    and the names are overridable so a differently named driver slots in
    without an edit here.
    """

    exe: str
    mef: str
    contract: str

    @classmethod
    def denoise(cls) -> MefStageSpec:
        """The staged whole-graph denoise driver (proven on device)."""
        return cls(
            exe="npu_flux_whole_graph_exec.exe",
            mef="npu.mef",
            contract="probe11",
        )

    @classmethod
    def encoder(cls) -> MefStageSpec:
        """The P1.1 encoder driver's expected staging (confirm at flip)."""
        return cls(
            exe="npu_flux_encoder_exec.exe",
            mef="encoder.mef",
            contract="encoder2",
        )

    @classmethod
    def vae(cls) -> MefStageSpec:
        """The P1.2 VAE driver's expected staging (confirm at flip)."""
        return cls(
            exe="npu_flux_vae_exec.exe",
            mef="vae.mef",
            contract="vae3",
        )


@dataclass
class StageLauncher:
    """Launches one stage executor per call, with the run script's protocol.

    Args:
        stage_dir: The Windows-visible staging directory holding the exe, the
            MEF, ``weights/``, ``inputs/`` and ``outputs/``.
        spec: Which executor/MEF/contract to launch.
        registry_mmap: Stage the ``NPU_REGISTRY_MMAP=1`` opt-in (#188): the
            Windows loader maps the weight files read-only so back-to-back
            per-step processes share the page cache's physical pages. Off by
            default, matching the run script.
        timeout_seconds: Budget per process; on expiry the device is
            considered poisoned.
        base_env: Environment to compose from; defaults to ``os.environ``.
        run_process: The exec seam; tests and dry runs inject a fake.
        sleep: Injectable wait between vsock retries.
        echo: Progress sink (one line per event).
    """

    stage_dir: Path
    spec: MefStageSpec = field(default_factory=MefStageSpec.denoise)
    registry_mmap: bool = False
    timeout_seconds: float = 1800.0
    base_env: Mapping[str, str] | None = None
    run_process: Callable[..., CompletedLaunch] = _run_windows_process
    sleep: Callable[[float], None] = time.sleep
    echo: Callable[[str], None] = print

    def __post_init__(self) -> None:
        self.stage_dir = Path(self.stage_dir)

    @property
    def inputs_dir(self) -> Path:
        return self.stage_dir / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.stage_dir / "outputs"

    def environment(self) -> dict[str, str]:
        """The launch environment, with one composed ``WSLENV``.

        Every knob that must reach the Windows process is appended to a
        single ``WSLENV`` value -- setting the variable without naming it in
        ``WSLENV`` never reaches the Windows side, and composing twice keeps
        only the last list.
        """
        return _compose_environment(self.base_env, self.registry_mmap)

    def launch(self, tag: str) -> CompletedLaunch:
        """Runs the stage executor once for ``tag``, retrying interop flakes.

        Args:
            tag: The step tag selecting ``inputs/inputN.step_<tag>.bin``.

        Returns:
            The completed launch (exit 0).

        Raises:
            LaunchError: The driver refused, or the vsock flake persisted
                through every attempt.
            LaunchTimeoutError: The driver outlived its budget (poisons).
        """
        argv = [
            f"./{self.spec.exe}",
            self.spec.mef,
            ".",
            tag,
            self.spec.contract,
        ]
        env = self.environment()
        done: CompletedLaunch | None = None
        for attempt in range(1, _VSOCK_ATTEMPTS + 1):
            done = self.run_process(
                argv, cwd=self.stage_dir, env=env, timeout=self.timeout_seconds
            )
            flaked = (
                "UtilAcceptVsock" in done.output and "ok:" not in done.output
            )
            if not flaked:
                break
            if attempt < _VSOCK_ATTEMPTS:
                self.echo(
                    f"  interop flake (vsock) on {self.spec.exe} tag {tag}, "
                    f"retrying in {_VSOCK_RETRY_SECONDS:.0f}s "
                    f"(attempt {attempt}/{_VSOCK_ATTEMPTS})"
                )
                self.sleep(_VSOCK_RETRY_SECONDS)
        else:
            raise LaunchError(
                f"{self.spec.exe} never started: the WSL vsock interop flake "
                f"(UtilAcceptVsock / accept4 failed 110) persisted through "
                f"{_VSOCK_ATTEMPTS} attempts. The interop channel is unwell; "
                "check the WSL session rather than retrying harder."
            )
        assert done is not None
        if done.returncode != 0:
            raise LaunchError(
                f"{self.spec.exe} tag {tag} exited {done.returncode}:\n"
                f"{_tail(done.output)}"
            )
        return done


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _compose_environment(
    base_env: Mapping[str, str] | None, registry_mmap: bool
) -> dict[str, str]:
    """Composes the launch environment with one WSLENV crossing list.

    Every ``MODULAR_NPU_*`` knob present in the environment is named in
    ``WSLENV`` automatically (absolute-path values with ``/p`` so WSL
    translates them) -- order-071 harness defect 1: a knob set but not
    named never reaches the Windows exe, so a device leg can run flag-off
    while reporting success. Naming is centralized here so "set the env
    var" in an order is sufficient again.
    """
    env = dict(os.environ if base_env is None else base_env)
    crossing: list[str] = []
    if registry_mmap:
        env["NPU_REGISTRY_MMAP"] = "1"
        crossing.append("NPU_REGISTRY_MMAP")
    for name in sorted(env):
        if name.startswith("MODULAR_NPU_"):
            crossing.append(
                f"{name}/p" if env[name].startswith("/") else name
            )
    existing = env.get("WSLENV", "")
    already = {
        token.split("/", 1)[0] for token in existing.split(":") if token
    }
    crossing = [
        token
        for token in crossing
        if token.split("/", 1)[0] not in already
    ]
    if crossing:
        env["WSLENV"] = ":".join(([existing] if existing else []) + crossing)
    return env


@dataclass
class PersistentStageSession:
    """One serve-mode process serving every step of a stage.

    The persistent-session half of EXECUTOR_DESIGN.md Phase 0 (P5,
    process-tax line): the exe initializes once -- registry, MEF compile,
    ``M_initModel`` -- and then serves steps over the file-sentinel protocol
    of design §2.2, with the runner keeping the loop, the scheduler, and
    every existing checkpoint instrument. Duck-types the surface
    :class:`MefDenoiseBlockStack` uses (``inputs_dir``, ``outputs_dir``,
    ``launch``), so the stage takes either; :class:`StageLauncher` stays the
    default and the permanent kill-switch fallback (``--step-process
    per-step``).

    The five review-hardened protocol requirements, each with a host test:

    * **R1** every sentinel name carries the session nonce; every stale
      ``*.go``/``*.done``/``serve.*`` is swept before launch; the exe unlinks
      each ``go`` after consuming it.
    * **R2** every wait loop polls the child's liveness and fails the step
      immediately on a dead child, with the exit code and the log tail --
      this is also what makes a stale per-step exe's ``argc`` refusal loud.
    * **R3** the child's stdout/stderr go to a file in the staging dir,
      never a pipe (an undrained pipe blocks the child at ~64 KiB mid-
      render); the runner tails the file for the serve self-report lines.
    * **R4** ``go`` is touched only after every staged input file is closed
      -- callers stage first, then call :meth:`launch`, and every staging
      writer closes before returning.
    * **R5** the exe brackets each execute in ``step_<tag>_start``/
      ``step_<tag>_done`` perf marks; ``npu_perf_summary.py`` renders the
      serve shape from them.

    Args:
        stage_dir: The Windows-visible staging directory (exe, MEF,
            ``weights/``, ``inputs/``, ``outputs/``).
        spec: Which executor/MEF/contract to serve.
        registry_mmap: Stage the ``NPU_REGISTRY_MMAP=1`` opt-in (#188).
        timeout_seconds: Budget per wait (startup and each step); on expiry
            the child is killed and the device is considered poisoned.
        base_env: Environment to compose from; defaults to ``os.environ``.
        popen: The process seam; tests and dry runs inject a fake with the
            ``Popen`` surface this class uses (``poll``/``kill``).
        sleep: Injectable wait between polls.
        clock: Injectable monotonic clock for the timeout arithmetic.
        echo: Progress sink (one line per event).
        nonce: The session nonce (R1); generated unless a test pins it.
    """

    stage_dir: Path
    spec: MefStageSpec = field(default_factory=MefStageSpec.denoise)
    registry_mmap: bool = False
    timeout_seconds: float = 1800.0
    base_env: Mapping[str, str] | None = None
    popen: Callable[..., Any] = subprocess.Popen
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    echo: Callable[[str], None] = print
    nonce: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self) -> None:
        self.stage_dir = Path(self.stage_dir)
        self._child: Any | None = None
        self._log_offset = 0
        self._steps_served = 0
        self._ended = False

    @property
    def inputs_dir(self) -> Path:
        return self.stage_dir / "inputs"

    @property
    def outputs_dir(self) -> Path:
        return self.stage_dir / "outputs"

    @property
    def log_path(self) -> Path:
        return self.outputs_dir / f"serve.{self.nonce}.log"

    def environment(self) -> dict[str, str]:
        """The launch environment; one composed ``WSLENV``, as ever."""
        return _compose_environment(self.base_env, self.registry_mmap)

    def start(self) -> None:
        """Launches the serve process and waits until it reports ready.

        Idempotent: a live session is left alone. The vsock interop flake is
        retried with the per-step launcher's bounds -- one flake window per
        render instead of six is part of what serve mode buys.

        Raises:
            LaunchError: The exe refused the serve argv (a stale per-step
                exe refuses ``argc != 5`` loudly -- restage the exe), died
                during init, the interop flake persisted, or this session
                already ended -- a session is single-use, because a
                relaunch under the SAME nonce would sweep the previous
                child's log, destroying the treatment self-report the
                device order quotes. A fresh session is a fresh nonce.
            LaunchTimeoutError: Startup outlived the budget (poisons).
        """
        if self._child is not None:
            return
        if self._ended:
            raise LaunchError(
                f"serve session {self.nonce} already ended; construct a "
                "fresh PersistentStageSession (fresh nonce) to relaunch"
            )
        self._sweep_stale_sentinels()
        self.inputs_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        argv = [
            f"./{self.spec.exe}",
            self.spec.mef,
            ".",
            "serve",
            self.spec.contract,
            f"serve:{self.nonce}",
        ]
        env = self.environment()
        for attempt in range(1, _VSOCK_ATTEMPTS + 1):
            self.log_path.unlink(missing_ok=True)
            self._log_offset = 0
            # R3: a file, never a pipe. The parent's handle closes right
            # after the exec; the child keeps writing through its own.
            with open(self.log_path, "wb") as log_file:
                child = self.popen(
                    argv,
                    cwd=str(self.stage_dir),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            if self._await_ready(child):
                self._child = child
                return
            text = self._log_text()
            flaked = "UtilAcceptVsock" in text and "ok:" not in text
            if flaked and attempt < _VSOCK_ATTEMPTS:
                self.echo(
                    f"  interop flake (vsock) on {self.spec.exe} serve "
                    f"start, retrying in {_VSOCK_RETRY_SECONDS:.0f}s "
                    f"(attempt {attempt}/{_VSOCK_ATTEMPTS})"
                )
                self.sleep(_VSOCK_RETRY_SECONDS)
                continue
            self._ended = True
            raise LaunchError(
                f"{self.spec.exe} serve session (nonce {self.nonce}) died "
                f"at startup with exit {child.poll()}. A stale per-step exe "
                "refuses the serve argv with its usage line -- if the tail "
                "shows one, restage the exe:\n"
                f"{_tail(text)}"
            )

    def launch(self, tag: str) -> CompletedLaunch:
        """Serves one step: signal ``go``, wait for ``done``.

        The caller stages every input for ``tag`` -- and closes it -- before
        calling (R4); the ``go`` sentinel is the only thing touched here.

        Raises:
            LaunchError: The serve child died mid-step (R2: reported
                immediately with the exit code and the log tail).
            LaunchTimeoutError: The step outlived its budget; the child is
                killed and the device is considered poisoned.
        """
        self.start()
        assert self._child is not None
        done_path = self.outputs_dir / f"step_{tag}.{self.nonce}.done"
        done_path.unlink(missing_ok=True)
        (self.inputs_dir / f"step_{tag}.{self.nonce}.go").touch()
        deadline = self.clock() + self.timeout_seconds
        while True:
            self._echo_new_log_lines()
            if done_path.exists():
                done_path.unlink(missing_ok=True)
                self._steps_served += 1
                return CompletedLaunch(returncode=0, output=self._log_text())
            code = self._child.poll()
            if code is not None:
                self._child = None
                self._ended = True
                raise LaunchError(
                    f"{self.spec.exe} serve session (nonce {self.nonce}) "
                    f"died during step {tag} with exit {code}:\n"
                    f"{_tail(self._log_text())}"
                )
            if self.clock() > deadline:
                self._child.kill()
                self._child = None
                self._ended = True
                raise LaunchTimeoutError(
                    f"{self.spec.exe} serve step {tag} did not finish within "
                    f"{self.timeout_seconds:.0f}s -- the device is likely "
                    "wedged. The serve process was killed. Recovery is a "
                    "full Windows reboot (human-gated); do not rerun."
                )
            self.sleep(_SERVE_POLL_SECONDS)

    def close(self) -> None:
        """Ends the session: finish sentinel, bounded wait, then the axe.

        Safe on a session that never started or already died; never raises
        -- by the time this runs the per-step outputs were already judged.
        A closed session stays closed: :meth:`start` refuses after this.
        """
        self._ended = True
        child, self._child = self._child, None
        if child is None:
            return
        if child.poll() is None:
            (self.inputs_dir / f"serve.{self.nonce}.finish").touch()
            deadline = self.clock() + _SERVE_FINISH_GRACE_SECONDS
            while child.poll() is None and self.clock() < deadline:
                self.sleep(_SERVE_POLL_SECONDS)
            if child.poll() is None:
                child.kill()
                self.echo(
                    f"serve session {self.nonce}: no exit within "
                    f"{_SERVE_FINISH_GRACE_SECONDS:.0f}s of finish; killed"
                )
        self._echo_new_log_lines()
        code = child.poll()
        if code not in (0, None):
            self.echo(
                f"serve session {self.nonce}: exited {code} at teardown "
                f"after {self._steps_served} step(s)"
            )

    def __enter__(self) -> PersistentStageSession:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _await_ready(self, child: Any) -> bool:
        """Waits for the ready sentinel plus the serve self-report line.

        Returns False on a dead child (the caller reads the log to decide
        flake-vs-refusal); raises on timeout, which poisons like any other
        expired device wait.
        """
        ready = self.outputs_dir / f"serve.{self.nonce}.ready"
        deadline = self.clock() + self.timeout_seconds
        while True:
            self._echo_new_log_lines()
            if (
                ready.exists()
                and f"ok: serve mode {self.nonce}" in self._log_text()
            ):
                return True
            if child.poll() is not None:
                return False
            if self.clock() > deadline:
                child.kill()
                self._ended = True
                raise LaunchTimeoutError(
                    f"{self.spec.exe} serve session did not become ready "
                    f"within {self.timeout_seconds:.0f}s -- the device is "
                    "likely wedged. Recovery is a full Windows reboot "
                    "(human-gated); do not rerun."
                )
            self.sleep(_SERVE_POLL_SECONDS)

    def _sweep_stale_sentinels(self) -> None:
        """R1's second arm: no sentinel survives into a new session.

        The staging dir is a fixed, reused path, so a prior render's ``done``
        would satisfy a bare wait instantly and hand back a stale velocity.
        The nonce already defeats that; the sweep keeps the directory from
        accumulating dead sentinels render over render.
        """
        for directory in (self.inputs_dir, self.outputs_dir):
            if not directory.is_dir():
                continue
            for pattern in ("*.go", "*.done", "serve.*"):
                for stale in directory.glob(pattern):
                    stale.unlink(missing_ok=True)
                    self.echo(f"swept stale sentinel {stale.name}")

    def _log_text(self) -> str:
        if not self.log_path.is_file():
            return ""
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def _echo_new_log_lines(self) -> None:
        """Tails the child's log for the serve self-report lines (R3)."""
        if not self.log_path.is_file():
            return
        with open(self.log_path, "rb") as log_file:
            log_file.seek(self._log_offset)
            chunk = log_file.read()
        if not chunk:
            return
        # Hold back a trailing partial line until its newline lands.
        head, sep, _ = chunk.rpartition(b"\n")
        if not sep:
            return
        self._log_offset += len(head) + len(sep)
        for line in head.decode("utf-8", errors="replace").splitlines():
            if line.startswith("ok: serve"):
                self.echo(f"  [serve {self.nonce}] {line}")


def _write_staged(
    directory: Path,
    name: str,
    array: npt.NDArray[Any],
    echo: Callable[[str], None],
) -> None:
    """Writes one staged tensor and echoes its sha256 -- the state-leak
    tripwire the plan requires: per-step bindings name their digest so a
    leak between per-step processes is attributable from the log alone."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = np.ascontiguousarray(array)
    payload.tofile(directory / name)
    digest = hashlib.sha256(payload.tobytes()).hexdigest()
    echo(
        f"staged inputs/{name} sha256={digest} "
        f"({payload.dtype} {list(payload.shape)})"
    )


def _read_staged_output(
    path: Path, dtype: str, shape: tuple[int, ...]
) -> npt.NDArray[Any]:
    """Reads one driver output with the byte count as the cross-check."""
    if not path.is_file():
        raise BridgeError(
            f"the driver exited 0 but wrote no {path.name} -- it died before "
            "its output, or wrote somewhere else"
        )
    values = np.fromfile(path, dtype=np.dtype(dtype).newbyteorder("<"))
    expected = int(np.prod(shape))
    if values.size != expected:
        raise BridgeError(
            f"{path.name} holds {values.size} elements, expected {expected} "
            f"for shape {shape}"
        )
    return np.ascontiguousarray(values.reshape(shape))


class MefDenoiseBlockStack:
    """The denoise loop's device call, through the whole-graph MEF.

    Implements the :class:`~flux_klein_host.interfaces.DiTBlockStack`
    protocol, so :func:`flux_klein_host.denoise.denoise` drives it and the
    Euler update stays with the landed scheduler mirror. Per step it stages
    the step-varying inputs (``input0`` latent, ``input3`` timestep,
    ``input8`` timestep projection), launches ONE process, and reads the
    fp16 velocity back; the invariant inputs (prompt embeddings, ids, rope
    tables, guidance) are staged once, on the first step, from the request
    itself -- the same bytes ``whole_graph_prep.py`` would stage, produced by
    the same imported functions.

    ``teacher_force`` binds ``input0`` per step from a recorded render's
    latents instead of the free-running chain (the P0.2 motion: a file swap,
    nothing else moves), isolating per-step wobble from Euler feedback. A
    step whose recording is absent falls back to the chain, so a partial
    bundle (``latent_init`` only) forces step 0 and free-runs the rest.
    """

    def __init__(
        self,
        launcher: StageLauncher | PersistentStageSession,
        config: KleinConfig = KLEIN_4B,
        *,
        guidance: float = float(DEFAULT_GUIDANCE),
        teacher_force: Path | str | None = None,
        mef_image_seq: int = 1024,
        echo: Callable[[str], None] = print,
    ) -> None:
        """Initializes the stage.

        Args:
            launcher: The per-process launcher or the serve-mode session,
                pointed at the staging dir. Which one is the runner's
                ``--step-process`` kill-switch; this stage stages and reads
                back identically either way.
            config: Model geometry.
            guidance: Staged ``input4``; read only by the bind guard.
            teacher_force: A recorded stage bundle to bind ``input0`` from,
                or ``None`` for the free-running chain. A step whose
                recorded file is absent free-runs from the chain.
            mef_image_seq: The image sequence length the staged MEF was
                compiled at. A request at any other geometry is refused --
                the fix is re-exporting the MEF, not re-staging.
            echo: Progress sink.
        """
        self.launcher = launcher
        self.config = config
        self.guidance = np.float32(guidance)
        self.teacher_force = (
            None if teacher_force is None else Path(teacher_force)
        )
        self.mef_image_seq = mef_image_seq
        self.echo = echo
        self._invariants_staged = False

    def __call__(
        self, request: DiTBlockStackRequest
    ) -> npt.NDArray[np.float32]:
        """Runs one denoise step as one device process.

        Args:
            request: The full block-stack argument bundle.

        Returns:
            ``(B, image_seq_len, 128)`` float32 velocity.

        Raises:
            ValueError: If the request does not match the staged MEF.
            BridgeError: If the velocity is missing, misshapen, or unusable.
            LaunchError, LaunchTimeoutError: From the launch itself.
        """
        request.validate(self.config)
        if request.geometry.image_seq_len != self.mef_image_seq:
            raise ValueError(
                f"the staged MEF is compiled at image_seq "
                f"{self.mef_image_seq}, got a request at "
                f"{request.geometry.image_seq_len}. Re-export the MEF for "
                "this geometry; the staging cannot adapt it."
            )
        if request.latent.shape[0] != 1:
            raise ValueError(
                f"the whole-graph driver runs one image per process, got "
                f"batch {request.latent.shape[0]}"
            )
        if not self._invariants_staged:
            self._stage_invariants(request)
            self._invariants_staged = True

        step = request.conditioning.step_index
        tag = f"{step:02d}"
        self._stage_step(request, tag)

        # A driver that dies without writing must not hand back the previous
        # run's velocity: remove the landing file before the launch.
        landing = self.launcher.outputs_dir / f"step_{tag}.velocity.bin"
        landing.unlink(missing_ok=True)

        self.launcher.launch(tag)

        shape = (1, self.mef_image_seq, self.config.in_channels)
        velocity16 = _read_staged_output(landing, "float16", shape)
        velocity = velocity16.astype(np.float32)
        _reject_unusable_payload("mef_denoise", "velocity", velocity)
        return velocity

    def _stage_invariants(self, request: DiTBlockStackRequest) -> None:
        inputs = self.launcher.inputs_dir

        def write(name: str, array: npt.NDArray[Any]) -> None:
            _write_staged(inputs, name, array, self.echo)

        text = request.text
        write(
            "input1.bin", np.zeros((1, 0, self.config.in_channels), np.float16)
        )
        write("input2.bin", text.prompt_embeds.astype(np.float16))
        write("input4.bin", np.full((1,), self.guidance, np.float32))
        write("input5.bin", request.image_ids.astype(np.int64))
        write("input6.bin", np.zeros((1, 0, 4), np.int64))
        write("input7.bin", text.text_ids.astype(np.int64))
        all_ids = np.concatenate(
            [
                text.text_ids[0].astype(np.float64),
                request.image_ids[0].astype(np.float64),
            ],
            axis=0,
        )
        write("input9.bin", packed_freqs_cis(all_ids).astype(np.float32))
        write(
            "input10.bin",
            np.arange(all_ids.shape[0], dtype=np.uint32)[None, :].copy(),
        )

    def _stage_step(self, request: DiTBlockStackRequest, tag: str) -> None:
        inputs = self.launcher.inputs_dir
        latent = self._step_latent(request)
        _write_staged(
            inputs,
            f"input0.step_{tag}.bin",
            np.ascontiguousarray(latent.astype(np.float16)),
            self.echo,
        )
        timestep = np.full((1,), request.conditioning.timestep, np.float32)
        _write_staged(inputs, f"input3.step_{tag}.bin", timestep, self.echo)
        # `timesteps_proj` carries the argument exactly as the DiT's
        # `Timesteps` block receives it -- fp16-rounded (finding D3, #196).
        _write_staged(
            inputs,
            f"input8.step_{tag}.bin",
            sinusoidal_embedding(
                scaled_timestep_argument(timestep),
                self.config.timestep_guidance_channels,
            ),
            self.echo,
        )

    def _step_latent(
        self, request: DiTBlockStackRequest
    ) -> npt.NDArray[np.float32]:
        if self.teacher_force is None:
            return request.latent
        step = request.conditioning.step_index
        name = (
            "denoise.latent_init.npy"
            if step == 0
            else f"denoise.step_{step - 1:02d}.latent.npy"
        )
        path = self.teacher_force / name
        if not path.is_file():
            # Partial-bundle fallback (293's seam): a fresh seed's bundle
            # carries only latent_init, because later steps cannot exist
            # until the device produces them -- so an absent recording
            # free-runs this step from the chain instead of refusing the
            # whole render.
            self.echo(
                f"teacher forcing: {name} absent, free-running this step"
            )
            return request.latent
        recorded = np.load(path).astype(np.float32)
        if recorded.shape != request.latent.shape:
            raise BridgeError(
                f"teacher-forced latent {name} has shape {recorded.shape}, "
                f"the loop runs at {request.latent.shape}"
            )
        self.echo(f"teacher-forced input0 from {name}")
        return recorded


class MefTextEncoder:
    """The text-encoder seam through its whole-graph MEF (the P1.1 flip).

    Stages the graph's two positional inputs exactly as
    ``Qwen3TextEncoderTransformer.input_types`` declares them at the pinned
    S=512 -- ``input0`` the int64 ``[S]`` token ids, ``input1`` the float32
    ``[1, 1, S, S]`` additive causal+padding bias -- and reads the
    concatenated hidden states back as ``outputs/encoder.out.bin`` (fp16,
    the recording's stage name). The driver itself does not exist until
    P1.1 closes; everything else about this seam is live today, which is
    what makes the flip a staging swap rather than rework.
    """

    def __init__(
        self,
        launcher: StageLauncher,
        config: KleinConfig = KLEIN_4B,
        *,
        tag: str = "00",
        echo: Callable[[str], None] = print,
    ) -> None:
        self.launcher = launcher
        self.config = config
        self.tag = tag
        self.echo = echo

    def __call__(self, request: TextEncodeRequest) -> TextConditioning:
        """Encodes one prompt through the encoder MEF.

        Raises:
            ValueError: If the request is not single-prompt at the pinned S.
            BridgeError: If the output is missing, misshapen, or unusable.
        """
        tokens = request.tokens
        tokens.validate(self.config)
        if request.batch_size != 1:
            raise ValueError(
                f"the encoder MEF is staged for one prompt per process, got "
                f"batch {request.batch_size}"
            )
        seq = self.config.text_seq_len
        inputs = self.launcher.inputs_dir
        _write_staged(
            inputs, "input0.bin", tokens.tokens.astype(np.int64), self.echo
        )
        _write_staged(
            inputs,
            "input1.bin",
            tokens.attention_bias.astype(np.float32),
            self.echo,
        )

        landing = self.launcher.outputs_dir / "encoder.out.bin"
        landing.unlink(missing_ok=True)
        self.launcher.launch(self.tag)

        width = self.config.joint_attention_dim
        hidden = _read_staged_output(
            landing, "float16", (1, seq, width)
        ).astype(np.float32)
        _reject_unusable_payload("mef_encoder", "hidden_states", hidden)
        return TextConditioning(
            prompt_embeds=hidden,
            text_ids=text_ids(seq),
            attention_bias=tokens.attention_bias,
        )


class MefVaeDecoder:
    """The VAE seam through its whole-graph MEF (the P1.2 flip).

    ``PostprocessAndDecode`` owns the batch-norm denorm, the 2x2 unpack, the
    conv stack AND the uint8 quantization in-graph, so this seam consumes the
    RAW packed latent: it re-packs the seam's unpacked input (``pack`` is
    ``unpack``'s exact inverse) and requires the host-side denorm to have
    been the identity -- the runner forces identity statistics when this
    backend is selected, because the real statistics are the graph's own
    weights and applying them twice is the silent plausible-wrong-image
    class. Inputs follow ``input_types``: the fp16 packed latent, then two
    float32 shape carriers whose LENGTHS convey ``latent_h``/``latent_w``.
    The uint8 ``(B, H, W, 3)`` output is handed back through the float
    protocol; ``to_rgb_uint8`` reproduces its exact bytes (``rint`` of the
    inverse map lands on the same integers).
    """

    def __init__(
        self,
        launcher: StageLauncher,
        config: KleinConfig = KLEIN_4B,
        *,
        tag: str = "00",
        echo: Callable[[str], None] = print,
    ) -> None:
        self.launcher = launcher
        self.config = config
        self.tag = tag
        self.echo = echo

    def __call__(self, request: VaeDecodeInput) -> npt.NDArray[np.float32]:
        """Decodes one latent through the VAE MEF.

        Raises:
            ValueError: If the request is not a single image.
            BridgeError: If the output is missing, misshapen, or unusable.
        """
        request.validate(self.config)
        if request.latent.shape[0] != 1:
            raise ValueError(
                f"the VAE MEF is staged for one image per process, got "
                f"batch {request.latent.shape[0]}"
            )
        geometry = request.geometry
        inputs = self.launcher.inputs_dir
        _write_staged(
            inputs,
            "input0.bin",
            pack(request.latent).astype(np.float16),
            self.echo,
        )
        # `latent_h`/`latent_w` in the graph are the TOKEN grid sides -- the
        # packed spatial extents before the in-graph 2x2 unpatchify.
        _write_staged(
            inputs,
            "input1.bin",
            np.zeros((geometry.token_rows,), np.float32),
            self.echo,
        )
        _write_staged(
            inputs,
            "input2.bin",
            np.zeros((geometry.token_cols,), np.float32),
            self.echo,
        )

        landing = self.launcher.outputs_dir / "vae.rgb.bin"
        landing.unlink(missing_ok=True)
        self.launcher.launch(self.tag)

        scale = self.config.vae_scale_factor
        height = geometry.latent_height * scale
        width = geometry.latent_width * scale
        rgb_u8 = _read_staged_output(landing, "uint8", (1, height, width, 3))
        # Back through the seam's float contract; float64 keeps the
        # round-trip through `to_rgb_uint8` exact on every 8-bit value.
        rgb = (rgb_u8.astype(np.float64) / 127.5 - 1.0).transpose(0, 3, 1, 2)
        return np.ascontiguousarray(rgb, dtype=np.float32)


__all__ = [
    "DEFAULT_GUIDANCE",
    "DEFAULT_STAGE_ROOT",
    "CompletedLaunch",
    "LaunchError",
    "LaunchTimeoutError",
    "MefDenoiseBlockStack",
    "MefStageSpec",
    "MefTextEncoder",
    "MefVaeDecoder",
    "PersistentStageSession",
    "StageLauncher",
]
