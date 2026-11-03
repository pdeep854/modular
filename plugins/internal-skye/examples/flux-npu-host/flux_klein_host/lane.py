"""The device-lane lock, as `DEVICE_LANE.md` prescribes it.

The Hexagon NPU is a single resource and concurrent claims corrupt each
other's results, so every run that will touch the device claims
``.agentwork/flux-npu/DEVICE_LANE.lock`` first. This module is that
discipline as a context manager: the atomic claim with the five fields the
protocol requires (holder, running, claimed, ``expect:``, ``yield:``), the
polite refusal naming whoever holds it, the physical `tasklist.exe` check
(never WSL ``ps`` -- WSL sees only the Linux wrapper while the Windows
``.exe`` survives and holds the lane), and the poison path: a timeout marks
the lock POISONED for the fleet instead of releasing it, because recovery is
a full Windows reboot and human-gated.

The reference implementation this mirrors is the claim/release/poison flow of
``run_whole_graph_denoise.sh``; the tests drive it with an injected busy
check so the discipline is proven on a machine with no Windows interop.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

#: The recovery instruction, verbatim where it matters: a poisoned lane is
#: NOT cleared by `pnputil /restart-device` (measured 2026-08-06); only a
#: full Windows reboot recovers it, and only a three-leg gate case proves it.
_RECOVERY = (
    "recovery: full Windows reboot -- human-gated "
    "(pnputil /restart-device measured NOT to clear the wedge)"
)


class LaneBusyError(RuntimeError):
    """The lane is held or physically busy; the run must not proceed."""


class LanePoisonedError(RuntimeError):
    """The lane is marked poisoned; recovery is human-gated."""


def _windows_npu_process() -> str | None:
    """Returns a running ``npu_*.exe`` name, checked via ``tasklist.exe``.

    WSL ``ps`` cannot see a surviving Windows process, so the check crosses
    to the Windows side. A machine without interop (no ``tasklist.exe``)
    reports free, matching the run script's ``2>/dev/null`` behavior.
    """
    try:
        listing = subprocess.run(
            ["tasklist.exe"], capture_output=True, text=True, timeout=30.0
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for line in listing.splitlines():
        if line.lower().startswith("npu_") and ".exe" in line.lower():
            return line.split()[0]
    return None


class LaneLock:
    """One claim of the device lane, released on every exit path.

    Args:
        path: The lock file, ``.agentwork/flux-npu/DEVICE_LANE.lock``.
        holder: Who is claiming, recorded for whoever finds the lane held.
        running: What the claim is for.
        expect: The protocol's duration estimate ("unknown" is acceptable;
            a missing line is not, so it is a required argument).
        yield_at: Where another agent could safely take over, or ``"none"``.
        busy_check: Returns the name of a Windows-side ``npu_*.exe`` still
            running, or ``None`` when the lane is physically free. Defaults
            to the ``tasklist.exe`` probe; tests inject their own.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        holder: str,
        running: str,
        expect: str,
        yield_at: str = "none",
        busy_check: Callable[[], str | None] = _windows_npu_process,
    ) -> None:
        self.path = Path(path)
        self.holder = holder
        self.running = running
        self.expect = expect
        self.yield_at = yield_at
        self.busy_check = busy_check
        self._held = False
        self._poisoned = False

    def claim(self) -> None:
        """Atomically claims the lane, or raises without touching it.

        Raises:
            LanePoisonedError: The lock is a poison marker; never cleared here.
            LaneBusyError: Held by someone else, or a Windows-side
                ``npu_*.exe`` is still running.
        """
        if self.path.exists():
            text = self.path.read_text(encoding="utf-8")
            if "POISONED" in text:
                raise LanePoisonedError(
                    f"the device lane is POISONED -- not proceeding.\n{text}\n"
                    f"{_RECOVERY}"
                )
            raise LaneBusyError(
                f"the device lane is held -- not proceeding. Wait for the "
                f"holder to release it (poll the lock; do not delete "
                f"it):\n{text}"
            )
        busy = self.busy_check()
        if busy is not None:
            raise LaneBusyError(
                f"the device lane is physically busy: {busy} is running on "
                "the Windows side (tasklist.exe). A WSL wrapper may have "
                "exited while the .exe survives -- wait for it."
            )
        claim = (
            f"holder: {self.holder}\n"
            f"running: {self.running}\n"
            f"claimed: {_utc_now()}\n"
            f"expect: {self.expect}\n"
            f"yield: {self.yield_at}\n"
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(
                self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644
            )
        except FileExistsError as error:
            # Lost the race between the existence check and the create.
            text = self.path.read_text(encoding="utf-8")
            raise LaneBusyError(
                f"the device lane was claimed while checking it:\n{text}"
            ) from error
        with os.fdopen(handle, "w", encoding="utf-8") as lock_file:
            lock_file.write(claim)
        self._held = True

    def poison(self, why: str) -> None:
        """Marks the lane poisoned for the fleet instead of releasing it.

        Called on a device timeout: the wedge outlives this process, so the
        lock is rewritten as a poison marker that :meth:`release` will not
        delete, exactly as the run script does.

        Args:
            why: What happened, recorded in the marker.
        """
        self.path.write_text(
            f"holder: POISONED -- {why}; do NOT claim\n"
            f"since: {_utc_now()}\n"
            f"{_RECOVERY}\n",
            encoding="utf-8",
        )
        self._poisoned = True

    def release(self) -> None:
        """Deletes the claim unless the lane was poisoned mid-run."""
        if self._held and not self._poisoned:
            self.path.unlink(missing_ok=True)
        self._held = False

    def __enter__(self) -> LaneLock:
        self.claim()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


__all__ = ["LaneBusyError", "LaneLock", "LanePoisonedError"]
