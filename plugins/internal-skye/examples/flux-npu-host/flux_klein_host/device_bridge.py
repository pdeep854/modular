"""The device-dispatch bridge: driving the real NPU driver from the host loop.

The host pipeline is Python on WSL Linux. The NPU driver is a Windows ARM64
``.exe``, because FastRPC to the Hexagon NSP only exists on the Windows side.
Nothing links the two and no RPC runs between them, so a seam crossing is
marshalled through the one thing both sides can see -- the shared filesystem --
and the call itself is a Windows-interop process exec.

One crossing is one *scratch directory*:

.. code-block:: text

    <scratch>/request.json     what follows, written here
    <scratch>/*.bin            one raw C-order little-endian array per tensor
    <scratch>/<output>.bin     written by the driver
    <scratch>/response.json    written by the driver, last

:class:`DeviceBridge` implements that crossing for *any* seam;
:class:`DeviceBlockStack` is the :class:`~flux_klein_host.interfaces.
DiTBlockStack` seam on top of it. The encoder and VAE seams are the same
marshalling with different tensor names -- see ``PIPELINE_IO.md`` for the wire
contract and the two worked examples.

Three properties are load-bearing and are the reason this is not just
``subprocess.run``:

1. **The manifest is self-describing.** Every tensor carries its dtype and
   shape, so a host/driver disagreement fails at load with both shapes named,
   rather than as reinterpreted bytes that produce a plausible wrong image.
2. **float32 crosses the seam.** ``interfaces.py`` fixes host-side arrays as
   float32 and leaves the fp16 cast to the device; the driver does the cast.
3. **``response.json`` is the verdict, not the exit code.** It is written last,
   so its absence means the driver died mid-run, and its ``<output>_source``
   says what produced the numbers. A ``--plan-only`` run writes a correctly
   shaped placeholder, and this module refuses to hand one back as a result
   unless the caller asked for exactly that.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .config import KLEIN_4B, KleinConfig, LatentGeometry
from .interfaces import (
    DiTBlockStackRequest,
    StepModulation,
    TextConditioning,
    VaeDecodeInput,
)
from .latents import text_ids
from .stages import TextEncodeRequest

#: Where ``build_devctx_dispatch.sh`` stages a driver, as WSL sees it. Also the
#: default scratch root: it is a real Windows directory, and a Windows process
#: cannot ``cd`` into the ``\\wsl.localhost`` UNC path a Linux-side directory
#: maps to.
DEFAULT_STAGE_DIR = Path("/mnt/c/Users/Public/mojonpu")

#: Marker the interop wrapper echoes so the driver's true exit status survives.
#: A Windows exit code does not reliably reach a WSL caller through ``cmd.exe``,
#: so the code is read out of stdout and the ``subprocess`` status is only a
#: cross-check.
EXIT_MARKER = "__NPU_EXIT__="

#: Order the shared adaLN vectors are packed in, matching ``unpackModulation``
#: in ``npu_block_stack_dispatch.cpp``. Each triple is (shift, scale, gate).
MODULATION_ORDER = (
    "double_img_attn",
    "double_img_mlp",
    "double_txt_attn",
    "double_txt_mlp",
    "single",
)


class BridgeError(RuntimeError):
    """A seam crossing failed: marshalling, the exec, or the driver itself."""


def _reject_unusable_payload(
    seam: str, name: str, value: npt.NDArray[Any]
) -> None:
    """Refuses a payload that cannot be a computation, whatever the driver said.

    ``output_source`` is the driver's own word for what it produced, so it
    cannot catch a driver that believes it succeeded. Two properties can be
    checked here without one: a tensor carrying non-finite elements is not a
    result, and a floating tensor of ONE repeated bit pattern was not computed
    from varying inputs.

    Both fired on the same render and neither was checked: a whole denoise step
    came back as 131072 identical NaNs tagged ``velocity_source: device``, and
    the loop integrated it into the latent and kept going for 40 minutes.

    Raises:
        BridgeError: If the payload is non-finite or single-valued.
    """
    if not np.issubdtype(value.dtype, np.floating) or value.size == 0:
        return
    if not np.isfinite(value).all():
        bad = int((~np.isfinite(value)).sum())
        raise BridgeError(
            f"{seam}: output {name!r} has {bad} of {value.size} non-finite "
            "elements. The driver reported success, so this is a device "
            "result that overflowed or was never computed -- refusing to "
            "integrate it."
        )
    if np.unique(value.view(_uint_view(value.dtype))).size == 1:
        raise BridgeError(
            f"{seam}: output {name!r} is {value.size} copies of a single bit "
            "pattern, which no forward pass produces. Refusing to integrate a "
            "buffer the device never meaningfully wrote."
        )


def _uint_view(dtype: np.dtype[Any]) -> str:
    """The same-width unsigned type, so bit patterns compare exactly."""
    return f"u{dtype.itemsize}"


def to_windows_path(path: Path) -> str:
    """Translates a WSL path to the Windows path a driver argument needs.

    Args:
        path: A path as WSL sees it.

    Returns:
        The Windows spelling, e.g. ``C:\\Users\\Public\\mojonpu``.

    Raises:
        BridgeError: If ``wslpath`` is unavailable or fails.
    """
    if shutil.which("wslpath") is None:
        raise BridgeError(
            "wslpath not found -- the bridge needs WSL interop to reach the "
            "Windows driver"
        )
    done = subprocess.run(
        ["wslpath", "-w", str(path)], capture_output=True, text=True
    )
    if done.returncode != 0:
        raise BridgeError(f"wslpath -w {path} failed: {done.stderr.strip()}")
    return done.stdout.strip()


#: numpy dtype names the wire format carries, and the manifest spelling of each.
_WIRE_DTYPES = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float16): "float16",
    np.dtype(np.int64): "int64",
    np.dtype(np.int32): "int32",
}


@dataclass(frozen=True)
class BridgeResponse:
    """What the driver reported, parsed from ``response.json``."""

    status: str
    plan_only: bool
    output_source: str
    """``device`` | ``host_reference`` | ``none``. ``none`` means the output
    ``.bin`` is a correctly shaped placeholder, not a result.

    Read out of ``<output>_source``, which is named after the seam's own output
    -- ``velocity_source`` for the DiT stack, ``hidden_states_source`` for the
    encoder, ``rgb_source`` for the VAE."""

    weight_source: str
    modulation_source: str
    """``request`` (the host's adaLN parameters) or ``synthetic``."""

    detail: str
    raw: Mapping[str, Any]

    @property
    def computed(self) -> bool:
        """Whether the outputs hold real numbers rather than a placeholder."""
        return self.output_source != "none"


@dataclass
class DeviceBridge:
    """Marshals one seam crossing to a Windows-side driver and back.

    Args:
        executable: The driver ``.exe``, as WSL sees it.
        scratch_root: Where per-call scratch directories are made. Must be
            Windows-visible; see :data:`DEFAULT_STAGE_DIR`.
        extra_args: Driver arguments added to every call (``--weights ...``).
        plan_only: Add ``--plan-only``. The driver then returns before it opens
            the device, so the crossing is exercised with no NPU -- and the
            outputs are placeholders, which :meth:`invoke` will not let a caller
            mistake for results.
        keep_scratch: Leave each call's directory behind for inspection.
        timeout_seconds: Cap on one driver run.
        interop: Reach the driver through Windows interop. Defaults to whether
            the executable is a ``.exe``. ``False`` runs it as a native process,
            which is how the host-only build of the same driver
            (``npu_block_stack_reference``) drives this identical protocol on a
            machine with neither Windows nor an NPU.
    """

    executable: Path
    scratch_root: Path = DEFAULT_STAGE_DIR
    extra_args: tuple[str, ...] = ()
    plan_only: bool = False
    keep_scratch: bool = False
    timeout_seconds: float = 3600.0
    interop: bool | None = None

    calls: int = field(default=0, init=False)
    last_seconds: float = field(default=0.0, init=False)
    last_scratch: Path | None = field(default=None, init=False)
    last_response: BridgeResponse | None = field(default=None, init=False)
    last_stdout: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.executable = Path(self.executable)
        self.scratch_root = Path(self.scratch_root)
        if self.interop is None:
            self.interop = self.executable.suffix.lower() == ".exe"
        if not self.executable.is_file():
            raise BridgeError(
                f"no driver at {self.executable} -- build and stage it with "
                "`build_devctx_dispatch.sh <driver>`"
            )

    def invoke(
        self,
        seam: str,
        geometry: Mapping[str, int],
        inputs: Mapping[str, npt.NDArray[Any]],
        outputs: Mapping[str, tuple[tuple[int, ...], str]],
        *,
        require_computed: bool = True,
    ) -> tuple[dict[str, npt.NDArray[Any]], BridgeResponse]:
        """Runs one crossing: marshal in, exec the driver, read the result.

        Args:
            seam: Seam name recorded in the manifest, e.g. ``dit_block_stack``.
            geometry: Integer geometry the driver configures itself from.
            inputs: Host arrays by manifest name. Written C-order; a
                non-contiguous or non-native-endian array is copied first.
            outputs: Expected shape and wire dtype per output name.
            require_computed: Reject a response whose outputs are placeholders.
                Only a deliberate plan-only run should pass ``False``.

        Returns:
            The outputs by name, and the parsed :class:`BridgeResponse`.

        Raises:
            BridgeError: On any marshalling, exec, or driver failure.
        """
        scratch = self._make_scratch(seam)
        self.last_scratch = scratch
        try:
            self._write_request(scratch, seam, geometry, inputs, outputs)
            self._exec(scratch)
            response = self._read_response(scratch, tuple(outputs))
            self.last_response = response
            if response.status != "ok":
                raise BridgeError(
                    f"{seam}: driver reported status {response.status!r}: "
                    f"{response.detail}"
                )
            if require_computed and not response.computed:
                raise BridgeError(
                    f"{seam}: the driver wrote placeholders "
                    f"(output_source={response.output_source!r}, "
                    f"{response.detail}). A plan-only run proves the bridge, "
                    "not the model -- construct the backend with "
                    "plan_only=True if that is what you meant."
                )
            values = {
                name: self._read_output(scratch, name, shape, dtype)
                for name, (shape, dtype) in outputs.items()
            }
            if require_computed:
                for name, value in values.items():
                    _reject_unusable_payload(seam, name, value)
            self.calls += 1
            return values, response
        finally:
            if not self.keep_scratch:
                shutil.rmtree(scratch, ignore_errors=True)

    def _make_scratch(self, seam: str) -> Path:
        try:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise BridgeError(
                f"cannot create the scratch root {self.scratch_root}: {error}"
            ) from error
        # Named rather than random: a kept scratch directory should say which
        # call it came from when several are left behind.
        scratch = self.scratch_root / f"io-{seam}-{self.calls:04d}"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        return scratch

    def _write_request(
        self,
        scratch: Path,
        seam: str,
        geometry: Mapping[str, int],
        inputs: Mapping[str, npt.NDArray[Any]],
        outputs: Mapping[str, tuple[tuple[int, ...], str]],
    ) -> None:
        manifest: dict[str, Any] = {
            "format": "skye-npu-io",
            "version": 1,
            "seam": seam,
            "geometry": {key: int(value) for key, value in geometry.items()},
            "inputs": {},
            "outputs": {},
        }
        for name, array in inputs.items():
            wire = _WIRE_DTYPES.get(array.dtype)
            if wire is None:
                raise BridgeError(
                    f"{seam}: input {name!r} has dtype {array.dtype}, which "
                    f"the bridge does not carry ({sorted(_WIRE_DTYPES.values())})"
                )
            # `tofile` writes native byte order and would silently emit
            # big-endian bytes on a big-endian host; the driver reads
            # little-endian, so make that explicit rather than assume the host.
            payload = np.ascontiguousarray(array).astype(
                array.dtype.newbyteorder("<"), copy=False
            )
            payload.tofile(scratch / f"{name}.bin")
            manifest["inputs"][name] = {
                "path": f"{name}.bin",
                "dtype": wire,
                "shape": [int(extent) for extent in array.shape],
            }
        for name, (shape, dtype) in outputs.items():
            manifest["outputs"][name] = {
                "path": f"{name}.bin",
                "dtype": dtype,
                "shape": [int(extent) for extent in shape],
            }
        (scratch / "request.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    def _write_runner(self, scratch: Path) -> Path:
        """Writes the batch file the interop exec runs.

        The command goes in a file rather than on the ``cmd.exe`` command line
        because WSL interop rewrites a Linux argv into a Windows command line
        and escapes every embedded quote on the way, so a quoted path arrives as
        a literal ``\\"C:\\...\\"`` and is not recognized as a program. A batch
        file crosses as one unquoted path and quotes its own arguments, which
        also leaves an exact record of the call for anyone reproducing it by
        hand.
        """
        exe_windows = to_windows_path(self.executable)
        scratch_windows = to_windows_path(scratch)
        if scratch_windows.startswith("\\\\"):
            raise BridgeError(
                f"the scratch directory maps to the UNC path "
                f"{scratch_windows}. Put it on a drive the Windows side owns "
                f"(default {DEFAULT_STAGE_DIR}); a Linux-side directory "
                "reaches Windows only over the slow 9P share, and cmd.exe "
                "cannot make one its working directory."
            )
        arguments = " ".join(
            ["--io", f'"{scratch_windows}"']
            + list(self.extra_args)
            + (["--plan-only"] if self.plan_only else [])
        )
        runner = scratch / "run.cmd"
        # Delayed expansion, and the status echoed into stdout as well as
        # returned: a Windows exit code does not reliably survive the trip back
        # to a WSL caller, so the echoed one is what the bridge believes.
        runner.write_bytes(
            "\r\n".join(
                [
                    "@echo off",
                    "setlocal enabledelayedexpansion",
                    f'"{exe_windows}" {arguments}',
                    "set NPU_STATUS=!ERRORLEVEL!",
                    f"echo {EXIT_MARKER}!NPU_STATUS!",
                    "exit /b !NPU_STATUS!",
                    "",
                ]
            ).encode("ascii")
        )
        return runner

    def _native_command(self, scratch: Path) -> list[str]:
        """The same call as :meth:`_write_runner`, run without cmd.exe."""
        return (
            [str(self.executable), "--io", str(scratch)]
            # The interop path quotes its arguments for cmd.exe; a native argv
            # must not carry those quotes into the argument itself.
            + [argument.replace('"', "") for argument in self.extra_args]
            + (["--plan-only"] if self.plan_only else [])
        )

    def _exec(self, scratch: Path) -> None:
        """Runs the driver against a scratch directory and checks its status."""
        if self.interop:
            command = [
                "cmd.exe",
                "/c",
                to_windows_path(self._write_runner(scratch)),
            ]
        else:
            command = self._native_command(scratch)
        started = time.perf_counter()
        try:
            done = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                # cmd.exe inherits the caller's directory, and a WSL cwd is a
                # UNC path it refuses; start it somewhere it can stand.
                cwd=str(self.executable.parent),
            )
        except FileNotFoundError as error:
            raise BridgeError(
                "cmd.exe not found -- Windows interop is off, so the bridge "
                "cannot reach the driver"
                if self.interop
                else f"no such executable: {self.executable}"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise BridgeError(
                f"the driver did not finish within {self.timeout_seconds}s"
            ) from error
        self.last_seconds = time.perf_counter() - started
        self.last_stdout = done.stdout

        status = done.returncode
        if self.interop:
            echoed = _parse_exit(done.stdout)
            if echoed is None:
                raise BridgeError(
                    "the interop wrapper printed no exit marker, so the "
                    f"driver's status is unknown:\n{_tail(done.stdout)}\n"
                    f"{_tail(done.stderr)}"
                )
            status = echoed
        if status != 0:
            raise BridgeError(
                f"the driver exited {status}:\n{_tail(done.stdout)}\n"
                f"{_tail(done.stderr)}"
            )

    def _read_response(
        self, scratch: Path, outputs: tuple[str, ...]
    ) -> BridgeResponse:
        path = scratch / "response.json"
        if not path.is_file():
            raise BridgeError(
                "the driver exited 0 but wrote no response.json, so it died "
                f"before it finished:\n{_tail(self.last_stdout)}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format") != "skye-npu-io-response":
            raise BridgeError(f"{path} is not a skye-npu-io response")
        # Named after the seam's output, so a driver that answered about a
        # different tensor than the one asked for reads as a placeholder rather
        # than being silently accepted.
        sources = [
            str(payload[f"{name}_source"])
            for name in outputs
            if f"{name}_source" in payload
        ]
        return BridgeResponse(
            status=str(payload.get("status", "")),
            plan_only=bool(payload.get("plan_only", False)),
            output_source=sources[0] if len(sources) == 1 else "none",
            weight_source=str(payload.get("weight_source", "unknown")),
            modulation_source=str(payload.get("modulation_source", "unknown")),
            detail=str(payload.get("detail", "")),
            raw=payload,
        )

    def _read_output(
        self, scratch: Path, name: str, shape: tuple[int, ...], dtype: str
    ) -> npt.NDArray[Any]:
        path = scratch / f"{name}.bin"
        if not path.is_file():
            raise BridgeError(f"the driver wrote no {name}.bin")
        values = np.fromfile(path, dtype=np.dtype(dtype).newbyteorder("<"))
        expected = int(np.prod(shape))
        if values.size != expected:
            raise BridgeError(_dtype_mismatch(path, name, shape, dtype))
        return np.ascontiguousarray(values.reshape(shape))


def _dtype_mismatch(
    path: Path, name: str, shape: tuple[int, ...], dtype: str
) -> str:
    """Names the dtype that WOULD fit, because that is always the real cause.

    An element-count mismatch on a seam buffer is almost never a shape bug: the
    driver wrote what the manifest declared. It is a reader that assumed a
    dtype. Order 134 read the encoder's float32 embeds as float16 and concluded
    the array attention arm produced uncorrelated output -- a stage that in fact
    agreed at cosine 0.999999 -- and order 137's first pass made the identical
    misread while checking it. Two independent readers, same assumption, and the
    generic "holds N elements, expected M" told neither of them why.

    ``float32`` read as ``float16`` yields exactly twice the elements, which is
    the signature to print rather than leave to be re-derived. The manifest is
    self-describing (see this module's header); a comparison that does not read
    the dtype from it has no business guessing one.
    """
    size = path.stat().st_size
    expected = int(np.prod(shape))
    lines = [
        f"{name}.bin holds {size} bytes, which is not "
        f"{expected} x {np.dtype(dtype).itemsize} for dtype {dtype!r} "
        f"and shape {shape}"
    ]
    for candidate in _WIRE_DTYPES:
        if expected and size == expected * candidate.itemsize:
            lines.append(
                f"  it IS {expected} x {candidate.itemsize} -- the driver wrote "
                f"{candidate.name!r}, not {dtype!r}. Read the dtype from the "
                f"seam's manifest instead of asserting one."
            )
    return "\n".join(lines)


def _parse_exit(stdout: str) -> int | None:
    """Pulls the driver's exit status out of the interop wrapper's echo."""
    for line in reversed(stdout.splitlines()):
        marker = line.strip()
        if marker.startswith(EXIT_MARKER):
            try:
                return int(marker[len(EXIT_MARKER) :])
            except ValueError:
                return None
    return None


def _tail(text: str, lines: int = 25) -> str:
    """The last few lines of driver output, for an error message."""
    kept = text.strip().splitlines()[-lines:]
    return "\n".join(kept)


def pack_modulation(
    modulation: StepModulation, dim: int
) -> npt.NDArray[np.float32]:
    """Flattens the shared adaLN parameters into the bridge's wire order.

    The wire carries 17 vectors: the five (shift, scale, gate) triples in
    :data:`MODULATION_ORDER`, then ``norm_out``'s scale and shift -- which is
    the concatenation of the four heads' linear outputs in module order, with
    no permutation anywhere. That order is the checkpoint's, read out of
    ``transformer_flux2`` and reproduced numerically against it in
    ``tests/test_modulation.py``; it is also exactly what ``unpackModulation``
    in ``npu_block_stack_dispatch.cpp`` reads.

    Args:
        modulation: The step's shared adaLN parameters.
        dim: DiT hidden width, the length of every vector.

    Returns:
        ``(17, dim)`` float32, in the driver's read order.

    Raises:
        ValueError: If any vector is not ``dim`` long.
    """
    triples = (
        modulation.double_img[0],
        modulation.double_img[1],
        modulation.double_txt[0],
        modulation.double_txt[1],
        modulation.single,
    )
    vectors: list[npt.NDArray[Any]] = []
    for triple in triples:
        vectors += [triple.shift, triple.scale, triple.gate]
    vectors += [modulation.out_scale, modulation.out_shift]

    wrong = [vector.shape for vector in vectors if vector.size != dim]
    if wrong:
        raise ValueError(
            f"every modulation vector must be {dim} long; got {wrong}"
        )
    return np.ascontiguousarray(
        np.stack([vector.reshape(dim).astype(np.float32) for vector in vectors])
    )


class DeviceBlockStack:
    """The ``DiTBlockStack`` seam, run on the real NPU driver.

    Drop-in for :class:`~flux_klein_host.stages.SyntheticBlockStack`: same
    Protocol, same shapes, but each call marshals the step's inputs to a scratch
    directory, execs ``npu_block_stack_dispatch.exe`` over Windows interop, and
    reads the velocity back.

    ``plan_only=True`` proves the crossing without an NPU -- the driver returns
    before it opens the device. The velocity it hands back is then a zero
    placeholder, and the caller has to opt in by constructing the backend that
    way, because a plan-only step must never be mistaken for a render.

    ``StepConditioning.modulation`` crosses when the host has it. A ``None``
    modulation is the seam's "the callee supplies it", and the driver answers
    with its own deterministic set (``modulation_source: synthetic``); a real
    one is packed by :func:`pack_modulation` into the 17-vector wire order and
    the driver reports ``modulation_source: request``. The response says which
    ran, so a synthetic-modulation step can never be read as a real one.
    """

    def __init__(
        self,
        executable: Path | str,
        config: KleinConfig = KLEIN_4B,
        *,
        weights: Path | str | None = None,
        scratch_root: Path | str = DEFAULT_STAGE_DIR,
        plan_only: bool = False,
        check: bool = False,
        keep_scratch: bool = False,
        timeout_seconds: float = 3600.0,
        interop: bool | None = None,
    ) -> None:
        """Initializes the backend.

        Args:
            executable: ``npu_block_stack_dispatch.exe``, as WSL sees it.
            config: Model geometry.
            weights: A converted ``skye-npu-weights`` manifest. Without it the
                driver runs on its own deterministic synthetic weights, which
                exercises the path but is not the model.
            scratch_root: Where per-step scratch directories are made.
            plan_only: Prove the crossing with no NPU. Returns placeholders.
            check: Have the driver also run its fp32 host reference and compare.
                Minutes per step at Klein dimensions; a bring-up switch.
            keep_scratch: Leave each step's directory behind.
            timeout_seconds: Cap on one driver run.
            interop: Force the Windows-interop or the native exec path; defaults
                to whether ``executable`` is a ``.exe``.
        """
        self.config = config
        self.bridge = _bridge(
            executable,
            weights=weights,
            scratch_root=scratch_root,
            plan_only=plan_only,
            check=check,
            keep_scratch=keep_scratch,
            timeout_seconds=timeout_seconds,
            interop=interop,
        )
        self.plan_only = plan_only
        self.weights = None if weights is None else Path(weights)

    def geometry(self, request: DiTBlockStackRequest) -> dict[str, int]:
        """The geometry block the driver configures itself from."""
        config = self.config
        return {
            "batch": int(request.latent.shape[0]),
            "heads": config.num_attention_heads,
            "head_dim": config.attention_head_dim,
            "text_seq": int(request.text.prompt_embeds.shape[1]),
            "image_seq": int(request.geometry.image_seq_len),
            "in_channels": config.in_channels,
            "joint_dim": config.joint_attention_dim,
            "mlp_ratio": int(config.mlp_ratio),
            "num_double": config.num_layers,
            "num_single": config.num_single_layers,
        }

    def __call__(
        self, request: DiTBlockStackRequest
    ) -> npt.NDArray[np.float32]:
        """Runs one denoise step on the device driver.

        Args:
            request: The full block-stack argument bundle.

        Returns:
            ``(B, image_seq_len, 128)`` velocity. Zeros if the backend was
            constructed ``plan_only=True``.

        Raises:
            ValueError: If the request is not something the driver can run.
            BridgeError: On any marshalling, exec, or driver failure.
        """
        request.validate(self.config)
        batch = int(request.latent.shape[0])
        if batch != 1:
            # The driver asserts this too: every row-wise kernel folds batch
            # into rows, but attention is per-sequence, so B > 1 needs one
            # dispatch per element and the driver does not issue them.
            raise ValueError(
                f"the block-stack driver runs one sequence per call, got "
                f"batch {batch}"
            )

        # Batch is folded into rows on the device, and the ids are per sequence
        # rather than per batch element, so the leading axis drops here.
        inputs: dict[str, npt.NDArray[Any]] = {
            "latent": _f32(request.latent.reshape(-1, self.config.in_channels)),
            "prompt_embeds": _f32(
                request.text.prompt_embeds.reshape(
                    -1, self.config.joint_attention_dim
                )
            ),
            "temb": _f32(request.conditioning.temb),
            "image_ids": _ids(request.image_ids[0]),
            "text_ids": _ids(request.text.text_ids[0]),
        }
        # Absent modulation is the seam's "the callee supplies it", and the
        # driver answers with its own synthetic set. Present modulation crosses
        # whole or not at all: a partial one would put the driver's numbers
        # behind some of the host's, which is exactly the silent mixture that
        # produces a plausible wrong image.
        if request.conditioning.modulation is not None:
            inputs["modulation"] = pack_modulation(
                request.conditioning.modulation, self.config.dim
            )

        shape = (
            batch,
            request.geometry.image_seq_len,
            self.config.in_channels,
        )
        values, response = self.bridge.invoke(
            "dit_block_stack",
            self.geometry(request),
            inputs,
            {"velocity": (shape, "float32")},
            require_computed=not self.plan_only,
        )
        del response
        return values["velocity"].astype(np.float32, copy=False)


class DeviceTextEncoder:
    """The ``TextEncoder`` seam, run on the real NPU driver.

    Drop-in for :class:`~flux_klein_host.stages.SyntheticTextEncoder`: marshals
    the tokenizer's output to a scratch directory, execs
    ``npu_qwen3_encoder_stack_dispatch.exe``, and reads the concatenated hidden
    states back as ``prompt_embeds``.

    Two tensors cross, and only two. The rope schedule, the per-head position
    map and the unit residual gate are functions of the geometry, so the driver
    derives them rather than the host sending a second source of truth for the
    same numbers. The ``text_ids`` come back from the host for the same reason
    the encoder owns them in the seam: it owns the padding.

    Without ``weights`` the driver has no embedding table, so it runs on a
    synthetic embedding and says so (``embedding_source`` in the response). The
    tokens still cross and are still range-checked; nothing about the answer can
    be read as a prompt's encoding.
    """

    def __init__(
        self,
        executable: Path | str,
        config: KleinConfig = KLEIN_4B,
        *,
        weights: Path | str | None = None,
        scratch_root: Path | str = DEFAULT_STAGE_DIR,
        plan_only: bool = False,
        check: bool = False,
        keep_scratch: bool = False,
        timeout_seconds: float = 3600.0,
        interop: bool | None = None,
    ) -> None:
        """Initializes the backend. Arguments as :class:`DeviceBlockStack`."""
        self.config = config
        self.plan_only = plan_only
        self.bridge = _bridge(
            executable,
            weights=weights,
            scratch_root=scratch_root,
            plan_only=plan_only,
            check=check,
            keep_scratch=keep_scratch,
            timeout_seconds=timeout_seconds,
            interop=interop,
        )

    def geometry(self, text_seq: int, valid_keys: int) -> dict[str, int]:
        """The geometry block the driver configures itself from.

        ``select`` is spelled as ``num_select`` plus ``select_i`` because the
        wire's geometry block is integers, not arrays.
        """
        config = self.config
        selected = config.text_hidden_state_layers
        geometry = {
            "batch": 1,
            "text_seq": int(text_seq),
            "hidden": config.text_hidden_size,
            "q_heads": config.text_num_attention_heads,
            "kv_heads": config.text_num_key_value_heads,
            "head_dim": config.text_head_dim,
            "inter": config.text_intermediate_size,
            "layers": config.text_num_layers,
            "valid_keys": int(valid_keys),
            "num_select": len(selected),
        }
        for slot, index in enumerate(selected):
            geometry[f"select_{slot}"] = int(index)
        return geometry

    def __call__(self, request: TextEncodeRequest) -> TextConditioning:
        """Encodes one prompt on the device driver.

        Args:
            request: The tokenizer's output and the requested batch.

        Returns:
            The :class:`~flux_klein_host.interfaces.TextConditioning`. Its
            ``prompt_embeds`` are zeros if the backend was constructed
            ``plan_only=True``.

        Raises:
            ValueError: If the request is not something the driver can run.
            BridgeError: On any marshalling, exec, or driver failure.
        """
        config = self.config
        tokens = request.tokens
        tokens.validate(config)
        if request.batch_size != 1:
            # The driver runs one sequence per call for the same reason the
            # block stack does: attention is per-sequence.
            raise ValueError(
                f"the encoder driver runs one prompt per call, got batch "
                f"{request.batch_size}"
            )

        seq = config.text_seq_len
        width = len(config.text_hidden_state_layers) * config.text_hidden_size
        values, _ = self.bridge.invoke(
            "qwen3_encoder",
            self.geometry(seq, tokens.num_valid_tokens),
            {
                "tokens": np.ascontiguousarray(tokens.tokens, dtype=np.int32),
                # The seam carries the bias as `(1, 1, S, S)`; the driver wants
                # the matrix itself, and the two leading axes are the batch and
                # head broadcast the driver does not have.
                "attention_bias": _f32(tokens.attention_bias.reshape(seq, seq)),
            },
            {"hidden_states": ((1, seq, width), "float32")},
            require_computed=not self.plan_only,
        )
        return TextConditioning(
            prompt_embeds=values["hidden_states"].astype(
                np.float32, copy=False
            ),
            text_ids=text_ids(seq),
            attention_bias=tokens.attention_bias,
        )


class DeviceVaeDecoder:
    """The ``VaeDecoder`` seam, run on the real NPU driver.

    Drop-in for :class:`~flux_klein_host.stages.SyntheticVaeDecoder`: marshals
    the decoder-facing latent to a scratch directory, execs
    ``npu_vae_decoder_dispatch.exe``, and reads the RGB back.

    One tensor crosses. :class:`~flux_klein_host.interfaces.VaeDecodeInput` is
    defined as the *already* denormalized and unpacked latent -- the batch-norm
    statistics are per packed channel, so that order is the host's to get right
    and it does, in ``vae_decode_input`` -- and the driver starts at its own
    ``post_quant_conv`` with nothing left to reshape.
    """

    def __init__(
        self,
        executable: Path | str,
        config: KleinConfig = KLEIN_4B,
        *,
        weights: Path | str | None = None,
        scratch_root: Path | str = DEFAULT_STAGE_DIR,
        plan_only: bool = False,
        check: bool = False,
        keep_scratch: bool = False,
        timeout_seconds: float = 3600.0,
        interop: bool | None = None,
    ) -> None:
        """Initializes the backend. Arguments as :class:`DeviceBlockStack`."""
        self.config = config
        self.plan_only = plan_only
        self.bridge = _bridge(
            executable,
            weights=weights,
            scratch_root=scratch_root,
            plan_only=plan_only,
            check=check,
            keep_scratch=keep_scratch,
            timeout_seconds=timeout_seconds,
            interop=interop,
        )

    def geometry(self, geometry: LatentGeometry) -> dict[str, int]:
        """The geometry block the driver configures itself from."""
        config = self.config
        block = {
            "batch": 1,
            "latent_channels": config.vae_latent_channels,
            "latent_height": geometry.latent_height,
            "latent_width": geometry.latent_width,
            "layers_per_block": config.vae_layers_per_block,
            "mid_attention": int(config.vae_mid_block_add_attention),
            "post_quant_conv": 1,
            "stages": len(config.vae_block_out_channels),
        }
        # Encoder order, as `vae/config.json` writes it; the decoder walks it
        # reversed and the driver does that itself.
        for stage, width in enumerate(config.vae_block_out_channels):
            block[f"block_out_{stage}"] = int(width)
        return block

    def __call__(self, request: VaeDecodeInput) -> npt.NDArray[np.float32]:
        """Decodes one latent on the device driver.

        Args:
            request: The denormalized, unpacked latent.

        Returns:
            ``(B, 3, height, width)`` RGB in ``[-1, 1]``. Zeros if the backend
            was constructed ``plan_only=True``.

        Raises:
            ValueError: If the request is not something the driver can run.
            BridgeError: On any marshalling, exec, or driver failure.
        """
        request.validate(self.config)
        batch = int(request.latent.shape[0])
        if batch != 1:
            raise ValueError(
                f"the VAE decoder driver runs one image per call, got batch "
                f"{batch}"
            )
        geometry = request.geometry
        scale = self.config.vae_scale_factor
        shape = (
            1,
            3,
            geometry.latent_height * scale,
            geometry.latent_width * scale,
        )
        values, _ = self.bridge.invoke(
            "vae_decoder",
            self.geometry(geometry),
            {"latent": _f32(request.latent[0])},
            {"rgb": (shape, "float32")},
            require_computed=not self.plan_only,
        )
        return values["rgb"].astype(np.float32, copy=False)


def _bridge(
    executable: Path | str,
    *,
    weights: Path | str | None,
    scratch_root: Path | str,
    plan_only: bool,
    check: bool,
    keep_scratch: bool,
    timeout_seconds: float,
    interop: bool | None,
) -> DeviceBridge:
    """Builds the bridge every seam backend drives its driver through."""
    if interop is None:
        interop = Path(executable).suffix.lower() == ".exe"
    arguments: list[str] = []
    if weights is not None:
        # The interop path builds a cmd.exe line, where a path has to be both
        # Windows-spelled and quoted; a native argv takes it raw.
        arguments += [
            "--weights",
            f'"{to_windows_path(Path(weights))}"'
            if interop
            else str(Path(weights)),
        ]
    if check:
        arguments.append("--check")
    return DeviceBridge(
        executable=Path(executable),
        scratch_root=Path(scratch_root),
        extra_args=tuple(arguments),
        plan_only=plan_only,
        keep_scratch=keep_scratch,
        timeout_seconds=timeout_seconds,
        interop=interop,
    )


def _f32(array: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """A contiguous float32 view of a host array, for the wire."""
    return np.ascontiguousarray(array, dtype=np.float32)


def _ids(array: npt.NDArray[Any]) -> npt.NDArray[np.int64]:
    """A contiguous int64 ``(S, 4)`` position-id array, for the wire."""
    return np.ascontiguousarray(array, dtype=np.int64)


__all__ = [
    "DEFAULT_STAGE_DIR",
    "MODULATION_ORDER",
    "BridgeError",
    "BridgeResponse",
    "DeviceBlockStack",
    "DeviceBridge",
    "DeviceTextEncoder",
    "DeviceVaeDecoder",
    "pack_modulation",
    "to_windows_path",
]
