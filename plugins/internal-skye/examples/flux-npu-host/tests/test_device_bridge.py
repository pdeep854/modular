"""The device-dispatch bridge, proven against a real driver with no NPU.

The bridge is the mechanism that turns the synthetic pipeline into a real-driver
one: the host marshals a seam's inputs into a scratch directory, execs the
driver, and reads the answer back. These tests run that mechanism against the
actual block-stack driver -- the host-only build of the same source file, so the
manifest reader, the bin parsing, the geometry override and the response are
exactly the ones the Windows ``.exe`` runs.

Two levels, because they falsify different things:

* ``--plan-only`` proves the round trip is WIRED: files land, the driver returns
  0, an output of the right shape comes back. Its velocity is a placeholder, so
  it cannot tell a delivered latent from an ignored one.
* the host-reference run proves the round trip is CONNECTED: the velocity is a
  function of the marshalled inputs, so perturbing one on the host side has to
  move it. That is what a shape-only check cannot catch.

The driver comes from bazel::

    ./bazelw build --config=plugin \\
      @plugin//examples/hexagon-ondevice:npu_block_stack_reference

Point ``NPU_BLOCK_STACK_DRIVER`` at it (or let the tests find it under
``bazel-bin``); without it they skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from flux_klein_host.config import (
    KleinConfig,
    LatentGeometry,
    resolve_geometry,
)
from flux_klein_host.device_bridge import (
    BridgeError,
    DeviceBlockStack,
    DeviceBridge,
)
from flux_klein_host.interfaces import (
    DiTBlockStackRequest,
    StepConditioning,
    TextConditioning,
)
from flux_klein_host.latents import latent_image_ids, text_ids

#: Where `bazel build` leaves the host-only driver, relative to the repo root.
_BAZEL_DRIVER = Path(
    "bazel-bin/external/plugin+/examples/hexagon-ondevice/"
    "npu_block_stack_reference"
)


def _mini_config() -> KleinConfig:
    """A geometry small enough to run a host reference per test.

    Only ``attention_head_dim`` is fixed -- the attention kernel bakes 128, and
    so does Klein. Everything else shrinks, keeping every matmul ``K`` a
    multiple of 64 (the HVX fp16 vector width), which is what the driver's plan
    check asserts. ``joint_attention_dim`` has to stay the stacked encoder
    width, so it shrinks with ``text_hidden_size``.
    """
    return KleinConfig(
        num_attention_heads=4,  # dim = 512
        attention_head_dim=128,
        text_seq_len=64,
        text_hidden_size=512,
        joint_attention_dim=3 * 512,
        num_layers=1,
        num_single_layers=1,
    )


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "bazelw").is_file():
            return candidate
    return Path.cwd()


def _driver() -> Path:
    override = os.environ.get("NPU_BLOCK_STACK_DRIVER")
    if override:
        return Path(override)
    return _repo_root() / _BAZEL_DRIVER


pytestmark = pytest.mark.skipif(
    not _driver().is_file(),
    reason=(
        "no block-stack driver; build "
        "@plugin//examples/hexagon-ondevice:npu_block_stack_reference or set "
        "NPU_BLOCK_STACK_DRIVER"
    ),
)


def _request(
    config: KleinConfig, *, seed: int = 0, scale: float = 1.0
) -> DiTBlockStackRequest:
    """One denoise step's inputs, at the reduced geometry."""
    geometry = resolve_geometry(config, 128, 128)
    rng = np.random.RandomState(seed)
    return DiTBlockStackRequest(
        latent=(
            rng.standard_normal((1, geometry.image_seq_len, config.in_channels))
            * scale
        ).astype(np.float32),
        image_ids=latent_image_ids(geometry),
        text=TextConditioning(
            prompt_embeds=(
                rng.standard_normal(
                    (1, config.text_seq_len, config.joint_attention_dim)
                )
                * 0.1
            ).astype(np.float32),
            text_ids=text_ids(config.text_seq_len),
        ),
        conditioning=StepConditioning(
            step_index=0,
            timestep=1.0,
            temb=rng.standard_normal((1, config.dim)).astype(np.float32),
        ),
        geometry=geometry,
    )


def _backend(
    tmp_path: Path, config: KleinConfig, **kwargs: object
) -> DeviceBlockStack:
    return DeviceBlockStack(
        _driver(),
        config,
        scratch_root=tmp_path,
        interop=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_plan_only_round_trip(tmp_path: Path) -> None:
    """Marshal out, exec, read back: the crossing runs and returns a shape."""
    config = _mini_config()
    request = _request(config)
    stack = _backend(tmp_path, config, plan_only=True, keep_scratch=True)

    velocity = stack(request)

    assert velocity.shape == request.latent.shape
    assert velocity.dtype == np.float32
    # A plan-only velocity is a placeholder, and the driver says so rather than
    # leaving the caller to guess from the values.
    assert not velocity.any()
    response = stack.bridge.last_response
    assert response is not None
    assert response.status == "ok"
    assert response.plan_only
    assert response.output_source == "none"
    assert not response.computed


def test_plan_only_marshals_every_seam_input(tmp_path: Path) -> None:
    """The scratch directory holds the whole request, self-described."""
    config = _mini_config()
    stack = _backend(tmp_path, config, plan_only=True, keep_scratch=True)
    stack(_request(config))

    scratch = stack.bridge.last_scratch
    assert scratch is not None
    manifest = json.loads((scratch / "request.json").read_text())
    assert manifest["format"] == "skye-npu-io"
    assert manifest["seam"] == "dit_block_stack"
    assert set(manifest["inputs"]) == {
        "latent",
        "prompt_embeds",
        "temb",
        "image_ids",
        "text_ids",
    }
    for spec in manifest["inputs"].values():
        payload = scratch / spec["path"]
        itemsize = np.dtype(spec["dtype"]).itemsize
        assert payload.stat().st_size == int(np.prod(spec["shape"])) * itemsize

    # The geometry the host asked for is the geometry the driver reports back,
    # so a driver silently running its own default shape would be caught.
    response = json.loads((scratch / "response.json").read_text())
    assert response["geometry"] == manifest["geometry"]


def test_placeholder_is_not_handed_back_as_a_result(tmp_path: Path) -> None:
    """A plan-only response must not satisfy a caller that wanted numbers."""
    config = _mini_config()
    bridge = DeviceBridge(
        executable=_driver(),
        scratch_root=tmp_path,
        extra_args=("--plan-only",),
        interop=False,
    )
    geometry = resolve_geometry(config, 128, 128)
    shape = (geometry.image_seq_len, config.in_channels)
    inputs: dict[str, npt.NDArray[np.float32]] = {
        "latent": np.zeros(shape, dtype=np.float32),
        "prompt_embeds": np.zeros(
            (config.text_seq_len, config.joint_attention_dim), dtype=np.float32
        ),
        "temb": np.zeros((1, config.dim), dtype=np.float32),
        "image_ids": latent_image_ids(geometry)[0],
        "text_ids": text_ids(config.text_seq_len)[0],
    }
    with pytest.raises(BridgeError, match="placeholders"):
        bridge.invoke(
            "dit_block_stack",
            _geometry_block(config, geometry),
            inputs,
            {"velocity": (shape, "float32")},
        )


def test_host_reference_velocity_depends_on_the_latent(tmp_path: Path) -> None:
    """The crossing DELIVERS: change an input, the answer has to move.

    A shape-only round trip passes just as happily when the driver ignores
    everything the host sent, which is the failure mode that would survive all
    the way to a wrong image. Here the driver computes the velocity from the
    marshalled request, so two different latents cannot produce one answer.
    """
    config = _mini_config()
    stack = _backend(tmp_path, config)

    first = stack(_request(config, seed=0))
    same = stack(_request(config, seed=0))
    other = stack(_request(config, seed=1))

    response = stack.bridge.last_response
    assert response is not None
    assert response.output_source == "host_reference"
    assert np.isfinite(first).all()

    # Deterministic in the request, as the seam's purity clause requires.
    np.testing.assert_array_equal(first, same)
    assert not np.allclose(first, other)


def test_host_reference_velocity_depends_on_the_prompt(tmp_path: Path) -> None:
    """The text stream crosses too, not just the latent."""
    config = _mini_config()
    stack = _backend(tmp_path, config)

    base = _request(config, seed=0)
    moved = DiTBlockStackRequest(
        latent=base.latent,
        image_ids=base.image_ids,
        text=TextConditioning(
            prompt_embeds=base.text.prompt_embeds + np.float32(0.25),
            text_ids=base.text.text_ids,
        ),
        conditioning=base.conditioning,
        geometry=base.geometry,
    )
    assert not np.allclose(stack(base), stack(moved))


def test_a_shape_disagreement_fails_at_the_seam(tmp_path: Path) -> None:
    """A manifest that lies about a shape is caught, not reinterpreted."""
    config = _mini_config()
    geometry = resolve_geometry(config, 128, 128)
    bridge = DeviceBridge(
        executable=_driver(),
        scratch_root=tmp_path,
        plan_only=True,
        interop=False,
    )
    shape = (geometry.image_seq_len, config.in_channels)
    inputs: dict[str, npt.NDArray[np.float32]] = {
        # One row short of what the geometry asks for.
        "latent": np.zeros((shape[0] - 1, shape[1]), dtype=np.float32),
        "prompt_embeds": np.zeros(
            (config.text_seq_len, config.joint_attention_dim), dtype=np.float32
        ),
        "temb": np.zeros((1, config.dim), dtype=np.float32),
        "image_ids": latent_image_ids(geometry)[0],
        "text_ids": text_ids(config.text_seq_len)[0],
    }
    with pytest.raises(BridgeError, match="the geometry needs"):
        bridge.invoke(
            "dit_block_stack",
            _geometry_block(config, geometry),
            inputs,
            {"velocity": (shape, "float32")},
            require_computed=False,
        )


def test_batched_requests_are_refused(tmp_path: Path) -> None:
    """Attention is per-sequence, so the driver runs one sequence per call."""
    config = _mini_config()
    request = _request(config)
    batched = DiTBlockStackRequest(
        latent=np.concatenate([request.latent, request.latent]),
        image_ids=np.concatenate([request.image_ids, request.image_ids]),
        text=TextConditioning(
            prompt_embeds=np.concatenate(
                [request.text.prompt_embeds, request.text.prompt_embeds]
            ),
            text_ids=np.concatenate(
                [request.text.text_ids, request.text.text_ids]
            ),
        ),
        conditioning=StepConditioning(
            step_index=0,
            timestep=1.0,
            temb=np.concatenate(
                [request.conditioning.temb, request.conditioning.temb]
            ),
        ),
        geometry=request.geometry,
    )
    stack = _backend(tmp_path, config, plan_only=True)
    with pytest.raises(ValueError, match="one sequence per call"):
        stack(batched)


def _geometry_block(
    config: KleinConfig, geometry: LatentGeometry
) -> dict[str, int]:
    """The geometry a raw :class:`DeviceBridge` call has to spell itself."""
    return {
        "batch": 1,
        "heads": config.num_attention_heads,
        "head_dim": config.attention_head_dim,
        "text_seq": config.text_seq_len,
        "image_seq": geometry.image_seq_len,
        "in_channels": config.in_channels,
        "joint_dim": config.joint_attention_dim,
        "mlp_ratio": int(config.mlp_ratio),
        "num_double": config.num_layers,
        "num_single": config.num_single_layers,
    }
