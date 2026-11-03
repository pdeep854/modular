"""Independent references the scaffold is validated against.

Two of them, deliberately:

``max_scheduler``
    MAX's own ``FlowMatchEulerDiscreteScheduler``, loaded straight from the
    monorepo by path. It is pure numpy with no MAX imports, so it costs no
    build. This is the oracle the P6 design names -- the on-device pipeline is
    checkpointed against MAX-in-WSL, so agreeing with *this* is what makes the
    scaffold substitutable for MAX's host prep.

``diffusers``
    The upstream ``FlowMatchEulerDiscreteScheduler`` and
    ``get_timestep_embedding``. Optional, since it needs torch. This one is the
    genuinely independent check: MAX's scheduler is itself a port, so matching
    only MAX would not catch a shared porting error.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_MAX_SCHEDULER_RELPATH = (
    "max/python/max/pipelines/diffusion/schedulers/"
    "scheduling_flow_match_euler_discrete.py"
)


def _repo_root() -> Path:
    """Walks up to the monorepo root from this file."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "max" / "python" / "max").is_dir():
            return parent
    raise RuntimeError("could not locate the monorepo root from tests/")


def load_max_scheduler_module() -> ModuleType:
    """Imports MAX's flow-match scheduler module by path.

    Returns:
        The loaded module.

    Raises:
        RuntimeError: If the file is missing or fails to load.
    """
    path = _repo_root() / _MAX_SCHEDULER_RELPATH
    if not path.is_file():
        raise RuntimeError(f"MAX scheduler reference not found at {path}")
    spec = importlib.util.spec_from_file_location("_max_flow_match_ref", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def diffusers_available() -> bool:
    """Reports whether the optional torch/diffusers reference is importable."""
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def make_diffusers_scheduler(config):
    """Builds a diffusers scheduler from our :class:`SchedulerConfig`.

    Only the fields that exist in the public ``scheduler_config.json`` are
    forwarded; ``use_empirical_mu`` is a MAX-side policy for choosing ``mu``,
    not a scheduler parameter, so ``mu`` is passed explicitly at
    ``set_timesteps`` time instead.
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=config.num_train_timesteps,
        shift=config.shift,
        use_dynamic_shifting=config.use_dynamic_shifting,
        base_shift=config.base_shift,
        max_shift=config.max_shift,
        base_image_seq_len=config.base_image_seq_len,
        max_image_seq_len=config.max_image_seq_len,
        shift_terminal=config.shift_terminal,
        time_shift_type=config.time_shift_type,
        invert_sigmas=False,
        stochastic_sampling=False,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
    )
