"""Bridges the V2 tokenizer slice to the pipeline's :class:`PromptTokenizer`.

Two routes, both producing the same :class:`PromptTokens`:

* :class:`KleinTokenizer` runs ``klein_tokenize.py`` in process. Needs the
  ``tokenizers`` package and the checkpoint's ``tokenizer.json`` -- cached
  locally after the first fetch.
* :class:`ArtifactTokenizer` reads a directory the V2 CLI already wrote. This
  is the route the P6 design actually specifies for the device: tokenization
  is ahead-of-time host work in WSL, and the Windows driver reads
  ``tokens.bin`` and ``attention_mask.bin`` off disk.

Both are imported by path from the sibling ``hexagon-ondevice`` example rather
than as a package, matching how the reference oracle reaches MAX's own helpers.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import KLEIN_4B, KleinConfig
from .stages import PromptTokens

_TOKENIZER_RELPATH = (
    "plugins/internal-skye/examples/hexagon-ondevice/klein_tokenize.py"
)


def _repo_root(start: Path) -> Path | None:
    """Finds the monorepo root by looking for the V2 tokenizer."""
    for candidate in [start, *start.parents]:
        if (candidate / _TOKENIZER_RELPATH).is_file():
            return candidate
    return None


def load_tokenizer_module() -> Any:
    """Imports ``klein_tokenize`` from the sibling hexagon-ondevice example.

    Returns:
        The imported module.

    Raises:
        ImportError: If this file is not inside the monorepo, or the module's
            own dependencies are missing.
    """
    cached = sys.modules.get("_skye_klein_tokenize")
    if cached is not None:
        return cached
    root = _repo_root(Path(__file__).resolve())
    if root is None:
        raise ImportError(
            "cannot locate the monorepo root; the tokenizer comes from "
            f"{_TOKENIZER_RELPATH} and this file must live in-tree"
        )
    spec = importlib.util.spec_from_file_location(
        "_skye_klein_tokenize", root / _TOKENIZER_RELPATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_skye_klein_tokenize"] = module
    spec.loader.exec_module(module)
    return module


class KleinTokenizer:
    """Runs the V2 tokenizer in process."""

    def __init__(
        self,
        config: KleinConfig = KLEIN_4B,
        *,
        spec: str | None = None,
        revision: str | None = None,
        allow_truncation: bool = False,
    ) -> None:
        """Resolves the Qwen3 tokenizer.

        Args:
            config: Model geometry, for the padded sequence length.
            spec: A local ``tokenizer.json``, a directory holding one, or a
                HuggingFace repo id. Defaults to the pinned Klein repo.
            revision: Revision to pin; defaults to the pinned Klein revision.
            allow_truncation: Truncate an over-length prompt instead of
                refusing it.

        Raises:
            ImportError: If ``tokenizers`` or the V2 module is unavailable.
        """
        module = load_tokenizer_module()
        self._module = module
        self.config = config
        self.allow_truncation = allow_truncation
        self.tokenizer, self.source = module.resolve_tokenizer(
            spec if spec is not None else module.KLEIN_REPO,
            revision if revision is not None else module.KLEIN_REVISION,
        )

    def __call__(self, prompt: str) -> PromptTokens:
        """Encodes one prompt.

        Args:
            prompt: The raw user prompt.

        Returns:
            The encoded :class:`PromptTokens`.
        """
        encoded = self._module.tokenize(
            self.tokenizer,
            prompt,
            seq_len=self.config.text_seq_len,
            allow_truncation=self.allow_truncation,
        )
        bias = self._module.build_attention_bias(encoded.attention_mask)
        return PromptTokens(
            tokens=encoded.token_ids.astype(np.int32),
            attention_mask=encoded.attention_mask,
            attention_bias=bias,
            num_valid_tokens=encoded.num_valid_tokens,
            rendered=encoded.rendered,
        )


class ArtifactTokenizer:
    """Replays a directory the V2 CLI wrote, the way the device driver does.

    Reads the same ``tokens.bin`` / ``attention_mask.bin`` the driver reads,
    and rebuilds the bias from the mask rather than requiring the optional
    1 MiB ``attention_bias.bin``.
    """

    def __init__(
        self, artifact_dir: Path | str, config: KleinConfig = KLEIN_4B
    ) -> None:
        """Loads the artifact.

        Args:
            artifact_dir: Directory holding the V2 buffers and manifest.
            config: Model geometry, checked against the buffers.

        Raises:
            FileNotFoundError: If the buffers are absent.
            ValueError: If a buffer disagrees with ``config``.
        """
        directory = Path(artifact_dir)
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest.json under {directory}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        dtype = np.int32
        for entry in self.manifest.get("buffers", ()):
            if entry["name"] == "tokens":
                dtype = np.dtype(entry["dtype"]).type
        tokens = np.fromfile(directory / "tokens.bin", dtype=dtype)
        mask = np.fromfile(directory / "attention_mask.bin", dtype=np.uint8)
        if tokens.size != config.text_seq_len:
            raise ValueError(
                f"{directory}/tokens.bin holds {tokens.size} ids, config "
                f"expects {config.text_seq_len}"
            )
        if mask.size != tokens.size:
            raise ValueError(
                f"attention_mask holds {mask.size} entries, tokens hold "
                f"{tokens.size}"
            )

        self._tokens = tokens.astype(np.int32)
        self._mask = mask
        self._bias = load_tokenizer_module().build_attention_bias(
            mask, self.manifest.get("mask_fill_value", -10000.0)
        )
        self.config = config

    def __call__(self, prompt: str) -> PromptTokens:
        """Returns the stored tokens, ignoring ``prompt``.

        Args:
            prompt: Checked against the manifest's recorded prompt so a
                mismatched artifact cannot be used silently.

        Returns:
            The stored :class:`PromptTokens`.

        Raises:
            ValueError: If ``prompt`` is not the prompt the artifact holds.
        """
        recorded = self.manifest.get("prompt")
        if recorded is not None and prompt != recorded:
            raise ValueError(
                f"this artifact tokenizes {recorded!r}, not {prompt!r}"
            )
        return PromptTokens(
            tokens=self._tokens,
            attention_mask=self._mask,
            attention_bias=self._bias,
            num_valid_tokens=int(self.manifest["num_valid_tokens"]),
            rendered=self.manifest.get("rendered_prompt", prompt),
        )


__all__ = ["ArtifactTokenizer", "KleinTokenizer", "load_tokenizer_module"]
