"""Checkpoint helpers for crash-resistant training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from utils.logger import get_logger

log = get_logger(__name__)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a PyTorch checkpoint via a temporary file, then atomically replace."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)
    log.debug("Checkpoint saved: %s", path)


def clean_state_dict(
    state_dict: dict[str, Any],
    model_state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of ``state_dict`` whose keys match ``model_state_dict``.

    Handles three common prefix mismatches that cause ``load_state_dict`` to
    silently drop all weights when used with ``strict=False``, or to crash when
    used with ``strict=True``:

    1. **DDP wrap** — ``DistributedDataParallel`` inserts ``"module."`` at the
       root, turning ``"model.encoder.xxx"`` into ``"module.model.encoder.xxx"``.
    2. **torch.compile** — inserts ``"_orig_mod."`` at the root.
    3. **DDP on inner model only** — Stage 1 wraps just ``module.model`` in DDP,
       so keys become ``"model.module.encoder.xxx"`` (prefix in the middle).
    4. **Any combination** of the above.

    The function tries the following remappings in priority order for each key:

    * Keep as-is (already matches).
    * Strip ``"module."`` from the root (standard DDP).
    * Strip ``"_orig_mod."`` from the root (torch.compile).
    * Replace ``"model.module."`` → ``"model."`` (inner-model DDP as in Stage 1).
    * Strip ``"module."`` *and* replace ``"model.module."`` → ``"model."``.

    Any key that still has no match in ``model_state_dict`` is kept as-is so
    that ``load_state_dict`` can report it (with ``strict=False``) rather than
    silently discarding it.

    Usage::

        model_state = ckpt.get("model_state") or ckpt.get("state_dict", {})
        cleaned = clean_state_dict(model_state, module.state_dict())
        incompatible = module.load_state_dict(cleaned, strict=False)
    """
    target_keys = set(model_state_dict.keys())

    def _candidates(k: str):
        yield k                                             # 1. keep as-is
        if k.startswith("module."):
            stripped = k[len("module."):]
            yield stripped                                  # 2. root DDP
            if stripped.startswith("model.module."):
                yield "model." + stripped[len("model.module."):]  # 2+3 combined
        if k.startswith("_orig_mod."):
            yield k[len("_orig_mod."):]                    # 3. torch.compile
        if k.startswith("model.module."):
            yield "model." + k[len("model.module."):]      # 4. inner DDP

    new_dict: dict[str, Any] = {}
    for k, v in state_dict.items():
        mapped = k
        for candidate in _candidates(k):
            if candidate in target_keys:
                mapped = candidate
                break
        new_dict[mapped] = v

    return new_dict
