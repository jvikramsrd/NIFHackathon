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
