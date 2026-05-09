"""Checkpoint helpers for crash-resistant training."""

from __future__ import annotations

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
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    log.debug("Checkpoint saved: %s", path)
