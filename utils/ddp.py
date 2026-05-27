"""DistributedDataParallel scaffolding.

These helpers keep single-GPU runs unchanged while making the training scripts
ready for ``torchrun --nproc_per_node=N``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from utils.logger import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class DDPState:
    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: Optional[torch.device] = None


def setup_ddp(seed: int = 42) -> DDPState:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DDPState(enabled=False)

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() and os.name != "nt" else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    log.info(
        "DDP initialized rank=%s local_rank=%s world_size=%s backend=%s",
        rank,
        local_rank,
        world_size,
        backend,
    )
    return DDPState(True, rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def is_main_process(state: Optional[DDPState] = None) -> bool:
    if state is not None:
        return state.rank == 0
    return int(os.getenv("RANK", "0")) == 0


def wrap_ddp(model: torch.nn.Module, state: DDPState) -> torch.nn.Module:
    if not state.enabled:
        return model
    kwargs = {}
    if state.device is not None and state.device.type == "cuda":
        kwargs["device_ids"] = [state.local_rank]
        kwargs["output_device"] = state.local_rank
    return DistributedDataParallel(model, **kwargs)


def make_sampler(dataset: Dataset, state: DDPState, shuffle: bool) -> Optional[DistributedSampler]:
    if not state.enabled:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=state.world_size,
        rank=state.rank,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def make_loader(
    dataset: Dataset,
    state: DDPState,
    batch_size: int,
    shuffle: bool,
    **kwargs,
) -> DataLoader:
    sampler = make_sampler(dataset, state, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        **kwargs,
    )


def set_epoch(loader: DataLoader, epoch: int) -> None:
    sampler = getattr(loader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
