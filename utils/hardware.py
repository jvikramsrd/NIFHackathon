"""
utils/hardware.py
──────────────────
One-stop hardware configuration for Windows + RTX A4000 + i9-13900.
Call `setup()` at the top of every train/inference script.
"""

import os
import platform
import sys

import torch
import torch.nn as nn

_SETUP_DONE = False


def setup(seed: int = 42, verbose: bool = True) -> torch.device:
    """
    Full hardware configuration for RTX A4000 + i9-13900K + 32 GB.
    Safe to call multiple times in the same process.
    Returns the torch.device to use.
    """

    global _SETUP_DONE

    # ── Windows multiprocessing + CPU thread config ──────────────────────────
    if platform.system() == "Windows":
        import multiprocessing

        # set_start_method may already be configured by an earlier stage
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        # i9-13900K: 8 P-cores (fast) + 16 E-cores (efficiency).
        # Pin OpenMP / MKL to P-core count so BLAS never lands on E-cores.
        os.environ["OMP_NUM_THREADS"] = "8"
        os.environ["MKL_NUM_THREADS"] = "8"
        os.environ["OPENBLAS_NUM_THREADS"] = "8"

    # PyTorch inter/intra-op thread pools must be configured before parallel work.
    # Make setup idempotent: set once, skip on subsequent calls.
    if not _SETUP_DONE:
        try:
            torch.set_num_threads(8)  # intra-op (single-op parallelism)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(4)  # inter-op (graph-level parallelism)
        except RuntimeError:
            pass
        _SETUP_DONE = True

    # ── CUDA + Ampere flags ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("[HW] WARNING: CUDA not available — running on CPU")
        return torch.device("cpu")

    device = torch.device("cuda:0")

    # TF32 — free ~3× matmul throughput on Ampere, no accuracy loss
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # PyTorch 2.x explicit matmul precision API (complements allow_tf32)
    # "high" = TF32 for matmul, "highest" = FP32 (slower), "medium" = bf16
    torch.set_float32_matmul_precision("high")

    # Flash Attention via SDPA (PyTorch 2.x) — speeds up Swin attention
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)  # disable slow fallback
    except AttributeError:
        pass

    # ── Reproducibility ─────────────────────────────────────────────────────
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Diagnostics ─────────────────────────────────────────────────────────
    if verbose:
        gpu = torch.cuda.get_device_properties(device)
        vram_gb = gpu.total_memory / 1024**3
        cc = f"{gpu.major}.{gpu.minor}"
        print("=" * 60)
        print(f"  GPU              : {gpu.name}")
        print(f"  VRAM             : {vram_gb:.1f} GB")
        print(f"  SM count         : {gpu.multi_processor_count}  (CC {cc})")
        print(f"  CUDA             : {torch.version.cuda}")
        print(f"  PyTorch          : {torch.__version__}")
        print(f"  bf16 native      : {gpu.major >= 8}")
        print(f"  TF32             : {torch.backends.cudnn.allow_tf32}")
        print(f"  matmul precision : {torch.get_float32_matmul_precision()}")
        print(f"  intra threads    : {torch.get_num_threads()}")
        print(f"  interop threads  : {torch.get_num_interop_threads()}")
        print("=" * 60)

    return device


def worker_init_fn(worker_id: int):
    """
    DataLoader worker initialiser — pins each worker to a P-core on i9-13900K.

    i9-13900K core layout (Windows logical processors):
      LP 0–7   → P-cores (fast, hyperthreaded: LP 0/1 = core 0, etc.)
      LP 8–23  → E-cores (efficiency)

    We assign worker 0→LP0, worker1→LP2, ... up to 4 workers per P-core pair.
    This keeps all data-loading work on fast P-cores.

    Usage:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    try:
        import os

        import psutil

        p = psutil.Process(os.getpid())
        # P-core logical processors: 0,1,2,3,4,5,6,7 (8 P-cores × 2 HT = 16 LPs)
        p_core_lps = list(range(16))  # LP 0–15 are P-core threads on 13900K
        lp = p_core_lps[worker_id % len(p_core_lps)]
        p.cpu_affinity([lp])
    except Exception:
        pass  # psutil not installed or affinity not supported — silently skip


def compile_model(
    model: nn.Module, mode: str = "reduce-overhead", fullgraph: bool = False
) -> nn.Module:
    """
    torch.compile() with graceful fallback.

    mode:
      "reduce-overhead"  — fastest to compile, best for training loops
      "max-autotune"     — ~20 min compile, +5% runtime throughput

    fullgraph=True → safe for ConvNeXt (no dynamic control flow) → +8%.
    fullgraph=False → required for Swin-B (has conditional ops in attention).
    """
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph)
        print(f"  torch.compile() applied (mode={mode}, fullgraph={fullgraph})")
        return compiled
    except Exception as e:
        print(f"  torch.compile() skipped: {e}")
        return model


def to_channels_last(model: nn.Module) -> nn.Module:
    """
    Convert model weights to channels-last (NHWC) memory format.

    Ampere tensor cores process NHWC convolutions 15–30% faster than NCHW.
    This is the single biggest remaining throughput gain on the A4000.

    Use for: ConvNeXt (Stage 2A), EfficientNet backbones.
    Do NOT use for: pure transformer models (Swin-B) — no spatial convs.
    """
    try:
        model = model.to(memory_format=torch.channels_last)
        print("  channels_last (NHWC) applied")
    except Exception as e:
        print(f"  channels_last skipped: {e}")
    return model


def cl_input(tensor: torch.Tensor) -> torch.Tensor:
    """Convert 4D batch tensor to channels_last before model forward pass."""
    return tensor.to(memory_format=torch.channels_last) if tensor.ndim == 4 else tensor


def get_amp_context(dtype: torch.dtype = torch.bfloat16):
    """
    Return the right AMP context.
    bfloat16 on Ampere needs no GradScaler.
    float16  on older cards needs GradScaler.

    Uses torch.amp.autocast (PyTorch 2.x API).
    torch.cuda.amp.autocast is deprecated in PyTorch 2.5.
    """
    # Use the new torch.amp.autocast API (works in PyTorch 2.0+)
    try:
        from torch.amp import GradScaler, autocast
    except ImportError:
        # Fallback for older PyTorch versions
        from torch.cuda.amp import GradScaler, autocast

    if dtype == torch.bfloat16:
        ctx = autocast("cuda", dtype=torch.bfloat16)
        scaler = None  # bf16 doesn't underflow → no scaling needed
    else:
        ctx = autocast("cuda", dtype=torch.float16)
        scaler = GradScaler("cuda")

    return ctx, scaler


class EMA:
    """
    Exponential Moving Average of model weights.
    Keeps a shadow copy that is updated after every optimizer step.
    Use shadow weights for validation / inference.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow: dict = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        """Temporarily load EMA weights into model (for val/inference)."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore training weights after val/inference."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}


def vram_stats() -> str:
    """Return a one-liner VRAM usage string."""
    if not torch.cuda.is_available():
        return "CUDA N/A"
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"VRAM  alloc={alloc:.1f}GB  reserved={reserv:.1f}GB  total={total:.1f}GB"


def get_cuda_streams():
    """
    Return two CUDA streams for H2D transfer / compute overlap.

    How to use:
        transfer_stream, compute_stream = get_cuda_streams()

        # In training loop:
        with torch.cuda.stream(transfer_stream):
            imgs = imgs.to(device, non_blocking=True)
        torch.cuda.current_stream().wait_stream(transfer_stream)
        with torch.cuda.stream(compute_stream):
            logits = model(imgs)

    On the A4000 (PCIe 4.0 x16 bus), this overlaps CPU→GPU transfers with
    the previous batch's backward pass, hiding ~15% of transfer latency.
    Only matters when batch sizes are small relative to compute time.
    For our large 640px patches the compute already dominates, so the gain
    is modest (~3-5%) — but it's free to enable.
    """
    if not torch.cuda.is_available():
        return None, None
    return torch.cuda.Stream(), torch.cuda.Stream()


def clear_cuda_cache():
    """Free cached but unused VRAM — call between stages."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc

        gc.collect()
        print(f"  Cache cleared. {vram_stats()}")
