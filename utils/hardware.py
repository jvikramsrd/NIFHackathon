"""
utils/hardware.py
──────────────────
One-stop hardware configuration for Windows + RTX A4000 + i9-13900.
Call `setup()` at the top of every train/inference script.
"""

import os
import platform
from typing import Optional

import torch
import torch.nn as nn
from utils.logger import get_logger

log = get_logger(__name__)


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

        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        except ValueError:
            pass

        os.environ.setdefault("OMP_NUM_THREADS", "8")
        os.environ.setdefault("MKL_NUM_THREADS", "8")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

    # ── GDAL / rasterio I/O performance ──────────────────────────────────────
    # These env vars only take effect before the GDAL driver registry is built.
    # GDAL_CACHEMAX: per-process block cache (MB). Default 40 MB — way too low
    #   for multi-GB orthos. 2048 MB amortises strip reads on TIFFs.
    # GDAL_DISABLE_READDIR_ON_OPEN: skip scanning the entire sibling-file list
    #   when opening a raster (huge speedup on folders with thousands of files).
    # CPL_VSIL_CURL_USE_HEAD: disable HEAD requests on remote VSI (no-op for
    #   local files, but cheap to leave).
    # See https://gdal.org/user/configoptions.html
    os.environ.setdefault("GDAL_CACHEMAX", "2048")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_USE_HEAD", "NO")
    os.environ.setdefault("GDAL_NUM_THREADS", "ALL_CPUS")

    if not _SETUP_DONE:
        try:
            torch.set_num_threads(8)
        except (RuntimeError, ValueError):
            pass
        try:
            torch.set_num_interop_threads(4)
        except (RuntimeError, ValueError):
            pass
        _SETUP_DONE = True

    # ── CUDA + Ampere flags ──────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log.info("[HW] WARNING: CUDA not available — running on CPU")
        return torch.device("cpu")

    # Defense-in-depth: if config.py wasn't imported first (e.g. unit tests,
    # standalone scripts), still apply the A4000-friendly allocator policy
    # before any CUDA allocation happens.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256",
    )

    try:
        device = torch.device("cuda:0")
    except Exception:
        log.info("[HW] WARNING: Cannot create CUDA device — falling back to CPU")
        return torch.device("cpu")

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    except Exception:
        pass

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except (AttributeError, Exception):
        pass

    # ── Reproducibility ─────────────────────────────────────────────────────
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # ── Diagnostics ─────────────────────────────────────────────────────────
    if verbose:
        try:
            gpu = torch.cuda.get_device_properties(device)
            vram_gb = gpu.total_memory / 1024**3
            cc = f"{gpu.major}.{gpu.minor}"
            log.info("=" * 60)
            log.info(f"  GPU              : {gpu.name}")
            log.info(f"  VRAM             : {vram_gb:.1f} GB")
            log.info(f"  SM count         : {gpu.multi_processor_count}  (CC {cc})")
            log.info(f"  CUDA             : {torch.version.cuda}")
            log.info(f"  PyTorch          : {torch.__version__}")
            log.info(f"  bf16 native      : {gpu.major >= 8}")
            log.info(f"  TF32             : {torch.backends.cudnn.allow_tf32}")
            log.info(f"  matmul precision : {torch.get_float32_matmul_precision()}")
            log.info(f"  intra threads    : {torch.get_num_threads()}")
            log.info(f"  interop threads  : {torch.get_num_interop_threads()}")
            log.info("=" * 60)
        except Exception as e:
            log.info(f"[HW] Could not get GPU diagnostics: {e}")

    return device


def worker_init_fn(worker_id: int):
    """
    DataLoader worker initialiser.

    1. Cap PyTorch's intra-op pool inside this worker to 1 thread.
       ``torch.set_num_threads`` is honoured at runtime, so this works even
       though the parent process set 8 threads for itself.
    2. Disable OpenCV's internal threading (``setNumThreads(0)``) and OpenCL,
       so each cv2 call inside a worker doesn't spawn ``num_cores`` threads
       on top of the N DataLoader workers.
    3. Force the OpenMP / BLAS env vars to "1" (overwrite the inherited "8"
       from the parent). Some BLAS libs read these at thread-pool creation
       time — if numpy hasn't been touched yet in this worker, this still
       takes effect on first use.
    4. Pin each worker to a fast P-core (LP 0-15) on the i9-13900K.
    """
    try:
        import cv2  # noqa: PLC0415
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
    except Exception:
        pass

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # Force-overwrite (not setdefault) — the parent set 8 for itself, but the
    # whole point of this function is to cap workers to 1.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"

    try:
        import psutil  # noqa: PLC0415

        p = psutil.Process(os.getpid())
        p_core_lps = list(range(16))
        lp = p_core_lps[worker_id % len(p_core_lps)]
        p.cpu_affinity([lp])
    except Exception:
        pass


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
        log.info(f"  torch.compile() applied (mode={mode}, fullgraph={fullgraph})")
        return compiled
    except Exception as e:
        log.info(f"  torch.compile() skipped: {e}")
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
        log.info("  channels_last (NHWC) applied")
    except Exception as e:
        log.info(f"  channels_last skipped: {e}")
    return model


def cl_input(tensor: torch.Tensor) -> torch.Tensor:
    """Convert 4D batch tensor to channels_last before model forward pass."""
    return tensor.to(memory_format=torch.channels_last) if tensor.ndim == 4 else tensor


def get_amp_context(dtype: torch.dtype = torch.bfloat16):
    """
    Return the right AMP context and a (possibly None) GradScaler.

    Returns ``(autocast_ctx, scaler)`` where:
      - ``scaler is None`` for bfloat16 (its 8-bit exponent matches fp32, so
        attention softmax probabilities don't underflow and gradients don't
        need scaling). This is the default for Ampere/Ada/Hopper.
      - ``scaler is a GradScaler`` for float16 (5-bit exponent → tiny attention
        gradients can flush to 0; the scaler multiplies the loss by a dynamic
        factor before backward so they survive).

    Callers MUST use the scaler when it is non-None, otherwise float16
    training will silently produce NaN losses on attention-heavy models.
    See ``maybe_backward`` / ``maybe_step`` helpers below for the safe pattern.
    """
    # torch.amp.* is the PyTorch 2.x API; the cuda.amp.* re-exports are
    # deprecated but still work on older builds.
    try:
        from torch.amp import GradScaler, autocast
    except ImportError:
        from torch.cuda.amp import GradScaler, autocast

    from contextlib import nullcontext

    cuda_available = torch.cuda.is_available()

    if dtype == torch.bfloat16 and cuda_available:
        ctx = autocast("cuda", dtype=torch.bfloat16)
        scaler = None  # bf16 doesn't underflow → no scaling needed
    elif dtype == torch.float16 and cuda_available:
        ctx = autocast("cuda", dtype=torch.float16)
        scaler = GradScaler("cuda")
    else:
        # fp32, unsupported dtypes, or CUDA absent: no autocast, no scaler.
        # nullcontext keeps the training-loop ``with amp_ctx:`` blocks
        # compiling cleanly even when there is no AMP work to do.
        ctx = nullcontext()
        scaler = None

    return ctx, scaler


def maybe_backward(loss: torch.Tensor, scaler) -> None:
    """Scale-then-backward when ``scaler`` is non-None, plain backward otherwise.

    Use this everywhere you would have written ``loss.backward()`` in a script
    that wants to be safe across bf16 and fp16.
    """
    if scaler is not None:
        scaler.scale(loss).backward()
    else:
        loss.backward()


def maybe_step(optimiser, scaler, max_grad_norm: Optional[float] = None,
               parameters=None) -> Optional[torch.Tensor]:
    """Take an optimiser step, honouring the AMP scaler when present.

    Returns the pre-clip gradient norm (``torch.Tensor`` scalar) when
    ``max_grad_norm`` is provided, else None. Callers can act on the norm
    (e.g. skip the step on a spike) before discarding it.
    """
    grad_norm = None
    if scaler is not None:
        # Unscale FIRST so clip_grad_norm sees the true gradients.
        scaler.unscale_(optimiser)
        if max_grad_norm is not None and parameters is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        scaler.step(optimiser)
        scaler.update()
    else:
        if max_grad_norm is not None and parameters is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
        optimiser.step()
    return grad_norm


class EMA:
    """
    Exponential Moving Average of model weights.

    Shadow weights are kept on the same device as the model (GPU) so that each
    update is a pure GPU in-place operation with no PCIe round-trips.
    `torch._foreach_mul_` / `torch._foreach_add_` batch all parameter updates
    into two CUDA kernels instead of one kernel-pair per parameter, cutting
    update overhead from ~500 kernel launches to 2.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.decay = decay
        self.shadow: dict = {}
        self._backup: dict = {}
        self._register(model)

    def _register(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    @torch.no_grad()
    def update(self, model: nn.Module):
        shadows, params = [], []
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                shadows.append(self.shadow[name])
                params.append(param.data)
        if not shadows:
            return
        # torch.lerp(start, end, w) == start + w*(end-start), so
        # _foreach_lerp_(shadow, param, 1-decay) folds the multiply+add
        # into a single fused kernel.  Two _foreach_* kernels (mul_ + add_)
        # are still issued by the fallback path.
        w = 1.0 - self.decay
        try:
            torch._foreach_lerp_(shadows, params, w)
        except (AttributeError, RuntimeError):
            try:
                torch._foreach_mul_(shadows, self.decay)
                torch._foreach_add_(shadows, params, alpha=w)
            except (AttributeError, RuntimeError):
                for s, p in zip(shadows, params):
                    s.mul_(self.decay).add_(p, alpha=w)

    def apply_shadow(self, model: nn.Module):
        """Swap EMA weights into model for validation."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """Restore training weights after validation."""
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
        log.info(f"  Cache cleared. {vram_stats()}")


def get_yolo_device() -> str:
    """Return the device string Ultralytics YOLO accepts for the active backend.

    YOLO's ``device=`` kwarg expects: a GPU ordinal string like ``"0"`` for CUDA,
    ``"mps"`` for Apple Silicon, or ``"cpu"``. Hardcoding ``"0"`` crashes on any
    machine without an NVIDIA GPU.
    """
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_inference_device_str() -> str:
    """Return a torch device string for libraries that take a string (e.g. SAHI).

    Mirrors the precedence in ``config.py``: CUDA → MPS → CPU.
    """
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
