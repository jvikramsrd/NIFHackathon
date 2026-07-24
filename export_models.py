"""Export trained models to ONNX and, when available, TensorRT engines."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

import config as CFG
from models.stage1_segmentation import Stage1Module
from models.stage2_models import RooftopClassifier
from utils.checkpointing import clean_state_dict
from utils.core import get_logger

log = get_logger(__name__)


def export_stage1_onnx(
    ckpt_path: Path = CFG.CKPT_DIR / f"stage1_{CFG.STAGE1['encoder']}_best.pth",
    out_path: Path = CFG.CKPT_DIR / "stage1.onnx",
    opset: int = 17,
) -> Path:
    # Override encoder_weights so smp doesn't try to download ImageNet weights
    # every export call — the checkpoint we load below already contains
    # fine-tuned weights.
    cfg = {**CFG.STAGE1, "encoder_weights": None}
    model = Stage1Module(cfg).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(clean_state_dict(raw_state, model.state_dict()), strict=False)
    dummy = torch.randn(1, 3, int(CFG.STAGE1["patch_size"]), int(CFG.STAGE1["patch_size"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=opset,
    )
    log.info("Stage 1 ONNX exported to %s", out_path)
    return out_path


def export_stage2a_onnx(
    ckpt_path: Path = CFG.CKPT_DIR / "stage2a_best.pth",
    out_path: Path = CFG.CKPT_DIR / "stage2a.onnx",
    opset: int = 17,
) -> Path:
    # Skip ImageNet download — the checkpoint already supplies fine-tuned weights.
    cfg = {**CFG.STAGE2A, "pretrained": False}
    model = RooftopClassifier(cfg).eval()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(clean_state_dict(raw_state, model.state_dict()), strict=False)
    crop_size = int(CFG.STAGE2A["crop_size"])
    dummy = torch.randn(1, 3, crop_size, crop_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    log.info("Stage 2A ONNX exported to %s", out_path)
    return out_path


def export_stage2b(
    ckpt_path: Path,
    fmt: str = "onnx",
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(ckpt_path))
    result = model.export(format=fmt, imgsz=int(CFG.STAGE2B["img_size"]))
    out_path = Path(result)
    log.info("Stage 2B exported to %s", out_path)
    return out_path


def export_tensorrt_from_onnx(onnx_path: Path, engine_path: Path | None = None) -> Path | None:
    trtexec = shutil.which("trtexec")
    if trtexec is None:
        log.warning("TensorRT export skipped: trtexec not found on PATH")
        return None
    engine_path = engine_path or onnx_path.with_suffix(".engine")
    subprocess.run(
        [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            "--fp16",
        ],
        check=True,
    )
    log.info("TensorRT engine exported to %s", engine_path)
    return engine_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["1", "2a", "2b"], required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    args = parser.parse_args()

    if args.stage == "1":
        onnx_path = export_stage1_onnx(args.ckpt or CFG.CKPT_DIR / f"stage1_{CFG.STAGE1['encoder']}_best.pth")
        if args.format == "engine":
            export_tensorrt_from_onnx(onnx_path)
    elif args.stage == "2a":
        onnx_path = export_stage2a_onnx(args.ckpt or CFG.CKPT_DIR / "stage2a_best.pth")
        if args.format == "engine":
            export_tensorrt_from_onnx(onnx_path)
    else:
        ckpt = args.ckpt or CFG.CKPT_DIR / f"stage2b_{CFG.STAGE2B['model_variant']}" / "weights" / "best.pt"
        export_stage2b(ckpt, fmt=args.format)


if __name__ == "__main__":
    main()
