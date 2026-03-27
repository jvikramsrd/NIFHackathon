"""
inference/pipeline.py  (A4000-optimised)
──────────────────────────────────────────
Full two-stage inference on a new village TIF.
Optimisations:
  • bfloat16 throughout
  • torch.compile (reduce-overhead) for segmentation model
  • 16-fold TTA for Stage 1
  • Tiled inference with overlap-add stitching
  • Stage 2B: 1280-px YOLO-l tiles
"""

import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from tqdm import tqdm

import config as CFG
from data.dataset import get_clf_val_transforms, get_val_transforms
from models.stage1_segmentation import Stage1Module, tta_predict
from models.stage2_models import InfrastructureDetector, RooftopClassifier
from utils.hardware import (
    cl_input,
    clear_cuda_cache,
    compile_model,
    get_amp_context,
    setup,
    to_channels_last,
    vram_stats,
)
from utils.postprocess import (
    apply_dense_crf,
    clean_segmentation_mask,
    detections_to_shapefile,
    mask_to_shapefile,
    merge_rooftop_labels,
)


class GeoIntelPipeline:
    def __init__(self, stage1_ckpt, stage2a_ckpt, stage2b_ckpt=None):
        self.device = setup()
        self.amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)

        print(f"[Pipeline] {vram_stats()}")

        # Stage 1 — Swin-B UNet++ (NO channels_last — transformer backbone)
        print("[1] Loading Stage-1 segmentation model …")
        ckpt = torch.load(stage1_ckpt, map_location=self.device, weights_only=False)
        self.seg = Stage1Module(CFG.STAGE1).to(self.device)
        self.seg.load_state_dict(ckpt["state_dict"], strict=False)
        self.seg.eval()
        if CFG.COMPILE_ENABLED:
            # fullgraph=False: Swin-B has conditional ops in attention
            self.seg.model = compile_model(
                self.seg.model, CFG.COMPILE_MODE, fullgraph=False
            )
        self.seg_tf = get_val_transforms(int(CFG.STAGE1["patch_size"]))  # type: ignore
        print(f"  {vram_stats()}")

        # Stage 2A — ConvNeXt-Base (channels_last: +15-25% on Ampere)
        print("[2] Loading Stage-2A rooftop classifier …")
        ckpt2a = torch.load(stage2a_ckpt, map_location=self.device, weights_only=False)
        self.clf = RooftopClassifier(CFG.STAGE2A).to(self.device)
        self.clf.load_state_dict(ckpt2a["state_dict"], strict=False)
        self.clf.eval()
        self.clf = to_channels_last(self.clf)  # NHWC for ConvNeXt
        if CFG.COMPILE_ENABLED:
            # fullgraph=True: ConvNeXt has no dynamic control flow
            self.clf = compile_model(self.clf, CFG.COMPILE_MODE, fullgraph=True)
        self.clf_tf = get_clf_val_transforms(int(CFG.STAGE2A["crop_size"]))  # type: ignore

        # Stage 2B — YOLOv8-l
        print("[3] Loading Stage-2B infrastructure detector …")
        self.detector = InfrastructureDetector(CFG.STAGE2B, str(CFG.CKPT_DIR))
        if stage2b_ckpt and Path(stage2b_ckpt).exists():
            from ultralytics.models.yolo.model import YOLO

            self.detector.model = YOLO(str(stage2b_ckpt))
            self.detector._backend = "yolo"

        print("✓ All models ready\n")

    def run(self, tif_path: str, out_dir: str):
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        prefix = Path(tif_path).stem

        with rasterio.open(tif_path) as src:
            meta = src.meta.copy()
            crs = src.crs
            transform = src.transform
            _H, _W = src.height, src.width
            bands = min(src.count, 3)
            img = src.read(list(range(1, bands + 1))).transpose(1, 2, 0)
            if bands < 3:
                img = np.stack([img[:, :, 0]] * 3, axis=-1)

        if img.dtype != np.uint8:
            img = _to_uint8(img)

        # ── Stage 1 ──────────────────────────────────────────────────────────
        print("[Stage 1] Tiled segmentation …")
        prob_map = self._segment(img)  # (C, H, W) float32
        print("  [DEBUG] Finished _segment. Calculating argmax...")

        if CFG.STAGE1.get("crf_inference"):
            print("  Dense CRF refinement …")
            prob_map = apply_dense_crf(
                img,
                prob_map,
                n_iter=int(CFG.STAGE1.get("crf_iter", 12)),  # type: ignore
            )
            print("  [DEBUG] Finished CRF.")

        seg_mask = prob_map.argmax(0).astype(np.uint8)
        print("  [DEBUG] Finished argmax. Running morphological cleanup...")
        seg_mask = clean_segmentation_mask(seg_mask, CFG.STAGE1)
        print("  [DEBUG] Finished morphological cleanup. Saving raster mask...")

        meta.update(count=1, dtype=rasterio.uint8)
        mask_path = out_dir_p / f"{prefix}_segmask.tif"
        try:
            with rasterio.open(str(mask_path), "w", **meta) as dst:
                dst.write(seg_mask[np.newaxis])
        except Exception as e:
            import time

            timestamp = int(time.time())
            mask_path = out_dir_p / f"{prefix}_segmask_{timestamp}.tif"
            print(
                f"  [WARN] Could not write to original mask path (perhaps open in QGIS?): {e}"
            )
            print(f"  [WARN] Saving to fallback path: {mask_path}")
            with rasterio.open(str(mask_path), "w", **meta) as dst:
                dst.write(seg_mask[np.newaxis])

        print("  [DEBUG] Saved raster mask. Converting to shapefile...")
        class_names = [str(x) for x in CFG.STAGE1["class_names"]]  # type: ignore
        mask_to_shapefile(seg_mask, transform, crs, class_names, str(out_dir_p), prefix)
        print("  [DEBUG] Converted to shapefile.")

        # Free prob_map (can be large: C × H × W float32 for a 6 GB ortho)
        del prob_map
        clear_cuda_cache()

        # ── Stage 2A ─────────────────────────────────────────────────────────
        print("[Stage 2A] Rooftop classification …")
        bld_shp = out_dir_p / f"{prefix}_building.shp"
        if bld_shp.exists():
            roof_preds = self._classify_rooftops(img, str(bld_shp), transform)
            merge_rooftop_labels(
                str(bld_shp),
                roof_preds,
                str(out_dir_p / f"{prefix}_building_rooftop.shp"),
            )

        clear_cuda_cache()

        # ── Stage 2B ─────────────────────────────────────────────────────────
        print("[Stage 2B] Infrastructure detection …")
        dets = self._detect(img)
        if dets:
            detections_to_shapefile(
                dets, transform, crs, str(out_dir_p / f"{prefix}_infrastructure.shp")
            )

        print(f"\n✓ Done → {out_dir_p}")

    # ── Stage 1 tiled inference ───────────────────────────────────────────────

    def _spline_window(self, window_size, overlap, power=2):
        """
        Create a 2D spline window for smooth blending.
        """
        intersection = int(overlap)
        wind_outer = (np.cos(np.pi * np.arange(intersection) / intersection) + 1) / 2
        wind = np.ones(window_size)
        wind[:intersection] = wind_outer[::-1]
        wind[-intersection:] = wind_outer
        wind = wind**power
        wind_2d = np.outer(wind, wind)
        return wind_2d

    def _segment(self, img_rgb: np.ndarray) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        ps = int(CFG.STAGE1["patch_size"])  # type: ignore
        # Reduce overlap for faster inference (smooth blending prevents seams anyway)
        overlap = min(int(CFG.STAGE1["overlap"]), 128)
        stride = ps - overlap  # type: ignore
        C = int(CFG.STAGE1["num_classes"])  # type: ignore
        prob_sum = np.zeros((C, H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        window = self._spline_window(ps, overlap)

        batch_size = 16
        batch_inputs = []
        batch_coords = []

        with torch.no_grad():
            for r in tqdm(range(0, H, stride), desc="  tiles"):
                for c in range(0, W, stride):
                    r2, c2 = min(r + ps, H), min(c + ps, W)
                    patch = img_rgb[r:r2, c:c2].copy()
                    ph, pw = patch.shape[:2]

                    if ph < ps or pw < ps:
                        pad = cv2.copyMakeBorder(
                            patch, 0, ps - ph, 0, ps - pw, cv2.BORDER_REFLECT_101
                        )
                    else:
                        pad = patch
                    aug = self.seg_tf(image=pad)
                    inp = aug["image"]

                    batch_inputs.append(inp)
                    batch_coords.append((r, r2, c, c2, ph, pw))

                    if len(batch_inputs) == batch_size:
                        inp_tensor = torch.stack(batch_inputs).to(self.device)
                        probs = (
                            tta_predict(
                                self.seg.model,
                                inp_tensor,
                                C,
                                CFG.AMP_DTYPE,
                                fast_tta=True,
                            )
                            .cpu()
                            .numpy()
                        )
                        for i, (br, br2, bc, bc2, bph, bpw) in enumerate(batch_coords):
                            wind_slice = window[:bph, :bpw]
                            prob_sum[:, br:br2, bc:bc2] += (
                                probs[i, :, :bph, :bpw] * wind_slice
                            )
                            count_map[br:br2, bc:bc2] += wind_slice
                        batch_inputs = []
                        batch_coords = []

            if len(batch_inputs) > 0:
                inp_tensor = torch.stack(batch_inputs).to(self.device)
                probs = (
                    tta_predict(
                        self.seg.model, inp_tensor, C, CFG.AMP_DTYPE, fast_tta=True
                    )
                    .cpu()
                    .numpy()
                )
                for i, (br, br2, bc, bc2, bph, bpw) in enumerate(batch_coords):
                    wind_slice = window[:bph, :bpw]
                    prob_sum[:, br:br2, bc:bc2] += probs[i, :, :bph, :bpw] * wind_slice
                    count_map[br:br2, bc:bc2] += wind_slice

        print("  [DEBUG] Tiling loop finished. Returning averaged probability map...")
        return prob_sum / np.maximum(count_map, 1e-6)

    # ── Stage 2A rooftop classification ──────────────────────────────────────

    def _classify_rooftops(self, img_rgb, bld_shp_path, transform):
        import geopandas as gpd

        gdf = gpd.read_file(bld_shp_path)
        preds = {}
        inv_transform = ~transform

        batch_size = 64
        batch_inputs = []
        batch_indices = []

        def _process_batch(inputs, indices):
            if not inputs:
                return
            inp_tensor = torch.stack(inputs).to(self.device)
            inp_tensor = cl_input(inp_tensor)
            with torch.no_grad():
                probs = self.clf.predict(
                    inp_tensor, int(CFG.STAGE2A["tta_steps"]), return_probs=True
                )  # type: ignore

            class_names_2a = [str(x) for x in CFG.STAGE2A["class_names"]]  # type: ignore
            max_probs, pids = torch.max(probs, dim=1)

            for i, i_idx in enumerate(indices):
                if max_probs[i].item() < 0.55:
                    preds[i_idx] = "Other"
                else:
                    preds[i_idx] = class_names_2a[pids[i].item()]

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  roofs"):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            geo_x1, geo_y1, geo_x2, geo_y2 = geom.bounds
            px_x1, px_y1 = inv_transform * (geo_x1, geo_y1)
            px_x2, px_y2 = inv_transform * (geo_x2, geo_y2)

            x1, x2 = sorted([int(px_x1), int(px_x2)])
            y1, y2 = sorted([int(px_y1), int(px_y2)])
            h, w = y2 - y1, x2 - x1

            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 -= pad_x
            x2 += pad_x
            y1 -= pad_y
            y2 += pad_y

            min_crop = int(CFG.STAGE2A["min_crop_px"])  # type: ignore
            if h < min_crop or w < min_crop:
                preds[idx] = "Other"
                continue
            x1c = max(0, x1)
            y1c = max(0, y1)
            x2c = min(img_rgb.shape[1], x2)
            y2c = min(img_rgb.shape[0], y2)
            crop_sz = int(CFG.STAGE2A["crop_size"])  # type: ignore

            img_slice = img_rgb[y1c:y2c, x1c:x2c]
            if (
                img_slice.size == 0
                or img_slice.shape[0] == 0
                or img_slice.shape[1] == 0
            ):
                preds[idx] = "Other"
                continue

            crop = cv2.resize(img_slice, (crop_sz, crop_sz))
            inp = self.clf_tf(image=crop)["image"]

            batch_inputs.append(inp)
            batch_indices.append(idx)

            if len(batch_inputs) == batch_size:
                _process_batch(batch_inputs, batch_indices)
                batch_inputs = []
                batch_indices = []

        if len(batch_inputs) > 0:
            _process_batch(batch_inputs, batch_indices)

        return preds

    # ── Stage 2B infrastructure detection (tiled) ────────────────────────────

    def _detect(self, img_rgb):
        import os
        import tempfile

        from models.stage2_models import soft_nms_gaussian

        tile = int(CFG.STAGE2B["img_size"])  # type: ignore
        overlap = int(CFG.STAGE2B.get("overlap", 256))  # type: ignore
        stride = tile - overlap

        H, W = img_rgb.shape[:2]
        dets = []

        # Use a proper temporary file path that works on Windows and Linux
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="geo_infra_tile_")
        os.close(tmp_fd)

        try:
            for r in tqdm(range(0, H, stride), desc="  det tiles"):
                for c in range(0, W, stride):
                    r2, c2 = min(r + tile, H), min(c + tile, W)
                    patch = img_rgb[r:r2, c:c2]
                    cv2.imwrite(tmp_path, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                    for d in self.detector.predict(tmp_path):
                        d["bbox_xyxy"][0] += c
                        d["bbox_xyxy"][2] += c
                        d["bbox_xyxy"][1] += r
                        d["bbox_xyxy"][3] += r
                        dets.append(d)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Per-class Soft-NMS — decays confidence of overlapping boxes instead
        # of hard-removing them, preserving closely-spaced infrastructure
        if dets:
            sigma = float(CFG.STAGE2B.get("soft_nms_sigma", 0.5))
            conf_thresh = float(CFG.STAGE2B["conf_thresh"])
            unique_classes = set(d["class_id"] for d in dets)
            final_dets = []

            for cls_id in unique_classes:
                cls_dets = [d for d in dets if d["class_id"] == cls_id]
                boxes = torch.tensor(
                    [d["bbox_xyxy"] for d in cls_dets], dtype=torch.float32
                )
                scores = torch.tensor(
                    [d["conf"] for d in cls_dets], dtype=torch.float32
                )
                keep_idx, keep_scores = soft_nms_gaussian(
                    boxes, scores, sigma=sigma, score_threshold=conf_thresh
                )
                for i, new_score in zip(keep_idx.tolist(), keep_scores.tolist()):
                    det = cls_dets[i].copy()
                    det["conf"] = new_score
                    final_dets.append(det)
        else:
            final_dets = []

        print(f"  Raw Detections: {len(dets)}  →  After Soft-NMS: {len(final_dets)}")
        return final_dets


def _to_uint8(arr):
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.ndim == 3:
        for i in range(arr.shape[2]):
            ch = arr[:, :, i].astype(np.float32)
            mn, mx = ch.min(), ch.max()
            out[:, :, i] = 0 if mx == mn else (ch - mn) / (mx - mn) * 255
    return out.astype(np.uint8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1_ckpt", default=str(CFG.CKPT_DIR / "stage1_best.pth"))
    ap.add_argument("--s2a_ckpt", default=str(CFG.CKPT_DIR / "stage2a_best.pth"))
    ap.add_argument(
        "--s2b_ckpt", default=str(CFG.CKPT_DIR / "stage2b_yolov8l/weights/best.pt")
    )
    args = ap.parse_args()
    pipe = GeoIntelPipeline(args.s1_ckpt, args.s2a_ckpt, args.s2b_ckpt)
    pipe.run(args.tif, args.out)
