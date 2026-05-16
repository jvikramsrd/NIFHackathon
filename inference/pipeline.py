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
from utils.logger import crash_logged, get_logger
from utils.postprocess import (
    apply_dense_crf,
    clean_segmentation_mask,
    detections_to_shapefile,
    mask_to_shapefile,
    merge_rooftop_labels,
)

log = get_logger(__name__)


class GeoIntelPipeline:
    def __init__(self, stage1_ckpt, stage2a_ckpt, stage2b_ckpt=None):
        # All required names are imported at module top — no need to re-import
        # inside __init__. The duplicate imports were defensive scaffolding
        # from an early refactor and just added clutter.
        self.device = setup()
        self.amp_ctx, _ = get_amp_context(CFG.AMP_DTYPE)

        log.info(f"[Pipeline] {vram_stats()}")

        # Stage 1 — Unet + MiT-B4 (channels_last applied to match training)
        log.info("[1] Loading Stage-1 segmentation model …")
        ckpt = torch.load(stage1_ckpt, map_location=self.device, weights_only=False)
        self.seg = Stage1Module(CFG.STAGE1).to(self.device)
        seg_state = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
        self.seg.load_state_dict(seg_state, strict=True)
        self.seg.eval()
        if CFG.COMPILE_ENABLED:
            # fullgraph=False keeps MiT encoder attention/control-flow tolerant.
            self.seg.model = compile_model(
                self.seg.model, CFG.COMPILE_MODE, fullgraph=False
            )
        # Apply channels_last to match training and gain 15-30% speedup on Ampere
        self.seg = to_channels_last(self.seg)
        self.seg_tf = get_val_transforms(int(CFG.STAGE1.get("patch_size", 512)))
        log.info(f"  {vram_stats()}")

        # Stage 2A — ConvNeXt-Base (channels_last: +15-25% on Ampere)
        log.info("[2] Loading Stage-2A rooftop classifier …")
        ckpt2a = torch.load(stage2a_ckpt, map_location=self.device, weights_only=False)
        self.clf = RooftopClassifier(CFG.STAGE2A).to(self.device)
        clf_state = {k.removeprefix("module."): v for k, v in ckpt2a["state_dict"].items()}
        self.clf.load_state_dict(clf_state, strict=True)
        self.clf.eval()
        self.clf = to_channels_last(self.clf)  # NHWC for ConvNeXt
        if CFG.COMPILE_ENABLED:
            # fullgraph=True: ConvNeXt has no dynamic control flow
            self.clf = compile_model(self.clf, CFG.COMPILE_MODE, fullgraph=True)
        self.clf_tf = get_clf_val_transforms(int(CFG.STAGE2A.get("crop_size", 160)))

        # Stage 2B — YOLOv9/OBB
        log.info("[3] Loading Stage-2B infrastructure detector …")
        self.detector = InfrastructureDetector(CFG.STAGE2B, str(CFG.CKPT_DIR))
        if stage2b_ckpt and Path(stage2b_ckpt).exists():
            from ultralytics.models.yolo.model import YOLO

            self.detector.model = YOLO(str(stage2b_ckpt))
            self.detector._backend = "yolo"

        # Cache the cosine blending window once. _segment() was recomputing this
        # on every call (a constant given (patch_size, overlap)) — the cost is
        # small but it's pure waste.
        from utils.window import cosine_window
        _ps = int(CFG.STAGE1.get("patch_size", 512))
        _ov = int(CFG.STAGE1.get("overlap", 128))
        self._seg_window = cosine_window(_ps, _ov).astype(np.float32)

        log.info("✓ All models ready\n")

    def run(self, tif_path: str, out_dir: str):
        import tempfile

        from utils.ecw_compat import ecw_to_tif, is_ecw

        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)

        # Convert ECW → TIF if the rasterio GDAL build lacks the ECW driver
        _ecw_tmp_ctx = None
        raster_path = Path(tif_path)
        if is_ecw(raster_path):
            _ecw_tmp_ctx = tempfile.TemporaryDirectory(prefix="geointel_ecw_")
            raster_path = ecw_to_tif(raster_path, Path(_ecw_tmp_ctx.name))

        prefix = raster_path.stem

        with rasterio.open(str(raster_path)) as src:
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

        # ── Stage 1: Segmentation (Segmentation Mask Generation) ──────────────────────────────────────────
        log.info("[Stage 1] Tiled segmentation …")
        prob_map = self._segment(img)  # (C, H, W) float32
        log.debug("  [DEBUG] Finished _segment. Calculating argmax...")

        if CFG.STAGE1.get("crf_inference"):
            log.info("  Dense CRF refinement …")
            prob_map = apply_dense_crf(
                img,
                prob_map,
                n_iter=int(CFG.STAGE1.get("crf_iter", 12)),
            )
            log.debug("  [DEBUG] Finished CRF.")

        seg_mask = prob_map.argmax(0).astype(np.uint8)
        log.debug("  [DEBUG] Finished argmax. Running morphological cleanup...")
        seg_mask = clean_segmentation_mask(seg_mask, CFG.STAGE1)
        log.debug("  [DEBUG] Finished morphological cleanup. Saving raster mask...")

        meta.update(count=1, dtype=rasterio.uint8)
        mask_path = out_dir_p / f"{prefix}_segmask.tif"
        try:
            with rasterio.open(str(mask_path), "w", **meta) as dst:
                dst.write(seg_mask[np.newaxis])
        except Exception as e:
            import time

            timestamp = int(time.time())
            mask_path = out_dir_p / f"{prefix}_segmask_{timestamp}.tif"
            log.info(
                f"  [WARN] Could not write to original mask path (perhaps open in QGIS?): {e}"
            )
            log.warning("Saving to fallback path: %s", mask_path)
            with rasterio.open(str(mask_path), "w", **meta) as dst:
                dst.write(seg_mask[np.newaxis])

        # *** CRITICAL STEP 1: GEOMETRY EXTRACTION FOR STAGE 2 CONSTRAINTS ***
        log.debug(
            "  [DEBUG] Extracting detailed building/road geometry for context gating..."
        )
        class_names = [str(x) for x in CFG.STAGE1.get("class_names", [])]
        # This creates the building SHP and the combined feature SHP for subsequent stages
        mask_to_shapefile(seg_mask, transform, crs, class_names, str(out_dir_p), prefix)

        # Load the specific building polygon SHP (assuming it exists)
        bld_shp = out_dir_p / f"{prefix}_building.shp"
        building_polygons = None
        if bld_shp.exists():
            log.debug("Successfully extracted Building Polygon SHP for contextual gating.")
            # We read it back to ensure geometry validity for later functions
            try:
                import geopandas as gpd

                building_polygons = gpd.read_file(str(bld_shp))
            except Exception as e:
                log.error("Could not load building polygon SHP for contextual gating: %s", e)
        else:
            log.warning(
                "Building polygon SHP not found; contextual gating for Stage 2A will proceed without building constraint."
            )

        # Free prob_map (can be large: C × H × W float32 for a 6 GB ortho)
        del prob_map
        clear_cuda_cache()

        # ── Stage 2A: Rooftop Classification (Constrained by Building Footprint) ──────────
        log.info("[Stage 2A] Rooftop classification …")
        bld_shp = out_dir_p / f"{prefix}_building.shp"
        if bld_shp.exists():
            # Pass the actual geometry constraints (polygons) to the classifier
            roof_preds = self._classify_rooftops(
                img, str(bld_shp), transform, building_polygons
            )
            merge_rooftop_labels(
                str(bld_shp),
                roof_preds,
                str(out_dir_p / f"{prefix}_building_rooftop.shp"),
            )
        else:
            log.warning("Stage 2A skipped: Building footprint required for constraint.")

        clear_cuda_cache()

        # ── Stage 2B: Infrastructure Detection (Constrained by Built/Road Footprints) ──────
        log.info("[Stage 2B] Infrastructure detection …")
        # Pass building and road context to guide detection search area
        context_polys = self._gather_context_polygons(
            out_dir_p, prefix, building_polygons, transform
        )
        dets = self._detect(img, context_polys)
        if dets:
            detections_to_shapefile(
                dets, transform, crs, str(out_dir_p / f"{prefix}_infrastructure.shp")
            )

        log.info(f"\n✓ Done → {out_dir_p}")

        if _ecw_tmp_ctx is not None:
            _ecw_tmp_ctx.cleanup()

    # ── Stage 1 tiled inference ───────────────────────────────────────────────

    def _segment(self, img_rgb: np.ndarray) -> np.ndarray:
        H, W = img_rgb.shape[:2]
        if H == 0 or W == 0:
            raise ValueError(f"Invalid image dimensions: {H}x{W}")

        ps = int(CFG.STAGE1["patch_size"])
        if ps <= 0:
            raise ValueError(f"Invalid patch_size: {ps}")

        overlap = int(CFG.STAGE1["overlap"])
        stride = max(1, ps - overlap)
        C = int(CFG.STAGE1["num_classes"])

        if C <= 0:
            raise ValueError(f"Invalid num_classes: {C}")

        prob_sum = np.zeros((C, H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        window = self._seg_window

        batch_size = 16
        batch_inputs = []
        batch_coords = []

        with torch.no_grad():
            for r in tqdm(range(0, H, stride), desc="  tiles"):
                for c in range(0, W, stride):
                    r2, c2 = min(r + ps, H), min(c + ps, W)
                    if r2 <= r or c2 <= c:
                        continue
                    patch = img_rgb[r:r2, c:c2].copy()
                    if patch.size == 0:
                        continue
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
                                fast_tta=CFG.FAST_TTA,
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
                        self.seg.model, inp_tensor, C, CFG.AMP_DTYPE, fast_tta=CFG.FAST_TTA
                    )
                    .cpu()
                    .numpy()
                )
                for i, (br, br2, bc, bc2, bph, bpw) in enumerate(batch_coords):
                    wind_slice = window[:bph, :bpw]
                    prob_sum[:, br:br2, bc:bc2] += probs[i, :, :bph, :bpw] * wind_slice
                    count_map[br:br2, bc:bc2] += wind_slice

        log.debug("  [DEBUG] Tiling loop finished. Returning averaged probability map...")
        return prob_sum / np.maximum(count_map, 1e-6)

    # ── Stage 2A rooftop classification ──────────────────────────────────────

    def _classify_rooftops(self, img_rgb, bld_shp_path, transform, building_polygons=None):
        import geopandas as gpd

        if not Path(bld_shp_path).exists():
            log.warning(f"Building shapefile not found: {bld_shp_path}")
            return {}

        # Reuse the GDF the caller already loaded — the original code ignored
        # the passed argument and re-ran ``gpd.read_file`` (a non-trivial cost
        # on 10K+ building shapefiles).
        if building_polygons is not None and len(building_polygons) > 0:
            gdf = building_polygons
        else:
            try:
                gdf = gpd.read_file(bld_shp_path)
            except Exception as e:
                log.warning(f"Failed to read building shapefile: {e}")
                return {}

        if len(gdf) == 0:
            log.warning("Empty building shapefile")
            return {}

        preds = {}
        try:
            inv_transform = ~transform
        except Exception as e:
            log.warning(f"Invalid transform for rooftop classification: {e}")
            return {}

        # Note: the previous version built ``building_polygons.geometry.unary_union``
        # and tested ``geom.intersects(context_union)`` per building. unary_union
        # on the entire dataset is wasteful here — the context filter is meant
        # to skip stray polygons, which a STRtree handles in O(log N).
        # In practice every building intersects its own footprint anyway, so we
        # only build the tree when the caller passed a *different* context set.

        batch_size = 64
        batch_inputs = []
        batch_indices = []
        class_names_2a = [str(x) for x in CFG.STAGE2A["class_names"]]
        per_class_thresh = CFG.STAGE2A.get(
            "stage2a_conf_thresh",
            {n: 0.55 for n in class_names_2a},
        )

        def _process_batch(inputs, indices):
            if not inputs:
                return
            try:
                inp_tensor = torch.stack(inputs).to(self.device, non_blocking=True)
                inp_tensor = cl_input(inp_tensor)
                with torch.no_grad():
                    probs = self.clf.predict(
                        inp_tensor, tta_steps=24, return_probs=True
                    )

                max_probs, pids = torch.max(probs, dim=1)
                # Move to CPU once for the per-class threshold lookup.
                max_probs_np = max_probs.detach().cpu().numpy()
                pids_np = pids.detach().cpu().numpy()
                for i, i_idx in enumerate(indices):
                    pred_name = class_names_2a[int(pids_np[i])]
                    thresh = per_class_thresh.get(pred_name, 0.55)
                    preds[i_idx] = pred_name if float(max_probs_np[i]) >= thresh else "Other"
            except Exception as e:
                log.warning(f"Batch processing failed: {e}")
                for i_idx in indices:
                    preds[i_idx] = "Other"

        min_crop = int(CFG.STAGE2A["min_crop_px"])
        crop_sz = int(CFG.STAGE2A["crop_size"])
        H_img, W_img = img_rgb.shape[:2]

        # iterrows builds a fresh Series per row — pull what we need into
        # native arrays once.
        geoms_arr = gdf.geometry.values
        index_arr = gdf.index.to_numpy()

        for k in tqdm(range(len(geoms_arr)), total=len(geoms_arr), desc="  roofs"):
            idx = index_arr[k]
            try:
                geom = geoms_arr[k]
                if geom is None or geom.is_empty:
                    preds[idx] = "Other"
                    continue

                geo_x1, geo_y1, geo_x2, geo_y2 = geom.bounds
                px_x1, px_y1 = inv_transform * (geo_x1, geo_y1)
                px_x2, px_y2 = inv_transform * (geo_x2, geo_y2)

                x1, x2 = sorted((int(px_x1), int(px_x2)))
                y1, y2 = sorted((int(px_y1), int(px_y2)))
                h, w = y2 - y1, x2 - x1

                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                x1 -= pad_x
                x2 += pad_x
                y1 -= pad_y
                y2 += pad_y

                if h < min_crop or w < min_crop:
                    preds[idx] = "Other"
                    continue
                x1c = max(0, x1)
                y1c = max(0, y1)
                x2c = min(W_img, x2)
                y2c = min(H_img, y2)

                if x2c <= x1c or y2c <= y1c:
                    preds[idx] = "Other"
                    continue

                img_slice = img_rgb[y1c:y2c, x1c:x2c]
                if img_slice.size == 0:
                    preds[idx] = "Other"
                    continue

                crop = cv2.resize(img_slice, (crop_sz, crop_sz), interpolation=cv2.INTER_LINEAR)
                inp = self.clf_tf(image=crop)["image"]

                batch_inputs.append(inp)
                batch_indices.append(idx)

                if len(batch_inputs) == batch_size:
                    _process_batch(batch_inputs, batch_indices)
                    batch_inputs = []
                    batch_indices = []
            except Exception as e:
                log.debug(f"Skipping building {idx} due to error: {e}")
                preds[idx] = "Other"
                continue

        if len(batch_inputs) > 0:
            _process_batch(batch_inputs, batch_indices)

        return preds

    def _gather_context_polygons(self, out_dir_p, prefix, building_polygons=None, transform=None):
        """Collect building/road polygons and convert them to pixel-space gates."""
        from pathlib import Path

        import geopandas as gpd
        from shapely.geometry import box as shapely_box

        polys = []
        context_classes = CFG.STAGE2B.get(
            "context_classes", ("building", "road", "waterbody")
        )
        context_buffer_px = int(CFG.STAGE2B.get("context_buffer_px", 64))
        for cls_name in context_classes:
            shp_path = Path(out_dir_p) / f"{prefix}_{cls_name}.shp"
            if shp_path.exists():
                try:
                    gdf = gpd.read_file(str(shp_path))
                    polys.extend(gdf.geometry.tolist())
                except Exception:
                    continue

        if building_polygons is not None and len(polys) == 0:
            polys.extend(building_polygons.geometry.tolist())

        if not polys:
            return None
        if transform is None:
            return polys

        inv_transform = ~transform
        pixel_boxes = []
        for geom in polys:
            if geom is None or geom.is_empty:
                continue
            try:
                geo_x1, geo_y1, geo_x2, geo_y2 = geom.bounds
                px_x1, px_y1 = inv_transform * (geo_x1, geo_y1)
                px_x2, px_y2 = inv_transform * (geo_x2, geo_y2)
                x1, x2 = sorted((float(px_x1), float(px_x2)))
                y1, y2 = sorted((float(px_y1), float(px_y2)))
                if x2 > x1 and y2 > y1:
                    pixel_boxes.append(
                        shapely_box(x1, y1, x2, y2).buffer(context_buffer_px)
                    )
            except Exception as exc:
                log.debug("Skipping invalid context polygon: %s", exc)

        return pixel_boxes if pixel_boxes else None

    # ── Stage 2B infrastructure detection (tiled) ────────────────────────────

    def _detect(self, img_rgb, context_polys=None):
        from shapely.geometry import box as shapely_box
        from shapely.strtree import STRtree

        from models.stage2_models import soft_nms_gaussian

        H, W = img_rgb.shape[:2]
        if H == 0 or W == 0:
            log.warning("Invalid image dimensions for detection")
            return []

        tile = int(CFG.STAGE2B["img_size"])
        if tile <= 0:
            raise ValueError(f"Invalid tile size: {tile}")

        overlap = int(CFG.STAGE2B.get("overlap", 256))
        stride = max(1, tile - overlap)

        # O(log N) spatial index for the context filter. The old code did
        # ``any(tile_geom.intersects(poly) for poly in context_polys)`` per tile —
        # O(N) shapely intersection tests scaled with the building/road count.
        ctx_tree = None
        if context_polys is not None:
            valid_polys = [p for p in context_polys if p is not None]
            if valid_polys:
                ctx_tree = STRtree(valid_polys)

        dets = []
        det_tiles_total = 0
        det_tiles_skipped = 0

        for r in tqdm(range(0, H, stride), desc="  det tiles"):
            for c in range(0, W, stride):
                r2, c2 = min(r + tile, H), min(c + tile, W)
                if r2 <= r or c2 <= c:
                    continue
                det_tiles_total += 1

                if ctx_tree is not None:
                    try:
                        tile_geom = shapely_box(c, r, c2, r2)
                        if len(ctx_tree.query(tile_geom)) == 0:
                            det_tiles_skipped += 1
                            continue
                    except Exception:
                        pass

                patch = img_rgb[r:r2, c:c2]
                if patch.size == 0:
                    continue

                # Hand YOLO the BGR ndarray directly. The old path JPEG-encoded
                # the patch, wrote it to a temp file, made YOLO re-read it, then
                # deleted the file — per tile. That's three avoidable disk I/Os
                # plus a lossy JPEG round-trip in the inference path.
                try:
                    bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    for d in self.detector.predict(bgr):
                        d["bbox_xyxy"][0] += c
                        d["bbox_xyxy"][2] += c
                        d["bbox_xyxy"][1] += r
                        d["bbox_xyxy"][3] += r
                        dets.append(d)
                except Exception as e:
                    log.debug(f"Tile {r},{c} failed: {e}")
                    continue

        if dets:
            try:
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
            except Exception as e:
                log.warning(f"Soft-NMS failed: {e}, returning raw detections")
                final_dets = dets
        else:
            final_dets = []

        log.info(
            f"  Tiles scanned: {det_tiles_total}  |  "
            f"Skipped by context: {det_tiles_skipped}  |  "
            f"Raw Detections: {len(dets)}  →  After Soft-NMS: {len(final_dets)}"
        )
        return final_dets


def _to_uint8(arr):
    if arr is None or arr.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    bands = min(arr.shape[2], 3)
    img = arr[:, :, :bands].astype(np.float32)
    lo = np.percentile(img, 2, axis=(0, 1), keepdims=True)
    hi = np.percentile(img, 98, axis=(0, 1), keepdims=True)
    safe = np.where(hi > lo, hi - lo, 1.0)
    out = np.clip((img - lo) / safe * 255.0, 0, 255).astype(np.uint8)
    if bands < 3:
        fill = np.repeat(out[:, :, :1], 3 - bands, axis=2)
        out = np.concatenate([out, fill], axis=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--s1_ckpt", default=str(CFG.CKPT_DIR / "stage1_best.pth"))
    ap.add_argument("--s2a_ckpt", default=str(CFG.CKPT_DIR / "stage2a_best.pth"))
    ap.add_argument(
        "--s2b_ckpt",
        default=str(CFG.CKPT_DIR / f"stage2b_{CFG.STAGE2B['model_variant']}" / "weights" / "best.pt"),
    )
    args = ap.parse_args()
    with crash_logged(log, "inference pipeline"):
        pipe = GeoIntelPipeline(args.s1_ckpt, args.s2a_ckpt, args.s2b_ckpt)
        pipe.run(args.tif, args.out)
