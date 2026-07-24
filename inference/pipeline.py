import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from tqdm import tqdm
import config as CFG
from data.dataset import get_clf_val_transforms, get_val_transforms
from models import Stage1Module, RooftopClassifier, InfrastructureDetector
from utils.core import get_logger, setup
from utils.inference import sliding_window_inference
from utils.postprocess import (
    apply_dense_crf,
    clean_segmentation_mask,
    detections_to_shapefile,
    mask_to_shapefile,
    merge_rooftop_labels,
)
from utils.image_utils import to_uint8

log = get_logger(__name__)

class GeoIntelPipeline:
    def __init__(self, stage1_ckpt, stage2a_ckpt, stage2b_ckpt=None):
        self.device = setup()
        
        log.info("[1] Loading Stage-1 segmentation model ...")
        self.seg = Stage1Module(CFG.STAGE1).to(self.device)
        self.seg.load_state_dict(torch.load(stage1_ckpt, map_location=self.device), strict=False)
        self.seg.eval()
        self.seg_tf = get_val_transforms(int(CFG.STAGE1.get("patch_size", 512)))

        log.info("[2] Loading Stage-2A rooftop classifier ...")
        self.clf = RooftopClassifier(CFG.STAGE2A).to(self.device)
        self.clf.load_state_dict(torch.load(stage2a_ckpt, map_location=self.device), strict=False)
        self.clf.eval()
        self.clf_tf = get_clf_val_transforms(int(CFG.STAGE2A.get("crop_size", 160)))

        log.info("[3] Loading Stage-2B infrastructure detector ...")
        self.detector = InfrastructureDetector(CFG.STAGE2B, str(CFG.CKPT_DIR))
        if stage2b_ckpt and Path(stage2b_ckpt).exists():
            from ultralytics import YOLO
            self.detector.model = YOLO(str(stage2b_ckpt))

        log.info("✓ All models ready")

    def run(self, tif_path: str, out_dir: str):
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        raster_path = Path(tif_path)
        prefix = raster_path.stem

        try:
            with rasterio.open(str(raster_path)) as src:
                meta = src.meta.copy()
                crs = src.crs
                transform = src.transform
                bands = min(src.count, 3)
                img = src.read(list(range(1, bands + 1))).transpose(1, 2, 0)
                if bands < 3:
                    img = np.stack([img[:, :, 0]] * 3, axis=-1)
        except Exception as e:
            log.error("Failed to read raster %s: %s", raster_path, e)
            return

        if img.dtype != np.uint8:
            img = to_uint8(img)

        # Stage 1: Segmentation
        log.info("[Stage 1] Tiled segmentation ...")
        prob_map = sliding_window_inference(
            model=self.seg.model,
            image=img,
            patch_size=int(CFG.STAGE1["patch_size"]),
            overlap=int(CFG.STAGE1["overlap"]),
            num_classes=int(CFG.STAGE1["num_classes"]),
            transform=self.seg_tf,
            batch_size=int(CFG.STAGE1.get("inference_batch_size", 16)),
            device=self.device,
            amp_dtype=torch.float16,
        )
        torch.cuda.empty_cache()

        seg_mask = prob_map.argmax(0).astype(np.uint8)
        seg_mask = clean_segmentation_mask(seg_mask, CFG.STAGE1)

        meta.update(count=1, dtype=rasterio.uint8)
        mask_path = out_dir_p / f"{prefix}_segmask.tif"
        with rasterio.open(str(mask_path), "w", **meta) as dst:
            dst.write(seg_mask[np.newaxis])

        class_names = [str(x) for x in CFG.STAGE1.get("class_names", [])]
        mask_to_shapefile(seg_mask, transform, crs, class_names, str(out_dir_p), prefix)

        bld_shp = out_dir_p / f"{prefix}_building.shp"
        building_polygons = None
        if bld_shp.exists():
            import geopandas as gpd
            building_polygons = gpd.read_file(str(bld_shp))

        del prob_map
        torch.cuda.empty_cache()

        # Stage 2A: Rooftop Classification
        log.info("[Stage 2A] Rooftop classification ...")
        if bld_shp.exists():
            roof_preds = self._classify_rooftops(img, str(bld_shp), transform, building_polygons)
            merge_rooftop_labels(str(bld_shp), roof_preds, str(out_dir_p / f"{prefix}_building_rooftop.shp"))
        torch.cuda.empty_cache()

        # Stage 2B: Infrastructure Detection
        log.info("[Stage 2B] Infrastructure detection ...")
        dets = self._detect(img)
        if dets:
            detections_to_shapefile(dets, transform, crs, str(out_dir_p / f"{prefix}_infrastructure.shp"))

        log.info(f"✓ Done → {out_dir_p}")

    def _classify_rooftops(self, img_rgb, bld_shp_path, transform, building_polygons=None):
        import geopandas as gpd
        gdf = building_polygons if building_polygons is not None else gpd.read_file(bld_shp_path)
        if len(gdf) == 0: return {}

        preds = {}
        inv_transform = ~transform
        crop_sz = int(CFG.STAGE2A["crop_size"])
        H_img, W_img = img_rgb.shape[:2]
        class_names_2a = [str(x) for x in CFG.STAGE2A["class_names"]]

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  roofs"):
            geom = row.geometry
            if geom is None or geom.is_empty:
                preds[idx] = "Other"
                continue

            geo_x1, geo_y1, geo_x2, geo_y2 = geom.bounds
            px_x1, px_y1 = inv_transform * (geo_x1, geo_y1)
            px_x2, px_y2 = inv_transform * (geo_x2, geo_y2)
            
            x1, x2 = sorted((int(px_x1), int(px_x2)))
            y1, y2 = sorted((int(px_y1), int(px_y2)))
            
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(W_img, x2), min(H_img, y2)
            if x2c <= x1c or y2c <= y1c:
                preds[idx] = "Other"
                continue

            img_slice = img_rgb[y1c:y2c, x1c:x2c]
            crop = cv2.resize(img_slice, (crop_sz, crop_sz), interpolation=cv2.INTER_LINEAR)
            inp = self.clf_tf(image=crop)["image"].unsqueeze(0).to(self.device)

            with torch.no_grad():
                pid = self.clf.predict(inp)[0].item()
            preds[idx] = class_names_2a[int(pid)]
            
        return preds

    def _detect(self, img_rgb):
        H, W = img_rgb.shape[:2]
        tile = int(CFG.STAGE2B["img_size"])
        stride = max(1, tile - int(CFG.STAGE2B.get("overlap", 256)))
        dets = []

        for r in tqdm(range(0, H, stride), desc="  det tiles"):
            for c in range(0, W, stride):
                r2, c2 = min(r + tile, H), min(c + tile, W)
                if r2 <= r or c2 <= c: continue

                patch = img_rgb[r:r2, c:c2]
                bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                for d in self.detector.predict(bgr):
                    d["bbox_xyxy"][0] += c
                    d["bbox_xyxy"][2] += c
                    d["bbox_xyxy"][1] += r
                    d["bbox_xyxy"][3] += r
                    dets.append(d)

        return dets
