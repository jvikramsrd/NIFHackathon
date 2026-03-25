import sys
from pathlib import Path
import config as CFG
from data.preprocessing import scan_folder, _extract_crops_streaming, _extract_infra_streaming
import re
import os

def extract_only(data_root: str):
    data_root_path = Path(data_root)
    candidates = [data_root_path] + [d for d in data_root_path.iterdir() if d.is_dir()]
    RASTER_EXTS = {".tif", ".tiff", ".ecw", ".img"}
    folders_with_rasters = [d for d in candidates if any(f.suffix.lower() in RASTER_EXTS for f in d.iterdir() if f.is_file())]
    
    os.makedirs(str(CFG.CROP_DIR), exist_ok=True)
    os.makedirs(str(CFG.YOLO_DIR / "images"), exist_ok=True)
    os.makedirs(str(CFG.YOLO_DIR / "labels"), exist_ok=True)

    for folder in folders_with_rasters:
        rasters, shps = scan_folder(folder)
        shp_by_stem = {s.stem.lower().rstrip("_"): s for s in shps}
        built_up_shp = shp_by_stem.get("built_up_area_type") or shp_by_stem.get("built_up_area_typ")
        utility_shps = [s for k, s in shp_by_stem.items() if k.startswith("utility")]

        for raster_path in rasters:
            print(f"--- Extracting from {raster_path.name} ---")
            
            n_crops = _extract_crops_streaming(
                raster_path, built_up_shp, CFG.STAGE2A["shp_roof_col"], 
                CFG.ROOF_TYPE_MAP, str(CFG.CROP_DIR), CFG.STAGE2A["crop_size"], CFG.STAGE2A["min_crop_px"]
            )
            print(f"Crops extracted: {n_crops}")

            if utility_shps:
                n_infra = _extract_infra_streaming(
                    raster_path, utility_shps, CFG.STAGE2B["shp_infra_col"],
                    CFG.INFRA_TYPE_MAP, CFG.STAGE2B["class_names"],
                    str(CFG.YOLO_DIR / "images"), str(CFG.YOLO_DIR / "labels"), CFG.STAGE2B["img_size"]
                )
                print(f"Infra extracted: {n_infra}")

if __name__ == "__main__":
    extract_only("C:/Users/Dell/Downloads/dataset")
