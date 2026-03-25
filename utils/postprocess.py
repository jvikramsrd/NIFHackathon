"""
utils/postprocess.py
─────────────────────
Post-processing for Stage 1 segmentation output:
  1. Dense CRF refinement (pydensecrf)
  2. Morphological cleanup (fill holes, remove noise)
  3. Vectorization: raster mask → GeoJSON / SHP polygons
  4. Rooftop label merge (Stage 2A results → building polygons)
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DENSE CRF REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────


def _process_crf_tile(args):
    print("  [DEBUG CRF] Entering _process_crf_tile")
    try:
        print("  [DEBUG CRF] Importing pydensecrf inside worker...")
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        print("  [DEBUG CRF] Import successful")
    except ImportError:
        print("  [DEBUG CRF] Import failed")
        return args[1], args[-2], args[-1], args[1].shape[1], args[1].shape[2]

    (
        image_rgb,
        prob_map,
        n_iter,
        pos_xy_std,
        pos_w,
        bi_xy_std,
        bi_rgb_std,
        bi_w,
        r,
        c,
    ) = args
    C, H, W = prob_map.shape
    print(f"  [DEBUG CRF] Creating DenseCRF2D({W}, {H}, {C})")
    d = dcrf.DenseCRF2D(W, H, C)

    print("  [DEBUG CRF] Computing unary_from_softmax")
    unary = unary_from_softmax(prob_map)

    print("  [DEBUG CRF] setUnaryEnergy")
    d.setUnaryEnergy(unary)

    print("  [DEBUG CRF] addPairwiseGaussian")
    d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)

    print("  [DEBUG CRF] addPairwiseBilateral")
    d.addPairwiseBilateral(
        sxy=bi_xy_std,
        srgb=bi_rgb_std,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=bi_w,
    )

    print(f"  [DEBUG CRF] Running inference({n_iter})")
    Q = d.inference(n_iter)

    print("  [DEBUG CRF] Reshaping and returning")
    refined = np.array(Q).reshape((C, H, W))
    return refined, r, c, H, W


def apply_dense_crf(
    image_rgb: np.ndarray,  # (H, W, 3) uint8
    prob_map: np.ndarray,  # (C, H, W) float32 softmax probabilities
    n_iter: int = 10,
    pos_xy_std: float = 3.0,
    pos_w: float = 3.0,
    bi_xy_std: float = 80.0,
    bi_rgb_std: float = 13.0,
    bi_w: float = 10.0,
) -> np.ndarray:
    """
    Apply DenseCRF post-processing to sharpen segment boundaries.
    Returns refined probability map (C, H, W).
    Falls back to identity if pydensecrf is not installed.
    Parallelizes over image tiles.
    """
    print("[DEBUG CRF] Entering apply_dense_crf")
    try:
        print("[DEBUG CRF] Trying to import pydensecrf...")
        import pydensecrf.densecrf as dcrf

        print("[DEBUG CRF] pydensecrf imported successfully")
    except ImportError:
        warnings.warn(
            "pydensecrf not installed; skipping CRF. "
            "Install with: pip install pydensecrf2"
        )
        return prob_map

    print("[DEBUG CRF] Importing tqdm...")
    from tqdm import tqdm

    print("[DEBUG CRF] tqdm imported. Preparing tasks...")

    C, H, W = prob_map.shape
    tile_size = 2048
    overlap = 256
    stride = tile_size - overlap

    intersection = overlap
    wind_outer = (np.cos(np.pi * np.arange(intersection) / intersection) + 1) / 2
    wind = np.ones(tile_size)
    wind[:intersection] = wind_outer[::-1]
    wind[-intersection:] = wind_outer
    wind = wind**2
    window = np.outer(wind, wind)

    tasks = []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r2 = min(r + tile_size, H)
            c2 = min(c + tile_size, W)
            img_tile = image_rgb[r:r2, c:c2].copy()
            prob_tile = prob_map[:, r:r2, c:c2].copy()
            tasks.append(
                (
                    img_tile,
                    prob_tile,
                    n_iter,
                    pos_xy_std,
                    pos_w,
                    bi_xy_std,
                    bi_rgb_std,
                    bi_w,
                    r,
                    c,
                )
            )

    print(f"[DEBUG CRF] Created {len(tasks)} tasks. Starting loop...")
    refined_map = np.zeros_like(prob_map)
    weight_map = np.zeros((H, W), dtype=np.float32)

    for i, t in enumerate(tqdm(tasks, desc="  CRF tiles")):
        print(
            f"  [DEBUG CRF] Submitting task {i + 1}/{len(tasks)} (r={t[-2]}, c={t[-1]})"
        )
        res, r, c, th, tw = _process_crf_tile(t)
        print(f"  [DEBUG CRF] Task {i + 1} completed")
        r2 = r + th
        c2 = c + tw
        wind_slice = window[:th, :tw]
        refined_map[:, r:r2, c:c2] += res * wind_slice
        weight_map[r:r2, c:c2] += wind_slice

    print("[DEBUG CRF] All tasks finished. Averaging map...")
    refined_map /= np.maximum(weight_map, 1e-6)
    print("[DEBUG CRF] Exiting apply_dense_crf")
    return refined_map


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MORPHOLOGICAL CLEANUP
# ─────────────────────────────────────────────────────────────────────────────


def clean_segmentation_mask(
    mask: np.ndarray,  # (H, W) uint8 class ids
    class_config: Dict,  # config.STAGE1 dict
) -> np.ndarray:
    """
    Per-class morphological operations:
      • Buildings  : close holes, remove tiny blobs
      • Roads      : skeletonise/dilate to ensure connectivity
      • Waterbodies: close gaps, remove tiny blobs
    """
    cleaned = mask.copy()
    min_bld = class_config.get("min_building_area_px", 100)
    min_road = class_config.get("min_road_width_px", 3)

    # --- Buildings (class 1) ---
    bld = (mask == 1).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bld = cv2.morphologyEx(bld, cv2.MORPH_CLOSE, kernel_close)
    bld = _remove_small_blobs(bld, min_bld)
    cleaned[bld == 1] = 1
    cleaned[bld == 0] &= ~np.uint8(1)  # don't reset other classes

    # --- Roads (class 2) ---
    road = (mask == 2).astype(np.uint8)
    # Aggressively bridge gaps in roads
    kernel_road_close = cv2.getStructuringElement(
        cv2.MORPH_RECT, (min_road + 15, min_road + 15)
    )
    kernel_road_open = cv2.getStructuringElement(cv2.MORPH_RECT, (min_road, min_road))
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel_road_close, iterations=2)
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel_road_open)
    road = _remove_small_blobs(road, min_bld // 2)
    cleaned = np.where(road == 1, 2, cleaned)

    # --- Waterbodies (class 3) ---
    water = (mask == 3).astype(np.uint8)
    kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_water)
    water = _remove_small_blobs(water, min_bld * 2)
    cleaned = np.where(water == 1, 3, cleaned)

    return cleaned.astype(np.uint8)


def _remove_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    # Fast vectorized lookup table
    valid_mask = (stats[:, cv2.CC_STAT_AREA] >= min_area).astype(np.uint8)
    valid_mask[0] = 0  # Background is always 0
    return valid_mask[labels]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RASTER MASK → VECTOR SHP / GeoJSON
# ─────────────────────────────────────────────────────────────────────────────


def mask_to_shapefile(
    mask: np.ndarray,  # (H, W) uint8 class ids
    transform,  # rasterio Affine transform
    crs,  # rasterio CRS
    class_names: List[str],
    out_dir: str,
    prefix: str = "output",
):
    """
    Vectorise each class layer from `mask` into a separate GeoPackage layer
    and a combined output shapefile.

    Requires: rasterio, shapely, geopandas, fiona
    """
    import geopandas as gpd
    import rasterio.features
    from shapely.affinity import affine_transform
    from shapely.geometry import shape

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_gdfs = []

    for class_id, class_name in enumerate(class_names):
        if class_id == 0:
            continue  # skip background

        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        shapes_gen = rasterio.features.shapes(binary, mask=binary, transform=transform)

        # Generator might be empty or valid, safely get geoms
        shapes_list = [(shape(s), v) for s, v in shapes_gen if v == 1]
        if not shapes_list:
            continue

        geoms, vals = zip(*shapes_list)
        geoms_list = list(geoms)

        gdf = gpd.GeoDataFrame(
            {
                "class_id": [class_id] * len(geoms_list),
                "class_name": [class_name] * len(geoms_list),
            },
            geometry=geoms_list,
            crs=crs,
        )
        # Simplify polygons slightly to reduce vertex count
        gdf["geometry"] = gdf["geometry"].simplify(
            tolerance=0.5, preserve_topology=True
        )
        gdf = gdf[gdf.geometry.is_valid]

        # Save per-class SHP
        cls_path = out_dir / f"{prefix}_{class_name}.shp"
        gdf.to_file(str(cls_path))
        all_gdfs.append(gdf)

    if all_gdfs:
        import pandas as pd

        combined = pd.concat(all_gdfs, ignore_index=True)
        combined_gdf = gpd.GeoDataFrame(combined, crs=crs)
        combined_path = out_dir / f"{prefix}_all_features.gpkg"
        combined_gdf.to_file(str(combined_path), driver="GPKG")
        print(f"  ✓ Combined vector saved: {combined_path}")

    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MERGE ROOFTOP LABELS INTO BUILDING POLYGONS
# ─────────────────────────────────────────────────────────────────────────────


def merge_rooftop_labels(
    building_shp_path: str,
    rooftop_predictions: Dict[int, str],  # {polygon_index: predicted_class}
    out_path: str,
):
    """
    Add rooftop material classification results as a new attribute
    to the building footprint shapefile.
    """
    import geopandas as gpd

    gdf = gpd.read_file(building_shp_path)
    gdf["roof_pred"] = gdf.index.map(lambda i: rooftop_predictions.get(i, "Unknown"))
    gdf.to_file(out_path)
    print(f"  ✓ Building SHP with roof labels → {out_path}")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INFRASTRUCTURE DETECTIONS → POINT SHP
# ─────────────────────────────────────────────────────────────────────────────


def detections_to_shapefile(
    detections: List[Dict],  # from InfrastructureDetector.predict()
    transform,  # rasterio Affine for the tile
    crs,
    out_path: str,
):
    """
    Convert bounding-box detections to point GeoDataFrame (centroid of bbox).
    """
    import geopandas as gpd
    from shapely.geometry import Point

    rows = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        # Pixel → geo coords via affine transform
        geo_x = transform.c + cx_px * transform.a
        geo_y = transform.f + cy_px * transform.e
        rows.append(
            {
                "geometry": Point(geo_x, geo_y),
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": round(det["conf"], 4),
            }
        )

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.to_file(out_path)
    print(f"  ✓ Infrastructure points → {out_path}")
    return gdf
