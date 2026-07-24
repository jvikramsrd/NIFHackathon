"""
utils/postprocess.py
─────────────────────
Post-processing for Stage 1 segmentation output:
  1. Dense CRF refinement (pydensecrf)
  2. Morphological cleanup (fill holes, remove noise)
  3. Vectorization: raster mask → GeoJSON / SHP polygons
  4. Rooftop label merge (Stage 2A results → building polygons)
"""

import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from utils.core import get_logger

# Optional geo stack — hoisted to module level so we pay the import cost once
# rather than on every call to mask_to_shapefile / clean_vector_geometries /
# detections_to_shapefile / merge_rooftop_labels. Each of those was importing
# the same modules locally — a measurable cost when post-processing many tiles.
try:
    import geopandas as gpd
    import pandas as pd
    import rasterio.features
    from shapely.affinity import affine_transform, rotate
    from shapely.affinity import scale as shapely_scale
    from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
    from shapely.geometry import box as shapely_box
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

try:
    from shapely.validation import make_valid as _make_valid
except ImportError:
    _make_valid = None

try:
    import config as _CFG
except ImportError:
    _CFG = None

warnings.filterwarnings("ignore")

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DENSE CRF REFINEMENT
# ─────────────────────────────────────────────────────────────────────────────


def _process_crf_tile(args):
    log.debug("  [DEBUG CRF] Entering _process_crf_tile")
    try:
        log.debug("  [DEBUG CRF] Importing pydensecrf inside worker...")
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        log.debug("  [DEBUG CRF] Import successful")
    except ImportError:
        log.debug("  [DEBUG CRF] Import failed")
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
    log.debug(f"  [DEBUG CRF] Creating DenseCRF2D({W}, {H}, {C})")
    d = dcrf.DenseCRF2D(W, H, C)

    log.debug("  [DEBUG CRF] Computing unary_from_softmax")
    unary = unary_from_softmax(prob_map)

    log.debug("  [DEBUG CRF] setUnaryEnergy")
    d.setUnaryEnergy(unary)

    log.debug("  [DEBUG CRF] addPairwiseGaussian")
    d.addPairwiseGaussian(sxy=pos_xy_std, compat=pos_w)

    # NOTE: an earlier version computed a Sobel-magnitude "texture_evidence"
    # tensor here and the docstring promised it would weight the bilateral
    # term. That tensor was never actually passed to addPairwiseBilateral —
    # the rgbim arg below uses the raw image_rgb. Removed to stop paying the
    # ~per-tile Sobel/normalize cost for a feature that wasn't wired up.
    # If we re-introduce texture-aware CRF, route through
    # pydensecrf.utils.create_pairwise_bilateral + d.addPairwiseEnergy.

    log.debug("  [DEBUG CRF] Building per-class bilateral compat matrix...")
    # Per-class compatibility matrix: heavier penalty for class confusions that
    # are most damaging to downstream metrics:
    #   building <-> waterbody: 3x   (catastrophic false positive)
    #   building <-> road:      1.5x  (common confusion in dense settlements)
    C_mat = np.ones((C, C), dtype=np.float32) * float(bi_w)
    if C > 3:
        C_mat[1, 3] = C_mat[3, 1] = float(bi_w) * 3.0  # building <-> waterbody
        C_mat[1, 2] = C_mat[2, 1] = float(bi_w) * 1.5  # building <-> road
    np.fill_diagonal(C_mat, 0.0)  # same-class: no penalty

    d.addPairwiseBilateral(
        sxy=bi_xy_std,
        srgb=bi_rgb_std,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=C_mat,
    )

    log.debug(f"  [DEBUG CRF] Running inference({n_iter})")
    Q = d.inference(n_iter)

    log.debug("  [DEBUG CRF] Reshaping and returning")
    refined = np.array(Q).reshape((C, H, W))
    return refined, r, c, H, W


def apply_dense_crf(
    image_rgb: np.ndarray,  # (H, W, 3) uint8
    prob_map: np.ndarray,  # (C, H, W) float32 softmax probabilities
    n_iter: int = 10,
    pos_xy_std: float = 3.0,
    pos_w: float = 3.0,
    bi_xy_std: float = 20.0,   # CORRECTED: 20px = 1m at 5cm drone GSD (was 80px = 4m)
    bi_rgb_std: float = 13.0,
    bi_w: float = 10.0,
) -> np.ndarray:
    """
    Apply DenseCRF post-processing to sharpen segment boundaries.
    Returns refined probability map (C, H, W).
    Falls back to identity if pydensecrf is not installed.
    Parallelizes over image tiles.
    """
    log.debug("[DEBUG CRF] Entering apply_dense_crf")
    try:
        log.debug("[DEBUG CRF] Trying to import pydensecrf...")
        import pydensecrf.densecrf as dcrf

        log.debug("[DEBUG CRF] pydensecrf imported successfully")
    except ImportError:
        warnings.warn(
            "pydensecrf not installed; skipping CRF. "
            "Install with: pip install pydensecrf2"
        )
        return prob_map

    log.debug("[DEBUG CRF] Importing tqdm...")
    from tqdm import tqdm

    log.debug("[DEBUG CRF] tqdm imported. Preparing tasks...")

    C, H, W = prob_map.shape
    tile_size = 1024  # CRF is O(n²); smaller tiles = 4× faster with bounded bilateral kernel
    overlap = 128
    stride = tile_size - overlap

    # Use the shared cosine window utility for consistent blending
    from utils.window import cosine_window

    window = cosine_window(tile_size, overlap, power=2)

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

    log.debug("[DEBUG CRF] Created %d tasks. Starting loop...", len(tasks))
    refined_map = np.zeros_like(prob_map)
    weight_map = np.zeros((H, W), dtype=np.float32)

    def _accumulate(res, r, c, th, tw):
        r2 = r + th
        c2 = c + tw
        wind_slice = window[:th, :tw]
        refined_map[:, r:r2, c:c2] += res * wind_slice
        weight_map[r:r2, c:c2] += wind_slice

    max_workers_cfg = int(getattr(_CFG, "CRF_WORKERS", 0) or 0) if _CFG is not None else 0
    max_workers = min(max_workers_cfg or (os.cpu_count() or 1), len(tasks), 8)
    if max_workers <= 1 or len(tasks) <= 1:
        for i, t in enumerate(tqdm(tasks, desc="  CRF tiles")):
            log.debug(
                "  [DEBUG CRF] Processing task %d/%d (r=%s, c=%s)",
                i + 1,
                len(tasks),
                t[-2],
                t[-1],
            )
            res, r, c, th, tw = _process_crf_tile(t)
            _accumulate(res, r, c, th, tw)
    else:
        log.debug("[DEBUG CRF] Launching %d workers for %d tiles", max_workers, len(tasks))
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process_crf_tile, t): t for t in tasks}
                for future in tqdm(
                    as_completed(futures), total=len(tasks), desc="  CRF tiles"
                ):
                    res, r, c, th, tw = future.result()
                    _accumulate(res, r, c, th, tw)
        except Exception as exc:
            log.warning("Parallel CRF failed (%s); falling back to serial", exc)
            for t in tqdm(tasks, desc="  CRF tiles (serial)"):
                res, r, c, th, tw = _process_crf_tile(t)
                _accumulate(res, r, c, th, tw)

    log.debug("[DEBUG CRF] All tasks finished. Averaging map...")
    refined_map /= np.maximum(weight_map, 1e-6)
    log.debug("[DEBUG CRF] Exiting apply_dense_crf")
    return refined_map


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MORPHOLOGICAL CLEANUP
# ─────────────────────────────────────────────────────────────────────────────


def clean_segmentation_mask(
    mask: np.ndarray,  # (H, W) uint8 class ids
    class_config: Dict,  # config.STAGE1 dict
) -> np.ndarray:
    """
    Per-class morphological operations, enhancing robustness using context-aware
    size thresholds and connectivity checks.
    """
    if mask is None or mask.size == 0:
        return np.zeros((256, 256), dtype=np.uint8)

    mask = np.asarray(mask)
    if mask.ndim != 2:
        return np.zeros((256, 256), dtype=np.uint8)

    cleaned = mask.copy()
    min_bld = class_config.get("min_building_area_px", 100)
    min_road = class_config.get("min_road_width_px", 3)

    try:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    except Exception:
        kernel_close = None
    try:
        road_close_sz = min_road + 15
        road_close_sz += 1 if road_close_sz % 2 == 0 else 0
        kernel_road_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (road_close_sz, road_close_sz)
        )
    except Exception:
        kernel_road_close = None
    try:
        kernel_road_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (min_road, min_road)
        )
    except Exception:
        kernel_road_open = None
    try:
        kernel_water = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    except Exception:
        kernel_water = None

    # --- Buildings (class 1) ---
    bld = (mask == 1).astype(np.uint8)
    if kernel_close is not None:
        bld = cv2.morphologyEx(bld, cv2.MORPH_CLOSE, kernel_close)
    bld = _remove_small_blobs(bld, min_bld)
    bld = separate_touching_buildings(bld)
    cleaned[bld == 1] = 1
    cleaned[(mask == 1) & (bld == 0)] = 0

    # --- Roads (class 2) ---
    road = (mask == 2).astype(np.uint8)
    if kernel_road_close is not None:
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel_road_close, iterations=2)
    if kernel_road_open is not None:
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel_road_open)
    road = _remove_small_blobs(road, min_bld // 2)
    cleaned[road == 1] = 2
    cleaned[(mask == 2) & (road == 0)] = 0

    # --- Waterbodies (class 3) ---
    water = (mask == 3).astype(np.uint8)
    if kernel_water is not None:
        water = cv2.morphologyEx(water, cv2.MORPH_CLOSE, kernel_water)
    water = _remove_small_blobs(water, min_bld * 2)
    cleaned[water == 1] = 3
    cleaned[(mask == 3) & (water == 0)] = 0

    return cleaned.astype(np.uint8)


def _remove_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    if binary is None or binary.size == 0:
        return np.zeros((256, 256), dtype=np.uint8)

    binary = np.asarray(binary)
    if binary.ndim != 2:
        return np.zeros((256, 256), dtype=np.uint8)

    min_area = max(1, int(min_area))

    try:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
    except Exception:
        return binary

    if n_labels <= 1:
        return binary

    valid_mask = np.zeros(n_labels, dtype=np.int32)
    valid_mask[stats[:, cv2.CC_STAT_AREA] >= min_area] = 1
    valid_mask[0] = 0
    return valid_mask[labels].astype(np.uint8)


def separate_touching_buildings(binary_mask: np.ndarray) -> np.ndarray:
    """
    Use marker-controlled watershed to separate touching building instances.

    Steps:
      1. Distance transform of the binary mask.
      2. Find local maxima of the distance map as instance seeds.
      3. Run watershed with the seeds as markers.

    Returns a binary mask where touching buildings are separated by a 1-px gap.
    Falls back to the input mask gracefully if scipy/skimage are unavailable.
    """
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError:
        log.warning("scipy/skimage not found; skipping watershed separation. "
                    "Install: pip install scipy scikit-image")
        return binary_mask

    if binary_mask.sum() == 0:
        return binary_mask

    dist = ndi.distance_transform_edt(binary_mask)
    # Minimum distance between seeds = 10px (avoids over-segmentation)
    min_dist = max(10, int(binary_mask.shape[0] * 0.01))
    coords = peak_local_max(dist, min_distance=min_dist, labels=binary_mask)
    if len(coords) == 0:
        return binary_mask

    markers = np.zeros_like(binary_mask, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    markers = ndi.label(markers)[0]
    # watershed_line=True leaves a 1-pixel background line between adjacent basins
    # (touching buildings), giving the polygon vectoriser clean boundaries to trace.
    # Without it the basins fill the exact same footprint as the input, leaving
    # buildings merged in the vector output.
    labels = watershed(-dist, markers, mask=binary_mask, watershed_line=True)
    return (labels > 0).astype(np.uint8)


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
    if mask is None or mask.size == 0:
        log.warning("Empty mask provided to mask_to_shapefile")
        return Path(out_dir)

    mask = np.asarray(mask)
    if mask.ndim != 2:
        log.warning(f"Invalid mask dimensions: {mask.shape}")
        return Path(out_dir)

    if not _GEO_AVAILABLE:
        log.warning("geopandas/shapely not available; skipping mask_to_shapefile")
        return Path(out_dir)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(_CFG.STAGE1) if _CFG is not None else {}
    simplify_tol = float(cfg.get("polygon_simplify_tolerance", 0.5))
    min_area_by_class = cfg.get("polygon_min_area_px", {})

    all_gdfs = []

    for class_id, class_name in enumerate(class_names):
        if class_id == 0:
            continue

        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        try:
            shapes_gen = rasterio.features.shapes(binary, mask=binary, transform=transform)
            shapes_list = [(shape(s), v) for s, v in shapes_gen if v == 1]
        except Exception as e:
            log.warning(f"Failed to extract shapes for class {class_name}: {e}")
            continue

        if not shapes_list:
            continue

        try:
            geoms, vals = zip(*shapes_list)
            geoms_list = list(geoms)

            gdf = gpd.GeoDataFrame(
                {
                    "class_id": [class_id] * len(geoms_list),
                    "class_name": [class_name] * len(geoms_list),
                },
                geometry=geoms_list,
                crs=crs or "EPSG:3857",
            )
            gdf = clean_vector_geometries(
                gdf,
                class_name=class_name,
                min_area=float(min_area_by_class.get(class_name, 0)),
                simplify_tolerance=simplify_tol,
            )
            if len(gdf) == 0:
                continue

            cls_path = out_dir / f"{prefix}_{class_name}.shp"
            gdf.to_file(str(cls_path))
            cls_path_json = out_dir / f"{prefix}_{class_name}.geojson"
            gdf.to_file(str(cls_path_json), driver="GeoJSON")
            all_gdfs.append(gdf)
        except Exception as e:
            log.warning(f"Failed to process class {class_name}: {e}")
            continue

    if all_gdfs:
        try:
            combined = pd.concat(all_gdfs, ignore_index=True)
            combined_gdf = gpd.GeoDataFrame(combined, crs=crs or "EPSG:3857")
            combined_path = out_dir / f"{prefix}_all_features.gpkg"
            combined_gdf.to_file(str(combined_path), driver="GPKG")
            log.info(f"  ✓ Combined vector saved: {combined_path}")
        except Exception as e:
            log.warning(f"Failed to save combined shapefile: {e}")

    return out_dir


def clean_vector_geometries(
    gdf,
    class_name: str = "",
    min_area: float = 0.0,
    simplify_tolerance: float = 0.5,
):
    """Fix topology, simplify polygons, explode multipart features, and filter area."""

    def _fix_geom(geom):
        if geom is None or geom.is_empty:
            return None
        try:
            fixed = _make_valid(geom) if _make_valid is not None else geom.buffer(0)
        except Exception:
            fixed = geom.buffer(0)
        if fixed is None or fixed.is_empty:
            return None
        if isinstance(fixed, GeometryCollection):
            polys = [g for g in fixed.geoms if isinstance(g, (Polygon, MultiPolygon))]
            if not polys:
                return None
            fixed = MultiPolygon(
                [p for geom_part in polys for p in (geom_part.geoms if isinstance(geom_part, MultiPolygon) else [geom_part])]
            )
        if simplify_tolerance > 0:
            fixed = fixed.simplify(simplify_tolerance, preserve_topology=True)
        try:
            if not fixed.is_valid:
                fixed = fixed.buffer(0)
        except Exception:
            return None
        return fixed if fixed is not None and not fixed.is_empty else None

    cleaned = gdf.copy()
    cleaned["geometry"] = cleaned.geometry.map(_fix_geom)
    cleaned = cleaned[cleaned.geometry.notna() & ~cleaned.geometry.is_empty]
    if len(cleaned) == 0:
        return cleaned
    cleaned = cleaned.explode(index_parts=False, ignore_index=True)
    if min_area > 0:
        before = len(cleaned)
        cleaned = cleaned[cleaned.geometry.area >= min_area]
        dropped = before - len(cleaned)
        if dropped:
            log.debug("Dropped %s %s polygons below area %.2f", dropped, class_name, min_area)
    cleaned = cleaned[cleaned.geometry.is_valid]
    return cleaned


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
    gdf = gpd.read_file(building_shp_path)
    gdf["roof_pred"] = gdf.index.map(lambda i: rooftop_predictions.get(i, "Unknown"))
    gdf.to_file(out_path)
    log.info(f"  ✓ Building SHP with roof labels → {out_path}")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INFRASTRUCTURE DETECTIONS → POINT SHP
# ─────────────────────────────────────────────────────────────────────────────


def detections_to_shapefile(
    detections: List[Dict],  # from InfrastructureDetector.predict()
    transform,
    crs,
    out_path: str,
) -> str | None:
    """
    Convert bounding-box / OBB detections to a GeoDataFrame.

    - OBB detections (``obb_xywhr`` present): stored as rotated rectangle polygons
      in geo-coordinates so orientation is preserved for GIS users.
    - Standard-box detections: stored as centroid points (legacy behaviour).
    """
    rows = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox_xyxy"]
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        # Pixel → geo coords via affine transform
        geo_cx = transform.c + cx_px * transform.a
        geo_cy = transform.f + cy_px * transform.e

        obb = det.get("obb_xywhr")  # [cx, cy, w, h, angle_rad] in pixel space
        if obb is not None and len(obb) == 5:
            px_cx, px_cy, pw, ph, angle_rad = obb
            # Build axis-aligned box in pixel space, then rotate
            half_w = pw / 2.0
            half_h = ph / 2.0
            # Convert pixel extents to geo units (assumes uniform pixel size)
            geo_half_w = half_w * abs(transform.a)
            geo_half_h = half_h * abs(transform.e)
            rect = shapely_box(
                geo_cx - geo_half_w, geo_cy - geo_half_h,
                geo_cx + geo_half_w, geo_cy + geo_half_h,
            )
            # Rotate around centroid; YOLO angle is clockwise in image space
            angle_deg = math.degrees(angle_rad)
            geom = rotate(rect, -angle_deg, origin=(geo_cx, geo_cy), use_radians=False)
        elif obb is not None and len(obb) == 8:
            # OBB-8 format: [x1, y1, x2, y2, x3, y3, x4, y4] in pixel coords
            corners_px = [(obb[i], obb[i+1]) for i in range(0, 8, 2)]
            corners_geo = []
            for px, py in corners_px:
                gx = transform.c + px * transform.a
                gy = transform.f + py * transform.e
                corners_geo.append((gx, gy))
            geom = Polygon(corners_geo)
        else:
            geom = Point(geo_cx, geo_cy)

        rows.append(
            {
                "geometry": geom,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": round(det["conf"], 4),
            }
        )

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf.to_file(out_path)
    out_path_json = str(Path(out_path).with_suffix(".geojson"))
    gdf.to_file(out_path_json, driver="GeoJSON")
    log.info("  ✓ Infrastructure features → %s and .geojson", out_path)
    return gdf
