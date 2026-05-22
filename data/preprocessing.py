"""
data/preprocessing.py  —  SVAMITVA dataset edition
════════════════════════════════════════════════════
Handles the exact layout found in cg/ and pb/:

  dataset/cg/  or  dataset/pb/
    ├── VILLAGE_NAME_ORTHO.tif     ← one or more ortho rasters
    ├── VILLAGE_NAME_3857.ecw      ← auto-converted to GeoTIFF at runtime
    ├── VILLAGE_NAME.tif.pyrx      ← pyramid overview    → SKIPPED
    ├── Built_Up_Area_type.shp     ← buildings (shared across all TIFs)
    ├── Road.shp                   ← roads
    ├── Water_Body.shp             ← waterbodies
    ├── Utility.shp                ← infrastructure points
    └── *.shp.*.lock               ← ArcGIS lock files   → IGNORED

Key behaviours:
  • SHPs are SHARED — one set annotates ALL rasters in the folder
  • ECW duplicates of TIFs are skipped (TIF preferred)
  • .pyrx, .lock, .aux, .sbx, .sbn, .cpg, .dbf, .prj, .shx ignored
  • Spaces and special chars in filenames handled via pathlib
  • Every raster open is wrapped in try/except → corrupt files skipped
  • Every SHP burn is wrapped in try/except → bad geometries skipped
  • Detailed per-file logging so you know exactly what succeeded/failed
"""

import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.enums
import rasterio.errors
import rasterio.features
import rasterio.transform
import rasterio.warp
import rasterio.windows
from tqdm import tqdm
from utils.logger import get_logger

log = get_logger(__name__)


warnings.filterwarnings("ignore")


def normalise_dbf_value(value) -> str:
    """Normalize noisy DBF categorical values for stable config-map lookup."""
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def find_attribute_column(gdf, candidates) -> Optional[str]:
    """Return the actual GeoDataFrame column matching any candidate name."""
    if gdf is None or not hasattr(gdf, "columns"):
        return None
    if isinstance(candidates, (str, bytes)):
        candidates = [candidates]

    cols = list(gdf.columns)
    if not cols:
        return None

    for candidate in candidates:
        if candidate in gdf.columns:
            return str(candidate)

    lower_map = {str(col).lower(): str(col) for col in cols}
    for candidate in candidates:
        match = lower_map.get(str(candidate).lower())
        if match is not None:
            return match

    norm_map = {normalise_dbf_value(col): str(col) for col in cols}
    for candidate in candidates:
        match = norm_map.get(normalise_dbf_value(candidate))
        if match is not None:
            return match
    return None


def canonical_mapped_label(raw_value, value_map: dict, default=None):
    """Map a raw DBF value to a canonical class label."""
    raw = normalise_dbf_value(raw_value)
    if not raw:
        return default

    if raw in value_map:
        return value_map[raw]

    for key, label in value_map.items():
        key_norm = normalise_dbf_value(key)
        if key_norm and (key_norm in raw or raw in key_norm):
            return label
    return default


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Only these extensions are treated as rasters
RASTER_EXTS = {".tif", ".tiff", ".ecw", ".img", ".vrt", ".jp2"}

# SHP sidecar extensions — never treat as main files
SHP_SIDECAR = {".dbf", ".prj", ".shx", ".sbn", ".sbx", ".cpg", ".xml"}

# Lock file patterns (regex applied to full filename)
LOCK_PATTERNS = [
    r"\.shp\..+\.lock$",  # .shp.DESKTOP-PC.xxx.lock
    r"\.shp\.ed\.lock$",
    r"\.shp\.NICNMM\..+\.lock$",
    r"\.tif\.pyrx$",
    r"\.ecw\.aux$",
    r"\.tif\.aux$",
    r"\.tif\.aux\.xml$",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FOLDER SCANNER
# ─────────────────────────────────────────────────────────────────────────────


def scan_folder(folder: str) -> Tuple[List[Path], List[Path]]:
    """
    Scan a folder (cg/ or pb/) and return:
      (raster_files, shp_files)

    Rules applied:
      • Skip .pyrx, .aux, .lock and all lock-pattern files
      • Skip ECW if a same-stem TIF exists (prefer TIF; ECW-only folders are fully supported via auto-conversion)
      • Skip any file whose stem ends with .shp (sidecar XML)
      • Handle spaces and special chars correctly via pathlib
    """
    folder_path = Path(folder)
    all_files = [f for f in folder_path.iterdir() if f.is_file()]

    rasters: List[Path] = []
    shps: List[Path] = []
    skipped: List[str] = []

    # Collect all TIF stems so we can deduplicate ECW
    tif_stems: Set[str] = set()
    for f in all_files:
        if f.suffix.lower() in {".tif", ".tiff"}:
            # Handle compound extensions like .tif.pyrx
            stem = f.name
            for ext in [".tif", ".tiff"]:
                if stem.lower().endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            tif_stems.add(stem.lower())

    for f in all_files:
        name = f.name
        name_low = name.lower()
        suffix = f.suffix.lower()

        # Skip lock files by pattern
        if any(re.search(pat, name, re.IGNORECASE) for pat in LOCK_PATTERNS):
            skipped.append(f"LOCK     {name}")
            continue

        # Skip pyramid overviews (compound extension .tif.pyrx)
        if ".pyrx" in name_low:
            skipped.append(f"PYRX     {name}")
            continue

        # Skip .aux stats files
        if name_low.endswith(".aux") or name_low.endswith(".aux.xml"):
            skipped.append(f"AUX      {name}")
            continue

        # Skip SHP sidecar files
        if suffix in SHP_SIDECAR:
            continue

        # Skip SHP metadata XML (filename.shp.xml)
        if ".shp.xml" in name_low:
            continue

        # Handle SHP files
        if suffix == ".shp":
            shps.append(f)
            continue

        # Handle raster files
        if suffix in RASTER_EXTS:
            # Skip ECW if a same-stem TIF already exists
            if suffix == ".ecw":
                # strip _3857 suffix that KUTRU has before comparing
                ecw_stem = f.stem.lower()
                if ecw_stem in tif_stems or ecw_stem.replace("_3857", "") in tif_stems:
                    skipped.append(f"ECW_DUP  {name}  (TIF exists)")
                    continue
            rasters.append(f)

    log.info(f"\n[Scan] {folder_path.name}/")
    log.info(f"  Rasters : {len(rasters)}")
    for r in rasters:
        log.info(f"    {r.name}  ({r.stat().st_size / 1e9:.1f} GB)")
    log.info(f"  SHPs    : {len(shps)}")
    for s in shps:
        log.info(f"    {s.name}")
    if skipped:
        log.info(f"  Skipped : {len(skipped)}")
        for s in skipped:
            log.info(f"    {s}")

    return rasters, shps


# ─────────────────────────────────────────────────────────────────────────────
# 2.  YOLO LABEL FORMATTER  (shared by streaming extractors below)
# ─────────────────────────────────────────────────────────────────────────────


def _format_yolo_label(
    cid: int,
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    use_obb: bool = False,
) -> str:
    """Return regular YOLO or YOLO-OBB label text for an axis-aligned box."""

    if not use_obb:
        return f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    x1 = max(0.0, min(1.0, cx - bw / 2.0))
    y1 = max(0.0, min(1.0, cy - bh / 2.0))
    x2 = max(0.0, min(1.0, cx + bw / 2.0))
    y2 = y1
    x3 = x2
    y3 = max(0.0, min(1.0, cy + bh / 2.0))
    x4 = x1
    y4 = y3
    return (
        f"{cid} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
        f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MASTER RUNNER  (strip-streamed for 70GB+ rasters)
# ─────────────────────────────────────────────────────────────────────────────


STRIP_ROWS = 4096  # rows per processing strip — 3.5 GB per strip on 213k-wide TIFs


def _raster_info(raster_path: Path):
    """Return (H, W, crs, transform, meta, dtype, n_bands) without reading pixels."""
    with rasterio.open(str(raster_path)) as src:
        return (
            src.height,
            src.width,
            src.crs,
            src.transform,
            src.meta.copy(),
            src.dtypes[0],
            src.count,
        )


def _read_strip_rgb(raster_path: Path, row_off: int, strip_h: int) -> np.ndarray:
    """
    Read a horizontal strip from a raster without loading the full file.
    Returns (strip_h, W, 3) uint8 RGB — the actual read height may be less
    at the bottom of the image.
    """
    with rasterio.open(str(raster_path)) as src:
        H, W = src.height, src.width
        actual_h = min(strip_h, H - row_off)
        win = rasterio.windows.Window(0, row_off, W, actual_h)  # type: ignore
        n = src.count
        if n >= 3:
            arr = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
        else:
            band = src.read(1, window=win)
            arr = np.stack([band] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr)
    return arr


def _burn_strip_mask(
    preloaded_shps: list,
    H_full: int,
    W_full: int,
    crs,
    transform,
    row_off: int,
    strip_h: int,
) -> np.ndarray:
    import numpy as np
    import rasterio.features

    actual_h = min(strip_h, H_full - row_off)
    mask = np.zeros((actual_h, W_full), dtype=np.uint8)

    # Shift the affine transform down by row_off pixels
    strip_transform = rasterio.transform.from_origin(
        transform.c,  # west (x origin unchanged)
        transform.f + row_off * transform.e,  # north shifts down
        transform.a,  # pixel width
        -transform.e,  # pixel height (positive value)
    )

    strip_bounds_minx = transform.c
    strip_bounds_maxy = transform.f + row_off * transform.e
    strip_bounds_maxx = transform.c + W_full * transform.a
    strip_bounds_miny = transform.f + (row_off + actual_h) * transform.e

    from shapely.geometry import box

    strip_box = box(
        strip_bounds_minx,
        min(strip_bounds_miny, strip_bounds_maxy),
        strip_bounds_maxx,
        max(strip_bounds_miny, strip_bounds_maxy),
    )

    for class_id, valid_geoms, tree in preloaded_shps:
        intersecting_indices = tree.query(strip_box)

        shapes_to_rasterize = []
        for idx in intersecting_indices:
            geom = valid_geoms[idx]
            if geom.intersects(strip_box):
                shapes_to_rasterize.append((geom, class_id))

        if shapes_to_rasterize:
            try:
                burned = rasterio.features.rasterize(
                    shapes_to_rasterize,
                    out_shape=(actual_h, W_full),
                    transform=strip_transform,
                    fill=0,
                    dtype=np.uint8,
                    merge_alg=rasterio.enums.MergeAlg.replace,
                )
                if burned is not None:
                    mask = np.where(burned > 0, burned, mask)
            except Exception:
                pass

    return mask


def _tile_strip(
    img_strip: np.ndarray,
    mask_strip: np.ndarray,
    row_off: int,
    patch_size: int,
    overlap: int,
    out_img_dir: str,
    out_mask_dir: str,
    safe_prefix: str,
    min_fg: float = 0.003,
) -> int:
    """Tile a single strip into patches. row_off is used in the filename."""
    stride = patch_size - overlap
    H, W = img_strip.shape[:2]

    # Pre-calculate all valid tile coordinates
    tiles = []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            tiles.append((r, c))

    def _process_tile(tile_coords):
        r, c = tile_coords
        r2, c2 = min(r + patch_size, H), min(c + patch_size, W)
        pi = img_strip[r:r2, c:c2]
        pm = mask_strip[r:r2, c:c2]
        ph, pw = pi.shape[:2]

        # Early exit before allocating padding memory
        if np.count_nonzero(pm) / (patch_size * patch_size) < min_fg:
            return 0

        if ph == patch_size and pw == patch_size:
            pad_i = pi
            pad_m = pm
        else:
            pad_i = cv2.copyMakeBorder(
                pi, 0, patch_size - ph, 0, patch_size - pw, cv2.BORDER_REFLECT_101
            )
            pad_m = cv2.copyMakeBorder(
                pm, 0, patch_size - ph, 0, patch_size - pw, cv2.BORDER_REFLECT_101
            )

        abs_r = row_off + r
        name = f"{safe_prefix}_{abs_r:06d}_{c:06d}.png"
        cv2.imwrite(
            str(Path(out_img_dir) / name), cv2.cvtColor(pad_i, cv2.COLOR_RGB2BGR)
        )
        cv2.imwrite(str(Path(out_mask_dir) / name), pad_m)
        return 1

    saved = 0
    # Use all P-Cores (ThreadPoolExecutor avoids multiprocessing spawn overhead for IO-bound cv2 writes)
    n_workers = max(1, (os.cpu_count() or 4) - 2)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(
            tqdm(
                executor.map(_process_tile, tiles),
                total=len(tiles),
                desc="      Tiling",
                leave=False,
            )
        )
        saved = sum(results)

    return saved


def _process_single_raster(raster_path, shps, built_up_shp, utility_shps):
    import gc
    import os
    import re
    import time

    import geopandas as gpd
    import numpy as np
    import rasterio
    from shapely.strtree import STRtree
    from tqdm import tqdm

    import config
    from config import INFRA_TYPE_MAP, ROOF_TYPE_MAP, SHP_LAYER_ROLES

    cfg1 = config.STAGE1
    cfg2a = config.STAGE2A
    cfg2b = config.STAGE2B

    log.info(f"\n  Processing {raster_path.name} on PID {os.getpid()}")

    # Convert ECW → TIF in-process before opening (rasterio lacks ECW driver by default)
    import tempfile as _tempfile
    _ecw_tmp = None
    if raster_path.suffix.lower() == ".ecw":
        from utils.ecw_compat import ecw_to_tif
        try:
            _ecw_tmp = _tempfile.TemporaryDirectory(prefix="geointel_ecw_preproc_")
            raster_path = ecw_to_tif(raster_path, Path(_ecw_tmp.name))
        except RuntimeError as ecw_err:
            log.info(str(ecw_err))
            _ecw_tmp.cleanup()
            return 0, 0, 0, 1

    try:
        H, W, crs, transform, meta, dtype, n_bands = _raster_info(raster_path)
    except Exception as e:
        log.info(f"    [SKIP] Cannot open {raster_path.name}: {e}")
        if _ecw_tmp is not None:
            _ecw_tmp.cleanup()
        return 0, 0, 0, 1  # patches, crops, infra, failed

    ram_gb = H * W * 3 / 1e9
    log.info(
        f"    {raster_path.name} : {W:,} x {H:,} px  ({ram_gb:.1f} GB if loaded whole)"
    )

    safe_prefix = re.sub(r"[^\w]", "_", raster_path.stem)[:40]
    mask_tif_path = str(
        config.TRAIN_MASKS / f"{safe_prefix}_{int(time.time())}_{os.getpid()}_mask.tif"
    )
    mask_meta = meta.copy()
    mask_meta.update(count=1, dtype=rasterio.uint8, compress="lzw")

    # --- PRELOAD SHAPEFILES FOR THIS RASTER ---
    log.info(
        f"    {raster_path.name}: Preloading and indexing SHP geometries into memory..."
    )
    preloaded_shps = []
    for shp_path in shps:
        stem_low = shp_path.stem.lower().rstrip("_")
        role = SHP_LAYER_ROLES.get(stem_low)
        if role is None:
            for key, val in SHP_LAYER_ROLES.items():
                if stem_low.startswith(key[:10]):
                    role = val
                    break
        if role is None:
            continue

        class_id, _ = role
        if class_id == 0:
            continue

        try:
            gdf = gpd.read_file(str(shp_path))
        except Exception:
            continue

        if len(gdf) == 0:
            continue

        if gdf.crs is not None and gdf.crs != crs:
            try:
                gdf = gdf.to_crs(crs)
            except Exception:
                continue

        valid_geoms = []
        for geom in gdf.geometry:
            if geom is not None and not geom.is_empty:
                if not geom.is_valid:
                    try:
                        geom = geom.buffer(0)
                    except Exception:
                        continue
                if geom.is_valid:
                    valid_geoms.append(geom)

        if valid_geoms:
            tree = STRtree(valid_geoms)
            preloaded_shps.append((class_id, valid_geoms, tree))

    patch_size = cfg1["patch_size"]
    overlap = cfg1["overlap"]
    strip_advance = STRIP_ROWS - patch_size
    n_strips = max(1, (H - patch_size + strip_advance - 1) // strip_advance + 1)

    raster_patches = 0
    with rasterio.open(mask_tif_path, "w", **mask_meta) as mask_dst:
        for strip_idx in range(n_strips):
            row_off = strip_idx * strip_advance
            if row_off >= H:
                break

            try:
                img_strip = _read_strip_rgb(raster_path, row_off, STRIP_ROWS)
            except Exception as e:
                log.info(f"    [SKIP strip {strip_idx} on {raster_path.name}] {e}")
                continue

            actual_h = img_strip.shape[0]

            mask_strip = _burn_strip_mask(
                preloaded_shps, H, W, crs, transform, row_off, actual_h
            )

            write_row_start = 0 if strip_idx == 0 else patch_size
            write_row_end = actual_h
            if write_row_end > write_row_start:
                WindowCls = getattr(rasterio.windows, "Window")
                win = WindowCls(
                    0, row_off + write_row_start, W, write_row_end - write_row_start
                )
                mask_dst.write(
                    mask_strip[write_row_start:write_row_end][np.newaxis],
                    window=win,
                )

            n = _tile_strip(
                img_strip,
                mask_strip,
                row_off,
                patch_size,
                overlap,
                str(config.PATCH_DIR),
                str(config.MASK_DIR),
                safe_prefix,
                min_fg=cfg1.get("min_fg_ratio", 0.003),
            )
            raster_patches += n

            pct = min(100, int((row_off + actual_h) / H * 100))
            log.info(
                f"    {raster_path.name} | Strip {strip_idx + 1:2d}/{n_strips} | patches={n} [{pct}%]"
            )

            del img_strip, mask_strip
            gc.collect()

    n_crops = _extract_crops_streaming(
        raster_path,
        built_up_shp,
        cfg2a.get("shp_roof_cols", cfg2a["shp_roof_col"]),
        ROOF_TYPE_MAP,
        str(config.CROP_DIR),
        cfg2a["crop_size"],
        cfg2a["min_crop_px"],
    )

    n_infra = 0
    if utility_shps:
        n_infra = _extract_infra_streaming(
            raster_path,
            utility_shps,
            cfg2b.get("shp_infra_cols", cfg2b["shp_infra_col"]),
            INFRA_TYPE_MAP,
            cfg2b["class_names"],
            str(config.YOLO_DIR / "images"),
            str(config.YOLO_DIR / "labels"),
            cfg2b["img_size"],
            class_buffer_px=cfg2b.get("class_buffer_px"),
            neg_tile_ratio=cfg2b.get("neg_tile_ratio", 0.0),
            use_obb=cfg2b.get("use_obb", False),
        )

    gc.collect()
    if _ecw_tmp is not None:
        _ecw_tmp.cleanup()
    return raster_patches, n_crops, n_infra, 0


def preprocess_folder(folder: str, config) -> Dict:
    """
    Process one folder (cg/ or pb/) in memory-safe strips.

    For giant TIFs (e.g. 213734x112836 = 72 GB RGB) we never load the full
    raster. Instead we iterate in STRIP_ROWS-row horizontal bands:
      1. Read strip RGB    (~2.6 GB per strip)
      2. Burn mask strip   (~0.9 GB per strip)
      3. Tile strip        (patches saved to disk)
      4. Discard both      (GC frees memory)
      5. Move to next strip

    Peak RAM per raster: ~3.5 GB  (safe for 32 GB)
    """
    import gc

    from config import INFRA_TYPE_MAP, ROOF_TYPE_MAP, SHP_LAYER_ROLES

    cfg1 = config.STAGE1
    cfg2a = config.STAGE2A
    cfg2b = config.STAGE2B

    rasters, shps = scan_folder(folder)
    if not rasters:
        log.warning("No rasters found in %s", folder)
        return {"rasters": 0, "patches": 0, "crops": 0, "infra": 0}

    shp_by_stem = {s.stem.lower().rstrip("_"): s for s in shps}
    built_up_shp = shp_by_stem.get("built_up_area_type") or shp_by_stem.get(
        "built_up_area_typ"
    )
    utility_shps = [s for k, s in shp_by_stem.items() if k.startswith("utility")]

    os.makedirs(str(config.PATCH_DIR), exist_ok=True)
    os.makedirs(str(config.MASK_DIR), exist_ok=True)
    os.makedirs(str(config.TRAIN_MASKS), exist_ok=True)

    total_patches = 0
    total_crops = 0
    total_infra = 0
    processed = 0
    failed = 0

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    max_workers = max(1, multiprocessing.cpu_count() // 2)
    max_workers = min(max_workers, 5)  # 5 concurrent rasters is roughly 17GB RAM
    log.info(f"  Spawning ProcessPoolExecutor with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_single_raster, r, shps, built_up_shp, utility_shps
            ): r
            for r in rasters
        }

        for future in as_completed(futures):
            raster_path = futures[future]
            try:
                p, c, i, f = future.result()
                total_patches += p
                total_crops += c
                total_infra += i
                failed += f
                if f == 0:
                    processed += 1
            except Exception as e:
                log.info(
                    f"    [CRASH] Raster {raster_path.name} failed with exception: {e}"
                )
                import traceback

                traceback.print_exc()
                failed += 1
    return {
        "folder": str(folder),
        "rasters": processed,
        "failed": failed,
        "patches": total_patches,
        "crops": total_crops,
        "infra": total_infra,
    }


def _extract_crops_streaming(
    raster_path: Path,
    built_up_shp: Optional[Path],
    roof_col,
    roof_type_map: dict,
    out_dir: str,
    crop_size: int,
    min_px: int,
) -> int:
    if built_up_shp is None or not built_up_shp.exists():
        return 0

    try:
        import geopandas as gpd

        gdf = gpd.read_file(str(built_up_shp))
    except Exception as e:
        log.info(f"      [SKIP crops] {e}")
        return 0

    saved = 0
    import re

    stem = re.sub(r"[^\w]", "_", raster_path.stem)[:20]

    try:
        from pathlib import Path

        import cv2
        import numpy as np
        import rasterio

        with rasterio.open(str(raster_path)) as src:
            crs = src.crs
            H, W = src.height, src.width

            if gdf.crs and gdf.crs != crs:
                try:
                    gdf = gdf.to_crs(crs)
                except Exception:
                    pass

            # gdf.iterrows builds a fresh Series per row — slow on 10K+ buildings.
            # Pull the columns we need into native arrays once, then iterate.
            geoms_arr = gdf.geometry.values
            roof_candidates = list(roof_col) if isinstance(roof_col, (list, tuple)) else [roof_col]
            roof_candidates.extend(["Roof_type", "roof_type", "type", "Type"])
            actual_roof_col = find_attribute_column(gdf, roof_candidates)
            if actual_roof_col:
                types_arr = gdf[actual_roof_col].astype(object).values
            else:
                log.warning(
                    "      [WARN crops] No rooftop type column found in %s; "
                    "writing crops as Other", built_up_shp.name
                )
                types_arr = np.full(len(gdf), "other", dtype=object)

            # Pre-resolve the label for every distinct raw type once. The
            # original code re-ran the dict + substring fallback per row.
            label_cache: dict = {}

            def _label_for(raw: str) -> str:
                norm_raw = normalise_dbf_value(raw)
                if norm_raw in label_cache:
                    return label_cache[norm_raw]
                lbl = canonical_mapped_label(norm_raw, roof_type_map, default="Other")
                lbl = lbl or "Other"
                label_cache[norm_raw] = lbl
                return lbl

            # GDAL Dataset is not thread-safe, so reads stay on the main thread.
            # The encode+write step (cv2.imwrite) releases the GIL, so we
            # background it onto a small thread pool to overlap with the next
            # tile's read.
            from concurrent.futures import ThreadPoolExecutor
            from tqdm import tqdm

            pool = ThreadPoolExecutor(max_workers=4)
            futures = []
            label_dir_cache: dict = {}

            try:
                for idx in tqdm(range(len(geoms_arr)), desc="      Crops", leave=False):
                    geom = geoms_arr[idx]
                    if geom is None or geom.is_empty:
                        continue

                    raw_type = types_arr[idx] if idx < len(types_arr) else "other"
                    label = _label_for(raw_type)
                    label_dir = label_dir_cache.get(label)
                    if label_dir is None:
                        label_dir = Path(out_dir) / label
                        label_dir.mkdir(parents=True, exist_ok=True)
                        label_dir_cache[label] = label_dir

                    try:
                        minx, miny, maxx, maxy = geom.bounds
                        r1, c1 = src.index(minx, maxy)
                        r2, c2 = src.index(maxx, miny)
                        if r1 > r2:
                            r1, r2 = r2, r1
                        if c1 > c2:
                            c1, c2 = c2, c1
                        r1, r2 = max(0, min(r1, r2)), min(H, max(r1, r2))
                        c1, c2 = max(0, min(c1, c2)), min(W, max(c1, c2))
                        if r2 - r1 < min_px or c2 - c1 < min_px:
                            continue

                        win = rasterio.windows.Window(c1, r1, c2 - c1, r2 - r1)
                        n_b = src.count
                        if n_b >= 3:
                            crop = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
                        else:
                            band = src.read(1, window=win)
                            crop = np.stack([band] * 3, axis=-1)

                        if crop.dtype != np.uint8:
                            crop = _to_uint8(crop)

                        crop = cv2.resize(
                            crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
                        )
                        out_p = str(label_dir / f"{stem}_{idx:06d}.png")
                        bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        futures.append(pool.submit(cv2.imwrite, out_p, bgr))
                        saved += 1
                    except Exception:
                        pass

                # Drain the write queue.
                for f in futures:
                    try:
                        f.result()
                    except Exception:
                        pass
            finally:
                pool.shutdown(wait=True)

    except Exception as e:
        log.info(f"      [SKIP crops] Cannot open raster: {e}")

    return saved


def _extract_infra_streaming(
    raster_path: Path,
    utility_shps: List[Path],
    infra_col,
    infra_type_map: dict,
    class_names: List[str],
    out_img_dir: str,
    out_label_dir: str,
    tile_size: int,
    buffer_px: int = 20,
    class_buffer_px: Optional[Dict[str, int]] = None,
    neg_tile_ratio: float = 0.0,
    use_obb: bool = False,
) -> int:
    """
    Generate YOLO labels for infrastructure by reading only the tiles that
    contain annotated features — never loads the full raster.

    Improvements over original:
      • Class-specific bounding box sizes (driven by ``class_buffer_px`` arg;
        config currently sets transformer=100, overhead_tank=80, well=40)
      • Tiles centered on objects instead of grid-snapped (no edge cropping)
      • Negative tile sampling to reduce false positive rate
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    cls_to_id = {c.lower(): i for i, c in enumerate(class_names)}
    id_to_name = {i: c.lower() for i, c in enumerate(class_names)}
    total = 0

    # Per-raster prefix prevents label collisions when multiple TIFs share the
    # same YOLO_DIR — without it, two rasters with an object at the same pixel
    # offset would overwrite each other's tiles silently.
    safe_prefix = re.sub(r"[^\w]", "_", raster_path.stem)[:40]

    try:
        with rasterio.open(str(raster_path)) as src:
            crs = src.crs
            geo_transform = src.transform
            H, W = src.height, src.width
    except Exception as e:
        log.info(f"      [SKIP infra] {e}")
        return 0

    from collections import defaultdict

    # Collect all infra objects with their pixel coordinates
    infra_objects: List[Tuple[int, int, int]] = []  # (class_id, px_col, px_row)

    # Inverse transform is constant for this raster — hoist out of the loop.
    inv_t = ~geo_transform
    # Pre-resolve every raw type to a class id once. ``infra_type_map`` lookups
    # ran per row before, including the substring fallback.
    _cid_cache: dict = {}

    def _cid_for(raw: str) -> int:
        norm_raw = normalise_dbf_value(raw)
        if norm_raw in _cid_cache:
            return _cid_cache[norm_raw]
        cls_name = canonical_mapped_label(norm_raw, infra_type_map, default=None)
        cid = cls_to_id.get(cls_name, -1) if cls_name else -1
        _cid_cache[norm_raw] = cid
        return cid

    for shp_path in utility_shps:
        try:
            gdf = gpd.read_file(str(shp_path))
        except Exception:
            continue

        if gdf.crs and gdf.crs != crs:
            try:
                gdf = gdf.to_crs(crs)
            except Exception:
                continue

        # iterrows materialises a Series per row; use the underlying arrays.
        geoms_arr = gdf.geometry.values
        infra_candidates = list(infra_col) if isinstance(infra_col, (list, tuple)) else [infra_col]
        infra_candidates.extend(
            ["Utility_Ty", "utility_type", "Utility_Type", "type", "Type", "name", "Name"]
        )
        actual_infra_col = find_attribute_column(gdf, infra_candidates)
        if actual_infra_col:
            raws = gdf[actual_infra_col].astype(object).fillna("").values
        else:
            log.warning(
                "      [WARN infra] No infrastructure type column found in %s; "
                "columns=%s", shp_path.name, list(gdf.columns)
            )
            raws = np.full(len(gdf), "", dtype=object)

        for k in range(len(geoms_arr)):
            raw = str(raws[k] or "").lower().strip()
            cid = _cid_for(raw)
            if cid < 0:
                continue

            geom = geoms_arr[k]
            if geom is None or geom.is_empty:
                continue

            try:
                pt = geom.centroid
                pc_f, pr_f = inv_t * (pt.x, pt.y)
                pr, pc = int(pr_f), int(pc_f)
                if pr < 0 or pr >= H or pc < 0 or pc >= W:
                    continue
                infra_objects.append((cid, int(pc), int(pr)))
                total += 1
            except Exception:
                continue

    if not infra_objects:
        log.info("      No infrastructure features found")
        return 0

    # ── Group objects into CENTERED tiles ──────────────────────────────────
    # Instead of grid-snapping, center each tile on the object cluster.
    # Objects close together share a tile; isolated ones get their own.
    tile_contents: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
    half = tile_size // 2

    for cid, pc, pr in infra_objects:
        # Center tile on this object
        tr = max(0, min(pr - half, H - tile_size))
        tc = max(0, min(pc - half, W - tile_size))
        # Snap to nearest existing tile if close (within half tile)
        merged = False
        for etr, etc in list(tile_contents.keys()):
            if abs(tr - etr) < half and abs(tc - etc) < half:
                tile_contents[(etr, etc)].append((cid, pc, pr))
                merged = True
                break
        if not merged:
            tile_contents[(tr, tc)] = [(cid, pc, pr)]

    # ── Write positive tiles with class-specific bounding boxes ───────────
    # GDAL Dataset isn't thread-safe, so reads stay sequential; the encode+
    # write step (cv2.imwrite + text-file write) releases the GIL, so it's
    # safe to background it onto a small thread pool.
    from concurrent.futures import ThreadPoolExecutor as _Pool

    write_pool = _Pool(max_workers=4)
    write_futures = []

    def _write_tile(img_path, bgr, lbl_path, label_text):
        cv2.imwrite(img_path, bgr)
        Path(lbl_path).write_text(label_text)

    try:
        with rasterio.open(str(raster_path)) as src:
            n_b = src.count
            for (tr, tc), pts in tqdm(
                tile_contents.items(), desc="      Infra+", leave=False
            ):
                r2, c2 = min(tr + tile_size, H), min(tc + tile_size, W)
                ph, pw = r2 - tr, c2 - tc
                if ph <= 0 or pw <= 0:
                    continue

                try:
                    win = rasterio.windows.Window(tc, tr, pw, ph)  # type: ignore
                    if n_b >= 3:
                        tile = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
                    else:
                        band = src.read(1, window=win)
                        tile = np.stack([band] * 3, axis=-1)
                    if tile.dtype != np.uint8:
                        tile = _to_uint8(tile)
                except Exception:
                    continue

                if tile is None or tile.size == 0 or 0 in tile.shape:
                    continue

                name = f"{safe_prefix}_infra_{tr:06d}_{tc:06d}"
                lines = []
                for cid, pc, pr in pts:
                    cls_name_lower = id_to_name.get(cid, "")
                    if class_buffer_px and cls_name_lower in class_buffer_px:
                        buf = class_buffer_px[cls_name_lower]
                    else:
                        buf = buffer_px

                    cx = (pc - tc) / pw
                    cy = (pr - tr) / ph
                    bw = (buf * 2) / pw
                    bh = (buf * 2) / ph
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    lines.append(_format_yolo_label(cid, cx, cy, bw, bh, use_obb=use_obb))

                bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                write_futures.append(write_pool.submit(
                    _write_tile,
                    str(Path(out_img_dir) / f"{name}.png"),
                    bgr,
                    str(Path(out_label_dir) / f"{name}.txt"),
                    "\n".join(lines),
                ))

            # ── Negative tile sampling ────────────────────────────────────
            if neg_tile_ratio > 0:
                import random

                n_neg = max(1, int(len(tile_contents) * neg_tile_ratio))
                occupied = set(tile_contents.keys())
                neg_count = 0
                attempts = 0
                max_attempts = n_neg * 10

                while neg_count < n_neg and attempts < max_attempts:
                    attempts += 1
                    rand_r = random.randint(0, max(0, H - tile_size))
                    rand_c = random.randint(0, max(0, W - tile_size))

                    too_close = any(
                        abs(rand_r - etr) < tile_size and abs(rand_c - etc) < tile_size
                        for etr, etc in occupied
                    )
                    if too_close:
                        continue

                    r2, c2 = min(rand_r + tile_size, H), min(rand_c + tile_size, W)
                    ph, pw = r2 - rand_r, c2 - rand_c
                    if ph <= 0 or pw <= 0:
                        continue

                    try:
                        win = rasterio.windows.Window(rand_c, rand_r, pw, ph)
                        if n_b >= 3:
                            tile = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
                        else:
                            band = src.read(1, window=win)
                            tile = np.stack([band] * 3, axis=-1)
                        if tile.dtype != np.uint8:
                            tile = _to_uint8(tile)
                    except Exception:
                        continue

                    if tile.max() < 10:
                        continue

                    name = f"{safe_prefix}_infra_neg_{rand_r:06d}_{rand_c:06d}"
                    bgr = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
                    write_futures.append(write_pool.submit(
                        _write_tile,
                        str(Path(out_img_dir) / f"{name}.png"),
                        bgr,
                        str(Path(out_label_dir) / f"{name}.txt"),
                        "",
                    ))
                    neg_count += 1

                if neg_count > 0:
                    log.info(f"      Negative tiles: {neg_count} (no-infra background)")

        # Drain queued writes before we return.
        for f in write_futures:
            try:
                f.result()
            except Exception:
                pass
    finally:
        write_pool.shutdown(wait=True)

    log.info(
        f"      Infrastructure objects: {total}  |  Positive tiles: {len(tile_contents)}"
    )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Convert any numeric raster array to uint8 with per-channel 2nd–98th percentile
    stretch.

    Using min-max stretch (the old approach) is distorted by sensor outliers and
    dead pixels, causing colour channels to shift relative to each other.
    Percentile stretch preserves natural colour ratios and is the industry standard
    for drone orthomosaic display.
    """
    if arr.dtype == np.uint8:
        return arr

    arr_f = arr.astype(np.float32)

    if arr_f.ndim == 3:
        C = arr_f.shape[2]
        # Build a (C, 2) array of (p2, p98) across non-zero pixels per channel.
        # Vectorising over the spatial axes drops C Python iterations + C percentile
        # calls — the per-strip cost was dominating large-raster preprocessing.
        lo = np.zeros(C, dtype=np.float32)
        hi = np.ones(C, dtype=np.float32)
        for i in range(C):
            ch = arr_f[:, :, i]
            nz = ch[ch > 0]
            if nz.size > 0:
                p2, p98 = np.percentile(nz, [2, 98])
                if p98 <= p2:
                    p2, p98 = float(ch.min()), float(ch.max())
            else:
                p2, p98 = 0.0, 1.0
            lo[i] = p2
            hi[i] = p98
        denom = np.where(hi > lo, hi - lo, 1.0).astype(np.float32)
        scaled = (arr_f - lo) / denom * 255.0
        out = np.clip(scaled, 0, 255)
        # Zero out channels that were degenerate (all-zero range)
        degenerate = (hi <= lo)
        if degenerate.any():
            out[:, :, degenerate] = 0
    else:
        flat = arr_f[arr_f > 0]
        if flat.size > 0:
            p2, p98 = np.percentile(flat, [2, 98])
            if p98 <= p2:
                p2, p98 = float(arr_f.min()), float(arr_f.max())
        else:
            p2, p98 = 0.0, 1.0
        if p98 == p2:
            out = np.zeros_like(arr_f)
        else:
            out = np.clip((arr_f - p2) / (p98 - p2) * 255.0, 0, 255)

    return out.astype(np.uint8)

