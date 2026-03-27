"""
data/preprocessing.py  —  SVAMITVA dataset edition
════════════════════════════════════════════════════
Handles the exact layout found in cg/ and pb/:

  dataset/cg/  or  dataset/pb/
    ├── VILLAGE_NAME_ORTHO.tif     ← one or more ortho rasters
    ├── VILLAGE_NAME_3857.ecw      ← compressed duplicate → SKIPPED
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

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Only these extensions are treated as rasters
RASTER_EXTS = {".tif", ".tiff", ".ecw", ".img", ".vrt", ".jp2"}

# Extensions that look like rasters but aren't
SKIP_EXTS = {
    ".pyrx",  # pyramid overviews
    ".aux",  # ESRI aux stats
    ".lock",  # ArcGIS locks
    ".sr.lock",  # spatial reference locks
    ".ed.lock",  # editor locks
}

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
      • Skip ECW if a same-stem TIF exists (prefer TIF — larger, uncompressed)
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

    print(f"\n[Scan] {folder_path.name}/")
    print(f"  Rasters : {len(rasters)}")
    for r in rasters:
        print(f"    {r.name}  ({r.stat().st_size / 1e9:.1f} GB)")
    print(f"  SHPs    : {len(shps)}")
    for s in shps:
        print(f"    {s.name}")
    if skipped:
        print(f"  Skipped : {len(skipped)}")
        for s in skipped:
            print(f"    {s}")

    return rasters, shps


# ─────────────────────────────────────────────────────────────────────────────
# 2.  RASTER READER  (with corruption check)
# ─────────────────────────────────────────────────────────────────────────────


def safe_read_raster(
    raster_path: Path,
) -> Optional[Tuple[np.ndarray, dict, object, object]]:
    """
    Read raster → (H,W,3) uint8 RGB, meta, crs, transform.
    Returns None if file is corrupt or unreadable.
    Logs the specific error so you know what went wrong.
    """
    try:
        with rasterio.open(str(raster_path)) as src:
            # Basic sanity checks
            if src.width == 0 or src.height == 0:
                print(f"    [SKIP] {raster_path.name}: zero-size raster")
                return None
            if src.count == 0:
                print(f"    [SKIP] {raster_path.name}: no bands")
                return None

            meta = src.meta.copy()
            crs = src.crs
            transform = src.transform
            n_bands = src.count

            # Read bands — try RGB first, fall back to grayscale
            try:
                if n_bands >= 3:
                    arr = src.read([1, 2, 3]).transpose(1, 2, 0)
                else:
                    band = src.read(1)
                    arr = np.stack([band] * 3, axis=-1)
            except Exception as e:
                print(f"    [SKIP] {raster_path.name}: read error — {e}")
                return None

        # Convert to uint8
        if arr.dtype != np.uint8:
            arr = _to_uint8(arr)

        # Check for all-zero / all-NaN (corrupt file symptom)
        if arr.max() == 0:
            print(
                f"    [WARN] {raster_path.name}: all-zero pixel values — file may be corrupt"
            )
            # Don't skip — might just be a very dark image

        print(f"    [OK]  {raster_path.name}  {arr.shape[1]}×{arr.shape[0]} px")
        return arr, meta, crs, transform

    except rasterio.errors.RasterioIOError as e:
        print(f"    [SKIP] {raster_path.name}: RasterioIOError — {e}")
        return None
    except Exception as e:
        print(f"    [SKIP] {raster_path.name}: {type(e).__name__} — {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SHP → RASTER MASK  (SVAMITVA-aware)
# ─────────────────────────────────────────────────────────────────────────────


def build_svamitva_mask(
    raster_path: Path,
    shp_files: List[Path],
    shp_roles: dict,  # config.SHP_LAYER_ROLES
    out_path: Optional[str] = None,
) -> np.ndarray:
    """
    Burns ALL relevant SHP layers into a single class mask.

    Uses SHP_LAYER_ROLES to know:
      • which seg class each SHP layer represents (0/1/2/3)
      • which attribute column to read
      • whether to skip the layer (class_id=0 → infrastructure, not seg)

    Skips individual geometries that fail (corrupt/empty) rather than
    crashing the entire raster.
    """
    with rasterio.open(str(raster_path)) as src:
        H, W = src.height, src.width
        crs = src.crs
        transform = src.transform
        meta = src.meta.copy()

    mask = np.zeros((H, W), dtype=np.uint8)
    total_burned = 0

    for shp_path in shp_files:
        stem_low = shp_path.stem.lower().rstrip("_")  # strip trailing _ (Utility_Poly_)
        role = shp_roles.get(stem_low)

        if role is None:
            # Try prefix match (handles truncated names)
            for key, val in shp_roles.items():
                if stem_low.startswith(key[:10]):
                    role = val
                    break

        if role is None:
            continue  # Unknown SHP — skip silently

        class_id, attr_col = role
        if class_id == 0:
            continue  # Infrastructure SHPs handled in Stage 2B

        # Read SHP safely
        try:
            gdf = gpd.read_file(str(shp_path))
        except Exception as e:
            print(f"      [SKIP SHP] {shp_path.name}: {e}")
            continue

        if len(gdf) == 0:
            continue

        # Reproject to raster CRS if needed
        if gdf.crs is not None and gdf.crs != crs:
            try:
                gdf = gdf.to_crs(crs)
            except Exception as e:
                print(f"      [WARN CRS] {shp_path.name}: {e}")

        layer_burned = 0
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            if not geom.is_valid:
                try:
                    geom = geom.buffer(0)  # fix self-intersections
                except Exception:
                    continue

            try:
                burned = rasterio.features.rasterize(
                    [(geom, class_id)],
                    out_shape=(H, W),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                    merge_alg=rasterio.enums.MergeAlg.replace,
                )
                if burned is not None:
                    mask = np.where(burned > 0, burned, mask)
                layer_burned += 1
            except Exception:
                # Individual geometry failure — skip and continue
                continue

        total_burned += layer_burned
        if layer_burned > 0:
            print(
                f"      {shp_path.name:<35} class={class_id}  features={layer_burned}"
            )

    fg = int((mask > 0).sum())
    tot = H * W
    print(
        f"      Total features burned: {total_burned}  |  "
        f"Coverage: {100 * fg / tot:.1f}%  ({fg:,}/{tot:,} px)"
    )

    if out_path:
        meta.update(count=1, dtype=rasterio.uint8)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(mask[np.newaxis])

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BUILDING CROP EXTRACTOR  (for Stage 2A, rooftop classification)
# ─────────────────────────────────────────────────────────────────────────────


def extract_building_crops_svamitva(
    img_rgb: np.ndarray,
    raster_path: Path,
    built_up_shp: Optional[Path],  # Built_Up_Area_type.shp
    roof_col: str,  # "type"
    roof_type_map: dict,  # ROOF_TYPE_MAP from config
    out_dir: str,
    crop_size: int = 160,
    min_px: int = 24,
) -> int:
    """
    For each polygon in Built_Up_Area_type.shp:
      • Read the 'type' attribute → map to RCC/Tiled/Tin/Other
      • Crop the building bbox from img_rgb
      • Save to out_dir/<class>/<stem>_<idx>.png

    Returns number of crops saved.
    """
    if built_up_shp is None or not built_up_shp.exists():
        print("      [SKIP crops] Built_Up_Area SHP not found")
        return 0

    try:
        gdf = gpd.read_file(str(built_up_shp))
    except Exception as e:
        print(f"      [SKIP crops] Cannot read {built_up_shp.name}: {e}")
        return 0

    H, W = img_rgb.shape[:2]

    # Reproject to pixel CRS if needed
    try:
        with rasterio.open(str(raster_path)) as src:
            crs = src.crs
            transform = src.transform
        if gdf.crs and gdf.crs != crs:
            gdf = gdf.to_crs(crs)
    except Exception as e:
        print(f"      [WARN crops CRS] {e}")
        transform = None

    saved = 0
    skipped = 0
    stem = raster_path.stem[:20]  # prefix crops with raster name

    for idx, row in gdf.iterrows():
        # Get roof type label
        raw_type = str(row.get(roof_col, "other") or "other").lower().strip()
        # Try the type map, fall back to "Other"
        label = roof_type_map.get(raw_type, None)
        if label is None:
            # Fuzzy fallback: check if any key is a substring
            for key, val in roof_type_map.items():
                if key in raw_type or raw_type in key:
                    label = val
                    break
            label = label or "Other"

        label_dir = Path(out_dir) / label
        label_dir.mkdir(parents=True, exist_ok=True)

        geom = row.geometry
        if geom is None or geom.is_empty:
            skipped += 1
            continue

        try:
            # Get bounding box in pixel coords
            minx, miny, maxx, maxy = geom.bounds
            # rasterio ~transform converts geo → pixel
            if transform is None:
                continue
            row_min, col_min = ~transform * (minx, maxy)
            row_max, col_max = ~transform * (maxx, miny)
            r1, r2 = int(min(row_min, row_max)), int(max(row_min, row_max))
            c1, c2 = int(min(col_min, col_max)), int(max(col_min, col_max))

            # Clamp to image bounds
            r1, r2 = max(0, r1), min(H, r2)
            c1, c2 = max(0, c1), min(W, c2)

            if r2 - r1 < min_px or c2 - c1 < min_px:
                skipped += 1
                continue

            crop = img_rgb[r1:r2, c1:c2]
            crop = cv2.resize(crop, (crop_size, crop_size))
            out_p = label_dir / f"{stem}_{idx:06d}.png"
            cv2.imwrite(str(out_p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            saved += 1

        except Exception:
            skipped += 1
            continue

    print(f"      Building crops: {saved} saved, {skipped} skipped")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 5.  UTILITY SHP → YOLO LABELS  (for Stage 2B)
# ─────────────────────────────────────────────────────────────────────────────


def extract_infra_yolo(
    img_rgb: np.ndarray,
    raster_path: Path,
    utility_shps: List[Path],  # Utility.shp + Utility_Poly.shp
    infra_col: str,  # "utility_type"
    infra_type_map: dict,
    class_names: List[str],
    out_img_dir: str,
    out_label_dir: str,
    tile_size: int = 1280,
    buffer_px: int = 20,
) -> int:
    """
    Convert Utility point/polygon features → YOLO format label files.
    Returns number of infrastructure objects written.
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    cls_to_id = {c.lower(): i for i, c in enumerate(class_names)}
    H, W = img_rgb.shape[:2]
    total = 0

    try:
        with rasterio.open(str(raster_path)) as src:
            transform = src.transform
            crs = src.crs
    except Exception as e:
        print(f"      [SKIP infra] Cannot open raster for CRS: {e}")
        return 0

    # Collect all infra points across both Utility and Utility_Poly
    infra_pts: List[Tuple[int, int, int]] = []  # (class_id, px_col, px_row)

    for shp_path in utility_shps:
        try:
            gdf = gpd.read_file(str(shp_path))
        except Exception as e:
            print(f"      [SKIP infra SHP] {shp_path.name}: {e}")
            continue

        if gdf.crs and gdf.crs != crs:
            try:
                gdf = gdf.to_crs(crs)
            except Exception:
                pass

        for _, row in gdf.iterrows():
            raw = str(row.get(infra_col, "") or "").lower().strip()
            cls_name = infra_type_map.get(raw, None)
            if cls_name is None:
                for key, val in infra_type_map.items():
                    if key in raw:
                        cls_name = val
                        break
            if cls_name is None:
                continue

            cid = cls_to_id.get(cls_name, -1)
            if cid == -1:
                continue

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            try:
                pt = geom.centroid
                px_row, px_col = ~transform * (pt.x, pt.y)
                infra_pts.append((cid, int(px_col), int(px_row)))
                total += 1
            except Exception:
                continue

    if not infra_pts:
        print("      No infrastructure features found")
        return 0

    # Tile the raster and write YOLO labels per tile
    from collections import defaultdict

    tile_groups = defaultdict(list)
    for cid, pc, pr in infra_pts:
        tr = (pr // tile_size) * tile_size
        tc = (pc // tile_size) * tile_size
        tile_groups[(tr, tc)].append((cid, pc, pr))

    for (tr, tc), pts in tile_groups.items():
        r2, c2 = min(tr + tile_size, H), min(tc + tile_size, W)
        tile = img_rgb[tr:r2, tc:c2]
        ph, pw = tile.shape[:2]
        name = f"infra_{tr:05d}_{tc:05d}"

        cv2.imwrite(
            str(Path(out_img_dir) / f"{name}.png"),
            cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
        )

        lines = []
        for cid, pc, pr in pts:
            cx = (pc - tc) / pw
            cy = (pr - tr) / ph
            bw = (buffer_px * 2) / pw
            bh = (buffer_px * 2) / ph
            cx, cy = max(0, min(1, cx)), max(0, min(1, cy))
            bw, bh = max(0.001, bw), max(0.001, bh)
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (Path(out_label_dir) / f"{name}.txt").write_text("\n".join(lines))

    print(f"      Infrastructure objects: {total}")
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 6.  PATCH TILER
# ─────────────────────────────────────────────────────────────────────────────


def tile_image_and_mask(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    patch_size: int,
    overlap: int,
    out_img_dir: str,
    out_mask_dir: str,
    prefix: str,
    min_fg: float = 0.003,
) -> int:
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    H, W = img_rgb.shape[:2]
    stride = patch_size - overlap
    saved = 0

    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r2, c2 = min(r + patch_size, H), min(c + patch_size, W)
            pi = img_rgb[r:r2, c:c2]
            pm = mask[r:r2, c:c2]
            ph, pw = pi.shape[:2]

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

            if np.sum(pad_m > 0) / patch_size**2 < min_fg:
                continue

            # Safe filename — replace spaces/special chars
            safe_prefix = re.sub(r"[^\w]", "_", prefix)[:40]
            name = f"{safe_prefix}_{r:05d}_{c:05d}.png"
            cv2.imwrite(
                str(Path(out_img_dir) / name), cv2.cvtColor(pad_i, cv2.COLOR_RGB2BGR)
            )
            cv2.imwrite(str(Path(out_mask_dir) / name), pad_m)
            saved += 1

    return saved


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MASTER RUNNER
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

    print(f"\n  Processing {raster_path.name} on PID {os.getpid()}")
    try:
        H, W, crs, transform, meta, dtype, n_bands = _raster_info(raster_path)
    except Exception as e:
        print(f"    [SKIP] Cannot open {raster_path.name}: {e}")
        return 0, 0, 0, 1  # patches, crops, infra, failed

    ram_gb = H * W * 3 / 1e9
    print(
        f"    {raster_path.name} : {W:,} x {H:,} px  ({ram_gb:.1f} GB if loaded whole)"
    )

    safe_prefix = re.sub(r"[^\w]", "_", raster_path.stem)[:40]
    mask_tif_path = str(
        config.TRAIN_MASKS / f"{safe_prefix}_{int(time.time())}_{os.getpid()}_mask.tif"
    )
    mask_meta = meta.copy()
    mask_meta.update(count=1, dtype=rasterio.uint8, compress="lzw")

    # --- PRELOAD SHAPEFILES FOR THIS RASTER ---
    print(
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
                print(f"    [SKIP strip {strip_idx} on {raster_path.name}] {e}")
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
                min_fg=0.003,
            )
            raster_patches += n

            pct = min(100, int((row_off + actual_h) / H * 100))
            print(
                f"    {raster_path.name} | Strip {strip_idx + 1:2d}/{n_strips} | patches={n} [{pct}%]"
            )

            del img_strip, mask_strip
            gc.collect()

    n_crops = _extract_crops_streaming(
        raster_path,
        built_up_shp,
        cfg2a["shp_roof_col"],
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
            cfg2b["shp_infra_col"],
            INFRA_TYPE_MAP,
            cfg2b["class_names"],
            str(config.YOLO_DIR / "images"),
            str(config.YOLO_DIR / "labels"),
            cfg2b["img_size"],
            class_buffer_px=cfg2b.get("class_buffer_px"),
            neg_tile_ratio=cfg2b.get("neg_tile_ratio", 0.0),
        )

    gc.collect()
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
        print(f"  [WARN] No rasters found in {folder}")
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
    print(f"  Spawning ProcessPoolExecutor with {max_workers} workers...")

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
                print(
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
    roof_col: str,
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
        print(f"      [SKIP crops] {e}")
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

            # Sequential reading is safe, stable, and often faster due to GDAL's block cache vs locking overhead
            from tqdm import tqdm

            for idx, row in tqdm(
                gdf.iterrows(), total=len(gdf), desc="      Crops", leave=False
            ):
                raw_type = str(row.get(roof_col, "other") or "other").lower().strip()
                label = roof_type_map.get(raw_type)
                if label is None:
                    for key, val in roof_type_map.items():
                        if key in raw_type or raw_type in key:
                            label = val
                            break
                label = label or "Other"

                label_dir = Path(out_dir) / label
                label_dir.mkdir(parents=True, exist_ok=True)

                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

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
                        if crop.dtype == np.uint16:
                            crop = (crop / 256).astype(np.uint8)
                        else:
                            crop = crop.astype(np.uint8)

                    # Better interpolation for downscaling/upscaling
                    crop = cv2.resize(
                        crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA
                    )
                    out_p = label_dir / f"{stem}_{idx:06d}.png"
                    cv2.imwrite(str(out_p), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    saved += 1
                except Exception:
                    pass

    except Exception as e:
        print(f"      [SKIP crops] Cannot open raster: {e}")

    return saved


def _extract_infra_streaming(
    raster_path: Path,
    utility_shps: List[Path],
    infra_col: str,
    infra_type_map: dict,
    class_names: List[str],
    out_img_dir: str,
    out_label_dir: str,
    tile_size: int,
    buffer_px: int = 20,
    class_buffer_px: Optional[Dict[str, int]] = None,
    neg_tile_ratio: float = 0.0,
) -> int:
    """
    Generate YOLO labels for infrastructure by reading only the tiles that
    contain annotated features — never loads the full raster.

    Improvements over original:
      • Class-specific bounding box sizes (transformer=60px, tank=50px, well=25px)
      • Tiles centered on objects instead of grid-snapped (no edge cropping)
      • Negative tile sampling to reduce false positive rate
    """
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    cls_to_id = {c.lower(): i for i, c in enumerate(class_names)}
    id_to_name = {i: c.lower() for i, c in enumerate(class_names)}
    total = 0

    try:
        with rasterio.open(str(raster_path)) as src:
            crs = src.crs
            H, W = src.height, src.width
    except Exception as e:
        print(f"      [SKIP infra] {e}")
        return 0

    from collections import defaultdict

    # Collect all infra objects with their pixel coordinates
    infra_objects: List[Tuple[int, int, int]] = []  # (class_id, px_col, px_row)

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

        for _, row in gdf.iterrows():
            raw = str(row.get(infra_col, "") or "").lower().strip()
            cls_name = infra_type_map.get(raw)
            if cls_name is None:
                for k, v in infra_type_map.items():
                    if k in raw:
                        cls_name = v
                        break
            if cls_name is None:
                continue

            cid = cls_to_id.get(cls_name, -1)
            if cid < 0:
                continue

            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            try:
                pt = geom.centroid
                pr, pc = src.index(pt.x, pt.y)
                if pr < 0 or pr >= H or pc < 0 or pc >= W:
                    continue
                infra_objects.append((cid, int(pc), int(pr)))
                total += 1
            except Exception:
                continue

    if not infra_objects:
        print("      No infrastructure features found")
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
    with rasterio.open(str(raster_path)) as src:
        for (tr, tc), pts in tqdm(
            tile_contents.items(), desc="      Infra+", leave=False
        ):
            r2, c2 = min(tr + tile_size, H), min(tc + tile_size, W)
            ph, pw = r2 - tr, c2 - tc
            if ph <= 0 or pw <= 0:
                continue

            try:
                win = rasterio.windows.Window(tc, tr, pw, ph)  # type: ignore
                n_b = src.count
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

            name = f"infra_{tr:06d}_{tc:06d}"
            cv2.imwrite(
                str(Path(out_img_dir) / f"{name}.png"),
                cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
            )
            lines = []
            for cid, pc, pr in pts:
                # Use class-specific bounding box size
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
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (Path(out_label_dir) / f"{name}.txt").write_text("\n".join(lines))

        # ── Negative tile sampling ────────────────────────────────────────
        # Sample random tiles that contain NO infrastructure objects.
        # These teach YOLO what buildings/roads look like WITHOUT infra.
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

                # Skip if too close to any positive tile
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
                    n_b = src.count
                    if n_b >= 3:
                        tile = src.read([1, 2, 3], window=win).transpose(1, 2, 0)
                    else:
                        band = src.read(1, window=win)
                        tile = np.stack([band] * 3, axis=-1)
                    if tile.dtype != np.uint8:
                        tile = _to_uint8(tile)
                except Exception:
                    continue

                # Skip blank tiles (all-zero = no-data area)
                if tile.max() < 10:
                    continue

                name = f"infra_neg_{rand_r:06d}_{rand_c:06d}"
                cv2.imwrite(
                    str(Path(out_img_dir) / f"{name}.png"),
                    cv2.cvtColor(tile, cv2.COLOR_RGB2BGR),
                )
                # Empty label file = no objects (negative tile)
                (Path(out_label_dir) / f"{name}.txt").write_text("")
                neg_count += 1

            if neg_count > 0:
                print(f"      Negative tiles: {neg_count} (no-infra background)")

    print(
        f"      Infrastructure objects: {total}  |  Positive tiles: {len(tile_contents)}"
    )
    return total


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.ndim == 3:
        for i in range(arr.shape[2]):
            ch = arr[:, :, i].astype(np.float32)
            mn, mx = ch.min(), ch.max()
            out[:, :, i] = 0 if mx == mn else (ch - mn) / (mx - mn) * 255
    else:
        mn, mx = float(arr.min()), float(arr.max())
        out = (
            np.zeros_like(arr, dtype=np.float32)
            if mx == mn
            else (arr.astype(np.float32) - mn) / (mx - mn) * 255
        )
    return out.astype(np.uint8)
