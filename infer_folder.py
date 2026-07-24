import argparse
import sys
from pathlib import Path

# Must be first so all project-relative imports below resolve correctly
sys.path.insert(0, str(Path(__file__).parent))

import tempfile

from utils.core import get_logger

log = get_logger(__name__)

import config as CFG
from inference.pipeline import GeoIntelPipeline
from utils.ecw_compat import ecw_to_tif, is_ecw


def infer_folder(test_folder: str, out_base_dir: str):
    test_folder_path = Path(test_folder)
    out_base_path = Path(out_base_dir)

    # 1. Find all raster files in the folder (including subdirectories)
    RASTER_GLOBS = ("*.tif", "*.tiff", "*.ecw", "*.TIF", "*.TIFF", "*.ECW")
    rasters = []
    for pattern in RASTER_GLOBS:
        rasters.extend(test_folder_path.rglob(pattern))
    # Deduplicate (case-insensitive globs on Windows can overlap)
    seen = set()
    tifs = []
    for r in rasters:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            tifs.append(r)

    # Filter out common non-image spatial files (like ArcGIS overviews)
    tifs = [
        t
        for t in tifs
        if not t.name.lower().endswith(".pyrx") and ".lock" not in t.name.lower()
    ]

    if not tifs:
        log.warning("No raster files (.tif, .tiff, .ecw) found in %s", test_folder)
        return

    log.info(f"\nFound {len(tifs)} image(s) to process in '{test_folder}'.")

    # 2. Load the models ONCE to save massive VRAM allocation overhead
    log.info("Loading models into VRAM (this happens only once)...")
    try:
        pipe = GeoIntelPipeline(
            str(CFG.CKPT_DIR / f"stage1_{CFG.STAGE1['encoder']}_best.pth"),
            str(CFG.CKPT_DIR / "stage2a_best.pth"),
            str(CFG.CKPT_DIR / f"stage2b_{CFG.STAGE2B['model_variant']}" / "weights" / "best.pt"),
        )
    except Exception as e:
        log.error(
            "Failed to load models — ensure training has finished and checkpoints exist.\n"
            "Error: %s", e, exc_info=True
        )
        return

    # 3. Process each image sequentially
    successful = 0
    failed = 0

    with tempfile.TemporaryDirectory(prefix="geointel_ecw_") as _tmp:
        tmp_dir = Path(_tmp)

        for i, raster in enumerate(tifs, 1):
            log.info(f"\n[{i}/{len(tifs)}] {'=' * 55}")
            log.info(f"Processing Image : {raster.name}")
            log.info(f"Source Path      : {raster}")

            # Create a unique output subfolder for this specific file
            out_dir = out_base_path / raster.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            # Convert ECW → TIF if needed (deleted after this image is done)
            process_path = raster
            converted_tif = None
            if is_ecw(raster):
                try:
                    converted_tif = ecw_to_tif(raster, tmp_dir)
                    process_path = converted_tif
                except RuntimeError as ecw_err:
                    log.error("ECW conversion failed for %s: %s", raster.name, ecw_err)
                    failed += 1
                    continue

            try:
                pipe.run(str(process_path), str(out_dir))
                log.info(f"[*] Done! Results saved to: {out_dir}")
                successful += 1
            except Exception as e:
                log.error("Failed to process %s: %s", raster.name, e, exc_info=True)
                failed += 1
            finally:
                # Clean up converted TIF immediately to free disk space
                if converted_tif and converted_tif.exists():
                    converted_tif.unlink(missing_ok=True)

    log.info(f"\n{'=' * 65}")
    log.info("  BATCH INFERENCE COMPLETE")
    log.info(f"{'=' * 65}")
    log.info(f"  Successfully processed : {successful}")
    log.info(f"  Failed to process      : {failed}")
    log.info(f"  All outputs saved to   : {out_base_path.absolute()}")
    log.info(f"{'=' * 65}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run full GeoIntel inference on a directory of raster images (.tif, .tiff, .ecw)."
    )
    ap.add_argument(
        "--test_folder",
        required=True,
        help="Path to the folder containing raster files (.tif, .tiff, .ecw) (e.g., 'C:/Users/Dell/Downloads/test_images')",
    )
    ap.add_argument(
        "--out_folder",
        default="./outputs/batch_inference",
        help="Base directory to save results. Each image will get its own subfolder here.",
    )
    args = ap.parse_args()

    infer_folder(args.test_folder, args.out_folder)
