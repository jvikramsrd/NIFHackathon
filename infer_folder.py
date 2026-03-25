import argparse
import os
import sys
import traceback
from pathlib import Path

# Ensure the root directory is in sys.path so internal imports work correctly
sys.path.insert(0, str(Path(__file__).parent))
import config as CFG
from inference.pipeline import GeoIntelPipeline


def infer_folder(test_folder: str, out_base_dir: str):
    test_folder_path = Path(test_folder)
    out_base_path = Path(out_base_dir)

    # 1. Find all TIF files in the folder (including subdirectories)
    tifs = list(test_folder_path.rglob("*.tif")) + list(
        test_folder_path.rglob("*.tiff")
    )

    # Filter out common non-image spatial files (like ArcGIS overviews)
    tifs = [
        t
        for t in tifs
        if not t.name.lower().endswith(".pyrx") and ".lock" not in t.name.lower()
    ]

    if not tifs:
        print(f"[WARN] No .tif or .tiff files found in {test_folder}")
        return

    print(f"\nFound {len(tifs)} image(s) to process in '{test_folder}'.")

    # 2. Load the models ONCE to save massive VRAM allocation overhead
    print("Loading models into VRAM (this happens only once)...")
    try:
        pipe = GeoIntelPipeline(
            str(CFG.CKPT_DIR / "stage1_best.pth"),
            str(CFG.CKPT_DIR / "stage2a_best.pth"),
            str(CFG.CKPT_DIR / "stage2b_yolov8l" / "weights" / "best.pt"),
        )
    except Exception as e:
        print(
            f"\n[FATAL ERROR] Failed to load models! Ensure training has finished and checkpoints exist."
        )
        print(f"Error details: {e}")
        return

    # 3. Process each image sequentially
    successful = 0
    failed = 0

    for i, tif in enumerate(tifs, 1):
        print(f"\n[{i}/{len(tifs)}] {'=' * 55}")
        print(f"Processing Image : {tif.name}")
        print(f"Source Path      : {tif}")

        # Create a unique output subfolder for this specific TIF
        out_dir = out_base_path / tif.stem
        os.makedirs(out_dir, exist_ok=True)

        try:
            # Run the 3-stage pipeline on the image
            pipe.run(str(tif), str(out_dir))
            print(f"[*] Done! Results saved to: {out_dir}")
            successful += 1
        except Exception as e:
            print(f"\n[ERROR] Failed to process {tif.name}!")
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 65}")
    print("  BATCH INFERENCE COMPLETE")
    print(f"{'=' * 65}")
    print(f"  Successfully processed : {successful}")
    print(f"  Failed to process      : {failed}")
    print(f"  All outputs saved to   : {out_base_path.absolute()}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Run full GeoIntel inference on a directory of TIF images."
    )
    ap.add_argument(
        "--test_folder",
        required=True,
        help="Path to the folder containing testing TIFs (e.g., 'C:/Users/Dell/Downloads/test_images')",
    )
    ap.add_argument(
        "--out_folder",
        default="./outputs/batch_inference",
        help="Base directory to save results. Each image will get its own subfolder here.",
    )
    args = ap.parse_args()

    infer_folder(args.test_folder, args.out_folder)
