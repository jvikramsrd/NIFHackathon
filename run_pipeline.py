import os
import sys
import multiprocessing
import argparse
from pathlib import Path

if __name__ == "__main__":
    multiprocessing.freeze_support()

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, str(Path(__file__).parent))
from utils.core import get_logger
import config as CFG

log = get_logger(__name__)

def preprocess(data_root: str):
    from data.preprocessing import preprocess_dataset_root
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        log.error("Data root does not exist: %s", data_root_path)
        return
    summary = preprocess_dataset_root(
        data_root=data_root_path,
        output_root=CFG.DATA_ROOT,
        patch_size=int(CFG.STAGE1["patch_size"]),
        overlap=float(CFG.STAGE1["overlap"]) / int(CFG.STAGE1["patch_size"]),
        role_map=CFG.SHP_LAYER_ROLES,
        resume=True,
    )
    log.info("PREPROCESSING COMPLETE")

def train_all():
    from train import train_stage1, train_stage2a, train_stage2b
    import torch
    
    log.info("STAGE 1 - Semantic Segmentation")
    train_stage1()
    torch.cuda.empty_cache()
    
    log.info("STAGE 2A - Rooftop Classifier")
    train_stage2a()
    torch.cuda.empty_cache()
    
    log.info("STAGE 2B - Infrastructure Detector")
    train_stage2b()

def evaluate():
    log.info("Evaluation functionality can be added back if required.")

def infer(tif_path: str, out_dir: str):
    from inference.pipeline import GeoIntelPipeline
    pipe = GeoIntelPipeline(
        str(CFG.CKPT_DIR / f"stage1_best.pth"),
        str(CFG.CKPT_DIR / "stage2a_best.pth"),
        str(CFG.CKPT_DIR / f"stage2b_{CFG.STAGE2B['model_variant']}" / "weights" / "best.pt"),
    )
    pipe.run(tif_path, out_dir)

def cli_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["preprocess", "train_all", "evaluate", "infer", "all"])
    ap.add_argument("--data_root", default="./dataset")
    ap.add_argument("--tif", default=None)
    ap.add_argument("--out", default="./outputs/test")
    args = ap.parse_args()

    if args.mode == "preprocess":
        preprocess(args.data_root)
    elif args.mode == "train_all":
        train_all()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "infer":
        if not args.tif:
            ap.error("--tif is required for --mode infer")
        infer(args.tif, args.out)
    elif args.mode == "all":
        preprocess(args.data_root)
        train_all()

if __name__ == "__main__":
    cli_main()
