"""
config.py  —  RTX A4000 16 GB  |  i9-13900  |  32 GB RAM
═══════════════════════════════════════════════════════════
Hardcoded to the actual SVAMITVA dataset structure:

  dataset/
    cg/   ← 5 TIF orthos + 1 ECW + shared SHPs
    pb/   ← 5 TIF orthos + 2 ECW + shared SHPs

SHP layers (same names in both cg/ and pb/):
  Built_Up_Area_type / Built_Up_Area_typ  → buildings  (col: type)
  Road                                    → roads       (col: road_type)
  Road_Centre_Line                        → roads       (col: road_type)
  Water_Body                              → waterbody   (col: water_type)
  Water_Body_Line                         → waterbody   (col: water_type)
  Waterbody_Point                         → waterbody   (col: water_type)
  Utility                                 → infra points(col: utility_type)
  Utility_Poly                            → infra poly  (col: utility_type)
  Bridge / Railway                        → road class  (infrastructure)
"""

import os
from pathlib import Path

import torch

# ─── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_ROOT = ROOT / "dataset"
PATCH_DIR = DATA_ROOT / "patches"
MASK_DIR = DATA_ROOT / "patch_masks"
CROP_DIR = DATA_ROOT / "building_crops"
YOLO_DIR = DATA_ROOT / "yolo_infra"
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "outputs" / "vectorized"
TRAIN_MASKS = DATA_ROOT / "masks"

for d in [
    PATCH_DIR,
    MASK_DIR,
    CROP_DIR,
    YOLO_DIR,
    CKPT_DIR,
    LOG_DIR,
    OUT_DIR,
    TRAIN_MASKS,
]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Hardware ────────────────────────────────────────────────────────────────
DEVICE = "cuda"
# Prevent CUDA memory fragmentation (recommended when OOM errors occur)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
AMP_DTYPE = torch.bfloat16
# torch.compile requires Triton which is not available on Windows.
# Disable it here — all other optimisations (TF32, bf16, cudnn.benchmark) still apply.
COMPILE_ENABLED = False
COMPILE_MODE = "reduce-overhead"
NUM_WORKERS = 10
PIN_MEMORY = True
PREFETCH_FACTOR = 3
PERSISTENT_WORKERS = True

# ─── SVAMITVA SHP → Class mapping ────────────────────────────────────────────
# Each SHP file uses a specific attribute column for classification.
# Values below are the actual strings found in SVAMITVA DBF files.

# Segmentation classes: 0=Background, 1=Building, 2=Road, 3=Waterbody
SHP_LAYER_ROLES = {
    # filename stem (lowercase, stripped)  → (seg_class_id, attribute_col)
    "built_up_area_type": (1, "type"),
    "built_up_area_typ": (1, "type"),  # PB truncated name
    "road": (2, "road_type"),
    "road_centre_line": (2, "road_type"),
    "water_body": (3, "water_type"),
    "water_body_line": (3, "water_type"),
    "waterbody_point": (3, "water_type"),
    "bridge": (2, "bridge_type"),  # treat as road class
    "railway": (2, "railway_type"),
    "utility": (0, "utility_type"),  # handled separately for Stage2B
    "utility_poly": (0, "utility_type"),
    "utility_poly_": (0, "utility_type"),  # PB trailing underscore
}

# Rooftop material values in Built_Up_Area_type.dbf → Stage2A class id
# SVAMITVA uses compound names like "Pucca_RCC", "Pucca_Tiled" etc.
ROOF_TYPE_MAP = {
    # Raw value (lowercase)  → Stage2A class name
    "pucca_rcc": "RCC",
    "rcc": "RCC",
    "pucca_rcc_slab": "RCC",
    "rcc_slab": "RCC",
    "pucca_tiled": "Tiled",
    "tiled": "Tiled",
    "mangalore_tile": "Tiled",
    "pucca_tin": "Tin",
    "tin": "Tin",
    "galvanized": "Tin",
    "pucca_asbestos": "Tin",  # treat asbestos as Tin category
    "asbestos": "Tin",
    "semi_pucca": "Other",
    "kuccha": "Other",
    "other": "Other",
    "others": "Other",
    "1": "RCC",
    "2": "Tiled",
    "3": "Tin",
    "4": "Other",
}

# Infrastructure values in Utility.dbf → Stage2B class id
INFRA_TYPE_MAP = {
    # Raw value (lowercase)  → class name
    "electric_transformer": "transformer",
    "transformer": "transformer",
    "electrical_transformer": "transformer",
    "overhead_water_tank": "overhead_tank",
    "water_tank": "overhead_tank",
    "overhead_tank": "overhead_tank",
    "hand_pump": "well",
    "well": "well",
    "tube_well": "well",
    "1": "transformer",
    "2": "well",
    "3": "overhead_tank",
    "11": "transformer",
    "14": "well",
}

# ─── Stage 1: Semantic Segmentation ─────────────────────────────────────────
STAGE1 = dict(
    num_classes=4,
    class_names=["background", "building", "road", "waterbody"],
    class_colors=[(0, 0, 0), (255, 0, 0), (128, 128, 128), (0, 0, 255)],
    # The SHP roles and class ids are driven by SHP_LAYER_ROLES above.
    # These two fields are kept for legacy compatibility:
    shp_class_col="type",
    shp_class_map={"building": 1, "road": 2, "waterbody": 3},
    # Model
    # UNet with mit_b4 (MixTransformer B4) gives excellent segmentation quality
    # and stays under the 15GB VRAM limit on the A4000 (batch 8, grad_accum 4).
    arch="Unet",
    encoder="mit_b4",
    encoder_weights="imagenet",
    in_channels=3,
    # Training
    # patch 512, batch 8, grad_accum 4 → effective batch 32
    # safely fits in 15 GB with margin for optimizer states + SCSE attention
    patch_size=512,
    overlap=256,
    batch_size=8,
    grad_accum=4,  # effective batch = 32
    lr=1e-4,
    encoder_lr_mult=0.1,
    weight_decay=1e-4,
    epochs=150,
    warmup_epochs=5,
    scheduler="cosine",
    # Loss
    dice_weight=0.5,
    bce_weight=0.25,
    focal_weight=0.25,
    focal_gamma=2.0,
    class_weights=[0.05, 1.6, 3.0, 2.0],
    # Regularisation
    use_swa=True,
    swa_lr=2e-5,
    swa_start_frac=0.75,
    use_ema=True,
    ema_decay=0.9998,
    cutmix_alpha=1.0,
    # Val
    val_fraction=0.15,
    seed=42,
    # Post-processing
    min_building_area_px=80,
    min_road_width_px=3,
    crf_inference=True,
    crf_iter=12,
)

# ─── Stage 2A: Rooftop Classification ───────────────────────────────────────
STAGE2A = dict(
    num_classes=4,
    class_names=["RCC", "Tiled", "Tin", "Other"],
    # Column in Built_Up_Area_type.dbf representing roof material
    shp_roof_col="Roof_type",
    # Map raw DBF values → class names (use ROOF_TYPE_MAP above)
    roof_type_map=ROOF_TYPE_MAP,
    arch="convnext_large",
    pretrained=True,
    crop_size=160,
    min_crop_px=24,
    batch_size=80,
    lr=5e-5,
    epochs=60,
    label_smoothing=0.1,
    mixup_alpha=0.4,
    cutmix_alpha=1.0,
    weight_decay=1e-1,
    grad_accum=1,
    tta_steps=16,
)

# ─── Stage 2B: Infrastructure Detection ─────────────────────────────────────
STAGE2B = dict(
    class_names=["transformer", "overhead_tank", "well"],
    num_classes=3,
    # Column in Utility.dbf / Utility_Poly.dbf
    shp_infra_col="Utility_Ty",
    infra_type_map=INFRA_TYPE_MAP,
    model_variant="yolov8x",
    img_size=1280,
    cache="ram",
    batch_size=4,
    epochs=150,
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=5,
    patience=20,
    cos_lr=True,
    mosaic=1.0,
    close_mosaic=20,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mixup=0.15,
    copy_paste=0.15,
    conf_thresh=0.25,
    iou_thresh=0.30,
    max_det=1000,
    overlap=256,
    # Per-class bounding box radius (pixels) — replaces fixed buffer_px=20
    # Sized to match actual object footprints in SVAMITVA drone imagery
    class_buffer_px={"transformer": 60, "overhead_tank": 50, "well": 25},
    # Fraction of negative tiles (no infra) to add for reducing false positives
    neg_tile_ratio=0.3,
    # Soft-NMS sigma for Gaussian confidence decay on overlapping detections
    soft_nms_sigma=0.5,
)
