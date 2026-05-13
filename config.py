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

# ─── Hardware — multi-backend (NVIDIA CUDA / AMD ROCm / Apple MPS / CPU) ─────
if torch.cuda.is_available():
    # Covers both NVIDIA (CUDA) and AMD (ROCm — reports as cuda in PyTorch).
    DEVICE = torch.device("cuda")
    # Reduce fragmentation on 16 GB cards; harmless on other VRAM sizes.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # bf16 is native on Ampere (NVIDIA) and CDNA2+ (AMD MI200+).
    # On older AMD cards PyTorch silently emulates it; fp16 is faster there,
    # but bf16 is kept as default because it's lossless for this workload.
    AMP_DTYPE = torch.bfloat16
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Apple Silicon (M1 / M2 / M3 / M4) — Metal Performance Shaders.
    # MPS does not support bfloat16 in all ops; use float16 instead.
    DEVICE = torch.device("mps")
    AMP_DTYPE = torch.float16
else:
    DEVICE = torch.device("cpu")
    AMP_DTYPE = torch.float32
# torch.compile requires Triton which is not available on Windows.
# Disable it here — all other optimisations (TF32, bf16, cudnn.benchmark) still apply.
COMPILE_ENABLED = False
COMPILE_MODE = "reduce-overhead"
FAST_TTA = False  # True = 2-scale×4-fold (8 passes); False = 3-scale×8-fold (24 passes, more accurate)
NUM_WORKERS = 8
PIN_MEMORY = True
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True
MAX_STEPS_PER_EPOCH = 1000  # cap steps/epoch to keep wall-time ≤ 30 min

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

# --- Stage 1: Semantic Segmentation ---
STAGE1 = dict(
    num_classes=4,
    class_names=['background', 'building', 'road', 'waterbody'],
    class_colors=[(0, 0, 0), (255, 0, 0), (128, 128, 128), (0, 0, 255)],
    shp_class_col='type',
    shp_class_map={'building': 1, 'road': 2, 'waterbody': 3},
    arch='Unet',
    encoder='mit_b5',
    encoder_weights='imagenet',
    in_channels=3,
    decoder_attention_type='scse',
    patch_size=512,
    patch_sizes=(512,),
    overlap=128,
    batch_size=6,
    grad_accum=4,
    lr=2e-4,
    encoder_lr_mult=0.1,
    weight_decay=1e-4,
    epochs=80,
    warmup_epochs=3,
    scheduler='cosine',
    use_sam=False,
    sam_rho=0.05,
    sam_adaptive=True,
    ms_training=True,
    ms_scales=(0.75, 1.0, 1.25),
    dice_weight=0.40,
    bce_weight=0.15,
    focal_weight=0.15,
    boundary_weight=0.15,
    lovasz_weight=0.15,
    touching_weight=0.10,
    focal_gamma=2.0,
    class_weights=[0.20, 2.00, 5.00, 2.50],
    label_smoothing=0.08,
    use_swa=True,
    swa_lr=2e-5,
    swa_start_frac=0.75,
    use_ema=True,
    ema_decay=0.9998,
    cutmix_alpha=1.0,
    drop_path_rate=0.2,
    val_fraction=0.15,
    seed=42,
    min_building_area_px=80,
    min_road_width_px=3,
    polygon_min_area_px={'building': 80, 'road': 120, 'waterbody': 160},
    polygon_simplify_tolerance=0.5,
    crf_inference=True,
    crf_iter=5,
    neg_tile_ratio=0.15,
    min_fg_ratio=0.01,    # minimum foreground fraction to keep a patch
)
# --- Stage 2A: Rooftop Classification ---
STAGE2A = dict(
    num_classes=4,
    class_names=['RCC', 'Tiled', 'Tin', 'Other'],
    shp_roof_col='Roof_type',
    roof_type_map=ROOF_TYPE_MAP,
    arch='convnext_large',
    pretrained=True,
    crop_size=224,
    min_crop_px=40,
    batch_size=32,
    lr=5e-5,
    epochs=80,
    label_smoothing=0.05,
    mixup_alpha=0.4,
    cutmix_alpha=1.0,
    weight_decay=1e-4,
    grad_accum=1,
    tta_steps=4,  # light TTA during training; full 24-fold used at final inference
    stage2a_conf_thresh={'RCC': 0.45, 'Tiled': 0.55, 'Tin': 0.50, 'Other': 0.40},
    use_arcface=True,
    arcface_s=30.0,
    arcface_m=0.55,
    use_sam=True,
    sam_rho=0.05,
    sam_adaptive=True,
    use_randaugment=True,
    randaugment_n=2,
    randaugment_m=7,
    drop_path_rate=0.4,
    use_ema=True,
    ema_decay=0.9995,
)
# --- Stage 2B: Infrastructure Detection ---
STAGE2B = dict(
    class_names=['transformer', 'overhead_tank', 'well'],
    num_classes=3,
    shp_infra_col='Utility_Ty',
    infra_type_map=INFRA_TYPE_MAP,
    model_variant='yolov9e',
    use_obb=True,
    obb_model_variant='yolov9e-obb',
    img_size=1280,
    cache='ram',
    batch_size=2,
    workers=0,
    epochs=120,
    lr0=1e-3,
    lrf=0.01,
    warmup_epochs=3,
    patience=20,
    cos_lr=True,
    mosaic=1.0,
    close_mosaic=20,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.3,
    degrees=15.0,
    translate=0.1,
    scale=0.6,
    fliplr=0.5,
    flipud=0.5,  # aerial images are rotation-invariant
    mixup=0.15,
    copy_paste=0.30,
    conf_thresh=0.10,
    iou_thresh=0.60,
    max_det=1000,
    overlap=512,
    class_buffer_px={'transformer': 100, 'overhead_tank': 80, 'well': 40},
    neg_tile_ratio=0.3,
    soft_nms_sigma=0.5,
    agnostic_nms=True,
    use_sahi=True,
    sahi_slice_size=640,
    sahi_overlap_ratio=0.40,
    class_conf_thresh={
        'transformer': 0.20,
        'overhead_tank': 0.12,
        'well': 0.10,
    },
)
