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

for _d in [
    PATCH_DIR,
    MASK_DIR,
    CROP_DIR,
    YOLO_DIR,
    CKPT_DIR,
    LOG_DIR,
    OUT_DIR,
    TRAIN_MASKS,
]:
    _d.mkdir(parents=True, exist_ok=True)
del _d

# ─── Hardware — multi-backend (NVIDIA CUDA / AMD ROCm / Apple MPS / CPU) ─────
if torch.cuda.is_available():
    # Covers both NVIDIA (CUDA) and AMD (ROCm — reports as cuda in PyTorch).
    DEVICE = torch.device("cuda")
    # A4000 (16 GB) fragmentation control. expandable_segments=True is the
    # PyTorch 2.1+ Ampere-friendly allocator that grows segments on demand
    # rather than carving fixed slabs — it virtually eliminates the long-run
    # fragmentation that hurts ConvNeXt-L + bf16 + TTA mega-batches. We keep
    # max_split_size_mb as a defensive cap for legacy paths. setdefault is
    # critical so a user-set value via env wins.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:256",
    )
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
MAX_STEPS_PER_EPOCH = 2000  # cap train steps/epoch; raised from 1000 for 30GB+ datasets
# Cap validation batches per epoch. Stage 1 val time scales linearly with the
# size of the val split, which on a 30 GB dataset dominates wall-clock. With
# val_loader shuffle=False the same 300 batches are sampled every epoch, which
# is exactly what we want for a stable best-checkpoint signal across runs.
MAX_VAL_STEPS = 300
# Run the (more expensive) batched TTA at validation every Nth epoch.
# Even-epoch TTA gives the best-model picker a high-fidelity signal at a
# fraction of the wall-clock cost.
VAL_TTA_EVERY = 2
# DenseCRF is CPU-heavy and tile-parallel. Four workers is a good default on
# 32 GB RAM: it reduces wall-clock without multiplying tile copies too much.
CRF_WORKERS = 4

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
    "pucca rcc": "RCC",
    "rcc": "RCC",
    "pucca_rcc_slab": "RCC",
    "rcc_slab": "RCC",
    "concrete": "RCC",
    "concrete_roof": "RCC",
    "pucca_tiled": "Tiled",
    "pucca tiled": "Tiled",
    "tiled": "Tiled",
    "mangalore_tile": "Tiled",
    "mangalore tile": "Tiled",
    "tile": "Tiled",
    "pucca_tin": "Tin",
    "pucca tin": "Tin",
    "tin": "Tin",
    "tin_roof": "Tin",
    "metal_roof": "Tin",
    "sheet_roof": "Tin",
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
    "electric transformer": "transformer",
    "transformer": "transformer",
    "transformers": "transformer",
    "electrical_transformer": "transformer",
    "distribution_transformer": "transformer",
    "dt": "transformer",
    "overhead_water_tank": "overhead_tank",
    "overhead water tank": "overhead_tank",
    "water_tank": "overhead_tank",
    "water tank": "overhead_tank",
    "overhead_tank": "overhead_tank",
    "oht": "overhead_tank",
    "ohsr": "overhead_tank",
    "hand_pump": "well",
    "hand pump": "well",
    "well": "well",
    "wells": "well",
    "tube_well": "well",
    "tube well": "well",
    "tubewell": "well",
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
    # MAnet + MiT-B4: best balance for the 16 GB A4000.
    # - MAnet decoder (Position-wise Attention + Multi-scale Feature Aggregation)
    #   sharpens irregular boundaries — rural building outlines, lake edges,
    #   variable-width roads — which a plain Unet decoder smears.
    # - MiT-B4 transformer encoder (~62M) keeps strong SegFormer-family features
    #   for rooftop / road / water texture while running ~20% faster than
    #   MiT-B5 and fitting batch_size=8 comfortably.
    # Note: smp's UnetPlusPlus *rejects* all MiT encoders (hardcoded check in
    # decoders/unetplusplus/model.py). If you ever want true UnetPlusPlus,
    # switch encoder to 'efficientnet-b4' (~21M params, even faster).
    # Note: MiT encoders in smp do NOT expose set_grad_checkpointing — the
    # training script's try/except falls through silently. With MAnet+MiT-B4
    # at 512px in bf16 the activation peak is ~10-11 GB at batch=8, well
    # within budget without checkpointing.
    arch='MAnet',
    encoder='mit_b4',
    encoder_weights='imagenet',
    in_channels=3,
    decoder_attention_type='scse',
    patch_size=512,
    patch_sizes=(512,),
    overlap=128,
    # batch_size=4 with grad_accum=8 → effective batch 32 (same as bs=8/accum=4)
    # but ~half the peak activation memory. Safer with MAnet+MiT (no smp
    # grad-checkpointing on MiT encoders) and leaves headroom for SAM later.
    batch_size=4,
    grad_accum=8,
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
    # touching_weight=0.0: disabled — this loss penalises building predictions at
    # ALL building edges (including isolated ones), conflicting with Dice + CE loss
    # and suppressing mIoU. Instance separation is handled more cleanly by the
    # watershed postprocessing in utils/postprocess.py (separate_touching_buildings).
    touching_weight=0.0,
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
    # 10 iterations: cleaner segment boundaries than 5 (Krähenbühl & Koltun
    # converge by ~10). Inference-time only, no training cost.
    crf_iter=10,
    neg_tile_ratio=0.15,
    min_fg_ratio=0.01,    # minimum foreground fraction to keep a patch
)
# --- Stage 2A: Rooftop Classification ---
STAGE2A = dict(
    num_classes=4,
    class_names=['RCC', 'Tiled', 'Tin', 'Other'],
    shp_roof_col='Roof_type',
    shp_roof_cols=('Roof_type', 'roof_type', 'type', 'Type', 'bldg_type', 'building_type'),
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
    # Validation TTA: 8 folds × 3 scales now go through a single batched
    # forward per scale (see RooftopClassifier.predict), so doubling tta_steps
    # only ~doubles val time — gives more representative best-epoch selection.
    tta_steps=8,
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
    shp_infra_cols=('Utility_Ty', 'utility_type', 'Utility_Type', 'type', 'Type', 'name', 'Name'),
    infra_type_map=INFRA_TYPE_MAP,
    # YOLOv11l chosen over YOLOv9e:
    #   • ~2× faster at parity accuracy on small aerial objects (~25M vs 58M params)
    #   • C2PSA attention gives a small but consistent recall bump on tiny objects
    #     (wells ~15px, transformers ~30px)
    #   • Active Ultralytics maintenance branch; yolov9e-obb isn't a first-party
    #     release and silently fell back to AABB anyway
    #   • Frees ~2.5 GB VRAM at imgsz=1280 → batch can double from 2 to 4
    model_variant='yolo11l',
    # OBB disabled: transformer/overhead_tank/well are rotationally symmetric
    # (circles) or near-square from a top-down drone view, so the angle target
    # is undefined or ambiguous. AABB outputs Point centroids in
    # detections_to_shapefile, which is what surveyors actually want for an
    # infrastructure inventory layer. obb_model_variant is kept as a no-op
    # fallback path in case OBB is ever re-enabled.
    use_obb=False,
    obb_model_variant='yolo11l-obb',
    img_size=1280,
    cache='ram',
    # Bumped from 2 → 4: YOLOv11l uses ~4.5 GB at 1280px (vs 7.2 GB for yolov9e),
    # leaving room for batch=4 on the A4000. Bigger gradient signal per step.
    batch_size=4,
    # Use the same parallel-loader budget as Stage 1/2A. workers=0 made YOLO
    # decode + Mosaic happen serially on the main thread, leaving the GPU idle
    # between batches.
    workers=NUM_WORKERS,
    # Light head dropout (0.1) and multi-scale training: documented YOLO
    # generalisation boosts. multi_scale resizes each batch within
    # ±50 % of img_size — combined with cache='ram' this is the most reliable
    # mAP lift on a fixed dataset. Watch VRAM at high img_size.
    dropout=0.1,
    multi_scale=True,
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
    context_classes=('building', 'road', 'waterbody'),
    context_buffer_px=128,
    neg_tile_ratio=0.3,
    # Lower sigma (sharper Gaussian decay) helps closely-spaced small objects
    # retain their score rather than being heavily attenuated by neighbours.
    soft_nms_sigma=0.40,
    agnostic_nms=True,
    use_sahi=True,
    # Smaller SAHI slices + higher overlap → better small-object recall on
    # transformers (~30-60 px) and wells (~15-30 px) at the cost of a few extra
    # YOLO forward passes per tile. The mAP win is consistently >2 % on small
    # infrastructure in published SAHI benchmarks.
    sahi_slice_size=512,
    sahi_overlap_ratio=0.45,
    class_conf_thresh={
        'transformer': 0.20,
        'overhead_tank': 0.12,
        'well': 0.10,
    },
)
