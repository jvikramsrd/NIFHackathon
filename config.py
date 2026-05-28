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

v2 — Maximum Accuracy Build:
  • Stage 1: mit_b5 encoder (82M), 4-scale MS training, Lovász weight boosted
  • Stage 2A: EfficientNetV2-L option, ensemble support, calibrated thresholds
  • Stage 2B: Improved per-class confidence floor, SAHI GREEDYNMM
  • Post-processing: bi_xy_std corrected to 20px (1m at 5cm GSD)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
        "max_split_size_mb:256",
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
# torch.compile uses the OpenAI Triton compiler internally to JIT-compile
# model graphs. It also enables CUDA Graphs via mode="reduce-overhead",
# which captures and replays GPU operations to eliminate CPU launch overhead.
# Available on Linux / WSL; not supported on native Windows (no Triton).
# WSL (Windows Subsystem for Linux) is detected as "Linux" by platform.system()
# so it will use Triton and CUDA Graphs normally.
COMPILE_ENABLED = True
COMPILE_MODE = "reduce-overhead"  # switch to "max-autotune" for final inference (+5% speed)
# Explicit CUDA Graphs warmup: runs a dummy batch through the compiled model
# to trigger compilation + graph capture before the main training loop.
# Only meaningful when COMPILE_ENABLED=True and COMPILE_MODE="reduce-overhead".
CUDA_GRAPHS_ENABLED = True
FAST_TTA = False  # True = 2-scale×4-fold (8 passes); False = 3-scale×8-fold (24 passes, more accurate)
# Enable model ensemble at inference: combines Stage2A predictions from multiple trained checkpoints.
# Set to True when multiple stage2a_*.pth checkpoints are available in CKPT_DIR.
ENSEMBLE_INFERENCE = False
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
    # MAnet + MiT-B5: maximum accuracy build for the 16 GB A4000.
    # - MAnet decoder (Position-wise Attention + Multi-scale Feature Aggregation)
    #   sharpens irregular boundaries — rural building outlines, lake edges,
    #   variable-width roads — which a plain Unet decoder smears.
    # - MiT-B5 transformer encoder (~82M): highest capacity in the MiT family.
    #   +1-2% mIoU vs MiT-B4 on dense urban/rural datasets (SegFormer paper table 4).
    #   Peak activation at 512px + bf16 + batch=2 = ~11 GB — fits A4000 with grad_accum=16.
    #   If VRAM is tight, revert to encoder='mit_b4' with batch_size=4, grad_accum=8.
    # Note: smp's UnetPlusPlus *rejects* all MiT encoders (hardcoded check in
    # decoders/unetplusplus/model.py). If you ever want true UnetPlusPlus,
    # switch encoder to 'efficientnet-b4' (~21M params, even faster).
    # Note: MiT encoders in smp do NOT expose set_grad_checkpointing — the
    # training script's try/except falls through silently. With MAnet+MiT-B5
    # at 512px in bf16 the activation peak is ~12-13 GB at batch=2+accum=16,
    # effective batch 32 — tight but viable.
    arch='MAnet',
    encoder='mit_b5',
    encoder_weights='imagenet',
    in_channels=3,
    decoder_attention_type='scse',
    patch_size=512,
    patch_sizes=(384, 512, 640),  # multi-resolution patch training for scale invariance
    overlap=128,
    # batch_size=2 with grad_accum=16 → effective batch 32, same gradient signal as
    # b4+accum8, but ~30% lower peak VRAM for the larger MiT-B5 encoder.
    batch_size=2,
    grad_accum=16,
    lr=1.5e-4,  # slightly lower LR for larger encoder (avoids early divergence)
    encoder_lr_mult=0.1,
    weight_decay=1e-4,
    epochs=90,  # 10 extra epochs for the larger encoder to converge
    warmup_epochs=5,  # longer warmup for stable training of 82M encoder
    scheduler='cosine',
    # Loss weights — SOTA recipe for IoU-maximising segmentation:
    # Lovász directly optimises the IoU metric (not a proxy), so we give it the
    # highest weight. Boundary loss sharpens building edges. Focal handles class imbalance.
    focal_weight=0.35,
    boundary_weight=0.25,  # raised: boundary precision is critical for polygon output
    lovasz_weight=0.40,   # raised: directly optimises mIoU — our primary metric
    # touching_weight=0.0: disabled — this loss penalises building predictions at
    # ALL building edges (including isolated ones), conflicting with Dice + CE loss
    # and suppressing mIoU. Instance separation is handled more cleanly by the
    # watershed postprocessing in utils/postprocess.py (separate_touching_buildings).
    touching_weight=0.0,
    focal_gamma=3.0,
    # class_weights: higher weight on road (5×) and waterbody (3×) to compensate
    # for their rarer occurrence in SVAMITVA rural imagery. Background kept at 0.4.
    class_weights=[0.40, 2.00, 5.00, 3.00],  # waterbody weight raised from 2.5 → 3.0
    label_smoothing=0.06,  # slightly lower — MiT-B5 is already well-regularized
    use_swa=True,
    swa_lr=1.5e-5,
    swa_start_frac=0.75,
    use_ema=True,
    ema_decay=0.9999,  # tighter EMA decay for larger model (slower shadow update = smoother)
    drop_path_rate=0.3,  # higher stochastic depth for 82M params encoder
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
    # CRF bilateral parameters (corrected for 5cm GSD drone imagery):
    # bi_xy_std=20: spatial kernel = 20px = 1m at 5cm resolution.
    # The old value of 80px = 4m, far too broad for fine-grained boundaries.
    crf_bi_xy_std=20.0,
    crf_bi_rgb_std=13.0,
    crf_bi_w=10.0,
    crf_pos_xy_std=3.0,
    crf_pos_w=3.0,
    neg_tile_ratio=0.15,
    min_fg_ratio=0.01,    # minimum foreground fraction to keep a patch
    inference_batch_size=12,  # reduced from 16: MiT-B5 is larger, keep VRAM headroom
)
# --- Stage 2A: Rooftop Classification ---
STAGE2A = dict(
    num_classes=4,
    class_names=['RCC', 'Tiled', 'Tin', 'Other'],
    shp_roof_col='Roof_type',
    shp_roof_cols=('Roof_type', 'roof_type', 'type', 'Type', 'bldg_type', 'building_type'),
    roof_type_map=ROOF_TYPE_MAP,
    # Architecture: ConvNeXt V2 Large — state-of-the-art for fine-grained
    # texture classification. ConvNeXt V2 uses fully-convolutional masked
    # autoencoder pretraining (FCMAE) which transfers exceptionally well to
    # aerial rooftop material classification (RCC vs Tiled vs Tin).
    # The backbone outputs 1536-dim features with Global Response Normalization
    # (GRN) for better feature competition.
    arch='convnextv2_large',
    pretrained=True,
    crop_size=256,  # raised from 224: EfficientNetV2-L was pre-trained at 256-480px
    min_crop_px=40,
    batch_size=24,  # slightly reduced for EfficientNetV2-L memory footprint
    lr=3e-5,  # lower LR: EfficientNetV2-L converges best with conservative LR
    epochs=100,   # extra epochs for new architecture to fully converge
    label_smoothing=0.05,
    mixup_alpha=0.4,
    cutmix_alpha=1.0,
    weight_decay=1e-4,
    grad_accum=2,  # effective batch 48: better gradient signal for 4-class problem
    # Validation TTA: 8 folds × 3 scales → single batched forward per scale.
    tta_steps=8,
    # Per-class confidence thresholds (calibrated for SVAMITVA class frequencies):
    # Tin and Other are rare → lower threshold to improve recall on minority classes.
    stage2a_conf_thresh={'RCC': 0.45, 'Tiled': 0.50, 'Tin': 0.42, 'Other': 0.35},
    use_arcface=True,
    arcface_s=32.0,  # slightly higher scale: tighter margin for 4-class problem
    arcface_m=0.55,
    use_sam=True,
    sam_rho=0.05,
    sam_adaptive=True,
    use_randaugment=True,
    randaugment_n=2,
    randaugment_m=9,  # stronger augmentation magnitude: rooftop textures benefit
    drop_path_rate=0.4,
    use_ema=True,
    ema_decay=0.9995,
    # Ensemble inference: list of checkpoint names to load for multi-model ensemble.
    # Add 'stage2a_best_v2.pth' etc. after training second model with different seed.
    # Leave empty list to use single model.
    ensemble_ckpts=[],
    # Class-balanced sampling: oversample rare classes during training.
    # Addresses severe class imbalance in SVAMITVA (mostly RCC, few Tin/Other).
    use_balanced_sampler=True,
    # Temperature scaling calibration (run utils/calibration.py after training).
    temperature=1.0,  # set by calibrate_and_save() after training
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
    # OBB enabled: YOLO11-OBB provides oriented bounding boxes for small
    # infrastructure objects (transformers, overhead tanks, wells). While many
    # of these are rotationally symmetric, OBB improves localization accuracy
    # at the edges and enables better non-max suppression for closely-spaced
    # objects. Falls back to AABB centroids for symmetric cases.
    use_obb=True,
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
    conf_thresh=0.08,  # lower global floor: per-class thresholds do the real filtering
    iou_thresh=0.60,
    max_det=1500,  # raised: dense villages can have many transformers/wells
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
    # SAHI: smaller slices + higher overlap → +2-4% mAP on small objects.
    # GREEDYNMM post-process: more accurate than NMM for overlapping small objects
    # (published +1.5% mAP@0.5 in SAHI benchmark on VisDrone dataset).
    sahi_slice_size=480,   # smaller than before: better for wells (~15-30px)
    sahi_overlap_ratio=0.50,  # higher overlap: reduces missed objects at tile seams
    sahi_postprocess_type='GREEDYNMM',  # upgraded from NMM
    # Per-class confidence thresholds (lowered for better recall on rare objects):
    # Transformers are more common and distinctive → higher threshold OK.
    # Wells are small and subtle → low threshold, rely on context gating.
    class_conf_thresh={
        'transformer': 0.18,
        'overhead_tank': 0.10,
        'well': 0.08,
    },
)

# ── Read-only config views ───────────────────────────────────────────────────
# Wrap stage configs in MappingProxyType to prevent accidental runtime mutation.
# All reads (dict.get, [], .keys, .values, .items, len, in, iter) work normally;
# writes ([]=, .update, .pop, .clear) raise TypeError.
from types import MappingProxyType
STAGE1 = MappingProxyType(STAGE1)    # type: ignore[assignment]
STAGE2A = MappingProxyType(STAGE2A)  # type: ignore[assignment]
STAGE2B = MappingProxyType(STAGE2B)  # type: ignore[assignment]
