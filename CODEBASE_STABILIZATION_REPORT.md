# CODEBASE_STABILIZATION_REPORT.md

## 1. Critical Execution Blockers

### Geospatial Integrity & CRS Mapping
- **Silent Failure on Projection Mismatch:** In `data/preprocessing.py` (lines ~650-653), the CRS verification block checks `gdf.crs != crs` and attempts `gdf.to_crs(crs)`. However, it catches all `Exception`s and executes a `pass`, silently keeping the original coordinates. If vector data cannot be projected, this will cause the raster and vector masks to be entirely misaligned, creating corrupted training masks without terminating the pipeline or alerting the user.
- **Missing CRS Check in Context Extraction:** In `inference/pipeline.py`, the `_gather_context_polygons` function loads shapefiles directly via `gpd.read_file(str(shp_path))` but lacks any `to_crs()` verification to ensure the context polygons mathematically align with the base raster's affine transform.

### Data Flow & Dimension Consistency
- **Unbounded Rescale in Preprocessing:** While `dataset.py` gracefully handles boundary crops via `cv2.copyMakeBorder` to hit the exact patch dimension, if `mask.shape[:2] != img.shape[:2]`, it forcefully uses `cv2.resize`. This resize does not check the scale factor—an extremely mismatched mask could be warped, shifting boundary pixels and corrupting edge-aware losses like the Hausdorff-ER loss in `stage1_segmentation.py`.
- **NaN Propagation via TTA Masking:** In `models/stage1_segmentation.py`, the `tta_predict` function catches any `Exception` during batched inference and assigns a uniform distribution (`1.0 / max(num_classes, 1)`). If a completely invalid tensor (e.g., filled with NaNs from a broken sensor) induces a crash, the pipeline will swallow it and generate false uniform predictions rather than flagging the corrupted input.

## 2. Efficiency & Resource Bottlenecks

### Single-Threaded CPU Bottleneck
- **Rooftop Classification Chokepoint:** In `inference/pipeline.py` (`_classify_rooftops`), the algorithm iterates through building polygons via a sequential Python loop (`for k in tqdm(range(len(geoms_arr))):`). For high-density urban grids (often exceeding 10,000+ building footprints), extracting bounds, computing the inverse affine transform per polygon, and running `cv2.resize` natively in Python will starve the GPU processor. The bounding box computations should be vectorized natively via `geopandas` or `shapely` block operations.

### Resource Allocation & Memory
- **Aggressive Dataloader Fallback:** In `train_stage1.py` and `train_stage2.py`, if the DDP multiprocess dataloader initialization triggers an exception, the pipeline catches it and forcefully falls back to `num_workers=0`. This will push all image reading and heavy Albumentations augmentations onto the main thread, resulting in catastrophic GPU starvation (bottlenecking the A4000 GPU).
- **VRAM Defragmentation:** While `train_stage1.py` and `inference/pipeline.py` correctly invoke `torch.cuda.empty_cache()` between major scale-shifts, the caching layer in `_classify_rooftops` accumulates raw RGB crops in a python `list` until it hits `batch_size`. This causes high heap-memory churn due to garbage collection before being stacked into the tensor batch.

### Exception Handling & Edge Cases
- **Zero-Division Handling:** The pipeline successfully mitigates zero-division in accuracy calculations via `.item() / max(total, 1)` in `train_stage2.py` and `np.maximum(count_map, 1e-6)` in `utils/inference.py`. Standardizing to PyTorch's `eps` parameter for division safety is recommended.

## 3. Pipeline Dependency Map

The overall structure of the analytical pipeline ensures predictable data propagation:

**A. Data Preparation & Normalization**
`data/preprocessing.py` 
 ↳ Extracts `.tif` and `.shp`
 ↳ Generates spatial masks (`_mask.tif`) and tiled `.png` patches
 ↳ Triggers `data/dataset.py` for PyTorch dataset mapping

**B. Stage 1: Core Semantic Segmentation**
`train/train_stage1.py`
 ↳ Imports model config from `models/stage1_segmentation.py` (MAnet + MiT-B5 + TriLoss)
 ↳ Optimized execution via `utils/hardware.py` and telemetry via `utils/logger.py`
 ↳ Validates checkpoints in `utils/checkpointing.py`

**C. Stage 2: Hierarchical Refinement**
`train/train_stage2.py`
 ↳ *Stage 2A*: Loads crops → `models/stage2_models.py` (RooftopClassifier)
 ↳ *Stage 2B*: Loads bounding boxes → `models/stage2_models.py` (InfrastructureDetector / YOLO)

**D. Inference Engine**
`inference/pipeline.py`
 ↳ Invokes `utils/inference.py` for scale-invariant sliding window segmentation
 ↳ Routes spatial map to `utils/postprocess.py` for DenseCRF polishing and geometry extraction
 ↳ Concludes by emitting unified `.shp` outputs containing structural and infrastructural inferences.
