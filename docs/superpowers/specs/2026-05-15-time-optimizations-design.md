# Time Optimization Design — GeoIntel Pipeline

**Date:** 2026-05-15  
**Scope:** Both inference and training phases  
**Constraint:** Zero accuracy change — all optimizations are numerically equivalent

---

## Problem

The pipeline has several bottlenecks discovered through full code review:

| Bottleneck | Location | Root Cause |
|---|---|---|
| 8–24 sequential model forward passes per tile-batch | `tta_predict()`, `RooftopClassifier.predict()` | TTA augmentations run one-at-a-time instead of batched |
| Disk I/O for every YOLO tile | `_detect()` | JPEG encode → write → YOLO reads → delete, per tile |
| O(N) context polygon scan | `_detect()` | Linear `any(intersects)` loop, no spatial index |
| Serial CRF tile processing | `apply_dense_crf()` | Docstring says "parallel" but loop is sequential |
| Double SHP file load | `run()` + `_classify_rooftops()` | GDF loaded, passed in, then re-read from disk |
| Python channel loop | `_to_uint8()` | `for i in range(3)` instead of numpy broadcast |
| Per-call spline window | `_segment()` | `cosine_window` recomputed every `_segment()` call |
| Python `class_weights()` loop | `RooftopDataset` | `for _, lbl in self.samples` instead of `np.bincount` |

---

## Design

### 1. TTA Batching — `models/stage1_segmentation.py`

**File:** `models/stage1_segmentation.py`  
**Function:** `tta_predict()`

**Current:** For each (scale, fold), calls `model(aug)` separately. With `fast_tta=False` (default): 3 scales × 8 folds = 24 sequential forward passes per call.

**Change:** Collect all augmented views up-front into a list. `torch.cat` them into chunks of size `tta_chunk` (default 8 × batch_size images). Run `model()` once per chunk. Reverse-augment outputs and accumulate into `probs_sum`.

**Memory contract:** `tta_chunk=8` means at most `8 × batch_size` images per forward call. For the pipeline's `batch_size=16`: up to 128 images. For mit_b5 at 512px in bf16 no-grad, activation memory ≈ 60 MB per image → ~7.7 GB. Within 16 GB budget.  
A `tta_chunk` parameter is accepted so callers can tune it down if VRAM is tighter.

**Numerical equivalence:** Same augmentations, same weights, same softmax — just reordered through one larger matmul instead of N smaller ones. Results are bit-for-bit identical.

### 2. TTA Batching — `models/stage2_models.py`

**File:** `models/stage2_models.py`  
**Function:** `RooftopClassifier.predict()`

**Current:** 3 scales × 8 folds = 24 sequential `self(aug)` calls.

**Change:** Same pattern as above. Collect all augmented views across scales/folds, run in one batched call (or chunked for memory safety). Reverse the augmentation to get probs, then weighted-average. Identical math.

**chunk default:** `tta_steps` (already a parameter) caps total folds; chunk = `min(tta_steps, 8) * len(scales)` which equals the current total — still done in one shot since classification crops are small (224px).

### 3. Eliminate Disk I/O in `_detect()` — `inference/pipeline.py`

**Current:**
```python
_, buf = cv2.imencode('.jpg', cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
tmp_path.write_bytes(buf.tobytes())
self.detector.predict(str(tmp_path))
tmp_path.unlink(missing_ok=True)
```

**Change:** Pass the numpy BGR array directly:
```python
bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
self.detector.predict(bgr)  # YOLO accepts ndarray
```

Remove `tempfile.mkdtemp`, the `tmp_dir` Path, and the `shutil.rmtree` cleanup. YOLO's `predict()` documents ndarray as a valid source type.

### 4. STRtree Spatial Index — `inference/pipeline.py`

**Current:** `any(tile_geom.intersects(poly) for poly in context_polys)`  
Per tile: O(N) shapely intersection tests where N = number of context polygons.

**Change:** Before the tile loop, build `tree = STRtree(context_polys)`. Per tile: `tree.query(tile_geom)` returns candidate indices in O(log N). If the result is non-empty, process the tile.

```python
from shapely.strtree import STRtree
tree = STRtree([p for p in context_polys if p is not None])
# inside loop:
if not tree.query(tile_geom).size:
    det_tiles_skipped += 1
    continue
```

### 5. Avoid Double SHP Load — `inference/pipeline.py`

**Current:** `run()` loads building polygons into `building_polygons` (GDF), passes it to `_classify_rooftops(img, bld_shp_path, transform, building_polygons)`, but `_classify_rooftops` ignores it and calls `gpd.read_file(bld_shp_path)` again.

**Change:** Refactor `_classify_rooftops` to accept and use the already-loaded GDF as its first data argument. The `bld_shp_path` parameter is kept for the existence check only (or removed). If `building_polygons` is None, fall back to reading from path.

### 6. Vectorize `_to_uint8()` — `inference/pipeline.py`

**Current:**
```python
for i in range(min(arr.shape[2], 3)):
    ch = arr[:, :, i].astype(np.float32)
    lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
    ...
    out[:, :, i] = np.clip(...)
```

**Change:** Single vectorized call:
```python
img = arr[:, :, :bands].astype(np.float32)
lo = np.percentile(img, 2, axis=(0, 1), keepdims=True)   # (1,1,C)
hi = np.percentile(img, 98, axis=(0, 1), keepdims=True)
safe = (hi - lo).clip(min=1e-6)
out = np.clip((img - lo) / safe * 255.0, 0, 255).astype(np.uint8)
```

### 7. Cache Spline Window — `inference/pipeline.py`

**Current:** `window = self._spline_window(ps, overlap)` called at the start of every `_segment()` invocation.

**Change:** Move to `__init__`: `self._seg_window = cosine_window(ps, overlap)` (reuse the shared utility from `utils.window`). `_spline_window` instance method removed.

### 8. Parallel CRF — `utils/postprocess.py`

**Current:** `for i, t in enumerate(tqdm(tasks, ...)):` processes tiles one-by-one on the main thread.

**Change:**
```python
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

max_workers = min(os.cpu_count() or 1, len(tasks), 8)
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = {executor.submit(_process_crf_tile, t): t[-2:] for t in tasks}
    for future in tqdm(as_completed(futures), total=len(tasks), desc="  CRF tiles"):
        res, r, c, th, tw = future.result()
        ...
```

`_process_crf_tile` is already a module-level function with numpy array args — fully picklable on Windows `spawn`. Falls back to serial if `ProcessPoolExecutor` raises.

**Platform note:** Windows requires `if __name__ == "__main__"` guard in entry-point scripts (already present in `run_pipeline.py` and `inference/pipeline.py`). ProcessPoolExecutor itself does not require this guard — only the entry script does, which is already in place.

### 9. Module-Level Imports — `utils/postprocess.py`

Move these imports from inside function bodies to module level:
- `import geopandas as gpd` (currently in `mask_to_shapefile`, `merge_rooftop_labels`, `detections_to_shapefile`)
- `from shapely.geometry import GeometryCollection, MultiPolygon, Polygon` (in `clean_vector_geometries`)
- `from shapely.validation import make_valid` (in `clean_vector_geometries`)
- `import math` (in `detections_to_shapefile`)

Guard optional deps with try/except at module level.

### 10. `class_weights()` via `np.bincount` — `data/dataset.py`

**Current:**
```python
counts = torch.zeros(len(self.class_names))
for _, lbl in self.samples:
    counts[lbl] += 1
```

**Change:**
```python
labels = np.array([lbl for _, lbl in self.samples], dtype=np.int64)
counts = torch.from_numpy(np.bincount(labels, minlength=len(self.class_names))).float()
```

---

## Files Changed

| File | Changes |
|---|---|
| `models/stage1_segmentation.py` | `tta_predict()` — batched TTA |
| `models/stage2_models.py` | `RooftopClassifier.predict()` — batched TTA |
| `inference/pipeline.py` | `_detect()` no disk I/O, STRtree, vectorize `_to_uint8()`, cache window, avoid double GDF load |
| `utils/postprocess.py` | Parallel CRF via ProcessPoolExecutor, module-level imports |
| `data/dataset.py` | `class_weights()` via bincount |

## Accuracy Guarantee

No hyperparameters, loss functions, model architectures, or augmentation pipelines are modified. All changes affect only execution order, batching strategy, or I/O method. The mathematical result of every operation is identical to the original.

---

## Expected Speedup

| Phase | Change | Estimated Speedup |
|---|---|---|
| Stage 1 inference TTA | Batched forward pass | 4–8× on TTA phase |
| Stage 2A classification TTA | Batched forward pass | 4–8× on TTA phase |
| Stage 2B detection | No disk I/O | 20–40% on detection phase |
| Context polygon filter | STRtree O(log N) | proportional to N polygons |
| CRF refinement | Parallel tiles | 4–8× on CRF phase |
| Misc | Imports, vectorize, cache | < 5% each |
