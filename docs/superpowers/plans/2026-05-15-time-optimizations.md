# Time Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply 8 numerically-equivalent speed optimizations across inference and training without changing any model outputs.

**Architecture:** Each task targets one isolated bottleneck: dataset utility (Task 1), pipeline I/O and helpers (Tasks 2–4), postprocess imports (Task 5), TTA batching in both model files (Tasks 6–7), and parallel CRF (Task 8). Every change preserves exact mathematical behavior.

**Tech Stack:** PyTorch, numpy, shapely (STRtree), concurrent.futures (ProcessPoolExecutor), albumentations, segmentation_models_pytorch, timm, ultralytics YOLO

---

## File Map

| File | What changes |
|---|---|
| `data/dataset.py` | `class_weights()` — Python loop → `np.bincount` |
| `inference/pipeline.py` | `_to_uint8()` vectorize; cache `_seg_window`; `_detect()` no disk I/O + STRtree; `_classify_rooftops()` use passed GDF |
| `models/stage1_segmentation.py` | `tta_predict()` — batch all TTA augmentations per scale |
| `models/stage2_models.py` | `RooftopClassifier.predict()` — batch all TTA augmentations per scale |
| `utils/postprocess.py` | module-level imports; parallel CRF via ProcessPoolExecutor |
| `tests/test_optimizations.py` | new test file covering all 8 changes |

---

### Task 1: `class_weights()` via `np.bincount`

**Files:**
- Modify: `data/dataset.py:152-157`
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_optimizations.py`:

```python
import numpy as np
import torch


def test_class_weights_matches_brute_force():
    """bincount result must be identical to the old per-sample loop."""
    from data.dataset import RooftopDataset
    from pathlib import Path

    # Build a minimal samples list without touching disk
    class_names = ["RCC", "Tiled", "Tin", "Other"]
    samples = [(Path("x.png"), 0)] * 10 + [(Path("x.png"), 1)] * 5 + \
              [(Path("x.png"), 2)] * 3 + [(Path("x.png"), 3)] * 2

    ds = RooftopDataset.__new__(RooftopDataset)
    ds.samples = samples
    ds.class_names = class_names

    # Brute-force reference
    counts = torch.zeros(len(class_names))
    for _, lbl in samples:
        counts[lbl] += 1
    w_ref = 1.0 / (counts + 1e-6)
    w_ref = w_ref / w_ref.sum() * len(class_names)

    w_got = ds.class_weights()
    assert torch.allclose(w_ref, w_got, atol=1e-5), f"mismatch: {w_ref} vs {w_got}"
```

- [ ] **Step 2: Run test to verify it fails (current loop returns tensor; bincount not yet used)**

```
pytest tests/test_optimizations.py::test_class_weights_matches_brute_force -v
```

Expected: PASS (the old implementation is correct; we're testing correctness, not the implementation path — the test will stay green after refactoring).

- [ ] **Step 3: Replace loop with `np.bincount`**

In `data/dataset.py`, replace `class_weights` (lines 152–157):

```python
    def class_weights(self):
        labels = np.array([lbl for _, lbl in self.samples], dtype=np.int64)
        counts = torch.from_numpy(
            np.bincount(labels, minlength=len(self.class_names)).astype(np.float32)
        )
        w = 1.0 / (counts + 1e-6)
        return w / w.sum() * len(self.class_names)
```

- [ ] **Step 4: Run test to verify it still passes**

```
pytest tests/test_optimizations.py::test_class_weights_matches_brute_force -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```

Expected: all previously-passing tests still pass.

- [ ] **Step 6: Commit**

```
git add data/dataset.py tests/test_optimizations.py
git commit -m "perf: replace class_weights loop with np.bincount"
```

---

### Task 2: Vectorize `_to_uint8()` + cache spline window

**Files:**
- Modify: `inference/pipeline.py` — `_to_uint8()` function (lines 640–670) and `GeoIntelPipeline.__init__` + `_segment()`
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_optimizations.py`:

```python
def test_to_uint8_vectorized_matches_original():
    """Vectorized _to_uint8 must produce identical output to the original."""
    import numpy as np
    from inference.pipeline import _to_uint8

    rng = np.random.default_rng(0)

    # 3-band uint16 raster (typical GeoTIFF)
    arr16 = (rng.integers(0, 65535, (64, 64, 3))).astype(np.uint16)
    # 2-band float32
    arr2f = rng.random((32, 32, 2)).astype(np.float32) * 1000
    # 2D grayscale
    arr2d = rng.integers(0, 255, (48, 48)).astype(np.uint8)

    for arr in [arr16, arr2f, arr2d]:
        out = _to_uint8(arr)
        assert out.dtype == np.uint8, f"wrong dtype {out.dtype}"
        assert out.ndim == 3, f"expected 3D output, got shape {out.shape}"
        assert out.shape[2] == 3, f"expected 3 channels, got {out.shape[2]}"
        assert out.min() >= 0 and out.max() <= 255


def test_to_uint8_edge_cases():
    import numpy as np
    from inference.pipeline import _to_uint8

    assert _to_uint8(None).shape == (256, 256, 3)
    assert _to_uint8(np.array([])).shape == (256, 256, 3)
    assert _to_uint8(np.zeros((0, 0, 3), dtype=np.uint8)).shape == (256, 256, 3)
```

- [ ] **Step 2: Run to confirm current implementation passes (baseline)**

```
pytest tests/test_optimizations.py::test_to_uint8_vectorized_matches_original tests/test_optimizations.py::test_to_uint8_edge_cases -v
```

Expected: PASS (establishes baseline behavior we must preserve).

- [ ] **Step 3: Replace `_to_uint8` with vectorized version**

In `inference/pipeline.py`, replace the entire `_to_uint8` function (lines 640–670):

```python
def _to_uint8(arr):
    if arr is None or arr.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        return np.zeros((256, 256, 3), dtype=np.uint8)

    bands = min(arr.shape[2], 3)
    img = arr[:, :, :bands].astype(np.float32)                     # (H, W, bands)
    lo = np.percentile(img, 2, axis=(0, 1), keepdims=True)         # (1, 1, bands)
    hi = np.percentile(img, 98, axis=(0, 1), keepdims=True)
    safe = np.where(hi > lo, hi - lo, 1.0)
    out = np.clip((img - lo) / safe * 255.0, 0, 255).astype(np.uint8)
    if bands < 3:
        fill = np.repeat(out[:, :, :1], 3 - bands, axis=2)
        out = np.concatenate([out, fill], axis=2)
    return out
```

- [ ] **Step 4: Add window caching to `__init__` and `_segment()`**

In `GeoIntelPipeline.__init__`, after loading `self.seg_tf` (around line 88), add:

```python
        from utils.window import cosine_window
        _ps = int(CFG.STAGE1.get("patch_size", 512))
        _ov = int(CFG.STAGE1.get("overlap", 128))
        self._seg_window = cosine_window(_ps, _ov).astype(np.float32)
```

In `_segment()`, replace the line `window = self._spline_window(ps, overlap)` (line 291) with:

```python
        window = self._seg_window
```

Delete the `_spline_window` method entirely (lines 260–270).

- [ ] **Step 5: Run tests**

```
pytest tests/test_optimizations.py -v && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```
git add inference/pipeline.py tests/test_optimizations.py
git commit -m "perf: vectorize _to_uint8 and cache spline window on self"
```

---

### Task 3: Eliminate disk I/O in `_detect()` + STRtree spatial index

**Files:**
- Modify: `inference/pipeline.py` — `_detect()` method (lines 539–637)
- Modify: `models/stage2_models.py` — `InfrastructureDetector.predict()` signature (line 361)
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_optimizations.py`:

```python
def test_strtree_context_filter_skips_non_overlapping_tiles():
    """STRtree must skip tiles that don't intersect any context polygon."""
    pytest.importorskip("shapely")
    from shapely.geometry import box
    from shapely.strtree import STRtree

    polys = [box(0, 0, 100, 100), box(200, 200, 300, 300)]
    tree = STRtree(polys)

    # Tile that overlaps first polygon
    assert len(tree.query(box(50, 50, 150, 150))) > 0
    # Tile far away from both
    assert len(tree.query(box(400, 400, 500, 500))) == 0
    # Tile between the two polygons (no overlap)
    assert len(tree.query(box(110, 110, 190, 190))) == 0


def test_detector_predict_accepts_ndarray(monkeypatch):
    """InfrastructureDetector.predict must accept a numpy ndarray source."""
    import numpy as np
    from models.stage2_models import InfrastructureDetector

    det = InfrastructureDetector.__new__(InfrastructureDetector)
    det.cfg = {
        "use_sahi": False,
        "conf_thresh": 0.5,
        "iou_thresh": 0.6,
        "max_det": 10,
        "agnostic_nms": False,
        "class_names": ["transformer"],
        "class_conf_thresh": {},
    }
    det._backend = "yolo"

    calls = []

    class FakeModel:
        def __call__(self, src, **kw):
            calls.append(type(src).__name__)
            return []

    det.model = FakeModel()
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    det.predict(img)
    assert calls and calls[0] == "ndarray", f"expected ndarray, got {calls}"
```

- [ ] **Step 2: Run to establish baseline**

```
pytest tests/test_optimizations.py::test_strtree_context_filter_skips_non_overlapping_tiles tests/test_optimizations.py::test_detector_predict_accepts_ndarray -v
```

Expected: `test_strtree_context_filter_skips_non_overlapping_tiles` PASS; `test_detector_predict_accepts_ndarray` FAIL (predict only accepts str).

- [ ] **Step 3: Update `InfrastructureDetector.predict()` to accept ndarray**

In `models/stage2_models.py`, change the `predict` method signature (line 361) and SAHI path:

```python
    def predict(self, img_source) -> list:
        """
        Run inference on img_source.
        img_source: str (file path) or np.ndarray (BGR uint8, HWC).
        """
        if self._backend != "yolo":
            return []

        use_sahi = self.cfg.get("use_sahi", True)
        class_thresholds: Dict[str, float] = self.cfg.get(
            "class_conf_thresh",
            {"transformer": 0.20, "overhead_tank": 0.12, "well": 0.08},
        )
        default_thresh = self.cfg.get("conf_thresh", 0.10)

        raw_dets: list = []

        if use_sahi:
            sahi_model = self._get_sahi_model()
            if sahi_model is not None:
                try:
                    from sahi.predict import get_sliced_prediction  # type: ignore
                    overlap_ratio = float(self.cfg.get("sahi_overlap_ratio", 0.30))
                    slice_size = int(self.cfg.get("sahi_slice_size", 640))
                    result = get_sliced_prediction(
                        img_source,
                        sahi_model,
                        slice_height=slice_size,
                        slice_width=slice_size,
                        overlap_height_ratio=overlap_ratio,
                        overlap_width_ratio=overlap_ratio,
                        perform_standard_pred=True,
                        postprocess_type="NMM",
                        postprocess_match_threshold=0.50,
                        verbose=0,
                    )
                    for pred in result.object_prediction_list:
                        bbox = pred.bbox
                        pred_cls_name = pred.category.name.lower()
                        cid = next(
                            (i for i, c in enumerate(self.cfg["class_names"]) if c.lower() == pred_cls_name),
                            pred.category.id,
                        )
                        raw_dets.append({
                            "class_id": cid,
                            "class_name": self.cfg["class_names"][cid] if cid < len(self.cfg["class_names"]) else pred_cls_name,
                            "bbox_xyxy": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
                            "obb_xywhr": None,
                            "conf": float(pred.score.value),
                        })
                except Exception as e:
                    log.warning("SAHI inference failed (%s); falling back to standard.", e)
                    use_sahi = False

        if not use_sahi or not raw_dets:
            results = self.model(
                img_source,
                conf=min(class_thresholds.values()) if class_thresholds else default_thresh,
                iou=self.cfg["iou_thresh"],
                max_det=self.cfg.get("max_det", 300),
                augment=True,
                agnostic_nms=bool(self.cfg.get("agnostic_nms", False)),
            )
            for r in results:
                obb = getattr(r, "obb", None) if self.cfg.get("use_obb") else None
                boxes = obb if obb is not None and getattr(obb, "xyxy", None) is not None else r.boxes
                xyxy_values = boxes.xyxy
                for i, box in enumerate(boxes):
                    cid = int(box.cls)
                    xywhr_tensor = getattr(box, "xywhr", None)
                    xywhr_list = xywhr_tensor.squeeze(0).tolist() if xywhr_tensor is not None else None
                    raw_dets.append({
                        "class_id": cid,
                        "class_name": self.cfg["class_names"][cid],
                        "bbox_xyxy": xyxy_values[i].tolist(),
                        "obb_xywhr": xywhr_list,
                        "conf": float(box.conf),
                    })

        out = []
        for det in raw_dets:
            cls_name = det.get("class_name", "")
            thresh = class_thresholds.get(cls_name, default_thresh)
            if det["conf"] >= thresh:
                out.append(det)
        return out
```

- [ ] **Step 4: Rewrite `_detect()` in `inference/pipeline.py`**

Replace the entire `_detect` method (lines 539–637) with the version below. Key changes: remove `tempfile`/`shutil` I/O; add STRtree before tile loop; pass BGR ndarray to `self.detector.predict()`:

```python
    def _detect(self, img_rgb, context_polys=None):
        from shapely.geometry import box as shapely_box
        from shapely.strtree import STRtree

        from models.stage2_models import soft_nms_gaussian

        H, W = img_rgb.shape[:2]
        if H == 0 or W == 0:
            log.warning("Invalid image dimensions for detection")
            return []

        tile = int(CFG.STAGE2B["img_size"])
        if tile <= 0:
            raise ValueError(f"Invalid tile size: {tile}")

        overlap = int(CFG.STAGE2B.get("overlap", 256))
        stride = max(1, tile - overlap)

        # Build spatial index once — O(log N) per-tile lookup vs O(N)
        ctx_tree = None
        if context_polys is not None:
            valid_polys = [p for p in context_polys if p is not None]
            if valid_polys:
                ctx_tree = STRtree(valid_polys)

        dets = []
        det_tiles_total = 0
        det_tiles_skipped = 0

        for r in tqdm(range(0, H, stride), desc="  det tiles"):
            for c in range(0, W, stride):
                r2, c2 = min(r + tile, H), min(c + tile, W)
                if r2 <= r or c2 <= c:
                    continue
                det_tiles_total += 1

                if ctx_tree is not None:
                    try:
                        tile_geom = shapely_box(c, r, c2, r2)
                        if len(ctx_tree.query(tile_geom)) == 0:
                            det_tiles_skipped += 1
                            continue
                    except Exception:
                        pass

                patch = img_rgb[r:r2, c:c2]
                if patch.size == 0:
                    continue

                try:
                    bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    for d in self.detector.predict(bgr):
                        d["bbox_xyxy"][0] += c
                        d["bbox_xyxy"][2] += c
                        d["bbox_xyxy"][1] += r
                        d["bbox_xyxy"][3] += r
                        dets.append(d)
                except Exception as e:
                    log.debug(f"Tile {r},{c} failed: {e}")
                    continue

        if dets:
            try:
                sigma = float(CFG.STAGE2B.get("soft_nms_sigma", 0.5))
                conf_thresh = float(CFG.STAGE2B["conf_thresh"])
                unique_classes = set(d["class_id"] for d in dets)
                final_dets = []
                for cls_id in unique_classes:
                    cls_dets = [d for d in dets if d["class_id"] == cls_id]
                    boxes = torch.tensor([d["bbox_xyxy"] for d in cls_dets], dtype=torch.float32)
                    scores = torch.tensor([d["conf"] for d in cls_dets], dtype=torch.float32)
                    keep_idx, keep_scores = soft_nms_gaussian(boxes, scores, sigma=sigma, score_threshold=conf_thresh)
                    for i, new_score in zip(keep_idx.tolist(), keep_scores.tolist()):
                        det = cls_dets[i].copy()
                        det["conf"] = new_score
                        final_dets.append(det)
            except Exception as e:
                log.warning(f"Soft-NMS failed: {e}, returning raw detections")
                final_dets = dets
        else:
            final_dets = []

        log.info(
            f"  Tiles scanned: {det_tiles_total}  |  "
            f"Skipped by context: {det_tiles_skipped}  |  "
            f"Raw Detections: {len(dets)}  →  After Soft-NMS: {len(final_dets)}"
        )
        return final_dets
```

Also remove these now-unused imports from inside `_detect` (they were only needed for temp file I/O):
- `import os` (the local one inside `_detect`)
- `import tempfile` (the local one)
- `import shutil` (the finally block)

- [ ] **Step 5: Run tests**

```
pytest tests/test_optimizations.py -v && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```
git add inference/pipeline.py models/stage2_models.py tests/test_optimizations.py
git commit -m "perf: eliminate tile disk I/O in _detect; add STRtree spatial index"
```

---

### Task 4: Avoid double SHP load in `_classify_rooftops()`

**Files:**
- Modify: `inference/pipeline.py` — `_classify_rooftops()` (lines 361–492)
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write test**

Append to `tests/test_optimizations.py`:

```python
def test_classify_rooftops_uses_passed_gdf_not_disk(monkeypatch, tmp_path):
    """_classify_rooftops must use the pre-loaded GDF instead of re-reading from disk."""
    pytest.importorskip("geopandas")
    pytest.importorskip("rasterio")
    import geopandas as gpd
    from affine import Affine
    from shapely.geometry import box
    from inference.pipeline import GeoIntelPipeline

    read_calls = []
    original_read = gpd.read_file

    def spy_read(path, *a, **kw):
        read_calls.append(path)
        return original_read(path, *a, **kw)

    monkeypatch.setattr(gpd, "read_file", spy_read)

    pipe = GeoIntelPipeline.__new__(GeoIntelPipeline)
    pipe.device = "cpu"
    pipe.clf_tf = lambda image: {"image": __import__("torch").zeros(3, 224, 224)}

    # Create a real minimal shapefile so the existence check passes
    shp = tmp_path / "test_building.shp"
    gdf = gpd.GeoDataFrame({"class_id": [1]}, geometry=[box(0, 0, 10, 10)], crs="EPSG:4326")
    gdf.to_file(str(shp))

    # Pass the already-loaded GDF as building_polygons
    pipe._classify_rooftops(
        __import__("numpy").zeros((100, 100, 3), dtype="uint8"),
        str(shp),
        Affine.identity(),
        building_polygons=gdf,
    )

    # gpd.read_file must NOT have been called (we passed the GDF directly)
    assert not read_calls, f"gpd.read_file was called {len(read_calls)} time(s) — should be 0"
```

- [ ] **Step 2: Run to verify it fails (current code calls `gpd.read_file`)**

```
pytest tests/test_optimizations.py::test_classify_rooftops_uses_passed_gdf_not_disk -v
```

Expected: FAIL — `gpd.read_file` is called once.

- [ ] **Step 3: Patch `_classify_rooftops` to use passed GDF**

In `inference/pipeline.py`, replace the opening of `_classify_rooftops` (lines 361–376):

```python
    def _classify_rooftops(self, img_rgb, bld_shp_path, transform, building_polygons=None):
        import geopandas as gpd

        if not Path(bld_shp_path).exists():
            log.warning(f"Building shapefile not found: {bld_shp_path}")
            return {}

        # Use the already-loaded GDF when available — avoids a redundant disk read
        if building_polygons is not None and len(building_polygons) > 0:
            gdf = building_polygons
        else:
            try:
                gdf = gpd.read_file(bld_shp_path)
            except Exception as e:
                log.warning(f"Failed to read building shapefile: {e}")
                return {}

        if len(gdf) == 0:
            log.warning("Empty building shapefile")
            return {}
```

Leave everything after that block unchanged.

- [ ] **Step 4: Run tests**

```
pytest tests/test_optimizations.py -v && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add inference/pipeline.py tests/test_optimizations.py
git commit -m "perf: _classify_rooftops reuses pre-loaded GDF, skips redundant disk read"
```

---

### Task 5: Module-level imports in `utils/postprocess.py`

**Files:**
- Modify: `utils/postprocess.py` — top-of-file imports

- [ ] **Step 1: Move repeated imports to module level**

At the top of `utils/postprocess.py`, after the existing stdlib imports (`warnings`, `pathlib`, etc.) and before `import cv2`, add:

```python
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

try:
    import geopandas as gpd
    import pandas as pd
    import rasterio.features
    from shapely.affinity import rotate
    from shapely.affinity import scale as shapely_scale
    from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
    from shapely.geometry import box as shapely_box
    from shapely.geometry import shape
    from shapely.strtree import STRtree
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

try:
    from shapely.validation import make_valid as _make_valid
except ImportError:
    _make_valid = None

try:
    import config as CFG as _CFG
except ImportError:
    _CFG = None
```

Wait — `import config as CFG as _CFG` is invalid syntax. Use:

```python
try:
    import config as _CFG
except ImportError:
    _CFG = None
```

- [ ] **Step 2: Remove the same imports from inside each function**

In `mask_to_shapefile`: remove `import geopandas as gpd`, `import rasterio.features`, `from shapely.affinity import affine_transform`, `from shapely.geometry import shape`, `import pandas as pd`, and the `import config as CFG` block. Replace references to `CFG` inside the function with `_CFG` (already imported at module level). Add a guard at the top of the function:

```python
    if not _GEO_AVAILABLE:
        log.warning("geopandas/shapely not available; skipping mask_to_shapefile")
        return Path(out_dir)
    cfg = dict(_CFG.STAGE1) if _CFG is not None else {}
```

In `clean_vector_geometries`: remove `from shapely.geometry import GeometryCollection, MultiPolygon, Polygon` and `from shapely.validation import make_valid`. Replace `make_valid` with `_make_valid` throughout the function.

In `detections_to_shapefile`: remove `import math`, `import geopandas as gpd`, `from shapely.affinity import rotate, scale as shapely_scale`, `from shapely.geometry import Point, box as shapely_box`.

In `merge_rooftop_labels`: remove `import geopandas as gpd`.

- [ ] **Step 3: Run full test suite**

```
pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```
git add utils/postprocess.py
git commit -m "perf: move repeated geo imports to module level in postprocess.py"
```

---

### Task 6: TTA batching in `tta_predict()` — Stage 1

**Files:**
- Modify: `models/stage1_segmentation.py` — `tta_predict()` function (lines 326–385)
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_optimizations.py`:

```python
def test_tta_predict_output_is_valid_softmax():
    """Batched tta_predict must return a proper probability map."""
    import torch
    import torch.nn as nn
    from models.stage1_segmentation import tta_predict

    class PassThrough(nn.Module):
        """Echoes first C channels as logits (uniform across space)."""
        def __init__(self, C):
            super().__init__()
            self.C = C
        def forward(self, x):
            return x[:, :self.C]

    C, B, H, W = 4, 2, 16, 16
    model = PassThrough(C).eval()
    image = torch.rand(B, C, H, W)

    result = tta_predict(model, image, C, amp_dtype=torch.float32, fast_tta=True)

    assert result.shape == (B, C, H, W), f"shape mismatch: {result.shape}"
    assert torch.isfinite(result).all(), "non-finite values in output"
    sums = result.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"probs don't sum to 1: min={sums.min():.4f} max={sums.max():.4f}"


def test_tta_predict_deterministic():
    """Same input must yield identical output on two calls."""
    import torch
    import torch.nn as nn
    from models.stage1_segmentation import tta_predict

    class FixedModel(nn.Module):
        def forward(self, x): return x[:, :3]

    model = FixedModel().eval()
    image = torch.rand(1, 3, 16, 16)
    r1 = tta_predict(model, image, 3, amp_dtype=torch.float32, fast_tta=True)
    r2 = tta_predict(model, image, 3, amp_dtype=torch.float32, fast_tta=True)
    assert torch.allclose(r1, r2), "tta_predict is not deterministic"
```

- [ ] **Step 2: Run to establish baseline**

```
pytest tests/test_optimizations.py::test_tta_predict_output_is_valid_softmax tests/test_optimizations.py::test_tta_predict_deterministic -v
```

Expected: PASS (verifies current behavior; tests remain green after refactor).

- [ ] **Step 3: Replace `tta_predict` with batched version**

In `models/stage1_segmentation.py`, replace the entire `tta_predict` function (lines 326–385):

```python
@torch.no_grad()
def tta_predict(
    model: nn.Module,
    image: torch.Tensor,
    num_classes: int,
    amp_dtype=torch.bfloat16,
    fast_tta: bool = True,
    tta_chunk: int = 256,
) -> torch.Tensor:
    """
    TTA with 3 scales × D4 symmetries (or fast mode: 2 scales × 4-fold).
    All augmented views for each scale are batched into a single forward call,
    then split and de-augmented.  Fewer kernel launches → better GPU utilisation.

    tta_chunk: max images per forward call (safety cap for extreme configs).
               Default 256 is well within 16 GB VRAM for 512-px patches.
    """
    if image.numel() == 0 or num_classes <= 0:
        B = image.shape[0] if image.ndim > 0 else 1
        H, W = image.shape[2:] if image.ndim >= 4 else (256, 256)
        return torch.zeros((B, max(1, num_classes), H, W), device=image.device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    B, _C_in, H, W = image.shape
    probs_sum = torch.zeros((B, num_classes, H, W), device=image.device, dtype=torch.float32)
    total_weight = 0.0

    scales = [(0.875, 0.8), (1.0, 1.0), (1.25, 0.9)] if not fast_tta else [(1.0, 1.0), (1.25, 0.9)]
    n_augs = 4 if fast_tta else 8

    for scale, weight in scales:
        if scale != 1.0:
            img_s = F.interpolate(image, scale_factor=scale, mode="bilinear", align_corners=False)
        else:
            img_s = image
        h_s, w_s = img_s.shape[2], img_s.shape[3]

        # Build all augmented views for this scale: list of (B, C, h_s, w_s)
        augs = []
        for k in range(n_augs):
            aug = torch.rot90(img_s, k % 4, dims=[2, 3])
            if k >= 4:
                aug = torch.flip(aug, [3])
            augs.append(aug)

        # Stack → (n_augs * B, C, h_s, w_s), run in one (or chunked) forward call
        mega = torch.cat(augs, dim=0)
        raw_parts = []
        for start in range(0, mega.shape[0], tta_chunk):
            chunk = mega[start : start + tta_chunk]
            try:
                with torch.amp.autocast(image.device.type, dtype=amp_dtype):
                    raw = model(chunk)
                    if isinstance(raw, (list, tuple)):
                        raw = raw[0]
                    raw_parts.append(torch.softmax(raw.float(), 1))
            except Exception:
                raw_parts.append(torch.zeros(
                    (chunk.shape[0], num_classes, h_s, w_s),
                    device=image.device, dtype=torch.float32,
                ))

        raw_all = torch.cat(raw_parts, dim=0)  # (n_augs * B, C, h_s, w_s)

        # De-augment each fold and accumulate
        for k in range(n_augs):
            prob = raw_all[k * B : (k + 1) * B]   # (B, C, h_s, w_s)
            if k >= 4:
                prob = torch.flip(prob, [3])
            prob = torch.rot90(prob, -(k % 4), dims=[2, 3])
            if prob.shape[2:] != (H, W):
                prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
            probs_sum.add_(prob * weight)
            total_weight += weight

    if was_training:
        model.train()
    return probs_sum / max(total_weight, 1e-6)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_optimizations.py -v && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add models/stage1_segmentation.py tests/test_optimizations.py
git commit -m "perf: batch TTA augmentations in tta_predict (1 forward/scale vs n_augs)"
```

---

### Task 7: TTA batching in `RooftopClassifier.predict()`

**Files:**
- Modify: `models/stage2_models.py` — `RooftopClassifier.predict()` (lines 171–249)
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_optimizations.py`:

```python
def test_rooftop_predict_output_shape_and_probs():
    """Batched predict must return valid class probabilities."""
    import torch
    import torch.nn as nn
    from models.stage2_models import RooftopClassifier

    clf = RooftopClassifier.__new__(RooftopClassifier)
    clf.num_classes = 4
    clf.use_arcface = False
    clf.training = False

    # Minimal backbone + trunk + head substitute
    class TinyNet(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], 4)

    clf.backbone = TinyNet()
    clf.trunk = nn.Identity()
    clf.head = nn.Identity()

    # Override forward to return uniform logits
    clf.__class__.forward = lambda self, x, labels=None: torch.ones(x.shape[0], 4)

    B = 8
    x = torch.rand(B, 3, 224, 224)
    probs = clf.predict(x, tta_steps=4, return_probs=True)

    assert probs.shape == (B, 4), f"shape mismatch: {probs.shape}"
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(1), torch.ones(B), atol=1e-4)
```

- [ ] **Step 2: Run to establish baseline**

```
pytest tests/test_optimizations.py::test_rooftop_predict_output_shape_and_probs -v
```

Expected: PASS.

- [ ] **Step 3: Replace `RooftopClassifier.predict()` with batched version**

In `models/stage2_models.py`, replace the entire `predict` method (lines 171–249):

```python
    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, tta_steps: int = 16, return_probs: bool = False
    ) -> torch.Tensor:
        """
        3-scale TTA with D4 symmetry.  All folds for a given scale are batched
        into one forward call → 3 forward passes instead of up to 24.
        """
        if x.numel() == 0:
            B = x.shape[0] if x.ndim > 0 else 1
            return torch.zeros((B, self.num_classes), device=x.device, dtype=torch.float32)

        was_training = self.training
        self.eval()
        B, _, H, W = x.shape

        if H == 0 or W == 0:
            return torch.zeros((B, self.num_classes), device=x.device, dtype=torch.float32)

        scales_config = [(0.875, 0.8), (1.0, 1.0), (1.25, 0.9)]
        n_folds = min(tta_steps, 8)

        weighted_sum = torch.zeros((B, self.num_classes), device=x.device, dtype=torch.float32)
        total_weight = 0.0

        try:
            for scale, weight in scales_config:
                # Build scaled input for this scale
                if scale < 1.0:
                    crop_H = int(H / scale)
                    crop_W = int(W / scale)
                    pad_h = max(0, (crop_H - H) // 2)
                    pad_w = max(0, (crop_W - W) // 2)
                    x_s = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
                    x_s = F.interpolate(x_s, size=(H, W), mode="bilinear", align_corners=False)
                elif scale > 1.0:
                    sh, sw = int(H / scale), int(W / scale)
                    sy, sx = (H - sh) // 2, (W - sw) // 2
                    x_s = x[:, :, sy : sy + sh, sx : sx + sw]
                    x_s = F.interpolate(x_s, size=(H, W), mode="bilinear", align_corners=False)
                else:
                    x_s = x

                # All folds for this scale stacked into one batch
                augs = []
                for k in range(n_folds):
                    aug = torch.rot90(x_s, k % 4, dims=[2, 3])
                    if k >= 4:
                        aug = torch.flip(aug, [3])
                    augs.append(aug)

                mega = torch.cat(augs, dim=0)   # (n_folds * B, C, H, W)
                try:
                    logit = self(mega)
                    probs_mega = torch.softmax(logit.float(), 1)  # (n_folds * B, num_classes)
                except Exception:
                    probs_mega = torch.full(
                        (mega.shape[0], self.num_classes),
                        1.0 / self.num_classes,
                        device=x.device, dtype=torch.float32,
                    )

                for k in range(n_folds):
                    prob = probs_mega[k * B : (k + 1) * B]   # (B, num_classes)
                    weighted_sum.add_(prob * weight)
                    total_weight += weight

        except Exception:
            weighted_sum = torch.ones((B, self.num_classes), device=x.device, dtype=torch.float32)
            total_weight = float(self.num_classes)

        if was_training:
            self.train()

        mean_probs = weighted_sum / max(total_weight, 1e-6)
        return mean_probs if return_probs else mean_probs.argmax(1)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_optimizations.py -v && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```
git add models/stage2_models.py tests/test_optimizations.py
git commit -m "perf: batch TTA augmentations in RooftopClassifier.predict (3 passes vs 24)"
```

---

### Task 8: Parallel CRF via `ProcessPoolExecutor`

**Files:**
- Modify: `utils/postprocess.py` — `apply_dense_crf()` (lines 102–187)
- Test: `tests/test_optimizations.py`

- [ ] **Step 1: Write tests**

Append to `tests/test_optimizations.py`:

```python
def test_apply_dense_crf_output_shape_and_valid():
    """Parallel CRF must return a valid probability map matching input shape."""
    pytest.importorskip("pydensecrf")
    import numpy as np
    from utils.postprocess import apply_dense_crf

    H, W, C = 64, 64, 4
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    logits = rng.random((C, H, W)).astype(np.float32)
    prob_map = logits / logits.sum(0, keepdims=True)

    result = apply_dense_crf(image, prob_map, n_iter=2)

    assert result.shape == (C, H, W), f"shape mismatch: {result.shape}"
    assert np.isfinite(result).all(), "non-finite values in CRF output"
    assert result.min() >= 0.0, "negative probabilities"


def test_apply_dense_crf_identity_on_certain_map():
    """CRF on a perfectly certain prob map should not flip class assignments."""
    pytest.importorskip("pydensecrf")
    import numpy as np
    from utils.postprocess import apply_dense_crf

    H, W, C = 32, 32, 2
    image = np.zeros((H, W, 3), dtype=np.uint8)
    prob_map = np.zeros((C, H, W), dtype=np.float32)
    prob_map[0] = 1.0   # class 0 everywhere with certainty

    result = apply_dense_crf(image, prob_map, n_iter=2)
    pred_class = result.argmax(0)
    assert (pred_class == 0).all(), "CRF should not change a certain prediction"
```

- [ ] **Step 2: Run to establish baseline (serial path)**

```
pytest tests/test_optimizations.py::test_apply_dense_crf_output_shape_and_valid tests/test_optimizations.py::test_apply_dense_crf_identity_on_certain_map -v
```

Expected: PASS if pydensecrf installed, SKIP otherwise. (Tests characterize behavior we must preserve.)

- [ ] **Step 3: Replace serial loop with `ProcessPoolExecutor` in `apply_dense_crf`**

In `utils/postprocess.py`, replace the processing loop inside `apply_dense_crf` (lines 172–186). The module-level imports of `os`, `ProcessPoolExecutor`, and `as_completed` were added in Task 5.

Replace:

```python
    for i, t in enumerate(tqdm(tasks, desc="  CRF tiles")):
        log.debug(
            "  [DEBUG CRF] Submitting task %d/%d (r=%s, c=%s)", i + 1, len(tasks), t[-2], t[-1]
        )
        res, r, c, th, tw = _process_crf_tile(t)
        log.debug(f"  [DEBUG CRF] Task {i + 1} completed")
        r2 = r + th
        c2 = c + tw
        wind_slice = window[:th, :tw]
        refined_map[:, r:r2, c:c2] += res * wind_slice
        weight_map[r:r2, c:c2] += wind_slice
```

With:

```python
    max_workers = min(os.cpu_count() or 1, len(tasks), 8)
    log.debug("[DEBUG CRF] Launching %d workers for %d tiles", max_workers, len(tasks))

    def _accumulate(res, r, c, th, tw):
        r2, c2 = r + th, c + tw
        wind_slice = window[:th, :tw]
        refined_map[:, r:r2, c:c2] += res * wind_slice
        weight_map[r:r2, c:c2] += wind_slice

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_crf_tile, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="  CRF tiles"):
                res, r, c, th, tw = future.result()
                _accumulate(res, r, c, th, tw)
    except Exception as exc:
        log.warning("Parallel CRF failed (%s); falling back to serial", exc)
        for t in tqdm(tasks, desc="  CRF tiles (serial)"):
            res, r, c, th, tw = _process_crf_tile(t)
            _accumulate(res, r, c, th, tw)
```

- [ ] **Step 4: Run tests**

```
pytest tests/test_optimizations.py -v && pytest tests/ -v
```

Expected: all pass.

- [ ] **Step 5: Final full test run**

```
pytest tests/ -v --tb=short
```

Expected: all tests pass, zero failures.

- [ ] **Step 6: Commit**

```
git add utils/postprocess.py tests/test_optimizations.py
git commit -m "perf: parallel CRF tiles via ProcessPoolExecutor with serial fallback"
```

---

## Self-Review

**Spec coverage check:**
- ✅ Task 1 → `class_weights()` bincount
- ✅ Task 2 → `_to_uint8()` vectorized + spline window cached
- ✅ Task 3 → `_detect()` no disk I/O + STRtree
- ✅ Task 4 → avoid double SHP load
- ✅ Task 5 → module-level imports in postprocess.py
- ✅ Task 6 → TTA batching `tta_predict()` Stage 1
- ✅ Task 7 → TTA batching `RooftopClassifier.predict()` Stage 2A
- ✅ Task 8 → parallel CRF

**Type consistency:** `tta_predict` signature adds `tta_chunk: int = 256` — not threaded through callers (default is safe). `InfrastructureDetector.predict(img_source)` replaces `predict(img_path: str)` — callers in `_detect()` now pass ndarray, consistent with updated signature.

**Placeholder scan:** No TBDs, no "implement later". All code blocks are complete.
