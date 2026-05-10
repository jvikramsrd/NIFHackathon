# Accuracy Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve end-to-end prediction accuracy across all three pipeline stages by fixing normalization, TTA, CRF, confidence thresholds, ArcFace config wiring, and training hyperparameters.

**Architecture:** Changes are split across three files: `config.py` (hyperparameters), `inference/pipeline.py` (inference logic), and `models/stage2_models.py` (ArcFace wiring). No architecture changes — same models, better configuration and inference logic.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV, segmentation-models-pytorch, timm, ultralytics, geopandas

---

## File Map

| File | What changes |
|---|---|
| `config.py` | crf_iter 5→10, min_fg_ratio 0.003→0.01, soft_nms_sigma 0.9→0.5, well threshold 0.03→0.10, add stage2a_conf_thresh, add FAST_TTA flag |
| `inference/pipeline.py` | _to_uint8 percentile clipping, remove overlap cap, read FAST_TTA flag, per-class Stage 2A conf thresholds |
| `models/stage2_models.py` | Wire ArcFace s and m from cfg instead of hardcoding |

---

### Task 1: Config — CRF, training noise, detection thresholds

**Files:**
- Modify: `config.py`

- [ ] **Step 1: Apply all config changes**

In `config.py`, locate `STAGE1 = dict(...)` and change:
```python
crf_iter=10,       # was 5 — 5 iterations is insufficient for CRF convergence
min_fg_ratio=0.01, # was 0.003 — 0.3% threshold includes near-empty patches
```

In `STAGE2A = dict(...)`, add after `tta_steps=24,`:
```python
stage2a_conf_thresh={'RCC': 0.45, 'Tiled': 0.55, 'Tin': 0.50, 'Other': 0.40},
```

In `STAGE2B = dict(...)`, change:
```python
soft_nms_sigma=0.5,   # was 0.9 — 0.9 barely penalises overlapping boxes
```
And inside `class_conf_thresh`:
```python
class_conf_thresh={
    'transformer': 0.20,
    'overhead_tank': 0.12,
    'well': 0.10,   # was 0.03 — 3% threshold floods output with false-positive wells
},
```

Add at module level (after `COMPILE_ENABLED = False`):
```python
FAST_TTA = False  # True = 2-scale×4-fold (8 passes); False = 3-scale×8-fold (24 passes)
```

- [ ] **Step 2: Verify config loads without error**

```bash
python -c "import config as CFG; print(CFG.STAGE1['crf_iter'], CFG.STAGE1['min_fg_ratio'], CFG.STAGE2B['soft_nms_sigma'], CFG.STAGE2B['class_conf_thresh']['well'], CFG.FAST_TTA)"
```
Expected output: `10 0.01 0.5 0.1 False`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "config: tighten CRF, detection thresholds, and training noise floor"
```

---

### Task 2: Fix ArcFace to read s and m from config

**Files:**
- Modify: `models/stage2_models.py` (line ~114)

- [ ] **Step 1: Wire ArcFace params from cfg**

In `RooftopClassifier.__init__`, find:
```python
        if self.use_arcface:
            self.head = ArcFaceHead(hidden_dim, self.num_classes, s=30.0, m=0.50)
```
Replace with:
```python
        if self.use_arcface:
            self.head = ArcFaceHead(
                hidden_dim,
                self.num_classes,
                s=float(cfg.get("arcface_s", 30.0)),
                m=float(cfg.get("arcface_m", 0.50)),
            )
```

- [ ] **Step 2: Verify the model builds with config values**

```bash
python -c "
import config as CFG
from models.stage2_models import RooftopClassifier
import torch
m = RooftopClassifier(CFG.STAGE2A)
print('ArcFace m:', m.head.m, 'ArcFace s:', m.head.s)
"
```
Expected output: `ArcFace m: 0.55 ArcFace s: 30.0`

- [ ] **Step 3: Commit**

```bash
git add models/stage2_models.py
git commit -m "fix: wire ArcFace s/m from config (was hardcoded, ignoring arcface_m=0.55)"
```

---

### Task 3: Fix _to_uint8 — percentile normalization

**Files:**
- Modify: `inference/pipeline.py` (function `_to_uint8` near bottom of file)

- [ ] **Step 1: Replace min-max with percentile clipping**

Find the function:
```python
def _to_uint8(arr):
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.ndim == 3:
        for i in range(arr.shape[2]):
            ch = arr[:, :, i].astype(np.float32)
            mn, mx = ch.min(), ch.max()
            out[:, :, i] = 0 if mx == mn else (ch - mn) / (mx - mn) * 255
    return out.astype(np.uint8)
```

Replace with:
```python
def _to_uint8(arr):
    out = np.zeros_like(arr, dtype=np.float32)
    if arr.ndim == 3:
        for i in range(arr.shape[2]):
            ch = arr[:, :, i].astype(np.float32)
            lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
            if hi <= lo:
                out[:, :, i] = 0
            else:
                out[:, :, i] = np.clip((ch - lo) / (hi - lo) * 255, 0, 255)
    return out.astype(np.uint8)
```

- [ ] **Step 2: Verify normalization logic**

```bash
python -c "
import numpy as np
import sys; sys.path.insert(0, '.')
from inference.pipeline import _to_uint8

# Simulate a band with one outlier pixel
arr = np.ones((10, 10, 3), dtype=np.float32) * 100.0
arr[0, 0, 0] = 10000.0  # extreme outlier
result = _to_uint8(arr)
# Most pixels should map near 0 with min-max (outlier dominates),
# but near 255 with percentile (outlier clipped away)
print('Center pixel band 0:', result[5, 5, 0])  # expect > 200 with percentile fix
print('Outlier pixel band 0:', result[0, 0, 0])  # expect 255 (clipped)
"
```
Expected: center pixel ≥ 200, outlier = 255.

- [ ] **Step 3: Commit**

```bash
git add inference/pipeline.py
git commit -m "fix: percentile normalization in _to_uint8 (robust to satellite outlier pixels)"
```

---

### Task 4: Remove overlap cap + wire FAST_TTA

**Files:**
- Modify: `inference/pipeline.py` (method `_segment`)

- [ ] **Step 1: Remove the 128-px overlap cap**

Find in `_segment`:
```python
        overlap = min(int(CFG.STAGE1["overlap"]), 128)
```
Replace with:
```python
        overlap = int(CFG.STAGE1["overlap"])
```

- [ ] **Step 2: Wire FAST_TTA flag for both tta_predict calls**

First call (inside the batch processing loop, `len(batch_inputs) == batch_size` block):
```python
                        probs = (
                            tta_predict(
                                self.seg.model,
                                inp_tensor,
                                C,
                                CFG.AMP_DTYPE,
                                fast_tta=True,
                            )
```
Change `fast_tta=True` to `fast_tta=CFG.FAST_TTA`.

Second call (the flush block after the loop, `len(batch_inputs) > 0`):
```python
                probs = (
                    tta_predict(
                        self.seg.model, inp_tensor, C, CFG.AMP_DTYPE, fast_tta=True
                    )
```
Change `fast_tta=True` to `fast_tta=CFG.FAST_TTA`.

- [ ] **Step 3: Verify the segment method references compile**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import config as CFG
print('overlap:', CFG.STAGE1['overlap'])
print('FAST_TTA:', CFG.FAST_TTA)
# Just import — no GPU needed to check syntax
from inference.pipeline import GeoIntelPipeline
print('Import OK')
"
```
Expected: prints overlap value (192), FAST_TTA (False), and "Import OK".

- [ ] **Step 4: Commit**

```bash
git add inference/pipeline.py
git commit -m "fix: remove 128px overlap cap and wire FAST_TTA flag for Stage 1 TTA"
```

---

### Task 5: Stage 2A per-class confidence thresholds

**Files:**
- Modify: `inference/pipeline.py` (method `_classify_rooftops`, inner function `_process_batch`)

- [ ] **Step 1: Replace the hardcoded 0.55 cutoff**

Find `_process_batch` inside `_classify_rooftops`:
```python
        def _process_batch(inputs, indices):
            if not inputs:
                return
            inp_tensor = torch.stack(inputs).to(self.device)
            inp_tensor = cl_input(inp_tensor)
            with torch.no_grad():
                probs = self.clf.predict(
                    inp_tensor, int(CFG.STAGE2A["tta_steps"]), return_probs=True
                )

            class_names_2a = [str(x) for x in CFG.STAGE2A["class_names"]]
            max_probs, pids = torch.max(probs, dim=1)

            for i, i_idx in enumerate(indices):
                if max_probs[i].item() < 0.55:
                    preds[i_idx] = "Other"
                else:
                    preds[i_idx] = class_names_2a[pids[i].item()]
```

Replace with:
```python
        def _process_batch(inputs, indices):
            if not inputs:
                return
            inp_tensor = torch.stack(inputs).to(self.device)
            inp_tensor = cl_input(inp_tensor)
            with torch.no_grad():
                probs = self.clf.predict(
                    inp_tensor, int(CFG.STAGE2A["tta_steps"]), return_probs=True
                )

            class_names_2a = [str(x) for x in CFG.STAGE2A["class_names"]]
            per_class_thresh = CFG.STAGE2A.get(
                "stage2a_conf_thresh",
                {n: 0.55 for n in class_names_2a},
            )
            max_probs, pids = torch.max(probs, dim=1)

            for i, i_idx in enumerate(indices):
                pred_name = class_names_2a[pids[i].item()]
                thresh = per_class_thresh.get(pred_name, 0.55)
                if max_probs[i].item() < thresh:
                    preds[i_idx] = "Other"
                else:
                    preds[i_idx] = pred_name
```

- [ ] **Step 2: Verify import is clean**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from inference.pipeline import GeoIntelPipeline
import config as CFG
print('stage2a_conf_thresh:', CFG.STAGE2A.get('stage2a_conf_thresh'))
print('Import OK')
"
```
Expected: prints the thresh dict and "Import OK".

- [ ] **Step 3: Commit**

```bash
git add inference/pipeline.py
git commit -m "fix: per-class confidence thresholds for Stage 2A (was hardcoded 0.55 for all classes)"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run the full import chain**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import config as CFG

# Stage 1 checks
assert CFG.STAGE1['crf_iter'] == 10, 'crf_iter not updated'
assert CFG.STAGE1['min_fg_ratio'] == 0.01, 'min_fg_ratio not updated'
assert CFG.FAST_TTA == False, 'FAST_TTA not set'

# Stage 2A checks
assert 'stage2a_conf_thresh' in CFG.STAGE2A, 'stage2a_conf_thresh missing'
assert CFG.STAGE2A['stage2a_conf_thresh']['RCC'] == 0.45

# Stage 2B checks
assert CFG.STAGE2B['soft_nms_sigma'] == 0.5, 'soft_nms_sigma not updated'
assert CFG.STAGE2B['class_conf_thresh']['well'] == 0.10, 'well threshold not updated'

# ArcFace check
from models.stage2_models import RooftopClassifier
m = RooftopClassifier(CFG.STAGE2A)
assert abs(m.head.m - 0.55) < 1e-5, 'ArcFace m not wired from config'

print('All assertions passed.')
"
```
Expected: `All assertions passed.`

- [ ] **Step 2: Commit spec and plan**

```bash
git add docs/
git commit -m "docs: add accuracy improvements spec and implementation plan"
```
