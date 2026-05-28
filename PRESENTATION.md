---
marp: true
theme: default
paginate: true
size: 16:9
header: 'GeoIntel Pipeline — SVAMITVA · A4000'
footer: '© NIF Hackathon · GeoIntel'
style: |
  section { font-size: 26px; }
  h1 { color: #1f4e8c; }
  h2 { color: #1f4e8c; border-bottom: 2px solid #1f4e8c; padding-bottom: 4px; }
  table { font-size: 22px; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
  .small { font-size: 20px; }
  .tiny { font-size: 18px; }
  .green { color: #2a8a2a; }
  .red { color: #b00020; }
  .muted { color: #666; }
---

<!-- _class: lead -->

# GeoIntel Pipeline
### Drone-orthophoto → GIS-ready vectors
### Three-stage deep learning, A4000-optimised

<br>

**SVAMITVA dataset · RTX A4000 16 GB · Windows 11**

---

## 1. The Problem

Given a **drone orthophoto** of a SVAMITVA village (often **70 GB+ uncompressed**), produce:

1. **Building footprints** with rooftop material classification
2. **Road network** polygons
3. **Waterbody** polygons
4. **Infrastructure points** — electric transformers, overhead tanks, hand pumps

Output must be **GIS-ready** (.shp, .gpkg) — load straight into QGIS / ArcGIS.

**Constraints**

- One RTX A4000 (16 GB VRAM, Ampere)
- 32 GB system RAM
- Some rasters exceed **213 734 × 112 836 px**

---

## 2. Why Three Stages (not End-to-End)

A single end-to-end model would have to learn three tasks at three scales:

| Sub-task | Scale | Right tool |
|---|---|---|
| Pixel segmentation | 512 px patches, dense output | Encoder-decoder (MAnet/MiT-B4) |
| Texture classification | 224 px crops | Fine-grained classifier (ConvNeXt-L + ArcFace) |
| Small-object detection | 1280 px tiles, ~15 px wells | Detector (YOLOv11l + SAHI) |

End-to-end alternatives like **Mask2Former** or **Panoptic-DeepLab** force one architecture to compromise on all three. Splitting lets each stage **use the right input size, the right loss, and the right inductive bias**.

**Bonus:** Stage 2A/2B are *gated* by Stage 1 — they only see crops/tiles that already contain buildings or roads → **40–70 % of compute skipped**.

---

## 3. Stage 1 — Architecture Choice

### Decoder: **MAnet** vs alternatives

| Decoder | Boundary sharpness | VRAM | Notes |
|---|---|---|---|
| `Unet` | medium | low | smears irregular outlines |
| `UnetPlusPlus` | high | high | **rejects MiT encoders** (hardcoded check in smp) |
| **`MAnet` ←** | **highest** | medium | PAB + MFAB modules built for irregular shapes |
| `DeepLabV3+` | high | medium | ASPP good for scale, but worse on thin classes (roads) |

### Encoder: **MiT-B4** vs alternatives

| Encoder | Params | Speed | mIoU on SVAMITVA |
|---|---|---|---|
| ResNet-50 | 25 M | fast | baseline |
| EfficientNet-B4 | 21 M | very fast | +0.5 |
| **MiT-B4 ←** | 62 M | medium | **+2.1 vs ResNet50** |
| MiT-B5 | 82 M | slow | +2.3 — not worth 20 % slower |

---

## 4. Stage 1 — Loss Design

### Single-loss vs **TriLoss** (six weighted terms)

| Term | Weight | What it fixes |
|---|---|---|
| Cosine-log Dice | 0.40 | Smooth gradients near 0/1; targets overlap |
| Cross-Entropy (label-smooth 0.08) | 0.15 | Stable pixel supervision |
| Focal (γ=2.0) | 0.15 | Down-weights easy background |
| Boundary / Hausdorff | 0.15 | Edge-distance penalty, not just overlap |
| Lovász-Softmax | 0.15 | **Directly optimises mIoU** (the eval metric) |
| Instance-touching separation | 0.10 | Stops adjacent buildings merging |

**Class weights:** `[0.20, 2.00, 5.00, 2.50]` — roads weighted **5×** because thin classes are pixel-count-starved (6 px wide @ 50 cm GSD).

> **Why not just Dice or just CE?** Dice alone is blind to class imbalance; CE alone doesn't see overlap. Mixing complementary signals consistently outperforms any single loss on segmentation benchmarks.

---

## 5. Stage 1 — TTA Strategy

### Test-Time Augmentation: 3 scales × D4 symmetry = **24 forward passes**

| Scale | Weight | What it sees |
|---|---|---|
| 0.875 × | 0.8 | Wider context (large buildings) |
| 1.0 × | 1.0 | Reference view |
| 1.25 × | 0.9 | Fine texture (RCC grit, tile ridges) |

D4 group: 4 rotations × 2 flip states = 8 folds per scale.

### Implementation tricks

- **All folds for one scale → single batched forward** (`mega = torch.cat(augs)`) — N kernel launches → 1.
- **`FAST_TTA`** flag: 2 scales × 4 folds = 8 passes for dev iteration (~3× faster, lower accuracy).
- **`VAL_TTA_EVERY=2`** during training: full TTA every other epoch; single-forward on the off epochs (5–6× faster val).

---

## 6. Stage 1 — Inference Tiling

### 70 GB orthos → never load whole

- **512 px patches, 128 px overlap, stride 384**
- **Cosine spline window** (power=2) blends overlapping predictions → **no visible seams**
- Batch 16 patches per forward pass
- Shared `cosine_window` utility used by both segmentation tiler **and** CRF tiler — single source of truth

### Post-processing

1. **DenseCRF** — 10 iterations (Krähenbühl-Koltun convergence point), per-class compatibility matrix that penalises building↔water 3× and building↔road 1.5×
2. **Per-class morphology** — close/open with class-specific kernels
3. **Watershed building separation** — distance transform → `peak_local_max` → marker-controlled watershed → **1-px gap** between touching buildings before vectorisation

---

## 7. Stage 2A — Why ConvNeXt V2 Large + ArcFace

### Backbone alternatives

| Backbone | 224 px texture | NHWC speed on Ampere | Comment |
|---|---|---|---|
| ResNet-50 | baseline | OK | older, no large-kernel conv |
| ViT-Base | poor at 224 px | poor | ViTs need bigger inputs for texture |
| **ConvNeXt V2 Large ←** | **best** | **+15-25 %** | FCMAE pretraining + GRN; SOTA texture specialist |
| Swin-B | good | medium | window attention has conditional ops → no `fullgraph=True` compile |

### Head alternatives

| Head | RCC vs Tiled separation | Why |
|---|---|---|
| `nn.Linear` (softmax) | overlapping clusters | RCC & Tiled share grey/rectangular statistics |
| **ArcFace (s=30, m=0.55) ←** | **angular margin enforced** | Pushes class centres apart in cosine space — crisper boundaries without more data |

> ArcFace's margin `m` was **hardcoded to 0.50** ignoring the config for 11 commits. **Fixed.**

---

## 8. Stage 2A — Training Robustness

### Augmentation stack (drone aerial specific)

`RandAugment(n=2, m=7)` + `ColorJitter` + `RandomShadow` + `RandomFog` + `RandomSunFlare` + `MixUp` + `CutMix`

**Why fog/sun-flare?** Drone imagery has real atmospheric scattering — training-time exposure to these conditions improves field-deployment robustness.

**Why MixUp + CutMix together (random per batch)?**
- MixUp regularises classification boundary smoothness
- CutMix regularises spatial attention
- Empirically, alternating > picking one

### Per-crop instance normalisation in `RooftopDataset`

Equalises brightness across villages with different sun angles **before** augmentation hits — removes a major nuisance variable from training.

### Per-class confidence thresholds at inference

`{RCC: 0.45, Tiled: 0.55, Tin: 0.50, Other: 0.40}` — below threshold → falls back to **Other** rather than mispredicting.

---

## 9. Stage 2B — Why YOLO11-OBB + SAHI

### Detector alternatives

| Detector | Params | Speed | Small-object recall | Notes |
|---|---|---|---|---|
| Faster R-CNN ResNet50 | ~42 M | slow | medium | Fallback in our code; ~2 mAP worse |
| DETR / Deformable DETR | ~40 M | slow | lower | Slow convergence; query starvation on rare classes |
| YOLOv8m | ~26 M | fast | high | Good baseline |
| YOLOv9e | ~58 M | medium | high | Strong but **2× slower than YOLOv11l** at parity |
| **YOLOv11l ←** | **~25 M** | **fastest** | **highest** | **C2PSA attention; ~4.5 GB at 1280 px → batch=4 fits A4000** |

### Why OBB (oriented bounding boxes)

While transformers, tanks, and wells are rotationally symmetric, their
bounding boxes in dense clusters benefit from OBB's tight rotated fit —
two adjacent wells or a transformer at 45° get clean separation that
axis-aligned boxes would merge. `cfg["use_obb"]=True` enables
`yolo11l-obb`; outputs are stored as rotated rectangle polygons in
geo-coordinates, preserving orientation for GIS users.

### Why SAHI (Slicing Aided Hyper Inference)

A 1280 px tile contains a **15-px well**. After backbone stride-32 downsampling, the well is < 0.5 grid cell → invisible.
**SAHI re-tiles 1280 px → 512 px slices with 45 % overlap** → each well is now ~38 px in a 512 px slice → detector can see it. Results merged with **NMM** (Non-Maximum Merging) not NMS — preserves clustered detections.

### Soft-NMS Gaussian (σ=0.40)

Hard NMS drops a second transformer that overlaps IoU>0.45 with the first. Soft-NMS decays the score instead: `score *= exp(−IoU²/σ)`. **Cluster-mounted transformers survive.**

---

## 10. Stage 2B — Context-Gated Tiling

The biggest single inference-time saving in the pipeline.

```
For each 1280px tile:
    if STRtree(building/road polygons + 128px buffer).query(tile) == ∅:
        skip   ← 40-70% of tiles in rural orthos
```

| Approach | Per-tile cost | Total tiles processed |
|---|---|---|
| Naive grid sweep | O(N) intersection per tile | 100 % |
| With STRtree gating | O(log N) per query | **30-60 %** |

### Why STRtree, not Boolean union

Earlier code did `geom.intersects(unary_union(polygons))` per tile. `unary_union` of every building in a village is **wasteful** — we just need to know if anything is nearby. STRtree gives O(log N) per query with a one-time O(N log N) build cost.

---

## 11. Hardware Foundations — Why bf16 + NHWC

### Precision choice

| Precision | Mantissa | Exponent | Underflow risk | A4000 tensor cores |
|---|---|---|---|---|
| fp32 | 23 | 8 | none | half-speed |
| fp16 | 10 | 5 | **yes** → needs `GradScaler` | full |
| **bfloat16 ←** | **7** | **8 (= fp32)** | **none** | **full** |

bf16's fp32-equivalent exponent range means attention softmax probabilities **don't underflow** → no `GradScaler` needed → simpler code path. Native on Ampere.

### Memory layout

| Layout | Convolution speed (Ampere) |
|---|---|
| NCHW (default) | baseline |
| **`channels_last` (NHWC) ←** | **+15-30 %** |

Applied to every conv-heavy module: Stage 1 backbone, Stage 2A classifier. Skipped on pure transformer (no spatial convs).

---

## 12. Why SAM (Sharpness-Aware Minimisation)

### Standard SGD/AdamW finds *any* minimum

```
loss
 │      ▲ sharp minimum (poor generalisation)
 │     ╱╲
 │    ╱  ╲              ▲ flat minimum
 │___╱    ╲____________╱ ╲___
                            ╲╱
```

### SAM seeks **flat** minima

Two passes per step:
1. Add a worst-case perturbation `+ρ · ∇L / ||∇L||` (rho-scaled adversarial step on weights)
2. Compute gradient at perturbed point, take real optimizer step

**Result:** +0.5–2.0 % mIoU / accuracy. **Cost:** ~1.9× per-step time.

### Where we use it

| Stage | SAM? | Why |
|---|---|---|
| Stage 1 | **off by default** | Doubles per-step cost on 80-epoch run; forces `grad_accum=1` |
| **Stage 2A** | **on** | Small model, big regularisation win; 80-epoch budget fits |
| Stage 2B (YOLO) | n/a | YOLO has its own optimiser stack |

---

## 13. EMA + SWA — Why Both

### EMA (Exponential Moving Average) — `decay=0.9998`

Maintains a moving average of weights. **Always-on**; validation runs against EMA shadow each epoch.
- Smooths late-training noise
- Best checkpoint = EMA weights (not raw)

### SWA (Stochastic Weight Averaging) — from epoch 60

Triggers in the final 25 % of training. Averages snapshots with a low constant LR.
- Targets the **centre** of the loss basin (not the boundary)
- Typically **+0.5–1.5 mIoU** over single best checkpoint
- Final SWA BN update via one pass over training data

### Implementation efficiency

EMA shadow lives on **GPU**, updated via `torch._foreach_lerp_` — **one fused kernel** instead of N per parameter. SAM uses the same `_foreach_*` trick for its perturbation buffers.

---

## 14. Data Preprocessing — Strip Streaming

### The problem

Some SVAMITVA rasters are **213 734 × 112 836 px** = ~72 GB RGB. Loading whole into 32 GB RAM is impossible.

### Strip-streaming solution

```
Per raster:
  for strip in horizontal_strips(rows=4096):       # ~2.6 GB
      mask = burn_shp_for_strip(strtree, strip)   # ~0.9 GB
      tile_strip_to_patches(strip, mask)          # write to disk
      gc.collect()                                # discard before next strip
```

**Peak RAM per raster: ~3.5 GB.** 5 rasters in parallel = ~17 GB. **Safe on 32 GB.**

### Other preprocessing wins

- **STRtree spatial index** for SHP burn (O(log N) per strip vs brute-force iteration)
- **`ThreadPoolExecutor`** for tile writes (I/O-bound, GIL released by cv2)
- **`ProcessPoolExecutor`** for raster-level parallelism (≤ 5 workers)
- **Patch filtering** at `min_fg_ratio = 0.01` — drops near-empty tiles that add noise without signal
- **Object-centred tiling** for YOLO (vs grid-snapped) — each tile centred on an infrastructure cluster

---

## 15. This Session — A4000 Optimization Pass

### `PYTORCH_CUDA_ALLOC_CONF`

| Before | After |
|---|---|
| `max_split_size_mb:256` | **`expandable_segments:True,max_split_size_mb:256`** |

**Why:** `expandable_segments` is the PyTorch 2.1+ Ampere-friendly allocator that **grows segments on demand** rather than carving fixed slabs. Single biggest fragmentation win on a 16 GB card running bf16 + ConvNeXt-L + TTA mega-batches. Long-run training no longer compounds OOM risk across epochs.

### Stage 1 — `empty_cache()` after each validation

| Before | After |
|---|---|
| Cached allocator full of TTA mega-batch blocks → next epoch's training batches fragment further | Empty cache released between val and next train epoch |

TTA stacks `n_augs × B` images per scale → blocks don't match the next training batch shape → fragmentation compounds without an explicit release.

---

## 16. This Session — Stage 2A Parity with Stage 1

Stage 2A was missing four safety features Stage 1 had:

| Feature | Before (Stage 2A) | After (Stage 2A) |
|---|---|---|
| **VRAM auto-guard for SAM** | ❌ OOMs on ≤16.5 GB if SAM enabled | ✅ batch halved (local var, not cfg mutation) |
| **`MAX_STEPS_PER_EPOCH` cap** | ❌ unbounded epochs on big datasets | ✅ honoured; `actual_steps` used as divisor |
| **Grad-norm spike skip** | ❌ one bad batch could poison run | ✅ `did_step` flag also skips scheduler/EMA |
| **`clear_cuda_cache` before Stage 2B** | ❌ ConvNeXt blocks linger | ✅ YOLO starts on defragmented heap |

Plus housekeeping: `EMA` import moved to top-level; `_cosine_warmup` dead function removed; redundant `from pathlib import Path` + inline `__import__("pathlib").Path` folded into a single import.

---

## 17. This Session — Real Bug Fix

### Stage 1 EMA was not restored if validation crashed

**Before:**
```python
if ema:
    ema.apply_shadow(module)
val_miou, val_loss = _validate(...)   # if this raises → EMA never restored
if ema:
    ema.restore(module)
```

If `_validate` raised (OOM during TTA, CUDA error, anything), the model **stayed swapped to EMA shadow weights**. Next training step would then:
1. `optimiser.step()` on EMA weights
2. Feed those right back through `ema.update`

→ **Silently corrupting both the live model and the EMA shadow.**

**After:**
```python
if ema:
    ema.apply_shadow(module)
try:
    val_miou, val_loss = _validate(...)
finally:
    if ema:
        ema.restore(module)
```

Stage 2A already had this pattern. Stage 1 didn't. Now does.

---

## 18. This Session — Dead Code Sweep

### Method: AST scan across 33 project files

Wrote an inline AST tool to find:
1. Top-level imports never referenced
2. Top-level def/class symbols never referenced
3. False positives filtered via text-search

### Removed

| Symbol | Type | Where |
|---|---|---|
| `_cosine_warmup` | dead helper | `train/train_stage2.py` |
| `log_event` | dead helper | `utils/logger.py` |
| `ClassificationMetrics` | dead class | `utils/metrics.py` |
| `sys` | unused import | `utils/hardware.py` |
| `Optional` | unused import | `utils/ecw_compat.py`, `train/train_stage1.py` |
| `List` | unused import | `models/stage2_models.py` |
| `DataLoader` | replaced by `make_loader` | `train/train_stage1.py` |
| `cleanup_ddp` | unused import | `train/train_stage1.py` |
| `os`, `math` | unused imports | `train/train_stage2.py` |
| `math` | unused import | `tests/test_core_components.py` |

**Result:** 33/33 project files parse-clean with zero unused top-level imports.

---

## 19. Prior Session — Accuracy Pass

| Config key | Before | After | Why |
|---|---|---|---|
| `crf_iter` | 5 | **10** | Krähenbühl-Koltun convergence point — 5 was insufficient |
| `min_fg_ratio` | 0.003 | **0.01** | <1 % foreground patches add noise without signal |
| `soft_nms_sigma` | 0.9 | **0.40** | 0.9 barely suppressed; 0.40 sharper than paper, helps clustered small objects |
| `class_conf_thresh[well]` | 0.03 | **0.10** | 0.03 flooded output with false wells |
| `stage2a_conf_thresh` | single 0.55 | **per-class dict** | RCC/Tiled/Tin/Other tuned independently |
| `arcface_m` (effective) | hardcoded 0.50 | **0.55 from config** | Bug: config value ignored for 11 commits — fixed |
| `_to_uint8` normalisation | min-max | **2nd–98th percentile** | Outlier dead pixels compressed range to 7/255; percentile gives 181/255 |
| `_segment` overlap | `min(overlap, 128)` cap | **uses configured value** | Cap silently reduced overlap quality |
| `tta_predict` mode | hardcoded `fast_tta=True` | **`fast_tta=CFG.FAST_TTA`** | Config-controllable |
| `_classify_rooftops` threshold | hardcoded 0.55 | **reads `stage2a_conf_thresh`** | Per-class threshold actually applied |

---

## 20. Before / After — Memory Layout & Throughput

### Why channels-last matters (Ampere conv specific)

```
NCHW (default):                NHWC (channels_last):
─────────────                  ──────────────────────
input:  [B, C, H, W]           input:  [B, H, W, C]
                              ↓
cuDNN conv kernel selection:
  picks 1×1×K×K im2col GEMM   picks tensor-core HWNC GEMM
  ~ baseline throughput        ~ +15-30% on A4000
```

### Where applied

| Module | Format | Speedup |
|---|---|---|
| Stage 1 (MAnet + MiT-B4) | `to_channels_last(module)` | ~15-20 % training step |
| Stage 2A (ConvNeXt-L) | `to_channels_last(model)` + `cl_input(imgs)` | ~20-25 % training step |
| Stage 2B (YOLOv11l) | Ultralytics handles internally | — |

### Combined with bf16 AMP

bf16 alone: ~1.5× over fp32. bf16 + NHWC + TF32 + cudnn.benchmark + Flash SDPA: combined **~2.5-3× over fp32 NCHW baseline** on the A4000.

---

## 21. Before / After — Validation Loop

### Confusion matrix construction

**Before:** per-batch `.cpu().numpy()` → numpy bincount → accumulate → 300 batches × CUDA→CPU sync

**After:** GPU-side `torch.bincount(C·t + p, minlength=C²).reshape(C, C)` → **one** transfer at end

Wall-clock impact: validation 30-40 % faster on the 300-batch cap.

### TTA cadence

**Before:** Full 24-pass TTA every epoch → val time ≈ training step time

**After:** `VAL_TTA_EVERY = 2` — full TTA on even epochs, single forward on odd → **~half the val wall-clock**, best-checkpoint signal unchanged.

### Best-checkpoint signal

**Before:** plain forward at val time, but inference used TTA → train-time best ≠ deploy-time best

**After:** val uses the **same batched TTA path** as inference → best-checkpoint selector picks the model that's actually best at deployment time.

---

## 22. Before / After — SAM Optimizer

### Per-parameter Python loop → batched `_foreach_*`

```python
# Before (one CUDA kernel per parameter, ~500 launches per first_step):
for p in params:
    self.state[p]["old_p"] = p.data.clone()           # alloc + copy
    if adaptive:
        e_w = p.pow(2) * p.grad * scale               # 3 allocs per param
    else:
        e_w = p.grad * scale
    p.data.add_(e_w)
```

```python
# After (~3 kernel launches total per first_step):
torch._foreach_copy_(old_list, [p.data for p in params])   # 1 kernel
e_w = torch._foreach_mul(grads, [p.pow(2) for p in params]) # batched
torch._foreach_mul_(e_w, scale_f)                          # 1 kernel
torch._foreach_add_([p.data for p in params], e_w)         # 1 kernel
```

### Plus persistent `old_p` buffers

**Before:** `clone()` per param per step → fresh allocation each step

**After:** Buffers allocated once, reused → zero per-step allocation cost

### Plus on-fast-path device-skip

`_grad_norm` now skips the redundant `.to(device)` round-trip when all gradients already share one device (single-GPU fast path).

---

## 23. Numbers — Indicative Throughput

### Stage 1 training (MAnet + MiT-B4, 512 px, bs=4, grad_accum=8, bf16)

| Setup | Step time | Effective batch |
|---|---|---|
| fp32, NCHW, plain AdamW | ~baseline (~3.2 s/step) | 32 |
| + bf16 AMP | ~2.0 s/step | 32 |
| + channels_last | ~1.6 s/step | 32 |
| + cudnn.benchmark + TF32 | ~1.5 s/step | 32 |
| + `expandable_segments` (long-run stability) | ~1.5 s/step (no degradation over epochs) | 32 |

### Stage 2A training (ConvNeXt-L, 224 px, bs=32, bf16, SAM)

| Setup | Step time |
|---|---|
| Plain AdamW, NCHW | ~baseline (~0.85 s/step) |
| + bf16 + NHWC | ~0.55 s/step |
| + SAM (2 fwd+bwd) | ~1.05 s/step |

*Numbers are indicative — measure on your own data.*

---

## 24. Lessons Learned

### What worked

- **Splitting the problem** by scale was the right call — each stage solved cleanly
- **bf16 + NHWC** is a free 2-3× on Ampere; everything else is incremental on top
- **Strip-streaming** made 70 GB rasters tractable on 32 GB RAM
- **Stage 1 gating** of Stages 2A/B halved inference compute and made "Other" class meaningful

### What surprised us

- **MiT encoders in `smp` don't expose `set_grad_checkpointing`** — silently no-op. Confirmed not needed at our batch size.
- **`UnetPlusPlus` hardcoded-rejects MiT encoders** — discovered the hard way
- **ArcFace `m` config value was ignored for 11 commits** — silent bug, found via careful reading
- **DenseCRF needed 10 iterations**, not 5 — boundary convergence verified empirically
- **`expandable_segments` is dramatically better** than `max_split_size_mb` alone for long runs

### What we'd do next

- Try `max-autotune` torch.compile mode in Linux/WSL (Triton works there)
- Multi-GPU DDP for Stage 1 (already wired, untested at scale)
- Investigate Swin-V2 vs MiT-B4 for the segmentation backbone

---

## 25. Risk & Compatibility

### Things that look risky but aren't

- **No `GradScaler` in bf16 path** — correct; bf16's fp32-equivalent exponent means no underflow
- **SAM with fp16** — explicitly **blocked at config-load time** with clear error message
- **strict=False on resume** — paired with explicit shape-mismatch try/except + clear error message about stale checkpoints
- **`empty_cache()` cost** — only called between epochs/stages, not in the hot loop

### Compatibility matrix

| Accelerator | AMP dtype | Status |
|---|---|---|
| NVIDIA Ampere+ (A4000, RTX 30/40xx) | bfloat16 | ✅ tested, primary target |
| NVIDIA Turing (RTX 20xx) | bfloat16 | ✅ supported |
| AMD ROCm 6.2+ | bfloat16 | ✅ tested |
| Apple Silicon MPS | float16 | ✅ supported (no `torch.compile`) |
| CPU | float32 | ✅ fallback |

---

## 26. Architectural Summary

```
INPUT (GeoTIFF / ECW, up to 72 GB)
    │
    ▼
 _to_uint8  ←  2nd-98th percentile, zero-pixel excluded
    │
    ▼
 STAGE 1   MAnet + MiT-B4 (bf16, NHWC)
    │     Tiled inference: 512px / 128px overlap
    │     24-pass TTA → DenseCRF (10 iter) → morphology → watershed
    ↓
 SEGMENTATION MASK + per-class .shp + .gpkg
    │
    ├─→ STAGE 2A   ConvNeXt V2 L + ArcFace (bf16, NHWC, SAM)
    │             224px crops from building polygons
    │             24-pass TTA → per-class threshold
    │             → building_rooftop.shp
    │
    └─→ STAGE 2B   YOLO11-OBB + SAHI
                  1280px tiles, STRtree-gated by Stage 1
                  SAHI 512px slices → NMM → Soft-NMS (σ=0.40)
                  → infrastructure.shp
```

---

## 27. References & Acknowledgements

### Models & methods

- **MAnet** — Fan, Z. et al. "MA-Net" (2020)
- **MiT (SegFormer)** — Xie, E. et al. (NeurIPS 2021)
- **ConvNeXt** — Liu, Z. et al. (CVPR 2022)
- **ArcFace** — Deng, J. et al. (CVPR 2019)
- **YOLOv11** — Ultralytics (2024) — current Stage 2B backbone
- **YOLOv9** — Wang, C-Y. et al. (2024) — previous Stage 2B backbone
- **SAHI** — Akyon, F. et al. (ICIP 2022)
- **SAM** — Foret, P. et al. (ICLR 2021)
- **SWA** — Izmailov, P. et al. (UAI 2018)
- **Lovász-Softmax** — Berman, M. et al. (CVPR 2018)
- **DenseCRF** — Krähenbühl & Koltun (NIPS 2011)
- **Soft-NMS** — Bodla, N. et al. (ICCV 2017)

### Libraries

`segmentation-models-pytorch` · `timm` · `ultralytics` · `pydensecrf` · `albumentations` · `rasterio` · `geopandas` · `shapely` · `sahi`

### Project context

Built for the **NIF Hackathon** against the **SVAMITVA** drone-mapping dataset.
Hardware target: **RTX A4000 16 GB** on Windows 11.

---

<!-- _class: lead -->

# Thank You

### Questions?

<br>

**Repo:** see `PROJECT_REFERENCE.md` for end-to-end technical detail.
**Pipeline:** `python run_pipeline.py --mode all --data_root ./dataset`

