"""
utils/benchmark.py
──────────────────────────────────────────────────────────────────────────────
Comprehensive accuracy benchmarking for the GeoIntel SVAMITVA pipeline.

Provides metric classes and report generators that go beyond simple mIoU /
pixel accuracy to measure what matters for GIS output quality:

  BoundaryF1Score       — F-score at 1/2/5px distance tolerance
  PolygonIoU            — polygon-level IoU (not pixel-level)
  RooftopPRCurves       — per-class precision-recall for Stage 2A
  InfraLocalizationAcc  — point-distance accuracy for Stage 2B
  generate_accuracy_report — full HTML accuracy summary

All metrics are computed in NumPy and are framework-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BOUNDARY F1 SCORE
# ─────────────────────────────────────────────────────────────────────────────


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """
    Binary dilation with a disk structuring element of given radius.
    Uses cv2 for speed; falls back to scipy if cv2 is unavailable.
    """
    try:
        import cv2
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    except ImportError:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        struct = generate_binary_structure(2, 1)
        result = mask.astype(bool)
        for _ in range(radius):
            result = binary_dilation(result, structure=struct)
        return result


def _boundary_mask(seg: np.ndarray) -> np.ndarray:
    """Extract binary boundary map from an integer segmentation mask."""
    import cv2
    # Gradient-based boundary: non-zero pixels are on a class boundary
    gx = cv2.Sobel(seg.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(seg.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return (np.abs(gx) + np.abs(gy)) > 0


class BoundaryF1Score:
    """
    Boundary F-measure (F-score at pixel distance tolerance d).

    Computes boundary precision/recall/F1 at three distance tolerances:
    1px, 2px, and 5px. The 1px score is strict; 5px is forgiving.

    For GIS output, the 2px score (≈ 10 cm at 5cm GSD) is the most
    practically meaningful: it measures whether polygon edges align with
    the visual boundaries in the orthophoto within about one roof tile width.

    Usage:
        scorer = BoundaryF1Score()
        scorer.update(pred_mask, gt_mask)
        results = scorer.compute()
        # {'f1_1px': 0.82, 'f1_2px': 0.89, 'f1_5px': 0.94, ...}
    """

    def __init__(self, tolerances: Tuple[int, ...] = (1, 2, 5)):
        self.tolerances = tolerances
        self._stats: Dict[int, Dict[str, float]] = {t: {"tp": 0.0, "pred": 0.0, "gt": 0.0}
                                                      for t in tolerances}

    def reset(self):
        for t in self.tolerances:
            self._stats[t] = {"tp": 0.0, "pred": 0.0, "gt": 0.0}

    def update(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """
        pred_mask, gt_mask: (H, W) integer class arrays.
        """
        pred_b = _boundary_mask(pred_mask)
        gt_b = _boundary_mask(gt_mask)

        for t in self.tolerances:
            # Precision: fraction of predicted boundary pixels within t px of GT boundary
            gt_dilated = _dilate_mask(gt_b, t)
            tp_prec = float((pred_b & gt_dilated).sum())

            # Recall: fraction of GT boundary pixels within t px of predicted boundary
            pred_dilated = _dilate_mask(pred_b, t)
            tp_rec = float((gt_b & pred_dilated).sum())

            self._stats[t]["tp"] += (tp_prec + tp_rec) / 2.0
            self._stats[t]["pred"] += float(pred_b.sum())
            self._stats[t]["gt"] += float(gt_b.sum())

    def compute(self) -> Dict[str, float]:
        out = {}
        for t in self.tolerances:
            s = self._stats[t]
            prec = s["tp"] / max(s["pred"], 1e-6)
            rec = s["tp"] / max(s["gt"], 1e-6)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)
            out[f"precision_{t}px"] = float(prec)
            out[f"recall_{t}px"] = float(rec)
            out[f"f1_{t}px"] = float(f1)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  POLYGON-LEVEL IoU
# ─────────────────────────────────────────────────────────────────────────────


def compute_polygon_iou(
    pred_polygons: list,
    gt_polygons: list,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute polygon-level precision, recall, and F1 at a given IoU threshold.

    A predicted polygon is a TP if it matches a GT polygon with IoU ≥ threshold.
    Uses STRtree for efficient candidate retrieval (O(n log n) vs O(n²)).

    Args:
        pred_polygons: List of Shapely Polygon objects (predicted buildings).
        gt_polygons:   List of Shapely Polygon objects (ground truth buildings).
        iou_threshold: IoU threshold for a match to count as TP.

    Returns:
        Dict with keys: precision, recall, f1, tp, fp, fn, n_pred, n_gt
    """
    try:
        from shapely.strtree import STRtree
    except ImportError:
        log.warning("shapely not available; polygon IoU skipped")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not pred_polygons or not gt_polygons:
        return {
            "precision": float(not pred_polygons),
            "recall": float(not gt_polygons),
            "f1": 0.0,
            "tp": 0, "fp": len(pred_polygons), "fn": len(gt_polygons),
            "n_pred": len(pred_polygons), "n_gt": len(gt_polygons),
        }

    tree = STRtree(gt_polygons)
    matched_gt: set = set()
    tp = 0

    for pred in pred_polygons:
        if pred is None or pred.is_empty:
            continue
        candidates = tree.query(pred)
        best_iou = 0.0
        best_j = -1
        for j in candidates:
            gt = gt_polygons[j]
            if j in matched_gt or gt is None or gt.is_empty:
                continue
            try:
                inter = pred.intersection(gt).area
                union = pred.union(gt).area
                iou = inter / max(union, 1e-6)
            except Exception:
                iou = 0.0
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            tp += 1
            matched_gt.add(best_j)

    fp = len(pred_polygons) - tp
    fn = len(gt_polygons) - tp
    precision = tp / max(tp + fp, 1e-6)
    recall = tp / max(tp + fn, 1e-6)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp, "fp": fp, "fn": fn,
        "n_pred": len(pred_polygons),
        "n_gt": len(gt_polygons),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  ROOFTOP CLASSIFICATION PR CURVES
# ─────────────────────────────────────────────────────────────────────────────


class RooftopPRCurves:
    """
    Per-class precision-recall curves and AUC for Stage 2A classification.

    Accumulates (confidence, label) pairs across all buildings processed,
    then computes PR curves and AUC-PR for each class.

    Usage:
        tracker = RooftopPRCurves(class_names=['RCC', 'Tiled', 'Tin', 'Other'])
        # For each batch:
        tracker.update(probs, true_labels)  # probs (N, 4), true_labels (N,)
        curves = tracker.compute()
        # {'RCC': {'auc_pr': 0.97, 'ap': 0.96}, 'Tiled': {...}, ...}
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self._scores: List[np.ndarray] = []   # (N, C) probs
        self._labels: List[np.ndarray] = []   # (N,) int labels

    def reset(self):
        self._scores.clear()
        self._labels.clear()

    def update(self, probs: np.ndarray, labels: np.ndarray):
        """probs: (N, C) float, labels: (N,) int."""
        self._scores.append(np.asarray(probs, dtype=np.float32))
        self._labels.append(np.asarray(labels, dtype=np.int64))

    def compute(self) -> Dict[str, Dict[str, float]]:
        if not self._scores:
            return {}
        all_scores = np.concatenate(self._scores, axis=0)  # (N, C)
        all_labels = np.concatenate(self._labels, axis=0)  # (N,)

        results = {}
        for c, name in enumerate(self.class_names):
            binary_gt = (all_labels == c).astype(np.float32)
            conf = all_scores[:, c]
            if binary_gt.sum() == 0:
                results[name] = {"auc_pr": 0.0, "ap": 0.0, "n_pos": 0}
                continue
            # Sort by descending confidence
            order = np.argsort(-conf)
            tp_cum = np.cumsum(binary_gt[order])
            fp_cum = np.cumsum(1 - binary_gt[order])
            n_pos = float(binary_gt.sum())
            precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
            recall = tp_cum / n_pos
            # AP = area under PR curve (trapezoidal)
            ap = float(np.trapz(precision[::-1], recall[::-1]))
            results[name] = {
                "auc_pr": ap,
                "ap": ap,
                "n_pos": int(n_pos),
                "n_total": len(all_labels),
            }
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INFRASTRUCTURE LOCALIZATION ACCURACY
# ─────────────────────────────────────────────────────────────────────────────


def compute_infra_localization_accuracy(
    pred_centroids: List[Tuple[float, float]],   # [(x, y), ...] in pixel coords
    gt_centroids: List[Tuple[float, float]],     # [(x, y), ...] in pixel coords
    pixel_to_meter: float = 0.05,               # meters per pixel (5cm GSD)
    distance_thresholds_m: Tuple[float, ...] = (1.0, 2.0, 5.0),
) -> Dict[str, float]:
    """
    Evaluate infrastructure detection localization accuracy.

    For each GT point, find the nearest predicted point. Compute:
      - Mean centroid error (meters)
      - % GT points localized within d meters (for each threshold)
      - % GT points with no prediction within 5m (miss rate)

    Args:
        pred_centroids: List of (x, y) pixel coordinates of predicted objects.
        gt_centroids:   List of (x, y) pixel coordinates of ground truth objects.
        pixel_to_meter: GSD in meters per pixel. Default 0.05 (5cm SVAMITVA).
        distance_thresholds_m: Distance tolerances for recall computation.

    Returns:
        Dict with keys: mean_error_m, median_error_m, miss_rate, recall@Xm
    """
    if not gt_centroids:
        return {"mean_error_m": 0.0, "median_error_m": 0.0, "miss_rate": 0.0}
    if not pred_centroids:
        return {
            "mean_error_m": float("inf"),
            "median_error_m": float("inf"),
            "miss_rate": 1.0,
            **{f"recall@{d}m": 0.0 for d in distance_thresholds_m},
        }

    pred_arr = np.array(pred_centroids, dtype=np.float32)
    gt_arr = np.array(gt_centroids, dtype=np.float32)

    # Pairwise distances (GT x Pred)
    diff = gt_arr[:, None, :] - pred_arr[None, :, :]  # (n_gt, n_pred, 2)
    dists_px = np.sqrt((diff ** 2).sum(axis=-1))        # (n_gt, n_pred)
    dists_m = dists_px * pixel_to_meter

    # For each GT point, find nearest prediction
    min_dists_m = dists_m.min(axis=1)  # (n_gt,)

    results: Dict[str, float] = {
        "mean_error_m": float(min_dists_m.mean()),
        "median_error_m": float(np.median(min_dists_m)),
        "miss_rate": float((min_dists_m > 5.0).mean()),
        "n_gt": len(gt_centroids),
        "n_pred": len(pred_centroids),
    }
    for d in distance_thresholds_m:
        results[f"recall@{d}m"] = float((min_dists_m <= d).mean())

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  HTML ACCURACY REPORT
# ─────────────────────────────────────────────────────────────────────────────


def generate_accuracy_report(
    seg_metrics: Optional[Dict] = None,
    boundary_metrics: Optional[Dict] = None,
    polygon_metrics: Optional[Dict] = None,
    clf_metrics: Optional[Dict] = None,
    pr_curves: Optional[Dict] = None,
    det_metrics: Optional[Dict] = None,
    infra_loc: Optional[Dict] = None,
    output_path: Optional[str] = None,
    village_name: str = "Village",
) -> str:
    """
    Generate a comprehensive HTML accuracy report for the full pipeline.

    Args:
        seg_metrics:    Output of SegmentationMetrics.compute()
        boundary_metrics: Output of BoundaryF1Score.compute()
        polygon_metrics:  Output of compute_polygon_iou()
        clf_metrics:    Stage 2A per-class metrics dict
        pr_curves:      Output of RooftopPRCurves.compute()
        det_metrics:    Stage 2B mAP metrics
        infra_loc:      Output of compute_infra_localization_accuracy()
        output_path:    Optional path to save the HTML file
        village_name:   Name for report title

    Returns:
        HTML string.
    """

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    def _status(v: float, target: float = 0.95) -> str:
        color = "#22c55e" if v >= target else "#ef4444"
        icon = "✓" if v >= target else "✗"
        return f'<span style="color:{color};font-weight:bold">{icon} {_pct(v)}</span>'

    sections = []

    # ── Section 1: Stage 1 Segmentation ─────────────────────────────────────
    if seg_metrics:
        rows = ""
        class_names = ["Background", "Building", "Road", "Waterbody"]
        for i, cls in enumerate(class_names):
            iou = seg_metrics.get("class_iou", [0, 0, 0, 0])[i] if i < len(seg_metrics.get("class_iou", [])) else 0
            f1 = seg_metrics.get("class_f1", [0, 0, 0, 0])[i] if i < len(seg_metrics.get("class_f1", [])) else 0
            rows += f"<tr><td>{cls}</td><td>{_pct(iou)}</td><td>{_pct(f1)}</td></tr>"
        miou = seg_metrics.get("mean_iou", 0.0)
        sections.append(f"""
        <h2>Stage 1 — Semantic Segmentation</h2>
        <p>Mean IoU (foreground): {_status(miou)} &nbsp;|&nbsp; Pixel Acc: {_pct(seg_metrics.get("pixel_acc", 0))}</p>
        <table border="1" cellpadding="6" style="border-collapse:collapse">
          <tr><th>Class</th><th>IoU</th><th>F1</th></tr>
          {rows}
        </table>
        """)
        if boundary_metrics:
            sections.append(f"""
        <h3>Boundary Quality</h3>
        <ul>
          <li>F1 @ 1px: {_pct(boundary_metrics.get("f1_1px", 0))}</li>
          <li>F1 @ 2px (≈10cm): {_pct(boundary_metrics.get("f1_2px", 0))}</li>
          <li>F1 @ 5px (≈25cm): {_pct(boundary_metrics.get("f1_5px", 0))}</li>
        </ul>
        """)
        if polygon_metrics:
            sections.append(f"""
        <h3>Polygon-Level Building Accuracy (IoU@0.5)</h3>
        <ul>
          <li>Precision: {_pct(polygon_metrics.get("precision", 0))}</li>
          <li>Recall: {_pct(polygon_metrics.get("recall", 0))}</li>
          <li>F1: {_pct(polygon_metrics.get("f1", 0))}</li>
          <li>TP: {polygon_metrics.get("tp", "?")} / FP: {polygon_metrics.get("fp", "?")} / FN: {polygon_metrics.get("fn", "?")}</li>
        </ul>
        """)

    # ── Section 2: Stage 2A Rooftop Classification ───────────────────────────
    if clf_metrics or pr_curves:
        rows = ""
        for cls_name, m in (clf_metrics or {}).items():
            acc = m.get("accuracy", m.get("f1", 0))
            n = m.get("n_pos", "?")
            ap = (pr_curves or {}).get(cls_name, {}).get("ap", None)
            ap_str = _pct(ap) if ap is not None else "—"
            rows += f"<tr><td>{cls_name}</td><td>{_status(acc, 0.92)}</td><td>{ap_str}</td><td>{n}</td></tr>"
        sections.append(f"""
        <h2>Stage 2A — Rooftop Classification</h2>
        <table border="1" cellpadding="6" style="border-collapse:collapse">
          <tr><th>Class</th><th>Accuracy/F1</th><th>AP</th><th>Count</th></tr>
          {rows}
        </table>
        """)

    # ── Section 3: Stage 2B Infrastructure Detection ─────────────────────────
    if det_metrics or infra_loc:
        det_html = ""
        if det_metrics:
            det_html = f"""
            <p>mAP@0.5: {_status(det_metrics.get("mAP_50", 0), 0.85)}
               &nbsp;|&nbsp; mAP@0.5:0.95: {_pct(det_metrics.get("mAP", 0))}</p>
            """
        loc_html = ""
        if infra_loc:
            loc_html = f"""
            <h3>Localization Accuracy</h3>
            <ul>
              <li>Mean error: {infra_loc.get("mean_error_m", "?"):.2f} m</li>
              <li>Median error: {infra_loc.get("median_error_m", "?"):.2f} m</li>
              <li>Recall@1m: {_pct(infra_loc.get("recall@1.0m", 0))}</li>
              <li>Recall@5m: {_pct(infra_loc.get("recall@5.0m", 0))}</li>
              <li>Miss rate (>5m): {_pct(infra_loc.get("miss_rate", 1.0))}</li>
            </ul>
            """
        sections.append(f"""
        <h2>Stage 2B — Infrastructure Detection</h2>
        {det_html}{loc_html}
        """)

    body = "\n".join(sections) if sections else "<p>No metrics provided.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>GeoIntel Accuracy Report — {village_name}</title>
  <style>
    body {{font-family: 'Segoe UI', sans-serif; max-width: 900px; margin: 40px auto; color: #1f1f1f;}}
    h1 {{color: #005FB8;}} h2 {{color: #1a3a5c; border-bottom: 2px solid #005FB8; padding-bottom: 6px;}}
    table {{margin: 12px 0;}} th {{background: #e8f0fe; padding: 8px 12px;}}
    td {{padding: 6px 12px;}} tr:nth-child(even) {{background: #f5f5f5;}}
    .target-note {{font-size: 0.85em; color: #666; margin-top: 4px;}}
  </style>
</head>
<body>
  <h1>GeoIntel SVAMITVA Accuracy Report</h1>
  <p><strong>Village:</strong> {village_name}</p>
  <p class="target-note">Target: mIoU ≥ 95%, Rooftop Acc ≥ 95%, mAP@0.5 ≥ 85%</p>
  {body}
</body>
</html>
"""
    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
        log.info("Accuracy report saved: %s", output_path)

    return html
