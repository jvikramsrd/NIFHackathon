"""
utils/metrics.py
─────────────────
Evaluation metrics for Stage 1 (segmentation) and Stage 2A (classification).
All metrics are computed in numpy for speed; no torch dependency at eval time.
"""

from typing import Dict, List

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION METRICS
# ─────────────────────────────────────────────────────────────────────────────


class SegmentationMetrics:
    """
    Accumulates confusion-matrix statistics across batches.
    Computes per-class IoU, mean IoU, pixel accuracy, and F1.
    """

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self._conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """
        preds, targets : integer arrays of any shape (B, H, W) or (H, W)
        """
        preds = preds.ravel().astype(np.int64)
        targets = targets.ravel().astype(np.int64)
        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        idx = self.num_classes * targets + preds
        self._conf_mat += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> Dict[str, float]:
        cm = self._conf_mat.astype(np.float64)

        tp = np.diag(cm)
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp

        iou = tp / np.maximum(tp + fp + fn, 1e-6)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-6)

        pix_acc = tp.sum() / np.maximum(cm.sum(), 1e-6)

        return {
            "class_iou": iou.tolist(),
            "class_f1": f1.tolist(),
            "mean_iou": float(iou[1:].mean()),  # exclude background
            "mean_f1": float(f1[1:].mean()),
            "pixel_acc": float(pix_acc),
            "fg_pixel_acc": float(tp[1:].sum() / np.maximum(cm[1:].sum(), 1e-6)),
        }

    def summary(self) -> str:
        r = self.compute()
        lines = ["─" * 50, f"{'Class':<14} {'IoU':>8} {'F1':>8}", "─" * 50]
        for i, name in enumerate(self.class_names):
            lines.append(
                f"{name:<14} {r['class_iou'][i]:>8.3f} {r['class_f1'][i]:>8.3f}"
            )
        lines += [
            "─" * 50,
            f"{'mIoU (fg)':.<14} {r['mean_iou']:>8.3f}",
            f"{'Pixel Acc':.<14} {r['pixel_acc']:>8.3f}",
            "─" * 50,
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION METRICS
# ─────────────────────────────────────────────────────────────────────────────


class ClassificationMetrics:
    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.n = len(class_names)
        self.reset()

    def reset(self):
        self._cm = np.zeros((self.n, self.n), dtype=np.int64)

    def update(self, preds: np.ndarray, labels: np.ndarray):
        for p, l in zip(preds.ravel(), labels.ravel()):
            if 0 <= l < self.n and 0 <= p < self.n:
                self._cm[l, p] += 1

    def compute(self) -> Dict:
        cm = self._cm.astype(np.float64)
        tp = np.diag(cm)
        prec = tp / np.maximum(cm.sum(0), 1e-6)
        rec = tp / np.maximum(cm.sum(1), 1e-6)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-6)
        acc = tp.sum() / np.maximum(cm.sum(), 1e-6)

        return {
            "accuracy": float(acc),
            "class_prec": prec.tolist(),
            "class_recall": rec.tolist(),
            "class_f1": f1.tolist(),
            "macro_f1": float(f1.mean()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION METRICS  (mAP @ IoU 0.5)
# ─────────────────────────────────────────────────────────────────────────────


def compute_map(
    pred_boxes: List[List],  # [[x1,y1,x2,y2,conf,cls], ...]
    gt_boxes: List[List],  # [[x1,y1,x2,y2,cls], ...]
    num_classes: int,
    iou_thresh: float = 0.5,
) -> Dict:
    """Simple mAP@0.5 implementation for infrastructure detection evaluation."""

    ap_per_class = []

    for cls_id in range(num_classes):
        preds_c = sorted(
            [p for p in pred_boxes if int(p[5]) == cls_id],
            key=lambda x: -x[4],  # descending confidence
        )
        gts_c = [g for g in gt_boxes if int(g[4]) == cls_id]

        if len(gts_c) == 0:
            ap_per_class.append(0.0)
            continue

        tp = np.zeros(len(preds_c))
        fp = np.zeros(len(preds_c))
        matched = set()

        for i, pred in enumerate(preds_c):
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts_c):
                if j in matched:
                    continue
                iou = _box_iou(pred[:4], gt[:4])
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= iou_thresh:
                tp[i] = 1
                matched.add(best_j)
            else:
                fp[i] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec = cum_tp / max(len(gts_c), 1)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
        ap_per_class.append(_voc_ap(rec, prec))

    return {
        "mAP_50": float(np.mean(ap_per_class)),
        "class_ap": ap_per_class,
    }


def _box_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-6)


def _voc_ap(rec, prec):
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def compute_map(
    pred_boxes: List[List],  # [[x1,y1,x2,y2,conf,cls], ...]
    gt_boxes: List[List],  # [[x1,y1,x2,y2,cls], ...]
    num_classes: int,
    iou_thresholds: List[float] = None,  # List of IoU thresholds to test
) -> Dict[str, float]:
    """
    Computes Average Precision (AP) and Mean Average Precision (mAP)
    across multiple IoU thresholds (COCO style).
    """

    if iou_thresholds is None:
        # Default to COCO standard: 0.5 to 0.95 in steps of 0.05
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    ap_per_class = []

    for cls_id in range(num_classes):
        preds_c = sorted(
            [p for p in pred_boxes if int(p[5]) == cls_id],
            key=lambda x: -x[4],  # descending confidence
        )
        gts_c = [g for g in gt_boxes if int(g[4]) == cls_id]

        if len(gts_c) == 0:
            ap_per_class.append(0.0)
            continue

        aps_for_class = []
        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(preds_c))
            fp = np.zeros(len(preds_c))
            matched = set()

            for i, pred in enumerate(preds_c):
                best_iou, best_j = 0.0, -1
                for j, gt in enumerate(gts_c):
                    if j in matched:
                        continue
                    # Pass the current threshold to the IoU calculation
                    iou = _box_iou(pred[:4], gt[:4], threshold=iou_thresh)
                    if iou > best_iou:
                        best_iou, best_j = iou, j

                if best_iou >= iou_thresh:
                    tp[i] = 1
                    matched.add(best_j)
                else:
                    fp[i] = 1

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / max(len(gts_c), 1)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-6)
            ap = _voc_ap(rec, prec)
            aps_for_class.append(ap)

        # Store the mean AP across all tested IoU thresholds for this class
        ap_per_class.append(np.mean(aps_for_class))

    # Mean Average Precision (mAP) across all classes
    mean_ap = np.mean(ap_per_class)

    return {
        "mAP_50": float(
            np.mean(
                np.array(
                    [_voc_ap(np.cumsum(tp), np.cumsum(fp)) for _ in range(len(gts_c))]
                )
            )
        ),  # Placeholder: Needs proper full mAP calculation structure if not using class_ap list
        "mAP": float(mean_ap),
        "class_ap": ap_per_class,
    }
