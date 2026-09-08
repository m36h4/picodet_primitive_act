"""
Compute Recall @ FPPI (False Positives Per Image) at a fixed IoU threshold,
from COCO-format ground truth + predictions (e.g. PaddleDetection's bbox.json).

This is a standard metric for single/few-class detectors (esp. small-object
detectors like ball detection) where you care about "how many true objects
do I catch, if I only tolerate N false positives per image on average" --
distinct from COCO's own AP metric, which averages over the full
precision-recall curve and doesn't let you pick an explicit operating point.

Definitions used here:
  - IoU threshold: a prediction counts as a match to a GT box only if
    IoU(pred, gt) >= iou_thresh (default 0.4, as requested).
  - Matching: greedy, highest-confidence-first, one prediction can match
    at most one GT box and vice versa (standard COCO-style matching).
  - FPPI curve: sweep the confidence threshold from high to low. At each
    threshold, compute total false positives / total images = FPPI, and
    recall = matched GT / total GT.
  - Recall @ FPPI=X: recall at the confidence threshold where FPPI is
    closest to (and does not exceed, by default) X.

Usage:
  python3 recall_at_fppi.py \
      --gt instances_eval.json \
      --pred bbox.json \
      --iou 0.4 \
      --fppi 0.05

Requires: pycocotools (pip install pycocotools)
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO


def compute_iou(box_a, box_b):
    """boxes in [x, y, w, h] format (COCO convention)."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Path to ground-truth COCO json")
    ap.add_argument("--pred", required=True, help="Path to predictions (COCO-format bbox.json)")
    ap.add_argument("--iou", type=float, default=0.4, help="IoU threshold for a match")
    ap.add_argument("--fppi", type=float, default=0.05,
                     help="Target false-positives-per-image operating point")
    ap.add_argument("--category-id", type=int, default=None,
                     help="Restrict to one category id (default: all categories combined, "
                          "correct for single-class 'ball' setups)")
    args = ap.parse_args()

    coco_gt = COCO(args.gt)
    with open(args.pred) as f:
        predictions = json.load(f)

    if args.category_id is not None:
        predictions = [p for p in predictions if p["category_id"] == args.category_id]
        gt_img_ids = coco_gt.getImgIds()
        gt_ann_ids = coco_gt.getAnnIds(imgIds=gt_img_ids, catIds=[args.category_id])
    else:
        gt_img_ids = coco_gt.getImgIds()
        gt_ann_ids = coco_gt.getAnnIds(imgIds=gt_img_ids)

    n_images = len(gt_img_ids)
    gt_anns = coco_gt.loadAnns(gt_ann_ids)

    # Group GT boxes by image
    gt_by_image = defaultdict(list)
    for ann in gt_anns:
        if ann.get("iscrowd", 0) == 1:
            continue
        gt_by_image[ann["image_id"]].append(ann["bbox"])
    total_gt = sum(len(v) for v in gt_by_image.values())

    print(f"Images: {n_images}")
    print(f"Ground-truth boxes: {total_gt}")
    print(f"Predictions (all confidences): {len(predictions)}")

    if total_gt == 0:
        raise SystemExit("No ground-truth boxes found -- check --gt path / --category-id.")
    if len(predictions) == 0:
        raise SystemExit("No predictions found -- check --pred path / --category-id.")

    # Sort all predictions by confidence, descending
    predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)

    # For each image, track which GT boxes have already been matched.
    # Processing predictions once in descending score order is equivalent
    # to sweeping the confidence threshold from 1.0 down to 0.0.
    matched_gt = {img_id: [False] * len(boxes) for img_id, boxes in gt_by_image.items()}

    n_tp = 0
    n_fp = 0
    curve = []  # (score, cumulative_recall, cumulative_fppi)

    for pred in predictions:
        img_id = pred["image_id"]
        pred_box = pred["bbox"]
        score = pred["score"]

        gt_boxes = gt_by_image.get(img_id, [])
        best_iou = 0.0
        best_idx = -1
        for i, gt_box in enumerate(gt_boxes):
            if matched_gt[img_id][i]:
                continue  # already matched to a higher-confidence prediction
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_iou >= args.iou and best_idx >= 0:
            matched_gt[img_id][best_idx] = True
            n_tp += 1
        else:
            n_fp += 1

        recall = n_tp / total_gt
        fppi = n_fp / n_images
        curve.append((score, recall, fppi))

    # Find the highest-recall point on the curve with fppi <= target.
    # fppi is monotonically non-decreasing as the threshold is lowered,
    # so the last point satisfying the constraint is the best one.
    best_point = None
    for score, recall, fppi in curve:
        if fppi <= args.fppi:
            best_point = (score, recall, fppi)
        else:
            break

    print(f"\n=== Recall @ FPPI={args.fppi}, IoU={args.iou} ===")
    if best_point is None:
        print(
            f"No confidence threshold achieves FPPI <= {args.fppi} "
            f"(even the single highest-confidence prediction already exceeds it). "
            f"Lowest achieved FPPI: {curve[0][2]:.4f} at score={curve[0][0]:.4f}, "
            f"recall at that point: {curve[0][1]:.4f}"
        )
    else:
        score, recall, fppi = best_point
        print(f"Confidence threshold: {score:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Achieved FPPI: {fppi:.4f} (target was {args.fppi})")

    print("\n=== Recall at other reference FPPI points (for context) ===")
    for target in [0.01, 0.05, 0.1, 0.5, 1.0]:
        point = None
        for score, recall, fppi in curve:
            if fppi <= target:
                point = (score, recall, fppi)
            else:
                break
        if point:
            print(f"  FPPI<={target:<5} -> recall={point[1]:.4f}  (score>={point[0]:.4f}, actual fppi={point[2]:.4f})")
        else:
            print(f"  FPPI<={target:<5} -> not achievable (min fppi = {curve[0][2]:.4f})")


if __name__ == "__main__":
    main()
