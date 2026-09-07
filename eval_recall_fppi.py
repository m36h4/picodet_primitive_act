import json
import os
import numpy as np
from collections import defaultdict

IOU_THRESHOLDS = [0.4, 0.5]
FPPI_LIMIT = 0.05

PRED_FILE = "output_nikon_eval/bbox.json"
GT_FILE = "dataset/labels/test.json"


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    inter = w * h
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]

    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


with open(GT_FILE) as f:
    gt_data = json.load(f)

with open(PRED_FILE) as f:
    predictions = json.load(f)

# image_id -> GT boxes
gt = defaultdict(list)

for ann in gt_data["annotations"]:
    gt[ann["image_id"]].append(ann["bbox"])

num_images = len(gt_data["images"])
num_gt = sum(len(v) for v in gt.values())

print(f"Images       : {num_images}")
print(f"GT boxes     : {num_gt}")
print(f"Predictions  : {len(predictions)}")
print()

# confidence thresholds
thresholds = np.arange(0.01, 1.00, 0.01)

for iou_threshold in IOU_THRESHOLDS:

    results = []

    for conf in thresholds:

        tp = 0
        fp = 0
        fn = 0

        preds_by_image = defaultdict(list)

        for p in predictions:
            if p["score"] >= conf:
                preds_by_image[p["image_id"]].append(p)

        for image_id in gt:

            gt_boxes = gt[image_id]

            preds = sorted(
                preds_by_image.get(image_id, []),
                key=lambda x: x["score"],
                reverse=True
            )

            matched_gt = set()

            for p in preds:

                best_iou = 0.0
                best_idx = -1

                for idx, g in enumerate(gt_boxes):

                    if idx in matched_gt:
                        continue

                    value = iou(p["bbox"], g)

                    if value > best_iou:
                        best_iou = value
                        best_idx = idx

                if best_iou >= iou_threshold:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched_gt)

        recall = tp / (tp + fn) if (tp + fn) else 0
        fppi = fp / num_images
        precision = tp / (tp + fp) if (tp + fp) else 0

        results.append(
            (conf, recall, fppi, precision, tp, fp, fn)
        )

    valid = [r for r in results if r[2] <= FPPI_LIMIT]

    if valid:
        best = max(valid, key=lambda x: x[1])

        print("=" * 60)
        print(f"IoU threshold : {iou_threshold}")
        print(f"FPPI limit    : {FPPI_LIMIT}")
        print("=" * 60)
        print(f"Confidence    : {best[0]:.2f}")
        print(f"Recall        : {best[1]:.4f} ({best[1]*100:.2f}%)")
        print(f"FPPI          : {best[2]:.5f}")
        print(f"Precision     : {best[3]:.4f}")
        print(f"TP            : {best[4]}")
        print(f"FP            : {best[5]}")
        print(f"FN            : {best[6]}")
    else:
        print(f"No threshold satisfies FPPI <= {FPPI_LIMIT}")
