import json
import numpy as np
from collections import defaultdict

PRED_FILE = "output_nikon_eval_merged/bbox.json"
GT_FILE = "dataset/labels/val.json"

IOU_THRESHOLDS = [0.4, 0.5]
FPPI_LIMIT = 0.05


def iou(a, b):
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]

    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter

    return inter / union if union > 0 else 0


with open(GT_FILE) as f:
    gt_data = json.load(f)

with open(PRED_FILE) as f:
    predictions = json.load(f)

gt = defaultdict(list)

for ann in gt_data["annotations"]:
    gt[ann["image_id"]].append(ann["bbox"])

gt_ids = sorted(im["id"] for im in gt_data["images"])

print("Images      :", len(gt_data["images"]))
print("GT boxes    :", len(gt_data["annotations"]))
print("Predictions :", len(predictions))

# PaddleDetection infer.py uses 0-based image IDs.
# Your COCO GT uses 1-based IDs.
pred_ids = sorted(set(p["image_id"] for p in predictions))

print("GT ID range :", min(gt_ids), "to", max(gt_ids))
print("Pred ID range:", min(pred_ids), "to", max(pred_ids))

if min(pred_ids) == 0 and min(gt_ids) == 1:
    print("\nMapping prediction image_id -> GT image_id: +1")
    for p in predictions:
        p["image_id"] += 1

print()


for iou_threshold in IOU_THRESHOLDS:

    best = None

    for conf in np.arange(0.01, 1.00, 0.01):

        preds_by_image = defaultdict(list)

        for p in predictions:
            if p["score"] >= conf:
                preds_by_image[p["image_id"]].append(p)

        tp = 0
        fp = 0
        fn = 0

        for image_id, gt_boxes in gt.items():

            preds = sorted(
                preds_by_image.get(image_id, []),
                key=lambda x: x["score"],
                reverse=True
            )

            matched = set()

            for p in preds:

                best_iou = 0
                best_gt = -1

                for i, gt_box in enumerate(gt_boxes):

                    if i in matched:
                        continue

                    v = iou(p["bbox"], gt_box)

                    if v > best_iou:
                        best_iou = v
                        best_gt = i

                if best_iou >= iou_threshold:
                    tp += 1
                    matched.add(best_gt)
                else:
                    fp += 1

            fn += len(gt_boxes) - len(matched)

        recall = tp / (tp + fn) if (tp + fn) else 0
        fppi = fp / len(gt_data["images"])
        precision = tp / (tp + fp) if (tp + fp) else 0

        if fppi <= FPPI_LIMIT:
            result = (recall, fppi, precision, conf, tp, fp, fn)

            if best is None or recall > best[0]:
                best = result

    print("=" * 60)
    print("IoU threshold :", iou_threshold)
    print("FPPI limit    :", FPPI_LIMIT)
    print("=" * 60)

    if best:
        recall, fppi, precision, conf, tp, fp, fn = best

        print(f"Confidence    : {conf:.2f}")
        print(f"Recall        : {recall:.4f} ({recall*100:.2f}%)")
        print(f"FPPI          : {fppi:.5f}")
        print(f"Precision     : {precision:.4f}")
        print(f"TP            : {tp}")
        print(f"FP            : {fp}")
        print(f"FN            : {fn}")
    else:
        print("No threshold satisfies FPPI <= 0.05")
