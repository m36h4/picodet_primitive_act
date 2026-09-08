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


# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------

with open(GT_FILE) as f:
    gt_data = json.load(f)

with open(PRED_FILE) as f:
    predictions = json.load(f)


# ---------------------------------------------------------
# Build GT dictionary
# ---------------------------------------------------------

gt = defaultdict(list)

for ann in gt_data["annotations"]:
    gt[ann["image_id"]].append(ann["bbox"])


# ---------------------------------------------------------
# IMPORTANT:
# PaddleDetection prediction image_id is the inference
# order: 0, 1, 2, ...
#
# val.json has its own COCO image IDs.
# Map prediction index -> actual COCO image ID
# using the SAME JSON image order.
# ---------------------------------------------------------

ordered_gt_ids = [
    img["id"]
    for img in gt_data["images"]
]


# ---------------------------------------------------------
# Information
# ---------------------------------------------------------

print("Images       :", len(gt_data["images"]))
print("GT boxes     :", len(gt_data["annotations"]))
print("Predictions  :", len(predictions))

gt_ids = sorted(ordered_gt_ids)

pred_ids_before = sorted(
    set(p["image_id"] for p in predictions)
)

print("GT ID range  :", min(gt_ids), "to", max(gt_ids))
print(
    "Pred ID range:",
    min(pred_ids_before),
    "to",
    max(pred_ids_before)
)


# ---------------------------------------------------------
# Map prediction image IDs
# ---------------------------------------------------------

mapped_predictions = []

invalid_prediction_ids = 0

for p in predictions:

    pred_index = int(p["image_id"])

    if pred_index < 0 or pred_index >= len(ordered_gt_ids):
        invalid_prediction_ids += 1
        continue

    new_p = p.copy()

    new_p["image_id"] = ordered_gt_ids[pred_index]

    mapped_predictions.append(new_p)


predictions = mapped_predictions


print()
print(
    "Mapping prediction image_id -> GT image_id:"
)
print(
    "JSON-order mapping: prediction index -> val.json image ID"
)

if invalid_prediction_ids > 0:
    print(
        "Invalid prediction IDs skipped:",
        invalid_prediction_ids
    )

print()


# ---------------------------------------------------------
# Verify mapping with first few predictions
# ---------------------------------------------------------

print("Mapping verification:")

for p in predictions[:5]:

    print(
        "prediction image_id -> GT image_id:",
        p["image_id"]
    )

print()


# ---------------------------------------------------------
# Nikon metric
# Recall @ FPPI <= 0.05
# ---------------------------------------------------------

for iou_threshold in IOU_THRESHOLDS:

    best = None

    for conf in np.arange(0.01, 1.00, 0.01):

        preds_by_image = defaultdict(list)

        for p in predictions:

            if p["score"] >= conf:

                preds_by_image[
                    p["image_id"]
                ].append(p)


        tp = 0
        fp = 0
        fn = 0


        # -------------------------------------------------
        # Evaluate every GT image
        # -------------------------------------------------

        for image_id, gt_boxes in gt.items():

            preds = sorted(
                preds_by_image.get(
                    image_id,
                    []
                ),
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

                    v = iou(
                        p["bbox"],
                        gt_box
                    )

                    if v > best_iou:

                        best_iou = v
                        best_gt = i


                if best_iou >= iou_threshold:

                    tp += 1
                    matched.add(best_gt)

                else:

                    fp += 1


            fn += (
                len(gt_boxes)
                - len(matched)
            )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        recall = (
            tp / (tp + fn)
            if (tp + fn)
            else 0
        )

        fppi = (
            fp / len(gt_data["images"])
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp)
            else 0
        )


        # -------------------------------------------------
        # Nikon constraint
        # -------------------------------------------------

        if fppi <= FPPI_LIMIT:

            result = (
                recall,
                fppi,
                precision,
                conf,
                tp,
                fp,
                fn
            )

            if (
                best is None
                or recall > best[0]
            ):

                best = result


    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    print("=" * 60)
    print("IoU threshold :", iou_threshold)
    print("FPPI limit    :", FPPI_LIMIT)
    print("=" * 60)


    if best:

        (
            recall,
            fppi,
            precision,
            conf,
            tp,
            fp,
            fn
        ) = best

        print(
            f"Confidence    : {conf:.2f}"
        )

        print(
            f"Recall        : {recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

        print(
            f"FPPI          : {fppi:.5f}"
        )

        print(
            f"Precision     : {precision:.4f}"
        )

        print(
            f"TP            : {tp}"
        )

        print(
            f"FP            : {fp}"
        )

        print(
            f"FN            : {fn}"
        )

    else:

        print(
            "No threshold satisfies "
            "FPPI <= 0.05"
        )

    print()
