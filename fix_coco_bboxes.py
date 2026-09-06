import argparse
import json
from pathlib import Path
from PIL import Image


DATASET_DIR = Path("dataset")

SPLITS = {
    "train": {
        "json": DATASET_DIR / "labels/train.json",
        "images": DATASET_DIR / "images/train",
    },
    "val": {
        "json": DATASET_DIR / "labels/val.json",
        "images": DATASET_DIR / "images/val",
    },
    "test": {
        "json": DATASET_DIR / "labels/test.json",
        "images": DATASET_DIR / "images/test",
    },
}


def process_split(split_name, cfg, dry_run):

    json_path = cfg["json"]
    image_dir = cfg["images"]

    print("\n" + "=" * 80)
    print(f"PROCESSING: {split_name.upper()}")
    print("=" * 80)

    if not json_path.exists():
        print(f"[ERROR] JSON not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {
        img["id"]: img
        for img in coco.get("images", [])
    }

    changed = 0
    removed = 0
    valid = 0
    missing = 0
    errors = 0

    new_annotations = []

    for ann in coco.get("annotations", []):

        img = images.get(ann["image_id"])

        if img is None:
            print(
                f"[ERROR] Annotation {ann['id']} "
                f"references missing image_id {ann['image_id']}"
            )
            errors += 1
            new_annotations.append(ann)
            continue

        filename = Path(img["file_name"]).name
        image_path = image_dir / filename

        if not image_path.exists():
            print(f"[MISSING] {image_path}")
            missing += 1
            new_annotations.append(ann)
            continue

        try:
            with Image.open(image_path) as im:
                image_w, image_h = im.size
        except Exception as e:
            print(f"[ERROR] {image_path}: {e}")
            errors += 1
            new_annotations.append(ann)
            continue

        x, y, w, h = ann["bbox"]

        # Original box boundaries
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        # Check whether bbox is outside image
        invalid = (
            x1 < 0
            or y1 < 0
            or x2 > image_w
            or y2 > image_h
            or w <= 0
            or h <= 0
        )

        if not invalid:
            valid += 1
            new_annotations.append(ann)
            continue

        # -----------------------------------------------------
        # CLIP bbox to actual image boundaries
        # -----------------------------------------------------

        new_x1 = max(0.0, x1)
        new_y1 = max(0.0, y1)
        new_x2 = min(float(image_w), x2)
        new_y2 = min(float(image_h), y2)

        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1

        # If nothing remains inside the image, remove bbox
        if new_w <= 0 or new_h <= 0:

            print(
                f"[REMOVE] ann_id={ann['id']} "
                f"{filename}\n"
                f"         image={image_w}x{image_h}\n"
                f"         old bbox={[x, y, w, h]}\n"
                f"         box has no valid area after clipping"
            )

            removed += 1

            if not dry_run:
                continue
            else:
                # In dry-run, don't actually remove it
                new_annotations.append(ann)
                continue

        new_bbox = [
            round(new_x1, 4),
            round(new_y1, 4),
            round(new_w, 4),
            round(new_h, 4),
        ]

        print(
            f"[CLIP] ann_id={ann['id']} {filename}\n"
            f"       image={image_w}x{image_h}\n"
            f"       old={ann['bbox']}\n"
            f"       new={new_bbox}"
        )

        changed += 1

        if not dry_run:
            ann["bbox"] = new_bbox

            # COCO area should match the corrected bbox
            ann["area"] = round(new_w * new_h, 4)

        new_annotations.append(ann)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n--- SUMMARY ---")
    print(f"Annotations checked : {len(coco.get('annotations', []))}")
    print(f"Already valid       : {valid}")
    print(f"Would clip          : {changed}" if dry_run
          else f"Clipped             : {changed}")
    print(f"Would remove        : {removed}" if dry_run
          else f"Removed             : {removed}")
    print(f"Missing images      : {missing}")
    print(f"Errors              : {errors}")

    # ---------------------------------------------------------
    # WRITE MODE
    # ---------------------------------------------------------

    if not dry_run:

        coco["annotations"] = new_annotations

        backup_path = json_path.with_suffix(".json.bbox_backup")

        # Backup ORIGINAL JSON
        with open(json_path, "r", encoding="utf-8") as f:
            original = json.load(f)

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2)

        # Write corrected JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        print(f"\nBackup : {backup_path}")
        print(f"Updated: {json_path}")

    else:
        print("\nDRY RUN: No files were modified.")


def main():

    parser = argparse.ArgumentParser(
        description="Check and repair out-of-range COCO bounding boxes."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show proposed changes without modifying JSON files."
    )

    args = parser.parse_args()

    print("\nCOCO BOUNDING BOX CHECK / REPAIR")
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")

    # Process ALL splits
    for split_name, cfg in SPLITS.items():
        process_split(
            split_name,
            cfg,
            args.dry_run
        )

    print("\n" + "=" * 80)

    if args.dry_run:
        print("DRY RUN COMPLETE")
        print("No JSON files were modified.")
        print()
        print("Review the [CLIP] and [REMOVE] entries.")
        print("If they are acceptable, run:")
        print()
        print("    python fix_coco_bboxes.py")
    else:
        print("ALL SPLITS UPDATED")
        print("Train + Val + Test processed.")
        print("Backups were created before modification.")

    print("=" * 80)


if __name__ == "__main__":
    main()
