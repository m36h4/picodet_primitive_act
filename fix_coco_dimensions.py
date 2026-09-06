import argparse
import json
from pathlib import Path
from PIL import Image


# ============================================================
# CONFIG
# ============================================================
DATASET_DIR = Path("dataset")

SPLITS = {
    "train": DATASET_DIR / "labels/train.json",
    "val": DATASET_DIR / "labels/val.json",
    "test": DATASET_DIR / "labels/test.json",
}


def process_split(split, json_path, dry_run):
    if not json_path.exists():
        print(f"[SKIP] {json_path} not found")
        return

    print(f"\n{'=' * 70}")
    print(f"Processing: {split}")
    print(f"JSON:       {json_path}")
    print(f"Mode:       {'DRY RUN' if dry_run else 'WRITE'}")
    print(f"{'=' * 70}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    changed = 0
    missing = 0
    errors = 0

    for img in coco.get("images", []):

        file_name = img["file_name"]
        image_path = DATASET_DIR / file_name

        if not image_path.exists():
            print(f"[MISSING] {image_path}")
            missing += 1
            continue

        try:
            with Image.open(image_path) as im:
                actual_w, actual_h = im.size
        except Exception as e:
            print(f"[ERROR] {image_path}: {e}")
            errors += 1
            continue

        old_w = img.get("width")
        old_h = img.get("height")

        if old_w != actual_w or old_h != actual_h:

            print(
                f"[CHANGE] {file_name}\n"
                f"         width : {old_w} -> {actual_w}\n"
                f"         height: {old_h} -> {actual_h}"
            )

            if not dry_run:
                img["width"] = actual_w
                img["height"] = actual_h

            changed += 1

    print(f"\nSummary for {split}:")
    print(f"  Images checked : {len(coco.get('images', []))}")
    print(f"  Would change   : {changed}" if dry_run else
          f"  Changed        : {changed}")
    print(f"  Missing files  : {missing}")
    print(f"  Errors         : {errors}")

    # --------------------------------------------------------
    # WRITE MODE ONLY
    # --------------------------------------------------------
    if not dry_run and changed > 0:

        backup_path = json_path.with_suffix(".json.backup")

        # Create backup BEFORE modifying the original
        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        # Write corrected JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        print(f"  Backup created : {backup_path}")
        print(f"  Updated JSON   : {json_path}")

    elif dry_run:
        print("\n  DRY RUN: No files were modified.")


def main():
    parser = argparse.ArgumentParser(
        description="Fix COCO image width/height metadata using actual image dimensions."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying any files."
    )

    args = parser.parse_args()

    print("\nCOCO IMAGE DIMENSION CHECK")
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")

    for split, json_path in SPLITS.items():
        process_split(split, json_path, args.dry_run)

    print("\n" + "=" * 70)

    if args.dry_run:
        print("DRY RUN COMPLETE")
        print("No files were modified.")
        print("If the changes look correct, run:")
        print("  python fix_coco_dimensions.py")
    else:
        print("UPDATE COMPLETE")
        print("COCO image width/height metadata has been updated.")

    print("=" * 70)


if __name__ == "__main__":
    main()
