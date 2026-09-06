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


def process_split(split_name, config, dry_run):

    json_path = config["json"]
    image_dir = config["images"]

    print("\n" + "=" * 70)
    print(f"PROCESSING SPLIT: {split_name.upper()}")
    print(f"JSON:   {json_path}")
    print(f"IMAGES: {image_dir}")
    print(f"MODE:   {'DRY RUN' if dry_run else 'WRITE'}")
    print("=" * 70)

    if not json_path.exists():
        print(f"[ERROR] JSON not found: {json_path}")
        return

    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: {image_dir}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    changed = 0
    already_correct = 0
    missing = 0
    errors = 0

    images = coco.get("images", [])

    print(f"Images in JSON: {len(images)}")

    for img in images:

        file_name = img["file_name"]

        # Only use the filename portion.
        # This handles file_name values containing
        # dataset/ or images/train/ prefixes.
        filename = Path(file_name).name

        image_path = image_dir / filename

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

        if old_w == actual_w and old_h == actual_h:
            already_correct += 1
            continue

        print(
            f"[CHANGE] {filename}\n"
            f"         {old_w}x{old_h} -> {actual_w}x{actual_h}"
        )

        if not dry_run:
            img["width"] = actual_w
            img["height"] = actual_h

        changed += 1

    print("\n--- SUMMARY ---")
    print(f"Images checked : {len(images)}")
    print(f"Already correct: {already_correct}")
    print(f"Would change   : {changed}" if dry_run else f"Changed        : {changed}")
    print(f"Missing files  : {missing}")
    print(f"Errors         : {errors}")

    # ---------------------------------------------------------
    # WRITE ONLY WHEN NOT DRY RUN
    # ---------------------------------------------------------
    if not dry_run and changed > 0:

        backup_path = json_path.with_suffix(".json.backup")

        # IMPORTANT:
        # Make backup of the ORIGINAL JSON before writing changes.
        #
        # Since coco has already been modified in memory,
        # reload the original file for the backup.
        with open(json_path, "r", encoding="utf-8") as f:
            original = json.load(f)

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(original, f, indent=2)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(coco, f, indent=2)

        print(f"\nBackup : {backup_path}")
        print(f"Updated: {json_path}")

    elif dry_run:
        print("\nDRY RUN: No files modified.")


def main():

    parser = argparse.ArgumentParser(
        description="Fix COCO image width/height metadata for train, val and test."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying any files."
    )

    args = parser.parse_args()

    print("\nCOCO IMAGE DIMENSION FIX")
    print(f"Dataset: {DATASET_DIR.resolve()}")

    if args.dry_run:
        print("Mode: DRY RUN")
    else:
        print("Mode: WRITE")

    # =========================================================
    # PROCESS ALL THREE SPLITS
    # =========================================================
    for split_name, config in SPLITS.items():
        process_split(
            split_name,
            config,
            args.dry_run
        )

    print("\n" + "=" * 70)

    if args.dry_run:
        print("DRY RUN COMPLETE")
        print("No JSON files were modified.")
        print()
        print("If the results look correct, run:")
        print("    python fix_coco_dimensions.py")
    else:
        print("ALL SPLITS UPDATED")
        print("Train + Val + Test processed.")

    print("=" * 70)


if __name__ == "__main__":
    main()
