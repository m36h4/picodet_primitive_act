import json
from pathlib import Path
from PIL import Image

DATASET = Path("dataset")

SPLITS = {
    "train": {
        "json": DATASET / "labels/train.json",
        "images": DATASET / "images/train",
    },
    "val": {
        "json": DATASET / "labels/val.json",
        "images": DATASET / "images/val",
    },
    "test": {
        "json": DATASET / "labels/test.json",
        "images": DATASET / "images/test",
    },
}


def check_split(split, cfg):

    print("\n" + "=" * 80)
    print(f"CHECKING: {split.upper()}")
    print("=" * 80)

    with open(cfg["json"], "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {x["id"]: x for x in coco["images"]}

    bad_boxes = []
    bad_classes = []
    missing_images = 0

    for ann in coco["annotations"]:

        image_id = ann["image_id"]
        img = images.get(image_id)

        if img is None:
            bad_boxes.append(
                (ann["id"], "image_id_not_found", image_id)
            )
            continue

        filename = Path(img["file_name"]).name
        image_path = cfg["images"] / filename

        if not image_path.exists():
            missing_images += 1
            continue

        # Use actual image dimensions
        with Image.open(image_path) as im:
            actual_w, actual_h = im.size

        x, y, w, h = ann["bbox"]

        problems = []

        if w <= 0:
            problems.append(f"width={w}")

        if h <= 0:
            problems.append(f"height={h}")

        if x < 0:
            problems.append(f"x={x}<0")

        if y < 0:
            problems.append(f"y={y}<0")

        if x + w > actual_w:
            problems.append(
                f"x+w={x+w:.2f}>image_width={actual_w}"
            )

        if y + h > actual_h:
            problems.append(
                f"y+h={y+h:.2f}>image_height={actual_h}"
            )

        if problems:
            bad_boxes.append(
                {
                    "ann_id": ann["id"],
                    "image_id": image_id,
                    "file": filename,
                    "image_size": f"{actual_w}x{actual_h}",
                    "bbox": [x, y, w, h],
                    "problems": problems,
                }
            )

        # Check class
        category_id = ann.get("category_id")

        if category_id != 1:
            bad_classes.append(
                {
                    "ann_id": ann["id"],
                    "file": filename,
                    "category_id": category_id,
                }
            )

    print(f"Images              : {len(coco['images'])}")
    print(f"Annotations          : {len(coco['annotations'])}")
    print(f"Invalid bboxes       : {len(bad_boxes)}")
    print(f"Invalid category IDs : {len(bad_classes)}")
    print(f"Missing images       : {missing_images}")

    # Show first 30 bad boxes
    if bad_boxes:
        print("\n--- FIRST BAD BOXES ---")

        for item in bad_boxes[:30]:
            print(item)

    if bad_classes:
        print("\n--- BAD CATEGORY IDs ---")

        for item in bad_classes[:30]:
            print(item)


def main():

    for split, cfg in SPLITS.items():
        check_split(split, cfg)


if __name__ == "__main__":
    main()
