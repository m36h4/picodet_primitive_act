python - <<'PY'
import json
from PIL import Image

ann_file = "dataset/labels/train.json"

with open(ann_file, "r") as f:
    data = json.load(f)

images = {x["id"]: x for x in data["images"]}

count = 0

for img_id, img in images.items():
    path = "dataset/" + img["file_name"]

    try:
        w_actual, h_actual = Image.open(path).size
    except Exception:
        continue

    w_ann = img["width"]
    h_ann = img["height"]

    if (w_actual, h_actual) != (w_ann, h_ann):
        print(
            f"{img['file_name']}: "
            f"actual={w_actual}x{h_actual}, "
            f"annotation={w_ann}x{h_ann}"
        )
        count += 1

print("\nTotal mismatches:", count)
PY
