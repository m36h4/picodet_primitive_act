try:
    batch = np.stack(batch, axis=0)
except ValueError:
    print("\n===== COLLATE SHAPE ERROR =====")
    print("KEY:", key)
    for i, x in enumerate(batch):
        print(" item", i, "shape:", getattr(x, "shape", None), "type:", type(x))
    print("================================\n")
    raise

cd /home/eng_megha/paddledetection

python - <<'PY'
import json, os

p = "dataset/labels/val.json"

with open(p, "r") as f:
    d = json.load(f)

imgs = {x["id"]: x for x in d["images"]}

for ann in d["annotations"]:
    if ann["image_id"] not in imgs:
        print("BAD annotation image_id:", ann["image_id"])
        continue

print("Images:", len(d["images"]))
print("Annotations:", len(d["annotations"]))
print("Categories:", d["categories"])

# Show images that have no annotations
ann_ids = {a["image_id"] for a in d["annotations"]}
no_ann = [x for x in d["images"] if x["id"] not in ann_ids]

print("Images with no annotations:", len(no_ann))
for x in no_ann[:20]:
    print(x["id"], x["file_name"], x.get("width"), x.get("height"))
PY
