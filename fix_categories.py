import glob, json

for fn in glob.glob("detr_dataset/*/annotations/*.json"):
    d = json.load(open(fn))
    # bump categories
    for cat in d["categories"]:
        cat["id"] += 1
    # bump annotations
    for ann in d["annotations"]:
        ann["category_id"] += 1
    with open(fn, "w") as f:
        json.dump(d, f, indent=2)
    print("Patched", fn)
