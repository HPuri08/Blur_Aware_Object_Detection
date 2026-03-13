import json
from pathlib import Path

root_dir = Path("detr_dataset")
json_files = list(root_dir.rglob("*.json"))

categories = [
    {"id": 0, "name": "Car", "supercategory": "vehicle"},
    {"id": 1, "name": "Pedestrian", "supercategory": "person"},
    {"id": 2, "name": "Cyclist", "supercategory": "person"},
]

info = {
    "description": "KITTI Object Detection in COCO format",
    "url": "https://www.cvlibs.net/datasets/kitti",
    "version": "1.0",
    "year": 2025,
    "contributor": "Malgorzata Galinska",
    "date_created": "2025-07-18"
}

for json_file in json_files:
    with open(json_file, "r") as f:
        data = json.load(f)

    data["info"] = info
    data["licenses"] = data.get("licenses", [])
    data["categories"] = categories

    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Fixed: {json_file}")
