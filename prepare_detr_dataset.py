import os
import json
import random
import shutil
from tqdm import tqdm
from PIL import Image

# === CONFIG ===
clear_img_dir = "kitti/images"
blurred_img_dir = "kitti_blur/images"
label_dir = "labels"

output_base = "detr_dataset"
splits = {"train": 0.7, "val": 0.2, "test": 0.1}
categories = [{"id": i, "name": name} for i, name in enumerate(["Car", "Pedestrian", "Cyclist"])]
category_map = {cat["name"]: cat["id"] for cat in categories}

# === CREATE OUTPUT STRUCTURE ===
for variant in ["clear", "blurred"]:
    for split in splits:
        os.makedirs(os.path.join(output_base, variant, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_base, variant, "annotations"), exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(output_base, "labels", split), exist_ok=True)

def convert_annotation(txt_file, image_id, ann_id_start):
    annotations = []
    ann_id = ann_id_start
    with open(txt_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        cls = parts[0]
        if cls not in category_map:
            continue

        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])
        w = x2 - x1
        h = y2 - y1
        area = w * h

        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_map[cls],
            "bbox": [x1, y1, w, h],
            "area": area,
            "iscrowd": 0
        })
        ann_id += 1
    return annotations, ann_id

# === SPLIT IMAGES ===
all_images = sorted([f for f in os.listdir(clear_img_dir) if f.endswith(".png")])
random.seed(42)
random.shuffle(all_images)
total = len(all_images)
train_end = int(total * splits["train"])
val_end = train_end + int(total * splits["val"])
split_map = {
    "train": all_images[:train_end],
    "val": all_images[train_end:val_end],
    "test": all_images[val_end:]
}

# === PROCESS SPLITS ===
for split in ["train", "val", "test"]:
    for variant, img_dir in zip(["clear", "blurred"], [clear_img_dir, blurred_img_dir]):
        json_data = {
            "images": [],
            "annotations": [],
            "categories": categories
        }
        image_id = 0
        ann_id = 0

        for img_name in tqdm(split_map[split], desc=f"{variant.upper()} - {split}"):
            img_path = os.path.join(img_dir, img_name)
            txt_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))
            out_img_path = os.path.join(output_base, variant, "images", split, img_name)
            out_lbl_path = os.path.join(output_base, "labels", split, img_name.replace(".png", ".txt"))

            if not os.path.exists(txt_path) or not os.path.exists(img_path):
                continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception as e:
                print(f"Could not open image {img_path}: {e}")
                continue

            # Copy image and label
            shutil.copy(img_path, out_img_path)
            shutil.copy(txt_path, out_lbl_path)

            json_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })

            anns, ann_id = convert_annotation(txt_path, image_id, ann_id)
            json_data["annotations"].extend(anns)
            image_id += 1

        # Save COCO JSON
        json_out_path = os.path.join(output_base, variant, "annotations", f"{split}.json")
        with open(json_out_path, "w") as f:
            json.dump(json_data, f, indent=2)

print("Dataset split and annotation conversion complete (no OpenCV)")
