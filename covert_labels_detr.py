import os
import json
from PIL import Image
from tqdm import tqdm

def convert_yolo_to_coco(image_dir, label_dir, output_json, category_names=["car"]):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')])

    images = []
    annotations = []
    categories = [{"id": i, "name": name} for i, name in enumerate(category_names)]

    ann_id = 0
    for img_id, img_file in enumerate(tqdm(image_files)):
        img_path = os.path.join(image_dir, img_file)
        width, height = Image.open(img_path).size
        images.append({
            "id": img_id,
            "file_name": img_file,
            "width": width,
            "height": height
        })

        label_file = os.path.join(label_dir, img_file.replace('.png', '.txt').replace('.jpg', '.txt'))
        if not os.path.exists(label_file):
            continue

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # skip bad lines
                class_id, x_center, y_center, w, h = map(float, parts)
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x_min = x_center - w / 2
                y_min = y_center - h / 2

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, "w") as f:
        json.dump(coco, f, indent=4)

# Clear
convert_yolo_to_coco(
    image_dir="split_dataset/clear/images/train",
    label_dir="split_dataset/labels/train",
    output_json="split_dataset/clear/annotations/train.json"
)

convert_yolo_to_coco(
    image_dir="split_dataset/clear/images/val",
    label_dir="split_dataset/labels/val",
    output_json="split_dataset/clear/annotations/val.json"
)

# Blurred
convert_yolo_to_coco(
    image_dir="split_dataset/blurred/images/train",
    label_dir="split_dataset/labels/train",
    output_json="split_dataset/blurred/annotations/train.json"
)

convert_yolo_to_coco(
    image_dir="split_dataset/blurred/images/val",
    label_dir="split_dataset/labels/val",
    output_json="split_dataset/blurred/annotations/val.json"
)
