import os
import json
from collections import defaultdict

def load_coco_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def summarize_coco(data):
    num_images = len(data.get('images', []))
    annotations = data.get('annotations', [])
    num_annotations = len(annotations)

    # Count annotations per category
    cat_counts = defaultdict(int)
    for ann in annotations:
        cat_counts[ann['category_id']] += 1

    return num_images, num_annotations, cat_counts

def print_summary(name, json_path):
    if not os.path.exists(json_path):
        print(f"[{name}] JSON not found: {json_path}")
        return

    data = load_coco_json(json_path)
    num_images, num_annotations, cat_counts = summarize_coco(data)

    print(f"\n== {name.upper()} ==")
    print(f"Images:       {num_images}")
    print(f"Annotations:  {num_annotations}")
    print("Per Category:")
    for cat_id, count in sorted(cat_counts.items()):
        print(f"  Category {cat_id}: {count}")

if __name__ == "__main__":
    base_path = "detr_dataset/clear/annotations"

    for split in ['train', 'val', 'test']:
        json_path = os.path.join(base_path, f"{split}.json")
        print_summary(split, json_path)
