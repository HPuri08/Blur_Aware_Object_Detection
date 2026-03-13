import os
import random
import shutil

random.seed(42)

# Directories
clear_dir = "kitti/images"
blurred_dir = "kitti_blur/images"
labels_dir = "labels_yolo"

out_base = "split_dataset"
splits = ['train', 'val']
ratios = [0.8, 0.2]  # 80% train, 20% val

# Get image list
images = sorted([f for f in os.listdir(clear_dir) if f.endswith(".png")])
random.shuffle(images)

# Split images
total = len(images)
train_end = int(total * ratios[0])
split_indices = {
    "train": images[:train_end],
    "val": images[train_end:]
}

# Create directories and copy files
for split in splits:
    for img_type in ["clear", "blurred"]:
        os.makedirs(os.path.join(out_base, img_type, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_base, "labels", split), exist_ok=True)

    for img_name in split_indices[split]:
        label_name = img_name.replace(".png", ".txt")

        # Copy clear and blurred images
        shutil.copy(os.path.join(clear_dir, img_name),
                    os.path.join(out_base, "clear", "images", split, img_name))
        shutil.copy(os.path.join(blurred_dir, img_name),
                    os.path.join(out_base, "blurred", "images", split, img_name))

        # Copy label (same for both)
        shutil.copy(os.path.join(labels_dir, label_name),
                    os.path.join(out_base, "labels", split, label_name))
