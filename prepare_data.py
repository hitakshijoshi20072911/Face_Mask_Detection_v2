import os
import shutil
import random

# Configuration
RAW_DIR = "./FMD_DATASET"          # <-- Your extracted dataset
OUT_DIR = "dataset"
SPLIT = (0.8, 0.1, 0.1)            # train / val / test
random.seed(42)

# The three target classes
CLASSES = ["incorrect_mask", "with_mask", "without_mask"]

for cls in CLASSES:
    # Collect all images from both 'simple' and 'complex' subfolders
    images = []
    for sub in ["simple", "complex"]:
        src_folder = os.path.join(RAW_DIR, cls, sub)
        if not os.path.exists(src_folder):
            print(f"⚠️ Missing folder: {src_folder}")
            continue
        for fname in os.listdir(src_folder):
            full_path = os.path.join(src_folder, fname)
            if os.path.isfile(full_path):
                images.append(full_path)

    print(f"{cls}: found {len(images)} images")

    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLIT[0])
    n_val   = int(n * SPLIT[1])

    splits = {
        "train": images[:n_train],
        "val":   images[n_train:n_train + n_val],
        "test":  images[n_train + n_val:]
    }

    for split, files in splits.items():
        dst_dir = os.path.join(OUT_DIR, split, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for src_path in files:
            shutil.copy2(src_path, os.path.join(dst_dir, os.path.basename(src_path)))
        print(f"  {split}: {len(files)} images")

print("Dataset prepared in", OUT_DIR)