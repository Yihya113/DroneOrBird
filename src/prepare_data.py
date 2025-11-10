"""
prepare_data.py
Splits the Kaggle 'Malicious Drones' dataset into train / val / test folders
for Drone vs Bird classification.
"""

import os
import shutil
import random
from pathlib import Path

# paths
SRC = Path("data/UAV_Dataset")
DST = Path("data/split")

# only use these two classes for now
CLASSES = ["Drone", "Bird"]

# split ratios
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

# create output folders
for split in RATIOS.keys():
    for cls in CLASSES:
        (DST / split / cls).mkdir(parents=True, exist_ok=True)

# reproducibility
random.seed(42)

for cls in CLASSES:
    imgs = list((SRC / cls).glob("*"))
    random.shuffle(imgs)
    n = len(imgs)

    train_end = int(n * RATIOS["train"])
    val_end = train_end + int(n * RATIOS["val"])

    splits = {
        "train": imgs[:train_end],
        "val": imgs[train_end:val_end],
        "test": imgs[val_end:],
    }

    for split, files in splits.items():
        for f in files:
            shutil.copy(f, DST / split / cls / f.name)

print("✅ Data split complete!")
print(f"Output saved to: {DST.resolve()}")
