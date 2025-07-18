#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 2025

@author: ibrayea
"""

import os
import shutil

# Paths
base_path = "/Users/ibrayea/Desktop/MPI/paper/arcade/stenosis"
yolo_base = "/Users/ibrayea/Desktop/arcade_yolo"
full_base = "/Users/ibrayea/Desktop/arcade_yolo_full"  # New directory for full dataset
subsets = {
    "train": {
        "src_img_dir": f"{base_path}/train/images",
        "src_label_dir": f"{yolo_base}/train/labels",
        "dst_img_dir": f"{full_base}/train/images",
        "dst_label_dir": f"{full_base}/train/labels"
    },
    "val": {
        "src_img_dir": f"{base_path}/val/images",
        "src_label_dir": f"{yolo_base}/val/labels",
        "dst_img_dir": f"{full_base}/val/images",
        "dst_label_dir": f"{full_base}/val/labels"
    },
    "test": {
        "src_img_dir": f"{base_path}/test/images",
        "src_label_dir": f"{yolo_base}/test/labels",
        "dst_img_dir": f"{full_base}/test/images",
        "dst_label_dir": f"{full_base}/test/labels"
    }
}

# Step 1: Create full dataset
def create_full_dataset():
    for subset, paths in subsets.items():
        # Create directories
        os.makedirs(paths["dst_img_dir"], exist_ok=True)
        os.makedirs(paths["dst_label_dir"], exist_ok=True)
        
        # Get images with annotations
        label_files = [f for f in os.listdir(paths["src_label_dir"]) if f.endswith(".txt")]
        img_files = [f.replace(".txt", ".png") for f in label_files]
        
        # Copy all images and labels
        copied = 0
        for img in img_files:
            src_img = os.path.join(paths["src_img_dir"], img)
            dst_img = os.path.join(paths["dst_img_dir"], img)
            src_label = os.path.join(paths["src_label_dir"], img.replace(".png", ".txt"))
            dst_label = os.path.join(paths["dst_label_dir"], img.replace(".png", ".txt"))
            
            if os.path.exists(src_img) and os.path.exists(src_label):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_label, dst_label)
                copied += 1
        
        print(f"{subset.upper()} full set: Copied {copied} images and labels")

# Step 2: Create data.yaml for full dataset
def create_full_yaml():
    config = f"""train: {full_base}/train/images
val: {full_base}/val/images
test: {full_base}/test/images
nc: 1
names: ["stenosis"]
"""
    config_path = f"{full_base}/data.yaml"
    with open(config_path, 'w') as f:
        f.write(config)
    print(f"Full dataset config file created at: {config_path}")
    return config_path

# Run tasks
print("Creating full dataset...")
create_full_dataset()

print("\nCreating full dataset config...")
yaml_path = create_full_yaml()

print("\nFull dataset processing complete")