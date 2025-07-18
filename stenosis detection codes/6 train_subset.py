#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:43:11 2025

@author: Ibrayeva Arman

Creates a smaller dataset and trains via YOLOv8 nano model
"""

import os
import shutil
from ultralytics import YOLO

# Paths
base_path = "/Users/ibrayea/Desktop/MPI/paper/arcade/stenosis"
yolo_base = "/Users/ibrayea/Desktop/arcade_yolo"
subset_base = "/Users/ibrayea/Desktop/arcade_yolo_subset"  # New subset directory
subsets = {
    "train": {
        "src_img_dir": f"{base_path}/train/images",
        "src_label_dir": f"{yolo_base}/train/labels",
        "dst_img_dir": f"{subset_base}/train/images",
        "dst_label_dir": f"{subset_base}/train/labels"
    },
    "val": {
        "src_img_dir": f"{base_path}/val/images",
        "src_label_dir": f"{yolo_base}/val/labels",
        "dst_img_dir": f"{subset_base}/val/images",
        "dst_label_dir": f"{subset_base}/val/labels"
    }
}

# Step 1: Create subset dataset
def create_subset_dataset():
    # Number of images to use
    train_limit = 200
    val_limit = 50
    
    for subset, paths in subsets.items():
        # Create directories
        os.makedirs(paths["dst_img_dir"], exist_ok=True)
        os.makedirs(paths["dst_label_dir"], exist_ok=True)
        
        # Get images with annotations
        label_files = [f for f in os.listdir(paths["src_label_dir"]) if f.endswith(".txt")]
        img_files = [f.replace(".txt", ".png") for f in label_files]
        
        # Limit number of images
        limit = train_limit if subset == "train" else val_limit
        selected_imgs = img_files[:min(len(img_files), limit)]
        
        # Copy images and labels
        for img in selected_imgs:
            src_img = os.path.join(paths["src_img_dir"], img)
            dst_img = os.path.join(paths["dst_img_dir"], img)
            src_label = os.path.join(paths["src_label_dir"], img.replace(".png", ".txt"))
            dst_label = os.path.join(paths["dst_label_dir"], img.replace(".png", ".txt"))
            
            if os.path.exists(src_img) and os.path.exists(src_label):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_label, dst_label)
        
        print(f"{subset.upper()} subset: Copied {len(selected_imgs)} images and labels")

# Step 2: Create subset data.yaml
def create_subset_yaml():
    config = f"""train: {subset_base}/train/images
val: {subset_base}/val/images
nc: 1
names: ["stenosis"]
"""
    config_path = f"{subset_base}/data.yaml"
    with open(config_path, 'w') as f:
        f.write(config)
    print(f"Subset config file created at: {config_path}")
    return config_path

# Step 3: Train YOLOv8 model
def train_yolo_model(yaml_path):
    # Load YOLOv8 nano model
    model = YOLO("yolov8n.pt")  # Pre-trained nano model
    
    # Train on subset
    model.train(
        data=yaml_path,
        epochs=10,  # Small number for quick training
        imgsz=512,  # Your image size
        batch=8,    # Small batch for CPU
        name="stenosis_subset",
        project=yolo_base,
        device="cpu"  # Change to 0 for GPU if available
    )
    print("Training complete")

# Run tasks
print("Creating subset dataset...")
create_subset_dataset()

print("\nCreating subset config...")
yaml_path = create_subset_yaml()

print("\nTraining YOLOv8 model...")
train_yolo_model(yaml_path)

print("\nTask complete")