#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:21:38 2025

@author: ibrayea
"""

import json
import os
import cv2

# Paths
base_path = "/Users/ibrayea/Desktop/MPI/paper/arcade/stenosis"
output_base = "/Users/ibrayea/Desktop/arcade_yolo"  # Output directory for YOLO format
subsets = {
    "train": {
        "json": f"{base_path}/train/annotations/train.json",
        "img_dir": f"{base_path}/train/images",
        "output_labels": f"{output_base}/train/labels"
    },
    "val": {
        "json": f"{base_path}/val/annotations/val.json",
        "img_dir": f"{base_path}/val/images",
        "output_labels": f"{output_base}/val/labels"
    },
    "test": {
        "json": f"{base_path}/test/annotations/test.json",
        "img_dir": f"{base_path}/test/images",
        "output_labels": f"{output_base}/test/labels"
    }
}

# Function to convert COCO to YOLO format
def convert_coco_to_yolo(coco_json_path, img_dir, output_label_dir):
    # Create output directory
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Map image IDs to file names
    img_id_to_name = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    # Group annotations by image
    img_annotations = {}
    for ann in coco_data["annotations"]:
        if ann["category_id"] == 26:  # Stenosis only
            img_id = ann["image_id"]
            if img_id not in img_annotations:
                img_annotations[img_id] = []
            img_annotations[img_id].append(ann)
    
    # Convert to YOLO format
    for img_id, anns in img_annotations.items():
        img_name = img_id_to_name.get(img_id)
        if not img_name:
            print(f"Image ID {img_id} not found in JSON")
            continue
        
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        # Create YOLO annotation file
        txt_path = os.path.join(output_label_dir, img_name.replace(".png", ".txt"))
        with open(txt_path, 'w') as f:
            for ann in anns:
                x, y, w, h = ann["bbox"]  # COCO bbox: [x, y, width, height]
                # Normalize coordinates (image size is 512x512)
                img_w, img_h = 512, 512
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                # YOLO format: class_id x_center y_center width height
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        print(f"Converted annotations for {img_name}: {len(anns)} stenosis annotations")

# Convert train and val subsets
for subset, paths in subsets.items():
    print(f"\nProcessing {subset.upper()} subset...")
    convert_coco_to_yolo(paths["json"], paths["img_dir"], paths["output_labels"])

# Verify one converted annotation
def visualize_yolo_annotation(img_dir, label_dir, img_name):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, img_name.replace(".png", ".txt"))
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    img_h, img_w = img.shape[:2]
    
    # Read YOLO annotations
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        print(f"Visualizing {img_name} with {len(lines)} annotations")
        for line in lines:
            class_id, x_center, y_center, w, h = map(float, line.split())
            # Denormalize coordinates
            x_center, y_center, w, h = x_center * img_w, y_center * img_h, w * img_w, h * img_h
            x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
            x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
            # Draw rectangle (green, thickness=2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "stenosis", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display image
    cv2.imshow("YOLO Annotation", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize one image from train subset
visualize_yolo_annotation(
    subsets["train"]["img_dir"],
    subsets["train"]["output_labels"],
    "1.png"  # Replace with a known image name if needed
)

print("Conversion and verification complete")