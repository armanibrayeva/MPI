#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 13:09:09 2025

@author: Ibrayeva Arman

Visualizes the .json file of arcade dataset. 
Outputs the images with boxes around stenosis created from manual label coordinates in the json file

"""

import json
import cv2
import os

# Paths to dataset
base_path = "/Users/ibrayea/Desktop/MPI/paper/arcade/stenosis"
json_path = f"{base_path}/train/annotations/train.json"
img_dir = f"{base_path}/train/images"

# Load COCO JSON
with open(json_path, 'r') as f:
    coco_data = json.load(f)

# Map image IDs to file names
img_id_to_name = {img["id"]: img["file_name"] for img in coco_data["images"]}

# Get annotations for stenosis (category_id: 26)
annotations = [ann for ann in coco_data["annotations"] if ann["category_id"] == 26]

# Visualize first 5 images with annotations
num_images_to_show = 5
images_processed = 0
processed_image_ids = set()

for ann in annotations:
    img_id = ann["image_id"]
    if img_id in processed_image_ids:
        continue  # Skip if image already processed
    
    # Get image file
    img_name = img_id_to_name.get(img_id)
    if not img_name:
        print(f"Image ID {img_id} not found in JSON")
        continue
    
    img_path = os.path.join(img_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    
    # Draw all bounding boxes for this image
    img_anns = [a for a in annotations if a["image_id"] == img_id]
    for ann in img_anns:
        x, y, w, h = map(int, ann["bbox"])  # COCO bbox is [x, y, width, height]
        # Draw rectangle (green, thickness=2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Add label
        cv2.putText(img, "stenosis", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display image
    cv2.imshow("Stenosis Annotation", img)
    print(f"Displaying: {img_name} with {len(img_anns)} stenosis annotations")
    cv2.waitKey(0)  # Press any key to move to next image
    
    # Mark image as processed
    processed_image_ids.add(img_id)
    images_processed += 1
    
    if images_processed >= num_images_to_show:
        break

# Close all windows
cv2.destroyAllWindows()
print("Visualization complete")