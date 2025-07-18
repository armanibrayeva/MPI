#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 2025

@author: ibrayea
"""

from ultralytics import YOLO

# Paths
full_base = "/Users/ibrayea/Desktop/arcade_yolo_full"
yolo_base = "/Users/ibrayea/Desktop/arcade_yolo"
yaml_path = f"{full_base}/data.yaml"

# Train YOLOv8 model
def train_yolo_model():
    # Load YOLOv8 nano model
    model = YOLO("yolov8n.pt")  # Pre-trained nano model
    
    # Train on full dataset
    model.train(
        data=yaml_path,
        epochs=50,  # Suitable for full dataset
        imgsz=512,  # Image size
        batch=16,   # Adjust based on hardware
        name="stenosis_full",
        project=yolo_base,
        device="cpu"  # Change to "0" for GPU if available
    )
    print("Training complete")

# Run training
print("Training YOLOv8 model on full dataset...")
train_yolo_model()
print("Task complete")