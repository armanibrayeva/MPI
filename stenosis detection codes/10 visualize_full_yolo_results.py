#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 2025

@author: ibrayea
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from ultralytics import YOLO

# Paths
project_dir = "/Users/ibrayea/Desktop/arcade_yolo/stenosis_full"
results_csv = f"{project_dir}/results.csv"
model_path = f"{project_dir}/weights/best.pt"
test_img_dir = "/Users/ibrayea/Desktop/arcade_yolo_full/test/images"
test_label_dir = "/Users/ibrayea/Desktop/arcade_yolo_full/test/labels"
output_dir = f"{project_dir}/visualizations"
os.makedirs(output_dir, exist_ok=True)

# Helper function: Compute IoU
def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

# Step 1: Visualize training metrics including precision and recall
def plot_training_metrics():
    if not os.path.exists(results_csv):
        print(f"Results CSV not found at {results_csv}")
        return
    
    df = pd.read_csv(results_csv)
    
    plt.figure(figsize=(15, 10))
    
    # Box Loss
    plt.subplot(2, 3, 1)
    plt.plot(df["epoch"], df["train/box_loss"], label="Train Box Loss")
    plt.plot(df["epoch"], df["val/box_loss"], label="Val Box Loss")
    plt.title("Box Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Classification Loss
    plt.subplot(2, 3, 2)
    plt.plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss")
    plt.plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss")
    plt.title("Classification Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # DFL Loss
    plt.subplot(2, 3, 3)
    plt.plot(df["epoch"], df["train/dfl_loss"], label="Train DFL Loss")
    plt.plot(df["epoch"], df["val/dfl_loss"], label="Val DFL Loss")
    plt.title("Distribution Focal Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # mAP
    plt.subplot(2, 3, 4)
    plt.plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5")
    plt.plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95")
    plt.title("mAP Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    
    # Precision
    plt.subplot(2, 3, 5)
    plt.plot(df["epoch"], df["metrics/precision(B)"], label="Precision")
    plt.title("Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.legend()
    
    # Recall
    plt.subplot(2, 3, 6)
    plt.plot(df["epoch"], df["metrics/recall(B)"], label="Recall")
    plt.title("Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    
    plt.tight_layout()
    plot_path = f"{output_dir}/training_metrics_with_accuracy.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Training metrics plot saved at: {plot_path}")

# Step 2: Measure detection time, accuracy, and visualize predictions
def evaluate_detection_time_and_accuracy():
    model = YOLO(model_path)
    
    img_files = [f for f in os.listdir(test_img_dir) if f.endswith(".png")]
    if not img_files:
        print(f"No images found in {test_img_dir}")
        return
    
    detection_times = []
    correct_detections = 0
    total_gt_boxes = 0
    iou_threshold = 0.5
    num_visualizations = 5
    visualized = 0
    
    for img_file in img_files:
        img_path = os.path.join(test_img_dir, img_file)
        label_path = os.path.join(test_label_dir, img_file.replace(".png", ".txt"))
        
        # Load ground truth
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    _, x_center, y_center, w, h = map(float, line.split())
                    # Denormalize (512x512 images)
                    x1 = (x_center - w/2) * 512
                    y1 = (y_center - h/2) * 512
                    x2 = (x_center + w/2) * 512
                    y2 = (y_center + h/2) * 512
                    gt_boxes.append([x1, y1, x2, y2])
        
        total_gt_boxes += len(gt_boxes)
        
        # Run inference
        start_time = time.time()
        results = model.predict(img_path, imgsz=512, device="cpu")  # Use "0" for GPU
        end_time = time.time()
        detection_times.append(end_time - start_time)
        
        # Evaluate accuracy
        pred_boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                pred_boxes.append([x1, y1, x2, y2])
        
        # Match predictions to ground truth
        matched_gt = set()
        for pred_box in pred_boxes:
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                if compute_iou(pred_box, gt_box) >= iou_threshold:
                    correct_detections += 1
                    matched_gt.add(i)
                    break
        
        # Visualize predictions
        if visualized < num_visualizations:
            img = cv2.imread(img_path)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"stenosis {conf:.2f}", (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            vis_path = f"{output_dir}/test_{img_file}"
            cv2.imwrite(vis_path, img)
            visualized += 1
    
    # Calculate metrics
    avg_time = np.mean(detection_times)
    detection_accuracy = correct_detections / total_gt_boxes if total_gt_boxes > 0 else 0
    
    print(f"Average detection time per image: {avg_time:.4f} seconds")
    print(f"Detection accuracy (IoU>{iou_threshold}): {detection_accuracy:.4f}")
    print(f"Correct detections: {correct_detections}/{total_gt_boxes}")
    print(f"Total images processed: {len(detection_times)}")
    print(f"Visualized predictions saved in: {output_dir}")

# Run tasks
print("Plotting training metrics...")
plot_training_metrics()

print("\nEvaluating detection time, accuracy, and visualizing predictions...")
evaluate_detection_time_and_accuracy()

print("\nVisualization and evaluation complete")