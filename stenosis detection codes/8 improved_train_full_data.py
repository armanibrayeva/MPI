#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 2025

@author: ibrayea
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import shutil

# Paths
full_base = "/Users/ibrayea/Desktop/arcade_yolo_full"
yolo_base = "/Users/ibrayea/Desktop/arcade_yolo"
project_dir = f"{yolo_base}/stenosis_improved"
yaml_path = f"{full_base}/data.yaml"
prev_results_csv = f"{yolo_base}/stenosis_full/results.csv"
combined_results_csv = f"{project_dir}/results_combined.csv"
diagnostics_dir = f"{project_dir}/diagnostics"
os.makedirs(project_dir, exist_ok=True)
os.makedirs(diagnostics_dir, exist_ok=True)

# Step 1: Train improved YOLOv8 model
def train_yolo_model():
    # Load YOLOv8 medium model
    model = YOLO("yolov8m.pt")  # Medium model for higher capacity
    
    # Train with improved settings
    results = model.train(
        data=yaml_path,
        epochs=100,           # More epochs for convergence
        imgsz=512,            # Image size
        batch=32,             # Larger batch size
        name="stenosis_improved",
        project=project_dir,
        device="cpu",         # Change to "0" for GPU if available
        patience=10,          # Early stopping
        dropout=0.1,          # Regularization
        lr0=0.001,            # Initial learning rate
        lrf=0.01,             # Final learning rate factor
        cos_lr=True,          # Cosine learning rate scheduler
        augment=True,         # Enable data augmentation
        amp=True              # Automatic Mixed Precision
    )
    print("Training complete")
    return results

# Step 2: Combine previous and new results
def combine_results():
    # Load previous results
    if os.path.exists(prev_results_csv):
        prev_df = pd.read_csv(prev_results_csv)
    else:
        print(f"Previous results not found at {prev_results_csv}")
        prev_df = pd.DataFrame()
    
    # Load new results
    new_results_csv = f"{project_dir}/results.csv"
    if os.path.exists(new_results_csv):
        new_df = pd.read_csv(new_results_csv)
        # Adjust epoch numbers to continue from previous
        new_df["epoch"] = new_df["epoch"] + len(prev_df)
    else:
        print(f"New results not found at {new_results_csv}")
        return
    
    # Combine and save
    combined_df = pd.concat([prev_df, new_df], ignore_index=True)
    combined_df.to_csv(combined_results_csv, index=False)
    print(f"Combined results saved at: {combined_results_csv}")
    return combined_df

# Step 3: Generate diagnostic clues
def generate_diagnostics(df):
    # Plot validation mAP50 vs. learning rate
    plt.figure(figsize=(8, 6))
    plt.scatter(df["lr/pg0"], df["metrics/mAP50(B)"], c=df["epoch"], cmap="viridis")
    plt.colorbar(label="Epoch")
    plt.xlabel("Learning Rate")
    plt.ylabel("mAP@0.5")
    plt.title("mAP@0.5 vs. Learning Rate")
    plot_path = f"{diagnostics_dir}/lr_vs_map50.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Learning rate plot saved at: {plot_path}")
    # Clue: If mAP50 peaks at high LR and drops, LR may be too high. If flat, LR may be too low.
    
    # Overfitting analysis
    overfitting = df["val/box_loss"] - df["train/box_loss"]
    avg_overfitting = overfitting.mean()
    print(f"Average overfitting (val_box_loss - train_box_loss): {avg_overfitting:.4f}")
    if avg_overfitting > 0.5:
        print("Clue: Significant overfitting. Increase dropout or augmentation.")
    elif avg_overfitting < 0:
        print("Clue: Underfitting. Increase model capacity or epochs.")
    
    # Confidence threshold analysis (run inference on test set)
    model = YOLO(f"{project_dir}/weights/best.pt")
    test_img_dir = f"{full_base}/test/images"
    img_files = [f for f in os.listdir(test_img_dir) if f.endswith(".png")][:50]  # Subset for speed
    thresholds = [0.25, 0.5, 0.75]
    precisions, recalls = [], []
    
    for thresh in thresholds:
        results = model.predict([os.path.join(test_img_dir, f) for f in img_files],
                               imgsz=512, conf=thresh, device="cpu", verbose=False)
        tp, fp, fn = 0, 0, 0
        for r in results:
            pred_boxes = len(r.boxes)
            # Simplified: Assume ground truth boxes exist (detailed IoU matching omitted for brevity)
            tp += min(pred_boxes, 1)  # Placeholder for actual TP calculation
            fp += max(0, pred_boxes - 1)
            fn += max(0, 1 - pred_boxes)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        precisions.append(prec)
        recalls.append(rec)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Metric Value")
    plt.title("Precision/Recall vs. Confidence Threshold")
    plt.legend()
    plot_path = f"{diagnostics_dir}/conf_threshold.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Confidence threshold plot saved at: {plot_path}")
    print("Clue: If precision drops at low thresholds, reduce false positives with NMS tuning.")
    
    # Per-class mAP (single class: stenosis)
    last_map50 = df["metrics/mAP50(B)"].iloc[-1]
    print(f"Final mAP@0.5 for stenosis: {last_map50:.4f}")
    if last_map50 < 0.5:
        print("Clue: Low mAP. Check stenosis label consistency or increase training data.")
    
    # Training time estimate
    if "time" in df.columns:
        per_epoch_time = (df["time"].iloc[-1] - df["time"].iloc[0]) / len(df)
        print(f"Average time per epoch: {per_epoch_time:.2f} seconds")
        print("Clue: If time is high, consider GPU or smaller model for faster iterations.")

# Run tasks
print("Training improved YOLOv8 model...")
train_yolo_model()

print("\nCombining results...")
combined_df = combine_results()

print("\nGenerating diagnostic clues...")
generate_diagnostics(combined_df)

print("\nImproved training and diagnostics complete")