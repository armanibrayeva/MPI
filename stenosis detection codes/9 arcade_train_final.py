#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 14:43:04 2025

@author: Ibrayeva Arman
yolo 10, stenosis detection using arcade dataset, following paper details


"""

import os
import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from skimage.morphology import white_tophat, disk
import matplotlib.pyplot as plt
import shutil

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# --- Main Path to your original dataset ---
ORIGINAL_DATASET_BASE_PATH = "/Users/ibrayea/Desktop/MPI/Stenosis Detection/arcade/stenosis/"

# --- Paths for the new YOLOv10-formatted dataset ---
# This is where the preprocessed images and labels will be stored
YOLO_DATASET_PATH = os.path.join(ORIGINAL_DATASET_BASE_PATH, "yolo_stenosis_dataset")

# --- Model Configuration ---
# Options: yolov10n.pt, yolov10s.pt, yolov10m.pt, yolov10b.pt, yolov10l.pt, yolov10x.pt
# 'x' is the largest and most accurate; 'n' is the smallest and fastest.
YOLO_MODEL = 'yolov10x.pt'

# --- Training Parameters ---
EPOCHS = 50
BATCH_SIZE = 8
IMAGE_SIZE = 640 # YOLOv10 was benchmarked on 640px images
PROJECT_NAME = 'stenosis_detection_project'
RUN_NAME = f'{os.path.splitext(YOLO_MODEL)[0]}_run'


# ==============================================================================
# 2. IMAGE PREPROCESSING (Based on the Paper)
# ==============================================================================
def preprocess_image_enhanced(image: np.ndarray) -> np.ndarray:
    """
    Applies the 'Enhanced' image preprocessing pipeline from the research paper.
    This was found to be the most effective for stenosis detection.

    Args:
        image (np.ndarray): A single-channel (grayscale) input image.

    Returns:
        np.ndarray: A 3-channel (RGB) processed image.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be grayscale (single-channel).")

    # 1. White Top-Hat transformation with a 50x50 kernel on the inverted image.
    # A disk kernel of radius 25 is a close approximation.
    kernel = disk(25)
    img_neg = 255 - image
    tophat_img = white_tophat(img_neg, kernel)

    # 2. Subtract the result from the original image.
    subtracted_img = cv2.subtract(image, tophat_img)

    # 3. Clip the result to the valid 0-255 range.
    clipped_img = np.clip(subtracted_img, 0, 255).astype(np.uint8)

    # 4. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    # The paper specifies a grid size of 8x8 and a clip limit of 2.0.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(clipped_img)

    # 5. Convert back to 3-channel RGB for the YOLO model.
    processed_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    return processed_image_rgb


# ==============================================================================
# 3. ONE-TIME DATASET PREPARATION
# ==============================================================================
def prepare_yolo_dataset():
    """
    Converts the original ARCADE dataset into the format required by YOLO.
    This involves creating specific directories, preprocessing images, converting
    annotations, and generating a dataset YAML file.

    This function only needs to be run once.
    """
    print("--- Starting YOLO Dataset Preparation ---")
    if os.path.exists(YOLO_DATASET_PATH):
        print(f"YOLO dataset path '{YOLO_DATASET_PATH}' already exists. Skipping setup.")
        print("If you need to regenerate the dataset, please delete this folder first.")
        return

    # Create directory structure
    print("Creating directory structure...")
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images/train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels/train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images/val"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels/val"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images/test"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels/test"), exist_ok=True)

    # Find the category ID for 'stenosis'
    with open(os.path.join(ORIGINAL_DATASET_BASE_PATH, "train/annotations/train.json")) as f:
        ann_data = json.load(f)
    stenosis_category_id = next((cat['id'] for cat in ann_data['categories'] if cat['name'] == 'stenosis'), None)
    if stenosis_category_id is None:
        raise ValueError("Category 'stenosis' not found in annotations.")
    # In YOLO format, since we only have one class, its index will be 0.
    yolo_class_id = 0

    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        print(f"Processing '{split}' split...")
        img_dir = os.path.join(ORIGINAL_DATASET_BASE_PATH, f"{split}/images")
        ann_path = os.path.join(ORIGINAL_DATASET_BASE_PATH, f"{split}/annotations/{split}.json")

        if not os.path.exists(ann_path):
            print(f"Annotation file for '{split}' not found. Skipping.")
            continue

        with open(ann_path) as f:
            data = json.load(f)

        # Create a map of image_id to annotations for that image
        img_to_anns = {}
        for ann in data['annotations']:
            if ann['category_id'] == stenosis_category_id:
                img_id = ann['image_id']
                if img_id not in img_to_anns:
                    img_to_anns[img_id] = []
                img_to_anns[img_id].append(ann)

        # Process each image and its annotations
        for img_info in data['images']:
            img_id = img_info['id']
            img_filename = img_info['file_name']
            img_path = os.path.join(img_dir, img_filename)

            # Load original image
            original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                print(f"Warning: Could not read image {img_path}. Skipping.")
                continue

            # Apply preprocessing
            processed_image = preprocess_image_enhanced(original_image)
            img_h, img_w, _ = processed_image.shape

            # Save processed image to the new YOLO directory
            cv2.imwrite(os.path.join(YOLO_DATASET_PATH, f"images/{split}/{img_filename}"), processed_image)

            # Convert annotations to YOLO format and save
            annotations = img_to_anns.get(img_id, [])
            label_filename = os.path.splitext(img_filename)[0] + '.txt'
            with open(os.path.join(YOLO_DATASET_PATH, f"labels/{split}/{label_filename}"), 'w') as f:
                for ann in annotations:
                    # COCO bbox is [x_min, y_min, width, height]
                    x, y, w, h = ann['bbox']
                    # YOLO format is [class_id, x_center_norm, y_center_norm, width_norm, height_norm]
                    x_center = (x + w / 2) / img_w
                    y_center = (y + h / 2) / img_h
                    w_norm = w / img_w
                    h_norm = h / img_h
                    f.write(f"{yolo_class_id} {x_center} {y_center} {w_norm} {h_norm}\n")

    # Create the dataset.yaml file for YOLO
    print("Creating dataset.yaml file...")
    yaml_content = f"""
path: {os.path.abspath(YOLO_DATASET_PATH)}
train: images/train
val: images/val
test: images/test

# Classes
names:
  0: stenosis
"""
    with open(os.path.join(YOLO_DATASET_PATH, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

    print("--- Dataset Preparation Complete ---")


# ==============================================================================
# 4. TRAINING, EVALUATION, and PREDICTION
# ==============================================================================
def main():
    """Main function to run the entire workflow."""

    # Step 1: Prepare the dataset (only runs if not already prepared)
    prepare_yolo_dataset()
    dataset_yaml_path = os.path.join(YOLO_DATASET_PATH, 'dataset.yaml')

    # Step 2: Train the YOLOv10 model
    print("\n--- Starting YOLOv10 Model Training ---")
    model = YOLO(YOLO_MODEL)

    # Launch training
    model.train(
        data=dataset_yaml_path,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,
        project=PROJECT_NAME,
        name=RUN_NAME
    )
    print("\n--- Training Complete ---")

    # Step 3: Evaluate the best model on the validation set
    print("\n--- Evaluating Best Model ---")
    # Path to the best model weights saved during training
    best_model_path = os.path.join(PROJECT_NAME, RUN_NAME, 'weights/best.pt')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError("Could not find best model weights. Training may have failed.")

    # Load the best model
    best_model = YOLO(best_model_path)
    metrics = best_model.val()
    print("\nValidation Metrics:")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:   {metrics.box.map50:.4f}")
    print(f"  Precision: {metrics.box.p[0]:.4f}") # Precision for class 0
    print(f"  Recall:    {metrics.box.r[0]:.4f}") # Recall for class 0


    # Step 4: Run prediction on a sample test image
    print("\n--- Running Prediction on a Sample Image ---")
    test_images_dir = os.path.join(YOLO_DATASET_PATH, 'images/test')
    test_images = os.listdir(test_images_dir)

    if not test_images:
        print("No test images found to run prediction.")
        return

    sample_image_path = os.path.join(test_images_dir, test_images[0])
    print(f"Predicting on: {sample_image_path}")

    # The predict function will save the output image with bounding boxes
    results = best_model.predict(
        source=sample_image_path,
        save=True,
        conf=0.25 # Confidence threshold for detection
    )
    print(f"\nPrediction complete. Output saved in the '{PROJECT_NAME}/{RUN_NAME}' directory.")


if __name__ == '__main__':
    main()