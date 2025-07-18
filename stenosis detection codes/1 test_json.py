# Reads json, outputs numbers of images, etc.

import json
import os

# Paths to JSON files
base_path = "/Users/ibrayea/Desktop/MPI/paper/arcade/stenosis"
json_files = {
    "train": f"{base_path}/train/annotations/train.json",
    "val": f"{base_path}/val/annotations/val.json",
    "test": f"{base_path}/test/annotations/test.json"
}

# Explore each JSON file
for subset, json_path in json_files.items():
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Count images and annotations
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        stenosis_annotations = [ann for ann in data['annotations'] if ann['category_id'] == 26]
        
        print(f"\n{subset.upper()} Subset:")
        print(f"Number of images: {num_images}")
        print(f"Number of annotations: {num_annotations}")
        print(f"Number of stenosis annotations: {len(stenosis_annotations)}")
    else:
        print(f"\n{subset.upper()} Subset: JSON file not found at {json_path}")