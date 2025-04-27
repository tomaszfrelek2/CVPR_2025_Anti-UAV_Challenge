import os
import random
import shutil
from pathlib import Path
import json

# ==== CONFIGURATION ====
dataset_dir = Path("/Users/tomaszfrelek/Downloads/train")  # Has images/ and labels/ folders
output_root = Path("/Users/tomaszfrelek/Downloads/test_images")
output_root.mkdir(parents=True, exist_ok=True)

# ========================
#random.seed(seed)

# Replace this with the actual path to your master folder
subfolders = [name for name in os.listdir(dataset_dir)
              if os.path.isdir(os.path.join(dataset_dir, name))]

# print("Subfolders found:")
# for folder in subfolders:
#     print(folder)

# Input folders
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"



# # Output folder setup
# for split in ["train", "val", "test"]:
#     (output_root / split / "images").mkdir(parents=True, exist_ok=True)
#     (output_root / split / "labels").mkdir(parents=True, exist_ok=True)

for i,folder in enumerate(subfolders):
    if i == 10: 
        break
    label_file = os.path.join(dataset_dir, folder,"IR_label.json")

    with open(label_file, 'r') as f:
        label_info = json.load(f)
    
    print(folder)
    print(label_info.keys())

    # List of keys you want to extract
    wanted_keys = ["exist", "gt_rect", "LR", "VE", "OC", "FM", "SV", "IC", "DBC", "TS"]

    # Build a new dictionary only with available keys
    data = {key: label_info[key][100:min(len(label_info[key]),500)] for key in wanted_keys if key in label_info}
    

    for i in range(101, min(len(data['gt_rect']),500)):
        if os.path.exists(os.path.join(dataset_dir, folder,f'{i:06d}.jpg')):
            image_name = f'{i:06d}.jpg'
        else:
            image_name = f'{i:06d}.png'
        
        label_file = "IR_label.json"
        Path(os.path.join(output_root,folder, image_name)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(os.path.join(dataset_dir, folder,image_name), os.path.join(output_root,folder, image_name))
        # Write dictionary to a .json file
        with open(os.path.join(output_root,folder,'IR_label.json'), 'w') as f:
            json.dump(data, f, indent=4)
        
