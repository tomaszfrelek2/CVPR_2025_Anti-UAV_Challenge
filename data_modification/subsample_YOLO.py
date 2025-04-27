import os
import random
import shutil
from pathlib import Path
import json
from PIL import Image



def convert_to_yolo(img_pth, exist, box):

    x, y, w, h = box[0], box[1], box[2], box[3]
    image = Image.open(img_pth)
    img_w, img_h = image.size

    # Convert to YOLO format (normalized)
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h

    # Example class ID (use 0 if you don't have class labels)
    class_id = 0
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

# ==== CONFIGURATION ====
dataset_dir = Path("/Users/tomaszfrelek/Downloads/train")  # Has images/ and labels/ folders
output_root = Path("/Users/tomaszfrelek/Downloads/MultiUAV_Baseline_code_and_submissi/data")          # Output folder root
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
seed = 42

# ========================
random.seed(seed)

# Replace this with the actual path to your master folder

# Get all subfolders (only directories) in the master folder
subfolders = [name for name in os.listdir(dataset_dir)
              if os.path.isdir(os.path.join(dataset_dir, name))]

# print("Subfolders found:")
# for folder in subfolders:
#     print(folder)

# Input folders
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

# Output folder setup
for split in ["train", "val", "test"]:
    (output_root / split / "images").mkdir(parents=True, exist_ok=True)
    (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
image_nums = 0
for folder in subfolders:
    label_file = os.path.join(dataset_dir, folder,"IR_label.json")

    with open(label_file, 'r') as f:
        label_info = json.load(f)
        
    generated_nums = []

    # Build a new dictionary only with available keys
    data = {'exist': [], 'gt_rect': []}
    
    num_files = len([f for f in os.listdir(os.path.join(dataset_dir, folder)) if os.path.isfile(os.path.join(os.path.join(dataset_dir, folder), f))])
    print(folder)

    for i in range(100):
        rand_num = random.randint(0, num_files - 2)
        while rand_num in generated_nums:
            rand_num = random.randint(0, num_files -2)
        generated_nums.append(rand_num)
        if os.path.exists(os.path.join(dataset_dir, folder,f'{rand_num+1:06d}.jpg')):
            image_name = f'{rand_num+1:06d}.jpg'
        else:
            image_name = f'{rand_num+1:06d}.png'

        img_path = os.path.join(dataset_dir, folder,image_name)

        print(rand_num)
        for key in data.keys():
            data[key].append(label_info[key][rand_num])
        
        if data['exist'][i]:
            yolo_label = convert_to_yolo(img_path,data['exist'][i],data['gt_rect'][i])
        else:
            yolo_label = ''
        #FIX
        
        #put image in train set
        if i < 80:
            shutil.copy2(img_path, os.path.join(output_root, "train","images",f'{image_nums:06d}.jpg'))
            with open(os.path.join(output_root,"train","labels",f'{image_nums:06d}.txt'), 'w') as f:
                f.write(yolo_label)
        #put image in 
        elif i - 80 < 10: 
            shutil.copy2(img_path, os.path.join(output_root,"val","images",f'{image_nums:06d}.jpg'))
            with open(os.path.join(output_root,"val","labels",f'{image_nums:06d}.txt'), 'w') as f:
                f.write(yolo_label)

        else:
            shutil.copy2(img_path, os.path.join(output_root, "test","images",f'{image_nums:06d}.jpg'))
            with open(os.path.join(output_root,"test","labels",f'{image_nums:06d}.txt'), 'w') as f:
                f.write(yolo_label)
        
        image_nums+=1






