import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import cv2
import json
from training import models as mdl

# Parse arguments
parser = argparse.ArgumentParser(description='SiamFC Drone Tracking Testing')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test images')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
parser.add_argument('--results_dir', type=str, default='tracking_results', help='Directory to save results')
parser.add_argument('--visualize', action='store_true', help='Visualize tracking results')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create results directory
os.makedirs(args.results_dir, exist_ok=True)

# Load the trained model
print("Loading model...")
model = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), upscale=False, corr_map_size=33, stride=4)
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"Model loaded from {args.model_path}")

# Function to preprocess images
def preprocess_image(image_path):
    image = np.array(Image.open(image_path).convert('RGB'))
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
    return image_tensor

# Function to get crop centered at target location
def get_crop(image, target_pos, size=127):
    h, w = image.shape[1:3]
    
    # Convert target position to int
    cx, cy = int(target_pos[0]), int(target_pos[1])
    
    # Get crop bounds
    half_size = size // 2
    x1 = max(0, cx - half_size)
    y1 = max(0, cy - half_size)
    x2 = min(w, cx + half_size)
    y2 = min(h, cy + half_size)
    
    # Crop image
    crop = image[:, y1:y2, x1:x2]
    
    # Pad if necessary
    pad_x1 = max(0, half_size - cx)
    pad_y1 = max(0, half_size - cy)
    pad_x2 = max(0, cx + half_size - w)
    pad_y2 = max(0, cy + half_size - h)
    
    if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
        crop = torch.nn.functional.pad(crop, (pad_x1, pad_x2, pad_y1, pad_y2), mode='constant', value=0)
    
    # Resize if necessary
    if crop.shape[1] != size or crop.shape[2] != size:
        crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
    
    return crop

# Function to get search area centered at target position
def get_search_area(image, target_pos, size=255):
    return get_crop(image, target_pos, size)

# Function to convert response map to target position
def response_to_position(response, target_pos, response_size=33, stride=4):
    # Get peak position
    response = response.cpu().detach().numpy()
    peak_idx = np.argmax(response.flatten())
    peak_y, peak_x = np.unravel_index(peak_idx, response.shape)
    
    # Calculate new target position
    half_size = response_size // 2
    offset_x = (peak_x - half_size) * stride
    offset_y = (peak_y - half_size) * stride
    
    new_x = target_pos[0] + offset_x
    new_y = target_pos[1] + offset_y
    
    return [new_x, new_y]

# Function to draw bounding box on image
def draw_bbox(image, pos, size=(30, 22)):
    image = image.copy()
    x, y = pos
    w, h = size
    x1, y1 = int(x - w/2), int(y - h/2)
    x2, y2 = int(x + w/2), int(y + h/2)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return image

# Function to track target in a sequence
def track_sequence(seq_path, initial_target=None):
    print(f"Processing sequence: {os.path.basename(seq_path)}")
    
    # Get all frame paths
    frame_paths = []
    for root, dirs, files in os.walk(seq_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                frame_paths.append(os.path.join(root, file))
    
    # Sort frame paths
    frame_paths.sort()
    
    if len(frame_paths) == 0:
        print(f"No frames found in {seq_path}")
        return None
    
    # If initial target not provided, select center of first frame
    if initial_target is None:
        first_frame = np.array(Image.open(frame_paths[0]).convert('RGB'))
        h, w, _ = first_frame.shape
        initial_target = [w/2, h/2]
        print(f"No initial target provided, using center: {initial_target}")
    
    # Initialize tracking
    target_pos = initial_target
    target_size = (30, 22)  # Default size based on your dataset
    tracking_results = []
    
    # Initialize visualization if enabled
    if args.visualize:
        seq_name = os.path.basename(seq_path)
        vis_dir = os.path.join(args.results_dir, 'visualizations', seq_name)
        os.makedirs(vis_dir, exist_ok=True)
    
    # Process first frame to get template
    first_frame = preprocess_image(frame_paths[0])
    template = get_crop(first_frame, target_pos)
    template = template.unsqueeze(0).to(device)
    
    # Save first position
    tracking_results.append({
        'frame': os.path.basename(frame_paths[0]),
        'position': [float(target_pos[0]), float(target_pos[1])],
        'size': [float(target_size[0]), float(target_size[1])]
    })
    
    # Visualize first frame if needed
    if args.visualize:
        first_frame_vis = np.array(Image.open(frame_paths[0]).convert('RGB'))
        first_frame_vis = draw_bbox(first_frame_vis, target_pos, target_size)
        cv2.imwrite(os.path.join(vis_dir, f"frame_0.jpg"), cv2.cvtColor(first_frame_vis, cv2.COLOR_RGB2BGR))
    
    # Track through the sequence
    for i in range(1, len(frame_paths)):
        # Load and preprocess current frame
        current_frame = preprocess_image(frame_paths[i])
        
        # Get search area
        search_area = get_search_area(current_frame, target_pos)
        search_area = search_area.unsqueeze(0).to(device)
        
        # Run model to get response map
        with torch.no_grad():
            response = model(template, search_area)
        
        # Update target position
        target_pos = response_to_position(response.squeeze(), target_pos)
        
        # Save tracking result
        tracking_results.append({
            'frame': os.path.basename(frame_paths[i]),
            'position': [float(target_pos[0]), float(target_pos[1])],
            'size': [float(target_size[0]), float(target_size[1])]
        })
        
        # Visualize if needed
        if args.visualize:
            current_frame_vis = np.array(Image.open(frame_paths[i]).convert('RGB'))
            current_frame_vis = draw_bbox(current_frame_vis, target_pos, target_size)
            cv2.imwrite(os.path.join(vis_dir, f"frame_{i}.jpg"), cv2.cvtColor(current_frame_vis, cv2.COLOR_RGB2BGR))
        
        # Print progress
        if i % 10 == 0:
            print(f"Processed {i}/{len(frame_paths)} frames")
    
    return tracking_results

# Process all test sequences
test_sequences = [os.path.join(args.test_dir, d) for d in os.listdir(args.test_dir) 
                  if os.path.isdir(os.path.join(args.test_dir, d))]

results = {}
for seq_path in test_sequences:
    seq_name = os.path.basename(seq_path)
    
    # Skip .DS_Store and other hidden files/directories
    if seq_name.startswith('.'):
        continue
    
    # Track the sequence
    seq_results = track_sequence(seq_path)
    
    if seq_results:
        # Save results to dictionary
        results[seq_name] = seq_results
        
        # Also save individual sequence results
        with open(os.path.join(args.results_dir, f"{seq_name}_tracking.json"), 'w') as f:
            json.dump(seq_results, f, indent=2)

# Save all results to a single file
with open(os.path.join(args.results_dir, "all_tracking_results.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"Tracking completed! Results saved to {args.results_dir}")