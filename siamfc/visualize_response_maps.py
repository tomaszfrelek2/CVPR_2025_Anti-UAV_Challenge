import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import json
from training import models as mdl
from pathlib import Path

parser = argparse.ArgumentParser(description='SiamFC Response Map Visualization')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test images')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
parser.add_argument('--results_dir', type=str, default='response_maps', help='Directory to save visualizations')
parser.add_argument('--sequence', type=str, default=None, help='Specific sequence to visualize')
parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to visualize')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs(args.results_dir, exist_ok=True)

print("Loading model...")
model = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), upscale=False, corr_map_size=33, stride=4)
checkpoint = torch.load(args.model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print(f"Model loaded from {args.model_path}")

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1))
    return image_tensor

def get_crop(image, bbox, size=127):
    c, h, w = image.shape
    x, y, target_w, target_h = bbox
    
    cx = x + target_w/2
    cy = y + target_h/2
    
    context_amount = 0.5
    crop_size = max(target_w, target_h) * (1 + context_amount * 2)
    
    scale = size / crop_size
    
    half_size = crop_size / 2
    x1 = int(cx - half_size)
    y1 = int(cy - half_size)
    x2 = int(cx + half_size)
    y2 = int(cy + half_size)
    
    x1_pad = max(0, -x1)
    y1_pad = max(0, -y1)
    x2_pad = max(0, x2 - w)
    y2_pad = max(0, y2 - h)
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    crop = image[:, y1:y2, x1:x2]
    
    if x1_pad > 0 or y1_pad > 0 or x2_pad > 0 or y2_pad > 0:
        padded = torch.nn.functional.pad(crop, 
                                       (x1_pad, x2_pad, y1_pad, y2_pad), 
                                       mode='constant', value=0)
    else:
        padded = crop
    
    resized = torch.nn.functional.interpolate(padded.unsqueeze(0), 
                                         size=(size, size), 
                                         mode='bilinear', 
                                         align_corners=False).squeeze(0)
    
    return resized, scale

def visualize_response_map(template_image, search_image, response_map, bbox, output_path):
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))

    axs[0, 0].imshow(template_image.permute(1, 2, 0).cpu().numpy())
    axs[0, 0].set_title('Template')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(search_image.permute(1, 2, 0).cpu().numpy())
    axs[0, 1].set_title('Search Region')
    axs[0, 1].axis('off')

    response_np = response_map.cpu().numpy()
    im = axs[1, 0].imshow(response_np, cmap='hot')
    peak_y, peak_x = np.unravel_index(np.argmax(response_np), response_np.shape)
    axs[1, 0].plot(peak_x, peak_y, 'g+', markersize=15, markeredgewidth=3)
    axs[1, 0].set_title('Response Map (+ marks peak)')
    fig.colorbar(im, ax=axs[1, 0])

    axs[1, 1].imshow(search_image.permute(1, 2, 0).cpu().numpy())

    response_size = response_np.shape[0]
    half_size = response_size // 2
    stride = 4

    disp_x = (peak_x - half_size) * stride
    disp_y = (peak_y - half_size) * stride

    center_x = search_image.shape[2] // 2
    center_y = search_image.shape[1] // 2

    pred_x = center_x + disp_x
    pred_y = center_y + disp_y
    
    axs[1, 1].plot(pred_x, pred_y, 'r+', markersize=15, markeredgewidth=3)
    axs[1, 1].set_title('Predicted Position')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(response_np, cmap='hot')
    plt.colorbar()
    plt.plot(peak_x, peak_y, 'g+', markersize=15, markeredgewidth=3)
    plt.title(f'Response Map (Peak: {response_np.max():.4f})')
    plt.savefig(output_path.replace('.png', '_response_only.png'))
    plt.close()

def visualize_sequence(seq_path):
    print(f"Visualizing sequence: {os.path.basename(seq_path)}")

    annotation_file = os.path.join(seq_path, 'IR_label.json')
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    gt_rects = annotations['gt_rect']
    exists = annotations['exist']

    frame_paths = []
    for root, dirs, files in os.walk(seq_path):
        if root == seq_path:
            continue
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                frame_paths.append(os.path.join(root, file))
    frame_paths.sort()

    seq_name = os.path.basename(seq_path)
    vis_dir = os.path.join(args.results_dir, seq_name)
    os.makedirs(vis_dir, exist_ok=True)

    initial_frame_idx = 0
    while initial_frame_idx < len(exists) and not exists[initial_frame_idx]:
        initial_frame_idx += 1
    
    if initial_frame_idx >= len(exists):
        print(f"No valid target found in sequence {os.path.basename(seq_path)}")
        return

    current_bbox = gt_rects[initial_frame_idx]

    first_frame = preprocess_image(frame_paths[initial_frame_idx])
    template, scale = get_crop(first_frame, current_bbox, size=127)
    template = template.unsqueeze(0).to(device)

    frame_count = 0
    for i in range(initial_frame_idx, min(len(frame_paths), len(gt_rects))):
        if frame_count >= args.num_frames:
            break
            
        if not exists[i]:
            continue

        current_frame = preprocess_image(frame_paths[i])

        search_area, search_scale = get_crop(current_frame, current_bbox, size=255)
        search_area = search_area.unsqueeze(0).to(device)

        with torch.no_grad():
            response = model(template, search_area)

        visualize_response_map(
            template.squeeze(0), 
            search_area.squeeze(0), 
            response.squeeze(), 
            current_bbox,
            os.path.join(vis_dir, f"frame_{i:04d}.png")
        )

        response_np = response.squeeze().cpu().numpy()
        peak_idx = np.argmax(response_np)
        peak_y, peak_x = np.unravel_index(peak_idx, response_np.shape)

        half_size = response_np.shape[0] // 2
        stride = 4
        disp_x = (peak_x - half_size) * stride / search_scale
        disp_y = (peak_y - half_size) * stride / search_scale

        center_x = current_bbox[0] + current_bbox[2]/2 + disp_x
        center_y = current_bbox[1] + current_bbox[3]/2 + disp_y
        current_bbox = [
            center_x - current_bbox[2]/2,
            center_y - current_bbox[3]/2,
            current_bbox[2],
            current_bbox[3]
        ]
        
        frame_count += 1
        print(f"Visualized frame {i} (response peak: {response_np.max():.4f})")

if args.sequence:
    seq_path = os.path.join(args.test_dir, args.sequence)
    if os.path.exists(seq_path):
        visualize_sequence(seq_path)
    else:
        print(f"Sequence {args.sequence} not found")
else:
    test_sequences = [os.path.join(args.test_dir, d) for d in os.listdir(args.test_dir) 
                      if os.path.isdir(os.path.join(args.test_dir, d)) and not d.startswith('.')]
    
    for seq_path in test_sequences:
        visualize_sequence(seq_path)

print(f"\nVisualization completed! Results saved to {args.results_dir}")