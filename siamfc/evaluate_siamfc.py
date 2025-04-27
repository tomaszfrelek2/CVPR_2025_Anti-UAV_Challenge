import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import argparse
import json
from training import models as mdl
from pathlib import Path
from collections import defaultdict

parser = argparse.ArgumentParser(description='SiamFC Drone Tracking Evaluation with mAP50')
parser.add_argument('--test_dir', type=str, required=True, help='Path to test images')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Directory to save results')
parser.add_argument('--visualize', action='store_true', help='Visualize tracking results')
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

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def calculate_ap(ious, threshold=0.5):
    true_positives = [1 if iou >= threshold else 0 for iou in ious]
    false_positives = [0 if iou >= threshold else 1 for iou in ious]

    valid_indices = [i for i, iou in enumerate(ious) if iou > 0 or true_positives[i] > 0 or false_positives[i] > 0]
    
    if not valid_indices:
        return 0.0

    true_positives = [true_positives[i] for i in valid_indices]
    false_positives = [false_positives[i] for i in valid_indices]

    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)

    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (len(valid_indices) + 1e-6)

    precisions = np.concatenate(([0], precisions))
    recalls = np.concatenate(([0], recalls))

    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11
    
    return ap

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

def response_to_bbox(response, center_pos, target_size, scale, response_size=33, stride=4):
    response = response.cpu().detach().numpy()
    peak_idx = np.argmax(response.flatten())
    peak_y, peak_x = np.unravel_index(peak_idx, response.shape)

    half_size = response_size // 2
    disp_x = (peak_x - half_size) * stride / scale
    disp_y = (peak_y - half_size) * stride / scale

    new_center_x = center_pos[0] + disp_x
    new_center_y = center_pos[1] + disp_y

    w, h = target_size
    x = new_center_x - w/2
    y = new_center_y - h/2
    
    return [x, y, w, h]

def evaluate_sequence(seq_path):
    print(f"Evaluating sequence: {os.path.basename(seq_path)}")

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

    pred_rects = []
    ious = []

    if args.visualize:
        seq_name = os.path.basename(seq_path)
        vis_dir = os.path.join(args.results_dir, 'visualizations', seq_name)
        os.makedirs(vis_dir, exist_ok=True)

    initial_frame_idx = 0
    while initial_frame_idx < len(exists) and not exists[initial_frame_idx]:
        initial_frame_idx += 1
    
    if initial_frame_idx >= len(exists):
        print(f"No valid target found in sequence {os.path.basename(seq_path)}")
        return None

    current_bbox = gt_rects[initial_frame_idx]

    first_frame = preprocess_image(frame_paths[initial_frame_idx])
    template, scale = get_crop(first_frame, current_bbox, size=127)
    template = template.unsqueeze(0).to(device)

    for i in range(initial_frame_idx, min(len(frame_paths), len(gt_rects))):
        if not exists[i]:
            pred_rects.append([0, 0, 0, 0])
            ious.append(0)
            continue

        current_frame = preprocess_image(frame_paths[i])

        center_x = current_bbox[0] + current_bbox[2]/2
        center_y = current_bbox[1] + current_bbox[3]/2

        search_area, search_scale = get_crop(current_frame, current_bbox, size=255)
        search_area = search_area.unsqueeze(0).to(device)

        with torch.no_grad():
            response = model(template, search_area)

        current_bbox = response_to_bbox(response.squeeze(), 
                                      [center_x, center_y], 
                                      [current_bbox[2], current_bbox[3]], 
                                      search_scale)

        iou = compute_iou(current_bbox, gt_rects[i])

        pred_rects.append(current_bbox)
        ious.append(iou)

        if args.visualize:
            img = Image.open(frame_paths[i]).convert('RGB')
            draw = ImageDraw.Draw(img)

            x, y, w, h = gt_rects[i]
            draw.rectangle([int(x), int(y), int(x+w), int(y+h)], outline='green', width=2)

            x, y, w, h = current_bbox
            draw.rectangle([int(x), int(y), int(x+w), int(y+h)], outline='red', width=2)

            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            draw.text((10, 10), f"IoU: {iou:.2f}", fill='white', font=font)

            img.save(os.path.join(vis_dir, f"frame_{i:04d}.jpg"))

    valid_ious = [iou for iou, exist in zip(ious, exists[initial_frame_idx:]) if exist]
    ap50 = calculate_ap(valid_ious, threshold=0.5)
    
    metrics = {
        'mean_iou': np.mean(valid_ious) if valid_ious else 0,
        'num_frames': len(valid_ious),
        'success_rate': sum(iou > 0.5 for iou in valid_ious) / len(valid_ious) if valid_ious else 0,
        'precision': sum(iou > 0.2 for iou in valid_ious) / len(valid_ious) if valid_ious else 0,
        'ap50': ap50
    }
    
    return metrics, pred_rects, ious

test_sequences = [os.path.join(args.test_dir, d) for d in os.listdir(args.test_dir) 
                  if os.path.isdir(os.path.join(args.test_dir, d)) and not d.startswith('.')]

overall_results = {}
all_metrics = []

for seq_path in test_sequences:
    seq_name = os.path.basename(seq_path)

    result = evaluate_sequence(seq_path)
    
    if result:
        metrics, pred_rects, ious = result
        overall_results[seq_name] = {
            'metrics': metrics,
            'predicted_rects': pred_rects,
            'ious': ious
        }
        all_metrics.append(metrics)
        
        print(f"Sequence {seq_name}:")
        print(f"  Mean IoU: {metrics['mean_iou']:.3f}")
        print(f"  Success Rate: {metrics['success_rate']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  AP50: {metrics['ap50']:.3f}")
        print("")

if all_metrics:
    overall_mean_iou = np.mean([m['mean_iou'] for m in all_metrics])
    overall_success_rate = np.mean([m['success_rate'] for m in all_metrics])
    overall_precision = np.mean([m['precision'] for m in all_metrics])
    overall_ap50 = np.mean([m['ap50'] for m in all_metrics])
    
    print("\nOverall Performance:")
    print(f"Mean IoU: {overall_mean_iou:.3f}")
    print(f"Success Rate: {overall_success_rate:.3f}")
    print(f"Precision: {overall_precision:.3f}")
    print(f"mAP50: {overall_ap50:.3f}")

    overall_results['overall'] = {
        'mean_iou': overall_mean_iou,
        'success_rate': overall_success_rate,
        'precision': overall_precision,
        'mAP50': overall_ap50
    }

    with open(os.path.join(args.results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=2)

    plt.figure(figsize=(10, 8))
    metrics_names = ['Mean IoU', 'Success Rate', 'Precision', 'mAP50']
    metrics_values = [overall_mean_iou, overall_success_rate, overall_precision, overall_ap50]
    
    plt.bar(metrics_names, metrics_values)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Overall Tracking Performance')
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.savefig(os.path.join(args.results_dir, 'performance_summary.png'))
    plt.close()

print(f"\nEvaluation completed! Results saved to {args.results_dir}")