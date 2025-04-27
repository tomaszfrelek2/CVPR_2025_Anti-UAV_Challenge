import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image, ImageDraw
from pathlib import Path

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize SiamFC Drone Tracking Results')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test images')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to tracking results')
    parser.add_argument('--output_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--sequence', type=str, default=None, help='Specific sequence to visualize (default: all)')
    parser.add_argument('--create_video', action='store_true', help='Create video of tracking results')
    return parser.parse_args()

def visualize_sequence(seq_name, test_dir, results_file, output_dir, create_video=False):
    # Load tracking results
    with open(results_file, 'r') as f:
        tracking_results = json.load(f)
    
    # Get sequence directory
    seq_dir = os.path.join(test_dir, seq_name)
    
    # Find all frames in the sequence
    frame_paths = []
    for root, dirs, files in os.walk(seq_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                frame_paths.append(os.path.join(root, file))
    
    # Sort frame paths
    frame_paths.sort()
    
    if len(frame_paths) == 0:
        print(f"No frames found in {seq_dir}")
        return
    
    # Create output directory for sequence
    seq_output_dir = os.path.join(output_dir, seq_name)
    os.makedirs(seq_output_dir, exist_ok=True)
    
    # Create visualization for each frame
    frames = []
    for i, result in enumerate(tracking_results):
        if i >= len(frame_paths):
            break
            
        # Load frame image
        img = Image.open(frame_paths[i]).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Get target position and size
        pos_x, pos_y = result['position']
        width, height = result['size']
        
        # Calculate bbox coordinates
        x1 = int(pos_x - width/2)
        y1 = int(pos_y - height/2)
        x2 = int(pos_x + width/2)
        y2 = int(pos_y + height/2)
        
        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        
        # Add frame number
        draw.text((10, 10), f"Frame: {i}", fill=(255, 0, 0))
        
        # Save visualized frame
        output_path = os.path.join(seq_output_dir, f"frame_{i:04d}.jpg")
        img.save(output_path)
        
        # Append for video if needed
        if create_video:
            frames.append(np.array(img))
    
    print(f"Saved {len(tracking_results)} visualized frames to {seq_output_dir}")
    
    # Create video if requested
    if create_video and frames:
        create_tracking_video(frames, os.path.join(output_dir, f"{seq_name}_tracking.mp4"))

def create_tracking_video(frames, output_path):
    """Create a video from frames using matplotlib animation"""
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    
    # Create animation
    ims = [[plt.imshow(frame, animated=True)] for frame in frames]
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    
    # Save animation
    ani.save(output_path)
    plt.close(fig)
    
    print(f"Saved tracking video to {output_path}")

def main():
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get sequences to process
    if args.sequence:
        sequences = [args.sequence]
    else:
        # Get all results files
        results_files = [f for f in os.listdir(args.results_dir) 
                         if f.endswith('_tracking.json') and not f.startswith('all')]
        sequences = [f.replace('_tracking.json', '') for f in results_files]
    
    # Process each sequence
    for seq_name in sequences:
        results_file = os.path.join(args.results_dir, f"{seq_name}_tracking.json")
        
        if not os.path.exists(results_file):
            print(f"Results file not found for {seq_name}")
            continue
        
        print(f"Visualizing results for {seq_name}...")
        visualize_sequence(seq_name, args.test_dir, results_file, 
                          args.output_dir, args.create_video)

if __name__ == "__main__":
    main()