import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from glob import glob

class DroneDataset(Dataset):
    """
    Custom dataset for drone tracking data.
    This dataset works with the drone data structure where each video sequence
    is in its own folder with an IR_label.json file and image frames.
    """
    
    def __init__(self, root_dir, reference_sz=127, search_sz=255, 
                 final_sz=33, pos_thr=16, max_frame_sep=50,
                 img_read_fcn=None, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the drone sequences
            reference_sz (int): Size of the reference/template image
            search_sz (int): Size of the search area image
            final_sz (int): Size of the final response map
            pos_thr (int): Positive threshold (pixels) for the target location
            max_frame_sep (int): Maximum frame separation for pairs
            img_read_fcn (callable): Function to read images
            transforms (callable, optional): Optional transform to be applied on samples
        """
        self.root_dir = root_dir
        self.reference_sz = reference_sz
        self.search_sz = search_sz
        self.final_sz = final_sz
        self.pos_thr = pos_thr
        self.max_frame_sep = max_frame_sep
        self.transforms = transforms
        
        if img_read_fcn is None:
            self.img_read_fcn = lambda x: np.array(Image.open(x).convert('RGB'))
        else:
            self.img_read_fcn = img_read_fcn
        
        # Find all sequences (folders in the root directory)
        self.sequences = []
        for seq_path in glob(os.path.join(root_dir, '*')):
            if os.path.isdir(seq_path):
                self.sequences.append(seq_path)
        
        if len(self.sequences) == 0:
            raise ValueError(f"No sequences found in {root_dir}")
        
        print(f"Found {len(self.sequences)} sequences in {root_dir}")
        
        # Build frame pairs for training
        self.frame_pairs = self._build_frame_pairs()
        
    def _build_frame_pairs(self):
        """
        Builds pairs of frames for training. Each pair consists of:
        - A reference frame with the target
        - A search frame with the target
        """
        frame_pairs = []
        
        for seq_path in self.sequences:
            # Load the annotation file
            annotation_file = os.path.join(seq_path, 'IR_label.json')
            if not os.path.exists(annotation_file):
                print(f"Warning: No IR_label.json found in {seq_path}, skipping")
                continue
            
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
            
            # Get bounding boxes
            gt_rects = annotations.get('gt_rect', [])
            exists = annotations.get('exist', [])
            
            if len(gt_rects) == 0:
                print(f"Warning: No bounding boxes found in {seq_path}, skipping")
                continue
            
            # Get frame paths
            frame_paths = []
            for root, dirs, files in os.walk(seq_path):
                # Skip the root directory itself
                if root == seq_path:
                    continue
                
                for file in files:
                    if file.endswith('.jpg') or file.endswith('.png'):
                        frame_paths.append(os.path.join(root, file))
            
            # Sort frame paths to ensure sequential order
            frame_paths.sort()
            
            # Match frames to annotations
            for i in range(len(frame_paths)):
                # Skip if the target doesn't exist in this frame
                if i >= len(exists) or not exists[i]:
                    continue
                
                # Use this frame as a reference
                ref_frame_idx = i
                ref_frame_path = frame_paths[ref_frame_idx]
                ref_bbox = gt_rects[ref_frame_idx]
                
                # Create pairs with frames within max_frame_sep distance
                for j in range(max(0, i - self.max_frame_sep), min(len(frame_paths), i + self.max_frame_sep + 1)):
                    # Skip reference frame itself
                    if j == i:
                        continue
                    
                    # Skip if the target doesn't exist in this frame
                    if j >= len(exists) or not exists[j]:
                        continue
                    
                    search_frame_idx = j
                    search_frame_path = frame_paths[search_frame_idx]
                    search_bbox = gt_rects[search_frame_idx]
                    
                    # Add the pair
                    frame_pairs.append({
                        'reference': {
                            'path': ref_frame_path,
                            'bbox': ref_bbox
                        },
                        'search': {
                            'path': search_frame_path,
                            'bbox': search_bbox
                        }
                    })
        
        print(f"Created {len(frame_pairs)} frame pairs for training")
        return frame_pairs
    
    def _get_crop(self, img, bbox, crop_size):
        """
        Get a crop centered on the target with context
        
        Args:
            img: Image to crop from
            bbox: Bounding box [x, y, w, h]
            crop_size: Size of the crop
        
        Returns:
            crop: Cropped image
        """
        x, y, w, h = bbox
        
        # Get center of the bounding box
        cx = x + w/2
        cy = y + h/2
        
        # Convert to integers for indexing
        cx, cy = int(cx), int(cy)
        
        # Calculate crop bounds
        half_crop = crop_size // 2
        x1 = max(0, cx - half_crop)
        y1 = max(0, cy - half_crop)
        x2 = min(img.shape[1], cx + half_crop)
        y2 = min(img.shape[0], cy + half_crop)
        
        # Crop the image
        crop = img[y1:y2, x1:x2]
        
        # Pad if necessary
        pad_x1 = max(0, half_crop - cx)
        pad_y1 = max(0, half_crop - cy)
        pad_x2 = max(0, cx + half_crop - img.shape[1])
        pad_y2 = max(0, cy + half_crop - img.shape[0])
        
        if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
            crop = np.pad(crop, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode='constant')
        
        # Resize if necessary
        if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
            pil_img = Image.fromarray(crop)
            pil_img = pil_img.resize((crop_size, crop_size), Image.BILINEAR)
            crop = np.array(pil_img)
        
        return crop
    
    def _create_label(self, response_size, pos_thr):
        """
        Create a label for training (Gaussian-shaped response map)
        
        Args:
            response_size: Size of the response map
            pos_thr: Positive threshold for the target location
        
        Returns:
            label: Gaussian-shaped response map
        """
        # Create a label with a Gaussian centered at the center of the response map
        label = np.zeros((response_size, response_size), dtype=np.float32)
        
        # Generate coordinates grid
        y, x = np.ogrid[-response_size//2:response_size//2, -response_size//2:response_size//2]
        
        # Calculate squared distance from center
        dist_from_center_sq = x*x + y*y
        
        # Create Gaussian response
        sigma = pos_thr / 3  # 3 sigma rule for the threshold
        label = np.exp(-0.5 * dist_from_center_sq / (sigma*sigma))
        
        return label
    
    def __len__(self):
        return len(self.frame_pairs)
    
    def __getitem__(self, idx):
        """
        Get a training pair consisting of reference and search images
        
        Args:
            idx: Index of the pair to get
        
        Returns:
            sample: Dictionary containing reference image, search image, and response label
        """
        pair = self.frame_pairs[idx]
        
        # Load the reference and search images
        ref_image = self.img_read_fcn(pair['reference']['path'])
        search_image = self.img_read_fcn(pair['search']['path'])
        
        # PIL already loads as RGB, so no conversion needed
        # Just make sure we have 3 channels
        if len(ref_image.shape) == 2:  # If grayscale
            ref_image = np.stack((ref_image,)*3, axis=-1)
        if len(search_image.shape) == 2:  # If grayscale
            search_image = np.stack((search_image,)*3, axis=-1)
        
        # Get crops
        ref_crop = self._get_crop(ref_image, pair['reference']['bbox'], self.reference_sz)
        search_crop = self._get_crop(search_image, pair['search']['bbox'], self.search_sz)
        
        # Create label
        label = self._create_label(self.final_sz, self.pos_thr)
        
        # Convert to torch tensors (C, H, W format)
        ref_crop = torch.from_numpy(ref_crop.transpose(2, 0, 1)).float()
        search_crop = torch.from_numpy(search_crop.transpose(2, 0, 1)).float()
        label = torch.from_numpy(label).float()
        
        # Apply transforms if any
        if self.transforms:
            ref_crop = self.transforms(ref_crop)
            search_crop = self.transforms(search_crop)
        
        # Create sample dictionary
        sample = {
            'template': ref_crop,
            'search': search_crop,
            'label': label
        }
        
        return sample


def create_drone_datasets(root_dir, reference_sz=127, search_sz=255, 
                         final_sz=33, pos_thr=16, max_frame_sep=50,
                         img_read_fcn=None, transforms=None, split=0.2):
    """
    Create training and validation datasets
    
    Args:
        root_dir: Root directory with sequences
        reference_sz: Size of reference image
        search_sz: Size of search image  
        final_sz: Size of the final response map
        pos_thr: Positive threshold
        max_frame_sep: Maximum frame separation
        img_read_fcn: Function to read images
        transforms: Transforms to apply
        split: Validation split ratio
    
    Returns:
        train_dataset, val_dataset: Training and validation datasets
    """
    # Get list of all sequence directories
    sequences = []
    for seq_path in glob(os.path.join(root_dir, '*')):
        if os.path.isdir(seq_path) and not os.path.basename(seq_path).startswith('_tmp_'):
            sequences.append(seq_path)
    
    # Shuffle and split
    random.shuffle(sequences)
    split_idx = int(len(sequences) * (1 - split))
    train_seqs = sequences[:split_idx]
    val_seqs = sequences[split_idx:]
    
    # Create temporary directories for train and val
    train_dir = os.path.join(root_dir, '_tmp_train')
    val_dir = os.path.join(root_dir, '_tmp_val')
    
    # Remove existing directories if they exist
    import shutil
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    
    # Create new directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create symlinks for train sequences
    for seq in train_seqs:
        seq_name = os.path.basename(seq)
        os.symlink(seq, os.path.join(train_dir, seq_name))
    
    # Create symlinks for val sequences
    for seq in val_seqs:
        seq_name = os.path.basename(seq)
        os.symlink(seq, os.path.join(val_dir, seq_name))
    
    # Create datasets
    train_dataset = DroneDataset(
        train_dir, reference_sz, search_sz, final_sz, pos_thr, 
        max_frame_sep, img_read_fcn, transforms
    )
    
    val_dataset = DroneDataset(
        val_dir, reference_sz, search_sz, final_sz, pos_thr,
        max_frame_sep, img_read_fcn, transforms
    )
    
    return train_dataset, val_dataset