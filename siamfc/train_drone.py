import os
import sys
import argparse
from training.drone_dataset import DroneDataset, create_drone_datasets
from torch.utils.data import DataLoader
from training import models as mdl
from training import losses
from training import trainer
from utils.timer import Timer
import logging
import torch
import numpy as np
from PIL import Image

def parse_arguments():
    parser = argparse.ArgumentParser(description='SiamFC Drone Tracking Training')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to the drone dataset directory')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Name of the experiment')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'val'],
                       help='Mode: train or val')
    parser.add_argument('--imutils_flag', type=str, default='safe',
                       choices=['fast', 'safe'],
                       help='Flag for image utilities (safe is recommended)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--max_epochs', type=int, default=5,
                       help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    
    return parser.parse_args()

def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s %(message)s',
                       datefmt='%Y-%m-%d %H:%M:%S')
    
    logging.info(f"----- Starting train script in mode: {args.mode} -----")

    setup_timer = Timer(convert=True)
    setup_timer.reset()
    logging.info("Loading datasets...")

    model = mdl.SiameseNet(mdl.BaselineEmbeddingNet(), upscale=False,
                          corr_map_size=33, stride=4)

    img_read_fcn = lambda x: np.array(Image.open(x).convert('RGB'))

    train_dataset, val_dataset = create_drone_datasets(
        args.data_dir,
        reference_sz=127, 
        search_sz=255,
        final_sz=33,
        pos_thr=16,
        max_frame_sep=50,
        img_read_fcn=img_read_fcn
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")

    criterion = losses.BalancedLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=5e-4
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    train_module = trainer.Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=args.max_epochs,
        exp_name=args.exp_name
    )

    if args.mode == 'train':
        train_module.train()
    else:
        train_module.validate()
    
    logging.info("Done!")

if __name__ == "__main__":
    main()