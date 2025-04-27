# SiameseFC Code for UAV/drone tracking

# This code is adapted from this GitHub repo: https://github.com/rafellerc/Pytorch-SiamFC

## Environment Setup

### Prerequisites
- Anaconda or Miniconda
- CUDA-compatible GPU

### Creating the Environment
To create the conda environment with all necessary dependencies:

```bash
conda env create -f environment.yaml
conda activate siamfc
```

## Project Structure

```
.
├── training/
│   ├── drone_dataset.py    # Custom dataset for drone tracking
│   ├── losses.py           # Loss functions
│   ├── trainer.py          # Training loops and utilities
│   └── models.py           # SiamFC network architecture
├── utils/
│   └── timer.py            # Timing utilities
├── train_drone.py          # Main training script
├── evaluate_siamfc.py      # Evaluation script
├── visualize_response_maps.py  # Visualization utilities
├── environment.yaml        # Conda environment specification
└── README.md
```

## Training

To train the SiamFC model on your drone dataset:

```bash
python3 train_drone.py \
  --data_dir PATH_TO_DATASET \
  --exp_name MY_EXPERIMENT
```

### Data Format

The dataset should be organized as follows:
```
dataset/
├── sequence_1/
│   ├── IR_label.json  # Annotations in [x, y, width, height] format
│   ├── frames/        # Containing image frames
├── sequence_2/
│   ├── IR_label.json
│   ├── frames/
└── ...
```

## Evaluation

To evaluate a trained model:

```bash
python3 evaluate_siamfc.py \
  --test_dir PATH_TO_TEST_DATA \
  --model_path PATH_TO_MODEL \
  --results_dir RESULTS_DIRECTORY \
  --visualize
```

This will calculate metrics including mean IoU, success rate, precision, and mAP50.

## Acknowledgments

- This code is adapted from this GitHub repo: https://github.com/rafellerc/Pytorch-SiamFC