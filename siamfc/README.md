### `SiamFC`

This folder contains all of our code for the **SiamFC** model.  

## SiameseFC Code for UAV/drone tracking

## This code is adapted from this GitHub repo: https://github.com/rafellerc/Pytorch-SiamFC

### Environment Setup

#### Prerequisites
- Anaconda or Miniconda
- CUDA-compatible GPU

#### Creating the Environment
To create the conda environment with all necessary dependencies:

```bash
# Create the environment first
conda env create -f baseline.yaml
conda activate siamfc

# You need to run this if you're on a Mac with Apple Silicon (AFTER CREATING AND ACTIVATING ENV)
pip install torch torchvision
```

### Project Structure

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

#### Data Format
You can find a small version of the dataset we used in the siamfc folder. It is called small_dataset.zip. Use this to run testing on our trained model. We cannot include the subsampled dataset we used for training because GitHub won't allow us to upload anything that large in size.
The dataset should be organized as follows:
```
small_dataset/
├── sequence_1/
│   ├── IR_label.json  # Annotations in [x, y, width, height] format
│   ├── frames/        # Containing image frames
├── sequence_2/
│   ├── IR_label.json
│   ├── frames/
└── ...
```

### Evaluation
You can find our trained model here: siamfc/models/siamDroneExp/best_model.pth

To evaluate our trained model:

```bash
python3 evaluate_siamfc.py \
  --test_dir PATH_TO_TEST_DATA \
  --model_path PATH_TO_MODEL \
  --results_dir RESULTS_DIRECTORY \
  --visualize
```

This will calculate metrics including mean IoU, success rate, precision, and mAP50.

### Acknowledgments

- This code is adapted from this GitHub repo: https://github.com/rafellerc/Pytorch-SiamFC
