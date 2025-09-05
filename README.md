# CVPR 2025 Anti-UAV Challenge
Read our project report at ``` report.pdf```
## A Note on Data

We were not able to provide our original training or testing data, as GitHub has file-size limits of 25MB, and even our zip files exceeded that. Instead, we've provided some small test data in siamfc and yolo folders for you to evaluate our models on.

---

## Folder Structure

### `data_modification`

This folder contains the scripts we used to subsample the original dataset into both **YOLO** and **SiamFC** formats.

### `SiamFC`

This folder contains all of our code for running the **SiamFC** model.  

### `yolo`

This folder contains all of our code for running the **YOLO11## model.

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

### Visualizations

You can unzip these files:
- siamfc/evaluation_results.zip
- siamfc/response_visualizations.zip

These folders will include validation images/visualizations for SiamFC

### Acknowledgments

- This code is adapted from this GitHub repo: https://github.com/rafellerc/Pytorch-SiamFC

---

## Running YOLO11

### 1. Install Jupyter Notebook Dependencies

You must have the **Jupyter Notebook** extension installed on your machine.  
If you don't, run the following commands:

```bash
pip install notebook
pip install ipykernel
```
Note- You may have to restart your IDE for this to work properly

### 2. Modify File Paths

Next you will need to modify the file paths in `../CV_FINAL_PROJECT_SP25/yolo_data/data.yaml` so that the ".." portion correspond with your machine path for that file.

### 3. Run Notebook
The notebook contains the commands for running/evaluating the model on test data, there's no need to train a new model as we've provided one for you.

Run each cell in `../CV_FINAL_PROJECT_SP25/yolo/CV_project.ipynb`. Again, be sure to update the data paths so that they correspond with your machine.

Note- After running the first cell ensure that your Ultralytics version is `8.3.40`, YOLO11 is a new architecture and requires the most up-to date version. If you have a cached version of ultralytics, or for whatever reason cannot install/upgrade to the correct version, you can upload the notebook to colab and run the model there, as colab should install it correctly by default. However to evaluate the model remember to upload the yolo_data folder to the colab instance.

### 4. Output
Model evaluating is done via ultralytics commandline, and it will output the evaluation metrics in the runs folder. 
Note that the val folder is the result of the model evaluation on our actual test set (which was too large to provide) while val-2 will be the results of the model evaluating on your test set. A good sanity check is to make sure the results line-up somewhat.
