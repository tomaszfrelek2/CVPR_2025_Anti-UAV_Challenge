# CV Final Project

## A Note on Data

We were not able to provide our original training or testing data, as GitHub has file-size limits of 25MB, and even our zip files exceeded that. Instead, we've provided some small test data in the `data` folder for you to evaluate our models on.

---

## Folder Structure

### `data_modification`

This folder contains the scripts we used to subsample the original dataset into both **YOLO** and **SiamFC** formats.

### `SiamFC`

This folder contains all of our code for the **SiamFC** model.  
Instructions for running it are provided in `siamfc/readme.md`.

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

### 2. Run Notebook
Run each cell in `../CV_FINAL_PROJECT_SP25/yolo/CV_project.ipynb`, with the exception of the training cell, as we've already provided the fully trained model for you. Again, be sure to update the data paths so that they correspond with your machine.

Model evaluating is done via ultralytics commandline, and it will output the evaluation metrics in the runs folder. 
Note that the val folder is the result of the model evaluation on our actual test set (which was too large to provide) while val-2 will be the results of the model evaluating on your test set. A good sanity check is to make sure the results line-up somewhat.
