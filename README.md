CV Final Project

A note on data: We were not able to provide our original training or testing data, as githib has file-size limits of 25mb, and even our zip files were larger than that. Instead, we've provided some short test data in the data folder for you to evaluate our models on.

data_modification: The data_modification folder contains the scripts that we used to subsample the original dataset into both YOLO and SiamFC formats


SiamFC: This folder contains all of our code for the SiamFC model. Instructions for running are in siamfc/readme.md



To run YOLO11: 
You must have the jupyter notebbok extension installed on your machine, if you don't run the follwoing commands:
pip install notebook
pip install ipykernel

Note- You may have to restart your IDE for this to work properly


Next you will need to modify the file paths in ../CV_FINAL_PROJECT_SP25/yolo_data/data.yaml so that the ".." portion correspond with your machine path for that file.

Run each cell in ../CV_FINAL_PROJECT_SP25/yolo/CV_project.ipynb, with the exception of the training cell, as we've already provided the fully trained model for you. Again, be sure to update the data paths so that they correspond with your machine.

Model evaluating is done via ultralytics commandline, and it will output the evaluation metrics in the runs folder. 
Note that the val folder is the result of the model evaluation on our actual test set (which was too large to provide) while val-2 will be the results of the model evaluating on your test set. A good sanity check is to make sure the results line-up somewhat.
