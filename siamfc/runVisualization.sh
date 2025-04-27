#!/bin/bash
#SBATCH --job-name=SiamFC_Visualization
#SBATCH --output=./visualize_run/output_%j.txt
#SBATCH --error=./visualize_run/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:2
#SBATCH --account=pas2985

# Print job details
echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"

# Make sure output directory exists
mkdir -p ./evaluation_run

# Activate your conda environment
source ~/.bashrc
conda activate siamfc

# # Define variables
# PROJECT_DIR="/fs/scratch/PAS2985/team11/Pytorch-SiamFC"
# TEST_DIR="/fs/scratch/PAS2985/team11/test_images"
# MODEL_PATH="/fs/scratch/PAS2985/team11/Pytorch-SiamFC/models/siamDroneExp/best_model.pth"
# RESULTS_DIR="/fs/scratch/PAS2985/team11/evaluation_results"

# # Navigate to project directory
# cd $PROJECT_DIR

# Run the evaluation script
python3 visualize_response_maps.py \
  --test_dir /fs/scratch/PAS2985/team11/test_images \
  --model_path /fs/scratch/PAS2985/team11/Pytorch-SiamFC/models/siamDroneExp/best_model.pth \
  --results_dir /fs/scratch/PAS2985/team11/response_visualizations \
  --num_frames 10

# Print completion message
echo "Job finished on $(date)"