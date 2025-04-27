#!/bin/bash
#SBATCH --job-name=SiamFC_Test
#SBATCH --output=./test_run/output_%j.txt
#SBATCH --error=./test_run/error_%j.txt
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
mkdir -p ./test_run

# Activate your conda environment
source ~/.bashrc
conda activate siamfc

# Define variables
PROJECT_DIR="/fs/scratch/PAS2985/team11/Pytorch-SiamFC"
TEST_DIR="/fs/scratch/PAS2985/team11/test_images"
MODEL_PATH="/fs/scratch/PAS2985/team11/Pytorch-SiamFC/models/siamDroneExp_debug/best_model.pth"
RESULTS_DIR="/fs/scratch/PAS2985/team11/tracking_results"

# Navigate to project directory
cd $PROJECT_DIR

# Copy the test script
cp test_siamfc.py test_siamfc.py.backup 2>/dev/null || :

# Run the tracking test script
python3 test_siamfc.py \
  --test_dir $TEST_DIR \
  --model_path $MODEL_PATH \
  --results_dir $RESULTS_DIR \
  --visualize

# Print completion message
echo "Job finished on $(date)"