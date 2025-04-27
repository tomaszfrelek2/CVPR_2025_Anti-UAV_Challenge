#!/bin/bash
#SBATCH --job-name=SiamFC_Drone 
#SBATCH --output=./trainingRun/output_%j.txt
#SBATCH --error=./trainingRun/error_%j.txt
#SBATCH --nodes=1                  
#SBATCH --ntasks=1                 
#SBATCH --cpus-per-task=4         
#SBATCH --mem=16G                  
#SBATCH --time=12:00:00            
#SBATCH --gres=gpu:2               
#SBATCH --account=pas2985

# Print job details
echo "Job started on $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Load necessary modules (adjust according to your environment)
# module load anaconda3
# module load cuda/11.3

# Activate your conda environment
source ~/.bashrc
conda activate siamfc

# Define variables
PROJECT_DIR="/fs/scratch/PAS2985/team11/Pytorch-SiamFC"
DATA_DIR="/fs/scratch/PAS2985/team11/first_100_images"
EXP_NAME="siamDroneExp"

# Navigate to project directory
cd $PROJECT_DIR

# Run the training script
python3 train_drone.py --data_dir $DATA_DIR --exp_name $EXP_NAME

# Print completion message
echo "Job finished on $(date)"