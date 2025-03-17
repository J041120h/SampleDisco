#!/bin/bash

############################################
# Slurm job configuration and directives   #
############################################

#SBATCH --job-name=test
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32              # Increased to 4 CPUs per task for better parallelism
#SBATCH --gres=gpu:1                    # Explicitly request 1 GPU
#SBATCH --time=2:00:00
#SBATCH --mem=400GB                     
#SBATCH --output=test.out            # Output file with Job ID
#SBATCH --error=test.err             # Error file with Job ID
#SBATCH --mail-type=END,FAIL            # Only notify on job completion or failure
#SBATCH --mail-user=hjiang55@jh.edu     # Replace with your email
#SBATCH --array=0

############################################
# Print job info for logging and debugging #
############################################
echo "======================================"
echo " Job ID:            ${SLURM_JOB_ID}"
echo " Job Name:          ${SLURM_JOB_NAME}"
echo " Partition:         ${SLURM_JOB_PARTITION}"
echo " Node List:         ${SLURM_NODELIST}"
echo " CPUs per Task:     ${SLURM_CPUS_PER_TASK}"
echo " Memory Alloc:      ${SLURM_MEM_PER_NODE}"
echo " GPU Requested:     1"
echo " Start Time:        $(date)"
echo "======================================"

############################################
# Load modules / activate environments     #
############################################
module load conda
conda activate hongkai

# Load CUDA and cuDNN modules (ensure compatibility)
module load cuda/12.4

# Check GPU availability
echo "Checking GPU status..."
nvidia-smi || { echo "GPU check failed! Exiting."; exit 1; }

############################################
# Move to the submission directory (good practice)
############################################
cd "${SLURM_SUBMIT_DIR}" || { echo "Failed to change directory! Exiting."; exit 1; }

# Optimize multi-threading settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

############################################
# Run your actual code                     #
############################################
echo "Starting main.py..."
python -u main.py || { echo "Python script failed! Exiting."; exit 1; }
echo "Finished main.py."

echo "End Time: $(date)"
