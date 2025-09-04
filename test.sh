#!/bin/bash

############################################
# Slurm job configuration and directives   #
############################################

#SBATCH --job-name=test
#SBATCH --partition=gpu                 # Use GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16              # CPU count
#SBATCH --gpus=1                        # Request 1 GPU
#SBATCH --time=24:00:00
#SBATCH --mem=500GB                     # Memory allocation
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hjiang55@jh.edu
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

# CUDA module removed — using system-installed CUDA instead
# Explicitly export CUDA paths (optional, but recommended)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Check GPU availability
echo "Checking GPU status..."
nvidia-smi || { echo "GPU check failed! Exiting."; exit 1; }

############################################
# Move to the submission directory         #
############################################
cd "${SLURM_SUBMIT_DIR}" || { echo "Failed to change directory! Exiting."; exit 1; }

# Optimize multi-threading settings
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}



############################################
# Run your actual code                     #
############################################
export PYTHONUNBUFFERED=1
echo "Starting main.py..."
python -u SampleDisc.py -m complex --config "/users/hjiang/GenoDistance/code/config/config_paired.yaml"
echo "Finished main.py."
echo "End Time: $(date)"