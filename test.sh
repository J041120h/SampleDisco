#!/bin/bash

############################################
# Slurm job configuration and directives   #
############################################

#SBATCH --job-name=test
#SBATCH --output=test.out            # Fixed output file
#SBATCH --error=test.err             # Fixed error file
#SBATCH --ntasks=1                   # 1 MPI task (if any)
#SBATCH --cpus-per-task=8            # 8 CPU cores
#SBATCH --time=72:00:00              # Time limit, e.g., 72 hours
#SBATCH --mail-type=ALL              # Email on job start/end/fail
#SBATCH --mail-user=hjiang55@jh.edu  # Replace with your email

############################################
# Print job info for logging and debugging #
############################################
echo "======================================"
echo " Job ID:            ${SLURM_JOB_ID}"
echo " Job Name:          ${SLURM_JOB_NAME}"
echo " Partition:         ${SLURM_JOB_PARTITION}"
echo " Node List:         ${SLURM_NODELIST}"
echo " CPUs per Task:     ${SLURM_CPUS_PER_TASK}"
echo " Memory Alloc:      800G"
echo " GPU Requested:     1"
echo " Start Time:        $(date)"
echo "======================================"

############################################
# Load modules / activate environments     #
############################################
nvidia-smi

module load conda
conda activate hongkai

############################################
# Move to the submission directory (good practice)
############################################
cd "${SLURM_SUBMIT_DIR}"

# If using multi-threading (e.g., OpenMP):
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

############################################
# Run your actual code                     #
############################################
echo "Starting main.py..."
python -u main.py
echo "Finished main.py."

echo "End Time: $(date)"
