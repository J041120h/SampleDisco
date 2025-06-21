#!/bin/bash

############################################
# Slurm job configuration and directives   #
############################################

#SBATCH --job-name=plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16              # Reduced CPU count
#SBATCH --time=1:00:00                  # 1 hour
#SBATCH --mem=100GB                     # Reduced memory
#SBATCH --output=plot.out
#SBATCH --error=plot.err
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
echo " GPU Requested:     None"
echo " Start Time:        $(date)"
echo "======================================"

############################################
# Load modules / activate environments     #
############################################
module load conda
conda activate hongkai

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
echo "Starting plot script..."
python integration_CCA_test.py
echo "Finished plot script."
echo "End Time: $(date)"
