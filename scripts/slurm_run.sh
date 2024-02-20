#!/bin/bash
# # SBATCH -o /home/%u/slogs/sl_%A.out
# # SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:1  # use 1 GPU
#SBATCH --mem=14000  # memory in Mb
#SBATCH --partition=ILCC_GPU_UBUNTU
#SBATCH -t 6-00:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=8

echo "Job running on ${SLURM_JOB_NODELIST}"

dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job started: $dt"

echo "Setting up bash enviroment"
source ~/.bashrc
set -e
SCRATCH_DISK=/disk/scratch
SCRATCH_HOME=${SCRATCH_DISK}/${USER}
mkdir -p ${SCRATCH_HOME}

# Activate your conda environment
PROJECT_NAME="semeval_nli4ct"
echo "Activating conda environment: ${PROJECT_NAME}"
conda activate ${PROJECT_NAME}

# echo "Moving input data to the compute node's scratch space: $SCRATCH_DISK"
src_path=/home/${USER}/${PROJECT_NAME}/data/
dest_path=${SCRATCH_HOME}/${PROJECT_NAME}/data/
mkdir -p ${dest_path}  # make it if required
rsync --archive --update --compress --progress ${src_path}/ ${dest_path}

echo "Running experiment"
python scripts/train.py experiment=$1

OUTPUT_DIR=${SCRATCH_HOME}/${PROJECT_NAME}/outputs/
OUTPUT_HOME=/home/${USER}/${PROJECT_NAME}/outputs/
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# Cleanup
rm -rf ${OUTPUT_DIR}

echo ""
echo "============"
echo "job finished successfully"
dt=$(date '+%d/%m/%Y %H:%M:%S')
echo "Job finished: $dt"