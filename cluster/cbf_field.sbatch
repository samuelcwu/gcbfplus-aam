#!/bin/bash
# Barrier-field video (scratch/cbf_field_video.py) on one GPU. Submit from the repo root:
#     sbatch cluster/cbf_field.sbatch                                  # episode 2, agent 1 (defaults)
#     sbatch --export=ALL,EPISODE=4,AGENT=3 cluster/cbf_field.sbatch   # another episode / agent
# Watch:   squeue --me        Log:  tail -f logs/cbf_field_<jobid>.out        Cancel:  scancel <jobid>
# Output:  scratch/videos/cbf_field/*.mp4 and scratch/figures/cbf_field_*_stills.png
#
# Lines starting with #SBATCH are options for the scheduler (same as passing them to sbatch on the command line):
#SBATCH --job-name=cbf_field
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail

echo "job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gcbfplus

cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg                     # no display on compute nodes
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # don't grab 75% of GPU memory up front

python -u scratch/cbf_field_video.py

echo "finished $(date)"
