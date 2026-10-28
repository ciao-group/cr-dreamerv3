#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512000
#SBATCH --gpus=a30:8
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH -e Offline_logs/jobfiles/log/%x.error-%A_%a
#SBATCH --partition=paula
# Phase 1 (and 4): train the bound-conditioned agent, one seed per array task.
# Memory models for phase 4: MEM_NOISE=0.1 sbatch 1_train_amortized.sh
SEED=${SLURM_ARRAY_TASK_ID:-0}
source "${REPO_DIR:-$HOME/cr-dreamer/cr-dreamerv3}/scripts/bound_inference/common.sh"

python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --logdir "${RUN_DIR}" \
  --run.train_ratio 32 \
  --jax.policy_devices 0,1,2,3,4,5,6,7 \
  --jax.train_devices 0,1,2,3,4,5,6,7
