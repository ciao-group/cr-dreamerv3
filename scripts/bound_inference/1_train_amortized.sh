#!/bin/bash
#SBATCH --job-name=cr_bounds_train
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --gpus=2
#SBATCH --time=3-00:00:00
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --constraint="ARCH:X86&GPU_SKU:A100"
#SBATCH --account=sci-ebel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%A_%a.out
# Phase 1 (and 4): train the bound-conditioned agent, one seed per array task.
#   sbatch 1_train_amortized.sh                       # memory model 0/0
#   sbatch 1_train_amortized.sh --mem-noise 0.1       # phase 4 model
# The job requeues itself before the time limit (keeping its arguments) and
# resumes from the last checkpoint; resubmitting the same command does too.
REPO_DIR=$HOME/cr-dreamerv3
source "${REPO_DIR}/scripts/bound_inference/common.sh" "$@"

run_requeueable python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --logdir "${RUN_DIR}" \
  --run.train_ratio 32 \
  --jax.policy_devices 0,1 \
  --jax.train_devices 0,1
