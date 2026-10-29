#!/bin/bash
#SBATCH --job-name=cr_bounds_train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --gpus=2
#SBATCH --time=7-00:00:00
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --constraint=ARCH:X86&GPU_SKU:A100
#SBATCH --account=sci-ebel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%A_%a.out
# Phase 1 (and 4): train the bound-conditioned agent, one seed per array task.
# Memory models for phase 4: MEM_NOISE=0.1 sbatch 1_train_amortized.sh
# The job requeues itself before the time limit and resumes from the last
# checkpoint; resubmitting with the same SEED/MEM_* resumes as well.
SEED=${SLURM_ARRAY_TASK_ID:-0}
source "${REPO_DIR:-$HOME/cr-dreamer/cr-dreamerv3}/scripts/bound_inference/common.sh"

run_requeueable python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --logdir "${RUN_DIR}" \
  --run.train_ratio 32 \
  --jax.policy_devices 0,1 \
  --jax.train_devices 0,1
