#!/bin/bash
#SBATCH --job-name=cr_bounds_eval
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --constraint="ARCH:X86&GPU_SKU:A100"
#SBATCH --account=sci-ebel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --output=%x_%j.out
# Phase 5: closed-loop rollouts at the inferred bounds against the held-out
# human trials. SUBJECT=RZ evaluates that subject's bounds.
SUBJECT=${SUBJECT:-}
source "${REPO_DIR:-$HOME/cr-dreamerv3}/scripts/bound_inference/common.sh"
THETA_FILE=${THETA_FILE:-${RUN_DIR}/infer_train/inference.json}

python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --script eval_human_likeness \
  --logdir "${RUN_DIR}/human_likeness${SUBJECT:+_${SUBJECT}}" \
  --run.from_checkpoint "$(latest_checkpoint)" \
  --jax.policy_devices 0 --jax.train_devices 0 \
  --inference.theta_file "${THETA_FILE}" \
  --inference.subject "${SUBJECT}" \
  --inference.split test \
  --inference.eval_steps 20000
