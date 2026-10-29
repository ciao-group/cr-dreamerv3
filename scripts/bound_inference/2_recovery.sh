#!/bin/bash
#SBATCH --job-name=cr_bounds_recovery
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
# Phase 2: parameter recovery on agent data with known bounds.
REPO_DIR=$HOME/cr-dreamerv3
source "${REPO_DIR}/scripts/bound_inference/common.sh" "$@"

python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --script bound_recovery \
  --logdir "${RUN_DIR}/recovery" \
  --run.from_checkpoint "$(latest_checkpoint)" \
  --jax.policy_devices 0 --jax.train_devices 0 \
  --inference.recovery_thetas 16 \
  --inference.recovery_steps 3000 \
  --inference.candidates 256 \
  --inference.rounds 3
