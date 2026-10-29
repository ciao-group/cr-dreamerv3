#!/bin/bash
#SBATCH --job-name=cr_bounds_infer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --constraint="ARCH:X86&GPU_SKU:A100"
#SBATCH --account=sci-ebel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --output=%x_%j.out
# Phase 3: infer bounds from the human training trials, then score the
# held-out trials at the inferred bounds (SPLIT=test with a THETA_FILE is
# phase 5). Run once per memory model to compare held-out likelihoods.
SPLIT=${SPLIT:-train}
source "${REPO_DIR:-$HOME/cr-dreamerv3}/scripts/bound_inference/common.sh"

python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --script infer_bounds \
  --logdir "${RUN_DIR}/infer_${SPLIT}" \
  --run.from_checkpoint "$(latest_checkpoint)" \
  --jax.policy_devices 0 --jax.train_devices 0 \
  --inference.split "${SPLIT}" \
  --inference.candidates 256 \
  --inference.rounds 3
