#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128000
#SBATCH --gpus=a30:1
#SBATCH --time=24:00:00
#SBATCH -e Offline_logs/jobfiles/log/%x.error-%j
#SBATCH --partition=paula
# Phase 3: infer bounds from the human training trials, then score the
# held-out trials at the inferred bounds (SPLIT=test with a THETA_FILE is
# phase 5). Run once per memory model to compare held-out likelihoods.
SPLIT=${SPLIT:-train}
source "${REPO_DIR:-$HOME/cr-dreamer/cr-dreamerv3}/scripts/bound_inference/common.sh"

python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --script infer_bounds \
  --logdir "${RUN_DIR}/infer_${SPLIT}" \
  --run.from_checkpoint "$(latest_checkpoint)" \
  --jax.policy_devices 0 --jax.train_devices 0 \
  --inference.split "${SPLIT}" \
  --inference.candidates 256 \
  --inference.rounds 3
