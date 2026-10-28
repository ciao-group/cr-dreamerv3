#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128000
#SBATCH --gpus=a30:1
#SBATCH --time=12:00:00
#SBATCH -e Offline_logs/jobfiles/log/%x.error-%j
#SBATCH --partition=paula
# Phase 2: parameter recovery on agent data with known bounds.
source "${REPO_DIR:-$HOME/cr-dreamer/cr-dreamerv3}/scripts/bound_inference/common.sh"

python dreamerv3/main.py "${AGENT_FLAGS[@]}" \
  --script bound_recovery \
  --logdir "${RUN_DIR}/recovery" \
  --run.from_checkpoint "$(latest_checkpoint)" \
  --jax.policy_devices 0 --jax.train_devices 0 \
  --inference.recovery_thetas 16 \
  --inference.recovery_steps 3000 \
  --inference.candidates 256 \
  --inference.rounds 3
