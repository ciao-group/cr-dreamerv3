#!/bin/bash
# End-to-end check of the pipeline on a laptop CPU with a tiny model:
# train briefly, infer bounds from two human trials, run recovery and a
# closed-loop evaluation. Results are meaningless, only the plumbing is tested.
# Expects Atari-HEAD asterix in data/Atari-HEAD/asterix_raw/asterix.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Usage: bash smoke_local.sh [PYTHON] [OUTDIR] [ATARI_HEAD_GAME_DIR]
PY=${1:-python}
OUT=${2:-data/smoke}
DATA=${3:-data/Atari-HEAD/asterix_raw/asterix}
FLAGS=(
  --configs cr-atari cr-atari-amortized debug --task cr-atari_asterix
  --jax.debug False --env.cr-atari.vision_square_size 24,24
  --agent.dyn.rssm.mem_noise 0.1 --agent.dyn.rssm.mem_decay 0.01
  --inference.data "${DATA}"
)
"${PY}" dreamerv3/main.py "${FLAGS[@]}" --logdir "${OUT}/train" \
  --run.steps 1000 --run.envs 2 || true  # the JAX profiler may segfault on macOS after saving
CKPT="${OUT}/train/ckpt/$(cat "${OUT}/train/ckpt/latest")"
"${PY}" dreamerv3/main.py "${FLAGS[@]}" --script infer_bounds \
  --logdir "${OUT}/infer" --run.from_checkpoint "${CKPT}" \
  --inference.trials 160,167 --inference.max_steps 300 \
  --inference.candidates 16 --inference.rounds 2
"${PY}" dreamerv3/main.py "${FLAGS[@]}" --script bound_recovery \
  --logdir "${OUT}/recovery" --run.from_checkpoint "${CKPT}" \
  --inference.recovery_thetas 3 --inference.recovery_steps 100 \
  --inference.candidates 8 --inference.rounds 1
"${PY}" dreamerv3/main.py "${FLAGS[@]}" --script eval_human_likeness \
  --logdir "${OUT}/eval" --run.from_checkpoint "${CKPT}" \
  --inference.theta_file "${OUT}/infer/inference.json" \
  --inference.trials 160,167 --inference.eval_steps 1000
echo "Smoke test finished: ${OUT}"
