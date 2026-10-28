#!/bin/bash
# Shared settings for the bound inference pipeline (see
# docs/bound-inference-plan.md). Source this file; override via environment:
#   GAME (asterix), MEM_NOISE (0.0), MEM_DECAY (0.0), SEED (0)

GAME=${GAME:-asterix}
MEM_NOISE=${MEM_NOISE:-0.0}
MEM_DECAY=${MEM_DECAY:-0.0}
SEED=${SEED:-0}

WORKSPACE_DIR=${WORKSPACE_DIR:-$(ws_find DreamerSpace)}
REPO_DIR=${REPO_DIR:-$HOME/cr-dreamer/cr-dreamerv3}
DATA_DIR=${DATA_DIR:-${WORKSPACE_DIR}/Atari-HEAD/${GAME}}

MODEL=amortized_mn${MEM_NOISE}_md${MEM_DECAY}
RUN_DIR=${WORKSPACE_DIR}/${GAME}/bounds/${MODEL}/seed${SEED}

cd "${REPO_DIR}"
if [ -f dreamerv3/bin/activate ]; then
  source dreamerv3/bin/activate
fi

# Flags that must be identical for training and every analysis of a run.
AGENT_FLAGS=(
  --configs cr-atari cr-atari-amortized
  --task "cr-atari_${GAME}"
  --env.cr-atari.vision_square_size 24,24
  --env.cr-atari.pooling 2
  --agent.dyn.rssm.mem_noise "${MEM_NOISE}"
  --agent.dyn.rssm.mem_decay "${MEM_DECAY}"
  --seed "${SEED}"
  --inference.data "${DATA_DIR}"
)

latest_checkpoint() {
  echo "${RUN_DIR}/ckpt/$(cat "${RUN_DIR}/ckpt/latest")"
}
