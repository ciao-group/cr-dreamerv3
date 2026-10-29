#!/bin/bash
# Shared settings for the bound inference pipeline (see
# docs/bound-inference-plan.md). Source this file; override via environment:
#   GAME (asterix), MEM_NOISE (0.0), MEM_DECAY (0.0), SEED (0)

GAME=${GAME:-asterix}
MEM_NOISE=${MEM_NOISE:-0.0}
MEM_DECAY=${MEM_DECAY:-0.0}
SEED=${SEED:-0}

# Checkpoints include the replay buffer (tens of GB per run), so point
# WORKSPACE_DIR to scratch storage rather than the home directory.
if [ -z "${WORKSPACE_DIR:-}" ]; then
  if command -v ws_find > /dev/null; then
    WORKSPACE_DIR=$(ws_find DreamerSpace)
  else
    echo "Set WORKSPACE_DIR to a directory for data, logs and checkpoints." >&2
    exit 1
  fi
fi
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

# Run a command so that SLURM can stop it shortly before the time limit
# (--signal=B:USR1@300) and requeue the job (--requeue). The requeued job
# resumes from the last checkpoint in the same RUN_DIR.
run_requeueable() {
  local job=${SLURM_JOB_ID:-}
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
    job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
  fi
  "$@" &
  local pid=$!
  trap 'echo "Time limit near, requeueing ${job}"; kill -TERM ${pid}; wait ${pid}; scontrol requeue "${job}"' USR1
  wait ${pid}
}
