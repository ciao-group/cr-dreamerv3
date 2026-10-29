#!/bin/bash
# Shared settings for the bound inference pipeline (see
# docs/bound-inference-plan.md). Sourced by the phase scripts, which pass
# their command line arguments on. No environment variables are read.
#
#   --game NAME        Atari game (asterix)
#   --mem-noise X      RSSM memory noise of the model (0.0)
#   --mem-decay X      RSSM memory decay of the model (0.0)
#   --seed N           seed (array task id for training, else 0)
#   --split S          human trials: train, test or all (phase specific)
#   --subject ID       Atari-HEAD subject, e.g. RZ (all subjects)
#   --theta-file PATH  inferred bounds for evaluation
#   --workspace DIR    data, logs and checkpoints (project directory)
#   --conda-env NAME   conda environment (cr-dreamer_env)

GAME=asterix
MEM_NOISE=0.0
MEM_DECAY=0.0
SEED=${SLURM_ARRAY_TASK_ID:-0}
SPLIT=${DEFAULT_SPLIT:-train}
SUBJECT=
THETA_FILE=
# Checkpoints include the replay buffer (tens of GB per run), so this must be
# project or scratch storage rather than the home directory.
WORKSPACE_DIR=/sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs
CONDA_ENV=cr-dreamer_env

while [ $# -gt 0 ]; do
  case "$1" in
    --game) GAME=$2 ;;
    --mem-noise) MEM_NOISE=$2 ;;
    --mem-decay) MEM_DECAY=$2 ;;
    --seed) SEED=$2 ;;
    --split) SPLIT=$2 ;;
    --subject) SUBJECT=$2 ;;
    --theta-file) THETA_FILE=$2 ;;
    --workspace) WORKSPACE_DIR=$2 ;;
    --conda-env) CONDA_ENV=$2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift 2
done

DATA_DIR=${WORKSPACE_DIR}/Atari-HEAD/${GAME}
MODEL=amortized_mn${MEM_NOISE}_md${MEM_DECAY}
RUN_DIR=${WORKSPACE_DIR}/${GAME}/bounds/${MODEL}/seed${SEED}

cd "${REPO_DIR}"
if command -v conda > /dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
elif [ -f dreamerv3/bin/activate ]; then
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
