#!/bin/bash
#SBATCH --job-name=cr_vision_grid
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=500G
#SBATCH --gpus=2
#SBATCH --time=3-00:00:00
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --constraint="ARCH:X86&GPU_SKU:A100"
#SBATCH --account=sci-ebel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --array=0-2
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --output=%x_%A_%a.out
# Train CR-Dreamer with a symmetric vision square grid (default: 24x24 px
# square on a 5x5 grid of positions, eye movement effort and EMMA on), one
# seed per array task. No environment variables are read.
#
#   sbatch scripts/vision_grid/train.sh                     # 5x5, seeds 0-2
#   sbatch --array=0 scripts/vision_grid/train.sh           # only seed 0
#   sbatch scripts/vision_grid/train.sh --count 4,4         # 4x4 baseline
#   sbatch scripts/vision_grid/train.sh --no-effort         # without effort
#   bash scripts/vision_grid/train.sh --dry-run             # print command
#   sbatch scripts/vision_grid/train.sh -- --run.steps 5000 # short test
#
#   --game NAME        Atari game (asterix)
#   --count X,Y        number of gaze positions (5,5)
#   --square W,H       vision square size in observation pixels (24,24)
#   --vision-model M   EMMA or None (EMMA)
#   --vision-mode M    foveated, periphery, periphery-cutoff (foveated)
#   --no-effort        disable the eye movement effort penalty
#   --seed N           seed (array task id, else 0)
#   --workspace DIR    logs and checkpoints (project directory)
#   --conda-env NAME   conda environment (cr-dreamer_env)
#   --dry-run          print the command instead of running it
#   -- FLAGS...        further flags passed on to dreamerv3/main.py last
#
# The job requeues itself before the time limit (keeping its arguments) and
# resumes from the last checkpoint; resubmitting the same command does too.

# sbatch runs a copy of this script, so the repo path is fixed.
REPO_DIR=$HOME/cr-dreamerv3

GAME=asterix
COUNT=5,5
SQUARE=24,24
VISION_MODEL=EMMA
VISION_MODE=foveated
EFFORT=True
SEED=${SLURM_ARRAY_TASK_ID:-0}
# Checkpoints include the replay buffer (tens of GB per run), so this must be
# project or scratch storage rather than the home directory.
WORKSPACE_DIR=/sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs
CONDA_ENV=cr-dreamer_env
DRY_RUN=0
EXTRA_FLAGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; EXTRA_FLAGS=("$@"); break ;;
    --no-effort) EFFORT=False; shift; continue ;;
    --dry-run) DRY_RUN=1; shift; continue ;;
    --game) GAME=$2 ;;
    --count) COUNT=$2 ;;
    --square) SQUARE=$2 ;;
    --vision-model) VISION_MODEL=$2 ;;
    --vision-mode) VISION_MODE=$2 ;;
    --seed) SEED=$2 ;;
    --workspace) WORKSPACE_DIR=$2 ;;
    --conda-env) CONDA_ENV=$2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift 2
done

EFFORT_NAME=effort
[ "${EFFORT}" = False ] && EFFORT_NAME=noeffort
MODEL=grid${COUNT/,/x}_square${SQUARE/,/x}_${VISION_MODEL}_${VISION_MODE}_${EFFORT_NAME}
RUN_DIR=${WORKSPACE_DIR}/${GAME}/vision_grid/${MODEL}/seed${SEED}

CMD=(
  python dreamerv3/main.py
  --configs cr-atari
  --task "cr-atari_${GAME}"
  --logdir "${RUN_DIR}"
  --seed "${SEED}"
  --run.train_ratio 32
  --env.cr-atari.vision_square_size "${SQUARE}"
  --env.cr-atari.vision_square_count "${COUNT}"
  --env.cr-atari.vision_model "${VISION_MODEL}"
  --env.cr-atari.vision_mode "${VISION_MODE}"
  --env.cr-atari.eye_movement_effort "${EFFORT}"
  --env.cr-atari.pooling 2
  --env.cr-atari.repeat 1
  --jax.policy_devices 0,1
  --jax.train_devices 0,1
  "${EXTRA_FLAGS[@]}"
)

if [ "${DRY_RUN}" = 1 ]; then
  echo "cd ${REPO_DIR}"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi

cd "${REPO_DIR}" || exit 1
if command -v conda > /dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
echo "Branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "Run dir: ${RUN_DIR}"

# Stop shortly before the time limit (--signal=B:USR1@300) and requeue
# (--requeue); the requeued job resumes from the checkpoint in RUN_DIR.
JOB=${SLURM_JOB_ID:-}
if [ -n "${SLURM_ARRAY_JOB_ID:-}" ]; then
  JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
fi
"${CMD[@]}" &
PID=$!
trap 'echo "Time limit near, requeueing ${JOB}"; kill -TERM ${PID}; wait ${PID}; scontrol requeue "${JOB}"' USR1
wait ${PID}
