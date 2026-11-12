#!/bin/bash
#SBATCH --job-name=cr_eval_objects
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --partition=pot-hpi-gpu-batch
#SBATCH --constraint=GPU_GEN:AMPERE
#SBATCH --account=sci-ebel
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=slack:martin.lorenz2
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
# Share of time the agent of a vision grid run looks at each object category
# (player, enemy, item, reward), from the latest checkpoint, one seed per
# array task. Compare with scripts/human_object_gaze.py. No environment
# variables are read.
#
#   sbatch scripts/vision_grid/eval_objects.sh                 # 5x5, seeds 0-2
#   sbatch --array=1 scripts/vision_grid/eval_objects.sh       # only seed 1
#   bash scripts/vision_grid/eval_objects.sh --dry-run         # print command
#
#   --model NAME       run name (grid5x5_square24x24_EMMA_foveated_effort)
#   --game NAME        Atari game (asterix)
#   --seed N           seed (array task id, else 0)
#   --episodes N       episodes to play (10)
#   --max-frames N     cut episodes after N frames, 0 = full (18000 = 15 min)
#   --repo-dir DIR     checkout with the object logging (~/cr-dreamerv3)
#   --workspace DIR    logs and checkpoints (project directory)
#   --conda-env NAME   conda environment (cr-dreamer_env)
#   --dry-run          print the command instead of running it
#
# Results: <workspace>/<game>/vision_grid/<model>/eval_objects/seed<N>/
# agent_object_gaze.csv. Only agent.pkl of the checkpoint is copied there,
# the training run is not touched.

# sbatch runs a copy of this script, so the repo path is not taken from $0.
REPO_DIR=$HOME/cr-dreamerv3

GAME=asterix
MODEL=grid5x5_square24x24_EMMA_foveated_effort
SEED=${SLURM_ARRAY_TASK_ID:-0}
EPISODES=10
MAX_FRAMES=18000
WORKSPACE_DIR=/sc/projects/sci-ebel/chair/Martin/CR-Atari_Logs
CONDA_ENV=cr-dreamer_env
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift; continue ;;
    --model) MODEL=$2 ;;
    --game) GAME=$2 ;;
    --seed) SEED=$2 ;;
    --episodes) EPISODES=$2 ;;
    --max-frames) MAX_FRAMES=$2 ;;
    --repo-dir) REPO_DIR=$2 ;;
    --workspace) WORKSPACE_DIR=$2 ;;
    --conda-env) CONDA_ENV=$2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift 2
done

BASE=${WORKSPACE_DIR}/${GAME}/vision_grid/${MODEL}
CMD=(
  python scripts/vision_grid/eval_objects.py
  --run-dir "${BASE}/seed${SEED}"
  --out-dir "${BASE}/eval_objects/seed${SEED}"
  --episodes "${EPISODES}"
  --envs "${EPISODES}"
  --max-frames "${MAX_FRAMES}"
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
exec "${CMD[@]}"
