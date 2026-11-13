#!/bin/bash
# Short sweep over the gaze penalties: one seed per setting, a limited number
# of agent steps. Each setting gets its own run directory through the run name
# of train.sh. Run it on the login node; it only calls sbatch.
#
#   bash scripts/vision_grid/sweep.sh --repo-dir ~/cr-dreamerv3-objects
#   bash scripts/vision_grid/sweep.sh --repo-dir ~/cr-dreamerv3-objects --dry-run
#
#   --repo-dir DIR   checkout the jobs run from (required, keep it separate
#                    from the checkout of running jobs)
#   --steps N        agent steps per run (1e7)
#   --seed N         seed for all runs (0)
#   --dry-run        print the train.sh commands instead of submitting
#
# Scale of the settings (5x5 grid, 24x24 squares, current agents): reward is
# about 7 per step, the mean squared eccentricity of the agent's gaze is about
# 160 deg^2 (humans 77), and the agent shifts its gaze nearly every step with
# about 15 deg amplitude. So per step:
#   effort  = kstatic * 160  -> 0.001: 0.16, 0.005: 0.8, 0.02: 3.2
#   saccade = cost * 15      -> 0.05: 0.75

REPO_DIR=
STEPS=1e7
SEED=0
DRY_RUN=

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=--dry-run; shift; continue ;;
    --repo-dir) REPO_DIR=$2 ;;
    --steps) STEPS=$2 ;;
    --seed) SEED=$2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift 2
done
[ -n "${REPO_DIR}" ] || { echo "--repo-dir is required" >&2; exit 1; }

SETTINGS=(
  "--effort-kstatic 0.005"
  "--effort-kstatic 0.02"
  "--saccade-cost 0.05"
  "--effort-kstatic 0.005 --saccade-cost 0.05"
  "--effort-kstatic 0.02 --saccade-cost 0.05"
  "--frame-discount --horizon 1000"
)

for SETTING in "${SETTINGS[@]}"; do
  # shellcheck disable=SC2086
  ARGS=(--repo-dir "${REPO_DIR}" --seed "${SEED}" ${SETTING} ${DRY_RUN} -- --run.steps "${STEPS}")
  if [ -n "${DRY_RUN}" ]; then
    bash "${REPO_DIR}/scripts/vision_grid/train.sh" "${ARGS[@]}" | tail -1
  else
    sbatch --array="${SEED}" "${REPO_DIR}/scripts/vision_grid/train.sh" "${ARGS[@]}"
  fi
done
