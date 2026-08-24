#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=512000
#SBATCH --gpus=a30:8
#SBATCH --time=48:00:00
#SBATCH -e Offline_logs/jobfiles/log/%x.error-%j
#SBATCH --partition=paula

# Define the game variable to be used throughout the script
GAME="seaquest"

# Extract the script filename without the extension to use as the log folder name
SCRIPT_NAME=$(basename "$0")
SCRIPT_NAME_NO_EXT="${SCRIPT_NAME%.*}"

WORKSPACE_DIR=$(ws_find DreamerSpace)

# Define the common log directory dynamically for both training and evaluation
LOG_DIR="${WORKSPACE_DIR}/${GAME}/logdir/cr-dreamer/${GAME}/${SCRIPT_NAME_NO_EXT}"

cd ~/cr-dreamer/cr-dreamerv3
source dreamerv3/bin/activate

# Training command
# python dreamerv3/main.py --logdir ${LOG_DIR} --run.train_ratio 32 --configs cr-atari --task cr-atari_${GAME} --env.cr-atari.vision_square_size 24,24 --jax.policy_devices 0,1,2,3,4,5,6,7 --jax.train_devices 0,1,2,3,4,5,6,7 --env.cr-atari.pooling 2 --env.cr-atari.vision_model EMMA --env.cr-atari.vision_mode foveated --env.cr-atari.repeat 4 --env.cr-atari.motor_action_delay True --env.cr-atari.reaction_time_delay True --env.cr-atari.eye_movement_effort True

# Dynamically find the most recently modified checkpoint directory
LATEST_CKPT=$(ls -td ${LOG_DIR}/ckpt/*/ | head -n 1)

# Evaluation command
python dreamerv3/main.py --logdir ${LOG_DIR} --run.train_ratio 32 --configs cr-atari --task cr-atari_${GAME} --env.cr-atari.vision_square_size 24,24 --jax.policy_devices 0 --jax.train_devices 0 --env.cr-atari.pooling 2 --script eval_only --run.from_checkpoint ${LATEST_CKPT} --env.cr-atari.vision_model EMMA --env.cr-atari.vision_mode periphery-cutoff --run.steps 5e5
# python dreamerv3/main.py  --logdir ${WORKSPACE_DIR}/${GAME}/logdir/cr-dreamer/${GAME}/emma_eval_gaze_model --run.train_ratio 32 --configs cr-atari --task cr-atari_${GAME} --env.cr-atari.vision_square_size 24,24 --env.cr-atari.pooling 4 --script eval_gaze --run.from_checkpoint /work2/wy39otun-CRDreamer/${GAME}/logdir/cr-dreamer/${GAME}/emma/ckpt/20260610T181156F882891/ --env.cr-atari.vision_model EMMA --env.cr-atari.vision_mode foveated --evaluation.apply_vision_square_from model