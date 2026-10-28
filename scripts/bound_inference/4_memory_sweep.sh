#!/bin/bash
# Phase 4: train one amortised agent (3 seeds) per memory model.
# Afterwards run 3_infer.sh with SPLIT=train and 5_eval_human_likeness.sh with
# the same MEM_NOISE / MEM_DECAY for every model and compare held-out results.
cd "$(dirname "$0")"
for MODEL in "0.05 0.0" "0.1 0.0" "0.2 0.0" "0.0 0.01" "0.0 0.05"; do
  read -r NOISE DECAY <<< "${MODEL}"
  MEM_NOISE=${NOISE} MEM_DECAY=${DECAY} sbatch --export=ALL,MEM_NOISE=${NOISE},MEM_DECAY=${DECAY} 1_train_amortized.sh
done
