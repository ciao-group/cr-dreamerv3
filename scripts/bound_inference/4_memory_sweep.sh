#!/bin/bash
# Phase 4: train one amortised agent (3 seeds) per memory model.
# Afterwards run 3_infer.sh and 5_eval_human_likeness.sh with the same
# --mem-noise / --mem-decay for every model and compare held-out results.
cd "$(dirname "$0")"
for MODEL in "0.05 0.0" "0.1 0.0" "0.2 0.0" "0.0 0.01" "0.0 0.05"; do
  read -r NOISE DECAY <<< "${MODEL}"
  sbatch 1_train_amortized.sh --mem-noise "${NOISE}" --mem-decay "${DECAY}"
done
