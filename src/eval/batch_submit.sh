#!/usr/bin/env bash
for T_a in 1 2 3 4 5 6 8 10 12 15; do
    sbatch src/eval/submit_evaluate_checkpoints.sbatch "${T_a}"
done
