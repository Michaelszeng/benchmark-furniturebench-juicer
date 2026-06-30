#!/usr/bin/env bash
for T_a in 3; do
    sbatch src/eval/submit_evaluate_checkpoints.sbatch "${T_a}"
done
