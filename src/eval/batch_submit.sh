#!/usr/bin/env bash
for T_a in 10; do
    sbatch src/eval/submit_evaluate_checkpoints.sbatch "${T_a}"
done
