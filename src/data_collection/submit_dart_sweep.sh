#!/bin/bash
# Submit collect_scripted.sbatch for multiple dart_amount values.

declare -A DART_SUFFIXES=(
    ["0.25"]="0_25"
    ["0.5"]="0_5"
    ["0.75"]="0_75"
    ["1.5"]="1_5"
)

for DART_AMOUNT in 0.25 0.5 0.75 1.5; do
    SUFFIX="${DART_SUFFIXES[$DART_AMOUNT]}"
    echo "Submitting dart_amount=${DART_AMOUNT} (suffix=${SUFFIX})"
    sbatch --export=ALL,DART_AMOUNT="${DART_AMOUNT}",SUFFIX="${SUFFIX}" \
        src/data_collection/submit_collect_scripted.sbatch
done

# Usage:
#   ./src/data_collection/submit_dart_sweep.sh