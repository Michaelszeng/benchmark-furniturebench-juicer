#!/bin/bash
# Submit collect_scripted.sbatch for multiple dart_amount values.

declare -A DART_SUFFIXES=(
    ["0.0"]="0"
    # ["0.03125"]="0_03125"
    # ["0.0625"]="0_0625"
    # ["0.125"]="0_125"
    # ["0.25"]="0_25"
    # ["0.5"]="0_5"
    # ["0.75"]="0_75"
)

for DART_AMOUNT in "${!DART_SUFFIXES[@]}"; do
    SUFFIX="${DART_SUFFIXES[$DART_AMOUNT]}"
    echo "Submitting dart_amount=${DART_AMOUNT} (suffix=${SUFFIX})"
    sbatch src/data_collection/submit_collect_scripted.sbatch \
        "${DART_AMOUNT}" "${SUFFIX}"
done

# Usage:
#   ./src/data_collection/submit_dart_sweep.sh