#!/bin/bash
# Submit collect_scripted.sbatch for multiple dart_amount values.

NON_MARKOVIAN="True"

# Hard-coded flag that selects scripted_dart.py + process_pickles_dart.py when "True",
# scripted.py + process_pickles.py when "False".  Independent of DART_AMOUNT.
USE_DART="True"

declare -A DART_SUFFIXES=(
    # ["0.0"]="0"
    # ["0.0625"]="0_0625"
    # ["0.125"]="0_125"
    # ["0.25"]="0_25"
    ["0.5"]="0_5"
    # ["1.0"]="1_0"
)

for DART_AMOUNT in "${!DART_SUFFIXES[@]}"; do
    SUFFIX="${DART_SUFFIXES[$DART_AMOUNT]}"
    echo "Submitting dart_amount=${DART_AMOUNT} (suffix=${SUFFIX}) non_markovian=${NON_MARKOVIAN} use_dart=${USE_DART}"
    sbatch src/data_collection/submit_collect_scripted.sbatch \
        "${DART_AMOUNT}" "${SUFFIX}" "" "${NON_MARKOVIAN}" "${USE_DART}"
done

# Usage:
#   ./src/data_collection/submit_dart_sweep.sh