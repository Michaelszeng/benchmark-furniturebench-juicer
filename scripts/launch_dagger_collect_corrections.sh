#!/bin/bash
# Submit N parallel instances of submit_dagger_collect_corrections.sbatch.
#
# Each instance handles candidates[i::N] of the labeled failures (modulo
# stride). The last instance to finish runs process_pickles via the lock
# pattern.
#
# Usage:
#   bash scripts/launch_dagger_collect_corrections.sh <num_procs> <iter> <action_horizon> <non_markovian>

set -e

# === HARDCODED CONSTANTS ================================================
# DART action-noise scale; exported into the env so sbatch / python inherit.
# Must match the value used during failure collection for consistent expert behavior.
DART_AMOUNT=0.0
export DART_AMOUNT
# Must match FURNITURE in submit_dagger_collect_corrections.sbatch.
FURNITURE="one_leg"
# ========================================================================

N="${1:?Usage: $0 <num_procs> <iter> <action_horizon> <non_markovian>}"
ITER="${2:?missing iter}"
ACTION_HORIZON="${3:?missing action_horizon}"
NON_MARKOVIAN="${4:?missing non_markovian (true/false)}"

NM_TAG=$([[ "${NON_MARKOVIAN,,}" == "true" || "${NON_MARKOVIAN}" == "1" || "${NON_MARKOVIAN,,}" == "yes" ]] && echo "nm" || echo "m")
DEMO_SOURCE="dagger_iter${ITER}_ah${ACTION_HORIZON}_${NM_TAG}"
CORR_DIR="dataset/raw/sim/${FURNITURE}/${DEMO_SOURCE}/low/correction"

# Clear any stale barrier/lock state from a previous (possibly-crashed) run.
rm -rf "${CORR_DIR}/.corrections_done" "${CORR_DIR}/.process_lock"

echo "Submitting $N parallel correction-collection jobs"
echo "  iter=$ITER  ah=$ACTION_HORIZON  nm=$NON_MARKOVIAN  dart=$DART_AMOUNT"
echo "  → $CORR_DIR"
for i in $(seq 0 $((N - 1))); do
    sbatch --export=ALL,DART_AMOUNT="$DART_AMOUNT" \
        scripts/submit_dagger_collect_corrections.sbatch \
        "$ITER" "$ACTION_HORIZON" "$NON_MARKOVIAN" "$i" "$N"
done
