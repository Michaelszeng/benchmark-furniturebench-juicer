#!/bin/bash
# Submit N parallel instances of submit_dagger_collect_failures.sbatch.
#
# Each instance auto-generates a UUID-derived seed (no two will collect the
# same failure). All instances share the on-disk num_failures quota; the last
# one to finish runs dagger_render_failures.py via the lock pattern.
#
# Usage:
#   bash scripts/launch_dagger_collect_failures.sh <num_procs> <iter> <action_horizon> <non_markovian>

set -e

# === HARDCODED CONSTANTS ================================================
# DART action-noise scale; exported into the env so sbatch / python inherit.
DART_AMOUNT=0.0
export DART_AMOUNT
# Must match FURNITURE in submit_dagger_collect_failures.sbatch.
FURNITURE="one_leg"
# ========================================================================

N="${1:?Usage: $0 <num_procs> <iter> <action_horizon> <non_markovian>}"
ITER="${2:?missing iter}"
ACTION_HORIZON="${3:?missing action_horizon}"
NON_MARKOVIAN="${4:?missing non_markovian (true/false)}"

NM_TAG=$([[ "${NON_MARKOVIAN,,}" == "true" || "${NON_MARKOVIAN}" == "1" || "${NON_MARKOVIAN,,}" == "yes" ]] && echo "nm" || echo "m")
DEMO_SOURCE="dagger_iter${ITER}_ah${ACTION_HORIZON}_${NM_TAG}"
FAILURE_DIR="dataset/raw/sim/${FURNITURE}/${DEMO_SOURCE}/low/failure"

# Clear any stale barrier/lock state from a previous (possibly-crashed) run
# so the new submission isn't immediately tripped by a leftover marker.
rm -rf "${FAILURE_DIR}/.collect_done" "${FAILURE_DIR}/.render_lock"

echo "Submitting $N parallel failure-collection jobs"
echo "  iter=$ITER  ah=$ACTION_HORIZON  nm=$NON_MARKOVIAN  dart=$DART_AMOUNT"
echo "  → $FAILURE_DIR"
for i in $(seq 0 $((N - 1))); do
    sbatch --export=ALL,DART_AMOUNT="$DART_AMOUNT" \
        scripts/submit_dagger_collect_failures.sbatch \
        "$ITER" "$ACTION_HORIZON" "$NON_MARKOVIAN" "$i" "$N"
done
