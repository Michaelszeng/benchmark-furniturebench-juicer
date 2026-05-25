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
N="${1:?Usage: $0 <num_procs> <iter> <action_horizon> <non_markovian>}"
ITER="${2:?missing iter}"
ACTION_HORIZON="${3:?missing action_horizon}"
NON_MARKOVIAN="${4:?missing non_markovian (true/false)}"

# Must match FURNITURE in submit_dagger_collect_failures.sbatch.
FURNITURE="one_leg"
FAILURE_DIR="dataset/raw/sim/${FURNITURE}/dagger_iter${ITER}/low/failure"

# Clear any stale barrier/lock state from a previous (possibly-crashed) run
# so the new submission isn't immediately tripped by a leftover marker.
rm -rf "${FAILURE_DIR}/.collect_done" "${FAILURE_DIR}/.render_lock"

echo "Submitting $N parallel failure-collection jobs (iter=$ITER, ah=$ACTION_HORIZON, nm=$NON_MARKOVIAN)"
for i in $(seq 0 $((N - 1))); do
    sbatch scripts/submit_dagger_collect_failures.sbatch \
        "$ITER" "$ACTION_HORIZON" "$NON_MARKOVIAN" "$i" "$N"
done
