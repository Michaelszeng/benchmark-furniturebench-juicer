#!/bin/bash
# Submit N parallel instances of submit_dagger_collect_corrections.sbatch.
#
# Each instance handles candidates[i::N] of the labeled failures (modulo
# stride). The last instance to finish runs process_pickles via the lock
# pattern.
#
# Usage:
#   bash scripts/launch_dagger_collect_corrections.sh <num_procs> <iter> <non_markovian>

set -e
N="${1:?Usage: $0 <num_procs> <iter> <non_markovian>}"
ITER="${2:?missing iter}"
NON_MARKOVIAN="${3:?missing non_markovian (true/false)}"

# Must match FURNITURE in submit_dagger_collect_corrections.sbatch.
FURNITURE="one_leg"
CORR_DIR="dataset/raw/sim/${FURNITURE}/dagger_iter${ITER}/low/correction"

# Clear any stale barrier/lock state from a previous (possibly-crashed) run.
rm -rf "${CORR_DIR}/.corrections_done" "${CORR_DIR}/.process_lock"

echo "Submitting $N parallel correction-collection jobs (iter=$ITER, nm=$NON_MARKOVIAN)"
for i in $(seq 0 $((N - 1))); do
    sbatch scripts/submit_dagger_collect_corrections.sbatch \
        "$ITER" "$NON_MARKOVIAN" "$i" "$N"
done
