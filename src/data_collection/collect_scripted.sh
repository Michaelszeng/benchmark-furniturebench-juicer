FURNITURE="one_leg"  # only one_leg is currently supported

# $1: dart_amount
# $2: suffix
# $3: non_markovian (True/False)
# $4: n_demos (default 100)
DART_AMOUNT=${1:-0.0}
SUFFIX=${2:-""}
NON_MARKOVIAN=${3:-"False"}
N_DEMOS=${4:-200}

N_ENVS=4

# SUFFIX=""  # manual override

echo "FURNITURE: ${FURNITURE}"
echo "N_DEMOS: ${N_DEMOS}"
echo "DART_AMOUNT: ${DART_AMOUNT}"
echo "SUFFIX: ${SUFFIX}"
echo "N_ENVS: ${N_ENVS}"
echo "NON_MARKOVIAN: ${NON_MARKOVIAN}"

# `stdbuf -oL -eL` line-buffers stdout/stderr of the child process and `python -u`
# disables Python's own output buffering. Together, these ensure that any
# log line emitted just before a native segfault actually reaches the
# .out / .err files instead of being lost in a flushed-at-exit buffer.
PY="stdbuf -oL -eL python -u"

NON_MARKOVIAN_FLAG=""
if [ "${NON_MARKOVIAN}" = "True" ]; then
    NON_MARKOVIAN_FLAG="--non-markovian"
    if [ -n "$SUFFIX" ]; then
        SUFFIX="${SUFFIX}_non_markovian"
    else
        SUFFIX="non_markovian"
    fi
fi

DEMO_SOURCE="scripted$( [ -n "${SUFFIX}" ] && echo "_${SUFFIX}" )"

${PY} src/data_collection/scripted.py \
    -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} \
    --n-video-trials 20 --dart-amount ${DART_AMOUNT} ${NON_MARKOVIAN_FLAG} \
    $( [ -n "${SUFFIX}" ] && echo "--output-dir-suffix ${SUFFIX}" ) \
    --headless

# Post-processing: convert pickles → zarr → translated zarr.
# Multiple instances of this script may run in parallel (separate SLURM jobs),
# all writing to the same output directory.  Only one instance should run
# post-processing.  We use an atomic mkdir lock + a done-marker file so that
# exactly the first instance to finish collection runs it; all others skip.
#
# If post-processing crashes and leaves a stale lock, delete the lock dir:
#   rm -rf "${LOCK_DIR}"
PROCESS_PICKLES="src/data_processing/process_pickles.py"

RAW_SUCCESS_DIR="dataset/raw/sim/${FURNITURE}/${DEMO_SOURCE}/low/success"
DONE_FILE="${RAW_SUCCESS_DIR}/.post_process_done"
LOCK_DIR="${RAW_SUCCESS_DIR}/.post_process_lock"

if [ -f "${DONE_FILE}" ]; then
    echo "[post-process] Already completed by another instance — skipping."
elif mkdir "${LOCK_DIR}" 2>/dev/null; then
    # Acquired the lock.  Double-check the done marker in case another instance
    # completed between our file-existence check and the mkdir.
    if [ ! -f "${DONE_FILE}" ]; then
        echo "[post-process] Lock acquired — running post-processing."
        ${PY} ${PROCESS_PICKLES} -f ${FURNITURE} -s "${DEMO_SOURCE}" -e "sim" --overwrite \
            && ${PY} src/data_processing/process_zarr.py \
                dataset/processed/sim/${FURNITURE}/${DEMO_SOURCE}/low/success.zarr \
                --output dataset/processed/sim/${FURNITURE}/${DEMO_SOURCE}/low/success_translated.zarr \
                --overwrite \
            && touch "${DONE_FILE}"
    else
        echo "[post-process] Already completed by another instance (inside lock) — skipping."
    fi
    rmdir "${LOCK_DIR}"
else
    echo "[post-process] Another instance holds the lock — skipping."
fi