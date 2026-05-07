FURNITURE="one_leg"
N_DEMOS=400

# $1: dart_amount  $2: suffix  $3: chunk_size (scripted_dart only, default 16)
DART_AMOUNT=${1:-0.0}
SUFFIX=${2:-""}

# The chunk size should be equal to (or greater than) your diffusion policy's prediction horizon; it consists of the clean
# future actions that the diffusion policy should learn to predict 
# Default value of 16 corresponds to a policy that predicts up to 16 actions into the future
CHUNK_SIZE=${3:-16}

# With N_ENVS=8 on H200 node, collecting more than 1 ep/min
N_ENVS=8

echo "FURNITURE: ${FURNITURE}"
echo "N_DEMOS: ${N_DEMOS}"
echo "DART_AMOUNT: ${DART_AMOUNT}"
echo "SUFFIX: ${SUFFIX}"
echo "CHUNK_SIZE: ${CHUNK_SIZE}"
echo "N_ENVS: ${N_ENVS}"

# `stdbuf -oL -eL` line-buffers stdout/stderr of the child process and `python -u`
# disables Python's own output buffering. Together, these ensure that any
# log line emitted just before a native segfault actually reaches the
# .out / .err files instead of being lost in a flushed-at-exit buffer.
PY="stdbuf -oL -eL python -u"

# Route to scripted_dart.py when DART_AMOUNT > 0, scripted.py otherwise.
USE_DART=$(echo "${DART_AMOUNT} > 0" | bc -l)
DEMO_SOURCE="scripted$( [ -n "${SUFFIX}" ] && echo "_${SUFFIX}" )"

if [ "${USE_DART}" = "1" ]; then
    ${PY} -m src.data_collection.scripted_dart \
        -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} \
        --chunk-size ${CHUNK_SIZE} --dart-amount ${DART_AMOUNT} \
        $( [ -n "${SUFFIX}" ] && echo "--output-dir-suffix ${SUFFIX}" ) \
        --headless
else
    ${PY} src/data_collection/scripted.py \
        -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} \
        --n-video-trials 20 --dart-amount ${DART_AMOUNT} \
        $( [ -n "${SUFFIX}" ] && echo "--output-dir-suffix ${SUFFIX}" ) \
        --headless
fi

# Process the pickles and convert to zarr
PROCESS_PICKLES=$( [ "${USE_DART}" = "1" ] \
    && echo "src/data_processing/process_pickles_dart.py" \
    || echo "src/data_processing/process_pickles.py" )
${PY} ${PROCESS_PICKLES} -f ${FURNITURE} -s "${DEMO_SOURCE}" -e "sim" --overwrite

# Translate Zarr format for diffusion policy
${PY} src/data_processing/process_zarr.py \
    dataset/processed/sim/${FURNITURE}/${DEMO_SOURCE}/low/success.zarr \
    --output dataset/processed/sim/${FURNITURE}/${DEMO_SOURCE}/low/success_translated.zarr \
    --overwrite