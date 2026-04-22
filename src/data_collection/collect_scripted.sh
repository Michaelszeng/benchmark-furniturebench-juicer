FURNITURE="one_leg"
N_DEMOS=400

# $1: dart_amount, $2: suffix
DART_AMOUNT=${1:-1.0}
SUFFIX=${2:-""}

# With N_ENVS=8 on H200 node, collecting more than 1 ep/min
N_ENVS=8

echo "FURNITURE: ${FURNITURE}"
echo "N_DEMOS: ${N_DEMOS}"
echo "DART_AMOUNT: ${DART_AMOUNT}"
echo "SUFFIX: ${SUFFIX}"
echo "N_ENVS: ${N_ENVS}"

# `stdbuf -oL -eL` line-buffers stdout/stderr of the child process and `python -u`
# disables Python's own output buffering. Together, these ensure that any
# log line emitted just before a native segfault actually reaches the
# .out / .err files instead of being lost in a flushed-at-exit buffer.
PY="stdbuf -oL -eL python -u"

if [ -n "$SUFFIX" ]; then
    # If suffix is provided, use it to create output directory suffix.
    ${PY} src/data_collection/scripted.py -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} --n-video-trials 20 --output-dir-suffix ${SUFFIX} --headless --dart-amount ${DART_AMOUNT}
    ${PY} src/data_processing/process_pickles.py -f ${FURNITURE} -s "scripted_${SUFFIX}" -e "sim" --overwrite
    ${PY} src/data_processing/process_zarr.py dataset/processed/sim/${FURNITURE}/scripted_${SUFFIX}/low/success.zarr --output dataset/processed/sim/${FURNITURE}/scripted_${SUFFIX}/low/success_translated.zarr --overwrite
else
    # Else, use default output directory suffix.
    ${PY} src/data_collection/scripted.py -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} --n-video-trials 20 --headless --dart-amount ${DART_AMOUNT}
    ${PY} src/data_processing/process_pickles.py -f ${FURNITURE} -s "scripted" -e "sim" --overwrite
    ${PY} src/data_processing/process_zarr.py dataset/processed/sim/${FURNITURE}/scripted/low/success.zarr --output dataset/processed/sim/${FURNITURE}/scripted/low/success_translated.zarr --overwrite
fi


###
# Run Scripted Policy (to generate `.pkl.xz` files) (`N` is the number of successful demos to record):
# ```bash
# python src/data_collection/scripted.py -f "FURNITURE" -n N --n-video-trials 0 --record-failures
# ```

# Convert `.pkl.xz` to `.zarr`:
# ```bash
# python src/data_processing/process_pickles.py -f "FURNITURE" -s "scripted" -e "sim"
# ```

#### Convert to Training Format

# Convert `zarr` to diffusion policy format (`<source_zarr_name>_translated/zarr`):
# ```bash
# python src/data_processing/process_zarr.py PATH.zarr --output PATH_translated.zarr
# ```
###