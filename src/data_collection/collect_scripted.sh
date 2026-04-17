FURNITURE="one_leg"
N_DEMOS=400
# Override by passing env vars: DART_AMOUNT=0.5 SUFFIX=0_5 sbatch ...
DART_AMOUNT=${DART_AMOUNT:-1.0}
SUFFIX=${SUFFIX:-""}

# With N_ENVS=8 on H200 node, collecting more than 1 ep/min
N_ENVS=8

echo "FURNITURE: ${FURNITURE}"
echo "N_DEMOS: ${N_DEMOS}"
echo "DART_AMOUNT: ${DART_AMOUNT}"
echo "SUFFIX: ${SUFFIX}"
echo "N_ENVS: ${N_ENVS}"

if [ -n "$SUFFIX" ]; then
    # If suffix is provided, use it to create output directory suffix.
    python src/data_collection/scripted.py -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} --n-video-trials 20 --output-dir-suffix ${SUFFIX} --headless --dart-amount ${DART_AMOUNT}
    python src/data_processing/process_pickles.py -f ${FURNITURE} -s "scripted_${SUFFIX}" -e "sim"
    python src/data_processing/process_zarr.py dataset/processed/sim/${FURNITURE}/scripted_${SUFFIX}/low/success.zarr --output dataset/processed/sim/${FURNITURE}/scripted_${SUFFIX}/low/success_translated.zarr
else
    # Else, use default output directory suffix.
    python src/data_collection/scripted.py -f ${FURNITURE} -n ${N_DEMOS} -e ${N_ENVS} --n-video-trials 20 --headless --dart-amount ${DART_AMOUNT}
    python src/data_processing/process_pickles.py -f ${FURNITURE} -s "scripted" -e "sim"
    python src/data_processing/process_zarr.py dataset/processed/sim/${FURNITURE}/scripted/low/success.zarr --output dataset/processed/sim/${FURNITURE}/scripted/low/success_translated.zarr
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