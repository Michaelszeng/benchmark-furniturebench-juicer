FURNITURE="one_leg"
N_DEMOS=500
SUFFIX=""
# SUFFIX="screw_z_pressure"

if [ -n "$SUFFIX" ]; then
    # If suffix is provided, use it to create output directory suffix.
    python src/data_collection/scripted.py -f ${FURNITURE} -n ${N_DEMOS} --record-video "failure" --output-dir-suffix ${SUFFIX}
    python src/data_processing/process_pickles.py -f ${FURNITURE} -s "scripted_${SUFFIX}" -e "sim"
    python src/data_processing/process_zarr.py dataset/processed/sim/${FURNITURE}/scripted_${SUFFIX}/low/success.zarr --output dataset/processed/sim/${FURNITURE}/scripted_${SUFFIX}/low/success_translated.zarr
else
    # Else, use default output directory suffix.
    python src/data_collection/scripted.py -f ${FURNITURE} -n ${N_DEMOS} --record-video "failure"
    python src/data_processing/process_pickles.py -f ${FURNITURE} -s "scripted" -e "sim"
    python src/data_processing/process_zarr.py dataset/processed/sim/${FURNITURE}/scripted/low/success.zarr --output dataset/processed/sim/${FURNITURE}/scripted/low/success_translated.zarr
fi


###
# Run Scripted Policy (to generate `.pkl.xz` files) (`N` is the number of successful demos to record):
# ```bash
# python src/data_collection/scripted.py -f "FURNITURE" -n N --record-video "failure"
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