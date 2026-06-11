#!/bin/bash
#SBATCH --job-name=dagger_corrections
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-titanrtx
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch scripts/submit_dagger_collect_corrections.sbatch <iter> <action_horizon> <non_markovian>
#
# Multi-instance parallel submission:
#   bash scripts/launch_dagger_collect_corrections.sh <num_procs> <iter> <action_horizon> <non_markovian>
#
# Each instance handles candidates[instance_idx::total_instances] of the labeled
# failures. The LAST instance to reach the barrier runs process_pickles once
# (see launch_dagger_collect_corrections.sh and the lock pattern below).

# === CONSTANTS (edit before running) ====================================
FURNITURE="one_leg"
# DEBUG flags (true/false). Both default false for normal training-data collection.
DEBUG_FULL_ROLLOUT="false"     # save full policy rollout (option B) → correction_debug/
SAVE_FAILED_ATTEMPTS="true"    # also save every failed expert attempt → correction[_debug]/failure/
# ========================================================================

ITER="${1:?Usage: sbatch ... <iter> <action_horizon> <non_markovian> [instance_idx total_instances]}"
ACTION_HORIZON="${2:?missing action_horizon}"
NON_MARKOVIAN="${3:?missing non_markovian (true/false)}"
INSTANCE_IDX="${4:-0}"
TOTAL_INSTANCES="${5:-1}"
DART_AMOUNT="${DART_AMOUNT:-0.0}"

NM_TAG=$([[ "${NON_MARKOVIAN,,}" == "true" || "${NON_MARKOVIAN}" == "1" || "${NON_MARKOVIAN,,}" == "yes" ]] && echo "nm" || echo "m")
DEMO_SOURCE="dagger_iter${ITER}_ah${ACTION_HORIZON}_${NM_TAG}"
OUTPUT_NAME="$DEMO_SOURCE"

echo "$(date +'%F %T')  job=$SLURM_JOB_ID  host=$HOSTNAME"
echo "ITER=$ITER  ACTION_HORIZON=$ACTION_HORIZON  NON_MARKOVIAN=$NON_MARKOVIAN"
echo "INSTANCE_IDX=$INSTANCE_IDX  TOTAL_INSTANCES=$TOTAL_INSTANCES"
echo "FURNITURE=$FURNITURE  DEMO_SOURCE=$DEMO_SOURCE  DART_AMOUNT=$DART_AMOUNT"
echo "DEBUG_FULL_ROLLOUT=$DEBUG_FULL_ROLLOUT  SAVE_FAILED_ATTEMPTS=$SAVE_FAILED_ATTEMPTS"

# --- Environment setup ---
cd "$SLURM_SUBMIT_DIR" || { echo "cd failed"; exit 1; }
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate py38
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
mkdir -p logs
export PYTHONPATH=$PWD:$PYTHONPATH
export PHYSX_MAX_GPU_CONTACT_PAIRS=$((1024 * 1024 * 32))
export HOME=/data/locomotion/michzeng
export LD_LIBRARY_PATH=/data/locomotion/michzeng/mamba/envs/py38/lib:${LD_LIBRARY_PATH:-}
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_INSTANCE_LAYERS=
ulimit -c unlimited
export PYTHONFAULTHANDLER=1

NM_ARG=()
case "${NON_MARKOVIAN,,}" in true|1|yes) NM_ARG=(--non-markovian) ;; esac
DEBUG_ARG=()
case "${DEBUG_FULL_ROLLOUT,,}" in true|1|yes) DEBUG_ARG=(--debug-full-rollout) ;; esac
FAILED_ARG=()
case "${SAVE_FAILED_ATTEMPTS,,}" in true|1|yes) FAILED_ARG=(--save-failed-attempts) ;; esac

FAILURE_DIR="dataset/raw/sim/${FURNITURE}/${DEMO_SOURCE}/low/failure"
if [ ! -d "$FAILURE_DIR" ]; then
    echo "ERROR: $FAILURE_DIR does not exist — run submit_dagger_collect_failures.sbatch first."
    exit 1
fi

echo "==== [1/2] dagger_collect_corrections.py (slice ${INSTANCE_IDX}/${TOTAL_INSTANCES}) ===="
python scripts/dagger_collect_corrections.py \
    --furniture "$FURNITURE" \
    --iter "$ITER" \
    --action-horizon "$ACTION_HORIZON" \
    --dart-amount "$DART_AMOUNT" \
    --instance-idx "$INSTANCE_IDX" \
    --total-instances "$TOTAL_INSTANCES" \
    --headless \
    "${NM_ARG[@]}" \
    "${DEBUG_ARG[@]}" \
    "${FAILED_ARG[@]}"

# --- Barrier + lock: only the last instance runs process_pickles. -------
CORR_DIR="dataset/raw/sim/${FURNITURE}/${DEMO_SOURCE}/low/correction"
mkdir -p "$CORR_DIR"
DONE_DIR="${CORR_DIR}/.corrections_done"
LOCK_DIR="${CORR_DIR}/.process_lock"
mkdir -p "$DONE_DIR"
touch "${DONE_DIR}/${SLURM_JOB_ID:-$$}"
N_DONE=$(ls "$DONE_DIR" 2>/dev/null | wc -l)
echo "Job coordination: ${N_DONE} out of ${TOTAL_INSTANCES} parallel instances have completed their assigned slice of failures."

ZARR_PATH="dataset/processed/sim/${FURNITURE}/${OUTPUT_NAME}/low/correction/success.zarr"
TRANSLATED_ZARR_PATH="dataset/processed/sim/${FURNITURE}/${OUTPUT_NAME}/low/correction/success_translated.zarr"

if [ "$N_DONE" -ge "$TOTAL_INSTANCES" ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "==== [2/3] process_pickles.py (last instance, lock acquired) ===="
    # Corrections are saved under {source}/{randomness}/correction/{outcome}, so
    # `correction` lives in the randomness slot of get_raw_paths' glob, not the source.
    python -m src.data_processing.process_pickles \
        --env sim --furniture "$FURNITURE" --randomness "low/correction" --demo-outcome success \
        --source "${DEMO_SOURCE}" --output-name "$OUTPUT_NAME" --overwrite
    PROCESS_RC=$?
    if [ "$PROCESS_RC" -ne 0 ]; then
        rm -rf "$DONE_DIR" "$LOCK_DIR"
        echo "ERROR: process_pickles failed with exit code ${PROCESS_RC}; no zarr written."
        exit "$PROCESS_RC"
    fi
    echo "done. Zarr: ${ZARR_PATH}"

    echo "==== [3/3] process_zarr.py (translate to diffusion-policy ReplayBuffer format) ===="
    # Training expects the ReplayBuffer layout (data/ + meta/ groups). Translate the
    # process_pickles output into success_translated.zarr, which the training config consumes.
    python src/data_processing/process_zarr.py "$ZARR_PATH" --overwrite
    TRANSLATE_RC=$?
    rm -rf "$DONE_DIR" "$LOCK_DIR"
    if [ "$TRANSLATE_RC" -ne 0 ]; then
        echo "ERROR: process_zarr failed with exit code ${TRANSLATE_RC}; translated zarr not written."
        exit "$TRANSLATE_RC"
    fi
    echo "done. Translated zarr: ${TRANSLATED_ZARR_PATH}"
else
    echo "skipping process_pickles/process_zarr (not the last instance, or lost lock race)"
fi
echo "$(date +'%F %T')  finished"
