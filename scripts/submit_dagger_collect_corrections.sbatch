#!/bin/bash
#SBATCH --job-name=dagger_corrections
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx3090
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1
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
#   sbatch scripts/submit_dagger_collect_corrections.sbatch <iter> <non_markovian>
#
# Multi-instance parallel submission:
#   bash scripts/launch_dagger_collect_corrections.sh <num_procs> <iter> <non_markovian>
#
# Each instance handles candidates[instance_idx::total_instances] of the labeled
# failures. The LAST instance to reach the barrier runs process_pickles once
# (see launch_dagger_collect_corrections.sh and the lock pattern below).

# === CONSTANTS (edit before running) ====================================
FURNITURE="one_leg"
# ========================================================================

ITER="${1:?Usage: sbatch ... <iter> <non_markovian> [instance_idx total_instances]}"
NON_MARKOVIAN="${2:?missing non_markovian (true/false)}"
INSTANCE_IDX="${3:-0}"
TOTAL_INSTANCES="${4:-1}"
OUTPUT_NAME="dagger_iter${ITER}"

echo "$(date +'%F %T')  job=$SLURM_JOB_ID  host=$HOSTNAME"
echo "ITER=$ITER  NON_MARKOVIAN=$NON_MARKOVIAN"
echo "INSTANCE_IDX=$INSTANCE_IDX  TOTAL_INSTANCES=$TOTAL_INSTANCES"
echo "FURNITURE=$FURNITURE  OUTPUT_NAME=$OUTPUT_NAME"

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

FAILURE_DIR="dataset/raw/sim/${FURNITURE}/dagger_iter${ITER}/low/failure"
if [ ! -d "$FAILURE_DIR" ]; then
    echo "ERROR: $FAILURE_DIR does not exist — run submit_dagger_collect_failures.sbatch first."
    exit 1
fi

echo "==== [1/2] dagger_collect_corrections.py (slice ${INSTANCE_IDX}/${TOTAL_INSTANCES}) ===="
python scripts/dagger_collect_corrections.py \
    --furniture "$FURNITURE" \
    --iter "$ITER" \
    --instance-idx "$INSTANCE_IDX" \
    --total-instances "$TOTAL_INSTANCES" \
    --headless \
    "${NM_ARG[@]}"

# --- Barrier + lock: only the last instance runs process_pickles. -------
CORR_DIR="dataset/raw/sim/${FURNITURE}/dagger_iter${ITER}/low/correction"
mkdir -p "$CORR_DIR"
DONE_DIR="${CORR_DIR}/.corrections_done"
LOCK_DIR="${CORR_DIR}/.process_lock"
mkdir -p "$DONE_DIR"
touch "${DONE_DIR}/${SLURM_JOB_ID:-$$}"
N_DONE=$(ls "$DONE_DIR" 2>/dev/null | wc -l)
echo "barrier: ${N_DONE}/${TOTAL_INSTANCES} instances finished"

if [ "$N_DONE" -ge "$TOTAL_INSTANCES" ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "==== [2/2] process_pickles.py (last instance, lock acquired) ===="
    python -m src.data_processing.process_pickles \
        --env sim --furniture "$FURNITURE" --randomness low --demo-outcome success \
        --source "dagger_iter${ITER}/correction" --output-name "$OUTPUT_NAME" --overwrite
    rm -rf "$DONE_DIR" "$LOCK_DIR"
    echo "done. Zarr: dataset/processed/sim/${FURNITURE}/${OUTPUT_NAME}/low/success.zarr"
else
    echo "skipping process_pickles (not the last instance, or lost lock race)"
fi
echo "$(date +'%F %T')  finished"
