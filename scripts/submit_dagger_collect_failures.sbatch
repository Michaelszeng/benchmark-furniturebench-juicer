#!/bin/bash
#SBATCH --job-name=dagger_failures
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx3090
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl,improbablex002
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch scripts/submit_dagger_collect_failures.sbatch <iter> <action_horizon> <non_markovian>
#
# Multi-instance parallel submission:
#   bash scripts/launch_dagger_collect_failures.sh <num_procs> <iter> <action_horizon> <non_markovian>
#
# After ALL parallel instances finish collecting, the LAST one to reach the
# barrier acquires a lock and runs dagger_render_failures.py (collect_scripted.sh
# pattern). Once that's done, run dagger_label_gates.py manually, then submit
# submit_dagger_collect_corrections.sbatch for the same iter.

# === CONSTANTS (edit before running) ====================================
# MARKOVIAN ITERATION 1
# Ta = 1,3
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_200_eps/checkpoints/epoch=050-val_loss=0.0076-val_ddim_mse=0.003549.ckpt"

# Ta = 6,10,15
CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_200_eps/checkpoints/epoch=115-val_loss=0.0089-val_ddim_mse=0.004272.ckpt"


# NON-MARKOVIAN ITERATION 1
# Ta = 1,2
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_non_markovian_200_eps/checkpoints/epoch=040-val_loss=0.0354-val_ddim_mse=0.011778.ckpt"
# Ta = 3,4,5,6,8,10,12,15
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m_non_markovian_200_eps/checkpoints/epoch=070-val_loss=0.0682-val_ddim_mse=0.010428.ckpt"

# NON-MARKOVIAN ITERATION 2
# Ta = 1
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah1/checkpoints/epoch=010-val_loss=0.0398-val_ddim_mse=0.022369.ckpt"
# Ta = 3
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah3/checkpoints/epoch=010-val_loss=0.0441-val_ddim_mse=0.016112.ckpt"
# Ta = 6
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah6/checkpoints/epoch=010-val_loss=0.0458-val_ddim_mse=0.014196.ckpt"
# Ta = 10
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah10/checkpoints/epoch=010-val_loss=0.0463-val_ddim_mse=0.011862.ckpt"
# Ta = 15
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah15/checkpoints/epoch=015-val_loss=0.0594-val_ddim_mse=0.006707.ckpt"

# NON-MARKOVIAN ITERATION 3
# Ta = 1
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah1/checkpoints/epoch=005-val_loss=0.0350-val_ddim_mse=0.020247.ckpt"
# Ta = 3
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah3/checkpoints/epoch=010-val_loss=0.0476-val_ddim_mse=0.013081.ckpt"
# Ta = 6
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah6/checkpoints/epoch=010-val_loss=0.0453-val_ddim_mse=0.017701.ckpt"
# Ta = 10
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah10/checkpoints/epoch=015-val_loss=0.0597-val_ddim_mse=0.018448.ckpt"
# Ta = 15
# CHECKPOINT="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/DAgger/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah15/checkpoints/epoch=010-val_loss=0.0470-val_ddim_mse=0.016986.ckpt"

FURNITURE="one_leg"
NUM_FAILURES=50
# ========================================================================

ITER="${1:?Usage: sbatch ... <iter> <action_horizon> <non_markovian> [instance_idx total_instances]}"
ACTION_HORIZON="${2:?missing action_horizon}"
NON_MARKOVIAN="${3:?missing non_markovian (true/false)}"
INSTANCE_IDX="${4:-0}"
TOTAL_INSTANCES="${5:-1}"
# DART_AMOUNT is inherited from the launcher's env (sbatch --export=ALL is default).
# Fallback to 0.0 for direct sbatch invocations.
DART_AMOUNT="${DART_AMOUNT:-0.0}"

echo "$(date +'%F %T')  job=$SLURM_JOB_ID  host=$HOSTNAME"
echo "ITER=$ITER  ACTION_HORIZON=$ACTION_HORIZON  NON_MARKOVIAN=$NON_MARKOVIAN"
echo "INSTANCE_IDX=$INSTANCE_IDX  TOTAL_INSTANCES=$TOTAL_INSTANCES"
echo "CHECKPOINT=$CHECKPOINT"
echo "FURNITURE=$FURNITURE  NUM_FAILURES=$NUM_FAILURES  DART_AMOUNT=$DART_AMOUNT"

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
NM_TAG=$([[ "${NON_MARKOVIAN,,}" == "true" || "${NON_MARKOVIAN}" == "1" || "${NON_MARKOVIAN,,}" == "yes" ]] && echo "nm" || echo "m")

# Demo-source dir name MUST mirror dagger_collect_failures.py's path construction.
FAILURE_DIR="dataset/raw/sim/${FURNITURE}/dagger_iter${ITER}_ah${ACTION_HORIZON}_${NM_TAG}/low/failure"

echo "==== [1/2] dagger_collect_failures.py ===="
python scripts/dagger_collect_failures.py \
    --checkpoint "$CHECKPOINT" \
    --furniture "$FURNITURE" \
    --iter "$ITER" \
    --num-failures "$NUM_FAILURES" \
    --n-action-steps "$ACTION_HORIZON" \
    --dart-amount "$DART_AMOUNT" \
    --headless \
    "${NM_ARG[@]}"

# --- Barrier + lock: only the last instance runs render. -----------------
# Each instance writes a unique marker on completion; the one that observes
# all markers and wins the mkdir lock runs render. Race-free even for single-
# instance runs (TOTAL_INSTANCES=1 → first marker == last → run render).
# Render failures into mp4s for human gating.
DONE_DIR="${FAILURE_DIR}/.collect_done"
LOCK_DIR="${FAILURE_DIR}/.render_lock"
mkdir -p "$DONE_DIR"
touch "${DONE_DIR}/${SLURM_JOB_ID:-$$}"
N_DONE=$(ls "$DONE_DIR" 2>/dev/null | wc -l)
echo "barrier: ${N_DONE}/${TOTAL_INSTANCES} instances finished"

if [ "$N_DONE" -ge "$TOTAL_INSTANCES" ] && mkdir "$LOCK_DIR" 2>/dev/null; then
    echo "==== [2/2] dagger_render_failures.py (last instance, lock acquired) ===="
    python scripts/dagger_render_failures.py "$FAILURE_DIR"
    rm -rf "$DONE_DIR" "$LOCK_DIR"
    echo "done. Next: label gates, then submit submit_dagger_collect_corrections.sbatch."
else
    echo "skipping render (not the last instance, or lost lock race)"
fi
echo "$(date +'%F %T')  finished"
