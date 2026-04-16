#!/bin/bash
#SBATCH --job-name=evaluate_checkpoints
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-a6000
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
#   sbatch src/eval/submit_evaluate_checkpoints.sbatch <action_horizon>
#
# SLRUM runner for `evaluate_checkpoints.sh`.

ACTION_HORIZON="${1:?Usage: sbatch src/eval/submit_evaluate_checkpoints.sbatch <action_horizon>}"

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "ACTION_HORIZON: $ACTION_HORIZON"

# ============================================================================
# Evaluation parameters
# ============================================================================
# path to a .ckpt file or a directory of .ckpt files
CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted_r3m/checkpoints"
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop/checkpoints"

FURNITURE="one_leg"
N_ENVS=32
N_ROLLOUTS=500
N_VIDEO_TRIALS=20
# ============================================================================

# Extract the experiment name as the directory immediately before "checkpoints".
EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }
# Conda setup
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate py38

mkdir -p logs


export PYTHONPATH=$PWD:$PYTHONPATH
export PHYSX_MAX_GPU_CONTACT_PAIRS=$((1024 * 1024 * 32))

export HOME=/data/locomotion/michzeng

# IsaacGym's native extension links against libpython3.8.so.1.0 at the C level.
# The conda activate step does not add the env's lib/ to LD_LIBRARY_PATH, so the
# dynamic linker can't find the library unless we set it explicitly.
export LD_LIBRARY_PATH=/data/locomotion/michzeng/mamba/envs/py38/lib:${LD_LIBRARY_PATH:-}

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "FURNITURE:       $FURNITURE"
echo "ACTION_HORIZON:  $ACTION_HORIZON"
echo "N_ENVS:          $N_ENVS"
echo "N_ROLLOUTS:      $N_ROLLOUTS"
echo "=========================================="

bash src/eval/evaluate_checkpoints.sh "${CHECKPOINT_PATH}" "${FURNITURE}" "${ACTION_HORIZON}" "${N_ENVS}" "${N_VIDEO_TRIALS}" "${N_ROLLOUTS}" --resume
