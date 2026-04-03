#!/bin/bash
#SBATCH --job-name=action_horizon_ablation
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx3090
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
#   sbatch src/eval/submit_action_horizon_ablation_csail.sbatch
#
# Runs 500 rollouts for each action horizon in [1,2,3,4,6,7,8,10,12,15],
# for every checkpoint found under CHECKPOINT_PATH (file or directory).
# Results are written to: outputs/<DATE>/<TIME>/T_a_<x>/<checkpoint_stem>/

# ============================================================================
# Evaluation parameters
# ============================================================================
# path to a .ckpt file or a directory of .ckpt files
# CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_scripted/checkpoints"
CHECKPOINT_PATH="/data/locomotion/michzeng/diffusion-policy-experiments/data/outputs/furniture_bench/2_obs_one_leg_teleop/checkpoints"
FURNITURE="one_leg"
N_ENVS=64
N_ROLLOUTS=500
N_VIDEO_TRIALS=20
# ============================================================================

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
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
echo "N_ENVS:          $N_ENVS"
echo "N_ROLLOUTS:      $N_ROLLOUTS"
echo "=========================================="

bash src/eval/action_horizon_ablation.sh "${CHECKPOINT_PATH}" "${FURNITURE}" "${N_ENVS}" "${N_VIDEO_TRIALS}" "${N_ROLLOUTS}"
