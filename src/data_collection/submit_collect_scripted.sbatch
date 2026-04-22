#!/bin/bash
#SBATCH --job-name=collect_scripted
#SBATCH --account=locomotion
#SBATCH --partition=locomotion-h200
#SBATCH --qos=locomotion-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256G
#SBATCH --time=120:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch src/data_collection/submit_collect_scripted.sbatch [dart_amount [suffix]]
#
# Arguments (both optional, positional):
#   $1  dart_amount  — e.g. 0.125
#   $2  suffix       — e.g. 0_125
#
# SLURM runner for `collect_scripted.sh`.

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

# Force the Vulkan loader to only see NVIDIA's ICD on this node. The system
# has 6 ICDs installed (intel_icd, intel_hasvk, radeon, lvp/lavapipe, virtio,
# nvidia), and enumerating all of them at instance creation has correlated
# with sporadic SIGSEGVs deep inside IsaacGym's render_all_camera_sensors
# on Hopper (H200) hardware. Restricting to nvidia_icd.json eliminates that
# whole class of loader-level interactions.
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
# Also clear any default Vulkan instance layers (validation, optimus, etc.)
# in case the cluster has any installed system-wide.
export VK_INSTANCE_LAYERS=

# Enable core dumps and Python's native-signal stack printer for debugging mystery segfault crashes
ulimit -c unlimited
export PYTHONFAULTHANDLER=1
echo "ulimit -c: $(ulimit -c)"
echo "core_pattern: $(cat /proc/sys/kernel/core_pattern 2>/dev/null)"
# NOTE: this cluster's core_pattern pipes to Ubuntu's apport, so any produced
# core typically lands in /var/crash/<exe>.<uid>.crash rather than cwd. The
# faulthandler trace written to the .err log is the reliable artifact here.

echo "=========================================="

bash src/data_collection/collect_scripted.sh "$@"

DATE_END=$(date +"%Y-%m-%d")
TIME_END=$(date +"%H-%M-%S")
echo "Finished DATE: $DATE_END"
echo "Finished TIME: $TIME_END"
