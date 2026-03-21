#!/bin/bash
#SBATCH --job-name=action_horizon_ablation
#SBATCH --account=locomotion
#SBATCH --partition=locomotion-h200
#SBATCH --qos=locomotion-main
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=384G
#SBATCH --time=48:00:00
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
CHECKPOINT_PATH=""   # path to a .ckpt file or a directory of .ckpt files
FURNITURE="one_leg"
N_ENVS=1
N_ROLLOUTS=500
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

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "FURNITURE:       $FURNITURE"
echo "N_ENVS:          $N_ENVS"
echo "N_ROLLOUTS:      $N_ROLLOUTS"
echo "=========================================="

# --- Collect checkpoints: single file or all .ckpt files in a directory ---
if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "Error: CHECKPOINT_PATH is not set." >&2
    exit 1
elif [[ -f "${CHECKPOINT_PATH}" ]]; then
    CHECKPOINTS=("${CHECKPOINT_PATH}")
elif [[ -d "${CHECKPOINT_PATH}" ]]; then
    mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_PATH}" -maxdepth 1 -name "*.ckpt" ! -name "latest.ckpt" | sort)
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "No .ckpt files found in ${CHECKPOINT_PATH}" >&2
        exit 1
    fi
else
    echo "Error: ${CHECKPOINT_PATH} is neither a file nor a directory." >&2
    exit 1
fi

echo "Checkpoints found: ${#CHECKPOINTS[@]}"

# Single base output dir shared across all checkpoints and horizons.
BASE_OUT="outputs/${DATE}/${TIME}"
echo "Base output directory: ${BASE_OUT}"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CKPT_STEM="$(basename "${CHECKPOINT}" .ckpt)"
    echo ""
    echo "########################################## "
    echo "Checkpoint: ${CKPT_STEM}"
    echo "##########################################"

    for N_ACTION_STEPS in 1 2 3 4 6 7 8 10 12 15; do
        OUT_DIR="${BASE_OUT}/T_a_${N_ACTION_STEPS}/${CKPT_STEM}"
        echo "=========================================="
        echo "Action horizon: ${N_ACTION_STEPS}  ->  ${OUT_DIR}"
        echo "=========================================="
        python src/eval/evaluate_model_custom.py \
            --checkpoint "${CHECKPOINT}" \
            --furniture "${FURNITURE}" \
            --n-rollouts "${N_ROLLOUTS}" \
            --n-envs "${N_ENVS}" \
            --n-action-steps "${N_ACTION_STEPS}" \
            --output-dir "${OUT_DIR}" \
            --headless
    done
done
