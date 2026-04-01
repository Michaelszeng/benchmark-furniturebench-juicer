#!/usr/bin/env bash
# Runs 500 rollouts for each action horizon in [1,2,3,4,6,7,8,10,12,15],
# for every checkpoint found under the given path (file or directory).
# Results are written to: outputs/<DATE>/<TIME>/T_a_<x>/<checkpoint_stem>/
#
# Example usage (single checkpoint):
#   ./src/eval/action_horizon_ablation.sh /home/michzeng/diffusion-policy/data/outputs/furniture_bench/2_obs_one_leg_scripted/checkpoints/epoch=095-val_loss=0.0659-val_ddim_mse=0.015579.ckpt one_leg 64 0 500
#
# Example usage (directory of checkpoints):
#   ./src/eval/action_horizon_ablation.sh /home/michzeng/diffusion-policy/data/outputs/furniture_bench/2_obs_one_leg_scripted/checkpoints/ one_leg 64 0 500
set -euo pipefail

CHECKPOINT_PATH="${1:?Usage: $0 <checkpoint_or_dir> [furniture] [n_envs] [n_video_trials] [n_rollouts]}"
FURNITURE="${2:-one_leg}"
N_ENVS="${3:-1}"
N_VIDEO_TRIALS="${4:-0}"
N_ROLLOUTS="${5:-500}"

# Collect checkpoints: single file or all .ckpt files in a directory.
if [[ -f "${CHECKPOINT_PATH}" ]]; then
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

# Single base output dir shared across all checkpoints and horizons.
BASE_OUT="outputs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)"
echo "Base output directory: ${BASE_OUT}"
echo "Checkpoints found: ${#CHECKPOINTS[@]}"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CKPT_STEM="$(basename "${CHECKPOINT}" .ckpt)"
    echo ""
    echo "########################################## "
    echo "Checkpoint: ${CKPT_STEM}"
    echo "##########################################"

    for N_ACTION_STEPS in 1 2 3 4 5 6 8 10 12 15; do
        OUT_DIR="${BASE_OUT}/T_a_${N_ACTION_STEPS}/${CKPT_STEM}"
        echo "=========================================="
        echo "Action horizon: ${N_ACTION_STEPS}  ->  ${OUT_DIR}"
        echo "=========================================="
        if ! python src/eval/evaluate_model_custom.py \
            --checkpoint "${CHECKPOINT}" \
            --furniture "${FURNITURE}" \
            --n-rollouts "${N_ROLLOUTS}" \
            --n-envs "${N_ENVS}" \
            --n-action-steps "${N_ACTION_STEPS}" \
            --n-video-trials "${N_VIDEO_TRIALS}" \
            --output-dir "${OUT_DIR}" \
            --headless; then
            echo "ERROR: evaluate_model_custom.py failed for action_horizon=${N_ACTION_STEPS}, checkpoint=${CKPT_STEM}" >&2
            continue
        fi
    done
done
