#!/usr/bin/env bash
# Runs N rollouts for a single given action horizon,
# for every checkpoint found under the given path (file or directory).
# Results are written to: outputs/<experiment_name>/T_a_<x>/<checkpoint_stem>/
# The experiment name is derived from the directory before "checkpoints/".
#
# Example usage (single checkpoint):
#   ./src/eval/evaluate_checkpoints.sh /path/to/checkpoints/epoch=095.ckpt one_leg 8 64 0 500
#
# Example usage (directory of checkpoints):
#   ./src/eval/evaluate_checkpoints.sh /path/to/checkpoints/ one_leg 8 64 0 500
#
# Debug mode (1 rollout, 1 env, no video, not headless):
#   ./src/eval/evaluate_checkpoints.sh <checkpoint_or_dir> one_leg 8 --debug
#
# Resume a previous run (skips completed checkpoints):
#   ./src/eval/evaluate_checkpoints.sh <checkpoint_or_dir> one_leg 8 64 0 500 --resume
set -euo pipefail
ulimit -s unlimited
ulimit -c unlimited

CHECKPOINT_PATH="${1:?Usage: $0 <checkpoint_or_dir> [furniture] <n_action_steps> [n_envs] [n_video_trials] [n_rollouts] [--debug] [--resume]}"
FURNITURE="${2:-one_leg}"
N_ACTION_STEPS="${3:?Usage: $0 <checkpoint_or_dir> [furniture] <n_action_steps> [n_envs] [n_video_trials] [n_rollouts] [--debug] [--resume]}"
N_ENVS="${4:-1}"
N_VIDEO_TRIALS="${5:-0}"
N_ROLLOUTS="${6:-500}"
DEBUG=false
RESUME=false
for arg in "$@"; do
    [[ "${arg}" == "--debug" ]] && DEBUG=true
    [[ "${arg}" == "--resume" ]] && RESUME=true
done

# Derive experiment name from the directory before "checkpoints/".
if [[ -f "${CHECKPOINT_PATH}" ]]; then
    EXPERIMENT_NAME="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
else
    EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
fi

if [[ "${DEBUG}" == "true" ]]; then
    echo "DEBUG mode enabled: n_envs=1, n_rollouts=1, n_video_trials=0, headless=false"
    N_ENVS=1 N_ROLLOUTS=1 N_VIDEO_TRIALS=0
fi

HEADLESS_FLAG="--headless"
[[ "${DEBUG}" == "true" ]] && HEADLESS_FLAG=""
RESUME_FLAG=""
[[ "${RESUME}" == "true" ]] && RESUME_FLAG="--resume"

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
BASE_OUT="outputs/${EXPERIMENT_NAME}"
echo "Base output directory: ${BASE_OUT}"
echo "Checkpoints found: ${#CHECKPOINTS[@]}"

for CHECKPOINT in "${CHECKPOINTS[@]}"; do
    CKPT_STEM="$(basename "${CHECKPOINT}" .ckpt)"
    OUT_DIR="${BASE_OUT}/T_a_${N_ACTION_STEPS}/${CKPT_STEM}"
    echo ""
    echo "########################################## "
    echo "Checkpoint: ${CKPT_STEM}"
    echo "Action horizon: ${N_ACTION_STEPS}  ->  ${OUT_DIR}"
    echo "##########################################"
    if ! python src/eval/evaluate_model_custom.py \
        --checkpoint "${CHECKPOINT}" \
        --furniture "${FURNITURE}" \
        --n-rollouts "${N_ROLLOUTS}" \
        --n-envs "${N_ENVS}" \
        --n-action-steps "${N_ACTION_STEPS}" \
        --n-video-trials "${N_VIDEO_TRIALS}" \
        --record-failures \
        --output-dir "${OUT_DIR}" \
        ${RESUME_FLAG} \
        ${HEADLESS_FLAG}; then
        echo "ERROR: evaluate_model_custom.py failed for action_horizon=${N_ACTION_STEPS}, checkpoint=${CKPT_STEM}" >&2
        continue
    fi
done
