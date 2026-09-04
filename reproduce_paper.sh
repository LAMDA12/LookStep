#!/usr/bin/env bash
# Staged reproduction entry point for the LookStep paper.
set -euo pipefail

LOOKSTEP_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_PARENT=$(cd -- "${LOOKSTEP_ROOT}/.." && pwd)
DATA_ROOT=${DATA_ROOT:-${LOOKSTEP_DATA_ROOT:-${PROJECT_PARENT}/data}}
PYTHON_BIN=${PYTHON_BIN:-python}
VERSION_ARGS=(--strict-versions)
if [ "${ALLOW_VERSION_DRIFT:-0}" = "1" ]; then
  VERSION_ARGS=()
fi

R2R_SOURCE=${R2R_SOURCE:-${LOOKSTEP_ROOT}/data_construction/outputs/r2r_event_fifo_full.jsonl}
RXR_SOURCE=${RXR_SOURCE:-${LOOKSTEP_ROOT}/data_construction/outputs/rxr_event_fifo_full.jsonl}
R2R_SWIFT=${R2R_SWIFT:-${LOOKSTEP_ROOT}/model_training/data/r2r_event_fifo_ms_swift}
RXR_SWIFT=${RXR_SWIFT:-${LOOKSTEP_ROOT}/model_training/data/rxr_event_fifo_ms_swift}

usage() {
  echo "Usage: bash reproduce_paper.sh STAGE  # or: bash LookStep/reproduce_paper.sh STAGE"
  echo
  echo "Stages:"
  echo "  test          Run dependency-light deterministic regression tests"
  echo "  check-data    Validate annotations and expert RGB trajectories"
  echo "  build-data    Build deterministic LookStep labels for R2R and RxR"
  echo "  prepare-data  Convert both datasets to the exact ms-swift splits"
  echo "  check-train   Validate the training environment and converted data"
  echo "  train         Run the released full-finetuning recipe"
  echo "  check-sim     Validate Habitat, MP3D, and MODEL_PATH"
  echo "  eval-r2r      Evaluate the downloaded checkpoint on R2R val-unseen"
  echo "  eval-rxr      Evaluate the downloaded checkpoint on RxR val-unseen"
  echo "  eval-all      Run both paper benchmarks"
  echo "  verify        Compare metrics.json files with the paper values"
  echo "  smoke-r2r     Run two R2R episodes to verify the end-to-end setup"
}

require_model_path() {
  if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: set MODEL_PATH to the downloaded LookStep checkpoint." >&2
    exit 2
  fi
}

cd "${PROJECT_PARENT}"

case "${1:-}" in
  test)
    PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/test_core.py"
    ;;
  check-data)
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/check_setup.py" \
      --stage data --data-root "${DATA_ROOT}" "${VERSION_ARGS[@]}"
    ;;
  build-data)
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/data_construction/build_short_label_dataset.py" \
      --dataset r2r \
      --dataset-root "${DATA_ROOT}/datasets" \
      --trajectory-root "${DATA_ROOT}/trajectory_data" \
      --history-policy event_fifo \
      --future-window 5 \
      --max-long-memory 6 \
      --recent-size 2 \
      --path-mode relative \
      --output-jsonl "${R2R_SOURCE}"
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/data_construction/build_short_label_dataset.py" \
      --dataset rxr \
      --dataset-root "${DATA_ROOT}/datasets" \
      --trajectory-root "${DATA_ROOT}/trajectory_data" \
      --history-policy event_fifo \
      --future-window 5 \
      --max-long-memory 6 \
      --recent-size 2 \
      --path-mode relative \
      --output-jsonl "${RXR_SOURCE}"
    ;;
  prepare-data)
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/model_training/prepare_ms_swift_data.py" \
      --input-jsonl "${R2R_SOURCE}" \
      --output-dir "${R2R_SWIFT}" \
      --image-root "${PROJECT_PARENT}" \
      --val-ratio 0.02 \
      --seed 7 \
      --path-mode absolute
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/model_training/prepare_ms_swift_data.py" \
      --input-jsonl "${RXR_SOURCE}" \
      --output-dir "${RXR_SWIFT}" \
      --image-root "${PROJECT_PARENT}" \
      --val-ratio 0 \
      --seed 7 \
      --path-mode absolute
    ;;
  check-train)
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/check_setup.py" \
      --stage training \
      --data-root "${DATA_ROOT}" \
      --training-data-root "${LOOKSTEP_ROOT}/model_training/data" \
      "${VERSION_ARGS[@]}"
    ;;
  train)
    DATA_DIR="${R2R_SWIFT}" RXR_DATA_DIR="${RXR_SWIFT}" \
      bash "${LOOKSTEP_ROOT}/model_training/scripts/train_qwen3vl_full.sh"
    ;;
  check-sim)
    require_model_path
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/check_setup.py" \
      --stage simulation \
      --data-root "${DATA_ROOT}" \
      --model-path "${MODEL_PATH}" \
      --processor-path "${PROCESSOR_PATH:-${MODEL_PATH}}" \
      "${VERSION_ARGS[@]}"
    ;;
  eval-r2r)
    require_model_path
    DATA_ROOT="${DATA_ROOT}" \
      bash "${LOOKSTEP_ROOT}/simulation/scripts/run_r2r.sh"
    ;;
  eval-rxr)
    require_model_path
    DATA_ROOT="${DATA_ROOT}" \
      bash "${LOOKSTEP_ROOT}/simulation/scripts/run_rxr.sh"
    ;;
  eval-all)
    require_model_path
    DATA_ROOT="${DATA_ROOT}" \
      bash "${LOOKSTEP_ROOT}/simulation/scripts/run_r2r.sh"
    DATA_ROOT="${DATA_ROOT}" \
      bash "${LOOKSTEP_ROOT}/simulation/scripts/run_rxr.sh"
    ;;
  verify)
    "${PYTHON_BIN}" "${LOOKSTEP_ROOT}/simulation/verify_paper_results.py"
    ;;
  smoke-r2r)
    require_model_path
    DATA_ROOT="${DATA_ROOT}" LIMIT_EPISODES=2 NPROC_PER_NODE=1 \
      OUTPUT_PATH="${LOOKSTEP_ROOT}/simulation/outputs/smoke_r2r" \
      bash "${LOOKSTEP_ROOT}/simulation/scripts/run_r2r.sh"
    ;;
  *)
    usage
    exit 2
    ;;
esac
