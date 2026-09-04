#!/usr/bin/env bash
# LookStep R2R-CE simulator evaluation entry point.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOOKSTEP_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
PROJECT_PARENT=$(cd -- "${LOOKSTEP_ROOT}/.." && pwd)
EVAL_SCRIPT="${LOOKSTEP_ROOT}/simulation/evaluate_short_label_sim.py"

MODEL_PATH=${MODEL_PATH:-${LOOKSTEP_ROOT}/model_training/checkpoints/qwen3vl_event_fifo_full}
PROCESSOR_PATH=${PROCESSOR_PATH:-${MODEL_PATH}}
CONFIG_PATH=${CONFIG_PATH:-${LOOKSTEP_ROOT}/simulation/configs/r2r.yaml}
DATA_ROOT=${DATA_ROOT:-${LOOKSTEP_DATA_ROOT:-${PROJECT_PARENT}/data}}
EVAL_SPLIT=${EVAL_SPLIT:-val_unseen}
OUTPUT_PATH=${OUTPUT_PATH:-${LOOKSTEP_ROOT}/simulation/outputs/r2r_${EVAL_SPLIT}}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_IDS[@]}}
MASTER_PORT=${MASTER_PORT:-29501}

MAX_LONG_MEMORY=${MAX_LONG_MEMORY:-6}
RECENT_SIZE=${RECENT_SIZE:-2}
MAX_STEPS=${MAX_STEPS:-400}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
NUM_BEAMS=${NUM_BEAMS:-1}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
SAVE_STEP_TRACES=${SAVE_STEP_TRACES:-0}
SAVE_VIDEO=${SAVE_VIDEO:-0}
LOG_STEP_TIME=${LOG_STEP_TIME:-0}
SEED=${SEED:-42}
LIMIT_EPISODES=${LIMIT_EPISODES:--1}

export CUDA_VISIBLE_DEVICES
cd "${PROJECT_PARENT}"
mkdir -p "${OUTPUT_PATH}"

CMD=(
  torchrun
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  "$EVAL_SCRIPT"
  --model_path "$MODEL_PATH"
  --processor_path "$PROCESSOR_PATH"
  --habitat_config_path "$CONFIG_PATH"
  --data_root "$DATA_ROOT"
  --eval_split "$EVAL_SPLIT"
  --output_path "$OUTPUT_PATH"
  --max_long_memory "$MAX_LONG_MEMORY"
  --recent_size "$RECENT_SIZE"
  --max_steps "$MAX_STEPS"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --num_beams "$NUM_BEAMS"
  --torch_dtype "$TORCH_DTYPE"
  --seed "$SEED"
  --limit_episodes "$LIMIT_EPISODES"
)

if [ "$LOG_STEP_TIME" = "1" ]; then
  CMD+=(--log_step_time)
fi

if [ "$SAVE_STEP_TRACES" = "1" ]; then
  CMD+=(--save_step_traces)
fi

if [ "$SAVE_VIDEO" = "1" ]; then
  CMD+=(--save_video)
fi

"${CMD[@]}" "$@"
