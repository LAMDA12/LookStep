#!/usr/bin/env bash
# LookStep full-parameter training entry point.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LOOKSTEP_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
DATA_DIR=${DATA_DIR:-${LOOKSTEP_ROOT}/model_training/data/r2r_event_fifo_ms_swift}
RXR_DATA_DIR=${RXR_DATA_DIR:-${LOOKSTEP_ROOT}/model_training/data/rxr_event_fifo_ms_swift}
OUTPUT_DIR=${OUTPUT_DIR:-${LOOKSTEP_ROOT}/model_training/checkpoints/qwen3vl_event_fifo_full}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
MAX_LENGTH=${MAX_LENGTH:-8192}
SAVE_STEPS=${SAVE_STEPS:-3000}
EVAL_STEPS=${EVAL_STEPS:-3000}
SEED=${SEED:-42}
DATA_SEED=${DATA_SEED:-42}
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
NPROC_PER_NODE=${NPROC_PER_NODE:-${#GPU_IDS[@]}}

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE

command -v swift >/dev/null 2>&1 || {
  echo "ERROR: 'swift' was not found. Install the pinned training environment first." >&2
  exit 1
}
for REQUIRED_FILE in "$DATA_DIR/train.jsonl" "$RXR_DATA_DIR/train.jsonl" "$DATA_DIR/val.jsonl"; do
  if [ ! -f "$REQUIRED_FILE" ]; then
    echo "ERROR: missing training file: $REQUIRED_FILE" >&2
    exit 1
  fi
done

swift sft \
  --model "$MODEL_PATH" \
  --model_type qwen3_vl \
  --tuner_type full \
  --dataset \
      "$DATA_DIR/train.jsonl" \
      "$RXR_DATA_DIR/train.jsonl" \
  --val_dataset "$DATA_DIR/val.jsonl" \
  --torch_dtype bfloat16 \
  --deepspeed zero2 \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --per_device_train_batch_size "$TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM_STEPS" \
  --learning_rate "$LEARNING_RATE" \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --optim adamw_torch_fused \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --max_grad_norm 1.0 \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy delete \
  --dataset_num_proc 1 \
  --packing false \
  --padding_free false \
  --freeze_vit true \
  --freeze_aligner true \
  --freeze_llm false \
  --gradient_checkpointing true \
  --seed "$SEED" \
  --data_seed "$DATA_SEED" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --eval_strategy steps \
  --eval_steps "$EVAL_STEPS" \
  --save_total_limit 3 \
  --logging_first_step true \
  --logging_steps 10 \
  --report_to tensorboard \
  --output_dir "$OUTPUT_DIR" \
  "$@"
