#!/usr/bin/env bash
set -euo pipefail

# Five-way prostate ablation:
# 1. SAM3 baseline with text prompt only
# 2. SAM3 + federated prompt tuning
# 3. SAM3 + federated prompt tuning + non-trained task encoder
# 4. SAM3 + federated prompt tuning + trained task encoder
# 5. SAM3 + trained task encoder

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CKPT:?Please set CKPT to the SAM3 checkpoint path}"
: "${OUT_ROOT:?Please set OUT_ROOT to the output root directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_ROOT="${DATA_ROOT:-/mnt/diskB/zhw/Prostate_82_256_JPG}"
SITES="${SITES:-BIDMC HK I2CVB ISBI ISBI_1.5 UCL}"
TEXT_PROMPT="${TEXT_PROMPT:-prostate}"
PROMPT_SYSTEM_MODE="${PROMPT_SYSTEM_MODE:-expanded}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
NUM_TOKENS="${NUM_TOKENS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-2}"
TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
TASK_ENCODER_SAMPLE_COUNT="${TASK_ENCODER_SAMPLE_COUNT:-16}"
TASK_ENCODER_NUM_TOKENS="${TASK_ENCODER_NUM_TOKENS:-4}"
TASK_ENCODER_SEED="${TASK_ENCODER_SEED:-0}"
TASK_ENCODER_BATCH_SIZE="${TASK_ENCODER_BATCH_SIZE:-4}"
TASK_ENCODER_SUPPORT_SIZE="${TASK_ENCODER_SUPPORT_SIZE:-8}"
TASK_ENCODER_QUERY_BATCH_SIZE="${TASK_ENCODER_QUERY_BATCH_SIZE:-1}"
TASK_ENCODER_EPOCHS="${TASK_ENCODER_EPOCHS:-$EPOCHS}"
TASK_ENCODER_LR="${TASK_ENCODER_LR:-1e-3}"
TASK_ENCODER_LIMIT="${TASK_ENCODER_LIMIT:-$TRAIN_LIMIT}"
NO_OBJECT_THRESHOLD="${NO_OBJECT_THRESHOLD:-0.2}"
MEMORY_TRIGGER_THRESHOLD="${MEMORY_TRIGGER_THRESHOLD:-0.8}"
IMAGE_STEM_PREFIX="${IMAGE_STEM_PREFIX:-}"
IMAGE_STEM_SUFFIX="${IMAGE_STEM_SUFFIX:-}"
MASK_STEM_PREFIX="${MASK_STEM_PREFIX:-}"
MASK_STEM_SUFFIX="${MASK_STEM_SUFFIX:-}"
PAIRING_MODE="${PAIRING_MODE:-stem}"
ATTR_JSON="${ATTR_JSON:-$PROJECT_ROOT/scripts/prompt_templates/attributes_template.json}"
ALIAS_JSON="${ALIAS_JSON:-$PROJECT_ROOT/scripts/prompt_templates/aliases_template.json}"
STAGES="${STAGES:-train,eval}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

has_stage() {
  local needle="$1"
  [[ ",${STAGES}," == *",${needle},"* ]]
}

maybe_limit_arg() {
  local limit_value="$1"
  if [[ -n "$limit_value" ]]; then
    printf '%s\n' --limit "$limit_value"
  fi
}

common_eval_args() {
  local site="$1"
  local output_dir="$2"
  local image_dir="$DATA_ROOT/$site/val_data_npy"
  local mask_dir="$DATA_ROOT/$site/val_label_npy"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval_medical_static_memory.py"
    --image-dir "$image_dir"
    --mask-dir "$mask_dir"
    --checkpoint-path "$CKPT"
    --output-dir "$output_dir"
    --text-prompt "$TEXT_PROMPT"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --no-object-threshold "$NO_OBJECT_THRESHOLD"
    --memory-trigger-threshold "$MEMORY_TRIGGER_THRESHOLD"
    --image-stem-prefix "$IMAGE_STEM_PREFIX"
    --image-stem-suffix "$IMAGE_STEM_SUFFIX"
    --mask-stem-prefix "$MASK_STEM_PREFIX"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --pairing-mode "$PAIRING_MODE"
    --prompt-system-mode "$PROMPT_SYSTEM_MODE"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
  )
  if [[ "$SAVE_PREDICTIONS" != "0" ]]; then
    cmd+=(--save-predictions)
  fi
  if [[ -n "$EVAL_LIMIT" ]]; then
    cmd+=(--limit "$EVAL_LIMIT")
  fi
  printf '%q ' "${cmd[@]}"
}

train_federated_prompt_tuning() {
  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_federated_free_memory_tokens_fedavg.py"
    --dataset-root "$DATA_ROOT"
    --checkpoint-path "$CKPT"
    --output-dir "$OUT_ROOT/prostate/fed_prompt_tuning"
    --run-all-holdouts
    --site-names $SITES
    --text-prompt "$TEXT_PROMPT"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --num-tokens "$NUM_TOKENS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --prompt-system-mode "$PROMPT_SYSTEM_MODE"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
    --skip-eval
  )
  if [[ -n "$TRAIN_LIMIT" ]]; then
    cmd+=(--limit-per-site "$TRAIN_LIMIT")
  fi

  echo "[TRAIN][prostate][fed_prompt_tuning] ${cmd[*]}"
  "${cmd[@]}"
}

train_task_encoder() {
  local site="$1"
  local output_dir="$OUT_ROOT/prostate/trained_task_encoder/$site"
  local output_path="$output_dir/task_encoder.pt"
  mkdir -p "$output_dir"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_task_encoder.py"
    --image-dir "$DATA_ROOT/$site/data_npy"
    --mask-dir "$DATA_ROOT/$site/label_npy"
    --output-path "$output_path"
    --checkpoint-path "$CKPT"
    --text-prompt "$TEXT_PROMPT"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --num-task-tokens "$TASK_ENCODER_NUM_TOKENS"
    --support-size "$TASK_ENCODER_SUPPORT_SIZE"
    --query-batch-size "$TASK_ENCODER_QUERY_BATCH_SIZE"
    --support-batch-size "$TASK_ENCODER_BATCH_SIZE"
    --epochs "$TASK_ENCODER_EPOCHS"
    --lr "$TASK_ENCODER_LR"
    --image-stem-prefix "$IMAGE_STEM_PREFIX"
    --image-stem-suffix "$IMAGE_STEM_SUFFIX"
    --mask-stem-prefix "$MASK_STEM_PREFIX"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --pairing-mode "$PAIRING_MODE"
    --prompt-system-mode "$PROMPT_SYSTEM_MODE"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
    --seed "$TASK_ENCODER_SEED"
  )
  if [[ -n "$TASK_ENCODER_LIMIT" ]]; then
    cmd+=(--limit "$TASK_ENCODER_LIMIT")
  fi

  echo "[TRAIN][prostate][$site][trained_task_encoder] ${cmd[*]}"
  "${cmd[@]}"
}

eval_baseline() {
  local site="$1"
  local output_dir="$OUT_ROOT/prostate/baseline/$site/eval"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$site" "$output_dir")"
  echo "[EVAL][prostate][$site][baseline] $cmd"
  eval "$cmd"
}

eval_prompt_tuning() {
  local site="$1"
  local output_dir="$OUT_ROOT/prostate/fed_prompt_tuning/$site/eval"
  local free_ckpt="$OUT_ROOT/prostate/fed_prompt_tuning/$site/fedavg_free_memory_tokens.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$site" "$output_dir")"
  cmd="$cmd --free-memory-ckpt $(printf '%q' "$free_ckpt") --free-memory-num-tokens $(printf '%q' "$NUM_TOKENS")"
  echo "[EVAL][prostate][$site][fed_prompt_tuning] $cmd"
  eval "$cmd"
}

eval_nontrained_task_encoder_with_prompt_tuning() {
  local site="$1"
  local output_dir="$OUT_ROOT/prostate/fed_prompt_tuning_nontrained_task_encoder/$site/eval"
  local free_ckpt="$OUT_ROOT/prostate/fed_prompt_tuning/$site/fedavg_free_memory_tokens.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$site" "$output_dir")"
  cmd="$cmd --free-memory-ckpt $(printf '%q' "$free_ckpt") --free-memory-num-tokens $(printf '%q' "$NUM_TOKENS")"
  cmd="$cmd --task-encoder-pool-image-dir $(printf '%q' "$DATA_ROOT/$site/data_npy")"
  cmd="$cmd --task-encoder-pool-mask-dir $(printf '%q' "$DATA_ROOT/$site/label_npy")"
  cmd="$cmd --task-encoder-sample-count $(printf '%q' "$TASK_ENCODER_SAMPLE_COUNT")"
  cmd="$cmd --task-encoder-seed $(printf '%q' "$TASK_ENCODER_SEED")"
  cmd="$cmd --task-encoder-batch-size $(printf '%q' "$TASK_ENCODER_BATCH_SIZE")"
  echo "[EVAL][prostate][$site][fed_prompt_tuning_nontrained_task_encoder] $cmd"
  eval "$cmd"
}

eval_trained_task_encoder() {
  local site="$1"
  local output_dir="$OUT_ROOT/prostate/trained_task_encoder/$site/eval"
  local task_ckpt="$OUT_ROOT/prostate/trained_task_encoder/$site/task_encoder.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$site" "$output_dir")"
  cmd="$cmd --task-encoder-pool-image-dir $(printf '%q' "$DATA_ROOT/$site/data_npy")"
  cmd="$cmd --task-encoder-pool-mask-dir $(printf '%q' "$DATA_ROOT/$site/label_npy")"
  cmd="$cmd --task-encoder-ckpt $(printf '%q' "$task_ckpt")"
  cmd="$cmd --task-encoder-sample-count $(printf '%q' "$TASK_ENCODER_SAMPLE_COUNT")"
  cmd="$cmd --task-encoder-seed $(printf '%q' "$TASK_ENCODER_SEED")"
  cmd="$cmd --task-encoder-batch-size $(printf '%q' "$TASK_ENCODER_BATCH_SIZE")"
  echo "[EVAL][prostate][$site][trained_task_encoder] $cmd"
  eval "$cmd"
}

eval_prompt_tuning_trained_task_encoder() {
  local site="$1"
  local output_dir="$OUT_ROOT/prostate/fed_prompt_tuning_trained_task_encoder/$site/eval"
  local free_ckpt="$OUT_ROOT/prostate/fed_prompt_tuning/$site/fedavg_free_memory_tokens.pt"
  local task_ckpt="$OUT_ROOT/prostate/trained_task_encoder/$site/task_encoder.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$site" "$output_dir")"
  cmd="$cmd --free-memory-ckpt $(printf '%q' "$free_ckpt") --free-memory-num-tokens $(printf '%q' "$NUM_TOKENS")"
  cmd="$cmd --task-encoder-pool-image-dir $(printf '%q' "$DATA_ROOT/$site/data_npy")"
  cmd="$cmd --task-encoder-pool-mask-dir $(printf '%q' "$DATA_ROOT/$site/label_npy")"
  cmd="$cmd --task-encoder-ckpt $(printf '%q' "$task_ckpt")"
  cmd="$cmd --task-encoder-sample-count $(printf '%q' "$TASK_ENCODER_SAMPLE_COUNT")"
  cmd="$cmd --task-encoder-seed $(printf '%q' "$TASK_ENCODER_SEED")"
  cmd="$cmd --task-encoder-batch-size $(printf '%q' "$TASK_ENCODER_BATCH_SIZE")"
  echo "[EVAL][prostate][$site][fed_prompt_tuning_trained_task_encoder] $cmd"
  eval "$cmd"
}

summarize_site() {
  local site="$1"
  "$PYTHON_BIN" - "$OUT_ROOT" "$site" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "prostate"
site = sys.argv[2]
rows = [
    ("baseline", "baseline_mean"),
    ("fed_prompt_tuning", "static_memory_mean"),
    ("fed_prompt_tuning_nontrained_task_encoder", "task_encoder_mean"),
    ("fed_prompt_tuning_trained_task_encoder", "task_encoder_mean"),
    ("trained_task_encoder", "task_encoder_mean"),
]
for name, key in rows:
    metrics_path = root / name / site / "eval" / "metrics.json"
    if not metrics_path.exists():
        print(f"{site}\t{name}\tMISSING\tMISSING")
        continue
    payload = json.loads(metrics_path.read_text())
    metrics = payload.get(key) or payload.get("static_memory_mean") or payload.get("baseline_mean")
    print(f"{site}\t{name}\t{metrics['dice']:.6f}\t{metrics['iou']:.6f}")
PY
}

main() {
  mkdir -p "$OUT_ROOT/prostate"
  if has_stage train; then
    train_federated_prompt_tuning
  fi
  for site in $SITES; do
    if has_stage train; then
      train_task_encoder "$site"
    fi
    if has_stage eval; then
      eval_baseline "$site"
      eval_prompt_tuning "$site"
      eval_nontrained_task_encoder_with_prompt_tuning "$site"
      eval_prompt_tuning_trained_task_encoder "$site"
      eval_trained_task_encoder "$site"
      summarize_site "$site" | tee "$OUT_ROOT/prostate/$site.summary.tsv"
    fi
  done
}

main "$@"
