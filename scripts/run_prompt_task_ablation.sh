#!/usr/bin/env bash
set -euo pipefail

# Five-way prompt/task ablation for medical segmentation datasets:
# 1. SAM3 baseline with text prompt only
# 2. SAM3 + federated prompt tuning
# 3. SAM3 + federated prompt tuning + non-trained task encoder
# 4. SAM3 + federated prompt tuning + trained task encoder
# 5. SAM3 + trained task encoder
#
# Examples:
#   EPOCHS=10 CUDA_VISIBLE_DEVICES=2 bash scripts/run_prompt_task_ablation.sh prostate
#   STAGES=eval bash scripts/run_prompt_task_ablation.sh breast_tumor fundus_disk
#   bash scripts/run_prompt_task_ablation.sh

CKPT="${CKPT:-/home/zhanghanwen/checkpoints/sam3.pt}"
OUT_ROOT="${OUT_ROOT:-/home/zhanghanwen/text-fedsam3}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CKPT:?Please set CKPT to the SAM3 checkpoint path}"
: "${OUT_ROOT:?Please set OUT_ROOT to the output root directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
PROMPT_SYSTEM_MODE="${PROMPT_SYSTEM_MODE:-expanded}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
NUM_TOKENS="${NUM_TOKENS:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-2}"
TRAIN_LIMIT="${TRAIN_LIMIT:-10000}"
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
PAIRING_MODE="${PAIRING_MODE:-}"
ATTR_JSON="${ATTR_JSON:-$PROJECT_ROOT/scripts/prompt_templates/attributes_template.json}"
ALIAS_JSON="${ALIAS_JSON:-$PROJECT_ROOT/scripts/prompt_templates/aliases_template.json}"
STAGES="${STAGES:-train,eval}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-1}"

SUPPORTED_DATASETS=(
  cervical
  breast_tumor
  prostate
  retinal_vessel
  fundus_cup
  fundus_disk
  nuclei
  brain_tumor
  polyp
)

has_stage() {
  local needle="$1"
  [[ ",${STAGES}," == *",${needle},"* ]]
}

dataset_selected() {
  local dataset="$1"
  shift || true
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  local selected
  for selected in "$@"; do
    if [[ "$selected" == "$dataset" ]]; then
      return 0
    fi
  done
  return 1
}

dataset_base_dir() {
  local dataset="$1"
  if [[ -n "${DATA_ROOT:-}" ]]; then
    echo "$DATA_ROOT"
    return
  fi

  case "$dataset" in
    cervical) echo "/mnt/diskB/zhw/CervicalCancer_JPG" ;;
    breast_tumor) echo "/home/zhanghanwen/BreastTumor_npy" ;;
    prostate) echo "/mnt/diskB/zhw/Prostate_82_256_JPG" ;;
    retinal_vessel) echo "/mnt/diskB/zhw/Retinal_vessel_1024" ;;
    fundus_cup) echo "/home/zhanghanwen/fundus_1024_256_cup" ;;
    fundus_disk) echo "/home/zhanghanwen/fundus_1024_256_disk" ;;
    nuclei) echo "/mnt/diskB/zhw/Nuclei_82_1024_JPG" ;;
    brain_tumor) echo "/mnt/diskB/zhw/FeTS2022_FedDG_82_240_2_JPG" ;;
    polyp) echo "/home/zhanghanwen/Polyp_1024_npy" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

dataset_sites() {
  local dataset="$1"
  if [[ -n "${SITES:-}" ]]; then
    echo "$SITES"
    return
  fi

  case "$dataset" in
    cervical) echo "A B C D" ;;
    breast_tumor) echo "BUSI UCLM" ;;
    prostate) echo "BIDMC HK I2CVB ISBI ISBI_1.5 UCL" ;;
    retinal_vessel) echo "CHASEDB1 DRIVE FIVES HRF LES-AV RETA STARE TRENDS" ;;
    fundus_cup) echo "fundus1 fundus2 fundus3 fundus4" ;;
    fundus_disk) echo "fundus1 fundus2 fundus3 fundus4" ;;
    nuclei) echo "MoNuSAC2018 MoNuSAC2020 PanNuke2Adrenal_gland PanNuke2Esophagus PanNuke3Bile-duct PanNuke3Uterus TNBC" ;;
    brain_tumor) echo "1 6 18 21" ;;
    polyp) echo "Clinic Etis Kvasir" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

dataset_prompt() {
  local dataset="$1"
  if [[ -n "${TEXT_PROMPT:-}" ]]; then
    echo "$TEXT_PROMPT"
    return
  fi

  case "$dataset" in
    cervical) echo "cervical tumor region in T2 MRI" ;;
    breast_tumor) echo "abnormal dark breast tumor region in ultrasound image" ;;
    prostate) echo "prostate gland" ;;
    retinal_vessel) echo "thin branching retinal blood vessel tree" ;;
    fundus_cup) echo "central optic cup inside the optic disc" ;;
    fundus_disk) echo "bright circular optic disc boundary in fundus image" ;;
    nuclei) echo "individual cell nuclei" ;;
    brain_tumor) echo "brain tumor lesion in MRI" ;;
    polyp) echo "abnormal protruding polyp region in colonoscopy image" ;;
    *) echo "Unknown dataset: $dataset" >&2; return 1 ;;
  esac
}

pairing_mode_for_dataset() {
  local dataset="$1"
  if [[ -n "$PAIRING_MODE" ]]; then
    echo "$PAIRING_MODE"
    return
  fi

  case "$dataset" in
    retinal_vessel) echo "sequential" ;;
    *) echo "stem" ;;
  esac
}

common_eval_args() {
  local dataset="$1"
  local site="$2"
  local output_dir="$3"
  local data_root="$4"
  local text_prompt="$5"
  local pairing_mode="$6"
  local image_dir="$data_root/$site/val_data_npy"
  local mask_dir="$data_root/$site/val_label_npy"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval_medical_static_memory.py"
    --image-dir "$image_dir"
    --mask-dir "$mask_dir"
    --checkpoint-path "$CKPT"
    --output-dir "$output_dir"
    --text-prompt "$text_prompt"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --no-object-threshold "$NO_OBJECT_THRESHOLD"
    --memory-trigger-threshold "$MEMORY_TRIGGER_THRESHOLD"
    --image-stem-prefix "$IMAGE_STEM_PREFIX"
    --image-stem-suffix "$IMAGE_STEM_SUFFIX"
    --mask-stem-prefix "$MASK_STEM_PREFIX"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --pairing-mode "$pairing_mode"
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
  local dataset="$1"
  local data_root="$2"
  local sites="$3"
  local text_prompt="$4"
  local pairing_mode="$5"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_federated_free_memory_tokens_fedavg.py"
    --dataset-root "$data_root"
    --checkpoint-path "$CKPT"
    --output-dir "$OUT_ROOT/$dataset/fed_prompt_tuning"
    --run-all-holdouts
    --site-names $sites
    --text-prompt "$text_prompt"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --num-tokens "$NUM_TOKENS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --image-stem-prefix "$IMAGE_STEM_PREFIX"
    --image-stem-suffix "$IMAGE_STEM_SUFFIX"
    --mask-stem-prefix "$MASK_STEM_PREFIX"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --pairing-mode "$pairing_mode"
    --prompt-system-mode "$PROMPT_SYSTEM_MODE"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
    --skip-eval
  )
  if [[ -n "$TRAIN_LIMIT" ]]; then
    cmd+=(--limit-per-site "$TRAIN_LIMIT")
  fi

  echo "[TRAIN][$dataset][fed_prompt_tuning] ${cmd[*]}"
  "${cmd[@]}"
}

train_task_encoder() {
  local dataset="$1"
  local site="$2"
  local data_root="$3"
  local text_prompt="$4"
  local pairing_mode="$5"
  local output_dir="$OUT_ROOT/$dataset/trained_task_encoder/$site"
  local output_path="$output_dir/task_encoder.pt"
  mkdir -p "$output_dir"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_task_encoder.py"
    --image-dir "$data_root/$site/data_npy"
    --mask-dir "$data_root/$site/label_npy"
    --output-path "$output_path"
    --checkpoint-path "$CKPT"
    --text-prompt "$text_prompt"
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
    --pairing-mode "$pairing_mode"
    --prompt-system-mode "$PROMPT_SYSTEM_MODE"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
    --seed "$TASK_ENCODER_SEED"
  )
  if [[ -n "$TASK_ENCODER_LIMIT" ]]; then
    cmd+=(--limit "$TASK_ENCODER_LIMIT")
  fi

  echo "[TRAIN][$dataset][$site][trained_task_encoder] ${cmd[*]}"
  "${cmd[@]}"
}

eval_baseline() {
  local dataset="$1"
  local site="$2"
  local data_root="$3"
  local text_prompt="$4"
  local pairing_mode="$5"
  local output_dir="$OUT_ROOT/$dataset/baseline/$site/eval"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$dataset" "$site" "$output_dir" "$data_root" "$text_prompt" "$pairing_mode")"
  echo "[EVAL][$dataset][$site][baseline] $cmd"
  eval "$cmd"
}

eval_prompt_tuning() {
  local dataset="$1"
  local site="$2"
  local data_root="$3"
  local text_prompt="$4"
  local pairing_mode="$5"
  local output_dir="$OUT_ROOT/$dataset/fed_prompt_tuning/$site/eval"
  local free_ckpt="$OUT_ROOT/$dataset/fed_prompt_tuning/$site/fedavg_free_memory_tokens.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$dataset" "$site" "$output_dir" "$data_root" "$text_prompt" "$pairing_mode")"
  cmd="$cmd --free-memory-ckpt $(printf '%q' "$free_ckpt") --free-memory-num-tokens $(printf '%q' "$NUM_TOKENS")"
  echo "[EVAL][$dataset][$site][fed_prompt_tuning] $cmd"
  eval "$cmd"
}

eval_nontrained_task_encoder_with_prompt_tuning() {
  local dataset="$1"
  local site="$2"
  local data_root="$3"
  local text_prompt="$4"
  local pairing_mode="$5"
  local output_dir="$OUT_ROOT/$dataset/fed_prompt_tuning_nontrained_task_encoder/$site/eval"
  local free_ckpt="$OUT_ROOT/$dataset/fed_prompt_tuning/$site/fedavg_free_memory_tokens.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$dataset" "$site" "$output_dir" "$data_root" "$text_prompt" "$pairing_mode")"
  cmd="$cmd --free-memory-ckpt $(printf '%q' "$free_ckpt") --free-memory-num-tokens $(printf '%q' "$NUM_TOKENS")"
  cmd="$cmd --task-encoder-pool-image-dir $(printf '%q' "$data_root/$site/data_npy")"
  cmd="$cmd --task-encoder-pool-mask-dir $(printf '%q' "$data_root/$site/label_npy")"
  cmd="$cmd --task-encoder-sample-count $(printf '%q' "$TASK_ENCODER_SAMPLE_COUNT")"
  cmd="$cmd --task-encoder-seed $(printf '%q' "$TASK_ENCODER_SEED")"
  cmd="$cmd --task-encoder-batch-size $(printf '%q' "$TASK_ENCODER_BATCH_SIZE")"
  echo "[EVAL][$dataset][$site][fed_prompt_tuning_nontrained_task_encoder] $cmd"
  eval "$cmd"
}

eval_trained_task_encoder() {
  local dataset="$1"
  local site="$2"
  local data_root="$3"
  local text_prompt="$4"
  local pairing_mode="$5"
  local output_dir="$OUT_ROOT/$dataset/trained_task_encoder/$site/eval"
  local task_ckpt="$OUT_ROOT/$dataset/trained_task_encoder/$site/task_encoder.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$dataset" "$site" "$output_dir" "$data_root" "$text_prompt" "$pairing_mode")"
  cmd="$cmd --task-encoder-pool-image-dir $(printf '%q' "$data_root/$site/data_npy")"
  cmd="$cmd --task-encoder-pool-mask-dir $(printf '%q' "$data_root/$site/label_npy")"
  cmd="$cmd --task-encoder-ckpt $(printf '%q' "$task_ckpt")"
  cmd="$cmd --task-encoder-sample-count $(printf '%q' "$TASK_ENCODER_SAMPLE_COUNT")"
  cmd="$cmd --task-encoder-seed $(printf '%q' "$TASK_ENCODER_SEED")"
  cmd="$cmd --task-encoder-batch-size $(printf '%q' "$TASK_ENCODER_BATCH_SIZE")"
  echo "[EVAL][$dataset][$site][trained_task_encoder] $cmd"
  eval "$cmd"
}

eval_prompt_tuning_trained_task_encoder() {
  local dataset="$1"
  local site="$2"
  local data_root="$3"
  local text_prompt="$4"
  local pairing_mode="$5"
  local output_dir="$OUT_ROOT/$dataset/fed_prompt_tuning_trained_task_encoder/$site/eval"
  local free_ckpt="$OUT_ROOT/$dataset/fed_prompt_tuning/$site/fedavg_free_memory_tokens.pt"
  local task_ckpt="$OUT_ROOT/$dataset/trained_task_encoder/$site/task_encoder.pt"
  mkdir -p "$output_dir"
  local cmd
  cmd="$(common_eval_args "$dataset" "$site" "$output_dir" "$data_root" "$text_prompt" "$pairing_mode")"
  cmd="$cmd --free-memory-ckpt $(printf '%q' "$free_ckpt") --free-memory-num-tokens $(printf '%q' "$NUM_TOKENS")"
  cmd="$cmd --task-encoder-pool-image-dir $(printf '%q' "$data_root/$site/data_npy")"
  cmd="$cmd --task-encoder-pool-mask-dir $(printf '%q' "$data_root/$site/label_npy")"
  cmd="$cmd --task-encoder-ckpt $(printf '%q' "$task_ckpt")"
  cmd="$cmd --task-encoder-sample-count $(printf '%q' "$TASK_ENCODER_SAMPLE_COUNT")"
  cmd="$cmd --task-encoder-seed $(printf '%q' "$TASK_ENCODER_SEED")"
  cmd="$cmd --task-encoder-batch-size $(printf '%q' "$TASK_ENCODER_BATCH_SIZE")"
  echo "[EVAL][$dataset][$site][fed_prompt_tuning_trained_task_encoder] $cmd"
  eval "$cmd"
}

summarize_site() {
  local dataset="$1"
  local site="$2"
  "$PYTHON_BIN" - "$OUT_ROOT" "$dataset" "$site" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / sys.argv[2]
site = sys.argv[3]
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

run_dataset() {
  local dataset="$1"
  local data_root
  local sites
  local text_prompt
  local pairing_mode

  data_root="$(dataset_base_dir "$dataset")"
  sites="$(dataset_sites "$dataset")"
  text_prompt="$(dataset_prompt "$dataset")"
  pairing_mode="$(pairing_mode_for_dataset "$dataset")"

  mkdir -p "$OUT_ROOT/$dataset"
  echo "[DATASET][$dataset] root=$data_root sites=[$sites] prompt=[$text_prompt] pairing=$pairing_mode"

  if has_stage train; then
    train_federated_prompt_tuning "$dataset" "$data_root" "$sites" "$text_prompt" "$pairing_mode"
  fi

  local site
  for site in $sites; do
    if has_stage train; then
      train_task_encoder "$dataset" "$site" "$data_root" "$text_prompt" "$pairing_mode"
    fi
    if has_stage eval; then
      eval_baseline "$dataset" "$site" "$data_root" "$text_prompt" "$pairing_mode"
      eval_prompt_tuning "$dataset" "$site" "$data_root" "$text_prompt" "$pairing_mode"
      eval_nontrained_task_encoder_with_prompt_tuning "$dataset" "$site" "$data_root" "$text_prompt" "$pairing_mode"
      eval_prompt_tuning_trained_task_encoder "$dataset" "$site" "$data_root" "$text_prompt" "$pairing_mode"
      eval_trained_task_encoder "$dataset" "$site" "$data_root" "$text_prompt" "$pairing_mode"
      summarize_site "$dataset" "$site" | tee "$OUT_ROOT/$dataset/$site.summary.tsv"
    fi
  done
}

main() {
  local requested=("$@")
  local dataset
  for dataset in "${SUPPORTED_DATASETS[@]}"; do
    if dataset_selected "$dataset" "${requested[@]}"; then
      run_dataset "$dataset"
    fi
  done
}

main "$@"
