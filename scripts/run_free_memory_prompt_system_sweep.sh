#!/usr/bin/env bash
set -euo pipefail

# Train + eval sweep for free-memory tokens with optional prompt-system modes.
#
# Usage examples:
#   CKPT=/path/to/sam3.pt OUT_ROOT=/path/to/out bash scripts/run_free_memory_prompt_system_sweep.sh prostate
#   CKPT=/path/to/sam3.pt OUT_ROOT=/path/to/out STAGES=train,eval bash scripts/run_free_memory_prompt_system_sweep.sh prostate breast_tumor
#
# Defaults:
# - If no dataset names are passed, all supported datasets are run.
# - Prostate runs raw/canonical/expanded by default because that is the best
#   place to validate the prompt-system change.
# - Other datasets run raw only unless PROMPT_MODES is set explicitly.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CKPT:?Please set CKPT to the SAM3 checkpoint path}"
: "${OUT_ROOT:?Please set OUT_ROOT to the output root directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
NUM_TOKENS="${NUM_TOKENS:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-2}"
TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
FREE_MEMORY_TOPK="${FREE_MEMORY_TOPK:-4}"
STATIC_MEMORY_TEXT_WEIGHT="${STATIC_MEMORY_TEXT_WEIGHT:-1.0}"
NO_OBJECT_THRESHOLD="${NO_OBJECT_THRESHOLD:-0.2}"
MEMORY_TRIGGER_THRESHOLD="${MEMORY_TRIGGER_THRESHOLD:-0.8}"
HYBRID_FUSION="${HYBRID_FUSION:-score_weighted}"
HYBRID_SCORE_TEMPERATURE="${HYBRID_SCORE_TEMPERATURE:-0.2}"
PROMPT_SYS_TEST_PROMPT="${PROMPT_SYS_TEST_PROMPT:-prostate gland}"
IMAGE_STEM_PREFIX="${IMAGE_STEM_PREFIX:-}"
IMAGE_STEM_SUFFIX="${IMAGE_STEM_SUFFIX:-}"
MASK_STEM_PREFIX="${MASK_STEM_PREFIX:-}"
MASK_STEM_SUFFIX="${MASK_STEM_SUFFIX:-}"
ATTR_JSON="${ATTR_JSON:-$PROJECT_ROOT/scripts/prompt_templates/attributes_template.json}"
ALIAS_JSON="${ALIAS_JSON:-$PROJECT_ROOT/scripts/prompt_templates/aliases_template.json}"
STAGES="${STAGES:-train,eval}"

SUPPORTED_DATASETS=(cervical breast_tumor prostate retinal_vessel fundus_cup fundus_disk nuclei)

has_stage() {
  local needle="$1"
  [[ ",${STAGES}," == *",${needle},"* ]]
}

dataset_selected() {
  local target="$1"
  shift
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  local wanted
  for wanted in "$@"; do
    if [[ "$wanted" == "$target" ]]; then
      return 0
    fi
  done
  return 1
}

prompt_modes_for_dataset() {
  local dataset="$1"
  if [[ -n "${PROMPT_MODES:-}" ]]; then
    echo "$PROMPT_MODES"
    return 0
  fi
  if [[ "$dataset" == "prostate" || "$dataset" == "breast_tumor" ]]; then
    echo "raw canonical expanded"
  else
    echo "raw"
  fi
}

dataset_base_dir() {
  case "$1" in
    cervical) echo "/mnt/diskB/zhw/CervicalCancer_JPG" ;;
    breast_tumor) echo "/home/zhanghanwen/BreastTumor_npy" ;;
    prostate) echo "/mnt/diskB/zhw/Prostate_82_256_JPG" ;;
    retinal_vessel) echo "/mnt/diskB/zhw/Retinal_vessel_1024" ;;
    fundus_cup) echo "/home/zhanghanwen/fundus_1024_256_cup" ;;
    fundus_disk) echo "/home/zhanghanwen/fundus_1024_256_disk" ;;
    nuclei) echo "/mnt/diskB/zhw/Nuclei_82_1024_JPG" ;;
    *) return 1 ;;
  esac
}

dataset_sites() {
  case "$1" in
    cervical) echo "A B C D" ;;
    breast_tumor) echo "BUSI UCLM" ;;
    prostate) echo "BIDMC HK I2CVB ISBI ISBI_1.5 UCL" ;;
    retinal_vessel) echo "CHASEDB1 DRIVE FIVES HRF LES-AV RETA STARE TRENDS" ;;
    fundus_cup) echo "fundus1 fundus2 fundus3 fundus4" ;;
    fundus_disk) echo "fundus1 fundus2 fundus3 fundus4" ;;
    nuclei) echo "MoNuSAC2018 MoNuSAC2020 PanNuke2Adrenal_gland PanNuke2Esophagus PanNuke3Bile-duct PanNuke3Uterus TNBC" ;;
    *) return 1 ;;
  esac
}

dataset_prompt() {
  case "$1" in
    cervical) echo "cervical lesion" ;;
    breast_tumor) echo "breast tumor" ;;
    prostate) echo "prostate" ;;
    retinal_vessel) echo "retinal vessel" ;;
    fundus_cup) echo "optic cup" ;;
    fundus_disk) echo "optic disc" ;;
    nuclei) echo "nucleus" ;;
    *) return 1 ;;
  esac
}

train_one() {
  local dataset="$1"
  local site="$2"
  local mode="$3"

  local base_dir
  base_dir="$(dataset_base_dir "$dataset")"
  local prompt
  prompt="$(dataset_prompt "$dataset")"
  if [[ "$dataset" == "prostate" && "$mode" != "raw" ]]; then
    prompt="$PROMPT_SYS_TEST_PROMPT"
  fi

  local image_dir="$base_dir/$site/data_npy"
  local mask_dir="$base_dir/$site/label_npy"
  local output_path="$OUT_ROOT/$dataset/$mode/$site/free_memory_tokens.pt"

  mkdir -p "$(dirname "$output_path")"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_free_memory_tokens.py"
    --image-dir "$image_dir"
    --mask-dir "$mask_dir"
    --output-path "$output_path"
    --checkpoint-path "$CKPT"
    --text-prompt "$prompt"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --num-tokens "$NUM_TOKENS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --image-stem-prefix "$IMAGE_STEM_PREFIX"
    --image-stem-suffix "$IMAGE_STEM_SUFFIX"
    --mask-stem-prefix "$MASK_STEM_PREFIX"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --prompt-system-mode "$mode"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
  )

  if [[ -n "$TRAIN_LIMIT" ]]; then
    cmd+=(--limit "$TRAIN_LIMIT")
  fi

  echo "[TRAIN][$dataset][$site][$mode] ${cmd[*]}"
  "${cmd[@]}"
}

eval_one() {
  local dataset="$1"
  local site="$2"
  local mode="$3"

  local base_dir
  base_dir="$(dataset_base_dir "$dataset")"
  local prompt
  prompt="$(dataset_prompt "$dataset")"
  if [[ "$dataset" == "prostate" && "$mode" != "raw" ]]; then
    prompt="$PROMPT_SYS_TEST_PROMPT"
  fi

  local image_dir="$base_dir/$site/val_data_npy"
  local mask_dir="$base_dir/$site/val_label_npy"
  local free_ckpt="$OUT_ROOT/$dataset/$mode/$site/free_memory_tokens.pt"
  local eval_dir="$OUT_ROOT/$dataset/$mode/$site/eval"

  mkdir -p "$eval_dir"

  local cmd=(
    "$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval_medical_static_memory.py"
    --image-dir "$image_dir"
    --mask-dir "$mask_dir"
    --checkpoint-path "$CKPT"
    --free-memory-ckpt "$free_ckpt"
    --output-dir "$eval_dir"
    --text-prompt "$prompt"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --free-memory-num-tokens "$NUM_TOKENS"
    --no-object-threshold "$NO_OBJECT_THRESHOLD"
    --memory-trigger-threshold "$MEMORY_TRIGGER_THRESHOLD"
    --save-predictions
    --image-stem-prefix "$IMAGE_STEM_PREFIX"
    --image-stem-suffix "$IMAGE_STEM_SUFFIX"
    --mask-stem-prefix "$MASK_STEM_PREFIX"
    --mask-stem-suffix "$MASK_STEM_SUFFIX"
    --prompt-system-mode "$mode"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
  )

  echo "[EVAL][$dataset][$site][$mode] ${cmd[*]}"
  "${cmd[@]}"
}

run_dataset() {
  local dataset="$1"
  local base_dir
  base_dir="$(dataset_base_dir "$dataset")"
  local sites
  sites="$(dataset_sites "$dataset")"
  local prompt_modes
  prompt_modes="$(prompt_modes_for_dataset "$dataset")"

  for mode in $prompt_modes; do
    for site in $sites; do
      if has_stage train; then
        train_one "$dataset" "$site" "$mode"
      fi
      if has_stage eval; then
        eval_one "$dataset" "$site" "$mode"
      fi
    done
  done
}

main() {
  mkdir -p "$OUT_ROOT"

  local selected=("$@")
  if [[ "${#selected[@]}" -eq 0 ]]; then
    selected=("${SUPPORTED_DATASETS[@]}")
  fi

  local dataset
  for dataset in "${selected[@]}"; do
    dataset_selected "$dataset" "${SUPPORTED_DATASETS[@]}" || {
      echo "Unknown dataset: $dataset" >&2
      echo "Supported: ${SUPPORTED_DATASETS[*]}" >&2
      exit 1
    }
    run_dataset "$dataset"
  done
}

main "$@"
