#!/usr/bin/env bash
set -euo pipefail

# Batch runner for prompt-system experiments with train_free_memory_tokens.py.
# Assumptions:
# - Each site contains `data_npy/` and `label_npy/` under the site directory.
# - The SAM3 checkpoint already exists.
# - You want to compare raw prompts against canonical/expanded prompt handling.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${CKPT:?Please set CKPT to the SAM3 checkpoint path}"
: "${OUT_ROOT:?Please set OUT_ROOT to the output root directory}"

ATTR_JSON="${ATTR_JSON:-$PROJECT_ROOT/scripts/prompt_templates/attributes_template.json}"
ALIAS_JSON="${ALIAS_JSON:-$PROJECT_ROOT/scripts/prompt_templates/aliases_template.json}"
PROMPT_SYS_TEST_PROMPT="${PROMPT_SYS_TEST_PROMPT:-prostate gland}"

DEVICE="${DEVICE:-cuda}"
NUM_TOKENS="${NUM_TOKENS:-4}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-1e-2}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
FREE_LIMIT="${FREE_LIMIT:-100}"

run_train() {
  local image_dir="$1"
  local mask_dir="$2"
  local output_path="$3"
  local text_prompt="$4"
  local prompt_mode="$5"
  local image_stem_suffix="${6:-}"
  local mask_stem_suffix="${7:-}"

  mkdir -p "$(dirname "$output_path")"

  local cmd=(
    python "$PROJECT_ROOT/scripts/train_free_memory_tokens.py"
    --image-dir "$image_dir"
    --mask-dir "$mask_dir"
    --output-path "$output_path"
    --checkpoint-path "$CKPT"
    --text-prompt "$text_prompt"
    --device "$DEVICE"
    --image-size "$IMAGE_SIZE"
    --num-tokens "$NUM_TOKENS"
    --epochs "$EPOCHS"
    --lr "$LR"
    --limit "$FREE_LIMIT"
    --prompt-system-mode "$prompt_mode"
    --attributes-json "$ATTR_JSON"
    --aliases-json "$ALIAS_JSON"
  )

  if [[ -n "$image_stem_suffix" ]]; then
    cmd+=(--image-stem-suffix "$image_stem_suffix")
  fi
  if [[ -n "$mask_stem_suffix" ]]; then
    cmd+=(--mask-stem-suffix "$mask_stem_suffix")
  fi

  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

run_prostate_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/mnt/diskB/zhw/Prostate_82_256_JPG"
  local sites=(BIDMC HK I2CVB ISBI ISBI_1.5 UCL)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/prostate/$tag/$site/free_memory_tokens.pt" \
      "prostate" \
      "$mode"
  done
}

run_cervical_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/mnt/diskB/zhw/CervicalCancer_JPG"
  local sites=(A B C D)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/cervical/$tag/$site/free_memory_tokens.pt" \
      "cervical lesion" \
      "$mode"
  done
}

run_breast_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/home/zhanghanwen/BreastTumor_npy"
  local sites=(BUSI UCLM)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/breast/$tag/$site/free_memory_tokens.pt" \
      "breast tumor" \
      "$mode"
  done
}

run_retinal_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/mnt/diskB/zhw/Retinal_vessel_1024"
  local sites=(CHASEDB1 DRIVE FIVES HRF LES-AV RETA STARE TRENDS)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/retinal/$tag/$site/free_memory_tokens.pt" \
      "retinal vessel" \
      "$mode"
  done
}

run_fundus_cup_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/home/zhanghanwen/fundus_1024_256_cup"
  local sites=(fundus1 fundus2 fundus3 fundus4)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/fundus_cup/$tag/$site/free_memory_tokens.pt" \
      "optic cup" \
      "$mode"
  done
}

run_fundus_disk_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/home/zhanghanwen/fundus_1024_256_disk"
  local sites=(fundus1 fundus2 fundus3 fundus4)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/fundus_disk/$tag/$site/free_memory_tokens.pt" \
      "optic disc" \
      "$mode"
  done
}

run_nuclei_suite() {
  local mode="$1"
  local tag="$2"
  local base_dir="/mnt/diskB/zhw/Nuclei_82_1024_JPG"
  local sites=(MoNuSAC2018 MoNuSAC2020 PanNuke2Adrenal_gland PanNuke2Esophagus PanNuke3Bile-duct PanNuke3Uterus TNBC)
  for site in "${sites[@]}"; do
    run_train \
      "$base_dir/$site/data_npy" \
      "$base_dir/$site/label_npy" \
      "$OUT_ROOT/nuclei/$tag/$site/free_memory_tokens.pt" \
      "nucleus" \
      "$mode"
  done
}

main() {
  mkdir -p "$OUT_ROOT"

  # Baseline raw prompts for all datasets.
  run_cervical_suite raw raw
  run_breast_suite raw raw
  run_prostate_suite raw raw
  run_retinal_suite raw raw
  run_fundus_cup_suite raw raw
  run_fundus_disk_suite raw raw
  run_nuclei_suite raw raw

  # Prompt-system check on prostate. This is the main sanity test for the
  # alias/canonical flow you asked about.
  local prostate_base_dir="/mnt/diskB/zhw/Prostate_82_256_JPG"
  local prostate_sites=(BIDMC HK I2CVB ISBI ISBI_1.5 UCL)
  for site in "${prostate_sites[@]}"; do
    run_train \
      "$prostate_base_dir/$site/data_npy" \
      "$prostate_base_dir/$site/label_npy" \
      "$OUT_ROOT/prostate/canonical/$site/free_memory_tokens.pt" \
      "$PROMPT_SYS_TEST_PROMPT" \
      canonical
    run_train \
      "$prostate_base_dir/$site/data_npy" \
      "$prostate_base_dir/$site/label_npy" \
      "$OUT_ROOT/prostate/expanded/$site/free_memory_tokens.pt" \
      "$PROMPT_SYS_TEST_PROMPT" \
      expanded
  done
}

main "$@"
