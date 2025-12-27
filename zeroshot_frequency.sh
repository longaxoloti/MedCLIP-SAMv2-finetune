#!/bin/bash

# Zeroshot pipeline runner: BiomedCLIP (stage1_best.pth) -> Saliency -> Postprocess -> SAM -> Eval
# Outputs are written under /home/long/projects/MedCLIP-SAMv2-finetune/saliency_results
# Usage:
#   bash run_zeroshot_to_sam.sh brain_tumors
#   bash run_zeroshot_to_sam.sh breast_tumors
#   # or run both
#   bash run_zeroshot_to_sam.sh all

set -euo pipefail

# Activate env (adjust if needed)
source /home/long/venv/zeroshot/bin/activate

ROOT_DIR="/home/long/projects/MedCLIP-SAMv2-finetune"
DATA_DIR="$ROOT_DIR/data"
PROMPTS_DIR="$ROOT_DIR/saliency_maps/text_prompts"
OUT_ROOT="$ROOT_DIR/saliency_results"
SAM_CKPT="$ROOT_DIR/segment-anything/sam_checkpoints/sam_vit_h_4b8939.pth"

mkdir -p "$OUT_ROOT"

run_dataset() {
  local NAME="$1"
  local DS_PATH="$DATA_DIR/$NAME"

  echo "Running zeroshot pipeline for: $NAME"

  local IN_IMAGES="$DS_PATH/test_images"
  local IN_MASKS_GT="$DS_PATH/test_masks"
  local PROMPTS_JSON="$PROMPTS_DIR/${NAME}_testing.json"

  local OUT_SAL="$OUT_ROOT/$NAME/saliency_maps"
  local OUT_COARSE="$OUT_ROOT/$NAME/coarse_masks"
  local OUT_SAM="$OUT_ROOT/$NAME/sam_output"

  mkdir -p "$OUT_SAL" "$OUT_COARSE" "$OUT_SAM"

  # 1) Saliency generation with frequency-aware trained model
  # Choose defaults per dataset
  local VVAR=1.0
  local VBETA=1.0
  local VLAYER=9
  if [[ "$NAME" == "brain_tumors" ]]; then
    VVAR=0.3
    VBETA=2.0
    VLAYER=9
  fi

  python "$ROOT_DIR/saliency_maps/inference_frequency_aware.py" \
    --checkpoint "$ROOT_DIR/saliency_maps/model/stage1_best.pth" \
    --input-path "$IN_IMAGES" \
    --output-path "$OUT_SAL" \
    --json-path "$PROMPTS_JSON" \
    --vvar "$VVAR" \
    --vbeta "$VBETA" \
    --vlayer "$VLAYER" \
    --seed 12 \
    --device cuda

  # 2) Postprocess saliency maps to coarse masks (kmeans, no CRF dependency)
  python "$ROOT_DIR/postprocessing/postprocess_saliency_maps.py" \
    --input-path "$IN_IMAGES" \
    --output-path "$OUT_COARSE" \
    --sal-path "$OUT_SAL" \
    --postprocess kmeans \
    --filter

  # 3) SAM prompting from coarse masks
  python "$ROOT_DIR/segment-anything/prompt_sam.py" \
    --input "$IN_IMAGES" \
    --mask-input "$OUT_COARSE" \
    --output "$OUT_SAM" \
    --model-type vit_h \
    --checkpoint "$SAM_CKPT" \
    --prompts boxes

  # 4) Evaluation
  python "$ROOT_DIR/evaluation/eval.py" \
    --gt_path "$IN_MASKS_GT" \
    --seg_path "$OUT_SAM"

  echo "Finished: $NAME -> $OUT_ROOT/$NAME"
}

TARGET="${1:-all}"
case "$TARGET" in
  brain_tumors)
    run_dataset brain_tumors
    ;;
  breast_tumors)
    run_dataset breast_tumors
    ;;
  all)
    run_dataset brain_tumors
    run_dataset breast_tumors
    ;;
  *)
    echo "Unknown target: $TARGET" >&2
    echo "Usage: bash run_zeroshot_to_sam.sh [brain_tumors|breast_tumors|all]" >&2
    exit 1
    ;;
 esac
