#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_openfoam_block_pilot_50/manifest.json}"
SPLIT_PATH="${SPLIT_PATH:-configs/d1_openfoam_block_pilot_50_split_seed42.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/d1_openfoam_block_scratch_smoke}"
EVAL_JSON="${EVAL_JSON:-outputs/logs/d1_openfoam_block_scratch_smoke_test.json}"
EPOCHS="${EPOCHS:-3}"
POINT_BUDGET="${POINT_BUDGET:-1024}"
EVAL_POINT_BUDGET="${EVAL_POINT_BUDGET:-1024}"
MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-8}"
MAX_VAL_CASES="${MAX_VAL_CASES:-4}"
N_HIDDEN="${N_HIDDEN:-128}"
N_LAYERS="${N_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
SLICE_NUM="${SLICE_NUM:-16}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-42}"

for path in "$D1_MANIFEST" "$SPLIT_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

"$PY" scripts/train_finetune_d1.py \
  --case-manifest "$D1_MANIFEST" \
  --split-path "$SPLIT_PATH" \
  --train-split train \
  --val-split val \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size 1 \
  --point-budget "$POINT_BUDGET" \
  --eval-point-budget "$EVAL_POINT_BUDGET" \
  --max-train-cases "$MAX_TRAIN_CASES" \
  --max-val-cases "$MAX_VAL_CASES" \
  --n-hidden "$N_HIDDEN" \
  --n-layers "$N_LAYERS" \
  --n-heads "$N_HEADS" \
  --slice-num "$SLICE_NUM" \
  --device "$DEVICE" \
  --seed "$SEED"

"$PY" scripts/evaluate_d1.py \
  --case-manifest "$D1_MANIFEST" \
  --split-path "$SPLIT_PATH" \
  --split test \
  --model-dir "$OUTPUT_DIR" \
  --output-json "$EVAL_JSON" \
  --point-budget "$EVAL_POINT_BUDGET" \
  --device "$DEVICE"

echo "M1 OpenFOAM downstream smoke complete"
echo "checkpoint: $OUTPUT_DIR"
echo "test_eval: $EVAL_JSON"
