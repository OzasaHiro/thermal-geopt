#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs
LOG_FILE="outputs/logs/gate_downstream_matrix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "FAILED at line ${LINENO}. See ${LOG_FILE}"' ERR

PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json}"
GATE_SPLIT="${GATE_SPLIT:-configs/d1_proxy_pilot_300_c5_n8192_label_scarcity_split.json}"
PRETRAIN_FULL="${PRETRAIN_FULL:-outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2}"
PRETRAIN_STATIC="${PRETRAIN_STATIC:-outputs/checkpoints/pretrain_gate_static_tdf_only_ep2}"
PRETRAIN_NO_BOUNDARY="${PRETRAIN_NO_BOUNDARY:-outputs/checkpoints/pretrain_gate_no_boundary_field_ep2}"
TRAIN_SIZES="${TRAIN_SIZES:-10 25 50 100}"
EPOCHS="${EPOCHS:-20}"
POINT_BUDGET="${POINT_BUDGET:-4096}"
EVAL_POINT_BUDGET="${EVAL_POINT_BUDGET:-4096}"
MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-0}"
MAX_VAL_CASES="${MAX_VAL_CASES:-0}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"

for path in "$D1_MANIFEST" "$GATE_SPLIT" "$PRETRAIN_FULL/best_model.pt" "$PRETRAIN_STATIC/best_model.pt" "$PRETRAIN_NO_BOUNDARY/best_model.pt"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

echo "Log: $LOG_FILE"
echo "TRAIN_SIZES=$TRAIN_SIZES EPOCHS=$EPOCHS POINT_BUDGET=$POINT_BUDGET MAX_TRAIN_CASES=$MAX_TRAIN_CASES MAX_VAL_CASES=$MAX_VAL_CASES"

for N in $TRAIN_SIZES; do
  echo "=== scratch train_${N} ==="
  "$PY" scripts/train_finetune_d1.py \
    --case-manifest "$D1_MANIFEST" \
    --split-path "$GATE_SPLIT" \
    --train-split "train_${N}" \
    --val-split val \
    --output-dir "outputs/checkpoints/d1_gate_scratch_train${N}_ep${EPOCHS}" \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --point-budget "$POINT_BUDGET" \
    --eval-point-budget "$EVAL_POINT_BUDGET" \
    --max-train-cases "$MAX_TRAIN_CASES" \
    --max-val-cases "$MAX_VAL_CASES" \
    --amp \
    --amp-dtype bfloat16 \
    --device "$DEVICE" \
    --seed "$SEED"

  echo "=== full train_${N} ==="
  "$PY" scripts/train_finetune_d1.py \
    --case-manifest "$D1_MANIFEST" \
    --split-path "$GATE_SPLIT" \
    --train-split "train_${N}" \
    --val-split val \
    --output-dir "outputs/checkpoints/d1_gate_full_train${N}_ep${EPOCHS}" \
    --pretrained-model-dir "$PRETRAIN_FULL" \
    --pretrained-checkpoint-file best_model.pt \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --point-budget "$POINT_BUDGET" \
    --eval-point-budget "$EVAL_POINT_BUDGET" \
    --max-train-cases "$MAX_TRAIN_CASES" \
    --max-val-cases "$MAX_VAL_CASES" \
    --amp \
    --amp-dtype bfloat16 \
    --device "$DEVICE" \
    --seed "$SEED"

  echo "=== static_tdf_only train_${N} ==="
  "$PY" scripts/train_finetune_d1.py \
    --case-manifest "$D1_MANIFEST" \
    --split-path "$GATE_SPLIT" \
    --train-split "train_${N}" \
    --val-split val \
    --output-dir "outputs/checkpoints/d1_gate_static_tdf_only_train${N}_ep${EPOCHS}" \
    --pretrained-model-dir "$PRETRAIN_STATIC" \
    --pretrained-checkpoint-file best_model.pt \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --point-budget "$POINT_BUDGET" \
    --eval-point-budget "$EVAL_POINT_BUDGET" \
    --max-train-cases "$MAX_TRAIN_CASES" \
    --max-val-cases "$MAX_VAL_CASES" \
    --amp \
    --amp-dtype bfloat16 \
    --device "$DEVICE" \
    --seed "$SEED"

  echo "=== no_boundary_field train_${N} ==="
  "$PY" scripts/train_finetune_d1.py \
    --case-manifest "$D1_MANIFEST" \
    --split-path "$GATE_SPLIT" \
    --train-split "train_${N}" \
    --val-split val \
    --output-dir "outputs/checkpoints/d1_gate_no_boundary_field_train${N}_ep${EPOCHS}" \
    --pretrained-model-dir "$PRETRAIN_NO_BOUNDARY" \
    --pretrained-checkpoint-file best_model.pt \
    --epochs "$EPOCHS" \
    --batch-size 1 \
    --point-budget "$POINT_BUDGET" \
    --eval-point-budget "$EVAL_POINT_BUDGET" \
    --max-train-cases "$MAX_TRAIN_CASES" \
    --max-val-cases "$MAX_VAL_CASES" \
    --amp \
    --amp-dtype bfloat16 \
    --device "$DEVICE" \
    --seed "$SEED"
done

echo "Completed downstream matrix. Log: $LOG_FILE"
