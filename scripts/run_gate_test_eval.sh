#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs
LOG_FILE="outputs/logs/gate_test_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "FAILED at line ${LINENO}. See ${LOG_FILE}"' ERR

PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json}"
GATE_SPLIT="${GATE_SPLIT:-configs/d1_proxy_pilot_300_c5_n8192_label_scarcity_split.json}"
TRAIN_SIZES="${TRAIN_SIZES:-10 25 50 100}"
GATE_GROUPS="${GATE_GROUPS:-scratch full static_tdf_only no_boundary_field}"
EPOCHS="${EPOCHS:-20}"
POINT_BUDGET="${POINT_BUDGET:-4096}"
DEVICE="${DEVICE:-cuda}"

for path in "$D1_MANIFEST" "$GATE_SPLIT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

missing=0
for N in $TRAIN_SIZES; do
  for GROUP in $GATE_GROUPS; do
    run_dir="outputs/checkpoints/d1_gate_${GROUP}_train${N}_ep${EPOCHS}"
    if [[ ! -e "$run_dir/config.json" || ! -e "$run_dir/best_model.pt" ]]; then
      echo "Missing checkpoint for eval: $run_dir"
      missing=1
    fi
  done
done
if [[ "$missing" -ne 0 ]]; then
  echo "Run scripts/run_gate_downstream_matrix.sh first, or check EPOCHS/TRAIN_SIZES/GATE_GROUPS."
  exit 1
fi

echo "Log: $LOG_FILE"
echo "TRAIN_SIZES=$TRAIN_SIZES GATE_GROUPS=$GATE_GROUPS EPOCHS=$EPOCHS POINT_BUDGET=$POINT_BUDGET"

for N in $TRAIN_SIZES; do
  for GROUP in $GATE_GROUPS; do
    echo "=== eval ${GROUP} train_${N} ==="
    "$PY" scripts/evaluate_d1.py \
      --case-manifest "$D1_MANIFEST" \
      --split-path "$GATE_SPLIT" \
      --split test \
      --model-dir "outputs/checkpoints/d1_gate_${GROUP}_train${N}_ep${EPOCHS}" \
      --checkpoint-file best_model.pt \
      --point-budget "$POINT_BUDGET" \
      --device "$DEVICE" \
      --output-json "outputs/logs/d1_gate_${GROUP}_train${N}_test_eval.json"
  done
done

"$PY" scripts/summarize_label_scarcity_results.py \
  --train-sizes $TRAIN_SIZES \
  --groups $GATE_GROUPS \
  --run-pattern "outputs/checkpoints/d1_gate_{group}_train{train_size}_ep${EPOCHS}" \
  --expected-case-count 150 \
  --expected-point-budget "$POINT_BUDGET"

echo "Completed test evaluation and summary. Log: $LOG_FILE"
