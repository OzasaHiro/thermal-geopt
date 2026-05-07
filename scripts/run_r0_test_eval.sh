#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs
LOG_FILE="outputs/logs/r0_test_eval_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "FAILED at line ${LINENO}. See ${LOG_FILE}"' ERR

PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json}"
BASE_SPLIT="${BASE_SPLIT:-configs/d1_proxy_pilot_300_c5_n8192_split.json}"
GATE_SPLIT_PATTERN="${GATE_SPLIT_PATTERN:-outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed}.json}"
GATE_GROUPS="${GATE_GROUPS:-scratch no_boundary_field}"
TRAIN_SIZES="${TRAIN_SIZES:-50 75 100 125}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42 43 44 45 46}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
EPOCHS="${EPOCHS:-20}"
POINT_BUDGET="${POINT_BUDGET:-4096}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_PREFIX="${RUN_PREFIX:-d1_r0_v2}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/logs/r0_replication_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-docs/r0_replication_results.md}"

for path in "$D1_MANIFEST" "$BASE_SPLIT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

split_path_for() {
  local split_seed="$1"
  local path="$GATE_SPLIT_PATTERN"
  path="${path//"{split_seed}"/$split_seed}"
  path="${path//"{split_seed.json}"/"${split_seed}.json"}"
  if [[ "$path" == *"{split_seed"* ]]; then
    path="outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed${split_seed}.json"
  fi
  if [[ "$path" == *"{"* || "$path" == *"}"* ]]; then
    echo "Unresolved placeholder in GATE_SPLIT_PATTERN result: $path" >&2
    return 1
  fi
  echo "$path"
}

missing=0
for SPLIT_SEED in $SPLIT_SEEDS; do
  GATE_SPLIT="$(split_path_for "$SPLIT_SEED")"
  if [[ ! -e "$GATE_SPLIT" ]]; then
    mkdir -p "$(dirname "$GATE_SPLIT")"
    echo "=== build split seed ${SPLIT_SEED}: ${GATE_SPLIT} ==="
    "$PY" scripts/build_label_scarcity_splits.py \
      --base-split "$BASE_SPLIT" \
      --output "$GATE_SPLIT" \
      --train-sizes $TRAIN_SIZES \
      --seed "$SPLIT_SEED"
  fi
  for TRAIN_SEED in $TRAIN_SEEDS; do
    for N in $TRAIN_SIZES; do
      for GROUP in $GATE_GROUPS; do
        run_dir="outputs/checkpoints/${RUN_PREFIX}_${GROUP}_split${SPLIT_SEED}_trainseed${TRAIN_SEED}_train${N}_ep${EPOCHS}"
        if [[ ! -e "$run_dir/config.json" || ! -e "$run_dir/best_model.pt" ]]; then
          echo "Missing checkpoint for eval: $run_dir"
          missing=1
        fi
      done
    done
  done
done
if [[ "$missing" -ne 0 ]]; then
  echo "Run scripts/run_r0_replication_matrix.sh first, or check EPOCHS/TRAIN_SIZES/GATE_GROUPS/SPLIT_SEEDS/TRAIN_SEEDS."
  exit 1
fi

echo "Log: $LOG_FILE"
echo "GATE_GROUPS=$GATE_GROUPS"
echo "TRAIN_SIZES=$TRAIN_SIZES SPLIT_SEEDS=$SPLIT_SEEDS TRAIN_SEEDS=$TRAIN_SEEDS EPOCHS=$EPOCHS POINT_BUDGET=$POINT_BUDGET"

for SPLIT_SEED in $SPLIT_SEEDS; do
  GATE_SPLIT="$(split_path_for "$SPLIT_SEED")"
  for TRAIN_SEED in $TRAIN_SEEDS; do
    for N in $TRAIN_SIZES; do
      for GROUP in $GATE_GROUPS; do
        output_json="outputs/logs/${RUN_PREFIX}_${GROUP}_split${SPLIT_SEED}_trainseed${TRAIN_SEED}_train${N}_test_eval.json"
        if [[ "$SKIP_EXISTING" == "1" && -e "$output_json" ]]; then
          echo "=== skip existing eval ${GROUP} split=${SPLIT_SEED} train_seed=${TRAIN_SEED} train_${N} ==="
          continue
        fi
        echo "=== eval ${GROUP} split=${SPLIT_SEED} train_seed=${TRAIN_SEED} train_${N} ==="
        "$PY" scripts/evaluate_d1.py \
          --case-manifest "$D1_MANIFEST" \
          --split-path "$GATE_SPLIT" \
          --split test \
          --model-dir "outputs/checkpoints/${RUN_PREFIX}_${GROUP}_split${SPLIT_SEED}_trainseed${TRAIN_SEED}_train${N}_ep${EPOCHS}" \
          --checkpoint-file best_model.pt \
          --point-budget "$POINT_BUDGET" \
          --device "$DEVICE" \
          --output-json "$output_json"
      done
    done
  done
done

"$PY" scripts/summarize_r0_replication_results.py \
  --train-sizes $TRAIN_SIZES \
  --groups $GATE_GROUPS \
  --split-seeds $SPLIT_SEEDS \
  --train-seeds $TRAIN_SEEDS \
  --eval-pattern "outputs/logs/${RUN_PREFIX}_{group}_split{split_seed}_trainseed{train_seed}_train{train_size}_test_eval.json" \
  --run-pattern "outputs/checkpoints/${RUN_PREFIX}_{group}_split{split_seed}_trainseed{train_seed}_train{train_size}_ep${EPOCHS}" \
  --expected-case-count 150 \
  --expected-point-budget "$POINT_BUDGET" \
  --output-json "$SUMMARY_JSON" \
  --output-md "$SUMMARY_MD"

echo "Completed R0 test evaluation and summary. Log: $LOG_FILE"
