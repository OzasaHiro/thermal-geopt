#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_openfoam_block_pilot_50/manifest.json}"
SPLIT_PATH="${SPLIT_PATH:-configs/d1_openfoam_block_pilot_50_split_seed42.json}"
RUN_PREFIX="${RUN_PREFIX:-d1_openfoam_block}"
EPOCHS="${EPOCHS:-20}"
POINT_BUDGET="${POINT_BUDGET:-1024}"
EVAL_POINT_BUDGET="${EVAL_POINT_BUDGET:-1024}"
MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-0}"
MAX_VAL_CASES="${MAX_VAL_CASES:-0}"
N_HIDDEN="${N_HIDDEN:-256}"
N_LAYERS="${N_LAYERS:-8}"
N_HEADS="${N_HEADS:-8}"
SLICE_NUM="${SLICE_NUM:-32}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
INIT_GROUPS="${INIT_GROUPS:-scratch full static_tdf_only no_boundary_field}"
PRETRAIN_FULL="${PRETRAIN_FULL:-outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2}"
PRETRAIN_STATIC="${PRETRAIN_STATIC:-outputs/checkpoints/pretrain_gate_static_tdf_only_ep2}"
PRETRAIN_NO_BOUNDARY="${PRETRAIN_NO_BOUNDARY:-outputs/checkpoints/pretrain_gate_no_boundary_field_ep2}"
PRETRAIN_DYNAMICS="${PRETRAIN_DYNAMICS:-outputs/checkpoints/pretrain_r1_dynamics_lifted_no_boundary_ep2}"

for path in "$D1_MANIFEST" "$SPLIT_PATH"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done

mkdir -p outputs/logs

pretrain_dir_for_group() {
  case "$1" in
    scratch) echo "" ;;
    full) echo "$PRETRAIN_FULL" ;;
    static_tdf_only) echo "$PRETRAIN_STATIC" ;;
    no_boundary_field) echo "$PRETRAIN_NO_BOUNDARY" ;;
    dynamics_lifted) echo "$PRETRAIN_DYNAMICS" ;;
    *)
      echo "Unknown group: $1" >&2
      return 1
      ;;
  esac
}

for group in $INIT_GROUPS; do
  pretrain_dir="$(pretrain_dir_for_group "$group")"
  if [[ -n "$pretrain_dir" && ! -f "$pretrain_dir/best_model.pt" ]]; then
    echo "Skipping $group: missing $pretrain_dir/best_model.pt"
    continue
  fi

  output_dir="outputs/checkpoints/${RUN_PREFIX}_${group}_ep${EPOCHS}"
  eval_json="outputs/logs/${RUN_PREFIX}_${group}_ep${EPOCHS}_test.json"
  echo "=== $group ==="

  train_args=(
    scripts/train_finetune_d1.py
    --case-manifest "$D1_MANIFEST"
    --split-path "$SPLIT_PATH"
    --train-split train
    --val-split val
    --output-dir "$output_dir"
    --epochs "$EPOCHS"
    --batch-size 1
    --point-budget "$POINT_BUDGET"
    --eval-point-budget "$EVAL_POINT_BUDGET"
    --max-train-cases "$MAX_TRAIN_CASES"
    --max-val-cases "$MAX_VAL_CASES"
    --n-hidden "$N_HIDDEN"
    --n-layers "$N_LAYERS"
    --n-heads "$N_HEADS"
    --slice-num "$SLICE_NUM"
    --device "$DEVICE"
    --seed "$SEED"
  )
  if [[ -n "$pretrain_dir" ]]; then
    train_args+=(--pretrained-model-dir "$pretrain_dir" --pretrained-checkpoint-file best_model.pt)
  fi

  "$PY" "${train_args[@]}"

  "$PY" scripts/evaluate_d1.py \
    --case-manifest "$D1_MANIFEST" \
    --split-path "$SPLIT_PATH" \
    --split test \
    --model-dir "$output_dir" \
    --output-json "$eval_json" \
    --point-budget "$EVAL_POINT_BUDGET" \
    --device "$DEVICE"
done

echo "M1 OpenFOAM initialization comparison complete"
