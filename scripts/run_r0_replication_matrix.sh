#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p outputs/logs
LOG_FILE="outputs/logs/r0_replication_matrix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
trap 'echo "FAILED at line ${LINENO}. See ${LOG_FILE}"' ERR

PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_proxy_pilot_300_c5_n8192/manifest.json}"
BASE_SPLIT="${BASE_SPLIT:-configs/d1_proxy_pilot_300_c5_n8192_split.json}"
if [[ -z "${GATE_SPLIT_PATTERN:-}" ]]; then
  GATE_SPLIT_PATTERN="outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed{split_seed}.json"
fi
PRETRAIN_FULL="${PRETRAIN_FULL:-outputs/checkpoints/pretrain_pilot_300_e20_n4096_ep2}"
PRETRAIN_STATIC="${PRETRAIN_STATIC:-outputs/checkpoints/pretrain_gate_static_tdf_only_ep2}"
PRETRAIN_NO_BOUNDARY="${PRETRAIN_NO_BOUNDARY:-outputs/checkpoints/pretrain_gate_no_boundary_field_ep2}"
PRETRAIN_DYNAMICS="${PRETRAIN_DYNAMICS:-outputs/checkpoints/pretrain_r1_dynamics_lifted_no_boundary_ep2}"
PRETRAIN_DIFFUSION="${PRETRAIN_DIFFUSION:-$PRETRAIN_DYNAMICS}"
GATE_GROUPS="${GATE_GROUPS:-scratch no_boundary_field}"
TRAIN_SIZES="${TRAIN_SIZES:-50 75 100 125}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42 43 44 45 46}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
EPOCHS="${EPOCHS:-20}"
POINT_BUDGET="${POINT_BUDGET:-4096}"
EVAL_POINT_BUDGET="${EVAL_POINT_BUDGET:-4096}"
MAX_TRAIN_CASES="${MAX_TRAIN_CASES:-0}"
MAX_VAL_CASES="${MAX_VAL_CASES:-0}"
DEVICE="${DEVICE:-cuda}"
SCRATCH_LR="${SCRATCH_LR:-1e-3}"
PRETRAINED_BACKBONE_LR="${PRETRAINED_BACKBONE_LR:-}"
PRETRAINED_HEAD_LR="${PRETRAINED_HEAD_LR:-}"
FREEZE_PRETRAINED_BACKBONE_EPOCHS="${FREEZE_PRETRAINED_BACKBONE_EPOCHS:-0}"
FINETUNE_SCHEDULER="${FINETUNE_SCHEDULER:-none}"
FINETUNE_WARMUP_RATIO="${FINETUNE_WARMUP_RATIO:-0.05}"
FINETUNE_MIN_LR_SCALE="${FINETUNE_MIN_LR_SCALE:-0.01}"
FINETUNE_PCT_START="${FINETUNE_PCT_START:-0.3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-}"
NORMALIZATION_PROTOCOL="${NORMALIZATION_PROTOCOL:-legacy_downstream}"
NORMALIZATION_CONFIG="${NORMALIZATION_CONFIG:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
RUN_PREFIX="${RUN_PREFIX:-d1_r0_v2}"

for path in "$D1_MANIFEST" "$BASE_SPLIT"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path"
    exit 1
  fi
done

split_path_for() {
  local split_seed="$1"
  local path="$GATE_SPLIT_PATTERN"
  path="${path//\{split_seed.json\}/${split_seed}.json}"
  path="${path//\{split_seed\}/$split_seed}"
  if [[ "$path" == *"{split_seed"* ]]; then
    path="outputs/logs/r0_splits/d1_proxy_pilot_300_c5_n8192_label_scarcity_seed${split_seed}.json"
  fi
  if [[ "$path" == *"{"* || "$path" == *"}"* ]]; then
    echo "Unresolved placeholder in GATE_SPLIT_PATTERN result: $path" >&2
    return 1
  fi
  echo "$path"
}

split_has_train_sizes() {
  local split_path="$1"
  "$PY" - "$split_path" $TRAIN_SIZES <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
train_sizes = [int(value) for value in sys.argv[2:]]
if not path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
for train_size in train_sizes:
    rows = payload.get(f"train_{train_size}")
    if not isinstance(rows, list) or len(rows) != train_size:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

build_split() {
  local split_seed="$1"
  local split_path="$2"
  mkdir -p "$(dirname "$split_path")"
  echo "=== build split seed ${split_seed}: ${split_path} ==="
  "$PY" scripts/build_label_scarcity_splits.py \
    --base-split "$BASE_SPLIT" \
    --output "$split_path" \
    --train-sizes $TRAIN_SIZES \
    --seed "$split_seed"
}

pretrain_dir_for() {
  local group="$1"
  case "$group" in
    full) echo "$PRETRAIN_FULL" ;;
    static_tdf_only) echo "$PRETRAIN_STATIC" ;;
    no_boundary_field) echo "$PRETRAIN_NO_BOUNDARY" ;;
    dynamics_lifted) echo "$PRETRAIN_DYNAMICS" ;;
    dynamics_lifted_no_boundary) echo "$PRETRAIN_DYNAMICS" ;;
    diffusion_lifted) echo "$PRETRAIN_DIFFUSION" ;;
    scratch) echo "" ;;
    *)
      echo "Unknown group: $group" >&2
      return 1
      ;;
  esac
}

echo "Log: $LOG_FILE"
echo "GATE_GROUPS=$GATE_GROUPS"
echo "TRAIN_SIZES=$TRAIN_SIZES SPLIT_SEEDS=$SPLIT_SEEDS TRAIN_SEEDS=$TRAIN_SEEDS EPOCHS=$EPOCHS"
echo "POINT_BUDGET=$POINT_BUDGET EVAL_POINT_BUDGET=$EVAL_POINT_BUDGET MAX_TRAIN_CASES=$MAX_TRAIN_CASES MAX_VAL_CASES=$MAX_VAL_CASES"

for SPLIT_SEED in $SPLIT_SEEDS; do
  GATE_SPLIT="$(split_path_for "$SPLIT_SEED")"
  if ! split_has_train_sizes "$GATE_SPLIT"; then
    echo "Split missing or incomplete for seed ${SPLIT_SEED}; rebuilding: ${GATE_SPLIT}"
    build_split "$SPLIT_SEED" "$GATE_SPLIT"
  fi

  for TRAIN_SEED in $TRAIN_SEEDS; do
    for N in $TRAIN_SIZES; do
      for GROUP in $GATE_GROUPS; do
        run_dir="outputs/checkpoints/${RUN_PREFIX}_${GROUP}_split${SPLIT_SEED}_trainseed${TRAIN_SEED}_train${N}_ep${EPOCHS}"
        if [[ "$SKIP_EXISTING" == "1" && -e "$run_dir/best_model.pt" ]]; then
          echo "=== skip existing ${GROUP} split=${SPLIT_SEED} train_seed=${TRAIN_SEED} train_${N} ==="
          continue
        fi

        pretrain_dir="$(pretrain_dir_for "$GROUP")"
        pretrained_args=()
        lr_args=(
          --lr "$SCRATCH_LR"
          --scheduler "$FINETUNE_SCHEDULER"
          --warmup-ratio "$FINETUNE_WARMUP_RATIO"
          --min-lr-scale "$FINETUNE_MIN_LR_SCALE"
          --pct-start "$FINETUNE_PCT_START"
        )
        if [[ -n "$MAX_GRAD_NORM" ]]; then
          lr_args+=(--max-grad-norm "$MAX_GRAD_NORM")
        fi
        lr_args+=(--normalization-protocol "$NORMALIZATION_PROTOCOL")
        if [[ -n "$NORMALIZATION_CONFIG" ]]; then
          lr_args+=(--normalization-config "$NORMALIZATION_CONFIG")
        fi
        if [[ -n "$pretrain_dir" ]]; then
          if [[ ! -e "$pretrain_dir/best_model.pt" ]]; then
            echo "Missing pretrained checkpoint for group ${GROUP}: ${pretrain_dir}/best_model.pt"
            exit 1
          fi
          pretrained_args=(--pretrained-model-dir "$pretrain_dir" --pretrained-checkpoint-file best_model.pt)
          if [[ -n "$PRETRAINED_BACKBONE_LR" ]]; then
            lr_args+=(--pretrained-backbone-lr "$PRETRAINED_BACKBONE_LR")
          fi
          if [[ -n "$PRETRAINED_HEAD_LR" ]]; then
            lr_args+=(--pretrained-head-lr "$PRETRAINED_HEAD_LR")
          fi
          if [[ "$FREEZE_PRETRAINED_BACKBONE_EPOCHS" != "0" ]]; then
            lr_args+=(--freeze-pretrained-backbone-epochs "$FREEZE_PRETRAINED_BACKBONE_EPOCHS")
          fi
          if [[ "$NORMALIZATION_PROTOCOL" == "pretrained" && -z "$NORMALIZATION_CONFIG" ]]; then
            lr_args+=(--normalization-config "$pretrain_dir/config.json")
          fi
        fi

        echo "=== ${GROUP} split=${SPLIT_SEED} train_seed=${TRAIN_SEED} train_${N} ==="
        "$PY" scripts/train_finetune_d1.py \
          --case-manifest "$D1_MANIFEST" \
          --split-path "$GATE_SPLIT" \
          --train-split "train_${N}" \
          --val-split val \
          --output-dir "$run_dir" \
          "${pretrained_args[@]}" \
          --epochs "$EPOCHS" \
          --batch-size 1 \
          --point-budget "$POINT_BUDGET" \
          --eval-point-budget "$EVAL_POINT_BUDGET" \
          --max-train-cases "$MAX_TRAIN_CASES" \
          --max-val-cases "$MAX_VAL_CASES" \
          "${lr_args[@]}" \
          --amp \
          --amp-dtype bfloat16 \
          --device "$DEVICE" \
          --seed "$TRAIN_SEED"
      done
    done
  done
done

echo "Completed R0 replication matrix. Log: $LOG_FILE"
