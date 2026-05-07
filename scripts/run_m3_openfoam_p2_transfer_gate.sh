#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-train}"
PY="${PY:-../../.venv/bin/python}"
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_openfoam_block_m3_300/manifest.json}"
BASE_SPLIT="${BASE_SPLIT:-configs/d1_openfoam_block_m3_300_split_seed42.json}"
if [[ -z "${GATE_SPLIT_PATTERN:-}" ]]; then
  GATE_SPLIT_PATTERN="outputs/logs/m3_splits/d1_openfoam_block_m3_300_label_scarcity_seed{split_seed}.json"
fi
PRETRAIN_DYNAMICS="${PRETRAIN_DYNAMICS:-outputs/checkpoints/pretrain_r1_d1_thermal_dynamics_p2_ep20}"
GATE_GROUPS="${GATE_GROUPS:-scratch dynamics_lifted}"
TRAIN_SIZES="${TRAIN_SIZES:-10 25 50 100 200}"
SPLIT_SEEDS="${SPLIT_SEEDS:-42 43 44}"
TRAIN_SEEDS="${TRAIN_SEEDS:-42}"
EPOCHS="${EPOCHS:-50}"
POINT_BUDGET="${POINT_BUDGET:-3072}"
EVAL_POINT_BUDGET="${EVAL_POINT_BUDGET:-3072}"
EXPECTED_CASE_COUNT="${EXPECTED_CASE_COUNT:-45}"
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
RUN_PREFIX="${RUN_PREFIX:-m3_openfoam_p2}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/logs/m3_openfoam_p2_transfer_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-docs/m3_openfoam_p2_transfer_results.md}"
SUMMARY_TITLE="${SUMMARY_TITLE:-M3 OpenFOAM P2 Transfer Gate Results}"
SUMMARY_DESCRIPTION="${SUMMARY_DESCRIPTION:-This report summarizes solver-backed D1 OpenFOAM label-scarcity transfer runs for the P2 dynamics-lifted Thermal GeoPT checkpoint.}"
SUMMARY_INTERPRETATION_RULE="${SUMMARY_INTERPRETATION_RULE:-A positive M3 signal means dynamics_lifted improves scratch by about 10% relative L2 at 25 or 50 labels, or reaches similar error with fewer labels, without degrading max-temperature error. If the signal is weak or negative, do not proceed to P3; review the pretext and prompt design first.}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

for path in "$D1_MANIFEST" "$BASE_SPLIT" "$PRETRAIN_DYNAMICS/best_model.pt"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    echo "Generate D1 labels with: OVERWRITE=1 bash scripts/run_m3_openfoam_d1_300.sh" >&2
    exit 1
  fi
done

run_train() {
  PY="$PY" \
  D1_MANIFEST="$D1_MANIFEST" \
  BASE_SPLIT="$BASE_SPLIT" \
  GATE_SPLIT_PATTERN="$GATE_SPLIT_PATTERN" \
  PRETRAIN_DYNAMICS="$PRETRAIN_DYNAMICS" \
  GATE_GROUPS="$GATE_GROUPS" \
  TRAIN_SIZES="$TRAIN_SIZES" \
  SPLIT_SEEDS="$SPLIT_SEEDS" \
  TRAIN_SEEDS="$TRAIN_SEEDS" \
  EPOCHS="$EPOCHS" \
  POINT_BUDGET="$POINT_BUDGET" \
  EVAL_POINT_BUDGET="$EVAL_POINT_BUDGET" \
  SCRATCH_LR="$SCRATCH_LR" \
  PRETRAINED_BACKBONE_LR="$PRETRAINED_BACKBONE_LR" \
  PRETRAINED_HEAD_LR="$PRETRAINED_HEAD_LR" \
  FREEZE_PRETRAINED_BACKBONE_EPOCHS="$FREEZE_PRETRAINED_BACKBONE_EPOCHS" \
  FINETUNE_SCHEDULER="$FINETUNE_SCHEDULER" \
  FINETUNE_WARMUP_RATIO="$FINETUNE_WARMUP_RATIO" \
  FINETUNE_MIN_LR_SCALE="$FINETUNE_MIN_LR_SCALE" \
  FINETUNE_PCT_START="$FINETUNE_PCT_START" \
  MAX_GRAD_NORM="$MAX_GRAD_NORM" \
  NORMALIZATION_PROTOCOL="$NORMALIZATION_PROTOCOL" \
  NORMALIZATION_CONFIG="$NORMALIZATION_CONFIG" \
  RUN_PREFIX="$RUN_PREFIX" \
  DEVICE="$DEVICE" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  bash scripts/run_r0_replication_matrix.sh
}

run_eval() {
  PY="$PY" \
  D1_MANIFEST="$D1_MANIFEST" \
  BASE_SPLIT="$BASE_SPLIT" \
  GATE_SPLIT_PATTERN="$GATE_SPLIT_PATTERN" \
  GATE_GROUPS="$GATE_GROUPS" \
  TRAIN_SIZES="$TRAIN_SIZES" \
  SPLIT_SEEDS="$SPLIT_SEEDS" \
  TRAIN_SEEDS="$TRAIN_SEEDS" \
  EPOCHS="$EPOCHS" \
  POINT_BUDGET="$EVAL_POINT_BUDGET" \
  RUN_PREFIX="$RUN_PREFIX" \
  SUMMARY_JSON="$SUMMARY_JSON" \
  SUMMARY_MD="$SUMMARY_MD" \
  SUMMARY_TITLE="$SUMMARY_TITLE" \
  SUMMARY_DESCRIPTION="$SUMMARY_DESCRIPTION" \
  SUMMARY_INTERPRETATION_RULE="$SUMMARY_INTERPRETATION_RULE" \
  EXPECTED_CASE_COUNT="$EXPECTED_CASE_COUNT" \
  DEVICE="$DEVICE" \
  SKIP_EXISTING="$SKIP_EXISTING" \
  bash scripts/run_r0_test_eval.sh
}

case "$MODE" in
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  all)
    run_train
    run_eval
    ;;
  *)
    echo "Unknown MODE=$MODE; expected train, eval, or all." >&2
    exit 1
    ;;
esac
