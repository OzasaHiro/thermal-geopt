#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRETRAIN_GEOPT_TRANSPORT="${PRETRAIN_GEOPT_TRANSPORT:-outputs/checkpoints/pretrain_r4_geopt_transport_p2_norm_val_ep100_wcos}"

MODE="${MODE:-all}" \
PY="${PY:-../../.venv/bin/python}" \
D1_MANIFEST="${D1_MANIFEST:-data/downstream_npz/d1_openfoam_heatsink_m4_300/manifest.json}" \
BASE_SPLIT="${BASE_SPLIT:-configs/d1_openfoam_heatsink_m4_300_split_seed42.json}" \
GATE_SPLIT_PATTERN="${GATE_SPLIT_PATTERN:-outputs/logs/m4_splits/d1_openfoam_heatsink_m4_300_label_scarcity_seed{split_seed}.json}" \
PRETRAIN_GEOPT_TRANSPORT="$PRETRAIN_GEOPT_TRANSPORT" \
PRETRAIN_GEOPT_ORIGINAL="${PRETRAIN_GEOPT_ORIGINAL:-../GeoPT/checkpoints}" \
PRETRAIN_GEOPT_ORIGINAL_CHECKPOINT="${PRETRAIN_GEOPT_ORIGINAL_CHECKPOINT:-GeoPT_8layers.pt}" \
GATE_GROUPS="${GATE_GROUPS:-scratch geopt_transport_lifted geopt_original}" \
TRAIN_SIZES="${TRAIN_SIZES:-10 25 50 100 200}" \
SPLIT_SEEDS="${SPLIT_SEEDS:-42 43 44}" \
TRAIN_SEEDS="${TRAIN_SEEDS:-42}" \
EPOCHS="${EPOCHS:-100}" \
POINT_BUDGET="${POINT_BUDGET:-4096}" \
EVAL_POINT_BUDGET="${EVAL_POINT_BUDGET:-4096}" \
EXPECTED_CASE_COUNT="${EXPECTED_CASE_COUNT:-45}" \
SCRATCH_LR="${SCRATCH_LR:-1e-3}" \
PRETRAINED_BACKBONE_LR="${PRETRAINED_BACKBONE_LR:-3e-4}" \
PRETRAINED_HEAD_LR="${PRETRAINED_HEAD_LR:-1e-3}" \
FREEZE_PRETRAINED_BACKBONE_EPOCHS="${FREEZE_PRETRAINED_BACKBONE_EPOCHS:-0}" \
FINETUNE_SCHEDULER="${FINETUNE_SCHEDULER:-onecycle}" \
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}" \
NORMALIZATION_PROTOCOL="${NORMALIZATION_PROTOCOL:-pretrained}" \
NORMALIZATION_CONFIG="${NORMALIZATION_CONFIG:-$PRETRAIN_GEOPT_TRANSPORT/config.json}" \
CONDITION_AUGMENTATION="${CONDITION_AUGMENTATION:-thermal_transport}" \
GEOPT_ORIGINAL_NORMALIZATION_PROTOCOL="${GEOPT_ORIGINAL_NORMALIZATION_PROTOCOL:-pretrained}" \
GEOPT_ORIGINAL_NORMALIZATION_CONFIG="${GEOPT_ORIGINAL_NORMALIZATION_CONFIG:-$PRETRAIN_GEOPT_TRANSPORT/config.json}" \
RUN_PREFIX="${RUN_PREFIX:-m4_heatsink_r4_vs_original}" \
SUMMARY_JSON="${SUMMARY_JSON:-outputs/logs/m4_heatsink_transfer_summary.json}" \
SUMMARY_MD="${SUMMARY_MD:-docs/m4_heatsink_transfer_results.md}" \
SUMMARY_TITLE="${SUMMARY_TITLE:-M4 Heat-Sink Solver-Backed D1 Transfer Gate Results}" \
SUMMARY_DESCRIPTION="${SUMMARY_DESCRIPTION:-This report summarizes geometry-varied solver-backed heat-sink D1 label-scarcity transfer runs comparing scratch, Thermal GeoPT pretraining, and the original GeoPT checkpoint.}" \
SUMMARY_INTERPRETATION_RULE="${SUMMARY_INTERPRETATION_RULE:-A positive M4 signal means a pretrained group improves scratch by about 10% relative L2 at 25 or 50 labels, or reaches similar error with fewer labels, without degrading max-temperature or hotspot metrics. Original GeoPT is a control for general GeoPT geometry-boundary priors; Thermal GeoPT must beat or complement it to support a thermal-specific pretraining claim.}" \
DEVICE="${DEVICE:-cuda}" \
SKIP_EXISTING="${SKIP_EXISTING:-1}" \
bash scripts/run_m3_openfoam_p2_transfer_gate.sh
