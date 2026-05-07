#!/usr/bin/env bash

PY="${PY:-../../.venv/bin/python}"
OPENFOAM_BASH="${OPENFOAM_BASH:-/opt/openfoam13/etc/bashrc}"
CASE_COUNT="${CASE_COUNT:-50}"
CELLS_X="${CELLS_X:-16}"
CELLS_Y="${CELLS_Y:-16}"
CELLS_Z="${CELLS_Z:-12}"
SEED="${SEED:-42}"
RAW_DIR="${RAW_DIR:-data/downstream_raw/d1_openfoam_block_pilot_50}"
NPZ_DIR="${NPZ_DIR:-data/downstream_npz/d1_openfoam_block_pilot_50}"
SPLIT_PATH="${SPLIT_PATH:-configs/d1_openfoam_block_pilot_50_split_seed42.json}"
BASELINE_JSON="${BASELINE_JSON:-outputs/logs/d1_openfoam_block_pilot_50_baseline_test.json}"
POINT_BUDGET="${POINT_BUDGET:-1024}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -f "$OPENFOAM_BASH" ]]; then
  echo "OpenFOAM bashrc not found: $OPENFOAM_BASH" >&2
  exit 1
fi

if ! source "$OPENFOAM_BASH"; then
  echo "Failed to source OpenFOAM bashrc: $OPENFOAM_BASH" >&2
  exit 1
fi

set -euo pipefail

generate_args=(
  scripts/generate_d1_openfoam_cases.py
  --case-count "$CASE_COUNT"
  --cells "$CELLS_X" "$CELLS_Y" "$CELLS_Z"
  --raw-dir "$RAW_DIR"
  --output-dir "$NPZ_DIR"
  --seed "$SEED"
  --openfoam-bash "$OPENFOAM_BASH"
)
if [[ "$OVERWRITE" == "1" ]]; then
  generate_args+=(--overwrite)
fi

"$PY" "${generate_args[@]}"

"$PY" scripts/inspect_artifacts.py "$NPZ_DIR/manifest.json"

"$PY" scripts/build_splits.py \
  --input-manifest "$NPZ_DIR/manifest.json" \
  --output "$SPLIT_PATH" \
  --train-frac 0.7 \
  --val-frac 0.15 \
  --seed "$SEED"

"$PY" scripts/evaluate_d1.py \
  --case-manifest "$NPZ_DIR/manifest.json" \
  --split-path "$SPLIT_PATH" \
  --split test \
  --baseline mean_temperature \
  --output-json "$BASELINE_JSON" \
  --point-budget "$POINT_BUDGET"

echo "M1 OpenFOAM pilot complete"
echo "manifest: $NPZ_DIR/manifest.json"
echo "split: $SPLIT_PATH"
echo "baseline: $BASELINE_JSON"
