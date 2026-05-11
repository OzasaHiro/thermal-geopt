#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PY="${PY:-../../.venv/bin/python}"
CASE_COUNT="${CASE_COUNT:-300}"
DATASET_ID="${DATASET_ID:-d1_openfoam_heatsink_m4_${CASE_COUNT}}"
RAW_DIR="${RAW_DIR:-data/downstream_raw/${DATASET_ID}}"
NPZ_DIR="${NPZ_DIR:-data/downstream_npz/${DATASET_ID}}"
SPLIT_PATH="${SPLIT_PATH:-configs/${DATASET_ID}_split_seed42.json}"
BASELINE_JSON="${BASELINE_JSON:-outputs/logs/${DATASET_ID}_baseline_test.json}"
FAMILIES="${FAMILIES:-plate_fin pin_fin}"
CELLS_X="${CELLS_X:-24}"
CELLS_Y="${CELLS_Y:-24}"
BASE_CELLS_Z="${BASE_CELLS_Z:-4}"
FEATURE_CELLS_Z="${FEATURE_CELLS_Z:-12}"
POINT_BUDGET="${POINT_BUDGET:-4096}"
SEED="${SEED:-42}"
OVERWRITE="${OVERWRITE:-0}"
OPENFOAM_BASH="${OPENFOAM_BASH:-/opt/openfoam13/etc/bashrc}"
SOURCE_TEMPERATURE_MIN="${SOURCE_TEMPERATURE_MIN:-360}"
SOURCE_TEMPERATURE_MAX="${SOURCE_TEMPERATURE_MAX:-430}"
SINK_TEMPERATURE_MIN="${SINK_TEMPERATURE_MIN:-280}"
SINK_TEMPERATURE_MAX="${SINK_TEMPERATURE_MAX:-310}"
SINK_VALUE_FRACTION="${SINK_VALUE_FRACTION:-1.0}"

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

"$PY" scripts/generate_d1_openfoam_heatsink_cases.py \
  --case-count "$CASE_COUNT" \
  --families $FAMILIES \
  --cells-x "$CELLS_X" \
  --cells-y "$CELLS_Y" \
  --base-cells-z "$BASE_CELLS_Z" \
  --feature-cells-z "$FEATURE_CELLS_Z" \
  --source-temperature-min "$SOURCE_TEMPERATURE_MIN" \
  --source-temperature-max "$SOURCE_TEMPERATURE_MAX" \
  --sink-temperature-min "$SINK_TEMPERATURE_MIN" \
  --sink-temperature-max "$SINK_TEMPERATURE_MAX" \
  --sink-value-fraction "$SINK_VALUE_FRACTION" \
  --seed "$SEED" \
  --raw-dir "$RAW_DIR" \
  --output-dir "$NPZ_DIR" \
  --openfoam-bash "$OPENFOAM_BASH" \
  "${overwrite_args[@]}"

"$PY" scripts/build_splits.py \
  --input-manifest "$NPZ_DIR/manifest.json" \
  --output "$SPLIT_PATH" \
  --train-frac 0.7 \
  --val-frac 0.15 \
  --seed 42

"$PY" scripts/evaluate_d1.py \
  --case-manifest "$NPZ_DIR/manifest.json" \
  --split-path "$SPLIT_PATH" \
  --split test \
  --baseline mean_temperature \
  --output-json "$BASELINE_JSON" \
  --point-budget "$POINT_BUDGET"

echo "M4 OpenFOAM heat-sink D1 complete"
echo "Manifest: $NPZ_DIR/manifest.json"
echo "Split: $SPLIT_PATH"
echo "Baseline: $BASELINE_JSON"
