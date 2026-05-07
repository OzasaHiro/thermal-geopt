#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CASE_COUNT="${CASE_COUNT:-300}" \
RAW_DIR="${RAW_DIR:-data/downstream_raw/d1_openfoam_block_m3_300}" \
NPZ_DIR="${NPZ_DIR:-data/downstream_npz/d1_openfoam_block_m3_300}" \
SPLIT_PATH="${SPLIT_PATH:-configs/d1_openfoam_block_m3_300_split_seed42.json}" \
BASELINE_JSON="${BASELINE_JSON:-outputs/logs/d1_openfoam_block_m3_300_baseline_test.json}" \
CELLS_X="${CELLS_X:-16}" \
CELLS_Y="${CELLS_Y:-16}" \
CELLS_Z="${CELLS_Z:-12}" \
POINT_BUDGET="${POINT_BUDGET:-3072}" \
OVERWRITE="${OVERWRITE:-0}" \
bash scripts/run_m1_openfoam_pilot.sh
