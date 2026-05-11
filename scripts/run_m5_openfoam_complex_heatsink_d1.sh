#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CASE_COUNT="${CASE_COUNT:-300}" \
DATASET_ID="${DATASET_ID:-d1_openfoam_complex_heatsink_m5_${CASE_COUNT}}" \
FAMILIES="${FAMILIES:-plate_fin pin_fin staggered_pin_fin}" \
CELLS_X="${CELLS_X:-28}" \
CELLS_Y="${CELLS_Y:-28}" \
BASE_CELLS_Z="${BASE_CELLS_Z:-4}" \
FEATURE_CELLS_Z="${FEATURE_CELLS_Z:-14}" \
bash scripts/run_m4_openfoam_heatsink_d1.sh
