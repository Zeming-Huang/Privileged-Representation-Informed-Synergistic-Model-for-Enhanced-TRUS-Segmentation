#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?Usage: ./scripts/reproduce_demo.sh <data_root> <checkpoint> [output_root] [python_exe] [device]}"
CHECKPOINT="${2:?Usage: ./scripts/reproduce_demo.sh <data_root> <checkpoint> [output_root] [python_exe] [device]}"
OUTPUT_ROOT="${3:-./outputs/reproduce_demo}"
PYTHON_EXE="${4:-python}"
DEVICE="${5:-cuda:0}"

PRED_DIR="${OUTPUT_ROOT}/predictions"
CSV_PATH="${OUTPUT_ROOT}/evaluation_results.csv"

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_EXE}" inference.py \
  -data_root "${DATA_ROOT}" \
  -checkpoint "${CHECKPOINT}" \
  -pred_save_dir "${PRED_DIR}" \
  -device "${DEVICE}"

"${PYTHON_EXE}" evaluate.py \
  -pred_dir "${PRED_DIR}" \
  -gt_dir "${DATA_ROOT}" \
  -output_csv "${CSV_PATH}"

echo "Prediction directory: ${PRED_DIR}"
echo "Evaluation CSV: ${CSV_PATH}"
