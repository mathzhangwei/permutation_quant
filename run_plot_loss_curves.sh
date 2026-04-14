#!/usr/bin/env bash

set -euo pipefail

INPUT_ROOT="${INPUT_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PLOT_EMA="${PLOT_EMA:-0}"
DPI="${DPI:-160}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${INPUT_ROOT}" ]]; then
  echo "Please provide INPUT_ROOT, for example:"
  echo "  INPUT_ROOT=PermuteDown_BatchScore_HardForward_SoftBackward_STE_MXFP4_v2/20260414_215530 bash run_plot_loss_curves.sh"
  exit 1
fi

CMD=(
  python
  plot_loss_curves.py
  --input-root "${INPUT_ROOT}"
  --dpi "${DPI}"
)

if [[ -n "${OUTPUT_DIR}" ]]; then
  CMD+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ "${PLOT_EMA}" == "1" ]]; then
  CMD+=(--plot-ema)
fi

echo "Generating loss plots with:"
echo "  INPUT_ROOT=${INPUT_ROOT}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  echo "  OUTPUT_DIR=${OUTPUT_DIR}"
else
  echo "  OUTPUT_DIR=<default: INPUT_ROOT/plots>"
fi
echo "  PLOT_EMA=${PLOT_EMA}"
echo "  DPI=${DPI}"

"${CMD[@]}"
