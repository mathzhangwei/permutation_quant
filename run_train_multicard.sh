#!/usr/bin/env bash

set -euo pipefail

NPUS_PER_NODE="${NPUS_PER_NODE:-4}"
LAYER_START="${LAYER_START:-2}"
LAYER_END="${LAYER_END:-2}"
NUM_EXPERTS="${NUM_EXPERTS:-64}"
ACT_SAMPLES_PATH="${ACT_SAMPLES_PATH:-/docker/w00888862/llm_quant_test/llm_quant/ptq/smoothquant/act_scales/deepseek-v2-lite_acts.pt}"
WEIGHTS_PATH="${WEIGHTS_PATH:-/docker/models/DeepSeek-V2-Lite/merged_model.safetensors}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
EPOCHS="${EPOCHS:-300}"
LR="${LR:-3e-3}"
SOFTSORT_METRIC="${SOFTSORT_METRIC:-l1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${OUTPUT_ROOT}" ]]; then
  RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="PermuteDown_BatchScore_HardForward_SoftBackward_STE_MXFP4_v2/${RUN_TIMESTAMP}"
fi

CMD=(
  torchrun
  --nproc_per_node "${NPUS_PER_NODE}"
  train.py
  --layer-start "${LAYER_START}"
  --layer-end "${LAYER_END}"
  --num-experts "${NUM_EXPERTS}"
  --act-samples-path "${ACT_SAMPLES_PATH}"
  --weights-path "${WEIGHTS_PATH}"
  --epochs "${EPOCHS}"
  --lr "${LR}"
  --softsort-metric "${SOFTSORT_METRIC}"
  --output-root "${OUTPUT_ROOT}"
)

echo "Running multi-card training with:"
echo "  NPUS_PER_NODE=${NPUS_PER_NODE}"
echo "  LAYER_START=${LAYER_START}"
echo "  LAYER_END=${LAYER_END}"
echo "  NUM_EXPERTS=${NUM_EXPERTS}"
echo "  ACT_SAMPLES_PATH=${ACT_SAMPLES_PATH}"
echo "  WEIGHTS_PATH=${WEIGHTS_PATH}"
echo "  EPOCHS=${EPOCHS}"
echo "  LR=${LR}"
echo "  SOFTSORT_METRIC=${SOFTSORT_METRIC}"
echo "  OUTPUT_ROOT=${OUTPUT_ROOT}"

"${CMD[@]}"
