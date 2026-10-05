#!/usr/bin/env bash
# Exact LLM runs behind the reported numbers, in the order they were run.
#   local:   bash scripts/llm_runs.sh <run>        (or "queue" for main + ablations back to back)
#   cluster: gpu submit -- bash scripts/llm_runs.sh <run>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p results/logs

llm() { uv run python -m pertpred.llm --run-name "$1" "${@:2}" > "results/logs/$1.log" 2>&1; }

# Main config, selected from the frozen probe on val: mean pooling over prompt tokens, final layer.
MAIN=(--pool mean --epochs 6)

case "${1:?run name}" in
  # 1st attempt: EOS pooling, final layer, 2 epochs. Under-trained (val still rising at the end).
  qwen_full)   llm qwen_full --pool eos --epochs 2 ;;
  # Frozen-representation probe (no weight updates) + Morgan||Qwen concat.
  probe)       uv run python -m pertpred.probe > results/logs/probe.log 2>&1 ;;
  # Main model.
  qwen_main)   llm qwen_main "${MAIN[@]}" ;;
  # Input-channel ablations at the main config.
  qwen_smiles) llm qwen_smiles "${MAIN[@]}" --fields cell,time,dose,smiles ;;
  qwen_name)   llm qwen_name "${MAIN[@]}" --fields cell,time,dose,name ;;
  # Leakage check: no compound information at all. Must score ~0 centered / ~0.5 retrieval.
  qwen_ctx)    llm qwen_ctx --pool mean --epochs 2 --fields cell,time,dose ;;
  queue)       for r in qwen_main qwen_smiles qwen_name qwen_ctx; do bash "$0" "$r"; done ;;
  *) echo "unknown run: $1" >&2; exit 1 ;;
esac
