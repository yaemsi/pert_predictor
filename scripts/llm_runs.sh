#!/usr/bin/env bash
# Exact LLM runs behind the reported numbers, in the order they were run.
# Locally: bash scripts/llm_runs.sh <name>   On the cluster: gpu submit -- bash scripts/llm_runs.sh <name>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
run() { uv run python -m pertpred.llm "$@" 2>&1 | tee "results/logs/$2.log"; }

case "${1:-all}" in
  # First attempt: EOS pooling on the final layer, 2 epochs. Under-trained (val still rising at the end).
  qwen_full)       run --run-name qwen_full ;;
  # Frozen-representation probe (no LoRA): picks pooling / layer on val.
  probe)           uv run python -m pertpred.probe 2>&1 | tee results/logs/probe.log ;;
  # Main model: probe-selected pooling/depth, trained to convergence on val.
  qwen_main)       run --run-name qwen_main  "${MAIN_ARGS[@]:---epochs 6}" ;;
  *) echo "unknown run: $1" >&2; exit 1 ;;
esac
