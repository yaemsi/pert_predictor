#!/usr/bin/env bash
# Exact LLM runs behind the reported numbers, in the order they were run.
#   local:   bash scripts/llm_runs.sh <run>        (or "queue" for main + ablations back to back)
#   cluster: gpu submit -- bash scripts/llm_runs.sh <run>
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
mkdir -p results/logs

llm() { uv run python main.py llm --run-name "$1" "${@:2}" > "results/logs/$1.log" 2>&1; }

# Main config, selected from the frozen probe on val: mean pooling over prompt tokens, final layer.
MAIN=(--pool mean --epochs 6)

case "${1:?run name}" in
  # 1st attempt: EOS pooling, final layer, 2 epochs. Under-trained (val still rising at the end).
  qwen_full)   llm qwen_full --pool eos --epochs 2 ;;
  # Frozen-representation probe (no weight updates) + Morgan||Qwen concat.
  probe)       uv run python main.py probe > results/logs/probe.log 2>&1 ;;
  # Main model.
  qwen_main)   llm qwen_main "${MAIN[@]}" ;;
  # Input-channel ablations at the main config.
  qwen_smiles) llm qwen_smiles "${MAIN[@]}" --fields cell,time,dose,smiles ;;
  # History: the first qwen_smiles run was cut at step 5850/7140 by a machine shutdown and its
  # best-on-val checkpoint was scored with the entry below (logs: results/logs/qwen_smiles_interrupted.*).
  # The reported numbers come from a later complete rerun of `qwen_smiles` above.
  qwen_smiles_eval) uv run python main.py llm --run-name qwen_smiles "${MAIN[@]}" --fields cell,time,dose,smiles \
                      --eval-only >> results/logs/qwen_smiles.log 2>&1 ;;
  qwen_name)   llm qwen_name "${MAIN[@]}" --fields cell,time,dose,name ;;
  # Leakage check: no compound information at all. Must score ~0 centered / ~0.5 retrieval.
  # Same 6-epoch config as the other ablations (a first 2-epoch version is kept as
  # results/logs/qwen_ctx_2ep.*: a longer run has more chances to exploit any leak).
  qwen_ctx)    llm qwen_ctx "${MAIN[@]}" --fields cell,time,dose ;;
  # Full fine-tuning: all ~494M weights trainable instead of LoRA adapters; 10x lower LR than LoRA.
  # Completes the adaptation ladder frozen probe -> LoRA -> all weights.
  qwen_allweights) llm qwen_allweights "${MAIN[@]}" --mode full --lr 2e-5 ;;
  queue)       for r in qwen_main qwen_smiles qwen_name qwen_ctx; do bash "$0" "$r"; done ;;
  resume)      for r in qwen_smiles_eval qwen_name qwen_ctx; do bash "$0" "$r"; done ;;
  queue2)      for r in qwen_ctx qwen_allweights; do bash "$0" "$r"; done ;;
  # Cell2Sentence-style generative variant (DECISIONS D8): Qwen + LoRA writes the top-25 up / down
  # genes as text, next-token loss, greedy decoding, tolerant parser. 2 epochs, val token loss.
  qwen_gen)    uv run python main.py gen --run-name qwen_gen --epochs 2 > results/logs/qwen_gen.log 2>&1 ;;
  *) echo "unknown run: $1" >&2; exit 1 ;;
esac
