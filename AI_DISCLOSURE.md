# AI disclosure

## What was used

Claude Code (Anthropic, Claude Opus 5.5) as a pair-programmer, running in this workspace with
shell access. It wrote the large majority of the code, ran the experiments, and drafted the
writeup and decision record; I directed the work, reviewed it, and own every decision in it.

## Where AI helped

- Environment setup (uv project, CUDA 12.8 torch for a Blackwell GPU), data loading from gctx/HDF5.
- The data pipeline, split, metrics, baselines, probe, and LoRA training code in `src/pertpred/`.
- Exploratory analysis that changed the plan: finding the bortezomib / MG-132 plate controls, the
  DMSO noise floor, the replicate ceiling, and the all-zeros predictor beating kNN on `delta_pearson`.
- First drafts of README, WRITEUP, DECISIONS.

## How it was verified

Checks built into the code (these run every time):
- Raw files verified against GEO's `SHA512SUMS.txt`; gctx matrix shape asserted against its id lists;
  metadata re-aligned to gctx column order and asserted equal.
- Split: asserted that no compound group appears in two splits; similarity-to-train reported.
- Metrics: the compound-agnostic models (`zero`, `context_mean`) score exactly 0 on
  `centered_pearson` and exactly 0.5 on `retrieval` — a built-in test that the headline metrics
  cannot be gamed by context information.
- LLM: zero-initialized head, so an untrained model reproduces the context-mean baseline exactly.
- Test split is only used for reporting; every hyperparameter (k, ridge alpha, MLP epoch, LoRA
  checkpoint, probe pooling) is chosen on val, and the choices are logged in `outputs/results/logs/`.

_To complete by me before submitting: what I re-derived or re-ran by hand, which parts of the
code I read line by line, and anything I changed after review._

## What was rejected or changed

- `delta_pearson` as the headline metric (AI's first proposal) — rejected after the all-zeros
  baseline beat structure-based models on it; replaced by within-context centering (DECISIONS D3).
- First LLM batching ran out of GPU memory; replaced with length-grouped batches and token-budget
  micro-batching with gradient accumulation (same effective batch).
- Raw Pearson as the headline — rejected once the context mean alone reached the replicate ceiling.
- The AI's interpretation "the LLM knows whether a compound acts but not what it does" — tested with
  an activity AUROC before writing it up; it was half wrong and the writeup reports the corrected
  version (direction gap, not activity gap).
- Two writeup claims drafted from memory ("every Qwen variant scores higher on val", "87% inactive")
  were checked against the result tables, found wrong, and corrected.
- After a machine shutdown interrupted one ablation, the AI first scored its saved best-on-val
  checkpoint (`--eval-only`) instead of retraining. I asked for the run to be completed instead; the
  complete run matched the shortcut (0.0392 vs 0.0389) and is what the writeup reports.
- The first success criterion for the generative variant (beat the regression model on top-25
  overlap) — rejected before the result existed, once rescoring showed top-k overlap is
  context-dominated; replaced by "beat the context mean on top-25 overlap, or the regression model on
  compound-specific metrics". The generative scoring path was validated with oracle inputs before its
  (negative) result was written up.

## External data

None beyond the supplied GEO release. `pert_info` was missing locally and was downloaded from GEO's
own GSE70138 page; its SHA-512 matches the supplied checksum file. No public target values,
annotations, or hosted predictors were looked up for any compound.
