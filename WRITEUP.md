# Perturbation-response prediction on L1000 with Qwen2.5-0.5B

_Status: draft — results sections are filled in from `results/report/` once the final runs finish._

## 1. Framing

**Question.** Given a cell line, a time point, a dose and a compound the model has never seen, predict
the Level 5 consensus signature over the 978 measured landmark genes.

**Why this version of the question.** It is the version that matters for using such a model: you
want to know what a new molecule will do. It is also the version where shortcuts are easiest to
mistake for skill, because cell line, dose and time explain a large share of what a signature looks
like regardless of which compound was applied. Everything in the evaluation is built to separate
"knows the context" from "knows the compound".

**Unit of evidence.** A compound, not a signature. A typical compound has 42 signatures (7 cell
lines x 6 doses), which are far from independent; all metrics are averaged within compound first and
confidence intervals come from a bootstrap over compounds.

## 2. Data and the pitfalls that shaped the evaluation

Source: GSE70138 Level 5 (118,050 signatures). After restricting to compound treatments and removing
plate controls: 102,797 signatures, 1,794 compounds, mostly 24 h, ~85% from 7 core cell lines.
Split by compound group: 1,316 train / 176 val / 264 test groups (76k / 11k / 15k signatures).
Test compounds are chemically far from training (median max-Tanimoto to train 0.36).

Pitfalls found, in the order they changed the plan:

1. **Plate positive controls.** The entire 20 µM dose level is bortezomib and MG-132, run on every
   REP plate (4,607 signatures, ~4x the norm of a typical signature). Left in, they were 6% of
   training rows, a large share of squared-error mass, and made the context mean look like it
   explained 29% of variance; with them removed it explains 3%. Removed (DECISIONS D4).

2. **Most signatures are noise.** Comparing treated signatures with the DMSO vehicle signatures in
   the same release, the median norm is nearly the same (32.1 vs 30.4) and only **16%** of treated
   signatures exceed the 95th percentile of vehicle (`results/eda/signal_vs_vehicle.png`). I report
   every metric on all signatures and on this "active" subset.

3. **The ceiling is low.** 4,456 conditions were measured twice on different plates. One measurement
   predicts the other with median Pearson **0.19** — 0.63 when both are active, 0.17 otherwise
   (`results/eda/replicate_ceiling.png`). No model should be expected to beat this on average.

4. **Raw Pearson measures the context.** The cell x time x dose training mean, which knows nothing
   about the compound, already scores Pearson 0.20 on held-out compounds — at the replicate ceiling.
   A model can look state-of-the-art on this metric without using the compound at all.

5. **The obvious fix is also broken.** Correlating residuals from the context mean (`delta_pearson`)
   gives an all-zeros predictor 0.069 — better than Tanimoto kNN and ridge on fingerprints. The
   context mean is pulled by a minority of strongly active training compounds; the typical held-out
   compound is inert, so (observed - mean) points along -mean and "predict nothing" correlates with it.

## 3. Metrics

| metric | what it answers | compound-agnostic model scores |
|---|---|---|
| `centered_pearson` (headline) | Within one cell x time x dose context, does the prediction say how *this* compound differs from the others? (centre predictions and observations per context, then correlate) | exactly 0 |
| `retrieval` (headline) | Given the observed signature, is this compound's prediction closer to it than other compounds' predictions in the same context? (0 = first, 0.5 = chance) | exactly 0.5 |
| `pearson` | conventional per-signature correlation | 0.20 (context mean) |
| `topk_dir` | overlap of predicted vs observed top-50 up and top-50 down genes (a CMap-style query) | 0.155 (context mean) |
| `rmse` | magnitude calibration | 1.124 (context mean) |

Both headline metrics are exactly null for any model that ignores the compound — this is checked
in the results table, where `zero` and `context_mean` score 0.000 and 0.500. Model selection
(kNN k, ridge alpha, MLP epoch, LLM checkpoint, probe pooling) uses val `centered_pearson` only;
test is used once per model for reporting.

## 4. Approach

_(filled in with final results)_

## 5. Results

_(filled in with final results)_

## 6. Failure modes

_(filled in with final results)_

## 7. What I tried and what changed

_(filled in with final results)_

## 8. What I would do next

_(filled in with final results)_
