# Decision record

Each entry: what was decided, the alternatives, why, and what result would show it was wrong.
Dated notes at the bottom record how the decisions came about.

---

## D1. Predict Level 5 consensus z-scores on the 978 landmark genes

**Chosen.** Target = Level 5 MODZ signature restricted to the 978 directly measured landmarks,
clipped to |z| <= 10 (5e-5 of values are affected).

**Alternatives.** (a) All 12,328 genes. (b) Level 3 replicate-level profiles. (c) A discretized
target (top-k up / down gene sets) generated as text.

**Why.** The other 11,350 genes are *inferred* from the landmarks by a fixed linear model, so they
add no independent information and multiply cost and apparent sample size by 12. Level 5 is the
replicate-collapsed signature people actually query; Level 3 would force me to redo plate
normalization and handle replicate noise before asking any modeling question. A continuous target
keeps magnitude (which carries potency) and lets every baseline and the LLM be scored identically;
top-k gene sets are still evaluated by deriving them from the prediction (`topk_dir`).

**Would be wrong if** a model scored on inferred genes showed gains that cannot be reproduced from
its landmark predictions, or if a gene-set generator beat the regression model on `topk_dir`.

## D2. Hold out compounds, grouped by molecule, not signatures

**Chosen.** 75 / 10 / 15 % split of *compound groups*. Two `pert_id`s are one group if they share an
InChIKey skeleton (first block: connectivity without stereo / protonation) or a non-BRD common name
(39 merges). Every signature of a test compound, in every cell, dose and time, is unseen in training.

**Alternatives.** (a) Random signature split. (b) Held-out cell lines. (c) Held-out
(compound, cell) pairs. (d) Murcko-scaffold split.

**Why.** "Given a condition, predict the response" is only interesting for a perturbation the model
has not seen; a random signature split lets a model memorize a compound from its other doses and
cells and would reward exactly the shortcut we want to rule out. Cell hold-out is a different
question with 7 core cells (too few groups to estimate anything). A scaffold split is stricter but
the compound split is already far from training chemistry (median max-Tanimoto of test compounds to
train = 0.36), and I report performance by similarity bin instead of hiding the dependency.

**Would be wrong if** results are driven by close analogues (performance concentrated in the
high-similarity bin) — then a scaffold split is required to support any "new chemistry" claim.

**Outcome (partly triggered).** Every model, LLM included, rises from ~0.02 (max Tanimoto <= 0.3,
75 compounds) to 0.08–0.10 (0.4–0.5). Signal on the most novel compounds is above zero but small, so
the writeup does not claim generalization to new chemistry; a scaffold split is next-steps item 2.

## D3. Headline metrics: centered Pearson and within-context retrieval, aggregated per compound

**Chosen.** Two headline metrics that are exactly 0 / exactly chance for any compound-agnostic model:
- `centered_pearson`: correlate prediction and observation after subtracting, within each
  cell x time x dose context of the evaluated set, the mean prediction and the mean observation.
- `retrieval`: rank the correct compound among all evaluated compounds in the same context
  (0 = first, 0.5 = chance).
Per-signature values are averaged within compound, then across compounds; 95% CIs and model-vs-model
differences come from a paired bootstrap over compounds. Everything is also reported on the subset
of signatures whose norm exceeds the 95th percentile of DMSO vehicle signatures in the same cell and
time ("active").

**Alternatives.** (a) Raw per-signature Pearson (the field default). (b) `delta_pearson`, the
correlation of residuals from the training context mean. (c) MSE / RMSE.

**Why.** Raw Pearson is dominated by context: the context mean alone scores 0.20, which is already
at the empirical replicate ceiling (two measurements of the same condition correlate at median
r = 0.19). `delta_pearson` looked right and is subtly broken: predicting all zeros scores 0.069,
better than kNN and ridge, because the training context mean is pulled by a minority of strongly
active compounds while the typical held-out compound is inert, so (truth - mean) points along -mean.
RMSE barely moves between models because 84% of signatures are indistinguishable from vehicle.
Compound-level aggregation matters because a typical compound contributes 42 strongly dependent
signatures.

**Would be wrong if** the centering made models look good that are useless in practice — e.g. a
model with high `centered_pearson` but no retrieval ability, or one whose gain is confined to
inactive signatures. Both are reported side by side to catch exactly that.

**Outcome.** The null checks hold exactly (`zero`, `context_mean`, `qwen_ctx`). The failure mode it
was designed to catch did occur: the single-channel LLMs earn their `centered_pearson` almost
entirely on inactive signatures (active-only ~0). That is visible only because the active stratum
is reported next to the overall number — keep reporting both.

## D4. Drop the per-plate positive controls

**Chosen.** Remove bortezomib and MG-132 at 20 µM (4,607 signatures, all train) from training and
evaluation. Bortezomib's 6-dose series stays.

**Alternatives.** Keep them as ordinary compounds; or down-weight them.

**Why.** They sit on every REP plate as assay QC, so 2 molecules were 6% of training rows. Their
signature norm (~120) is ~4x a typical one (~32), so ~14x per-row squared-error mass — they would
have dominated the MSE objective and *defined* the 20 µM context. Before removal the context mean
appeared to explain 29% of training variance; after removal, 3%. That number was an artifact of two
control compounds.

**Would be wrong if** held-out proteasome-inhibitor-like compounds became systematically worse
predicted (none are in val/test, so this is untested).

## D5. Qwen as a conditional encoder with a regression head over the context mean (not a gene-list generator)

**Chosen.** Serialize the condition as plain text, run Qwen2.5-0.5B with LoRA (r=16, all linear
layers), pool a hidden state, linear head -> 978-gene *residual* over the training context mean.
Head zero-initialized: at step 0 the model **is** the context-mean baseline.

**Alternatives.** (a) Generate ranked up/down gene symbols as text (Cell2Sentence-style). (b) Frozen
Qwen embeddings + a small model (run as the `probe_*` experiment). (c) Full fine-tuning.

**Why.** Generation turns a 978-d continuous target into an ordering problem in which most of the
loss sits on near-tied, noisy genes, adds decoding and malformed-output failure modes unrelated to
biology, and still has to be converted back to a vector to compare against baselines. A regression
head uses every gene and every magnitude and makes the LLM directly comparable to the MLP baseline,
which sees the same information as Morgan bits. Predicting the residual with a zero-initialized
head makes "the LLM adds nothing" a measurable null instead of a confound. LoRA because ~1,300
training compounds is a small dataset for 500M parameters, and the weights are not submitted anyway.

**Would be wrong if** the frozen probe matches or beats LoRA (fine-tuning buys nothing), or if a
generative variant beats the head on `topk_dir`.

**Outcome (partly triggered).** LoRA vs frozen probe: `centered_pearson` +0.006 [-0.004, +0.016]
(no detectable gain), retrieval -0.020 [-0.040, 0.000] (borderline gain). Fine-tuning buys little on
this data. The generative alternative was not run, so the second condition is untested.

## D6. Choose the LLM's read-out with a cheap frozen probe before spending GPU on LoRA

**Chosen.** Mean pooling over prompt tokens at the final layer, picked on val from a frozen-Qwen
probe (4 pooling x layer combinations, each embedding fed to the *same* MLP as the fingerprint
baseline). Main LoRA run trained 6 epochs (the 2-epoch first attempt was still improving).

**Alternatives.** (a) Keep the first design (EOS token, final layer) and just train longer.
(b) Grid-search pooling with full LoRA runs (~1 GPU-hour each).

**Why.** The first LoRA run sat at zero for half an epoch and finished under the fingerprint MLP, so
the question "is the representation the problem or the training?" had to be answered before
committing GPU hours. The probe answers the representation half for ~10 minutes of compute, and it
doubles as a clean ablation: frozen Qwen vs Morgan bits under an identical downstream model.

**What it showed.** Every frozen-Qwen variant is *below* Morgan bits on test (`centered_pearson`
0.035–0.040 vs 0.054; paired differences -0.014 to -0.020, CIs exclude 0). Name-only and
SMILES-only embeddings are about equally (un)informative. Concatenating Qwen to Morgan bits gives
+0.010 [-0.002, +0.022] — not distinguishable from zero, and worse retrieval and RMSE. Pooling
choice on val did not transfer cleanly to test (winner's curse across 4 near-tied options).

**Would be wrong if** LoRA from the probe-selected pooling does no better than LoRA from EOS
pooling — then pooling was not the bottleneck and the probe told us little about fine-tuning.

**Outcome (mixed).** On all signatures `qwen_main` and `qwen_full` tie (+0.001 [-0.014, +0.015]);
on active signatures mean pooling moved the LoRA model from -0.020 to 0.027 and retrieval from
0.468 to 0.449. The probe's pooling choice helped where the first model was worst, but did not
change the headline number. Note `qwen_main` also trained 3x longer, so the two changes are
confounded.

---

## Dated notes

**2026-10-05**
- Verified all raw files against `SHA512SUMS.txt`. `pert_info` was listed in `data/README.md` and
  the checksums but missing locally; fetched it from GEO and confirmed the SHA-512 matches the
  supplied checksum. It is the only source of SMILES / InChIKey. No other external data used.
- Landmark extraction: 118,050 signatures x 978 genes; `trt_cp` = 107,404 signatures, 1,796 compounds.
  Mostly 24 h (95%); 7 core cell lines carry ~85% of signatures.
- Found the 20 µM block = bortezomib + MG-132 plate controls -> D4.
- Treated vs DMSO norms: median 32.1 vs 30.4; only 16% of treated signatures exceed the DMSO 95th
  percentile. Replicate-pair Pearson 0.63 when both replicates are active, 0.17 otherwise -> D3's
  "active" stratum.
- Baselines run; `zero` beats kNN and ridge on `delta_pearson` -> replaced it as headline (D3).
- First Qwen run (EOS pooling, final layer, 2 epochs): val `centered_pearson` 0 for half an epoch,
  then rising steadily to 0.043 at the end of the schedule — under-trained, below the MLP on
  Morgan bits (0.060 val). Rather than guess, ran a frozen-embedding probe to pick pooling / layer.
- Tool note: one probe run was killed by a 10-minute foreground timeout and its log was empty
  because stdout was block-buffered. Long jobs now run in the background with `PYTHONUNBUFFERED=1`.
- Probe: mean pooling at the final layer chosen on val; every frozen-Qwen variant is below Morgan
  bits in the same MLP (D6). Morgan||Qwen: +0.010 [-0.002, +0.022], not significant.
- `mlp_fp` seed spread (4 seeds): 0.049–0.053 test `centered_pearson` — smaller than the gaps
  being compared, so single-seed MLP comparisons are meaningful; LLM seeds not measured.
- `qwen_main` (mean pooling, 6 epochs): val peaked at epoch 3 (0.054) and declined after; test
  0.045, tied with `mlp_fp`, behind on active signatures.

**2026-10-06**
- Machine shutdown overnight cut `qwen_smiles` at step 5,850/7,140. Rather than spend 45 min
  retraining, added `--eval-only` to score its best-on-val checkpoint (step 1,782; val flat or
  declining for 4,000 steps after, and the main run's last 18% was flat too). Recorded in
  `scripts/llm_runs.sh`.
- Ablations: SMILES matters (+0.014 [+0.004, +0.026] over name-only); the name does not add
  measurably over SMILES (+0.006, n.s.); name-only still carries signal (0.031). `qwen_ctx` scores
  the null on every compound-specific metric: no metadata leakage.
- Hypothesis "the LLM knows *whether* a compound acts but not *what* it does" was half wrong. An
  activity AUROC check (now in `evaluate.py`) shows `qwen_main` ranks activity as well as Morgan
  bits (0.585 vs 0.592); the gap is in direction on active signatures.
- Corrected two writeup claims after checking them against the tables: val > test is not universal
  (val is easier: 18.5% vs 12.7% active), and 83% (not 87%) of training signatures are inactive.
- Retrained `qwen_smiles` to completion so every reported number comes from a finished run
  (interrupted logs kept as `results/logs/qwen_smiles_interrupted.*`). Test centered Pearson 0.0392
  vs 0.0389 from the interrupted checkpoint, retrieval 0.471 vs 0.466: the shortcut would not have
  changed any conclusion, and the pair doubles as a run-to-run noise estimate for the LLM. With the
  complete run, the name's contribution to retrieval moved from n.s. to borderline
  (-0.022 [-0.044, -0.000]); the writeup reports it as borderline.
- While building the data-tour notebook, found that the compound grouping (D2) misses salt /
  prodrug forms with different InChIKey skeletons: dexamethasone / -acetate (both train),
  estramustine / -phosphate (train / val), ixazomib / -citrate (val / test). None links train to
  test, so test scores are not leaked; val-based selection saw one near-duplicate of a test
  compound. Not worth ~5 GPU-hours of reruns; a name-stem rule should be added to the grouping.
- Cache moved to `data/pre-processed/` (the workspace's own slot for derived data; `submit` skips
  `data/`). Level 3 decompressed there too (17 GB) for the notebook's raw-vs-differential section.
