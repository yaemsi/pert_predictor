# pert_predictor

Predicting LINCS L1000 Phase II transcriptional responses to **unseen compounds** with
Qwen2.5-0.5B, against strong non-LLM baselines.

- Writeup (framing, results, failure modes, next steps): [`WRITEUP.md`](WRITEUP.md)
- Decision record: [`DECISIONS.md`](DECISIONS.md)
- AI disclosure: [`AI_DISCLOSURE.md`](AI_DISCLOSURE.md)
- Numbers: [`results/report/`](results/report/) (tables, plots), [`results/eda/`](results/eda/) (data diagnostics),
  [`results/per_signature/`](results/per_signature/) (every model's per-signature metrics, re-aggregatable),
  [`results/logs/`](results/logs/) (training curves and run logs)

## Result in one paragraph

On 264 held-out compounds, Qwen2.5-0.5B with LoRA reading the condition as text reaches centered
Pearson 0.045 and retrieval rank 0.449 (0.5 = chance), statistically tied with an MLP on Morgan
fingerprints (0.054 / 0.442) and weaker on signatures with real signal. A context-only version of
the same model scores exactly the null, so the gain comes from the compound text. Details, failure
modes and next steps in [`WRITEUP.md`](WRITEUP.md); headline figure `results/report/models_test.png`.

## Task in one paragraph

Input: a cell line, a time point, a dose, and a compound (common name if it has one, SMILES).
Output: the Level 5 consensus z-score signature over the 978 measured landmark genes. Evaluation
is on compounds never seen in training (grouped so salts / stereo variants / re-registrations of
one molecule cannot straddle the split), and the headline metrics measure what a prediction says
about *this compound* beyond what the cell / time / dose context already implies.

## Environment

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). PyTorch comes from the CUDA 12.8 wheel
index (works on any driver >= 12.8, including CUDA 13.x drivers; required for Blackwell GPUs).

```bash
uv sync                      # creates .venv from uv.lock
```

Paths are environment variables so the same code runs locally and under Slurm:

| variable | default | holds |
|---|---|---|
| `PERTPRED_DATA` | `./data/raw` | the raw GEO files (read-only) |
| `PERTPRED_CACHE` | `./data/pre-processed` | decompressed gctx (Level 5: 5.8 GB; Level 3: 17 GB, notebook only), matrices, predictions, checkpoints |
| `PERTPRED_RESULTS` | `./results` | small outputs that belong in the submission |

The cache lives under `data/`, which `submit` does not package, so weights and matrices never reach the submission.
If `data/` is read-only on the cluster, point `PERTPRED_CACHE` at a writable path (e.g. `/scratch/slurm/...` or `$HOME`).

## Reproduce

```bash
uv run python -m pertpred.data        # decompress Level 5, extract 978 landmarks       (~2 min, CPU)
uv run python -m pertpred.split       # compound-grouped split, Morgan fingerprints     (~10 s)
uv run python -m pertpred.eda         # data diagnostics -> results/eda/                (~30 s)
uv run python -m pertpred.baselines   # zero, context mean, kNN, ridge, MLP             (~3 min, GPU optional)
uv run python -m pertpred.probe       # frozen Qwen embeddings -> same MLP              (~15 min, GPU)
bash scripts/llm_runs.sh queue        # main LoRA model + 3 ablations                   (~3 h, GPU)
uv run python -m pertpred.evaluate    # tables, paired tests, plots -> results/report/  (~2 min)
```

The exact LLM runs behind the reported numbers are listed, with their command lines, in
[`scripts/llm_runs.sh`](scripts/llm_runs.sh). Each run's config is also the first line of
`results/logs/<run>.jsonl`.

On the remote workspace, wrap any GPU step with the provided launcher, e.g.

```bash
gpu submit -- uv run python -m pertpred.llm --run-name qwen_full
```

## Model

Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`) reads the
condition as text:

```
cell line: A375 (skin)
time: 24 h
dose: 10 uM
compound: oxaprozin
smiles: OC(=O)CCc1nc(c(o1)-c1ccccc1)-c1ccccc1
```

A pooled hidden state feeds a linear head that predicts the 978-gene residual over the training mean
of the same cell x time x dose context. LoRA (r=16) on every linear layer of the backbone; the head
is zero-initialized so training starts exactly at the context-mean baseline. Weights are not part of
the submission.

## Data tour

[`notebooks/01_data_tour.ipynb`](notebooks/01_data_tour.ipynb) walks through every raw file, how
they link, and what the measurements look like, with plots (saved in the notebook). Re-run with
`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_tour.ipynb`; section 9
needs Level 3 decompressed into `data/pre-processed/level3.gctx` and is skipped otherwise.

## Code map

| module | role |
|---|---|
| `src/pertpred/config.py` | paths, constants, pinned model revision |
| `src/pertpred/data.py` | gctx -> (signature metadata, 978-gene z matrix) |
| `src/pertpred/split.py` | compound grouping, held-out split, fingerprints, novelty (max Tanimoto to train) |
| `src/pertpred/task.py` | rows, clipped targets, context-mean offset |
| `src/pertpred/metrics.py` | per-signature metrics, compound-level aggregation, bootstrap |
| `src/pertpred/evaluate.py` | DMSO activity threshold, scoring, report tables and plots |
| `src/pertpred/baselines.py` | non-LLM baselines |
| `src/pertpred/probe.py` | frozen-Qwen embeddings through the baseline MLP |
| `src/pertpred/llm.py` | Qwen LoRA regressor: serialization, training, inference |
| `src/pertpred/eda.py` | diagnostics that shaped the evaluation |
