"""Paths and constants. Override paths with env vars so the same code runs locally and under Slurm."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # src/pertpred/utils/config.py -> repo

# Raw GEO files (read-only in the remote workspace).
DATA_DIR = Path(os.environ.get("PERTPRED_DATA", REPO_ROOT / "data"))
OUTPUTS_DIR = REPO_ROOT / "outputs"
# Large derived files (decompressed gctx, matrices, predictions, checkpoints; ~26 GB). Git-ignored.
# NOT part of the submission (checkpoints must not be submitted): move or delete it before `submit`,
# or point PERTPRED_CACHE outside the workspace (e.g. /scratch on the cluster).
CACHE_DIR = Path(os.environ.get("PERTPRED_CACHE", OUTPUTS_DIR / "processed_data"))
# Small, reviewable outputs (metrics, tables, plots, logs) that ARE part of the submission.
RESULTS_DIR = Path(os.environ.get("PERTPRED_RESULTS", OUTPUTS_DIR / "results"))

PREFIX = "GSE70138_Broad_LINCS_"
LEVEL5_GZ = DATA_DIR / f"{PREFIX}Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz"
LEVEL3_GZ = DATA_DIR / f"{PREFIX}Level3_INF_mlr12k_n345976x12328_2017-03-06.gctx.gz"
INST_INFO = DATA_DIR / f"{PREFIX}inst_info_2017-03-06.txt.gz"
SIG_INFO = DATA_DIR / f"{PREFIX}sig_info_2017-03-06.txt.gz"
PERT_INFO = DATA_DIR / f"{PREFIX}pert_info_2017-03-06.txt.gz"
GENE_INFO = DATA_DIR / f"{PREFIX}gene_info_2017-03-06.txt.gz"
CELL_INFO = DATA_DIR / f"{PREFIX}cell_info_2017-04-28.txt.gz"

LEVEL5_GCTX = CACHE_DIR / "level5.gctx"
LEVEL3_GCTX = CACHE_DIR / "level3.gctx"  # only needed by the data-tour notebook (17 GB decompressed)
Y_PATH = CACHE_DIR / "landmark_z.npy"  # (n_sigs, 978) float32, rows aligned with SIGS_PATH
SIGS_PATH = CACHE_DIR / "sigs.parquet"
GENES_PATH = CACHE_DIR / "landmark_genes.parquet"
SPLIT_PATH = CACHE_DIR / "split.parquet"  # one row per trt_cp signature, with split + features
FP_PATH = CACHE_DIR / "fingerprints.npz"
PRED_DIR = CACHE_DIR / "predictions"
CKPT_DIR = CACHE_DIR / "checkpoints"

QWEN_MODEL = "Qwen/Qwen2.5-0.5B"
QWEN_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

# Level 5 MODZ values have long tails (|z| > 20 occurs). Clip so a handful of extreme genes do not
# dominate MSE training or Pearson evaluation. Applied identically to every model and baseline.
Z_CLIP = 10.0

# Phase II's nominal dose ladder (µM). Off-ladder doses are snapped to the nearest rung in log space
# for context grouping only; the text prompt keeps the recorded dose.
DOSE_LADDER = (0.04, 0.12, 0.37, 1.11, 3.33, 10.0, 20.0)

SEED = 17
