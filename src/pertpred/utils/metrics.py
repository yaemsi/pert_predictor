"""Evaluation metrics for predicted signatures.

All per-signature metrics take aligned (n, 978) arrays. Aggregation is two-level: average over a
compound's signatures first, then over compounds, because signatures of one compound (doses x cells
x times) are strongly dependent and treating them as independent overstates the evidence. Confidence
intervals come from a bootstrap over compounds for the same reason.

Metric menu (why each one exists):
- pearson: the conventional per-signature correlation across genes. Reported, but it rewards
  getting the shared cell/time/dose response right and says little about the compound itself.
- centered_pearson (headline): within each evaluated cell x time x dose context, subtract the mean
  prediction from predictions and the mean observation from observations, then correlate. Any
  compound-agnostic predictor scores exactly 0, so this measures only what distinguishes one
  compound from the others measured in the same context.
- delta_pearson: correlation of (pred - mu) with (true - mu), mu = training context mean. Kept
  because it is the obvious choice and it is subtly broken here: predicting all-zeros scores ~0.07,
  because mu is pulled by a few strongly active training compounds while the typical held-out
  compound is inert, so (true - mu) points along -mu. See WRITEUP.md.
- retrieval (headline): inside each test context, rank the right compound's prediction among all
  test compounds' predictions by cosine to the observed residual. 0 = always first, 0.5 = chance.
- topk_dir / topk25_dir: overlap of predicted vs observed top-k up and top-k down genes (k = 50 / 25;
  a CMap-style query set). k = 25 matches the gene sentences the generative variant writes.
"""

import numpy as np
import pandas as pd

from pertpred.data.task import context_key


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    num = (a * b).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)


def rowwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    num = (a * b).sum(axis=1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-8)


def topk_direction_overlap(pred: np.ndarray, true: np.ndarray, k: int = 50) -> np.ndarray:
    """Mean of |top-k up overlap| / k and |top-k down overlap| / k. Chance ~= k / n_genes."""
    out = np.empty(len(pred))
    for i in range(len(pred)):
        pu = set(np.argpartition(-pred[i], k)[:k])
        tu = set(np.argpartition(-true[i], k)[:k])
        pd_ = set(np.argpartition(pred[i], k)[:k])
        td = set(np.argpartition(true[i], k)[:k])
        out[i] = (len(pu & tu) + len(pd_ & td)) / (2 * k)
    return out


def retrieval_percentile(pred_res: np.ndarray, true_res: np.ndarray, context: np.ndarray, compound: np.ndarray) -> np.ndarray:
    """Per-signature normalized rank of the correct compound among compounds in the same context.

    Candidates are compounds (a compound's prediction = mean of its predictions in that context).
    Ties count half, so a model whose predictions are identical within a context scores exactly 0.5.
    Signatures in contexts with fewer than 2 compounds get NaN.
    """
    out = np.full(len(pred_res), np.nan)
    df = pd.DataFrame({"ctx": context, "cmp": compound, "i": np.arange(len(context))})
    for _, g in df.groupby("ctx", sort=False):
        cmps, inv = np.unique(g["cmp"].to_numpy(), return_inverse=True)
        if len(cmps) < 2:
            continue
        idx = g["i"].to_numpy()
        cand = np.zeros((len(cmps), pred_res.shape[1]))
        np.add.at(cand, inv, pred_res[idx])
        cand /= np.bincount(inv)[:, None]
        cand_n = cand / np.maximum(np.linalg.norm(cand, axis=1, keepdims=True), 1e-8)
        cand_n[np.linalg.norm(cand, axis=1) < 1e-8] = 0.0
        tr = true_res[idx]
        tr_n = tr / np.maximum(np.linalg.norm(tr, axis=1, keepdims=True), 1e-8)
        sims = tr_n @ cand_n.T  # (n_sig_in_ctx, n_cmp)
        correct = sims[np.arange(len(idx)), inv][:, None]
        better = (sims > correct + 1e-9).sum(axis=1)
        ties = (np.abs(sims - correct) <= 1e-9).sum(axis=1) - 1  # exclude self
        out[idx] = (better + 0.5 * ties) / (len(cmps) - 1)
    return out


def center_within_context(x: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Subtract each context's mean row (over the evaluated signatures) from its rows."""
    codes, inv = np.unique(context, return_inverse=True)
    sums = np.zeros((len(codes), x.shape[1]))
    np.add.at(sums, inv, x)
    counts = np.bincount(inv)
    out = x - (sums / counts[:, None])[inv]
    out[counts[inv] < 2] = np.nan  # a lone signature has nothing to be compared against
    return out


def centered_pearson(pred: np.ndarray, true: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Per-signature Pearson after within-context centering of predictions and observations."""
    pc = center_within_context(np.asarray(pred, dtype=np.float64), context)
    tc = center_within_context(np.asarray(true, dtype=np.float64), context)
    out = rowwise_pearson(np.nan_to_num(pc), np.nan_to_num(tc))
    out[np.isnan(pc).any(axis=1)] = np.nan
    return out


def per_signature_metrics(pred: np.ndarray, true: np.ndarray, mu: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    """One row per signature. `meta` must carry sig_id, group_id, cell_id, time_h, dose_bin."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    pred_res, true_res = pred - mu, true - mu
    context = context_key(meta)
    centered = centered_pearson(pred, true, context)
    return pd.DataFrame(
        {
            "sig_id": meta["sig_id"].to_numpy(),
            "group_id": meta["group_id"].to_numpy(),
            "pearson": rowwise_pearson(pred, true),
            "centered_pearson": centered,
            "delta_pearson": rowwise_pearson(pred_res, true_res),
            "retrieval": retrieval_percentile(pred_res, true_res, context, meta["group_id"].to_numpy()),
            "topk_dir": topk_direction_overlap(pred, true),
            "topk25_dir": topk_direction_overlap(pred, true, k=25),
            "rmse": np.sqrt(((pred - true) ** 2).mean(axis=1)),
        }
    )


METRICS = ["pearson", "centered_pearson", "delta_pearson", "retrieval", "topk_dir", "topk25_dir", "rmse"]


def compound_level(per_sig: pd.DataFrame, by: str = "group_id") -> pd.DataFrame:
    return per_sig.groupby(by)[METRICS].mean()


def summarize(per_sig: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Compound-level macro mean with 95% bootstrap CI over compounds."""
    comp = compound_level(per_sig)
    rng = np.random.default_rng(seed)
    vals = comp.to_numpy()
    boots = np.stack([np.nanmean(vals[rng.integers(0, len(vals), len(vals))], axis=0) for _ in range(n_boot)])
    return pd.DataFrame(
        {
            "mean": np.nanmean(vals, axis=0),
            "ci_lo": np.nanpercentile(boots, 2.5, axis=0),
            "ci_hi": np.nanpercentile(boots, 97.5, axis=0),
        },
        index=comp.columns,
    )


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Compound-level paired bootstrap of (a - b) for every metric. a and b are per-signature tables."""
    ca, cb = compound_level(a), compound_level(b)
    ca, cb = ca.align(cb, join="inner")
    d = (ca - cb).to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.stack([np.nanmean(d[rng.integers(0, len(d), len(d))], axis=0) for _ in range(n_boot)])
    return pd.DataFrame(
        {
            "diff": np.nanmean(d, axis=0),
            "ci_lo": np.nanpercentile(boots, 2.5, axis=0),
            "ci_hi": np.nanpercentile(boots, 97.5, axis=0),
            "p_le_0": (boots <= 0).mean(axis=0),
        },
        index=ca.columns,
    )
