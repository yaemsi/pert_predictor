"""Score saved predictions and build the comparison report.

Every model writes predictions for val+test rows via `save_predictions`. `score` turns them into a
per-signature metric table (kept in results/ so the numbers can be re-aggregated without the model).
`python -m pertpred.evaluate` builds the final tables and plots from whatever has been scored.
"""

import functools
import json

import numpy as np
import pandas as pd

from pertpred import config as C
from pertpred.data import load_signatures
from pertpred.metrics import METRICS, paired_difference, per_signature_metrics, summarize
from pertpred.task import Task

PER_SIG_DIR = C.RESULTS_DIR / "per_signature"


def save_predictions(name: str, idx: np.ndarray, pred: np.ndarray) -> None:
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.PRED_DIR / f"{name}.npz", idx=idx, pred=pred.astype(np.float32))


def load_predictions(name: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(C.PRED_DIR / f"{name}.npz")
    return z["idx"], z["pred"]


@functools.cache
def dmso_activity_threshold(q: float = 0.95) -> pd.DataFrame:
    """Per (cell, time) q-quantile of DMSO consensus-signature norms: the measured noise floor.

    A treated signature whose norm does not exceed what vehicle alone produces in the same cell and
    time point carries no detectable compound effect; scoring a model on it measures noise.
    """
    sigs, Y, _ = load_signatures()
    ctl = np.flatnonzero((sigs["pert_type"] == "ctl_vehicle").to_numpy())
    norms = np.linalg.norm(np.clip(np.asarray(Y[ctl]), -C.Z_CLIP, C.Z_CLIP), axis=1)
    df = pd.DataFrame({"cell_id": sigs["cell_id"].to_numpy()[ctl], "time_h": sigs["time_h"].to_numpy()[ctl], "norm": norms})
    thr = df.groupby(["cell_id", "time_h"])["norm"].agg(["size", lambda s: s.quantile(q)])
    thr.columns = ["n", "thr"]
    thr.attrs["global"] = float(np.quantile(norms, q))
    return thr


def activity_flags(task: Task, idx: np.ndarray) -> np.ndarray:
    thr = dmso_activity_threshold()
    m = task.meta.iloc[idx]
    keys = list(zip(m["cell_id"], m["time_h"]))
    t = np.array([thr["thr"].get(k, thr.attrs["global"]) if thr["n"].get(k, 0) >= 20 else thr.attrs["global"] for k in keys])
    return np.linalg.norm(task.Y[idx], axis=1) > t


STRATA_COLS = ["cell_id", "time_h", "dose_bin", "is_named", "max_tani_train", "pert_iname"]


def score(name: str, task: Task, mu: np.ndarray, idx: np.ndarray, pred: np.ndarray, split: str) -> pd.DataFrame:
    meta = task.meta.iloc[idx]
    per_sig = per_signature_metrics(pred, task.Y[idx], mu[idx], meta)
    for col in STRATA_COLS:
        per_sig[col] = meta[col].to_numpy()
    per_sig["active"] = activity_flags(task, idx)
    PER_SIG_DIR.mkdir(parents=True, exist_ok=True)
    per_sig.to_csv(PER_SIG_DIR / f"{name}.{split}.csv.gz", index=False, float_format="%.5f")
    return per_sig


def score_split(name: str, task: Task, mu: np.ndarray, idx_all: np.ndarray, pred_all: np.ndarray, split: str) -> pd.DataFrame:
    """Score the rows of `split` out of a val+test prediction block; print the compound-level summary."""
    sel = task.meta["split"].to_numpy()[idx_all] == split
    per_sig = score(name, task, mu, idx_all[sel], pred_all[sel], split)
    s = summarize(per_sig)
    print(f"[{name} | {split}] " + "  ".join(f"{m}={s.loc[m, 'mean']:.4f}" for m in METRICS))
    return per_sig


def val_selection_score(task: Task, mu: np.ndarray, idx: np.ndarray, pred: np.ndarray) -> float:
    """Model-selection criterion on val: compound-level mean centered_pearson."""
    meta = task.meta.iloc[idx]
    per_sig = per_signature_metrics(pred, task.Y[idx], mu[idx], meta)
    return float(per_sig.groupby("group_id")["centered_pearson"].mean().mean())


# ---------------------------------------------------------------- report


def _load_per_sig(split: str) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(PER_SIG_DIR.glob(f"*.{split}.csv.gz")):
        out[p.name.removesuffix(f".{split}.csv.gz")] = pd.read_csv(p)
    return out


def _fmt(s: pd.DataFrame) -> pd.Series:
    return s.apply(lambda r: f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1)


def report(split: str = "test", reference: str = "context_mean") -> None:
    tables = _load_per_sig(split)
    if not tables:
        raise SystemExit(f"no scored predictions for split={split}")
    out_dir = C.RESULTS_DIR / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, raw = {}, {}
    for name, df in tables.items():
        s = summarize(df)
        rows[name] = _fmt(s)
        raw[name] = s["mean"].to_dict() | {"n_signatures": len(df), "n_compounds": int(df["group_id"].nunique())}
        act = df[df["active"]]
        if len(act):
            sa = summarize(act)
            for m in ("pearson", "centered_pearson", "retrieval"):
                rows[name][f"{m} (active)"] = f"{sa.loc[m, 'mean']:.3f} [{sa.loc[m, 'ci_lo']:.3f}, {sa.loc[m, 'ci_hi']:.3f}]"
                raw[name][f"{m}_active"] = float(sa.loc[m, "mean"])
    main = pd.DataFrame(rows).T
    order = main.index.to_series().map(lambda n: raw[n]["centered_pearson"]).sort_values().index
    main = main.loc[order]
    (out_dir / f"summary_{split}.md").write_text(main.to_markdown())
    (out_dir / f"summary_{split}.json").write_text(json.dumps(raw, indent=2))
    print(main.to_string())

    if reference in tables:
        diffs = {}
        for name, df in tables.items():
            if name == reference:
                continue
            d = paired_difference(df, tables[reference])
            diffs[name] = d.apply(lambda r: f"{r['diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]", axis=1)
        dtab = pd.DataFrame(diffs).T
        (out_dir / f"paired_vs_{reference}_{split}.md").write_text(dtab.to_markdown())
        print(f"\npaired (model - {reference}), compound bootstrap:\n{dtab.to_string()}")

    _strata(tables, out_dir, split)


def _strata(tables: dict[str, pd.DataFrame], out_dir, split: str) -> None:
    def comp_mean(df, metric):
        return df.groupby("group_id")[metric].mean().mean()

    pieces = []
    for name, df in tables.items():
        df = df.copy()
        df["tani_bin"] = pd.cut(df["max_tani_train"], [0, 0.3, 0.4, 0.5, 0.7, 1.0], include_lowest=True).astype(str)
        for col in ("cell_id", "time_h", "dose_bin", "is_named", "active", "tani_bin"):
            for level, g in df.groupby(col):
                pieces.append(
                    {
                        "model": name,
                        "stratum": col,
                        "level": str(level),
                        "n_sig": len(g),
                        "n_cmp": g["group_id"].nunique(),
                        "centered_pearson": comp_mean(g, "centered_pearson"),
                        "pearson": comp_mean(g, "pearson"),
                        "retrieval": comp_mean(g, "retrieval"),
                    }
                )
    strata = pd.DataFrame(pieces)
    strata.to_csv(out_dir / f"strata_{split}.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--reference", default="context_mean")
    a = ap.parse_args()
    report(a.split, a.reference)
