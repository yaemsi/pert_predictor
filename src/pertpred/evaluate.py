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
from pertpred.metrics import METRICS, centered_pearson, paired_difference, per_signature_metrics, summarize
from pertpred.task import Task, context_key

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
    """Model-selection criterion on val: compound-level mean centered_pearson (only that metric, for speed)."""
    meta = task.meta.iloc[idx]
    r = centered_pearson(pred, task.Y[idx], context_key(meta))
    return float(pd.Series(r).groupby(meta["group_id"].to_numpy()).mean().mean())


# ---------------------------------------------------------------- report


def _load_per_sig(split: str) -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(PER_SIG_DIR.glob(f"*.{split}.csv.gz")):
        out[p.name.removesuffix(f".{split}.csv.gz")] = pd.read_csv(p)
    return out


def _fmt(s: pd.DataFrame) -> pd.Series:
    return s.apply(lambda r: f"{r['mean']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]", axis=1)


def report(split: str = "test", reference: str = "mlp_fp") -> None:
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

    strata = _strata(tables, out_dir, split)
    _plot_summary(tables, out_dir, split)
    _plot_similarity(strata, out_dir, split)
    _activity_auroc(list(tables), out_dir, split)


def _activity_auroc(names: list[str], out_dir, split: str) -> None:
    """Does a model know *whether* a compound acts? AUROC of predicted response strength
    (||pred - context mean||) for the DMSO-based active flag, computed within each cell x time x dose
    context (activity rises steeply with dose) and size-weighted across contexts. This separates
    "predicts activity" from "predicts which genes move", which centered_pearson mixes together."""
    from sklearn.metrics import roc_auc_score

    from pertpred.task import fit_context_mean, load_task

    task = load_task()
    _, mu = fit_context_mean(task)
    rows = task.idx(split)
    active = activity_flags(task, rows)
    ctx = context_key(task.meta.iloc[rows])
    out = {}
    for name in names:
        if not (C.PRED_DIR / f"{name}.npz").exists():
            continue
        idx, pred = load_predictions(name)
        pos = pd.Series(np.arange(len(idx)), index=idx).loc[rows].to_numpy()
        strength = np.linalg.norm(pred[pos] - mu[rows], axis=1)
        aucs, w = [], []
        for c in np.unique(ctx):
            m = ctx == c
            if active[m].min() != active[m].max() and np.ptp(strength[m]) > 0:
                aucs.append(roc_auc_score(active[m], strength[m]))
                w.append(m.sum())
        out[name] = float(np.average(aucs, weights=w)) if aucs else 0.5
    tab = pd.Series(out, name="within-context AUROC (active | predicted strength)").sort_values().to_frame()
    (out_dir / f"activity_auroc_{split}.md").write_text(tab.to_markdown(floatfmt=".3f"))
    print(f"\n{tab.to_string(float_format='%.3f')}")


MODEL_ORDER = ["zero", "context_mean", "knn_tanimoto", "ridge_fp", "mlp_fp"]
# Headline figure: baselines, then frozen Qwen (val-selected pooling only), then LoRA Qwen.
PLOT_GROUPS = [
    ("baselines", ["zero", "context_mean", "knn_tanimoto", "ridge_fp", "mlp_fp"]),
    ("frozen Qwen + MLP", ["probe_name+smiles_mean24", "probe_smiles_mean24", "probe_name_mean24", "probe_fp+qwen_mean24"]),
    ("Qwen fine-tuned", ["qwen_full", "qwen_main", "qwen_allweights", "qwen_smiles", "qwen_name", "qwen_ctx"]),
]


def _ordered(names) -> list[str]:
    base = [n for n in MODEL_ORDER if n in names]
    return base + sorted(n for n in names if n not in MODEL_ORDER)


def _plot_summary(tables: dict[str, pd.DataFrame], out_dir, split: str) -> None:
    """Dot + 95% CI per model for the two compound-specific metrics, all signatures and active only."""
    from pertpred.plotstyle import INK2, NEUTRAL, SERIES, plt, setup

    setup()
    names, ypos, group_rows, y = [], [], [], 0.0
    for label, members in PLOT_GROUPS:
        present = [n for n in members if n in tables]
        if not present:
            continue
        group_rows.append((label, y))
        y += 0.9
        for n in present:
            names.append(n)
            ypos.append(y)
            y += 1.0
        y += 0.4
    ypos = -np.asarray(ypos)
    panels = [("centered_pearson", "centered Pearson (higher is better)", 0.0), ("retrieval", "retrieval rank (lower is better)", 0.5)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 0.36 * y + 1.2), sharey=True)
    for ax, (metric, label, chance) in zip(axes, panels):
        for subset, color, dy in [("all", SERIES[0], 0.14), ("active", SERIES[1], -0.14)]:
            for yi, n in zip(ypos, names):
                df = tables[n] if subset == "all" else tables[n][tables[n]["active"]]
                s = summarize(df, n_boot=500).loc[metric]
                ax.plot([s["ci_lo"], s["ci_hi"]], [yi + dy] * 2, color=color, lw=2, solid_capstyle="round")
                ax.plot(s["mean"], yi + dy, "o", ms=6, color=color, mec="white", mew=1.2,
                        label=("all test signatures" if subset == "all" else "active only (norm > DMSO 95th pct)") if yi == ypos[0] else None)
        ax.axvline(chance, color=NEUTRAL, lw=1.2, ls="--")
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, names)
    for label, gy in group_rows:
        axes[0].text(-0.02, -gy, label, transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                     fontsize=9, fontweight="bold", color=INK2)
    axes[1].text(0.5, ypos[0] + 0.6, "chance", color=INK2, fontsize=8, ha="center")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.995), fontsize=8.5)
    fig.suptitle(f"Held-out compounds ({split}): compound-level mean, 95% bootstrap CI over compounds", fontsize=11, y=1.03)
    fig.savefig(out_dir / f"models_{split}.png")
    plt.close(fig)


def _plot_similarity(strata: pd.DataFrame, out_dir, split: str) -> None:
    """centered_pearson vs similarity of the test compound to its nearest training compound."""
    from pertpred.plotstyle import SERIES, plt, setup

    setup()
    order = ["<=0.3", "0.3-0.4", "0.4-0.5", "0.5-0.7", ">0.7"]
    s = strata[(strata["stratum"] == "tani_bin") & strata["level"].isin(order)].copy()
    s["pos"] = s["level"].map({lv: i for i, lv in enumerate(order)})
    s = s.sort_values("pos")
    show = [n for n in ("knn_tanimoto", "mlp_fp", "qwen_main") if n in set(s["model"])]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for color, name in zip(SERIES, show):
        g = s[s["model"] == name]
        ax.plot(g["pos"], g["centered_pearson"], marker="o", ms=6, color=color, label=name)
    counts = s[s["model"] == show[0]][["pos", "level", "n_cmp"]] if show else []
    if len(counts):
        ax.set_xticks(counts["pos"], [f"{lv}\n{n} cmpds" for lv, n in zip(counts["level"], counts["n_cmp"])])
    ax.axhline(0, color="#8a8984", lw=1.2, ls="--")
    ax.set_xlabel("max Tanimoto similarity to any training compound")
    ax.set_ylabel("centered Pearson")
    ax.set_title("Compound-specific accuracy vs chemical novelty")
    ax.legend(loc="upper left")
    fig.savefig(out_dir / f"similarity_{split}.png")
    plt.close(fig)


def _strata(tables: dict[str, pd.DataFrame], out_dir, split: str) -> pd.DataFrame:
    def comp_mean(df, metric):
        return df.groupby("group_id")[metric].mean().mean()

    pieces = []
    for name, df in tables.items():
        df = df.copy()
        df["tani_bin"] = pd.cut(
            df["max_tani_train"], [0, 0.3, 0.4, 0.5, 0.7, 1.0], include_lowest=True,
            labels=["<=0.3", "0.3-0.4", "0.4-0.5", "0.5-0.7", ">0.7"],
        ).astype(str)
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
    return strata


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--reference", default="mlp_fp", help="model every other model is paired against")
    a = ap.parse_args()
    report(a.split, a.reference)
