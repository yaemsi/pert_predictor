"""Data diagnostics that shaped the evaluation design. Writes outputs/results/eda/.

1. How much of a signature is signal? Treated vs DMSO consensus-signature norms.
2. Empirical noise ceiling: conditions measured twice (same compound, cell, time, dose on different
   plates). How well does one measurement predict the other, as a function of signal strength?
3. How much does the cell x time x dose context explain, in-sample vs on held-out compounds?

Run: `python main.py eda`
"""

import json

import numpy as np
import pandas as pd

from pertpred.data.load import load_signatures
from pertpred.data.task import fit_context_mean, load_task
from pertpred.utils import config as C
from pertpred.utils.evaluate import activity_flags, dmso_activity_threshold
from pertpred.utils.metrics import rowwise_pearson
from pertpred.utils.plotstyle import INK2, NEUTRAL, SERIES, plt, setup

OUT = C.RESULTS_DIR / "eda"


def replicate_pairs(task) -> pd.DataFrame:
    """First two signatures of every exactly-repeated (compound, cell, time, dose) condition."""
    m = task.meta
    key = m["pert_id"] + "|" + m["cell_id"] + "|" + m["time_h"].astype(str) + "|" + m["dose_um"].astype(str)
    rows = []
    for _, g in m.groupby(key):
        if len(g) >= 2:
            i, j = g.index[:2]
            rows.append((i, j))
    return pd.DataFrame(rows, columns=["i", "j"])


def _log_ticks(ax, ticks) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def run(args=None) -> None:
    setup()
    OUT.mkdir(parents=True, exist_ok=True)
    task = load_task()
    _, mu = fit_context_mean(task)
    Y, m = task.Y, task.meta
    summary = {}

    # ---- 1. signal vs vehicle noise
    sigs, Y_all, _ = load_signatures()
    ctl = np.flatnonzero((sigs["pert_type"] == "ctl_vehicle").to_numpy())
    dmso_norm = np.linalg.norm(np.clip(np.asarray(Y_all[ctl]), -C.Z_CLIP, C.Z_CLIP), axis=1)
    trt_norm = np.linalg.norm(Y, axis=1)
    active = activity_flags(task, np.arange(len(m)))
    summary["n_trt_signatures"] = int(len(m))
    summary["n_dmso_signatures"] = int(len(ctl))
    summary["norm_median"] = {"dmso": float(np.median(dmso_norm)), "treated": float(np.median(trt_norm))}
    summary["dmso_norm_q95_global"] = float(dmso_activity_threshold().attrs["global"])
    summary["frac_active"] = {s: float(active[task.idx(s)].mean()) for s in ("train", "val", "test")}

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.geomspace(15, 250, 60)
    ax.hist(dmso_norm, bins=bins, density=True, histtype="step", color=SERIES[0], lw=2, label="DMSO vehicle")
    ax.hist(trt_norm, bins=bins, density=True, histtype="step", color=SERIES[1], lw=2, label="compound-treated")
    ax.axvline(summary["dmso_norm_q95_global"], color=NEUTRAL, lw=1.2, ls="--")
    ax.text(summary["dmso_norm_q95_global"] * 1.04, ax.get_ylim()[1] * 0.9, "DMSO 95th pct", color=INK2, fontsize=8.5)
    _log_ticks(ax, [20, 30, 45, 70, 100, 200])
    ax.set_xlabel("L2 norm of 978-gene consensus signature (clipped z)")
    ax.set_ylabel("density")
    ax.set_title(f"Most treated signatures look like vehicle: {100 * active.mean():.0f}% exceed the DMSO 95th percentile")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "signal_vs_vehicle.png")
    plt.close(fig)

    # ---- 2. replicate noise ceiling
    pairs = replicate_pairs(task)
    a, b = Y[pairs["i"].to_numpy()], Y[pairs["j"].to_numpy()]
    pairs["r"] = rowwise_pearson(a, b)
    pairs["r_residual"] = rowwise_pearson(a - mu[pairs["i"]], b - mu[pairs["j"]])
    pairs["min_norm"] = np.minimum(trt_norm[pairs["i"]], trt_norm[pairs["j"]])
    pairs["both_active"] = active[pairs["i"]] & active[pairs["j"]]
    summary["replicate_pairs"] = {
        "n_pairs": int(len(pairs)),
        "pearson_median": float(pairs["r"].median()),
        "pearson_mean": float(pairs["r"].mean()),
        "pearson_mean_both_active": float(pairs.loc[pairs["both_active"], "r"].mean()),
        "pearson_mean_not_both_active": float(pairs.loc[~pairs["both_active"], "r"].mean()),
        "residual_pearson_mean": float(pairs["r_residual"].mean()),
    }
    pairs["bin"] = pd.qcut(pairs["min_norm"], 8)
    curve = pairs.groupby("bin", observed=True).agg(x=("min_norm", "median"), r=("r", "median"), r_res=("r_residual", "median"))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(curve["x"], curve["r"], marker="o", ms=5, color=SERIES[0], label="raw signature")
    ax.plot(curve["x"], curve["r_res"], marker="o", ms=5, color=SERIES[1], label="minus context mean")
    ax.axvline(summary["dmso_norm_q95_global"], color=NEUTRAL, lw=1.2, ls="--")
    ax.text(summary["dmso_norm_q95_global"] * 1.02, 0.05, "DMSO 95th pct", color=INK2, fontsize=8.5)
    _log_ticks(ax, [25, 30, 40, 50, 60, 70])
    ax.set_xlabel("weaker signature norm of the pair (octile median)")
    ax.set_ylabel("median Pearson between replicates")
    ax.set_title(f"Noise ceiling: re-measuring the same condition ({len(pairs)} pairs)")
    ax.legend(loc="upper left")
    fig.savefig(OUT / "replicate_ceiling.png")
    plt.close(fig)

    # ---- 3. context mean, in-sample vs held-out compounds
    tr, va = task.idx("train"), task.idx("val")
    gm = Y[tr].mean(0)
    summary["context_r2"] = {
        "train_in_sample": float(1 - ((Y[tr] - mu[tr]) ** 2).sum() / ((Y[tr] - gm) ** 2).sum()),
        "val_heldout_compounds": float(1 - ((Y[va] - mu[va]) ** 2).sum() / ((Y[va] - gm) ** 2).sum()),
    }
    summary["counts"] = {
        "by_time_h": m["time_h"].value_counts().sort_index().to_dict(),
        "by_cell_top10": m["cell_id"].value_counts().head(10).to_dict(),
        "signatures_per_compound_median": float(m.groupby("group_id").size().median()),
        "test_max_tanimoto_to_train_median": float(m.drop_duplicates("pert_id").query("split=='test'")["max_tani_train"].median()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))

