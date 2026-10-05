"""Non-LLM baselines. Hyperparameters are chosen on val (compound-level centered_pearson); test is scored once.

- zero:          predict 0 for every gene (Level 5 z-scores are centred on the plate population).
- context_mean:  training mean for the cell x time x dose context. Knows everything except the compound.
- knn_tanimoto:  context mean + mean residual of the k most structurally similar training compounds
                 measured in the same context (Morgan r=2, Tanimoto). The classic chemistry baseline.
- ridge_fp:      context mean + ridge regression from [Morgan bits, context one-hots] to the residual.
- mlp_fp:        context mean + MLP on Morgan bits with learned cell / time / dose embeddings. Same
                 information as the LLM's SMILES channel, no language-model prior.

Run: `python -m pertpred.baselines`
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

from pertpred import config as C
from pertpred.evaluate import save_predictions, score_split, val_selection_score
from pertpred.task import Task, as_str, context_key, fit_context_mean, load_task


def load_fps(meta: pd.DataFrame) -> np.ndarray:
    z = np.load(C.FP_PATH, allow_pickle=True)
    pos = pd.Series(np.arange(len(z["pert_id"])), index=z["pert_id"])
    return z["fp"][pos.loc[meta["pert_id"]].to_numpy()]


def tanimoto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


# ------------------------------------------------------------------ kNN


def knn_predict(task: Task, mu: np.ndarray, query_idx: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """Return {k: predictions for query_idx}. Neighbours are compounds measured in the same context."""
    meta = task.meta
    tr = task.idx("train")
    res = task.Y - mu

    # Mean training residual per (context, compound).
    tr_df = pd.DataFrame({"ctx": context_key(meta.iloc[tr]), "pert": meta["pert_id"].to_numpy()[tr], "row": tr})
    cmp_ids = np.unique(np.concatenate([meta["pert_id"].to_numpy()[tr], meta["pert_id"].to_numpy()[query_idx]]))
    cmp_pos = pd.Series(np.arange(len(cmp_ids)), index=cmp_ids)
    fp_tab = load_fps(pd.DataFrame({"pert_id": cmp_ids}))
    has_fp = fp_tab.sum(1) > 0
    S = tanimoto(fp_tab, fp_tab)

    ctx_tables = {}
    for ctx, g in tr_df.groupby("ctx"):
        perts, inv = np.unique(g["pert"].to_numpy(), return_inverse=True)
        R = np.zeros((len(perts), res.shape[1]), dtype=np.float32)
        np.add.at(R, inv, res[g["row"].to_numpy()])
        R /= np.bincount(inv)[:, None]
        ctx_tables[ctx] = (cmp_pos.loc[perts].to_numpy(), R)

    q_ctx = context_key(meta.iloc[query_idx])
    q_cmp = cmp_pos.loc[meta["pert_id"].to_numpy()[query_idx]].to_numpy()
    out = {k: mu[query_idx].copy() for k in ks}
    for i, (ctx, c) in enumerate(zip(q_ctx, q_cmp)):
        if ctx not in ctx_tables or not has_fp[c]:
            continue  # no structure or no training compounds in this context -> context mean
        cand, R = ctx_tables[ctx]
        order = np.argsort(-S[c, cand], kind="stable")
        for k in ks:
            out[k][i] += R[order[:k]].mean(axis=0)
    return out


# ------------------------------------------------------------------ ridge


def _design(meta: pd.DataFrame, fps: np.ndarray, levels: dict) -> np.ndarray:
    parts = [fps.astype(np.float32)]
    for col, values in levels.items():
        parts.append((as_str(meta[col])[:, None] == np.asarray(values)[None, :]).astype(np.float32))
    return np.concatenate(parts, axis=1)


def ridge_predict(task: Task, mu: np.ndarray, query_idx: np.ndarray, alphas: list[float]) -> dict[float, np.ndarray]:
    meta = task.meta
    tr = task.idx("train")
    levels = {c: sorted(np.unique(as_str(meta[c]))) for c in ("cell_id", "time_h", "dose_bin")}
    X_tr = _design(meta.iloc[tr], load_fps(meta.iloc[tr]), levels)
    X_q = _design(meta.iloc[query_idx], load_fps(meta.iloc[query_idx]), levels)
    target = task.Y[tr] - mu[tr]
    out = {}
    for a in alphas:
        model = Ridge(alpha=a).fit(X_tr, target)
        out[a] = mu[query_idx] + model.predict(X_q).astype(np.float32)
    return out


# ------------------------------------------------------------------ MLP


class FPMLP(torch.nn.Module):
    def __init__(self, n_cells: int, n_times: int, n_doses: int, n_genes: int, hidden: int = 1024, dropout: float = 0.2):
        super().__init__()
        self.fp = torch.nn.Sequential(torch.nn.Linear(2048, 512), torch.nn.GELU(), torch.nn.Dropout(dropout))
        self.cell = torch.nn.Embedding(n_cells, 32)
        self.time = torch.nn.Embedding(n_times, 8)
        self.dose = torch.nn.Embedding(n_doses, 8)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(512 + 32 + 8 + 8, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, n_genes)
        )

    def forward(self, fp, cell, time, dose):
        h = torch.cat([self.fp(fp), self.cell(cell), self.time(time), self.dose(dose)], dim=1)
        return self.head(h)


def mlp_predict(task: Task, mu: np.ndarray, epochs: int = 40, seed: int = C.SEED) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train on train, early-stop on val centered_pearson. Returns (val+test idx, predictions, val curve)."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = task.meta
    codes = {c: pd.Categorical(as_str(meta[c])).codes for c in ("cell_id", "time_h", "dose_bin")}
    fps = load_fps(meta)
    res = task.Y - mu

    def tensors(idx):
        return [
            torch.tensor(fps[idx], dtype=torch.float32, device=dev),
            *(torch.tensor(codes[c][idx], dtype=torch.long, device=dev) for c in ("cell_id", "time_h", "dose_bin")),
        ]

    tr, va = task.idx("train"), task.idx("val")
    q = np.concatenate([va, task.idx("test")])
    X_tr, Y_tr = tensors(tr), torch.tensor(res[tr], device=dev)
    X_q = tensors(q)
    n_lvls = [int(codes[c].max()) + 1 for c in ("cell_id", "time_h", "dose_bin")]
    model = FPMLP(*n_lvls, n_genes=res.shape[1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def predict():
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(*(t[s : s + 4096] for t in X_q)).cpu().numpy() for s in range(0, len(q), 4096)])

    best, best_pred, curve = -np.inf, None, []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), device=dev)
        for s in range(0, len(tr), 512):
            b = perm[s : s + 512]
            loss = torch.nn.functional.mse_loss(model(*(t[b] for t in X_tr)), Y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        pred = mu[q] + predict()
        v = val_selection_score(task, mu, va, pred[: len(va)])
        curve.append(v)
        if v > best:
            best, best_pred = v, pred
        print(f"  mlp epoch {ep + 1}/{epochs} train_loss={loss.item():.4f} val_centered_pearson={v:.4f}")
    return q, best_pred, curve


# ------------------------------------------------------------------ driver


def run_tuned(name, task, mu, preds_by_param: dict, q: np.ndarray, log: dict) -> None:
    va_n = len(task.idx("val"))
    scores = {p: val_selection_score(task, mu, q[:va_n], pr[:va_n]) for p, pr in preds_by_param.items()}
    best = max(scores, key=scores.get)
    print(f"{name}: val centered_pearson by param {({k: round(v, 4) for k, v in scores.items()})} -> best={best}")
    log[name] = {"val_scores": {str(k): v for k, v in scores.items()}, "chosen": str(best)}
    save_predictions(name, q, preds_by_param[best])
    for split in ("val", "test"):
        score_split(name, task, mu, q, preds_by_param[best], split)


def main() -> None:
    import json

    task = load_task()
    _, mu = fit_context_mean(task)
    q = np.concatenate([task.idx("val"), task.idx("test")])
    log = {}

    for name, pred in {
        "zero": np.zeros((len(q), task.Y.shape[1]), dtype=np.float32),
        "context_mean": mu[q],
    }.items():
        save_predictions(name, q, pred)
        for split in ("val", "test"):
            score_split(name, task, mu, q, pred, split)

    run_tuned("knn_tanimoto", task, mu, knn_predict(task, mu, q, ks=[1, 3, 5, 10, 20, 50]), q, log)
    run_tuned("ridge_fp", task, mu, ridge_predict(task, mu, q, alphas=[10.0, 100.0, 1000.0, 10000.0]), q, log)

    q_mlp, pred_mlp, curve = mlp_predict(task, mu)
    assert (q_mlp == q).all()
    log["mlp_fp"] = {"val_curve": curve, "chosen_epoch": int(np.argmax(curve)) + 1}
    save_predictions("mlp_fp", q, pred_mlp)
    for split in ("val", "test"):
        score_split("mlp_fp", task, mu, q, pred_mlp, split)

    (C.RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (C.RESULTS_DIR / "logs" / "baselines_tuning.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
