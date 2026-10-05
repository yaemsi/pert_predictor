"""Frozen-Qwen probe: does the pretrained representation of a compound carry response information?

Each compound is embedded once by frozen Qwen2.5-0.5B (no weight updates). The embedding replaces
Morgan bits in exactly the same MLP as `mlp_fp`, so the comparison isolates the representation.

Variants:
  text:    name+smiles | smiles | name     (what Qwen reads about the compound)
  pooling: eos  (hidden state at an appended end-of-text token)
           mean (average over the compound's tokens)
  layer:   12 (middle of 24) | 24 (final, after the last norm)

Pooling/layer is chosen on val with the name+smiles text; the other texts reuse that choice.
Run (GPU): `python -m pertpred.probe`
"""

import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from pertpred import config as C
from pertpred.baselines import load_fps, mlp_predict
from pertpred.evaluate import save_predictions, score_split, val_selection_score
from pertpred.task import fit_context_mean, load_task

TEXTS = ("name+smiles", "smiles", "name")
POOLS = (("eos", 12), ("eos", 24), ("mean", 12), ("mean", 24))


def compound_text(row, variant: str) -> str:
    lines = []
    if "name" in variant.split("+") and row.is_named:
        lines.append(f"compound: {row.pert_iname}")
    if "smiles" in variant.split("+") and isinstance(row.canonical_smiles, str):
        lines.append(f"smiles: {row.canonical_smiles}")
    return "\n".join(lines)


@torch.no_grad()
def embed(texts: list[str], bs: int = 64) -> dict[tuple[str, int], np.ndarray]:
    """{(pool, layer): (n, d)}; empty texts (nothing to read) get a zero vector."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION)
    model = AutoModel.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION, dtype=torch.bfloat16).to(dev).eval()
    d = model.config.hidden_size
    out = {p: np.zeros((len(texts), d), dtype=np.float32) for p in POOLS}
    nonempty = [i for i, t in enumerate(texts) if t]
    for s in range(0, len(nonempty), bs):
        idx = nonempty[s : s + bs]
        ids = [tok(texts[i], add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for i in idx]
        L = max(map(len, ids))
        x = torch.full((len(ids), L), tok.eos_token_id, dtype=torch.long)
        mask = torch.zeros((len(ids), L), dtype=torch.long)
        for j, seq in enumerate(ids):
            x[j, : len(seq)] = torch.tensor(seq)
            mask[j, : len(seq)] = 1
        x, mask = x.to(dev), mask.to(dev)
        hs = model(input_ids=x, attention_mask=mask, output_hidden_states=True).hidden_states
        last = mask.sum(1) - 1
        for pool, layer in POOLS:
            h = hs[layer].float()
            if pool == "eos":
                v = h[torch.arange(len(idx), device=dev), last]
            else:  # mean over text tokens, excluding the appended EOS
                m = mask.clone()
                m[torch.arange(len(idx), device=dev), last] = 0
                v = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
            out[(pool, layer)][idx] = v.cpu().numpy()
    return out


def standardize(feat_cmp: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    mu, sd = feat_cmp[train_mask].mean(0), feat_cmp[train_mask].std(0) + 1e-6
    return ((feat_cmp - mu) / sd).astype(np.float32)


def run(name: str, task, mu, feat_rows: np.ndarray, log: dict) -> float:
    q, pred, curve = mlp_predict(task, mu, features=feat_rows)
    va_n = len(task.idx("val"))
    v = val_selection_score(task, mu, q[:va_n], pred[:va_n])
    save_predictions(name, q, pred)
    for split in ("val", "test"):
        score_split(name, task, mu, q, pred, split)
    log[name] = {"val_centered_pearson": v, "val_curve": curve}
    return v


EMB_PATH = C.CACHE_DIR / "probe_embeddings.npz"
LOG_PATH = C.RESULTS_DIR / "logs" / "probe.json"


def main(stage: str) -> None:
    task = load_task()
    _, mu = fit_context_mean(task)
    cmp = task.meta.drop_duplicates("pert_id").reset_index(drop=True)
    row_to_cmp = pd.Series(np.arange(len(cmp)), index=cmp["pert_id"]).loc[task.meta["pert_id"]].to_numpy()
    train_mask = (cmp["split"] == "train").to_numpy()

    if stage == "all":
        log = {}
        feats = {v: embed([compound_text(r, v) for r in cmp.itertuples()]) for v in TEXTS}
        np.savez(EMB_PATH, pert_id=cmp["pert_id"].to_numpy(), **{f"{v}|{p}{l}": a for v, d in feats.items() for (p, l), a in d.items()})

        scores = {}
        for pool, layer in POOLS:
            name = f"probe_name+smiles_{pool}{layer}"
            f = standardize(feats["name+smiles"][(pool, layer)], train_mask)
            scores[(pool, layer)] = run(name, task, mu, f[row_to_cmp], log)
        best = max(scores, key=scores.get)
        print(f"probe pooling chosen on val: {best} ({scores[best]:.4f})")
        log["chosen_pooling"] = list(best)

        for v in ("smiles", "name"):
            f = standardize(feats[v][best], train_mask)
            run(f"probe_{v}_{best[0]}{best[1]}", task, mu, f[row_to_cmp], log)
        LOG_PATH.write_text(json.dumps(log, indent=2))

    # Does Qwen add anything on top of chemistry? Morgan bits concatenated with the chosen embedding.
    log = json.loads(LOG_PATH.read_text())
    pool, layer = log["chosen_pooling"]
    emb = np.load(EMB_PATH, allow_pickle=True)
    assert (emb["pert_id"] == cmp["pert_id"].to_numpy()).all()
    q = standardize(emb[f"name+smiles|{pool}{layer}"], train_mask)
    fp = load_fps(cmp).astype(np.float32)
    run(f"probe_fp+qwen_{pool}{layer}", task, mu, np.concatenate([fp, q], axis=1)[row_to_cmp], log)
    LOG_PATH.write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "fp_concat"], default="all", help="fp_concat reuses cached embeddings")
    main(ap.parse_args().stage)
