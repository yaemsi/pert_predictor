"""Cell2Sentence-style generative variant: Qwen *writes* the response as a gene sentence.

CellType's Cell2Sentence (C2S) represents expression as text (gene names ordered by expression)
and trains an LLM to read and generate it. The perturbation-response analogue used here: for each
signature, the top-K most up-regulated and top-K most down-regulated landmark genes, in order:

    up: HMOX1 INSIG1 FOXO4 ...
    down: SPDEF UGDH NPEPL1 ...

Qwen2.5-0.5B + LoRA is trained with the ordinary next-token loss on that text (prompt tokens are
masked out of the loss), conditioned on exactly the prompt the regression model sees. At test time
it decodes greedily. The output is parsed tolerantly (unknown symbols and duplicates dropped,
missing sections left empty; parse statistics are reported) and turned into a 978-vector so it is
scored exactly like every other model: unlisted genes get the context mean; the gene written at
up-rank r gets the training-average z-score of the r-th most up-regulated gene (likewise for down).

Checkpoints are chosen by val token loss: generating val at every evaluation would be too slow.

Run (GPU): `python -m pertpred.gen --run-name qwen_gen`
"""

import argparse
import json
import math
import re
import time

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from pertpred import config as C
from pertpred.data import load_signatures
from pertpred.evaluate import save_predictions, score_split
from pertpred.llm import ALL_FIELDS, length_grouped_batches, micro_batches, serialize
from pertpred.task import fit_context_mean, load_task

K = 25
RESPONSE = "\nresponse:\n"
LINE = re.compile(r"^\s*(up|down)\s*:(.*)$", re.MULTILINE | re.IGNORECASE)


# ------------------------------------------------------------------ text <-> genes


def gene_sentence(y: np.ndarray, symbols: np.ndarray, k: int = K) -> str:
    up = np.argsort(-y, kind="stable")[:k]
    down = np.argsort(y, kind="stable")[:k]
    return f"up: {' '.join(symbols[up])}\ndown: {' '.join(symbols[down])}"


def parse(text: str, index: dict, k: int = K) -> tuple[list[int], list[int], dict]:
    """Generated text -> (up gene indices, down gene indices, parse statistics).

    Tolerant by design: the first `up:` / `down:` line wins, unknown symbols and repeats are skipped
    (a gene already listed as up is not accepted as down), and each list is cut at k.
    """
    found = {}
    for m in LINE.finditer(text):
        found.setdefault(m.group(1).lower(), m.group(2).split())
    stats = {"has_up": "up" in found, "has_down": "down" in found, "n_tokens": 0, "n_invalid": 0, "n_duplicate": 0}
    seen, lists = set(), {}
    for key in ("up", "down"):
        genes = []
        for tok in found.get(key, []):
            stats["n_tokens"] += 1
            g = index.get(tok)
            if g is None:
                stats["n_invalid"] += 1
            elif g in seen:
                stats["n_duplicate"] += 1
            else:
                seen.add(g)
                genes.append(g)
        lists[key] = genes[:k]
    stats["n_up"], stats["n_down"] = len(lists["up"]), len(lists["down"])
    return lists["up"], lists["down"], stats


def rank_profiles(Y_train: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Average z-score of the r-th most up- (and down-) regulated gene across training signatures."""
    s = np.sort(Y_train, axis=1)
    return s[:, ::-1][:, :k].mean(axis=0), s[:, :k].mean(axis=0)


def to_vector(mu_row: np.ndarray, up: list[int], down: list[int], prof_up: np.ndarray, prof_down: np.ndarray) -> np.ndarray:
    v = mu_row.astype(np.float32).copy()
    v[up] = prof_up[: len(up)]
    v[down] = prof_down[: len(down)]
    return v


# ------------------------------------------------------------------ training data


class SeqData:
    """prompt + target + EOS, with the prompt masked out of the labels; dynamic right padding."""

    def __init__(self, prompts: list[str], targets: list[str], tok):
        p = tok(prompts, add_special_tokens=False)["input_ids"]
        t = tok(targets, add_special_tokens=False)["input_ids"]
        eos = tok.eos_token_id
        self.ids = [a + b + [eos] for a, b in zip(p, t)]
        self.labels = [[-100] * len(a) + b + [eos] for a, b in zip(p, t)]
        self.n_target = np.array([len(b) + 1 for b in t])
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else eos

    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.ids])

    def __call__(self, rows: np.ndarray, device):
        L = max(len(self.ids[i]) for i in rows)
        ids = torch.full((len(rows), L), self.pad, dtype=torch.long)
        lab = torch.full((len(rows), L), -100, dtype=torch.long)
        mask = torch.zeros((len(rows), L), dtype=torch.long)
        for j, i in enumerate(rows):
            n = len(self.ids[i])
            ids[j, :n] = torch.tensor(self.ids[i])
            lab[j, :n] = torch.tensor(self.labels[i])
            mask[j, :n] = 1
        return ids.to(device), mask.to(device), lab.to(device)


@torch.no_grad()
def mean_token_loss(model, data: SeqData, rows: np.ndarray, device, max_tokens: int) -> float:
    model.eval()
    lengths = data.lengths()
    total, n = 0.0, 0
    for mb in micro_batches(rows, lengths, max_tokens):
        ids, mask, lab = data(mb, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=mask, labels=lab)
        k = int(data.n_target[mb].sum())
        total += out.loss.float().item() * k
        n += k
    return total / n


# ------------------------------------------------------------------ generation


@torch.no_grad()
def generate(model, tok, prompts: list[str], device, max_new: int, bs: int) -> list[str]:
    tok.padding_side = "left"  # decoder-only generation needs the prompt flush against the new tokens
    order = np.argsort([len(p) for p in prompts], kind="stable")
    out = [""] * len(prompts)
    t0 = time.time()
    for s in range(0, len(prompts), bs):
        sel = order[s : s + bs]
        enc = tok([prompts[i] for i in sel], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new, do_sample=False, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id
        )
        for i, text in zip(sel, tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)):
            out[i] = text
        print(f"  generated {min(s + bs, len(prompts))}/{len(prompts)} ({time.time() - t0:.0f}s)", flush=True)
    tok.padding_side = "right"
    return out


# ------------------------------------------------------------------ main


def run(args) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = load_task()
    _, mu = fit_context_mean(task)
    _, _, genes = load_signatures()
    symbols = genes["pr_gene_symbol"].to_numpy()
    index = {s: i for i, s in enumerate(symbols)}
    assert len(index) == len(symbols), "landmark symbols must be unique"

    tok = AutoTokenizer.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION)
    prompts = [serialize(r, ALL_FIELDS) + RESPONSE for r in task.meta.itertuples()]
    tr, va, te = task.idx("train"), task.idx("val"), task.idx("test")
    if args.max_train_rows:
        tr = rng.choice(tr, size=min(args.max_train_rows, len(tr)), replace=False)
    va_loss_rows = np.sort(rng.choice(va, size=min(args.val_loss_rows, len(va)), replace=False))

    # Sanity: the target format round-trips through the parser exactly.
    for i in tr[:200]:
        up, down, st = parse(gene_sentence(task.Y[i], symbols), index)
        assert len(up) == K and len(down) == K and st["n_invalid"] == 0, st

    fit_rows = np.concatenate([tr, va_loss_rows])
    data = SeqData([prompts[i] for i in fit_rows], [gene_sentence(task.Y[i], symbols) for i in fit_rows], tok)
    pos = {r: j for j, r in enumerate(fit_rows)}  # task row -> SeqData row
    tr_d = np.array([pos[r] for r in tr])
    va_d = np.array([pos[r] for r in va_loss_rows])
    lengths = data.lengths()
    max_new = int(data.n_target.max()) + 8
    print(f"example target:\n{gene_sentence(task.Y[te[0]], symbols)}\n---")
    print(f"sequence tokens: median={np.median(lengths):.0f} max={lengths.max()}; target tokens max={data.n_target.max()}")

    model = AutoModelForCausalLM.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION, dtype=torch.float32)
    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lcfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable parameters: {sum(p.numel() for p in params) / 1e6:.2f}M")

    log_dir = C.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt = C.CKPT_DIR / f"{args.run_name}.pt"
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(log_dir / f"{args.run_name}.jsonl", "a" if args.eval_only else "w")
    log_f.write(json.dumps({"config": vars(args), "n_train_rows": int(len(tr)), "K": K}) + "\n")

    t0 = time.time()
    if not args.eval_only:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        steps_per_epoch = math.ceil(len(tr_d) / args.batch_size)
        total = steps_per_epoch * args.epochs
        warmup = max(1, int(0.03 * total))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
        )
        eval_every = max(1, int(steps_per_epoch * args.eval_every_epochs))
        best, step = math.inf, 0
        for epoch in range(args.epochs):
            for rows in length_grouped_batches(tr_d, lengths, args.batch_size, rng):
                model.train()
                opt.zero_grad(set_to_none=True)
                n_batch, loss_sum = int(data.n_target[rows].sum()), 0.0
                for mb in micro_batches(rows, lengths, args.max_tokens):
                    ids, mask, lab = data(mb, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        out = model(input_ids=ids, attention_mask=mask, labels=lab)
                    # token-weighted so the accumulated gradient equals the full-batch mean token loss
                    loss = out.loss.float() * (int(data.n_target[mb].sum()) / n_batch)
                    loss.backward()
                    loss_sum += loss.item()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                step += 1
                if step % 50 == 0:
                    rate = step * args.batch_size / (time.time() - t0)
                    print(f"step {step}/{total} epoch {epoch} loss {loss_sum:.4f} ({rate:.0f} rows/s)", flush=True)
                    log_f.write(json.dumps({"step": step, "loss": loss_sum}) + "\n")
                if step % eval_every == 0 or step == total:
                    v = mean_token_loss(model, data, va_d, device, args.max_tokens)
                    print(f"  [val] step {step} token_loss={v:.4f}", flush=True)
                    log_f.write(json.dumps({"step": step, "val_token_loss": v}) + "\n")
                    log_f.flush()
                    if v < best:
                        best = v
                        torch.save({"state": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                                    "step": step, "val_token_loss": v}, ckpt)

    state = torch.load(ckpt)
    assert not model.load_state_dict(state["state"], strict=False).unexpected_keys
    print(f"best val token loss={state['val_token_loss']:.4f} at step {state['step']}")
    model = model.merge_and_unload().to(torch.bfloat16).eval()

    q = np.concatenate([va, te])
    if args.gen_limit:  # smoke tests: a small, fixed subset of val and of test
        q = np.concatenate([va[: args.gen_limit], te[: args.gen_limit]])
    texts = generate(model, tok, [prompts[i] for i in q], device, max_new, args.gen_batch)

    gen_dir = C.CACHE_DIR / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    with open(gen_dir / f"{args.run_name}.jsonl", "w") as f:
        for i, t in zip(q, texts):
            f.write(json.dumps({"row": int(i), "sig_id": task.meta.sig_id.iloc[i], "text": t}) + "\n")

    prof_up, prof_down = rank_profiles(task.Y[tr])
    preds, stats = [], []
    for i, t in zip(q, texts):
        up, down, st = parse(t, index)
        preds.append(to_vector(mu[i], up, down, prof_up, prof_down))
        stats.append(st)
    st = pd.DataFrame(stats)
    parse_summary = {
        "n": int(len(st)),
        "well_formed_frac": float((st.has_up & st.has_down & (st.n_up == K) & (st.n_down == K)).mean()),
        "missing_section_frac": float((~(st.has_up & st.has_down)).mean()),
        "invalid_symbol_frac": float(st.n_invalid.sum() / max(1, st.n_tokens.sum())),
        "duplicate_frac": float(st.n_duplicate.sum() / max(1, st.n_tokens.sum())),
        "mean_valid_up": float(st.n_up.mean()),
        "mean_valid_down": float(st.n_down.mean()),
        "distinct_up_lists": int(len(set(tuple(parse(t, index)[0]) for t in texts))),
    }
    print(json.dumps(parse_summary, indent=2))
    (log_dir / f"{args.run_name}_parse.json").write_text(json.dumps(parse_summary, indent=2))

    # A few outputs next to the truth, for reading.
    sample = [f"### {task.meta.pert_iname.iloc[i]} | {task.meta.cell_id.iloc[i]} | {task.meta.dose_um.iloc[i]:g} uM | "
              f"{task.meta.time_h.iloc[i]:g} h\n\ngenerated:\n```\n{t.strip()}\n```\ntrue:\n```\n{gene_sentence(task.Y[i], symbols)}\n```\n"
              for i, t in list(zip(q, texts))[-8:]]
    (log_dir / f"{args.run_name}_samples.md").write_text("\n".join(sample))

    pred = np.stack(preds)
    save_predictions(args.run_name, q, pred)
    for split in ("val", "test"):
        score_split(args.run_name, task, mu, q, pred, split)
    log_f.write(json.dumps({"best_step": state["step"], "best_val_token_loss": state["val_token_loss"],
                            "parse": parse_summary, "minutes": (time.time() - t0) / 60}) + "\n")
    log_f.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=4096, help="padded-token cap per micro-batch (memory)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every-epochs", type=float, default=0.25)
    ap.add_argument("--val-loss-rows", type=int, default=2000)
    ap.add_argument("--max-train-rows", type=int, default=0, help="subsample train rows (smoke tests)")
    ap.add_argument("--gen-limit", type=int, default=0, help="generate only this many val and test rows (smoke tests)")
    ap.add_argument("--gen-batch", type=int, default=128)
    ap.add_argument("--eval-only", action="store_true", help="skip training; generate from the saved checkpoint")
    ap.add_argument("--seed", type=int, default=C.SEED)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
