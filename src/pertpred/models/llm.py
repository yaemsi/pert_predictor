"""Qwen2.5-0.5B as a conditional encoder of a serialized perturbation, with a regression read-out.

The condition (cell line, time, dose, compound name, SMILES) is written as plain text. Qwen reads it;
the hidden state at an appended end-of-text token goes through a linear head that predicts the
978-gene residual over the context mean. Prediction = context_mean + head(h).

The head is zero-initialized, so before any training the model *is* the context-mean baseline and
every gain has to come from what the backbone extracts from the text. Fields can be dropped from the
prompt (`--fields`) to ablate which channel carries the signal.

Run (GPU): `python main.py llm --run-name qwen_main --pool mean --epochs 6` (options: utils/arguments.py LLMArgs)
"""

import dataclasses
import json
import math
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer

from pertpred.data.task import Task, fit_context_mean, load_task
from pertpred.utils import config as C
from pertpred.utils.arguments import LLMArgs
from pertpred.utils.evaluate import save_predictions, score_split, val_selection_score

ALL_FIELDS = ("cell", "time", "dose", "name", "smiles")


def serialize(row, fields) -> str:
    """One condition -> prompt text. Unnamed compounds (BRD ids only) get no name line: a BRD id is a
    registration number, and including it would only let the model memorize training compounds."""
    lines = []
    if "cell" in fields:
        lines.append(f"cell line: {row.cell_id} ({row.primary_site})")
    if "time" in fields:
        lines.append(f"time: {row.time_h:g} h")
    if "dose" in fields:
        lines.append(f"dose: {row.dose_um:g} uM")
    if "name" in fields and row.is_named:
        lines.append(f"compound: {row.pert_iname}")
    if "smiles" in fields and isinstance(row.canonical_smiles, str):
        lines.append(f"smiles: {row.canonical_smiles}")
    return "\n".join(lines)


class QwenRegressor(torch.nn.Module):
    def __init__(
        self, n_out: int, mode: str = "lora", lora_r: int = 16, init: str = "pretrained", pool: str = "eos", n_layers: int = 0
    ):
        super().__init__()
        if init == "pretrained":
            backbone = AutoModel.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION, dtype=torch.float32)
        else:  # architecture-only control: same network, no pretraining
            cfg = AutoConfig.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION)
            backbone = AutoModel.from_config(cfg, dtype=torch.float32)
        if n_layers:  # keep only the first n decoder blocks (the final norm is kept and applied to block n)
            backbone.layers = backbone.layers[:n_layers]
            backbone.config.num_hidden_layers = n_layers
            if getattr(backbone.config, "layer_types", None):
                backbone.config.layer_types = backbone.config.layer_types[:n_layers]
        self.pool = pool
        if mode == "lora":
            lcfg = LoraConfig(
                r=lora_r,
                lora_alpha=2 * lora_r,
                lora_dropout=0.0,  # dropout here copies every adapted input; not worth the activation memory
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            backbone = get_peft_model(backbone, lcfg)
        elif mode == "frozen":
            backbone.requires_grad_(False)
        elif mode != "full":
            raise ValueError(mode)
        self.backbone = backbone
        d = backbone.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, n_out))
        torch.nn.init.zeros_(self.head[1].weight)
        torch.nn.init.zeros_(self.head[1].bias)

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        last = attention_mask.sum(dim=1) - 1  # right padding -> index of the appended EOS token
        rows = torch.arange(h.size(0), device=h.device)
        if self.pool == "eos":
            pooled = h[rows, last]
        else:  # mean over prompt tokens, excluding the appended EOS
            m = attention_mask.clone()
            m[rows, last] = 0
            pooled = (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        return self.head(pooled.float())


class Batcher:
    """Pre-tokenized prompts with dynamic right padding."""

    def __init__(self, texts: list[str], tok):
        enc = tok(texts, add_special_tokens=False)["input_ids"]
        self.ids = [e + [tok.eos_token_id] for e in enc]
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __call__(self, rows: np.ndarray, device):
        seqs = [self.ids[i] for i in rows]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), self.pad, dtype=torch.long)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
            mask[j, : len(s)] = 1
        return ids.to(device), mask.to(device)

    def lengths(self) -> np.ndarray:
        return np.array([len(s) for s in self.ids])


def length_grouped_batches(rows: np.ndarray, lengths: np.ndarray, batch_size: int, rng, mega: int = 50) -> list[np.ndarray]:
    """Shuffle, then sort by length inside windows of `mega` batches, so batches pad little but stay random."""
    rows = rng.permutation(rows)
    batches = []
    for s in range(0, len(rows), batch_size * mega):
        chunk = rows[s : s + batch_size * mega]
        chunk = chunk[np.argsort(lengths[chunk], kind="stable")]
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def micro_batches(rows: np.ndarray, lengths: np.ndarray, max_tokens: int) -> list[np.ndarray]:
    """Split a batch so each piece has at most `max_tokens` padded tokens (gradient accumulation)."""
    rows = rows[np.argsort(lengths[rows])]
    out, start = [], 0
    for end in range(1, len(rows) + 1):
        if end - start > 1 and (end - start) * lengths[rows[end - 1]] > max_tokens:
            out.append(rows[start : end - 1])
            start = end - 1
    out.append(rows[start:])
    return out


@torch.no_grad()
def predict_residual(model, batcher, rows: np.ndarray, device, bs: int = 256) -> np.ndarray:
    model.eval()
    order = np.argsort(batcher.lengths()[rows])  # length-sorted inference wastes less padding
    out = np.empty((len(rows), model.head[1].out_features), dtype=np.float32)
    for s in range(0, len(rows), bs):
        sel = order[s : s + bs]
        ids, mask = batcher(rows[sel], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out[sel] = model(ids, mask).float().cpu().numpy()
    return out


def run(args: LLMArgs) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task: Task = load_task()
    _, mu = fit_context_mean(task)
    fields = tuple(args.fields.split(","))
    assert set(fields) <= set(ALL_FIELDS), fields

    texts = [serialize(r, fields) for r in task.meta.itertuples()]
    tok = AutoTokenizer.from_pretrained(C.QWEN_MODEL, revision=C.QWEN_REVISION)
    batcher = Batcher(texts, tok)
    print(f"example prompt ({args.run_name}):\n{texts[task.idx('test')[0]]}\n---")
    print(f"prompt tokens: median={np.median(batcher.lengths()):.0f} max={batcher.lengths().max()}")

    tr, va, te = task.idx("train"), task.idx("val"), task.idx("test")
    if args.max_train_rows:
        tr = np.random.default_rng(args.seed).choice(tr, size=min(args.max_train_rows, len(tr)), replace=False)
    res = torch.tensor(task.Y - mu)  # residual targets, CPU

    model = QwenRegressor(
        task.Y.shape[1], mode=args.mode, lora_r=args.lora_r, init=args.init, pool=args.pool, n_layers=args.n_layers
    ).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {n_train / 1e6:.2f}M")
    head_params = list(model.head.parameters())
    body_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")]
    opt = torch.optim.AdamW(
        [{"params": body_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}], weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(tr) / args.batch_size)
    total = steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))
    )
    eval_every = max(1, int(steps_per_epoch * args.eval_every_epochs))

    log_dir = C.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # --eval-only appends to an existing run's log (keeping its training curve) and skips training:
    # it scores the best-on-val checkpoint saved so far, e.g. after a run was interrupted.
    log_f = open(log_dir / f"{args.run_name}.jsonl", "a" if args.eval_only else "w")
    log_f.write(json.dumps({"config": dataclasses.asdict(args), "n_train_rows": int(len(tr)), "trainable_params": n_train}) + "\n")
    ckpt = C.CKPT_DIR / f"{args.run_name}.pt"
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best, step, t0 = -np.inf, 0, time.time()
    rng = np.random.default_rng(args.seed)
    lengths = batcher.lengths()
    for epoch in range(0 if args.eval_only else args.epochs):
        for rows in length_grouped_batches(tr, lengths, args.batch_size, rng):
            model.train()
            opt.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for mb in micro_batches(rows, lengths, args.max_tokens):
                ids, mask = batcher(mb, device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    pred = model(ids, mask)
                # Weight by micro-batch share so the accumulated gradient equals the full-batch mean.
                loss = torch.nn.functional.mse_loss(pred.float(), res[mb].to(device)) * (len(mb) / len(rows))
                loss.backward()
                loss_sum += loss.item()
            loss = torch.tensor(loss_sum)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                rate = step * args.batch_size / (time.time() - t0)
                print(f"step {step}/{total} epoch {epoch} loss {loss.item():.4f} ({rate:.0f} rows/s)", flush=True)
                log_f.write(json.dumps({"step": step, "loss": loss.item()}) + "\n")
            if step % eval_every == 0 or step == total:
                v_pred = mu[va] + predict_residual(model, batcher, va, device)
                v = val_selection_score(task, mu, va, v_pred)
                v_mse = float(((v_pred - task.Y[va]) ** 2).mean())
                print(f"  [val] step {step} centered_pearson={v:.4f} mse={v_mse:.4f}", flush=True)
                log_f.write(json.dumps({"step": step, "val_centered_pearson": v, "val_mse": v_mse}) + "\n")
                log_f.flush()
                if v > best:
                    best = v
                    trainable = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
                    torch.save({"state": trainable, "step": step, "val": v}, ckpt)

    # Reload the best-on-val trainable weights, then predict val + test once.
    state = torch.load(ckpt)
    missing = model.load_state_dict(state["state"], strict=False)
    assert not missing.unexpected_keys
    print(f"best val centered_pearson={state['val']:.4f} at step {state['step']}")
    q = np.concatenate([va, te])
    pred = mu[q] + predict_residual(model, batcher, q, device)
    save_predictions(args.run_name, q, pred)
    for split in ("val", "test"):
        score_split(args.run_name, task, mu, q, pred, split)
    log_f.write(json.dumps({"best_step": state["step"], "best_val_centered_pearson": state["val"], "minutes": (time.time() - t0) / 60}) + "\n")
    log_f.close()
