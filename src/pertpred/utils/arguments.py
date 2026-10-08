"""Every command's options, as dataclasses. `main.py` (repo root) turns them into the command-line interface.

One dataclass per command; each field is one `--option` (underscores become dashes, e.g.
`run_name` -> `--run-name`). Defaults are the settings behind the reported results; the exact runs
are listed in `scripts/llm_runs.sh`. To add an option: add a field here, then read it from `args`
in the command's `run(args)`.
"""

import argparse
import dataclasses
import typing
from dataclasses import dataclass, field

from pertpred.utils.config import SEED


def opt(default, help: str, **argparse_kwargs):
    """A dataclass field that carries its command-line help (and e.g. `choices=` / `required=`)."""
    return field(default=default, metadata={"help": help, **argparse_kwargs})


# ------------------------------------------------------------------ data


@dataclass
class SplitArgs:
    """Compound-grouped train / val / test split (pertpred.data.split)."""

    val_frac: float = opt(0.10, "fraction of compound groups held out for validation")
    test_frac: float = opt(0.15, "fraction of compound groups held out for test")
    seed: int = opt(SEED, "seed of the group shuffle")


# ------------------------------------------------------------------ models


@dataclass
class BaselineArgs:
    """Non-LLM baselines (pertpred.models.baselines)."""

    knn_ks: tuple[int, ...] = opt((1, 3, 5, 10, 20, 50), "neighbour counts tried on val for knn_tanimoto")
    ridge_alphas: tuple[float, ...] = opt((10.0, 100.0, 1000.0, 10000.0), "ridge penalties tried on val for ridge_fp")
    mlp_epochs: int = opt(40, "maximum epochs for mlp_fp (best epoch chosen on val)")
    seed: int = opt(SEED, "seed for mlp_fp")
    seed_variance: bool = opt(False, "instead of the baselines, retrain mlp_fp with seeds 1-4 and log the spread")


@dataclass
class ProbeArgs:
    """Frozen-Qwen embeddings through the baseline MLP (pertpred.models.probe)."""

    stage: str = opt(
        "all", "all: embed, pick pooling on val, text variants, Morgan concat; fp_concat: concat only, from cached embeddings",
        choices=("all", "fp_concat"),
    )


@dataclass
class LLMArgs:
    """Qwen2.5-0.5B regressor: text prompt -> pooled hidden state -> 978-gene head (pertpred.models.llm)."""

    run_name: str = opt(None, "name for logs, checkpoint and predictions", required=True)
    fields: str = opt("cell,time,dose,name,smiles", "comma-separated prompt lines to include (ablations drop some)")
    mode: str = opt("lora", "lora adapters, all weights (full), or frozen backbone", choices=("lora", "full", "frozen"))
    init: str = opt("pretrained", "pretrained weights or an architecture-only random init", choices=("pretrained", "random"))
    lora_r: int = opt(16, "LoRA rank")
    pool: str = opt("eos", "read-out: hidden state at the appended EOS token, or mean over prompt tokens", choices=("eos", "mean"))
    n_layers: int = opt(0, "truncate the backbone to its first N blocks (0 = all 24)")
    epochs: int = opt(2, "training epochs")
    batch_size: int = opt(64, "rows per optimizer step")
    max_tokens: int = opt(6144, "padded-token cap per micro-batch (memory); gradients are accumulated")
    lr: float = opt(2e-4, "learning rate of the backbone (LoRA or full weights)")
    head_lr: float = opt(1e-3, "learning rate of the regression head")
    eval_every_epochs: float = opt(0.25, "evaluate on val every this many epochs")
    max_train_rows: int = opt(0, "subsample training rows (smoke tests; 0 = all)")
    eval_only: bool = opt(False, "skip training; score the run's saved best-on-val checkpoint")
    seed: int = opt(SEED, "random seed")


@dataclass
class GenArgs:
    """Cell2Sentence-style generative variant: Qwen writes the top-25 up / down genes (pertpred.models.gen)."""

    run_name: str = opt(None, "name for logs, checkpoint, generations and predictions", required=True)
    epochs: int = opt(2, "training epochs")
    batch_size: int = opt(64, "rows per optimizer step")
    max_tokens: int = opt(4096, "padded-token cap per micro-batch (memory); gradients are accumulated")
    lr: float = opt(2e-4, "LoRA learning rate")
    lora_r: int = opt(16, "LoRA rank")
    eval_every_epochs: float = opt(0.25, "compute val token loss every this many epochs")
    val_loss_rows: int = opt(2000, "val rows used for the token-loss checkpoint criterion")
    max_train_rows: int = opt(0, "subsample training rows (smoke tests; 0 = all)")
    gen_limit: int = opt(0, "generate only this many val and test rows (smoke tests; 0 = all)")
    gen_batch: int = opt(128, "prompts per generation batch")
    eval_only: bool = opt(False, "skip training; generate from the saved checkpoint")
    seed: int = opt(SEED, "random seed")


# ------------------------------------------------------------------ evaluation


@dataclass
class EvalArgs:
    """Tables, paired comparisons and plots from the scored predictions (pertpred.utils.evaluate)."""

    split: str = opt("test", "which split to report", choices=("val", "test"))
    reference: str = opt("mlp_fp", "model every other model is paired against")
    rescore: bool = opt(False, "recompute every per-signature table from saved predictions first")


# ------------------------------------------------------------------ dataclass <-> argparse


def add_arguments(parser: argparse.ArgumentParser, cls) -> None:
    """One `--option` per dataclass field: bools become flags, tuples take one or more values."""
    for f in dataclasses.fields(cls):
        meta = dict(f.metadata)
        help_ = meta.pop("help", "")
        flag = "--" + f.name.replace("_", "-")
        if f.type is bool:
            parser.add_argument(flag, dest=f.name, action="store_true", help=help_)
        elif typing.get_origin(f.type) is tuple:
            parser.add_argument(flag, dest=f.name, type=typing.get_args(f.type)[0], nargs="+", default=f.default, help=help_, **meta)
        else:
            parser.add_argument(flag, dest=f.name, type=f.type, default=f.default, help=help_, **meta)


def build(cls, namespace: argparse.Namespace):
    """Parsed namespace -> dataclass instance (lists become tuples so instances stay hashable)."""
    values = {f.name: getattr(namespace, f.name) for f in dataclasses.fields(cls)}
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items()})
