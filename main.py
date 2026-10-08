"""Single entry point: `uv run python main.py <command> [--options]`.

Pipeline, in order:
    prepare    decompress Level 5, extract the 978 landmark genes          (CPU, ~2 min)
    split      compound-grouped split + Morgan fingerprints                (CPU, ~10 s)
    eda        data diagnostics -> results/eda/                            (CPU, ~30 s)
    baselines  zero, context mean, kNN, ridge, MLP on fingerprints         (GPU optional, ~3 min)
    probe      frozen Qwen embeddings through the baseline MLP             (GPU, ~10 min)
    llm        Qwen2.5-0.5B regressor (LoRA / all weights / frozen)        (GPU, ~45 min per run)
    gen        Cell2Sentence-style generative variant                      (GPU, ~1.5 h)
    evaluate   tables, paired tests, plots -> results/report/              (CPU, ~2 min)

`python main.py <command> --help` lists a command's options; they are defined as dataclasses in
src/pertpred/utils/arguments.py. Every command module exposes `run(args)`.
"""

import argparse
import importlib

from pertpred.utils.arguments import BaselineArgs, EvalArgs, GenArgs, LLMArgs, ProbeArgs, SplitArgs, add_arguments, build

# command -> (module exposing run(args), options dataclass or None, one-line help)
COMMANDS = {
    "prepare": ("pertpred.data.load", None, "decompress Level 5 and extract the 978 landmark genes"),
    "split": ("pertpred.data.split", SplitArgs, "compound-grouped train / val / test split and fingerprints"),
    "eda": ("pertpred.data.eda", None, "data diagnostics: noise floor, replicate ceiling, context effects"),
    "baselines": ("pertpred.models.baselines", BaselineArgs, "non-LLM baselines (zero, context mean, kNN, ridge, MLP)"),
    "probe": ("pertpred.models.probe", ProbeArgs, "frozen-Qwen embeddings through the baseline MLP"),
    "llm": ("pertpred.models.llm", LLMArgs, "train and score a Qwen regressor"),
    "gen": ("pertpred.models.gen", GenArgs, "train and score the Cell2Sentence-style generative variant"),
    "evaluate": ("pertpred.utils.evaluate", EvalArgs, "build the report from all scored predictions"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for name, (_, args_cls, help_) in COMMANDS.items():
        sub = commands.add_parser(name, help=help_, description=help_, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        if args_cls is not None:
            add_arguments(sub, args_cls)
    namespace = parser.parse_args(argv)

    module, args_cls, _ = COMMANDS[namespace.command]
    args = build(args_cls, namespace) if args_cls is not None else None
    importlib.import_module(module).run(args)  # imported lazily: `--help` never loads torch


if __name__ == "__main__":
    main()
