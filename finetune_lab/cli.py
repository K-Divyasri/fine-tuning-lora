"""Command line: `python -m finetune_lab ...`.

    python -m finetune_lab compare              # the full comparison table
    python -m finetune_lab predict "my card was charged twice"
"""

from __future__ import annotations

import argparse

from . import CLASSES, DEFAULT_SEED
from .dataset import make_splits
from .evaluate import compare_all, format_table
from .features import HashedVectorizer
from .model import TwoLayerNet
from .train import fit_full


def cmd_compare(args: argparse.Namespace) -> int:
    train, _val, test = make_splits(seed=args.seed)
    results = compare_all(train, test, lora_rank=args.rank, seed=args.seed)
    print(format_table(results))
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    train, _val, _test = make_splits(seed=args.seed)
    vectorizer = HashedVectorizer()
    net = TwoLayerNet(dim=vectorizer.dim, seed=args.seed)
    fit_full(net, vectorizer.transform(train.texts), train.labels)
    features = vectorizer.transform([args.text])
    pred = net.predict(features)[0]
    print(f"'{args.text}' -> {CLASSES[pred]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune_lab",
        description="Fine-tune a tiny classifier from scratch and compare it to prompt/RAG baselines.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_compare = sub.add_parser("compare", help="train all methods and print the comparison table")
    p_compare.add_argument("--rank", type=int, default=4, help="LoRA rank")
    p_compare.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_compare.set_defaults(func=cmd_compare)

    p_predict = sub.add_parser("predict", help="classify one ticket with a fully fine-tuned model")
    p_predict.add_argument("text", help="the ticket text, in quotes")
    p_predict.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_predict.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
