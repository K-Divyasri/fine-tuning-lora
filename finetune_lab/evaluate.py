"""Score every method honestly on the SAME held-out test set, and build the comparison table.

This file is where the project's actual claim gets checked: does fine-tuning beat
the cheaper alternatives? Every method here is scored on the test split only, which
none of them has ever seen during training -- that's what makes the numbers honest.

`compare_all` runs prompt, RAG, head-only, full fine-tune, and LoRA on the same
train/val/test split and returns one table: accuracy and trainable-parameter count
for each. That table is the deliverable -- it's what goes in the README and what an
interviewer will ask you to explain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import PromptClassifier, RagClassifier
from .dataset import Dataset
from .features import HashedVectorizer
from .model import TwoLayerNet
from .train import fit_full, fit_head_only, fit_lora


@dataclass
class MethodResult:
    """One row of the comparison table."""

    name: str
    test_accuracy: float
    trainable_params: int
    needs_training: bool


def per_class_accuracy(predictions: np.ndarray, labels: np.ndarray, classes: tuple[str, ...]) -> dict[str, float]:
    """Accuracy broken down by true class -- shows WHICH classes a method struggles on."""
    out: dict[str, float] = {}
    labels = np.asarray(labels)
    for i, name in enumerate(classes):
        mask = labels == i
        out[name] = float((predictions[mask] == labels[mask]).mean()) if mask.any() else float("nan")
    return out


def compare_all(
    train: Dataset,
    test: Dataset,
    *,
    dim: int = 512,
    hidden: int = 64,
    lora_rank: int = 4,
    seed: int = 0,
) -> list[MethodResult]:
    """Train/evaluate every method on the same split and return the comparison table."""
    vectorizer = HashedVectorizer(dim=dim)
    F_train = vectorizer.transform(train.texts)
    F_test = vectorizer.transform(test.texts)

    results: list[MethodResult] = []

    # 1. Prompt -- no training, no data needed.
    prompt = PromptClassifier()
    results.append(MethodResult(
        "prompt (hand keywords)", prompt.accuracy(test.texts, test.labels), 0, False,
    ))

    # 2. RAG -- "trains" by storing examples, not by adjusting weights.
    rag = RagClassifier(k=5).fit(F_train, train.labels)
    results.append(MethodResult(
        "RAG (kNN over labelled examples)", rag.accuracy(F_test, test.labels), 0, False,
    ))

    # 3. Base: frozen backbone, head only.
    net_base = TwoLayerNet(dim=dim, hidden=hidden, seed=seed)
    fit_head_only(net_base, F_train, train.labels)
    results.append(MethodResult(
        "base (frozen backbone, head only)",
        net_base.accuracy(F_test, test.labels),
        net_base.W2.size,
        True,
    ))

    # 4. Full fine-tune: same starting point, but train everything.
    net_full = TwoLayerNet(dim=dim, hidden=hidden, seed=seed)
    fit_full(net_full, F_train, train.labels)
    results.append(MethodResult(
        "full fine-tune (all weights)",
        net_full.accuracy(F_test, test.labels),
        net_full.param_count(),
        True,
    ))

    # 5. LoRA: same starting point, freeze W1, train only a low-rank patch + head.
    net_lora = TwoLayerNet(dim=dim, hidden=hidden, seed=seed)
    lora = fit_lora(net_lora, F_train, train.labels, rank=lora_rank, seed=seed)
    lora_acc = net_lora.accuracy(F_test, test.labels, delta_W1=lora.delta())
    results.append(MethodResult(
        f"LoRA fine-tune (rank={lora_rank})",
        lora_acc,
        lora.n_params() + net_lora.W2.size,
        True,
    ))

    return results


def format_table(results: list[MethodResult]) -> str:
    """Render the comparison as a plain-text table, widest column first for readability."""
    full_params = next(r.trainable_params for r in results if r.name.startswith("full"))
    lines = [f"{'method':38s} {'test acc':>9s} {'params':>10s} {'vs full':>10s}"]
    lines.append("-" * len(lines[0]))
    for r in results:
        ratio = f"{full_params / r.trainable_params:.1f}x" if r.trainable_params else "-"
        lines.append(f"{r.name:38s} {r.test_accuracy:9.3f} {r.trainable_params:10d} {ratio:>10s}")
    return "\n".join(lines)
