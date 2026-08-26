"""Make the synthetic support-ticket dataset and split it train / val / test.

Why synthetic? Because we want a task where the RIGHT answer is knowable (so we can
score honestly) and where fine-tuning has something real to learn. Each ticket is
built from a class's vocabulary plus shared filler words, and -- crucially -- 40% of
tickets also borrow a word or two from another class. That overlap is what makes the
task realistic and non-trivial: a human's hand-written keyword rules will trip over
it, but a model trained on labelled examples can learn the subtler weighting.

The three splits do different jobs, and keeping them separate is the single most
important discipline in all of machine learning:
    train  the model learns from these
    val    you peek at these while tuning settings (how long to train, the rank...)
    test   you touch these ONCE, at the very end, to get an honest score

If you tune against the test set, you've cheated and your number is a lie. The whole
project is built to make that separation obvious.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from . import CLASSES, DEFAULT_SEED

# The rich vocabulary tickets are actually generated from: lots of synonyms, and
# deliberate overlap (e.g. "login"/"password"/"reset" belong to BOTH technical and
# account; "address" to both account and shipping). This overlap is the difficulty.
GENERATIVE_VOCAB: dict[str, list[str]] = {
    "billing": ["charge", "charged", "refund", "invoice", "payment", "pay", "price",
                "subscription", "card", "receipt", "overcharged", "fee", "billed",
                "cost", "plan", "renew", "discount", "transaction", "coupon", "bill"],
    "technical": ["error", "crash", "bug", "broken", "freeze", "loading", "glitch",
                  "500", "blank", "stuck", "login", "password", "reset", "sync",
                  "update", "install", "version", "screen", "fails", "timeout"],
    "account": ["account", "profile", "email", "username", "settings", "deactivate",
                "verify", "permissions", "login", "password", "reset", "name",
                "address", "update", "preferences", "access", "locked", "2fa", "phone"],
    "shipping": ["package", "delivery", "tracking", "shipment", "arrive", "late",
                 "courier", "lost", "delayed", "address", "box", "dispatch",
                 "warehouse", "return", "damaged", "parcel", "eta", "carrier", "order"],
}

FILLER = ["the", "a", "my", "please", "help", "hi", "thanks", "cannot", "need",
          "with", "for", "and", "is", "to", "today", "again", "really", "still", "now"]

# What a person would HAND-WRITE as prompt keywords: just the obvious few per class.
# It misses most of the synonyms above -- which is exactly why the prompt baseline is
# weak, and why fine-tuning (which sees labels) can do better.
PROMPT_KEYWORDS: dict[str, list[str]] = {
    "billing": ["charge", "refund", "payment", "invoice"],
    "technical": ["error", "crash", "bug", "broken"],
    "account": ["account", "profile", "password", "settings"],
    "shipping": ["package", "delivery", "tracking", "shipment"],
}


@dataclass
class Dataset:
    """A split of the data: the ticket texts and their integer labels (0..3)."""

    texts: list[str]
    labels: list[int]

    def __len__(self) -> int:
        return len(self.texts)


def _make_ticket(class_index: int, rng: random.Random) -> str:
    """Build one ticket for a class: its own words + filler + sometimes a stray word."""
    cls = CLASSES[class_index]
    vocab = GENERATIVE_VOCAB[cls]
    words = [rng.choice(vocab) for _ in range(rng.randint(3, 5))]
    words += [rng.choice(FILLER) for _ in range(rng.randint(4, 7))]
    if rng.random() < 0.40:  # overlap/ambiguity: borrow from another class
        other = rng.choice([c for c in CLASSES if c != cls])
        words += [rng.choice(GENERATIVE_VOCAB[other]) for _ in range(rng.randint(1, 2))]
    rng.shuffle(words)
    return " ".join(words)


def make_split(n_per_class: int, seed: int) -> Dataset:
    """Create a balanced dataset with `n_per_class` tickets for each of the 4 classes."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[int] = []
    for class_index in range(len(CLASSES)):
        for _ in range(n_per_class):
            texts.append(_make_ticket(class_index, rng))
            labels.append(class_index)
    # Shuffle so the classes aren't in blocks.
    order = list(range(len(texts)))
    rng.shuffle(order)
    return Dataset([texts[i] for i in order], [labels[i] for i in order])


def make_splits(
    n_train: int = 80,
    n_val: int = 20,
    n_test: int = 30,
    seed: int = DEFAULT_SEED,
) -> tuple[Dataset, Dataset, Dataset]:
    """Return (train, val, test), each built from a DIFFERENT seed so they don't overlap."""
    train = make_split(n_train, seed=seed)
    val = make_split(n_val, seed=seed + 1)
    test = make_split(n_test, seed=seed + 2)
    return train, val, test
