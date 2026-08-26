"""The small two-layer neural network you fine-tune -- written out in numpy.

It is deliberately tiny so nothing is hidden:

    features F  --(W1)-->  hidden  --tanh-->  --(W2)-->  scores  --softmax-->  probabilities
       (dim)              (hidden)                       (classes)

W1 is the "backbone": a big matrix that turns raw word-counts into richer features.
In a real project this is the giant pretrained network you downloaded. Here we start
it from a fixed random seed and treat it as our "pretrained" starting point.

W2 is the "head": a small matrix that reads the features and picks a class.

That split matters for the whole project:
  * PROMPT / RAG baselines don't touch either matrix (no training at all).
  * "Base" (head-only) training adjusts ONLY the small head W2.
  * FULL fine-tuning trains the big W1 and the head W2 -- lots of parameters.
  * LoRA freezes W1 and trains a tiny low-rank patch for it (see lora.py) -- almost
    the same result for a fraction of the parameters.

This file just defines the network and how to run it forward. The actual training
loops live in train.py, so you can read "what the model is" separately from "how it
learns".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import CLASSES


def softmax(scores: np.ndarray) -> np.ndarray:
    """Turn raw scores into probabilities that sum to 1 (row-wise), stably."""
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


@dataclass
class Forward:
    """What one forward pass produced -- kept so backprop can reuse it."""

    hidden: np.ndarray   # tanh(F @ W1_effective), shape (n, hidden)
    probs: np.ndarray    # softmax(hidden @ W2), shape (n, n_classes)


class TwoLayerNet:
    """features -> hidden (W1, frozen-able) -> class probabilities (W2, the head)."""

    def __init__(self, dim: int, hidden: int = 64, n_classes: int = len(CLASSES),
                 seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        # W1 is our stand-in for a pretrained backbone: fixed random features.
        self.W1 = rng.standard_normal((dim, hidden)) / np.sqrt(dim)
        self.W2 = np.zeros((hidden, n_classes))  # head starts blank; training fills it
        self.dim = dim
        self.hidden = hidden
        self.n_classes = n_classes

    # --- running the network forward -----------------------------------------
    def forward(self, features: np.ndarray, delta_W1: np.ndarray | None = None) -> Forward:
        """Run features through the net. `delta_W1` is LoRA's patch added to W1, if any."""
        effective_W1 = self.W1 if delta_W1 is None else (self.W1 + delta_W1)
        hidden = np.tanh(features @ effective_W1)
        probs = softmax(hidden @ self.W2)
        return Forward(hidden=hidden, probs=probs)

    def predict(self, features: np.ndarray, delta_W1: np.ndarray | None = None) -> np.ndarray:
        """Return the predicted class index for each row."""
        return self.forward(features, delta_W1).probs.argmax(axis=1)

    def accuracy(self, features: np.ndarray, labels: list[int] | np.ndarray,
                 delta_W1: np.ndarray | None = None) -> float:
        """Fraction of rows predicted correctly."""
        preds = self.predict(features, delta_W1)
        labels = np.asarray(labels)
        return float((preds == labels).mean())

    def copy(self) -> "TwoLayerNet":
        """A deep copy, so an experiment can't mutate the original weights."""
        clone = TwoLayerNet.__new__(TwoLayerNet)
        clone.W1 = self.W1.copy()
        clone.W2 = self.W2.copy()
        clone.dim, clone.hidden, clone.n_classes = self.dim, self.hidden, self.n_classes
        return clone

    def param_count(self) -> int:
        """Total number of weights in the network (W1 + W2)."""
        return self.W1.size + self.W2.size


def cross_entropy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Average negative-log-likelihood of the correct class -- the loss we minimise."""
    n = len(labels)
    correct = probs[np.arange(n), labels]
    return float(-np.log(np.clip(correct, 1e-12, 1.0)).mean())
