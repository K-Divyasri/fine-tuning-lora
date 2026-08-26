"""The training loops -- the "how the model learns" half of the project.

Training is a loop of four steps, repeated until the model is good:

    1. forward   : run the data through the net to get predicted probabilities
    2. loss      : measure how wrong they are (cross-entropy)
    3. backward  : work out which way to nudge each weight to reduce the loss (gradients)
    4. step      : nudge the weights a little in that direction (learning rate)

We do it three ways, and comparing them is the whole lesson:

    fit_head_only  train ONLY the small head W2 (the frozen backbone stays fixed).
                   This is our "base" -- minimal adaptation.
    fit_full       train the big backbone W1 AND the head W2. Most parameters, best fit.
    fit_lora       freeze W1, train a small LoRA patch for it PLUS the head. Almost
                   the quality of full fine-tuning for a fraction of the parameters.

The maths is plain numpy backprop through one tanh layer. You do not need to follow
every line -- but it's all here, and the notebooks walk through it slowly.
"""

from __future__ import annotations

import numpy as np

from .lora import LoRA
from .model import TwoLayerNet, cross_entropy, softmax


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[labels]


def _epoch_grads_head(net: TwoLayerNet, hidden: np.ndarray, Y: np.ndarray,
                      probs: np.ndarray) -> np.ndarray:
    """Gradient of the loss w.r.t. the head W2."""
    n = Y.shape[0]
    return hidden.T @ (probs - Y) / n


def fit_head_only(net: TwoLayerNet, features: np.ndarray, labels: list[int],
                  *, epochs: int = 300, lr: float = 0.5) -> TwoLayerNet:
    """Train only the head W2 on a frozen backbone. Mutates and returns `net`."""
    labels = np.asarray(labels)
    Y = _one_hot(labels, net.n_classes)
    hidden = np.tanh(features @ net.W1)  # backbone is frozen, so compute once
    for _ in range(epochs):
        probs = softmax(hidden @ net.W2)
        net.W2 -= lr * _epoch_grads_head(net, hidden, Y, probs)
    return net


def fit_full(net: TwoLayerNet, features: np.ndarray, labels: list[int],
             *, epochs: int = 500, lr: float = 0.5) -> TwoLayerNet:
    """Full fine-tune: train the backbone W1 and the head W2. Mutates and returns `net`."""
    labels = np.asarray(labels)
    Y = _one_hot(labels, net.n_classes)
    n = len(labels)
    for _ in range(epochs):
        hidden = np.tanh(features @ net.W1)
        probs = softmax(hidden @ net.W2)
        d_scores = (probs - Y) / n                     # dL/d(hidden@W2)
        grad_W2 = hidden.T @ d_scores
        d_hidden = (d_scores @ net.W2.T) * (1 - hidden ** 2)  # through tanh
        grad_W1 = features.T @ d_hidden
        net.W1 -= lr * grad_W1
        net.W2 -= lr * grad_W2
    return net


def fit_lora(net: TwoLayerNet, features: np.ndarray, labels: list[int],
             *, rank: int = 4, epochs: int = 500, lr: float = 0.5,
             seed: int = 0) -> LoRA:
    """LoRA fine-tune: freeze W1, train a low-rank patch + the head. Returns the LoRA.

    The backbone `net.W1` is never modified. We train the patch (A, B) and the head
    W2 together. To use the result, call net.forward(F, delta_W1=lora.delta()).
    """
    labels = np.asarray(labels)
    Y = _one_hot(labels, net.n_classes)
    n = len(labels)
    lora = LoRA.init(net.dim, net.hidden, rank=rank, seed=seed)
    for _ in range(epochs):
        delta = lora.delta()
        hidden = np.tanh(features @ (net.W1 + delta))
        probs = softmax(hidden @ net.W2)
        d_scores = (probs - Y) / n
        grad_W2 = hidden.T @ d_scores
        d_hidden = (d_scores @ net.W2.T) * (1 - hidden ** 2)
        grad_delta = features.T @ d_hidden             # gradient w.r.t. the whole patch
        # Chain the patch gradient into A and B (delta = scaling * B @ A).
        grad_B = lora.scaling * (grad_delta @ lora.A.T)
        grad_A = lora.scaling * (lora.B.T @ grad_delta)
        lora.B -= lr * grad_B
        lora.A -= lr * grad_A
        net.W2 -= lr * grad_W2                          # the head is trained in both cases
    return lora


def loss_curve(net: TwoLayerNet, features: np.ndarray, labels: list[int],
               *, epochs: int = 500, lr: float = 0.5) -> list[float]:
    """Full fine-tune while recording the loss each epoch -- for the 'watch it learn' plot."""
    labels = np.asarray(labels)
    Y = _one_hot(labels, net.n_classes)
    n = len(labels)
    history: list[float] = []
    for _ in range(epochs):
        hidden = np.tanh(features @ net.W1)
        probs = softmax(hidden @ net.W2)
        history.append(cross_entropy(probs, labels))
        d_scores = (probs - Y) / n
        d_hidden = (d_scores @ net.W2.T) * (1 - hidden ** 2)
        net.W1 -= lr * (features.T @ d_hidden)
        net.W2 -= lr * (hidden.T @ d_scores)
    return history
