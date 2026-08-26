"""LoRA -- Low-Rank Adaptation -- built by hand so it stops being magic.

The idea in one breath: to adapt a big frozen weight matrix W (shape dim x hidden)
you do NOT retrain all of it. You leave W exactly as it is and learn a small "patch"
that gets added to it. The patch is forced to be low-rank -- it's the product of two
skinny matrices:

    patch  =  B @ A          B is (dim x r),  A is (r x hidden),   r is tiny (e.g. 4)

So the effective weight the network uses is  W + B@A, but the only things you TRAIN
are A and B. Count the parameters:

    full W        : dim x hidden           (e.g. 512 x 64 = 32,768)
    LoRA A and B  : r x (dim + hidden)      (e.g. 4 x (512 + 64) = 2,304)

That's why LoRA is cheap: you train a tiny fraction of the weights and get almost the
same result. On real LLMs this is the difference between needing a data-centre and
fine-tuning on a single free GPU.

Two standard details, both here:
  * B starts at ZERO, so at the very first step the patch is 0 and the model behaves
    exactly like the untouched pretrained one -- training only ever nudges it.
  * A `scaling` factor (alpha / r) lets you turn the patch's strength up or down
    without retraining; we keep it simple with alpha = r (scaling = 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoRA:
    """A low-rank patch (A, B) for one weight matrix of shape (dim x hidden)."""

    A: np.ndarray        # (rank, hidden)
    B: np.ndarray        # (dim, rank)
    rank: int
    scaling: float = 1.0

    @classmethod
    def init(cls, dim: int, hidden: int, rank: int = 4, alpha: float | None = None,
             seed: int = 0) -> "LoRA":
        """Create a fresh patch: A small random, B zero (so it starts as a no-op)."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((rank, hidden)) * 0.01
        B = np.zeros((dim, rank))
        scaling = 1.0 if alpha is None else alpha / rank
        return cls(A=A, B=B, rank=rank, scaling=scaling)

    def delta(self) -> np.ndarray:
        """The actual patch added to W: scaling * (B @ A), shape (dim x hidden)."""
        return self.scaling * (self.B @ self.A)

    def n_params(self) -> int:
        """How many numbers we actually train (A plus B)."""
        return self.A.size + self.B.size

    def merge_into(self, W: np.ndarray) -> np.ndarray:
        """Return W + patch. 'Merging' is how you ship a LoRA: fold it back into W."""
        return W + self.delta()
