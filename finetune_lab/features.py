"""Turn ticket text into number vectors, because models do maths, not language.

Every model in this project (and every real one) works on numbers, so first we turn
each ticket into a fixed-length vector. We use "feature hashing" (the hashing trick):
run each word through a hash function to pick one of `dim` slots, and count how often
each slot is hit. It needs no vocabulary file and handles words it has never seen.

We use hashlib.md5 for the hash, NOT Python's built-in hash(), because hash() of a
string is randomised per run for security -- it would give different vectors every
time and make results impossible to reproduce. md5 is identical on every machine and
every run, which is exactly what we want.

Vectors are L2-normalised (scaled to length 1) so that a long ticket and a short one
are compared on their word MIX, not their length.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case and split into word/number tokens."""
    return _TOKEN_RE.findall(text.lower())


def _bucket(token: str, dim: int) -> int:
    """Stable hash of a word into one of `dim` buckets (md5, so it's reproducible)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim


class HashedVectorizer:
    """Text -> a fixed-length, L2-normalised bag-of-words vector."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def transform_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for token in tokenize(text):
            vec[_bucket(token, self.dim)] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def transform(self, texts: list[str]) -> np.ndarray:
        """Stack a list of texts into a (n_texts, dim) matrix, one row per text."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.vstack([self.transform_one(t) for t in texts])
