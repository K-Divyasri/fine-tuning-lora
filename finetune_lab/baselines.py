"""The two rivals fine-tuning has to beat: prompt rules and RAG. No training in either.

To claim fine-tuning is worth it, you have to compare it against the cheaper options
on the SAME task. These are those options, done honestly:

  PromptClassifier   Stands in for "prompt engineering": a human writes a few obvious
                     keywords per class and you classify by which list matches most.
                     No data, no training. Its weakness here is real -- the tickets use
                     many synonyms the hand-written list never thought of.

  RagClassifier      Stands in for RAG (retrieval): keep all the labelled training
                     examples, and to classify a new ticket, find the most similar
                     stored tickets and take a vote of their labels (k-nearest-
                     neighbours). No weights are trained; it just looks things up.

Comparing these to the fine-tuned model is the interview-grade lesson of the project:
prompt is cheapest and weakest, RAG needs no training but must store every example,
fine-tuning needs labelled data and a training run but wins on a fixed task.
"""

from __future__ import annotations

import numpy as np

from . import CLASSES
from .dataset import PROMPT_KEYWORDS
from .features import tokenize


class PromptClassifier:
    """Classify by hand-written keywords -- the 'just prompt it' baseline."""

    def __init__(self, keywords: dict[str, list[str]] | None = None) -> None:
        self.keywords = keywords or PROMPT_KEYWORDS

    def predict(self, texts: list[str]) -> np.ndarray:
        preds = []
        for text in texts:
            toks = set(tokenize(text))
            scores = [sum(kw in toks for kw in self.keywords[c]) for c in CLASSES]
            # If nothing matches, fall back to class 0 (a real system would 'refuse').
            preds.append(int(np.argmax(scores)) if max(scores) > 0 else 0)
        return np.array(preds)

    def accuracy(self, texts: list[str], labels: list[int]) -> float:
        return float((self.predict(texts) == np.asarray(labels)).mean())


class RagClassifier:
    """Classify by nearest labelled neighbours -- the 'retrieval' baseline (kNN)."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def fit(self, train_features: np.ndarray, train_labels: list[int]) -> "RagClassifier":
        """'Training' here is just remembering the examples -- no weights change."""
        self._features = train_features
        self._labels = np.asarray(train_labels)
        return self

    def predict(self, query_features: np.ndarray) -> np.ndarray:
        if self._features is None:
            raise RuntimeError("Call fit() with the training examples first.")
        sims = query_features @ self._features.T   # cosine (rows are unit vectors)
        preds = []
        for row in sims:
            nearest = self._labels[np.argsort(-row)[:self.k]]
            preds.append(int(np.bincount(nearest, minlength=len(CLASSES)).argmax()))
        return np.array(preds)

    def accuracy(self, query_features: np.ndarray, labels: list[int]) -> float:
        return float((self.predict(query_features) == np.asarray(labels)).mean())
