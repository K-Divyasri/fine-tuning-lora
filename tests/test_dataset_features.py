"""Tests for the dataset generator, the splits, and the feature hasher."""

from __future__ import annotations

import numpy as np

from finetune_lab import CLASSES
from finetune_lab.dataset import make_split, make_splits
from finetune_lab.features import HashedVectorizer, tokenize


def test_make_split_is_balanced():
    ds = make_split(10, seed=1)
    assert len(ds) == 10 * len(CLASSES)
    # Exactly n_per_class of each label.
    counts = [ds.labels.count(i) for i in range(len(CLASSES))]
    assert counts == [10] * len(CLASSES)


def test_make_split_is_deterministic():
    a = make_split(5, seed=42)
    b = make_split(5, seed=42)
    assert a.texts == b.texts
    assert a.labels == b.labels


def test_make_splits_train_val_test_dont_overlap():
    train, val, test = make_splits(n_train=10, n_val=5, n_test=5, seed=1)
    assert set(train.texts).isdisjoint(set(test.texts))
    assert set(val.texts).isdisjoint(set(test.texts))


def test_tokenize_lowercases_and_splits():
    assert tokenize("Charge ERROR-500") == ["charge", "error", "500"]


def test_hashed_vectorizer_is_deterministic():
    vec = HashedVectorizer(dim=64)
    a = vec.transform_one("my card was charged twice")
    b = vec.transform_one("my card was charged twice")
    assert np.array_equal(a, b)


def test_hashed_vectorizer_unit_length():
    vec = HashedVectorizer(dim=64)
    v = vec.transform_one("a billing question about my invoice")
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_hashed_vectorizer_similar_texts_score_higher():
    vec = HashedVectorizer(dim=256)
    a = vec.transform_one("my card was charged twice please refund")
    b = vec.transform_one("i was charged twice on my card refund please")
    c = vec.transform_one("the app crashes every time i open it")
    assert (a @ b) > (a @ c)


def test_transform_empty_list_has_right_shape():
    vec = HashedVectorizer(dim=32)
    out = vec.transform([])
    assert out.shape == (0, 32)
