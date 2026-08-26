"""Shared fixtures: a small, fast, deterministic split for every test."""

from __future__ import annotations

import pytest

from finetune_lab.dataset import make_splits
from finetune_lab.features import HashedVectorizer


@pytest.fixture(scope="session")
def splits():
    """A small dataset (fast tests): 20/10/15 per class."""
    return make_splits(n_train=20, n_val=10, n_test=15, seed=1)


@pytest.fixture(scope="session")
def vectorizer():
    return HashedVectorizer(dim=256)


@pytest.fixture(scope="session")
def train_features(splits, vectorizer):
    train, _val, _test = splits
    return vectorizer.transform(train.texts)


@pytest.fixture(scope="session")
def test_features(splits, vectorizer):
    _train, _val, test = splits
    return vectorizer.transform(test.texts)
