"""Tests for the prompt/RAG baselines and the full comparison table."""

from __future__ import annotations

from finetune_lab.baselines import PromptClassifier, RagClassifier
from finetune_lab.dataset import make_splits
from finetune_lab.evaluate import compare_all, format_table, per_class_accuracy
from finetune_lab.features import HashedVectorizer


def test_prompt_classifier_predicts_something_for_every_ticket(splits):
    _train, _val, test = splits
    clf = PromptClassifier()
    preds = clf.predict(test.texts)
    assert len(preds) == len(test)


def test_prompt_classifier_better_than_random(splits):
    _train, _val, test = splits
    clf = PromptClassifier()
    assert clf.accuracy(test.texts, test.labels) > 0.25  # better than 1-in-4 chance


def test_rag_classifier_needs_fit_before_predict(train_features):
    rag = RagClassifier(k=3)
    try:
        rag.predict(train_features)
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass


def test_rag_classifier_beats_random(train_features, test_features, splits):
    train, _val, test = splits
    rag = RagClassifier(k=5).fit(train_features, train.labels)
    assert rag.accuracy(test_features, test.labels) > 0.25


def test_per_class_accuracy_has_all_classes():
    import numpy as np
    preds = np.array([0, 1, 1, 2, 3])
    labels = np.array([0, 1, 0, 2, 3])
    scores = per_class_accuracy(preds, labels, ("a", "b", "c", "d"))
    assert set(scores) == {"a", "b", "c", "d"}
    # True class "a" (label 0) is indices 0 and 2; predictions there are [0, 1] -> 1/2 right.
    assert scores["a"] == 0.5
    assert scores["b"] == 1.0  # true class "b" is only index 1, predicted correctly


def test_compare_all_runs_and_orders_methods_sensibly():
    train, _val, test = make_splits(n_train=15, n_val=5, n_test=10, seed=2)
    results = compare_all(train, test, dim=128, hidden=16, lora_rank=2, seed=0)
    names = [r.name for r in results]
    assert any("prompt" in n for n in names)
    assert any("RAG" in n for n in names)
    assert any("full fine-tune" in n for n in names)
    assert any("LoRA" in n for n in names)
    full = next(r for r in results if r.name.startswith("full"))
    lora = next(r for r in results if r.name.startswith("LoRA"))
    # LoRA must have far fewer trainable params than full fine-tuning.
    assert lora.trainable_params < full.trainable_params / 3


def test_format_table_contains_every_method():
    train, _val, test = make_splits(n_train=10, n_val=5, n_test=8, seed=3)
    results = compare_all(train, test, dim=64, hidden=8, lora_rank=2, seed=0)
    table = format_table(results)
    for r in results:
        assert r.name in table
