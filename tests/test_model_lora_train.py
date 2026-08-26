"""Tests for the network, LoRA patch, and the three training loops."""

from __future__ import annotations

import numpy as np

from finetune_lab.lora import LoRA
from finetune_lab.model import TwoLayerNet, cross_entropy, softmax
from finetune_lab.train import fit_full, fit_head_only, fit_lora, loss_curve


def test_softmax_rows_sum_to_one():
    probs = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_forward_shapes(train_features, splits):
    train, _val, _test = splits
    net = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    out = net.forward(train_features)
    assert out.hidden.shape == (len(train), 16)
    assert out.probs.shape == (len(train), 4)


def test_untrained_head_is_no_better_than_chance(train_features, test_features, splits):
    _train, _val, test = splits
    net = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    # W2 starts at all zeros -> uniform probabilities -> ~25% accuracy on 4 classes.
    acc = net.accuracy(test_features, test.labels)
    assert acc < 0.5


def test_fit_head_only_improves_over_untrained(train_features, test_features, splits):
    train, _val, test = splits
    net = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    before = net.accuracy(test_features, test.labels)
    fit_head_only(net, train_features, train.labels, epochs=200)
    after = net.accuracy(test_features, test.labels)
    assert after > before


def test_fit_full_beats_head_only(train_features, test_features, splits):
    train, _val, test = splits
    net_head = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    fit_head_only(net_head, train_features, train.labels, epochs=200)
    head_acc = net_head.accuracy(test_features, test.labels)

    net_full = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    fit_full(net_full, train_features, train.labels, epochs=300)
    full_acc = net_full.accuracy(test_features, test.labels)

    assert full_acc >= head_acc


def test_fit_full_does_not_touch_original_backbone_copy(train_features, splits):
    train, _val, _test = splits
    net = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    original_W1 = net.W1.copy()
    fit_full(net, train_features, train.labels, epochs=50)
    # Full fine-tuning MUST change the backbone -- that's the whole point.
    assert not np.array_equal(net.W1, original_W1)


def test_lora_starts_as_a_no_op():
    lora = LoRA.init(dim=32, hidden=8, rank=2, seed=0)
    # B starts at zero, so the patch is exactly zero before any training.
    assert np.array_equal(lora.delta(), np.zeros((32, 8)))


def test_lora_has_far_fewer_params_than_full():
    dim, hidden, rank = 512, 64, 4
    full_params = dim * hidden
    lora = LoRA.init(dim=dim, hidden=hidden, rank=rank, seed=0)
    assert lora.n_params() < full_params / 5  # at least 5x fewer


def test_fit_lora_leaves_backbone_frozen(train_features, splits):
    train, _val, _test = splits
    net = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    original_W1 = net.W1.copy()
    fit_lora(net, train_features, train.labels, rank=2, epochs=50)
    # LoRA must NOT touch the backbone -- only the patch and the head change.
    assert np.array_equal(net.W1, original_W1)


def test_fit_lora_competitive_with_full(train_features, test_features, splits):
    train, _val, test = splits
    net_full = TwoLayerNet(dim=train_features.shape[1], hidden=32, seed=0)
    fit_full(net_full, train_features, train.labels, epochs=300)
    full_acc = net_full.accuracy(test_features, test.labels)

    net_lora = TwoLayerNet(dim=train_features.shape[1], hidden=32, seed=0)
    lora = fit_lora(net_lora, train_features, train.labels, rank=4, epochs=300, seed=0)
    lora_acc = net_lora.accuracy(test_features, test.labels, delta_W1=lora.delta())

    # LoRA should get within a reasonable margin of full fine-tuning.
    assert lora_acc >= full_acc - 0.15


def test_cross_entropy_lower_for_confident_correct_prediction():
    probs_confident = np.array([[0.9, 0.05, 0.05]])
    probs_unsure = np.array([[0.4, 0.3, 0.3]])
    labels = np.array([0])
    assert cross_entropy(probs_confident, labels) < cross_entropy(probs_unsure, labels)


def test_loss_curve_decreases(train_features, splits):
    train, _val, _test = splits
    net = TwoLayerNet(dim=train_features.shape[1], hidden=16, seed=0)
    history = loss_curve(net, train_features, train.labels, epochs=200)
    assert history[-1] < history[0]
