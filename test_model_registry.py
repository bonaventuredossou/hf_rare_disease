from __future__ import annotations

import importlib

import numpy as np

from models.registry import MODEL_REGISTRY, SELECTED_SUBMISSION_MODEL
from models.temporal_consensus import WEIGHTS, predict_temporal_consensus
from models.wide_rank50_sbnn import (
    HIDDEN,
    KL_WEIGHT,
    POSTERIOR_DRAWS,
    POSTERIOR_SCALE_MULTIPLIER,
    PRIOR_SCALE,
    RANKS,
)


EXPECTED_MODELS = {
    "gradient_boost",
    "dense_mlp",
    "deep_ensemble",
    "small_sbnn",
    "dense_bbb",
    "wide_rank50_sbnn",
    "wide_dense_bbb",
    "temporal_consensus",
}


def test_every_reported_model_has_an_importable_module() -> None:
    assert set(MODEL_REGISTRY) == EXPECTED_MODELS
    for specification in MODEL_REGISTRY.values():
        assert importlib.import_module(specification.module)


def test_selected_submission_configuration_is_explicit() -> None:
    assert SELECTED_SUBMISSION_MODEL == "wide_rank50_sbnn"
    assert HIDDEN == (200, 100)
    assert RANKS == (50, 50, 1)
    assert KL_WEIGHT == 0.01
    assert POSTERIOR_SCALE_MULTIPLIER == 0.1
    assert PRIOR_SCALE == 0.3
    assert POSTERIOR_DRAWS == 128


def test_temporal_consensus_uses_the_reported_weights() -> None:
    assert WEIGHTS == {
        "gradient_boost": 0.3,
        "deep_ensemble": 0.2,
        "small_sbnn": 0.5,
    }
    predictions = {
        "gradient_boost": np.array([0.2, 0.7]),
        "deep_ensemble": np.array([0.4, 0.5]),
        "small_sbnn": np.array([0.8, 0.1]),
    }
    expected = 0.3 * predictions["gradient_boost"] + 0.2 * predictions["deep_ensemble"] + 0.5 * predictions["small_sbnn"]
    np.testing.assert_allclose(predict_temporal_consensus(predictions), expected)
