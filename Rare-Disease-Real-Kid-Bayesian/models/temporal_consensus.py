"""Validated combination of gradient boosting, deep ensemble, and small SBNN."""

import numpy as np


MODEL_NAME = "temporal_consensus"
WEIGHTS = {
    "gradient_boost": 0.3,
    "deep_ensemble": 0.2,
    "small_sbnn": 0.5,
}


def predict_temporal_consensus(predictions: dict[str, np.ndarray]) -> np.ndarray:
    missing = set(WEIGHTS) - set(predictions)
    if missing:
        raise KeyError(f"Consensus inputs are missing: {sorted(missing)}")
    return sum(WEIGHTS[name] * predictions[name] for name in WEIGHTS)
