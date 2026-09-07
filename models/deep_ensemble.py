"""Five independently initialized deterministic neural networks."""

import numpy as np

from models.dense_mlp import fit_dense_mlp


MODEL_NAME = "deep_ensemble"
MEMBERS = 5


def fit_deep_ensemble(x_train, y_train, *, seed, members=MEMBERS):
    """Fit independent MLP members with consecutive random seeds."""
    return [
        fit_dense_mlp(x_train, y_train, seed=seed + member)
        for member in range(members)
    ]


def predict_deep_ensemble(models, x):
    """Return the ensemble mean, mutual information, and member standard deviation."""
    samples = np.stack([model.predict_proba(x)[:, 1] for model in models])
    mean = samples.mean(axis=0)

    def entropy(probability):
        clipped = np.clip(probability, 1e-12, 1.0 - 1e-12)
        return -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))

    mutual_information = np.maximum(
        entropy(mean) - entropy(samples).mean(axis=0), 0.0
    )
    return mean, mutual_information, samples.std(axis=0, ddof=1)
