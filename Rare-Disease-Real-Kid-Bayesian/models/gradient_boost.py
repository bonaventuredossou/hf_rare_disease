"""Gradient-boosted tree baseline used on unscaled ClinVar features."""

from sklearn.ensemble import HistGradientBoostingClassifier


MODEL_NAME = "gradient_boost"


def fit_gradient_boost(x_train, y_train, *, seed):
    """Fit the exact gradient-boosting configuration used in the report."""
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    ).fit(x_train, y_train)


def predict_gradient_boost(model, x):
    return model.predict_proba(x)[:, 1]
