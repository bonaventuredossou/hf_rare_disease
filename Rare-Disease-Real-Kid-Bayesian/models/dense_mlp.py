"""Single deterministic neural-network baseline."""

from sklearn.neural_network import MLPClassifier


MODEL_NAME = "dense_mlp"


def fit_dense_mlp(x_train, y_train, *, seed):
    """Fit the 50-32-16-1 deterministic MLP used in the comparison."""
    return MLPClassifier(
        hidden_layer_sizes=(32, 16),
        alpha=1e-4,
        batch_size=512,
        max_iter=120,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=12,
        random_state=seed,
    ).fit(x_train, y_train)


def predict_dense_mlp(model, x):
    return model.predict_proba(x)[:, 1]
