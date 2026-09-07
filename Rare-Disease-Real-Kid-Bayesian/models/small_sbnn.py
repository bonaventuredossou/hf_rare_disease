"""Original 50-32-16-1 low-rank Singular Bayesian Neural Network."""

from clinvar_temporal_experiment import fit_sbnn


MODEL_NAME = "small_sbnn"
HIDDEN = (32, 16)
RANKS = (8, 8, 1)
KL_WEIGHT = 0.01
POSTERIOR_SCALE_MULTIPLIER = 1.0
PRIOR_SCALE = 1.0


def fit_small_sbnn(x_train, y_train, x_val, y_val, *, seed, epochs=80):
    return fit_sbnn(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=seed,
        epochs=epochs,
        hidden=HIDDEN,
        ranks=RANKS,
        kl_weight=KL_WEIGHT,
        posterior_scale_multiplier=POSTERIOR_SCALE_MULTIPLIER,
        prior_sigma1=PRIOR_SCALE,
    )


def predict_small_sbnn(model, x, *, draws=128, seed):
    return model.predict_statistics(x, draws=draws, seed=seed)
