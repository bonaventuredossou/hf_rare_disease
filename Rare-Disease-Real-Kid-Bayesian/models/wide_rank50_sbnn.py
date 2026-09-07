"""Selected 50-200-100-1 low-rank SBNN used for the submission."""

from clinvar_temporal_experiment import fit_sbnn


MODEL_NAME = "wide_rank50_sbnn"
HIDDEN = (200, 100)
RANKS = (50, 50, 1)
KL_WEIGHT = 0.01
POSTERIOR_SCALE_MULTIPLIER = 0.1
PRIOR_SCALE = 0.3
POSTERIOR_DRAWS = 128


def fit_wide_rank50_sbnn(x_train, y_train, x_val, y_val, *, seed, epochs=60):
    """Fit the SBNN configuration selected by the temporal rank sweep."""
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


def predict_wide_rank50_sbnn(model, x, *, draws=POSTERIOR_DRAWS, seed):
    """Return posterior mean, mutual information, and predictive standard deviation."""
    return model.predict_statistics(x, draws=draws, seed=seed)
