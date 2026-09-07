"""Full-rank Bayes-by-Backprop network at the original hidden width."""

from clinvar_temporal_experiment import fit_bbb


MODEL_NAME = "dense_bbb"
HIDDEN = (32, 16)
RHO_INIT = -4.0
KL_WEIGHT = 0.1
PRIOR_SCALE = 0.3


def fit_full_rank_bbb(x_train, y_train, x_val, y_val, *, seed, epochs=60):
    """Fit independent Gaussian posteriors for every weight and bias."""
    return fit_bbb(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=seed,
        epochs=epochs,
        hidden=HIDDEN,
        rho_init=RHO_INIT,
        kl_weight=KL_WEIGHT,
        prior_sigma1=PRIOR_SCALE,
    )


def predict_full_rank_bbb(model, x, *, draws=128, seed):
    return model.predict_statistics(x, draws=draws, seed=seed)
