"""Full-weight BBB network matched to the selected SBNN's hidden width."""

from clinvar_temporal_experiment import fit_bbb


MODEL_NAME = "wide_dense_bbb"
HIDDEN = (200, 100)
RHO_INIT = -4.0
KL_WEIGHT = 0.1
PRIOR_SCALE = 0.3


def fit_wide_dense_bbb(x_train, y_train, x_val, y_val, *, seed, epochs=60):
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


def predict_wide_dense_bbb(model, x, *, draws=128, seed):
    return model.predict_statistics(x, draws=draws, seed=seed)
