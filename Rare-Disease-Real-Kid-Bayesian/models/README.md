# Model implementations

Every model named in the methods report has a visible module in this directory.

| Report name | Implementation | Training entry point |
|---|---|---|
| Gradient boosting | `gradient_boost.py` | `fit_gradient_boost` |
| Dense MLP | `dense_mlp.py` | `fit_dense_mlp` |
| Five-network deep ensemble | `deep_ensemble.py` | `fit_deep_ensemble` |
| Original small SBNN | `small_sbnn.py` | `fit_small_sbnn` |
| Dense/full-rank BBB | `full_rank_bbb.py` | `fit_full_rank_bbb` |
| Wide rank-50 SBNN | `wide_rank50_sbnn.py` | `fit_wide_rank50_sbnn` |
| Wide full-weight BBB | `wide_dense_bbb.py` | `fit_wide_dense_bbb` |
| Temporal consensus | `temporal_consensus.py` | `predict_temporal_consensus` |

`registry.py` is the machine-readable index. Running `python -m models.registry` prints every method, architecture, role, and the selected submission model.

The low-level variational implementation is shared rather than copied across four files. `sbnn_pair_ranker.py` defines the low-rank and dense Bayesian layers, posterior sampling, mixture-prior KL, reparameterized gradients, and Adam optimizer. `clinvar_temporal_experiment.py` defines the binary SBNN and BBB networks and their training loops. The model modules in this directory fix the exact architecture and hyperparameters used for each reported comparison.

The public command `python generate_track1_submission.py --ranking-zip /path/to/ranking.zip` applies the fitted model suite through `verify_temporal_models_track1.py`. The selected `wide_rank50_sbnn` prediction is then validated and published as the canonical challenge CSV by `export_track1_sbnn_submission.py`.
