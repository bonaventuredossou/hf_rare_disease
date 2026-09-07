# Change log

## 2026-09-07: public repository release

- Added a repository-level overview of the complete genome-to-submission pipeline.
- Exposed seven fitted models and the temporal consensus as separate modules under `models/`.
- Added a model registry and tests tying report names to importable implementations.
- Added one canonical SBNN submission at `submission/track1_wide_rank50_sbnn.csv`.
- Added a guarded exporter and machine-readable provenance manifest for the submission.
- Added public model, results, data, and reproducibility documentation.
- Removed private coordination notes and duplicate cluster-package files from the public release.
- Updated checkpoint inference so the ClinVar training archive is required only for retraining.
- Added the selected SBNN output to exact Track 1 score verification.
- Verified 29 tests, the submission schema, rank 1, 100 Rank Points, and F-max 1.000.

## 2026-09-06: wide Bayesian comparison and generalization

- Evaluated a 50-200-100-1 network so low-rank choices were meaningful at the tested scale.
- Screened coupled rank, KL, posterior-scale, and prior-scale configurations.
- Added dense Bayes-by-Backprop and width-matched comparisons.
- Selected the rank `(50, 50, 1)` SBNN for its temporal top-1 ranking and PR-AUC.
- Retained the five-network ensemble as the strongest tested unseen-gene detector.
- Added repeated post-2023 Phenopacket/HPO ranking experiments.

## 2026-09-05: temporal benchmark and Track 1 transfer

- Added a leakage-controlled ClinVar temporal split with exact Variation ID exclusion.
- Added gradient boosting, dense MLP, ensemble, SBNN, BBB, and consensus baselines.
- Added classification, ranking, calibration, selective-prediction, and OOD metrics.
- Transferred the trained allele models to strict and broad child candidate tables.
- Verified that the accepted `BUB1B` pair remains rank 1 across the tested adapters.

## 2026-09-02: Bayesian pipeline foundation

- Added the shared candidate-pair feature contract.
- Added the conditional-logit linear Bayesian baseline and Laplace posterior.
- Added the low-rank neural Bayesian implementation and Monte Carlo inference.
- Added independent submission-schema validation and exact score verification.
- Preserved the original genome-wide pipeline and leaderboard-confirmed deterministic CSV.
