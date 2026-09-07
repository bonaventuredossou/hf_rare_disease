#!/usr/bin/env python3
"""Fit the Bayesian pair ranker on a labeled case-level NPZ benchmark.

Expected NPZ keys: ``features`` with shape [cases, candidates, features] and
``targets`` containing the causal candidate index for each case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bayesian_pair_ranker import Posterior, ranking_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
        help="Optional held-out NPZ with features and targets; never used for fitting.",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable train-only feature standardization (not recommended).",
    )
    args = parser.parse_args()
    data = np.load(args.benchmark, allow_pickle=False)
    features = np.asarray(data["features"], dtype=np.float64)
    if args.no_standardize:
        mean = np.zeros(features.shape[-1], dtype=np.float64)
        scale = np.ones(features.shape[-1], dtype=np.float64)
    else:
        # Fit normalization on training cases only; reuse it for evaluation.
        mean = features.mean(axis=(0, 1))
        scale = features.std(axis=(0, 1))
        scale = np.where(scale > 1e-8, scale, 1.0)
    standardized = (features - mean) / scale
    model = Posterior.fit(standardized, data["targets"], prior_scale=args.prior_scale)
    probabilities = model.map_probabilities(standardized)
    predictive = model.posterior_predictive(
        standardized, draws=args.draws, seed=args.seed
    )
    metrics_evaluation = None
    if args.evaluation is not None:
        held_out = np.load(args.evaluation, allow_pickle=False)
        evaluation_features = (
            np.asarray(held_out["features"], dtype=np.float64) - mean
        ) / scale
        evaluation_probabilities = model.map_probabilities(evaluation_features)
        metrics_evaluation = ranking_metrics(
            evaluation_probabilities, held_out["targets"]
        )
    result = {
        "model": "laplace_bayesian_conditional_logit",
        "prior_scale": args.prior_scale,
        "posterior_draws": args.draws,
        "seed": args.seed,
        "standardized": not args.no_standardize,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "converged": model.converged,
        "objective": model.objective,
        "metrics_map": ranking_metrics(probabilities, data["targets"]),
        "metrics_evaluation": metrics_evaluation,
        "mean_mutual_information": float(predictive["mutual_information"].mean()),
        "mean_target_probability_std": float(
            predictive["probability_std"][
                np.arange(len(data["targets"])), data["targets"]
            ].mean()
        ),
        "weights_map": model.weights_map.tolist(),
        "posterior_covariance": model.covariance.tolist(),
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"weights_map", "posterior_covariance"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
