#!/usr/bin/env python3
"""Bayesian case-level ranking for rare-disease candidate pairs.

Each case has one observed causal candidate among a fixed candidate set.  The
model assigns utility ``x @ w`` to each candidate and fits a conditional-softmax
likelihood with an isotropic Gaussian prior.  A Laplace approximation around
the MAP estimate provides posterior draws and uncertainty diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class Posterior:
    weights_map: np.ndarray
    covariance: np.ndarray
    prior_scale: float
    converged: bool
    objective: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        prior_scale: float = 1.0,
        max_iterations: int = 500,
    ) -> "Posterior":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.int64)
        if x.ndim != 3 or y.shape != (x.shape[0],):
            raise ValueError(
                "features must be [cases, candidates, features] and targets must align"
            )
        if min(x.shape) < 1 or not np.isfinite(x).all():
            raise ValueError("features must be nonempty and finite")
        if not np.array_equal(y, np.asarray(targets)):
            raise ValueError("targets must be integer indices")
        if np.any(y < 0) or np.any(y >= x.shape[1]):
            raise ValueError("targets contain an invalid candidate index")
        if prior_scale <= 0 or not math.isfinite(prior_scale):
            raise ValueError("prior_scale must be finite and positive")
        rows = np.arange(x.shape[0])
        precision = 1.0 / prior_scale**2

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            scores = np.einsum("ncf,f->nc", x, w, optimize=True)
            probs = _softmax(scores)
            loss = np.sum(logsumexp(scores, axis=1) - scores[rows, y])
            loss += 0.5 * precision * float(w @ w)
            residual = probs.copy()
            residual[rows, y] -= 1.0
            gradient = np.einsum("ncf,nc->f", x, residual, optimize=True)
            gradient += precision * w
            return float(loss), gradient

        result = minimize(
            objective,
            np.zeros(x.shape[2]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iterations, "ftol": 1e-12, "gtol": 1e-8},
        )
        w = np.asarray(result.x, dtype=np.float64)
        scores = np.einsum("ncf,f->nc", x, w, optimize=True)
        probs = _softmax(scores)
        second = np.einsum("nci,nc,ncj->ij", x, probs, x, optimize=True)
        means = np.einsum("nci,nc->ni", x, probs, optimize=True)
        hessian = second - np.einsum("ni,nj->ij", means, means)
        hessian += precision * np.eye(x.shape[2])
        covariance = np.linalg.inv(0.5 * (hessian + hessian.T))
        covariance = 0.5 * (covariance + covariance.T)
        return cls(w, covariance, prior_scale, bool(result.success), float(result.fun))

    def map_probabilities(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if (
            x.ndim != 3
            or x.shape[-1] != len(self.weights_map)
            or not np.isfinite(x).all()
        ):
            raise ValueError("invalid prediction features")
        return _softmax(np.einsum("ncf,f->nc", x, self.weights_map, optimize=True))

    def posterior_predictive(
        self, features: np.ndarray, *, draws: int = 256, seed: int = 20260902
    ) -> dict[str, np.ndarray]:
        if draws < 2:
            raise ValueError("draws must be at least 2")
        x = np.asarray(features, dtype=np.float64)
        if (
            x.ndim != 3
            or x.shape[-1] != len(self.weights_map)
            or not np.isfinite(x).all()
        ):
            raise ValueError("invalid prediction features")
        rng = np.random.default_rng(seed)
        sampled = rng.multivariate_normal(
            self.weights_map, self.covariance, size=draws, check_valid="raise"
        )
        probabilities = _softmax(np.einsum("ncf,sf->snc", x, sampled, optimize=True))
        mean = probabilities.mean(axis=0)
        predictive_entropy = -np.sum(mean * np.log(np.clip(mean, 1e-15, 1.0)), axis=1)
        expected_entropy = (
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0)), axis=2)
        ).mean(axis=0)
        return {
            "mean_probabilities": mean,
            "probability_std": probabilities.std(axis=0, ddof=1),
            "probability_lower_95": np.quantile(probabilities, 0.025, axis=0),
            "probability_upper_95": np.quantile(probabilities, 0.975, axis=0),
            "probability_draws": probabilities,
            "expected_entropy": expected_entropy,
            "predictive_entropy": predictive_entropy,
            "mutual_information": np.maximum(
                predictive_entropy - expected_entropy, 0.0
            ),
        }


def ranking_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int64)
    order = np.argsort(-p, axis=1, kind="stable")
    ranks = np.argmax(order == y[:, None], axis=1) + 1
    return {
        "case_count": float(len(y)),
        "top1_accuracy": float(np.mean(ranks == 1)),
        "top5_accuracy": float(np.mean(ranks <= min(5, p.shape[1]))),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "negative_log_likelihood": float(
            -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0)))
        ),
    }
