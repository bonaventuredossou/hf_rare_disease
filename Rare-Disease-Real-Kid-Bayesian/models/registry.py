"""Auditable registry for every model named in the methods report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    family: str
    architecture: str
    role: str
    produces_track1_csv: bool = True


MODEL_REGISTRY = {
    spec.name: spec
    for spec in (
        ModelSpec(
            "gradient_boost",
            "models.gradient_boost",
            "deterministic tree",
            "220 boosting iterations, at most 31 leaves",
            "tabular pathogenicity baseline",
        ),
        ModelSpec(
            "dense_mlp",
            "models.dense_mlp",
            "deterministic neural network",
            "50-32-16-1",
            "single-network neural baseline",
        ),
        ModelSpec(
            "deep_ensemble",
            "models.deep_ensemble",
            "deep ensemble",
            "five independently initialized 50-32-16-1 MLPs",
            "epistemic uncertainty and OOD baseline",
        ),
        ModelSpec(
            "small_sbnn",
            "models.small_sbnn",
            "low-rank Bayesian neural network",
            "50-32-16-1 with ranks 8-8-1",
            "original SBNN baseline",
        ),
        ModelSpec(
            "dense_bbb",
            "models.full_rank_bbb",
            "full-weight Bayesian neural network",
            "50-32-16-1 with an independent posterior per weight",
            "full-weight Bayes-by-Backprop comparison",
        ),
        ModelSpec(
            "wide_rank50_sbnn",
            "models.wide_rank50_sbnn",
            "low-rank Bayesian neural network",
            "50-200-100-1 with ranks 50-50-1",
            "selected Track 1 submission model",
        ),
        ModelSpec(
            "wide_dense_bbb",
            "models.wide_dense_bbb",
            "full-weight Bayesian neural network",
            "50-200-100-1 with an independent posterior per weight",
            "width-matched full-weight Bayesian comparison",
        ),
        ModelSpec(
            "temporal_consensus",
            "models.temporal_consensus",
            "validated probability mixture",
            "0.3 gradient boost + 0.2 deep ensemble + 0.5 small SBNN",
            "combined ranking model",
        ),
    )
}

SELECTED_SUBMISSION_MODEL = "wide_rank50_sbnn"


def main() -> None:
    print(
        json.dumps(
            {
                "selected_submission_model": SELECTED_SUBMISSION_MODEL,
                "models": {name: asdict(spec) for name, spec in MODEL_REGISTRY.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
