#!/usr/bin/env python3
"""Train an end-to-end SBNN candidate-pair ranker on the external benchmark."""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from bayesian_pair_ranker import ranking_metrics
from sbnn_pair_ranker import (
    SingularBayesianRanker,
    fit_sbnn,
    train_dense_ranker,
)
from shared_pair_features import FEATURE_NAMES, SCHEMA
from train_shared_pair_model import PhenotypeScores, dataset


def load_splits(project_root, seed):
    sys.path.insert(0, str(project_root / "src"))
    from rare_kid_modeling.molecular_benchmark import (
        load_molecular_cache,
        split_cases_by_gene_and_source,
    )
    from rare_kid_modeling.pheval_adapter import load_pheval_solved_cases

    paths = {
        "phenopackets": project_root / "data/open/phenopacket_store/0.1.27",
        "annotations": project_root
        / "data/open/molecular/vep_annotations_phenopacket_store_0.1.27.json",
        "obo": project_root / "data/open/hpo/v2026-06-23/hp-base.obo",
        "genes": project_root / "data/open/hpo/v2026-06-23/genes_to_phenotype.txt",
    }
    cases, load_report = load_pheval_solved_cases(paths["phenopackets"])
    annotations, decoys, _ = load_molecular_cache(paths["annotations"])
    knowledge = PhenotypeScores(paths["obo"], paths["genes"])
    splits = split_cases_by_gene_and_source(cases, seed=seed)
    data = {
        name: dataset(values, annotations, decoys, knowledge, seed)
        for name, values in splits.items()
    }
    for left, right in itertools.combinations(data, 2):
        for column in (2, 3, 4):
            if set(data[left][column]) & set(data[right][column]):
                raise AssertionError("Gene, source or case leakage between splits")
    return data, load_report.to_dict()


def evaluate(model, x, y, draws, seed):
    prediction = model.predict_samples(x, draws=draws, seed=seed)
    metrics = ranking_metrics(prediction["mean"], y)
    metrics["mean_mutual_information"] = float(prediction["mutual_information"].mean())
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "models/sbnn"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--draws", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()

    data, load_report = load_splits(args.project_root, args.seed)
    train_x, train_y, *_ = data["train"]
    validation_x, validation_y, *_ = data["validation"]
    test_x, test_y, *_ = data["test"]
    mean = train_x.mean(axis=(0, 1))
    scale = train_x.std(axis=(0, 1))
    scale[scale < 1e-8] = 1.0
    train_x = (train_x - mean) / scale
    validation_x = (validation_x - mean) / scale
    test_x = (test_x - mean) / scale

    hidden = (32, 16)
    ranks = (8, 8, 1)
    dense_weights, dense_biases = train_dense_ranker(
        train_x, train_y, hidden=hidden, epochs=400, seed=args.seed
    )
    runs = []
    models = []
    for initialization in ("random", "svd"):
        model = SingularBayesianRanker(
            len(FEATURE_NAMES), hidden=hidden, ranks=ranks, seed=args.seed
        )
        if initialization == "svd":
            model.initialize_from_dense(dense_weights, dense_biases)
        history = fit_sbnn(
            model,
            train_x,
            train_y,
            validation_x,
            validation_y,
            epochs=args.epochs,
            kl_weight=0.01,
            warmup_epochs=max(20, args.epochs // 10),
            seed=args.seed,
        )
        validation = evaluate(
            model, validation_x, validation_y, args.draws, args.seed + 1
        )
        runs.append(
            {
                "initialization": initialization,
                "validation": validation,
                "history": history,
            }
        )
        models.append(model)
        print(initialization, validation, flush=True)

    selected = min(
        range(len(runs)),
        key=lambda index: runs[index]["validation"]["negative_log_likelihood"],
    )
    model = models[selected]
    test_metrics = evaluate(model, test_x, test_y, args.draws, args.seed + 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "method": "Singular Bayesian Neural Network",
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "state": model.state_dict(),
        "selected_initialization": runs[selected]["initialization"],
        "training": {
            "seed": args.seed,
            "epochs": args.epochs,
            "kl_weight": 0.01,
            "kl_scaling": "linear warmup, then kl_weight / number of training cases",
            "posterior": "independent Gaussian factors A, B and bias",
            "prior": "0.5 N(0,1) + 0.5 N(0,exp(-6)^2)",
        },
    }
    (args.output_dir / "sbnn_pair_model.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )
    report = {
        "status": "executed",
        "loaded_cases": load_report,
        "split": {
            name: {"cases": len(values[1]), "genes": len(set(values[4]))}
            for name, values in data.items()
        },
        "gene_and_source_disjoint": True,
        "candidate_pairs_per_case": int(train_x.shape[1]),
        "architecture": {"hidden": hidden, "ranks": ranks},
        "initialization_runs": runs,
        "selected_initialization": runs[selected]["initialization"],
        "test": test_metrics,
        "challenge_gene_excluded": "BUB1B",
        "limitations": [
            "Candidate sets are constructed from known benign decoys, not full WGS backgrounds.",
            "The test split has appeared in earlier development experiments.",
            "A fresh external test and harder candidate generation are required before comparison with the challenge pipeline.",
        ],
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {"selected": runs[selected]["initialization"], "test": test_metrics},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
