#!/usr/bin/env python3
"""Select a deterministic/SBNN consensus on one temporal half, test on the other."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from clinvar_temporal_experiment import ece, fmax, ranking_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()

    data = pd.read_csv(args.predictions, sep="\t")
    model_columns = ["Gradient boost", "Deep ensemble", "SBNN"]
    selection = (data["variation_id"].to_numpy(dtype=np.int64) % 2) == 0
    y_selection = data.loc[selection, "label"].to_numpy()

    candidates = []
    grid = np.linspace(0, 1, 11)
    for tree_weight in grid:
        for ensemble_weight in grid:
            sbnn_weight = 1.0 - tree_weight - ensemble_weight
            if sbnn_weight < -1e-9:
                continue
            weights = np.array([tree_weight, ensemble_weight, max(0.0, sbnn_weight)])
            probability = data.loc[selection, model_columns].to_numpy() @ weights
            candidates.append((average_precision_score(y_selection, probability), weights))
    _, selected_weights = max(candidates, key=lambda item: item[0])

    holdout = ~selection
    y = data.loc[holdout, "label"].to_numpy()
    individual = {name: data.loc[holdout, name].to_numpy() for name in model_columns}
    individual["Selected consensus"] = data.loc[holdout, model_columns].to_numpy() @ selected_weights
    report = {
        "selection_rule": "maximize PR-AUC on even ClinVar Variation IDs from the temporal test; report odd IDs only",
        "selection_n": int(selection.sum()),
        "holdout_n": int(holdout.sum()),
        "weights": dict(zip(model_columns, selected_weights.tolist())),
        "holdout": {},
    }
    for name, probability in individual.items():
        maximum_f, threshold = fmax(y, probability)
        report["holdout"][name] = {
            "roc_auc": float(roc_auc_score(y, probability)),
            "pr_auc": float(average_precision_score(y, probability)),
            "f_max": maximum_f,
            "f_max_threshold": threshold,
            "nll": float(log_loss(y, probability, labels=[0, 1])),
            "brier": float(brier_score_loss(y, probability)),
            "ece_15": ece(y, probability),
            "ranking": ranking_metrics(y, probability, seed=args.seed),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
