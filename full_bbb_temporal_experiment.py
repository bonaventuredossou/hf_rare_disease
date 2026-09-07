#!/usr/bin/env python3
"""Evaluate a dense Bayes-by-Backprop network on the temporal ClinVar split."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from clinvar_temporal_experiment import (
    BinaryBBB,
    balanced_cap,
    ece,
    featurize,
    fit_bbb,
    fmax,
    gene_statistics,
    parse_release,
    ranking_metrics,
    selective_metrics,
)


@dataclass(frozen=True)
class BBBConfig:
    rho_init: float
    kl_weight: float
    prior_sigma1: float

    @property
    def name(self):
        return f"bbb_rho{self.rho_init:g}_kl{self.kl_weight:g}_prior{self.prior_sigma1:g}"


def configs():
    return [
        BBBConfig(rho, kl, prior)
        for rho in (-8.0, -6.0, -5.0, -4.0, -3.0)
        for kl in (0.001, 0.01, 0.1, 1.0, 3.0)
        for prior in (0.3, 1.0, 3.0)
    ]


def summarize(y, probability, uncertainty, standard_deviation, genes, train_genes):
    unseen = np.asarray([gene not in train_genes for gene in genes])
    maximum_f, _ = fmax(y, probability)
    selective = selective_metrics(y, probability, uncertainty)
    return {
        "n": int(len(y)),
        "pr_auc": float(average_precision_score(y, probability)),
        "f_max": maximum_f,
        "nll": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "ece_15": ece(y, probability),
        "balanced_accuracy_at_80pct_coverage": selective["balanced_accuracy_at_80pct_coverage"],
        "uncertainty_unseen_gene_auroc": float(roc_auc_score(unseen, uncertainty)),
        "std_unseen_gene_auroc": float(roc_auc_score(unseen, standard_deviation)),
        "known_gene_mean_std": float(standard_deviation[~unseen].mean()),
        "unseen_gene_mean_std": float(standard_deviation[unseen].mean()),
    }


def score(metrics):
    return (
        metrics["nll"] + metrics["brier"] + metrics["ece_15"]
        + 0.2 * (1.0 - metrics["uncertainty_unseen_gene_auroc"])
        + 0.5 * (1.0 - metrics["balanced_accuracy_at_80pct_coverage"])
    )


def run(config, x_train, y_train, x_val, y_val, x_eval, y_eval, genes, train_genes, *, seed, epochs, draws):
    model, history = fit_bbb(
        x_train, y_train, x_val, y_val, seed=seed, epochs=epochs,
        kl_weight=config.kl_weight, rho_init=config.rho_init,
        prior_sigma1=config.prior_sigma1,
    )
    probability, uncertainty, standard_deviation = model.predict_statistics(
        x_eval, draws=draws, seed=seed + 99
    )
    metrics = summarize(
        y_eval, probability, uncertainty, standard_deviation, genes, train_genes
    )
    metrics["score"] = score(metrics)
    metrics["epochs_completed"] = history[-1]["epoch"] if history else 0
    return model, metrics


def task_valid(results):
    best_pr = max(item["metrics"]["pr_auc"] for item in results)
    best_f = max(item["metrics"]["f_max"] for item in results)
    return [
        item for item in results
        if item["metrics"]["pr_auc"] >= best_pr - 0.005
        and item["metrics"]["f_max"] >= best_f - 0.005
    ]


def shortlist(results, limit=8):
    valid = task_valid(results) or results
    selected = []
    for key, reverse in (
        ("uncertainty_unseen_gene_auroc", True),
        ("nll", False),
        ("score", False),
    ):
        item = sorted(valid, key=lambda value: value["metrics"][key], reverse=reverse)[0]
        if item["config"]["name"] not in {value["config"]["name"] for value in selected}:
            selected.append(item)
    for rho in (-8.0, -6.0, -5.0, -4.0, -3.0):
        group = [item for item in valid if item["config"]["rho_init"] == rho]
        if group:
            item = max(group, key=lambda value: value["metrics"]["uncertainty_unseen_gene_auroc"])
            if item["config"]["name"] not in {value["config"]["name"] for value in selected}:
                selected.append(item)
    return selected[:limit]


def plot_grid(output_dir, results):
    lookup = {
        (item["config"]["rho_init"], item["config"]["kl_weight"], item["config"]["prior_sigma1"]): item["metrics"]
        for item in results
    }
    rhos = (-8.0, -6.0, -5.0, -4.0, -3.0)
    kls = (0.001, 0.01, 0.1, 1.0, 3.0)
    for prior in (0.3, 1.0, 3.0):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for axis, key, title in (
            (axes[0], "uncertainty_unseen_gene_auroc", "Unseen-gene AUROC"),
            (axes[1], "nll", "NLL"),
        ):
            matrix = np.asarray([[lookup[(rho, kl, prior)][key] for rho in rhos] for kl in kls])
            image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
            axis.set_xticks(range(len(rhos)), [str(value) for value in rhos])
            axis.set_yticks(range(len(kls)), [str(value) for value in kls])
            axis.set_xlabel("rho init")
            axis.set_ylabel("KL weight")
            axis.set_title(title)
            fig.colorbar(image, ax=axis)
        fig.suptitle(f"Dense BBB, broad prior sigma = {prior}")
        fig.tight_layout()
        fig.savefig(output_dir / f"bbb_grid_prior_{prior:g}.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    valid_names = {item["config"]["name"] for item in task_valid(results)}
    for item in results:
        color = "#C43B3B" if item["config"]["name"] in valid_names else "#888888"
        axis.scatter(item["metrics"]["nll"], item["metrics"]["uncertainty_unseen_gene_auroc"], color=color, alpha=0.7)
    axis.axhline(0.70, color="black", linestyle="--")
    axis.set_xlabel("NLL (lower is better)")
    axis.set_ylabel("Unseen-gene AUROC (higher is better)")
    axis.set_title("Dense BBB calibration and OOD tradeoff")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "bbb_pareto.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading ClinVar snapshots", flush=True)
    historical, historical_ids = parse_release(args.historical, collect_all_ids=True)
    current, _ = parse_release(args.current, collect_all_ids=False)
    temporal = [record for record in current if record.variation_id not in historical_ids]
    labels = np.asarray([record.label for record in historical])
    groups = np.asarray([record.gene for record in historical])
    train_index, val_index = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=args.seed
    ).split(np.zeros(len(labels)), labels, groups))
    train_index = balanced_cap(train_index, labels, 30000, args.seed)
    val_index = balanced_cap(val_index, labels, 7500, args.seed + 1)
    stats = gene_statistics(historical, train_index)
    train_records = [historical[int(i)] for i in train_index]
    val_records = [historical[int(i)] for i in val_index]
    tune_all = [record for record in temporal if record.variation_id % 2 == 0]
    test_records = [record for record in temporal if record.variation_id % 2 == 1]
    rng = np.random.default_rng(args.seed)
    tune_records = [tune_all[int(i)] for i in rng.choice(len(tune_all), 60000, replace=False)]

    x_train_raw = featurize(train_records, stats)
    x_val_raw = featurize(val_records, stats)
    x_tune_raw = featurize(tune_records, stats)
    x_test_raw = featurize(test_records, stats)
    y_train = np.asarray([record.label for record in train_records])
    y_val = np.asarray([record.label for record in val_records])
    y_tune = np.asarray([record.label for record in tune_records])
    y_test = np.asarray([record.label for record in test_records])
    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)
    x_tune = scaler.transform(x_tune_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)
    train_genes = {record.gene for record in train_records}
    tune_genes = [record.gene for record in tune_records]
    test_genes = [record.gene for record in test_records]
    screen_train = balanced_cap(np.arange(len(y_train)), y_train, 10000, args.seed + 2)
    screen_val = rng.choice(len(y_val), 3000, replace=False)
    screen_tune = rng.choice(len(y_tune), 20000, replace=False)
    screen_genes = [tune_genes[int(i)] for i in screen_tune]

    stage1 = []
    candidates = configs()
    print(f"Screening {len(candidates)} dense BBB configurations", flush=True)
    for number, config in enumerate(candidates, 1):
        _, metrics = run(
            config,
            x_train[screen_train], y_train[screen_train], x_val[screen_val], y_val[screen_val],
            x_tune[screen_tune], y_tune[screen_tune], screen_genes, train_genes,
            seed=args.seed, epochs=12, draws=12,
        )
        stage1.append({"config": {**asdict(config), "name": config.name}, "metrics": metrics})
        with (args.output_dir / "bbb_stage1_checkpoint.json").open("w") as handle:
            json.dump({"stage1": stage1}, handle, indent=2)
            handle.write("\n")
        print(f"  {number:02d}/{len(candidates)} {config.name}: NLL={metrics['nll']:.4f}, OOD={metrics['uncertainty_unseen_gene_auroc']:.4f}", flush=True)
    plot_grid(args.output_dir, stage1)

    stage2 = []
    selected = shortlist(stage1)
    print(f"Confirming {len(selected)} configurations", flush=True)
    for item in selected:
        values = item["config"]
        config = BBBConfig(values["rho_init"], values["kl_weight"], values["prior_sigma1"])
        _, metrics = run(
            config, x_train, y_train, x_val, y_val,
            x_tune, y_tune, tune_genes, train_genes,
            seed=args.seed, epochs=36, draws=24,
        )
        stage2.append({"config": {**asdict(config), "name": config.name}, "metrics": metrics})
        print(f"  {config.name}: NLL={metrics['nll']:.4f}, OOD={metrics['uncertainty_unseen_gene_auroc']:.4f}", flush=True)

    winner_item = max(task_valid(stage2) or stage2, key=lambda item: item["metrics"]["uncertainty_unseen_gene_auroc"])
    values = winner_item["config"]
    winner = BBBConfig(values["rho_init"], values["kl_weight"], values["prior_sigma1"])
    print(f"Final BBB fit: {winner.name}", flush=True)
    model, history = fit_bbb(
        x_train, y_train, x_val, y_val, seed=args.seed, epochs=60,
        kl_weight=winner.kl_weight, rho_init=winner.rho_init,
        prior_sigma1=winner.prior_sigma1,
    )
    probability, uncertainty, standard_deviation = model.predict_statistics(
        x_test, draws=48, seed=args.seed + 99
    )
    final = summarize(
        y_test, probability, uncertainty, standard_deviation, test_genes, train_genes
    )
    final["ranking"] = ranking_metrics(y_test, probability, seed=args.seed)
    with args.ensemble_report.open() as handle:
        ensemble = json.load(handle)["holdout"]["deep_ensemble"]
    dense_parameters = 50 * 32 + 32 + 32 * 16 + 16 + 16 + 1
    report = {
        "model": "Dense Bayes by Backprop",
        "grid_size": len(stage1),
        "winner": {**asdict(winner), "name": winner.name},
        "stage1": stage1,
        "stage2": stage2,
        "final_history": history,
        "holdout": {"full_bbb": final, "deep_ensemble": ensemble},
        "parameter_counts": {
            "deterministic_dense_network": dense_parameters,
            "five_member_deep_ensemble": 5 * dense_parameters,
            "dense_bbb_variational_scalars": model.parameter_count(),
            "tuned_low_rank_sbnn_variational_scalars": 1828,
            "full_rank_factorized_sbnn_variational_scalars": 6916,
        },
    }
    with (args.output_dir / "full_bbb_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    np.savez_compressed(
        args.output_dir / "full_bbb_holdout_predictions.npz",
        variation_id=np.asarray([record.variation_id for record in test_records]),
        label=y_test, probability=probability,
        mutual_information=uncertainty, predictive_std=standard_deviation,
    )
    print(json.dumps({"winner": report["winner"], "holdout": report["holdout"], "parameter_counts": report["parameter_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
