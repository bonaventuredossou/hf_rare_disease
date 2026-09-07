#!/usr/bin/env python3
"""Tune the SBNN on one temporal split and test it once on the other split."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from clinvar_temporal_experiment import (
    balanced_cap,
    ece,
    featurize,
    fit_sbnn,
    fmax,
    gene_statistics,
    parse_release,
    ranking_metrics,
    selective_metrics,
)


@dataclass(frozen=True)
class Config:
    name: str
    posterior_scale_multiplier: float
    prior_sigma1: float
    kl_weight: float
    ranks: tuple[int, int, int]
    prior_sigma2: float = math.exp(-6.0)


def candidate_configs() -> list[Config]:
    base = dict(posterior_scale_multiplier=1.0, prior_sigma1=1.0, kl_weight=0.01, ranks=(8, 8, 1))
    values = [Config("baseline", **base)]
    for value in (0.35, 0.60, 1.50):
        values.append(Config(f"posterior_{value:g}", value, 1.0, 0.01, (8, 8, 1)))
    for value in (0.5, 2.0):
        values.append(Config(f"prior_{value:g}", 1.0, value, 0.01, (8, 8, 1)))
    for value in (0.001, 0.003, 0.03):
        values.append(Config(f"kl_{value:g}", 1.0, 1.0, value, (8, 8, 1)))
    for ranks in ((4, 4, 1), (8, 4, 1), (16, 8, 1), (16, 16, 1)):
        values.append(Config("rank_" + "_".join(map(str, ranks)), 1.0, 1.0, 0.01, ranks))
    values.extend(
        [
            Config("mixed_a", 0.60, 2.0, 0.003, (16, 8, 1)),
            Config("mixed_b", 0.35, 0.5, 0.003, (8, 4, 1)),
            Config("mixed_c", 1.50, 2.0, 0.001, (16, 16, 1)),
        ]
    )
    return values


def broad_candidate_configs() -> list[Config]:
    values = [Config("baseline", 1.0, 1.0, 0.01, (8, 8, 1))]
    for kl in (0.03, 0.10, 0.30, 1.0, 3.0):
        values.append(Config(f"kl_{kl:g}", 1.0, 1.0, kl, (8, 8, 1)))
    for scale in (0.10, 0.30, 3.0, 10.0):
        values.append(Config(f"rho_scale_{scale:g}", scale, 1.0, 0.30, (8, 8, 1)))
    for ranks in ((8, 4, 1), (16, 8, 1), (16, 16, 1), (32, 16, 1)):
        values.append(Config("rank_" + "_".join(map(str, ranks)), 1.0, 1.0, 0.30, ranks))
    # A deterministic space-filling design covers interactions without a 500-run grid.
    posterior = (0.10, 0.30, 1.0, 3.0, 10.0)
    kls = (0.03, 0.10, 0.30, 1.0, 3.0)
    ranks = ((8, 4, 1), (8, 8, 1), (16, 8, 1), (16, 16, 1), (32, 16, 1))
    prior1 = (0.3, 1.0, 3.0)
    prior2 = (math.exp(-8.0), math.exp(-6.0), math.exp(-4.0), math.exp(-2.0))
    for i in range(20):
        values.append(
            Config(
                f"interaction_{i + 1:02d}",
                posterior[i % len(posterior)],
                prior1[(2 * i + 1) % len(prior1)],
                kls[(3 * i + 1) % len(kls)],
                ranks[(2 * i + 3) % len(ranks)],
                prior2[(3 * i + 2) % len(prior2)],
            )
        )
    return values


def factorial_candidate_configs() -> list[Config]:
    values = []
    for ranks in ((8, 4, 1), (8, 8, 1), (16, 8, 1), (16, 16, 1), (32, 16, 1)):
        for kl in (0.001, 0.01, 0.1, 1.0, 3.0):
            for posterior in (0.1, 0.3, 1.0, 3.0, 10.0):
                for prior1 in (0.3, 1.0, 3.0):
                    rank_name = "_".join(map(str, ranks))
                    values.append(
                        Config(
                            f"grid_r{rank_name}_k{kl:g}_q{posterior:g}_p{prior1:g}",
                            posterior,
                            prior1,
                            kl,
                            ranks,
                        )
                    )
    return values


def uncertainty_summary(y, p, uncertainty, genes, train_genes):
    unseen = np.asarray([gene not in train_genes for gene in genes])
    maximum_f, _ = fmax(y, p)
    selective = selective_metrics(y, p, uncertainty)
    return {
        "n": int(len(y)),
        "positive_fraction": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "f_max": maximum_f,
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_15": ece(y, p),
        "balanced_accuracy_at_80pct_coverage": selective["balanced_accuracy_at_80pct_coverage"],
        "uncertainty_unseen_gene_auroc": float(roc_auc_score(unseen, uncertainty)),
    }


def selection_score(result):
    return (
        result["nll"]
        + result["brier"]
        + result["ece_15"]
        + 0.20 * (1.0 - result["uncertainty_unseen_gene_auroc"])
        + 0.50 * (1.0 - result["balanced_accuracy_at_80pct_coverage"])
    )


def dispersion_summary(standard_deviation, genes, train_genes):
    unseen = np.asarray([gene not in train_genes for gene in genes])
    return {
        "known_gene_mean_std": float(np.mean(standard_deviation[~unseen])),
        "unseen_gene_mean_std": float(np.mean(standard_deviation[unseen])),
        "known_gene_median_std": float(np.median(standard_deviation[~unseen])),
        "unseen_gene_median_std": float(np.median(standard_deviation[unseen])),
        "std_unseen_gene_auroc": float(roc_auc_score(unseen, standard_deviation)),
    }


def run_config(config, x_train, y_train, x_val, y_val, x_tune, y_tune, tune_genes, train_genes, *, seed, epochs, draws):
    model, history = fit_sbnn(
        x_train,
        y_train,
        x_val,
        y_val,
        seed=seed,
        epochs=epochs,
        kl_weight=config.kl_weight,
        ranks=config.ranks,
        posterior_scale_multiplier=config.posterior_scale_multiplier,
        prior_sigma1=config.prior_sigma1,
        prior_sigma2=config.prior_sigma2,
    )
    prediction, uncertainty = model.predict_samples(x_tune, draws=draws, seed=seed + 99)
    result = uncertainty_summary(y_tune, prediction, uncertainty, tune_genes, train_genes)
    result["score"] = selection_score(result)
    result["epochs_completed"] = history[-1]["epoch"] if history else 0
    return model, result


def eligible(results, baseline):
    return [
        item
        for item in results
        if item["metrics"]["pr_auc"] >= baseline["pr_auc"] - 0.003
        and item["metrics"]["f_max"] >= baseline["f_max"] - 0.005
    ]


def make_shortlist(results, baseline, limit=4):
    pool = eligible(results, baseline) or results
    choices = []
    for key, reverse in (("score", False), ("nll", False), ("uncertainty_unseen_gene_auroc", True)):
        item = sorted(pool, key=lambda value: value["metrics"][key], reverse=reverse)[0]
        if item["config"]["name"] not in {value["config"]["name"] for value in choices}:
            choices.append(item)
    baseline_item = next(item for item in results if item["config"]["name"] == "baseline")
    if baseline_item["config"]["name"] not in {value["config"]["name"] for value in choices}:
        choices.append(baseline_item)
    for item in sorted(pool, key=lambda value: value["metrics"]["score"]):
        if len(choices) >= limit:
            break
        if item["config"]["name"] not in {value["config"]["name"] for value in choices}:
            choices.append(item)
    return choices[:limit]


def make_factorial_shortlist(results, baseline, limit=10):
    pool = eligible(results, baseline) or results
    choices = []
    for key, reverse in (("uncertainty_unseen_gene_auroc", True), ("score", False)):
        item = sorted(pool, key=lambda value: value["metrics"][key], reverse=reverse)[0]
        if item["config"]["name"] not in {value["config"]["name"] for value in choices}:
            choices.append(item)
    baseline_item = min(
        results,
        key=lambda item: (
            tuple(item["config"]["ranks"]) != (8, 8, 1),
            abs(item["config"]["kl_weight"] - 0.01),
            abs(item["config"]["posterior_scale_multiplier"] - 1.0),
            abs(item["config"]["prior_sigma1"] - 1.0),
        ),
    )
    if baseline_item["config"]["name"] not in {value["config"]["name"] for value in choices}:
        choices.append(baseline_item)
    rank_values = sorted({tuple(item["config"]["ranks"]) for item in pool})
    for ranks in rank_values:
        group = [item for item in pool if tuple(item["config"]["ranks"]) == ranks]
        for key, reverse in (("uncertainty_unseen_gene_auroc", True), ("score", False)):
            item = sorted(group, key=lambda value: value["metrics"][key], reverse=reverse)[0]
            if item["config"]["name"] not in {value["config"]["name"] for value in choices}:
                choices.append(item)
    return choices[:limit]


def find_baseline(results):
    return min(
        results,
        key=lambda item: (
            tuple(item["config"]["ranks"]) != (8, 8, 1),
            abs(item["config"]["kl_weight"] - 0.01),
            abs(item["config"]["posterior_scale_multiplier"] - 1.0),
            abs(item["config"]["prior_sigma1"] - 1.0),
        ),
    )["metrics"]


def plot_factorial(output_dir, results):
    ranks = sorted({tuple(item["config"]["ranks"]) for item in results})
    kls = sorted({item["config"]["kl_weight"] for item in results})
    posterior = sorted({item["config"]["posterior_scale_multiplier"] for item in results})
    priors = sorted({item["config"]["prior_sigma1"] for item in results})
    lookup = {
        (
            tuple(item["config"]["ranks"]),
            item["config"]["kl_weight"],
            item["config"]["posterior_scale_multiplier"],
            item["config"]["prior_sigma1"],
        ): item["metrics"]
        for item in results
    }
    for prior in priors:
        fig, axes = plt.subplots(2, len(ranks), figsize=(4 * len(ranks), 8), squeeze=False)
        for column, rank in enumerate(ranks):
            for row, (key, title) in enumerate((("uncertainty_unseen_gene_auroc", "OOD AUROC"), ("nll", "NLL"))):
                matrix = np.asarray([[lookup[(rank, kl, rho, prior)][key] for rho in posterior] for kl in kls])
                image = axes[row, column].imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
                axes[row, column].set_xticks(range(len(posterior)), [str(x) for x in posterior])
                axes[row, column].set_yticks(range(len(kls)), [str(x) for x in kls])
                axes[row, column].set_xlabel("posterior scale multiplier")
                axes[row, column].set_ylabel("KL weight")
                axes[row, column].set_title(f"rank {rank[:2]}: {title}")
                fig.colorbar(image, ax=axes[row, column], fraction=0.046)
        fig.suptitle(f"Full KL x posterior grid, broad prior sigma = {prior}")
        fig.tight_layout()
        fig.savefig(output_dir / f"factorial_heatmaps_prior_{prior:g}.png", dpi=160)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    for rank in ranks:
        group = [item for item in results if tuple(item["config"]["ranks"]) == rank]
        axis.scatter(
            [item["metrics"]["nll"] for item in group],
            [item["metrics"]["uncertainty_unseen_gene_auroc"] for item in group],
            label=str(rank[:2]), alpha=0.65, s=28,
        )
    def front(items):
        ordered = sorted(items, key=lambda item: item["metrics"]["nll"])
        selected = []
        best_ood = -math.inf
        for item in ordered:
            ood = item["metrics"]["uncertainty_unseen_gene_auroc"]
            if ood > best_ood:
                selected.append(item)
                best_ood = ood
        return selected
    all_front = front(results)
    baseline = find_baseline(results)
    valid = eligible(results, baseline)
    valid_front = front(valid)
    axis.plot(
        [item["metrics"]["nll"] for item in all_front],
        [item["metrics"]["uncertainty_unseen_gene_auroc"] for item in all_front],
        color="black", linewidth=1.5, marker="o", markersize=3, label="all-model Pareto front",
    )
    axis.plot(
        [item["metrics"]["nll"] for item in valid_front],
        [item["metrics"]["uncertainty_unseen_gene_auroc"] for item in valid_front],
        color="#C43B3B", linewidth=2.0, marker="o", markersize=4, label="task-valid Pareto front",
    )
    axis.axhline(0.70, color="black", linestyle="--", linewidth=1, label="0.70 target")
    axis.set_xlabel("NLL (lower is better)")
    axis.set_ylabel("Unseen-gene AUROC (higher is better)")
    axis.set_title("SBNN calibration and OOD tradeoff")
    axis.legend(title="layer ranks")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "factorial_pareto_scatter.png", dpi=180)
    plt.close(fig)


def load_baseline_predictions(path, keep_ids):
    output = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            variant_id = int(row["variation_id"])
            if variant_id in keep_ids:
                output[variant_id] = float(row["SBNN"])
    return output


def plot_sweep(output_dir, stage2, baseline, final_result):
    ordered = sorted(stage2, key=lambda item: item["metrics"]["score"])
    names = [item["config"]["name"] for item in ordered]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, key, title in (
        (axes[0], "nll", "Temporal tuning NLL"),
        (axes[1], "ece_15", "Temporal tuning ECE"),
        (axes[2], "uncertainty_unseen_gene_auroc", "Unseen-gene detection"),
    ):
        axis.bar(x, [item["metrics"][key] for item in ordered], color="#466B8A")
        axis.set_xticks(x, names, rotation=35, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "sweep_stage2.png", dpi=180)
    plt.close(fig)

    keys = ("nll", "brier", "ece_15", "pr_auc")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, key in zip(axes, keys):
        axis.bar(["Original", "Tuned"], [baseline[key], final_result[key]], color=["#777777", "#B24C63"])
        axis.set_title(key.replace("_", " "))
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tuned_vs_original_holdout.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--train-cap-per-class", type=int, default=30000)
    parser.add_argument("--screen-cap-per-class", type=int, default=10000)
    parser.add_argument("--screen-validation-cap", type=int, default=3000)
    parser.add_argument("--tune-cap", type=int, default=60000)
    parser.add_argument("--screen-tune-cap", type=int, default=20000)
    parser.add_argument("--screen-draws", type=int, default=12)
    parser.add_argument("--stage1-epochs", type=int, default=14)
    parser.add_argument("--stage2-epochs", type=int, default=36)
    parser.add_argument("--final-epochs", type=int, default=60)
    parser.add_argument("--draws", type=int, default=48)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--search-space", choices=("initial", "broad", "factorial"), default="initial")
    parser.add_argument("--shortlist-size", type=int, default=4)
    parser.add_argument("--stage1-results", type=Path)
    parser.add_argument("--continue-stage1", action="store_true")
    parser.add_argument("--stage2-results", type=Path)
    parser.add_argument("--continue-stage2", action="store_true")
    parser.add_argument("--winner-metric", choices=("score", "uncertainty"), default="score")
    parser.add_argument("--compare-ensemble-std", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading ClinVar snapshots", flush=True)
    historical, historical_ids = parse_release(args.historical, collect_all_ids=True)
    current, _ = parse_release(args.current, collect_all_ids=False)
    temporal = [record for record in current if record.variation_id not in historical_ids]
    historical_y = np.asarray([record.label for record in historical])
    groups = np.asarray([record.gene for record in historical])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_index, val_index = next(splitter.split(np.zeros(len(historical_y)), historical_y, groups))
    train_index = balanced_cap(train_index, historical_y, args.train_cap_per_class, args.seed)
    val_index = balanced_cap(val_index, historical_y, min(args.train_cap_per_class // 4, 15000), args.seed + 1)
    stats = gene_statistics(historical, train_index)
    train_records = [historical[int(i)] for i in train_index]
    val_records = [historical[int(i)] for i in val_index]
    train_genes = {record.gene for record in train_records}

    tune_records_all = [record for record in temporal if record.variation_id % 2 == 0]
    test_records = [record for record in temporal if record.variation_id % 2 == 1]
    rng = np.random.default_rng(args.seed)
    if len(tune_records_all) > args.tune_cap:
        tune_indices = rng.choice(len(tune_records_all), args.tune_cap, replace=False)
        tune_records = [tune_records_all[int(i)] for i in tune_indices]
    else:
        tune_records = tune_records_all

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
    tune_genes = [record.gene for record in tune_records]
    test_genes = [record.gene for record in test_records]

    screen_index = balanced_cap(np.arange(len(y_train)), y_train, args.screen_cap_per_class, args.seed + 2)
    screen_val_index = np.random.default_rng(args.seed + 3).choice(
        len(y_val), min(args.screen_validation_cap, len(y_val)), replace=False
    )
    screen_tune_index = np.random.default_rng(args.seed + 4).choice(
        len(y_tune), min(args.screen_tune_cap, len(y_tune)), replace=False
    )
    screen_tune_genes = [tune_genes[int(i)] for i in screen_tune_index]
    if args.search_space == "factorial":
        configs = factorial_candidate_configs()
    elif args.search_space == "broad":
        configs = broad_candidate_configs()
    else:
        configs = candidate_configs()
    if args.candidate_limit:
        configs = configs[: args.candidate_limit]
    if args.stage1_results:
        with args.stage1_results.open() as handle:
            stage1 = json.load(handle)["stage1"]
        print(f"Stage 1: reused {len(stage1)} saved configurations", flush=True)
        pending = configs if args.continue_stage1 else []
    else:
        stage1 = []
        print(f"Stage 1: {len(configs)} configurations", flush=True)
        pending = configs
    completed = {item["config"]["name"] for item in stage1}
    pending = [config for config in pending if config.name not in completed]
    if pending:
        print(f"Stage 1: running {len(pending)} remaining configurations", flush=True)
        for number, config in enumerate(pending, len(stage1) + 1):
            _, result = run_config(
                config,
                x_train[screen_index], y_train[screen_index], x_val[screen_val_index], y_val[screen_val_index],
                x_tune[screen_tune_index], y_tune[screen_tune_index], screen_tune_genes, train_genes,
                seed=args.seed, epochs=args.stage1_epochs, draws=args.screen_draws,
            )
            stage1.append({"config": asdict(config), "metrics": result})
            with (args.output_dir / "stage1_checkpoint.json").open("w") as handle:
                json.dump({"stage1": stage1}, handle, indent=2)
                handle.write("\n")
            print(f"  {number:03d}/{len(configs)} {config.name}: score={result['score']:.5f}, NLL={result['nll']:.5f}, OOD={result['uncertainty_unseen_gene_auroc']:.4f}", flush=True)

    baseline1 = find_baseline(stage1)
    if args.search_space == "factorial":
        plot_factorial(args.output_dir, stage1)
        shortlist = make_factorial_shortlist(stage1, baseline1, limit=args.shortlist_size)
    else:
        shortlist = make_shortlist(stage1, baseline1, limit=args.shortlist_size)

    if args.stage2_results:
        with args.stage2_results.open() as handle:
            stage2 = json.load(handle)["stage2"]
        print(f"Stage 2: reused {len(stage2)} saved configurations", flush=True)
        pending_stage2 = shortlist if args.continue_stage2 else []
    else:
        stage2 = []
        print("Stage 2: full training subset", flush=True)
        pending_stage2 = shortlist
    completed_stage2 = {item["config"]["name"] for item in stage2}
    pending_stage2 = [item for item in pending_stage2 if item["config"]["name"] not in completed_stage2]
    if pending_stage2:
        print(f"Stage 2: running {len(pending_stage2)} remaining configurations", flush=True)
        for item in pending_stage2:
            config = Config(**{**item["config"], "ranks": tuple(item["config"]["ranks"])})
            _, result = run_config(
                config, x_train, y_train, x_val, y_val, x_tune, y_tune, tune_genes, train_genes,
                seed=args.seed, epochs=args.stage2_epochs, draws=args.draws,
            )
            stage2.append({"config": asdict(config), "metrics": result})
            print(f"  {config.name}: score={result['score']:.5f}, NLL={result['nll']:.5f}, OOD={result['uncertainty_unseen_gene_auroc']:.4f}", flush=True)

    baseline2 = find_baseline(stage2)
    finalists = eligible(stage2, baseline2)
    if args.winner_metric == "uncertainty":
        winner_item = max(finalists or stage2, key=lambda item: item["metrics"]["uncertainty_unseen_gene_auroc"])
    else:
        winner_item = min(finalists or stage2, key=lambda item: item["metrics"]["score"])
    winner = Config(**{**winner_item["config"], "ranks": tuple(winner_item["config"]["ranks"])})
    print(f"Final fit: {winner.name}", flush=True)
    winner_model, history = fit_sbnn(
        x_train, y_train, x_val, y_val,
        seed=args.seed, epochs=args.final_epochs, kl_weight=winner.kl_weight,
        ranks=winner.ranks, posterior_scale_multiplier=winner.posterior_scale_multiplier,
        prior_sigma1=winner.prior_sigma1,
        prior_sigma2=winner.prior_sigma2,
    )
    final_p, final_u, final_std = winner_model.predict_statistics(x_test, draws=args.draws, seed=args.seed + 99)
    final_result = uncertainty_summary(y_test, final_p, final_u, test_genes, train_genes)
    final_result["ranking"] = ranking_metrics(y_test, final_p, seed=args.seed)
    final_result["dispersion"] = dispersion_summary(final_std, test_genes, train_genes)

    ensemble_result = None
    ensemble_std = None
    if args.compare_ensemble_std:
        print("Training five independent networks for the dispersion comparison", flush=True)
        members = []
        for member in range(5):
            model = MLPClassifier(
                hidden_layer_sizes=(32, 16), alpha=1e-4, batch_size=512,
                max_iter=120, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=12, random_state=args.seed + member,
            ).fit(x_train, y_train)
            members.append(model.predict_proba(x_test)[:, 1])
        ensemble_samples = np.stack(members)
        ensemble_p = ensemble_samples.mean(axis=0)
        ensemble_std = ensemble_samples.std(axis=0)
        entropy = lambda p: -(p*np.log(np.clip(p, 1e-12, 1)) + (1-p)*np.log(np.clip(1-p, 1e-12, 1)))
        ensemble_u = np.maximum(entropy(ensemble_p) - entropy(ensemble_samples).mean(axis=0), 0.0)
        ensemble_result = uncertainty_summary(y_test, ensemble_p, ensemble_u, test_genes, train_genes)
        ensemble_result["dispersion"] = dispersion_summary(ensemble_std, test_genes, train_genes)

    baseline_map = load_baseline_predictions(args.baseline_predictions, {record.variation_id for record in test_records})
    baseline_p = np.asarray([baseline_map[record.variation_id] for record in test_records])
    baseline_result = {
        "n": int(len(y_test)),
        "pr_auc": float(average_precision_score(y_test, baseline_p)),
        "f_max": fmax(y_test, baseline_p)[0],
        "nll": float(log_loss(y_test, baseline_p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_test, baseline_p)),
        "ece_15": ece(y_test, baseline_p),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, baseline_p >= 0.5)),
        "ranking": ranking_metrics(y_test, baseline_p, seed=args.seed),
    }
    np.savez_compressed(
        args.output_dir / "tuned_holdout_predictions.npz",
        variation_id=np.asarray([record.variation_id for record in test_records]),
        label=y_test,
        probability=final_p,
        mutual_information=final_u,
        predictive_std=final_std,
        ensemble_predictive_std=ensemble_std if ensemble_std is not None else np.asarray([]),
    )
    report = {
        "protocol": {
            "training": "ClinVar 2023-12-26 only",
            "tuning": "new 2026-08-29 Variation IDs with even ID parity; fixed random sample for the sweep",
            "final_test": "all new 2026-08-29 Variation IDs with odd ID parity; touched once after selection",
            "screen_train_n": int(len(screen_index)),
            "full_train_n": int(len(y_train)),
            "tune_n": int(len(y_tune)),
            "test_n": int(len(y_test)),
            "selection_score": "NLL + Brier + ECE + 0.20*(1-unseen-gene AUROC) + 0.50*(1-balanced accuracy at 80% coverage)",
            "task_guardrail": "PR-AUC no more than 0.003 below baseline and F-max no more than 0.005 below baseline",
            "winner_metric": args.winner_metric,
        },
        "stage1": stage1,
        "stage2": stage2,
        "winner": asdict(winner),
        "winner_history": history,
        "holdout": {"original_sbnn": baseline_result, "tuned_sbnn": final_result, "deep_ensemble": ensemble_result},
    }
    with (args.output_dir / "sweep_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    plot_sweep(args.output_dir, stage2, baseline_result, final_result)
    print(json.dumps(report["holdout"], indent=2), flush=True)


if __name__ == "__main__":
    main()
