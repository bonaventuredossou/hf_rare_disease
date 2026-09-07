#!/usr/bin/env python3
"""Compare meaningful low-rank choices in a wider Bayesian network."""

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
    BinarySBNN,
    balanced_cap,
    ece,
    featurize,
    fit_bbb,
    fit_sbnn,
    fmax,
    gene_statistics,
    parse_release,
    ranking_metrics,
    selective_metrics,
)


HIDDEN = (200, 100)
RANKS = ((8, 8, 1), (16, 16, 1), (32, 32, 1), (50, 50, 1), (50, 100, 1))


@dataclass(frozen=True)
class RankConfig:
    ranks: tuple[int, int, int]
    kl_weight: float
    posterior_scale_multiplier: float
    prior_sigma1: float = 0.3

    @property
    def name(self):
        rank = "_".join(map(str, self.ranks))
        return f"wide_rank{rank}_kl{self.kl_weight:g}_post{self.posterior_scale_multiplier:g}"


def parameter_count(model):
    return int(sum(value.size for _, value in model.parameters()))


def configs():
    return [
        RankConfig(rank, kl, posterior)
        for rank in RANKS
        for kl in (0.01, 0.1, 1.0)
        for posterior in (0.1, 0.3, 1.0)
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


def run_rank(config, x_train, y_train, x_val, y_val, x_eval, y_eval, genes, train_genes, *, seed, epochs, draws):
    model, history = fit_sbnn(
        x_train, y_train, x_val, y_val,
        seed=seed, epochs=epochs, hidden=HIDDEN, ranks=config.ranks,
        kl_weight=config.kl_weight,
        posterior_scale_multiplier=config.posterior_scale_multiplier,
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


def shortlist(results):
    valid = task_valid(results) or results
    selected = []
    for rank in RANKS:
        group = [item for item in valid if tuple(item["config"]["ranks"]) == rank]
        if not group:
            group = [item for item in results if tuple(item["config"]["ranks"]) == rank]
        for key, reverse in (("uncertainty_unseen_gene_auroc", True), ("score", False)):
            item = sorted(group, key=lambda value: value["metrics"][key], reverse=reverse)[0]
            if item["config"]["name"] not in {value["config"]["name"] for value in selected}:
                selected.append(item)
    return selected


def plot_screen(output_dir, results):
    fig, axes = plt.subplots(1, len(RANKS), figsize=(20, 4.5), sharey=True)
    for axis, rank in zip(axes, RANKS):
        group = [item for item in results if tuple(item["config"]["ranks"]) == rank]
        for kl in (0.01, 0.1, 1.0):
            row = sorted(
                [item for item in group if item["config"]["kl_weight"] == kl],
                key=lambda item: item["config"]["posterior_scale_multiplier"],
            )
            axis.plot(
                [item["config"]["posterior_scale_multiplier"] for item in row],
                [item["metrics"]["uncertainty_unseen_gene_auroc"] for item in row],
                marker="o", label=f"KL {kl:g}",
            )
        axis.axhline(0.70, color="black", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_title(f"rank {rank[:2]}")
        axis.set_xlabel("posterior scale multiplier")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Unseen-gene AUROC")
    axes[-1].legend()
    fig.suptitle("Wide SBNN rank, KL, and posterior-scale interactions")
    fig.tight_layout()
    fig.savefig(output_dir / "wide_rank_screen.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--stage1-results", type=Path)
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

    screen_train = balanced_cap(np.arange(len(y_train)), y_train, 5000, args.seed + 2)
    screen_val = rng.choice(len(y_val), 3000, replace=False)
    screen_tune = rng.choice(len(y_tune), 10000, replace=False)
    screen_genes = [tune_genes[int(i)] for i in screen_tune]
    if args.stage1_results:
        with args.stage1_results.open() as handle:
            stage1 = json.load(handle)["stage1"]
        print(f"Reused {len(stage1)} screened configurations", flush=True)
    else:
        stage1 = []
        candidates = configs()
        print(f"Screening {len(candidates)} wide SBNN configurations", flush=True)
        for number, config in enumerate(candidates, 1):
            _, metrics = run_rank(
                config,
                x_train[screen_train], y_train[screen_train], x_val[screen_val], y_val[screen_val],
                x_tune[screen_tune], y_tune[screen_tune], screen_genes, train_genes,
                seed=args.seed, epochs=16, draws=12,
            )
            item = {"config": {**asdict(config), "name": config.name}, "metrics": metrics}
            stage1.append(item)
            with (args.output_dir / "wide_rank_stage1_checkpoint.json").open("w") as handle:
                json.dump({"stage1": stage1}, handle, indent=2)
                handle.write("\n")
            print(f"  {number:02d}/{len(candidates)} {config.name}: NLL={metrics['nll']:.4f}, OOD={metrics['uncertainty_unseen_gene_auroc']:.4f}", flush=True)
    plot_screen(args.output_dir, stage1)

    stage2 = []
    selected = shortlist(stage1)
    print(f"Confirming {len(selected)} configurations", flush=True)
    for item in selected:
        values = item["config"]
        config = RankConfig(
            tuple(values["ranks"]), values["kl_weight"],
            values["posterior_scale_multiplier"], values["prior_sigma1"],
        )
        _, metrics = run_rank(
            config, x_train, y_train, x_val, y_val,
            x_tune, y_tune, tune_genes, train_genes,
            seed=args.seed, epochs=36, draws=24,
        )
        stage2.append({"config": {**asdict(config), "name": config.name}, "metrics": metrics})
        print(f"  {config.name}: NLL={metrics['nll']:.4f}, OOD={metrics['uncertainty_unseen_gene_auroc']:.4f}", flush=True)

    winner_item = max(task_valid(stage2) or stage2, key=lambda item: item["metrics"]["uncertainty_unseen_gene_auroc"])
    values = winner_item["config"]
    winner = RankConfig(
        tuple(values["ranks"]), values["kl_weight"],
        values["posterior_scale_multiplier"], values["prior_sigma1"],
    )
    print(f"Final wide SBNN: {winner.name}", flush=True)
    sbnn, history = fit_sbnn(
        x_train, y_train, x_val, y_val,
        seed=args.seed, epochs=60, hidden=HIDDEN, ranks=winner.ranks,
        kl_weight=winner.kl_weight,
        posterior_scale_multiplier=winner.posterior_scale_multiplier,
        prior_sigma1=winner.prior_sigma1,
    )
    p, u, sd = sbnn.predict_statistics(x_test, draws=48, seed=args.seed + 99)
    final_sbnn = summarize(y_test, p, u, sd, test_genes, train_genes)
    final_sbnn["ranking"] = ranking_metrics(y_test, p, seed=args.seed)

    print("Training matching-width dense BBB", flush=True)
    bbb, bbb_history = fit_bbb(
        x_train, y_train, x_val, y_val,
        seed=args.seed, epochs=60, hidden=HIDDEN,
        rho_init=-4.0, kl_weight=0.1, prior_sigma1=0.3,
    )
    bp, bu, bsd = bbb.predict_statistics(x_test, draws=48, seed=args.seed + 99)
    final_bbb = summarize(y_test, bp, bu, bsd, test_genes, train_genes)
    final_bbb["ranking"] = ranking_metrics(y_test, bp, seed=args.seed)

    counts = {
        "wide_deterministic_network": 30401,
        "wide_five_member_ensemble": 152005,
        "wide_dense_bbb": parameter_count(BinaryBBB(50, hidden=HIDDEN, seed=0)),
        "wide_low_rank": {
            str(rank): parameter_count(BinarySBNN(50, hidden=HIDDEN, ranks=rank, seed=0))
            for rank in RANKS
        },
    }
    report = {
        "architecture": [50, 200, 100, 1],
        "stage1": stage1,
        "stage2": stage2,
        "winner": {**asdict(winner), "name": winner.name},
        "holdout": {"wide_low_rank_sbnn": final_sbnn, "wide_dense_bbb": final_bbb},
        "parameter_counts": counts,
        "history": {"sbnn": history, "bbb": bbb_history},
    }
    with (args.output_dir / "wide_rank_results.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"winner": report["winner"], "holdout": report["holdout"], "parameter_counts": counts}, indent=2), flush=True)


if __name__ == "__main__":
    main()
