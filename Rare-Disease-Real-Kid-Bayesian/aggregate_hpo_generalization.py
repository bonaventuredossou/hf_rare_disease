#!/usr/bin/env python3
"""Summarize repeated candidate-resampling runs of the HPO benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def describe(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    names = list(runs[0]["results"])
    summary = {}
    for name in names:
        model_runs = [run["results"][name] for run in runs]
        summary[name] = {
            "molecular_top1": describe(item["molecular_only"]["top1"] for item in model_runs),
            "hpo_conditioned_top1": describe(item["phenotype_conditioned"]["top1"] for item in model_runs),
            "top1_change": describe(item["paired_top1_test"]["top1_change"] for item in model_runs),
            "hpo_conditioned_top5": describe(item["phenotype_conditioned"]["top5"] for item in model_runs),
            "molecular_two_variant_top1": describe(
                item["molecular_only"]["two_variant_ranking"]["top1"] for item in model_runs
            ),
            "hpo_conditioned_two_variant_top1": describe(
                item["phenotype_conditioned"]["two_variant_ranking"]["top1"] for item in model_runs
            ),
            "molecular_known_gene_top1": describe(
                item["molecular_only"]["phenotype_reference_known"]["top1"] for item in model_runs
            ),
            "hpo_conditioned_known_gene_top1": describe(
                item["phenotype_conditioned"]["phenotype_reference_known"]["top1"] for item in model_runs
            ),
            "molecular_unseen_gene_top1": describe(
                item["molecular_only"]["phenotype_reference_unseen"]["top1"] for item in model_runs
            ),
            "hpo_conditioned_unseen_gene_top1": describe(
                item["phenotype_conditioned"]["phenotype_reference_unseen"]["top1"] for item in model_runs
            ),
            "selected_beta": [item["selected_beta"] for item in model_runs],
            "selected_missing_gene_score": [item["selected_missing_gene_score"] for item in model_runs],
            "paired_test_p_values": [item["paired_top1_test"]["mcnemar_exact_two_sided_p"] for item in model_runs],
        }

    report = {
        "runs": len(runs),
        "candidate_seeds": [run["design"]["candidate_seed"] for run in runs],
        "cases_per_run": runs[0]["data"]["untouched_post_2023_cases"],
        "summary": summary,
        "interpretation": (
            "Means and sample standard deviations reflect five independently sampled benign candidate cohorts. "
            "They measure robustness to candidate composition, not uncertainty from independent patient cohorts."
        ),
    }
    (args.output_dir / "repeated_seed_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    x = np.arange(len(names))
    width = 0.36
    molecular_mean = [summary[name]["molecular_top1"]["mean"] for name in names]
    molecular_std = [summary[name]["molecular_top1"]["std"] for name in names]
    hpo_mean = [summary[name]["hpo_conditioned_top1"]["mean"] for name in names]
    hpo_std = [summary[name]["hpo_conditioned_top1"]["std"] for name in names]
    fig, axis = plt.subplots(figsize=(14, 6.5))
    axis.bar(x - width / 2, molecular_mean, width, yerr=molecular_std, capsize=3, label="molecular only", color="#496A81")
    axis.bar(x + width / 2, hpo_mean, width, yerr=hpo_std, capsize=3, label="molecular + HPO", color="#D17B49")
    axis.axhline(0.1, color="#333333", linestyle="--", linewidth=1, label="random top-1")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean top-1 accuracy across five candidate cohorts")
    axis.set_title("Temporal patient generalization with candidate-resampling robustness")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "repeated_seed_top1.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
