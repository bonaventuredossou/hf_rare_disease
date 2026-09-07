#!/usr/bin/env python3
"""Rebuild the figures used in the hackathon methods report from saved results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent


def read_json(relative_path: str) -> dict:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analysis_funnel(output: Path) -> None:
    scoring = read_json("results/scoring_audit.json")
    annotation = read_json("results/annotation_join_audit.json")
    stages = [
        "WGS variants",
        "Heterozygous alleles",
        "Gene annotations",
        "Same-gene pairs",
        "Strict QC pairs",
        "Final shortlist",
    ]
    counts = [
        5_079_064,
        annotation["matched_heterozygous_alleles"] + annotation["unmatched_heterozygous_alleles"],
        annotation["candidate_gene_annotations"],
        scoring["all_pair_count"],
        scoring["strict_primary_pair_count"],
        6,
    ]
    colors = ["#284B63", "#356B83", "#4D8AA0", "#78A6B5", "#D69F5D", "#B85C38"]
    fig, axis = plt.subplots(figsize=(11.5, 5.8))
    y = np.arange(len(stages))
    axis.barh(y, counts, color=colors)
    axis.set_xscale("log")
    axis.set_yticks(y, stages)
    axis.invert_yaxis()
    axis.set_xlabel("Number retained (log scale)")
    axis.set_title("From 5.1 million variants to six reportable compound-heterozygous hypotheses")
    for position, count in zip(y, counts):
        axis.text(count * 1.08, position, f"{count:,}", va="center", fontsize=10)
    axis.grid(axis="x", alpha=0.2)
    save(fig, output, "analysis_funnel.png")


def system_pipeline(output: Path) -> None:
    stages = [
        ("1  Candidate generation", "FASTQ / VCF\nalignment, calling and QC", "Bonaventure pipeline", "#284B63"),
        ("2  Clinical filtering", "frequency, consequence,\nClinVar, inheritance, phase", "Bonaventure pipeline", "#356B83"),
        ("3  Molecular ranking", "gradient boost, ensemble,\nSBNN and dense BBB", "learned comparison", "#8B5E83"),
        ("4  OOD warning", "ensemble disagreement and\nBayesian posterior uncertainty", "uncertainty layer", "#B85C38"),
        ("5  Patient reasoning", "HPO similarity, disease validity,\ncompound-heterozygous logic", "combined evidence", "#4F8A5B"),
        ("6  Track 1 decision", "ranked candidate pairs, EPCR,\nand evidence trace", "challenge output", "#7A8B3A"),
    ]
    fig, axis = plt.subplots(figsize=(15, 5.2))
    axis.set_xlim(0, len(stages))
    axis.set_ylim(0, 1)
    axis.axis("off")
    for index, (title, body, owner, color) in enumerate(stages):
        x = index + 0.08
        box = FancyBboxPatch(
            (x, 0.24), 0.84, 0.54,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.4,
            edgecolor=color,
            facecolor=color + "18",
        )
        axis.add_patch(box)
        axis.text(x + 0.42, 0.68, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=color)
        axis.text(x + 0.42, 0.49, body, ha="center", va="center", fontsize=9)
        axis.text(x + 0.42, 0.30, owner, ha="center", va="center", fontsize=8.5, color="#555555", style="italic")
        if index < len(stages) - 1:
            axis.annotate("", xy=(index + 1.07, 0.51), xytext=(index + 0.94, 0.51), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#555555"})
    axis.text(3, 0.92, "The submitted system: evidence is narrowed, ranked, checked for uncertainty, and explained", ha="center", fontsize=14, fontweight="bold")
    axis.text(3, 0.10, "Deterministic validity checks remain separate from learned pathogenicity and uncertainty estimates.", ha="center", fontsize=10)
    save(fig, output, "system_pipeline.png")


def track1_scores(output: Path) -> None:
    deterministic = read_json("results/exact_track1_score_audit.json")["results"]["deterministic_genomewide"]
    transferred = read_json(
        "results/track1_temporal_transfer_20260906/track1_temporal_model_audit.json"
    )["results"]["strict"]
    rows = [("Deterministic WGS", deterministic["rank_points"], deterministic["f_max"], deterministic["f_max_threshold"])]
    labels = {
        "gradient_boost": "Gradient boost",
        "dense_mlp": "Dense MLP",
        "deep_ensemble": "Deep ensemble",
        "small_sbnn": "Small SBNN",
        "dense_bbb": "Dense BBB",
        "wide_rank50_sbnn": "Wide rank-50 SBNN",
        "wide_dense_bbb": "Wide dense BBB",
        "temporal_consensus": "Temporal consensus",
    }
    rows.extend(
        (labels[name], item["rank_points"], item["f_max"], item["f_max_threshold"])
        for name, item in transferred.items()
    )
    names = [row[0] for row in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.5, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(rows))
    left.barh(y, [row[1] for row in rows], color="#3D6D8A")
    left.set_yticks(y, names)
    left.invert_yaxis()
    left.set_xlim(0, 105)
    left.set_xlabel("Rank Points")
    left.set_title("All methods recover the accepted pair at rank 1")
    for i, row in enumerate(rows):
        left.text(row[1] - 2, i, f"{row[1]:.0f}", ha="right", va="center", color="white", fontweight="bold")
    thresholds = [row[3] for row in rows]
    right.barh(y, thresholds, color="#D27A45")
    right.set_yticks(y, [""] * len(y))
    right.invert_yaxis()
    right.set_xlim(0, 1.04)
    right.set_xlabel("EPCR threshold at F-max")
    right.set_title("Perfect F-max does not imply equal calibration")
    for i, value in enumerate(thresholds):
        right.text(value + 0.015, i, f"{value:.3f}", va="center", fontsize=9)
    save(fig, output, "track1_scores.png")


def temporal_model_comparison(output: Path) -> None:
    consensus = read_json("results/clinvar_temporal_20260905/consensus_holdout.json")["holdout"]
    full_bbb = read_json("results/full_bbb_temporal_20260906/full_bbb_results.json")["holdout"]
    wide = read_json("results/wide_rank_temporal_20260906/wide_rank_results.json")["holdout"]
    data = {
        "Gradient boost": (consensus["Gradient boost"]["ranking"]["top1"], consensus["Gradient boost"]["f_max"], np.nan),
        "Deep ensemble": (consensus["Deep ensemble"]["ranking"]["top1"], consensus["Deep ensemble"]["f_max"], full_bbb["deep_ensemble"]["uncertainty_unseen_gene_auroc"]),
        "Small SBNN": (consensus["SBNN"]["ranking"]["top1"], consensus["SBNN"]["f_max"], 0.6020),
        "Consensus": (consensus["Selected consensus"]["ranking"]["top1"], consensus["Selected consensus"]["f_max"], np.nan),
        "Dense BBB": (full_bbb["full_bbb"]["ranking"]["top1"], full_bbb["full_bbb"]["f_max"], full_bbb["full_bbb"]["uncertainty_unseen_gene_auroc"]),
        "Wide rank-50 SBNN": (wide["wide_low_rank_sbnn"]["ranking"]["top1"], wide["wide_low_rank_sbnn"]["f_max"], wide["wide_low_rank_sbnn"]["uncertainty_unseen_gene_auroc"]),
        "Wide dense BBB": (wide["wide_dense_bbb"]["ranking"]["top1"], wide["wide_dense_bbb"]["f_max"], wide["wide_dense_bbb"]["uncertainty_unseen_gene_auroc"]),
    }
    names = list(data)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    metrics = [(0, "Top-1 among 100", (0.75, 0.87)), (1, "F-max", (0.89, 0.93)), (2, "Unseen-gene OOD AUROC", (0.55, 0.78))]
    colors = ["#477998", "#477998", "#8B5E83", "#7A8B3A", "#8B5E83", "#B85C38", "#8B5E83"]
    for axis, (index, title, limits) in zip(axes, metrics):
        values = [data[name][index] for name in names]
        axis.bar(x, values, color=colors)
        axis.set_xticks(x, [name.replace(" ", "\n") for name in names], fontsize=8)
        axis.set_ylim(*limits)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
        for i, value in enumerate(values):
            if np.isfinite(value):
                axis.text(i, value + (limits[1] - limits[0]) * 0.018, f"{value:.3f}", ha="center", fontsize=8)
    fig.suptitle("Untouched odd-ID temporal holdout: ranking and uncertainty answer different questions", fontsize=14)
    save(fig, output, "temporal_model_comparison.png")


def parameter_tradeoff(output: Path) -> None:
    full = read_json("results/full_bbb_temporal_20260906/full_bbb_results.json")
    wide = read_json("results/wide_rank_temporal_20260906/wide_rank_results.json")
    points = {
        "Deep ensemble": (full["parameter_counts"]["five_member_deep_ensemble"], 0.8025, 0.7485),
        "Small SBNN": (2212, 0.8210, 0.6020),
        "Dense BBB": (full["parameter_counts"]["dense_bbb_variational_scalars"], 0.8300, 0.6665),
        "Wide rank-50 SBNN": (wide["parameter_counts"]["wide_low_rank"]["(50, 50, 1)"], 0.8435, 0.6284),
        "Wide dense BBB": (wide["parameter_counts"]["wide_dense_bbb"], 0.8135, 0.6567),
    }
    fig, axis = plt.subplots(figsize=(9, 6))
    for name, (parameters, top1, ood) in points.items():
        axis.scatter(parameters, top1, s=90 + 500 * (ood - 0.58), c=ood, vmin=0.58, vmax=0.76, cmap="viridis", edgecolor="black")
        axis.annotate(name, (parameters, top1), xytext=(6, 5), textcoords="offset points", fontsize=9)
    axis.set_xscale("log")
    axis.set_xlabel("Fitted scalar parameters (log scale)")
    axis.set_ylabel("Top-1 accuracy among 100 candidates")
    axis.set_ylim(0.79, 0.855)
    axis.set_title("More parameters improve ranking only up to a point")
    axis.grid(alpha=0.2)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0.58, 0.76), cmap="viridis"), ax=axis)
    colorbar.set_label("Unseen-gene OOD AUROC")
    save(fig, output, "parameter_tradeoff.png")


def external_generalization(output: Path) -> None:
    report = read_json(
        "results/phenopacket_hpo2023_generalization_repeated_20260906/repeated_seed_summary.json"
    )["summary"]
    names = list(report)
    labels = [name.replace("_", "\n") for name in names]
    x = np.arange(len(names))
    width = 0.36
    molecular = [report[name]["molecular_top1"]["mean"] for name in names]
    molecular_sd = [report[name]["molecular_top1"]["std"] for name in names]
    hpo = [report[name]["hpo_conditioned_top1"]["mean"] for name in names]
    hpo_sd = [report[name]["hpo_conditioned_top1"]["std"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].bar(x - width / 2, molecular, width, yerr=molecular_sd, capsize=3, label="molecular only", color="#496A81")
    axes[0].bar(x + width / 2, hpo, width, yerr=hpo_sd, capsize=3, label="molecular + HPO", color="#D17B49")
    axes[0].axhline(0.1, color="#333333", linestyle="--", linewidth=1, label="random")
    axes[0].set_xticks(x, labels, fontsize=8)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_title("All 719 temporally new solved cases")
    axes[0].legend()
    pair_molecular = [report[name]["molecular_two_variant_top1"]["mean"] for name in names]
    pair_hpo = [report[name]["hpo_conditioned_two_variant_top1"]["mean"] for name in names]
    axes[1].bar(x - width / 2, pair_molecular, width, label="molecular only", color="#496A81")
    axes[1].bar(x + width / 2, pair_hpo, width, label="molecular + HPO", color="#D17B49")
    axes[1].axhline(0.1, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, labels, fontsize=8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Top-1 accuracy")
    axes[1].set_title("64 challenge-like two-variant cases")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("External Phenopacket generalization across five resampled candidate cohorts", fontsize=14)
    save(fig, output, "external_generalization.png")


def phenotype_coverage(output: Path) -> None:
    report = read_json(
        "results/phenopacket_hpo2023_generalization_repeated_20260906/repeated_seed_summary.json"
    )["summary"]
    names = list(report)
    x = np.arange(len(names))
    width = 0.36
    known = [report[name]["hpo_conditioned_known_gene_top1"]["mean"] for name in names]
    unseen = [report[name]["hpo_conditioned_unseen_gene_top1"]["mean"] for name in names]
    fig, axis = plt.subplots(figsize=(13.5, 5.8))
    axis.bar(x - width / 2, known, width, label="causal gene represented in older phenotype cases", color="#4F8A5B")
    axis.bar(x + width / 2, unseen, width, label="causal gene absent from phenotype reference", color="#9A6B78")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Phenotype-conditioned top-1 accuracy")
    axis.set_title("Phenotype matching is powerful for known genes, but it does not solve novel genes")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    save(fig, output, "phenotype_coverage.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_pipeline(args.output_dir)
    analysis_funnel(args.output_dir)
    track1_scores(args.output_dir)
    temporal_model_comparison(args.output_dir)
    parameter_tradeoff(args.output_dir)
    external_generalization(args.output_dir)
    phenotype_coverage(args.output_dir)
    print(f"Wrote seven figures to {args.output_dir}")


if __name__ == "__main__":
    main()
