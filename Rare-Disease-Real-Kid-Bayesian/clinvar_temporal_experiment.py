#!/usr/bin/env python3
"""Temporal ClinVar benchmark for deterministic and Bayesian variant models.

The historical release defines training data. The later release contributes only
variants whose ClinVar Variation ID did not exist in the historical snapshot.
No CLNSIG field or disease name is used as an input feature.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from sbnn_pair_ranker import Adam, DenseBayesianLayer, SingularBayesianLayer, _sigmoid


POSITIVE_LABELS = {"pathogenic", "likely_pathogenic", "pathogenic/likely_pathogenic"}
NEGATIVE_LABELS = {"benign", "likely_benign", "benign/likely_benign"}

CONSEQUENCES = (
    "missense",
    "frameshift",
    "nonsense",
    "stop_gained",
    "splice_donor",
    "splice_acceptor",
    "splice_region",
    "inframe",
    "synonymous",
    "intron",
    "utr",
    "non-coding",
    "regulatory",
)
VARIANT_TYPES = (
    "single_nucleotide_variant",
    "deletion",
    "insertion",
    "indel",
    "duplication",
    "microsatellite",
    "copy_number",
)
SUBSTITUTIONS = tuple(a + b for a in "ACGT" for b in "ACGT" if a != b)


def parse_info(value: str) -> dict[str, str]:
    result = {}
    for field in value.split(";"):
        if "=" in field:
            key, item = field.split("=", 1)
            result[key] = item
    return result


def clean_label(value: str | None) -> int | None:
    if not value:
        return None
    values = {part.strip().lower() for part in value.split("|")}
    if values and values <= POSITIVE_LABELS:
        return 1
    if values and values <= NEGATIVE_LABELS:
        return 0
    return None


def first_gene(info: dict[str, str]) -> str:
    value = info.get("GENEINFO", "")
    return value.split("|")[0].split(":")[0] or "UNKNOWN"


@dataclass(frozen=True)
class Variant:
    variation_id: int
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: str
    label: int
    consequence: str
    variant_type: str
    review: str
    origin: int
    af_esp: float | None
    af_exac: float | None
    af_tgp: float | None
    has_rs: bool


def parse_release(path: Path, *, collect_all_ids: bool) -> tuple[list[Variant], set[int]]:
    records = []
    all_ids: set[int] = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, pos, identifier, ref, alt, _, _, info_text = line.rstrip().split("\t")[:8]
            if not identifier.isdigit():
                continue
            variation_id = int(identifier)
            if collect_all_ids:
                all_ids.add(variation_id)
            info = parse_info(info_text)
            label = clean_label(info.get("CLNSIG"))
            if label is None or "," in alt:
                continue
            def frequency(name: str) -> float | None:
                try:
                    return float(info[name])
                except (KeyError, ValueError):
                    return None
            try:
                origin = int(info.get("ORIGIN", "0").split(",")[0])
            except ValueError:
                origin = 0
            records.append(
                Variant(
                    variation_id=variation_id,
                    chrom=chrom,
                    pos=int(pos),
                    ref=ref.upper(),
                    alt=alt.upper(),
                    gene=first_gene(info),
                    label=label,
                    consequence=info.get("MC", "").lower(),
                    variant_type=info.get("CLNVC", "").lower(),
                    review=info.get("CLNREVSTAT", "").lower(),
                    origin=origin,
                    af_esp=frequency("AF_ESP"),
                    af_exac=frequency("AF_EXAC"),
                    af_tgp=frequency("AF_TGP"),
                    has_rs="RS" in info,
                )
            )
    return records, all_ids


def gene_statistics(records: list[Variant], indices: np.ndarray) -> dict[str, tuple[int, float]]:
    counts: dict[str, list[int]] = {}
    for index in indices:
        record = records[int(index)]
        counts.setdefault(record.gene, [0, 0])
        counts[record.gene][0] += 1
        counts[record.gene][1] += record.label
    return {
        gene: (total, (positive + 2.0) / (total + 4.0))
        for gene, (total, positive) in counts.items()
    }


def feature_names() -> list[str]:
    numeric = [
        "log_ref_length",
        "log_alt_length",
        "log_length_change",
        "is_snv",
        "is_transition",
        "max_population_af",
        "log10_population_af",
        "population_af_missing",
        "has_dbsnp_id",
        "origin_germline",
        "origin_inherited",
        "origin_de_novo",
        "review_multiple_submitters",
        "review_expert_panel",
        "review_practice_guideline",
        "log_gene_training_count",
        "gene_training_pathogenic_rate",
        "gene_unseen",
    ]
    return numeric + [f"mc_{x}" for x in CONSEQUENCES] + [f"type_{x}" for x in VARIANT_TYPES] + [f"sub_{x}" for x in SUBSTITUTIONS]


def featurize(records: list[Variant], stats: dict[str, tuple[int, float]]) -> np.ndarray:
    rows = []
    transitions = {"AG", "GA", "CT", "TC"}
    for record in records:
        frequencies = [x for x in (record.af_esp, record.af_exac, record.af_tgp) if x is not None]
        maximum = max(frequencies, default=0.0)
        count, rate = stats.get(record.gene, (0, 0.5))
        substitution = record.ref + record.alt if len(record.ref) == len(record.alt) == 1 else ""
        row = [
            math.log1p(len(record.ref)),
            math.log1p(len(record.alt)),
            math.log1p(abs(len(record.ref) - len(record.alt))),
            float(len(record.ref) == len(record.alt) == 1),
            float(substitution in transitions),
            maximum,
            math.log10(maximum + 1e-7),
            float(not frequencies),
            float(record.has_rs),
            float(bool(record.origin & 1)),
            float(bool(record.origin & 4)),
            float(bool(record.origin & 32)),
            float("multiple_submitters" in record.review),
            float("expert_panel" in record.review),
            float("practice_guideline" in record.review),
            math.log1p(count),
            rate,
            float(count == 0),
        ]
        row.extend(float(item in record.consequence) for item in CONSEQUENCES)
        row.extend(float(item in record.variant_type) for item in VARIANT_TYPES)
        row.extend(float(item == substitution) for item in SUBSTITUTIONS)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


class BinarySBNN:
    def __init__(
        self,
        input_dim: int,
        hidden=(32, 16),
        ranks=(8, 8, 1),
        seed=0,
        posterior_scale_multiplier=1.0,
        prior_sigma1=1.0,
        prior_sigma2=math.exp(-6.0),
    ):
        rng = np.random.default_rng(seed)
        widths = (input_dim, *hidden, 1)
        self.layers = [
            SingularBayesianLayer(
                widths[i],
                widths[i + 1],
                ranks[i],
                rng,
                relu=i < len(hidden),
                posterior_scale_multiplier=posterior_scale_multiplier,
                prior_sigma1=prior_sigma1,
                prior_sigma2=prior_sigma2,
            )
            for i in range(len(widths) - 1)
        ]

    def parameters(self):
        for i, layer in enumerate(self.layers):
            for name in ("A", "B", "bias"):
                posterior = getattr(layer, name)
                yield f"layer{i}.{name}.mu", posterior.mu
                yield f"layer{i}.{name}.rho", posterior.rho

    def forward(self, x: np.ndarray, rng: np.random.Generator):
        h = x
        cache = []
        for i, layer in enumerate(self.layers):
            sampled = layer.sample(rng)
            z = h @ sampled["weight"] + sampled["bias"]
            cache.append((h, z, sampled))
            h = np.maximum(z, 0.0) if i < len(self.layers) - 1 else z
        return h[:, 0], cache

    def loss_and_gradients(self, x, y, rng, kl_scale):
        logits, cache = self.forward(x, rng)
        probabilities = _sigmoid(logits)
        data_loss = -float(np.mean(y * np.log(np.clip(probabilities, 1e-12, 1)) + (1-y) * np.log(np.clip(1-probabilities, 1e-12, 1))))
        gradient = ((probabilities - y) / len(y))[:, None]
        layer_data = []
        for i in range(len(self.layers) - 1, -1, -1):
            h, _, sampled = cache[i]
            grad_weight = h.T @ gradient
            grad_bias = gradient.sum(axis=0)
            layer_data.append({
                "A.mu": grad_weight @ sampled["B"],
                "B.mu": grad_weight.T @ sampled["A"],
                "bias.mu": grad_bias,
                "A.rho": (grad_weight @ sampled["B"]) * sampled["A_epsilon"] * _sigmoid(self.layers[i].A.rho),
                "B.rho": (grad_weight.T @ sampled["A"]) * sampled["B_epsilon"] * _sigmoid(self.layers[i].B.rho),
                "bias.rho": grad_bias * sampled["bias_epsilon"] * _sigmoid(self.layers[i].bias.rho),
            })
            gradient = gradient @ sampled["weight"].T
            if i:
                gradient *= cache[i - 1][1] > 0
        layer_data.reverse()
        gradients = {}
        kl = 0.0
        for i, (layer, (_, _, sampled), data_gradient) in enumerate(zip(self.layers, cache, layer_data)):
            layer_kl, layer_gradient = layer.kl_and_gradients(sampled)
            kl += layer_kl
            for name in data_gradient:
                gradients[f"layer{i}.{name}"] = data_gradient[name] + kl_scale * layer_gradient[name]
        return data_loss + kl_scale * kl, gradients

    def predict_samples(self, x, draws=64, seed=0):
        rng = np.random.default_rng(seed)
        samples = np.stack([_sigmoid(self.forward(x, rng)[0]) for _ in range(draws)])
        mean = samples.mean(axis=0)
        entropy = lambda p: -(p*np.log(np.clip(p, 1e-12, 1)) + (1-p)*np.log(np.clip(1-p, 1e-12, 1)))
        return mean, np.maximum(entropy(mean) - entropy(samples).mean(axis=0), 0.0)

    def predict_statistics(self, x, draws=64, seed=0):
        rng = np.random.default_rng(seed)
        samples = np.stack([_sigmoid(self.forward(x, rng)[0]) for _ in range(draws)])
        mean = samples.mean(axis=0)
        entropy = lambda p: -(p*np.log(np.clip(p, 1e-12, 1)) + (1-p)*np.log(np.clip(1-p, 1e-12, 1)))
        mutual_information = np.maximum(entropy(mean) - entropy(samples).mean(axis=0), 0.0)
        return mean, mutual_information, samples.std(axis=0)


class BinaryBBB:
    """Dense Bayes-by-Backprop network with independent weight posteriors."""

    def __init__(
        self,
        input_dim: int,
        hidden=(32, 16),
        seed=0,
        rho_init=-5.0,
        prior_sigma1=1.0,
        prior_sigma2=math.exp(-6.0),
    ):
        rng = np.random.default_rng(seed)
        widths = (input_dim, *hidden, 1)
        self.layers = [
            DenseBayesianLayer(
                widths[i], widths[i + 1], rng,
                relu=i < len(hidden), rho_init=rho_init,
                prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2,
            )
            for i in range(len(widths) - 1)
        ]

    def parameters(self):
        for i, layer in enumerate(self.layers):
            for name in ("weight", "bias"):
                posterior = getattr(layer, name)
                yield f"layer{i}.{name}.mu", posterior.mu
                yield f"layer{i}.{name}.rho", posterior.rho

    def parameter_count(self):
        return int(sum(value.size for _, value in self.parameters()))

    def forward(self, x, rng):
        h = x
        cache = []
        for i, layer in enumerate(self.layers):
            sampled = layer.sample(rng)
            z = h @ sampled["weight"] + sampled["bias"]
            cache.append((h, z, sampled))
            h = np.maximum(z, 0.0) if i < len(self.layers) - 1 else z
        return h[:, 0], cache

    def loss_and_gradients(self, x, y, rng, kl_scale):
        logits, cache = self.forward(x, rng)
        probabilities = _sigmoid(logits)
        data_loss = -float(np.mean(
            y * np.log(np.clip(probabilities, 1e-12, 1))
            + (1-y) * np.log(np.clip(1-probabilities, 1e-12, 1))
        ))
        gradient = ((probabilities - y) / len(y))[:, None]
        layer_data = []
        for i in range(len(self.layers) - 1, -1, -1):
            h, _, sampled = cache[i]
            grad_weight = h.T @ gradient
            grad_bias = gradient.sum(axis=0)
            layer_data.append({
                "weight.mu": grad_weight,
                "bias.mu": grad_bias,
                "weight.rho": grad_weight * sampled["weight_epsilon"] * _sigmoid(self.layers[i].weight.rho),
                "bias.rho": grad_bias * sampled["bias_epsilon"] * _sigmoid(self.layers[i].bias.rho),
            })
            gradient = gradient @ sampled["weight"].T
            if i:
                gradient *= cache[i - 1][1] > 0
        layer_data.reverse()
        gradients = {}
        kl = 0.0
        for i, (layer, (_, _, sampled), data_gradient) in enumerate(zip(self.layers, cache, layer_data)):
            layer_kl, layer_gradient = layer.kl_and_gradients(sampled)
            kl += layer_kl
            for name in data_gradient:
                gradients[f"layer{i}.{name}"] = data_gradient[name] + kl_scale * layer_gradient[name]
        return data_loss + kl_scale * kl, gradients

    def predict_statistics(self, x, draws=64, seed=0):
        rng = np.random.default_rng(seed)
        samples = np.stack([_sigmoid(self.forward(x, rng)[0]) for _ in range(draws)])
        mean = samples.mean(axis=0)
        entropy = lambda p: -(p*np.log(np.clip(p, 1e-12, 1)) + (1-p)*np.log(np.clip(1-p, 1e-12, 1)))
        mutual_information = np.maximum(entropy(mean) - entropy(samples).mean(axis=0), 0.0)
        return mean, mutual_information, samples.std(axis=0)


def fit_sbnn(
    x,
    y,
    x_val,
    y_val,
    *,
    seed,
    epochs=80,
    batch_size=1024,
    kl_weight=0.01,
    ranks=(8, 8, 1),
    posterior_scale_multiplier=1.0,
    prior_sigma1=1.0,
    prior_sigma2=math.exp(-6.0),
    hidden=(32, 16),
):
    model = BinarySBNN(
        x.shape[1],
        hidden=hidden,
        ranks=ranks,
        seed=seed,
        posterior_scale_multiplier=posterior_scale_multiplier,
        prior_sigma1=prior_sigma1,
        prior_sigma2=prior_sigma2,
    )
    optimizer = Adam(model.parameters(), learning_rate=8e-4)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_parameters = None
    patience = 12
    stale = 0
    history = []
    for epoch in range(epochs):
        for start in range(0, len(y), batch_size):
            if start == 0:
                order = rng.permutation(len(y))
            index = order[start:start + batch_size]
            warmup = min(1.0, (epoch + 1) / 10.0)
            _, gradients = model.loss_and_gradients(x[index], y[index], rng, warmup * kl_weight / len(y))
            optimizer.step(gradients)
        if epoch % 2 == 1:
            prediction, _ = model.predict_samples(x_val, draws=16, seed=seed + epoch)
            value = log_loss(y_val, prediction, labels=[0, 1])
            history.append({"epoch": epoch + 1, "validation_nll": value})
            if value < best_loss - 1e-4:
                best_loss = value
                best_parameters = {name: parameter.copy() for name, parameter in model.parameters()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_parameters is not None:
        for name, parameter in model.parameters():
            parameter[...] = best_parameters[name]
    return model, history


def fit_bbb(
    x, y, x_val, y_val, *, seed, epochs=80, batch_size=1024,
    kl_weight=0.01, rho_init=-5.0, prior_sigma1=1.0,
    prior_sigma2=math.exp(-6.0), hidden=(32, 16),
):
    model = BinaryBBB(
        x.shape[1], hidden=hidden, seed=seed, rho_init=rho_init,
        prior_sigma1=prior_sigma1, prior_sigma2=prior_sigma2,
    )
    optimizer = Adam(model.parameters(), learning_rate=8e-4)
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_parameters = None
    stale = 0
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch_size):
            index = order[start:start + batch_size]
            warmup = min(1.0, (epoch + 1) / 10.0)
            _, gradients = model.loss_and_gradients(
                x[index], y[index], rng, warmup * kl_weight / len(y)
            )
            optimizer.step(gradients)
        if epoch % 2 == 1:
            prediction = model.predict_statistics(x_val, draws=16, seed=seed + epoch)[0]
            value = log_loss(y_val, prediction, labels=[0, 1])
            history.append({"epoch": epoch + 1, "validation_nll": value})
            if value < best_loss - 1e-4:
                best_loss = value
                best_parameters = {name: parameter.copy() for name, parameter in model.parameters()}
                stale = 0
            else:
                stale += 1
                if stale >= 12:
                    break
    if best_parameters is not None:
        for name, parameter in model.parameters():
            parameter[...] = best_parameters[name]
    return model, history


def ece(y, p, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def fmax(y, p):
    order = np.argsort(-p)
    y_ordered = y[order]
    tp = np.cumsum(y_ordered)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / max(int(y.sum()), 1)
    values = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    index = int(np.argmax(values))
    return float(values[index]), float(p[order[index]])


def ranking_metrics(y, p, seed=0, negatives_per_case=99, max_cases=2000):
    rng = np.random.default_rng(seed)
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]
    if len(positives) > max_cases:
        positives = rng.choice(positives, max_cases, replace=False)
    ranks = []
    for positive in positives:
        sampled = rng.choice(negatives, negatives_per_case, replace=len(negatives) < negatives_per_case)
        scores = np.concatenate(([p[positive]], p[sampled]))
        ranks.append(1 + int(np.sum(scores[1:] > scores[0])) + 0.5 * int(np.sum(scores[1:] == scores[0])))
    ranks = np.asarray(ranks)
    return {
        "cases": int(len(ranks)),
        "candidates_per_case": negatives_per_case + 1,
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "top10": float(np.mean(ranks <= 10)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def selective_metrics(y, p, uncertainty):
    order = np.argsort(uncertainty)
    result = {}
    for coverage in (0.5, 0.8, 1.0):
        n = max(1, int(round(coverage * len(y))))
        index = order[:n]
        result[f"balanced_accuracy_at_{int(coverage*100)}pct_coverage"] = float(balanced_accuracy_score(y[index], p[index] >= 0.5))
    return result


def metrics(y, p, *, uncertainty, genes, train_genes, seed):
    prediction = p >= 0.5
    maximum_f, threshold = fmax(y, p)
    result = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "f1_at_0.5": float(f1_score(y, prediction)),
        "f_max": maximum_f,
        "f_max_threshold": threshold,
        "nll": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "ece_15": ece(y, p),
        "ranking": ranking_metrics(y, p, seed=seed),
        "selective": selective_metrics(y, p, uncertainty),
    }
    unseen = np.asarray([gene not in train_genes for gene in genes])
    result["unseen_gene_fraction"] = float(unseen.mean())
    if unseen.any() and (~unseen).any():
        result["uncertainty_unseen_gene_auroc"] = float(roc_auc_score(unseen, uncertainty))
        for name, mask in (("known_gene", ~unseen), ("unseen_gene", unseen)):
            if len(np.unique(y[mask])) == 2:
                result[name] = {
                    "n": int(mask.sum()),
                    "roc_auc": float(roc_auc_score(y[mask], p[mask])),
                    "nll": float(log_loss(y[mask], p[mask], labels=[0, 1])),
                    "brier": float(brier_score_loss(y[mask], p[mask])),
                    "ece_15": ece(y[mask], p[mask]),
                }
    return result


def balanced_cap(indices, labels, cap_per_class, seed):
    rng = np.random.default_rng(seed)
    selected = []
    for value in (0, 1):
        group = indices[labels[indices] == value]
        if len(group) > cap_per_class:
            group = rng.choice(group, cap_per_class, replace=False)
        selected.append(group)
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def plot_results(output_dir, y, predictions, uncertainties):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    names = list(predictions)
    x = np.arange(len(names))
    rank_top1 = [ranking_metrics(y, predictions[name], seed=17)["top1"] for name in names]
    briers = [brier_score_loss(y, predictions[name]) for name in names]
    retained = [selective_metrics(y, predictions[name], uncertainties[name])["balanced_accuracy_at_50pct_coverage"] for name in names]
    for axis, values, title, ylabel in zip(
        axes,
        (rank_top1, briers, retained),
        ("Genome-style ranking", "Probability quality", "Selective referral"),
        ("Top-1 rate among 100 candidates", "Brier score (lower is better)", "Balanced accuracy at 50% coverage"),
    ):
        axis.bar(x, values, color=["#355C7D", "#6C5B7B", "#C06C84", "#F67280", "#2A9D8F"][:len(names)])
        axis.set_xticks(x, names, rotation=30, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 6))
    for name, probability in predictions.items():
        observed, predicted = calibration_curve(y, probability, n_bins=12, strategy="quantile")
        axis.plot(predicted, observed, marker="o", label=name)
    axis.plot([0, 1], [0, 1], "k--", label="ideal")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed pathogenic fraction")
    axis.set_title("Temporal ClinVar calibration")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--cap-per-class", type=int, default=50000)
    parser.add_argument("--sbnn-epochs", type=int, default=80)
    parser.add_argument("--draws", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    historical, historical_ids = parse_release(args.historical, collect_all_ids=True)
    current, _ = parse_release(args.current, collect_all_ids=False)
    temporal_test = [record for record in current if record.variation_id not in historical_ids]
    labels = np.asarray([record.label for record in historical])
    groups = np.asarray([record.gene for record in historical])
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_index, validation_index = next(splitter.split(np.zeros(len(labels)), labels, groups))
    train_index = balanced_cap(train_index, labels, args.cap_per_class, args.seed)
    validation_index = balanced_cap(validation_index, labels, min(args.cap_per_class // 4, 15000), args.seed + 1)

    stats = gene_statistics(historical, train_index)
    train_records = [historical[int(i)] for i in train_index]
    validation_records = [historical[int(i)] for i in validation_index]
    test_records = temporal_test
    x_train = featurize(train_records, stats)
    x_validation = featurize(validation_records, stats)
    x_test = featurize(test_records, stats)
    y_train = np.asarray([record.label for record in train_records])
    y_validation = np.asarray([record.label for record in validation_records])
    y_test = np.asarray([record.label for record in test_records])
    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train).astype(np.float32)
    x_validation_scaled = scaler.transform(x_validation).astype(np.float32)
    x_test_scaled = scaler.transform(x_test).astype(np.float32)

    predictions = {}
    uncertainties = {}

    tree = HistGradientBoostingClassifier(max_iter=220, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=args.seed).fit(x_train, y_train)
    predictions["Gradient boost"] = tree.predict_proba(x_test)[:, 1]
    uncertainties["Gradient boost"] = 1.0 - np.abs(2.0 * predictions["Gradient boost"] - 1.0)

    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-4, batch_size=512, max_iter=120, early_stopping=True, validation_fraction=0.1, n_iter_no_change=12, random_state=args.seed).fit(x_train_scaled, y_train)
    predictions["Dense MLP"] = mlp.predict_proba(x_test_scaled)[:, 1]
    uncertainties["Dense MLP"] = 1.0 - np.abs(2.0 * predictions["Dense MLP"] - 1.0)

    ensemble_members = []
    for member in range(5):
        model = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-4, batch_size=512, max_iter=120, early_stopping=True, validation_fraction=0.1, n_iter_no_change=12, random_state=args.seed + member).fit(x_train_scaled, y_train)
        ensemble_members.append(model.predict_proba(x_test_scaled)[:, 1])
    ensemble_samples = np.stack(ensemble_members)
    predictions["Deep ensemble"] = ensemble_samples.mean(axis=0)
    entropy = lambda p: -(p*np.log(np.clip(p, 1e-12, 1)) + (1-p)*np.log(np.clip(1-p, 1e-12, 1)))
    uncertainties["Deep ensemble"] = np.maximum(entropy(predictions["Deep ensemble"]) - entropy(ensemble_samples).mean(axis=0), 0.0)

    sbnn, history = fit_sbnn(x_train_scaled, y_train, x_validation_scaled, y_validation, seed=args.seed, epochs=args.sbnn_epochs)
    predictions["SBNN"] , uncertainties["SBNN"] = sbnn.predict_samples(x_test_scaled, draws=args.draws, seed=args.seed + 99)

    train_genes = set(record.gene for record in train_records)
    genes_test = [record.gene for record in test_records]
    report = {
        "experiment": "ClinVar temporal pathogenicity and ranking benchmark",
        "historical_release": args.historical.name,
        "current_release": args.current.name,
        "seed": args.seed,
        "feature_names": feature_names(),
        "data": {
            "historical_labeled_total": len(historical),
            "historical_all_variant_ids": len(historical_ids),
            "training": {"n": len(y_train), "positive": int(y_train.sum()), "genes": len(train_genes)},
            "validation_gene_disjoint": {"n": len(y_validation), "positive": int(y_validation.sum()), "genes": len(set(record.gene for record in validation_records))},
            "temporal_test_new_ids": {"n": len(y_test), "positive": int(y_test.sum()), "genes": len(set(genes_test))},
        },
        "models": {
            name: metrics(y_test, probability, uncertainty=uncertainties[name], genes=genes_test, train_genes=train_genes, seed=args.seed)
            for name, probability in predictions.items()
        },
        "sbnn": {
            "architecture": [x_train.shape[1], 32, 16, 1],
            "ranks": [8, 8, 1],
            "posterior_draws": args.draws,
            "history": history,
        },
        "limitations": [
            "This is allele-level ClinVar classification, not patient-level causal diagnosis.",
            "ClinVar labels and annotation density are affected by ascertainment and submission practices.",
            "Genome-style ranking uses one temporal positive plus 99 sampled temporal benign variants.",
            "The benchmark does not test phenotype, pedigree, phase, compound-heterozygous pairing, structural variants, or drug repurposing.",
            "Review-status and gene-history features are available evidence but may not transfer to novel genes or poorly curated conditions; unseen-gene results are reported separately.",
        ],
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with (args.output_dir / "predictions.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["variation_id", "gene", "label", *predictions])
        for i, record in enumerate(test_records):
            writer.writerow([record.variation_id, record.gene, record.label, *[float(predictions[name][i]) for name in predictions]])
    plot_results(args.output_dir, y_test, predictions, uncertainties)
    print(json.dumps(report["data"], indent=2))
    for name, result in report["models"].items():
        print(name, {key: result[key] for key in ("roc_auc", "pr_auc", "f_max", "nll", "brier", "ece_15")}, result["ranking"])


if __name__ == "__main__":
    main()
