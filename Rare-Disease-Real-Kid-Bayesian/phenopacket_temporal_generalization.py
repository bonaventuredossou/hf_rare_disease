#!/usr/bin/env python3
"""Patient-level temporal generalization benchmark for the saved model families."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from clinvar_temporal_experiment import Variant, featurize, fmax, parse_release


SEED = 20260905
GENE_FEATURE_COLUMNS = (15, 16, 17)


def key(record: Variant) -> tuple[str, int, str, str]:
    return (record.chrom.removeprefix("chr"), record.pos, record.ref.upper(), record.alt.upper())


def raw_vcf_keys(path: Path) -> set[tuple[str, int, str, str]]:
    result = set()
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, pos, _, ref, alt = line.split("\t", 5)[:5]
            if "," not in alt:
                result.add((chrom.removeprefix("chr"), int(pos), ref.upper(), alt.upper()))
    return result


def causal_keys(document: dict) -> list[tuple[str, int, str, str]]:
    result = []
    for interpretation in document.get("interpretations", []):
        for genomic in interpretation.get("diagnosis", {}).get("genomicInterpretations", []):
            if genomic.get("interpretationStatus") != "CAUSATIVE":
                continue
            descriptor = genomic.get("variantInterpretation", {}).get("variationDescriptor", {})
            record = descriptor.get("vcfRecord", {})
            if str(record.get("genomeAssembly", "")).lower() not in {"hg38", "grch38"}:
                continue
            try:
                result.append(
                    (
                        str(record["chrom"]).removeprefix("chr"),
                        int(record["pos"]),
                        str(record["ref"]).upper(),
                        str(record["alt"]).upper(),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return result


def load_cases(
    archive_path: Path,
    current: dict[tuple[str, int, str, str], Variant],
    historical_keys: set[tuple[str, int, str, str]],
) -> tuple[list[dict], dict[str, int]]:
    unique = {}
    counts = defaultdict(int)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            document = json.loads(archive.read(name))
            variants = causal_keys(document)
            if not variants:
                counts["no_grch38_causal_variant"] += 1
                continue
            if len(variants) > 2:
                counts["more_than_two_causal_variants"] += 1
                continue
            if not all(item in current for item in variants):
                counts["not_cleanly_annotated_in_current_clinvar"] += 1
                continue
            if not all(current[item].label == 1 for item in variants):
                counts["not_pathogenic_in_current_clinvar"] += 1
                continue
            if any(item in historical_keys for item in variants):
                counts["present_in_2023_snapshot"] += 1
                continue
            hypothesis = tuple(sorted(variants))
            if hypothesis in unique:
                counts["duplicate_causal_hypothesis"] += 1
                continue
            unique[hypothesis] = {
                "case_id": document.get("id", name),
                "source_path": name,
                "hpo_count": sum(
                    not feature.get("excluded", False)
                    for feature in document.get("phenotypicFeatures", [])
                ),
                "causal": tuple(current[item] for item in hypothesis),
            }
    counts["retained_unique_cases"] = len(unique)
    return list(unique.values()), dict(counts)


def consequence_group(record: Variant) -> str:
    value = record.consequence
    if any(item in value for item in ("frameshift", "nonsense", "stop_gained", "splice_donor", "splice_acceptor")):
        return "lof"
    if "missense" in value:
        return "missense"
    if "synonymous" in value:
        return "synonymous"
    return "other"


def build_decoy_indices(records: list[Variant]):
    by_signature = defaultdict(list)
    by_type = defaultdict(list)
    by_gene = defaultdict(list)
    for index, record in enumerate(records):
        by_signature[(record.variant_type, consequence_group(record))].append(index)
        by_type[record.variant_type].append(index)
        by_gene[record.gene].append(index)
    pair_genes = [gene for gene, indices in by_gene.items() if len(indices) >= 2]
    return by_signature, by_type, by_gene, pair_genes


def sample_hypotheses(cases: list[dict], benign: list[Variant], seed: int = SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    by_signature, by_type, by_gene, pair_genes = build_decoy_indices(benign)
    output = []
    for case in cases:
        causal = case["causal"]
        candidates = [causal]
        seen = {tuple(sorted(key(record) for record in causal))}
        attempts = 0
        while len(candidates) < 10 and attempts < 10000:
            attempts += 1
            if len(causal) == 1:
                signature = (causal[0].variant_type, consequence_group(causal[0]))
                pool = by_signature.get(signature, [])
                if len(pool) < 10:
                    pool = by_type.get(causal[0].variant_type, [])
                if len(pool) < 10:
                    pool = list(range(len(benign)))
                hypothesis = (benign[int(rng.choice(pool))],)
            else:
                gene = pair_genes[int(rng.integers(len(pair_genes)))]
                selected = rng.choice(by_gene[gene], 2, replace=False)
                hypothesis = tuple(benign[int(index)] for index in selected)
            hypothesis_key = tuple(sorted(key(record) for record in hypothesis))
            if hypothesis_key not in seen:
                candidates.append(hypothesis)
                seen.add(hypothesis_key)
        if len(candidates) != 10:
            raise RuntimeError(f"Could not generate nine unique decoys for {case['case_id']}")
        shuffled = rng.permutation(10)
        output.append({**case, "hypotheses": [candidates[int(i)] for i in shuffled], "truth_index": int(np.where(shuffled == 0)[0][0])})
    return output


def matrices(records: list[Variant], scaler, stats, neutralize_gene_history: bool) -> tuple[np.ndarray, np.ndarray]:
    raw = featurize(records, stats)
    if neutralize_gene_history:
        raw[:, GENE_FEATURE_COLUMNS[0]] = 0.0
        raw[:, GENE_FEATURE_COLUMNS[1]] = 0.5
        raw[:, GENE_FEATURE_COLUMNS[2]] = 1.0
    return raw, scaler.transform(raw).astype(np.float32)


def predict_all(models, records: list[Variant], neutralize_gene_history: bool) -> dict[str, np.ndarray]:
    tree, mlp, ensemble, small_sbnn, scaler, stats = models["main"]
    bbb, wide_sbnn, wide_bbb, scaler2, stats2 = models["wide"]
    raw, scaled = matrices(records, scaler, stats, neutralize_gene_history)
    _, scaled2 = matrices(records, scaler2, stats2, neutralize_gene_history)
    member_predictions = np.stack([member.predict_proba(scaled)[:, 1] for member in ensemble])
    result = {
        "gradient_boost": tree.predict_proba(raw)[:, 1],
        "dense_mlp": mlp.predict_proba(scaled)[:, 1],
        "deep_ensemble": member_predictions.mean(axis=0),
        "small_sbnn": small_sbnn.predict_samples(scaled, draws=32, seed=SEED + 99)[0],
        "dense_bbb": bbb.predict_statistics(scaled2, draws=32, seed=SEED + 99)[0],
        "wide_rank50_sbnn": wide_sbnn.predict_statistics(scaled2, draws=32, seed=SEED + 99)[0],
        "wide_dense_bbb": wide_bbb.predict_statistics(scaled2, draws=32, seed=SEED + 99)[0],
    }
    result["temporal_consensus"] = (
        0.3 * result["gradient_boost"]
        + 0.2 * result["deep_ensemble"]
        + 0.5 * result["small_sbnn"]
    )
    return result


def hypothesis_scores(cases: list[dict], allele_scores: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    cursor = 0
    case_scores = []
    labels = []
    flat_scores = []
    for case in cases:
        scores = []
        for hypothesis in case["hypotheses"]:
            values = allele_scores[cursor : cursor + len(hypothesis)]
            cursor += len(hypothesis)
            scores.append(float(np.exp(np.mean(np.log(np.clip(values, 1e-12, 1.0))))))
        scores = np.asarray(scores)
        case_scores.append(scores)
        labels.extend(int(i == case["truth_index"]) for i in range(10))
        flat_scores.extend(scores)
    return case_scores, np.asarray(labels), np.asarray(flat_scores)


def summarize(cases: list[dict], scores: list[np.ndarray], labels: np.ndarray, flat: np.ndarray) -> dict:
    ranks = []
    for case, values in zip(cases, scores):
        truth = values[case["truth_index"]]
        rank = 1 + int(np.sum(values > truth)) + 0.5 * int(np.sum(values == truth) - 1)
        ranks.append(rank)
    ranks = np.asarray(ranks)
    maximum_f, threshold = fmax(labels, flat)

    def rank_metrics(mask: np.ndarray) -> dict[str, float | int | list[float]]:
        selected = ranks[mask]
        successes = int(np.sum(selected <= 1))
        n = len(selected)
        rate = successes / n
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (rate + z * z / (2 * n)) / denominator
        half_width = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
        return {
            "cases": n,
            "top1": float(rate),
            "top1_wilson_95": [float(center - half_width), float(center + half_width)],
            "top3": float(np.mean(selected <= 3)),
            "top5": float(np.mean(selected <= 5)),
            "mrr": float(np.mean(1.0 / selected)),
            "mean_rank": float(np.mean(selected)),
        }

    single_mask = np.asarray([len(case["causal"]) == 1 for case in cases])
    pair_mask = ~single_mask
    overall = rank_metrics(np.ones(len(cases), dtype=bool))
    return {
        **overall,
        "single_variant_cases": sum(len(case["causal"]) == 1 for case in cases),
        "two_variant_cases": sum(len(case["causal"]) == 2 for case in cases),
        "single_variant_ranking": rank_metrics(single_mask),
        "two_variant_ranking": rank_metrics(pair_mask),
        "f_max": maximum_f,
        "f_max_threshold": threshold,
        "brier": float(brier_score_loss(labels, flat)),
        "nll": float(log_loss(labels, np.clip(flat, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def plot_results(output_dir: Path, results: dict) -> None:
    names = list(results["gene_history_neutralized"])
    labels = [name.replace("_", "\n") for name in names]
    x = np.arange(len(names))
    width = 0.38
    fig, axis = plt.subplots(figsize=(13, 6))
    for offset, condition, color in (
        (-width / 2, "gene_history_neutralized", "#355C7D"),
        (width / 2, "gene_history_operational", "#C06C84"),
    ):
        axis.bar(
            x + offset,
            [results[condition][name]["top1"] for name in names],
            width,
            label=condition.replace("_", " "),
            color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Top-1 causal-hypothesis accuracy")
    axis.set_ylim(0, 1)
    axis.set_title("Post-2023 Phenopacket patient-level generalization")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "patient_generalization_top1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--phenopackets", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading ClinVar snapshots", flush=True)
    historical_keys = raw_vcf_keys(args.historical)
    _, historical_ids = parse_release(args.historical, collect_all_ids=True)
    current_records, _ = parse_release(args.current, collect_all_ids=False)
    current = {key(record): record for record in current_records}
    cases, exclusions = load_cases(args.phenopackets, current, historical_keys)
    benign = [
        record for record in current_records
        if record.label == 0 and record.variation_id not in historical_ids and key(record) not in historical_keys
    ]
    cases = sample_hypotheses(cases, benign)
    records = [record for case in cases for hypothesis in case["hypotheses"] for record in hypothesis]
    with args.model_checkpoint.open("rb") as handle:
        models = pickle.load(handle)

    results = {}
    for neutralized in (True, False):
        condition = "gene_history_neutralized" if neutralized else "gene_history_operational"
        print(f"Scoring {condition}", flush=True)
        predictions = predict_all(models, records, neutralized)
        results[condition] = {}
        for name, allele_scores in predictions.items():
            scores, labels, flat = hypothesis_scores(cases, allele_scores)
            results[condition][name] = summarize(cases, scores, labels, flat)
            print(name, results[condition][name]["top1"], flush=True)

    hpo_counts = [case["hpo_count"] for case in cases]
    report = {
        "claim": "Molecular causal-hypothesis ranking transfers to post-2023 solved rare-disease cases.",
        "scope": "Patient-level molecular ranking only; HPO terms identify real cases but are not yet model inputs.",
        "data": {
            "phenopacket_store_release": args.phenopackets.name,
            "training_snapshot": args.historical.name,
            "evaluation_snapshot": args.current.name,
            "unique_cases": len(cases),
            "single_variant_cases": sum(len(case["causal"]) == 1 for case in cases),
            "two_variant_cases": sum(len(case["causal"]) == 2 for case in cases),
            "unique_disease_genes": len({record.gene for case in cases for record in case["causal"]}),
            "median_observed_hpo_terms": float(np.median(hpo_counts)),
            "new_benign_decoy_pool": len(benign),
            "candidate_hypotheses_per_case": 10,
            "exclusions": exclusions,
        },
        "design": {
            "causal_variants": "Exact GRCh38 variants absent from the raw 2023 ClinVar snapshot and pathogenic in 2026 ClinVar.",
            "decoys": "Nine benign hypotheses absent from the 2023 snapshot; single-variant decoys matched on variant type and consequence group; two-variant decoys contain two alleles in one gene.",
            "pair_score": "Geometric mean of allele pathogenicity probabilities.",
            "primary_condition": "All gene-history inputs set to unseen/neutral to prevent gene-label memorization.",
            "seed": SEED,
            "posterior_draws": 32,
        },
        "results": results,
        "limitations": [
            "The benchmark does not yet use each patient's HPO terms for phenotype matching.",
            "Benign candidates come from ClinVar rather than each patient's complete exome or genome.",
            "The causal case variants and decoys are temporally clean by exact genomic coordinate, but ClinVar curation practices can still introduce dataset shift.",
            "Repeated reports of the same causal hypothesis were deduplicated.",
        ],
    }
    output = args.output_dir / "patient_generalization_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_results(args.output_dir, results)
    print(json.dumps(report["data"], indent=2), flush=True)


if __name__ == "__main__":
    main()
