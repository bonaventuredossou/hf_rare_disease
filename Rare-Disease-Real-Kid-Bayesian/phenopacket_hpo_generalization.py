#!/usr/bin/env python3
"""Add phenotype matching to the temporal patient benchmark without test-set tuning."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

from clinvar_temporal_experiment import Variant, parse_release
from phenopacket_temporal_generalization import (
    SEED,
    causal_keys,
    hypothesis_scores,
    key,
    predict_all,
    raw_vcf_keys,
    sample_hypotheses,
    summarize,
)


BETA_GRID = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0)
MISSING_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def observed_hpo_terms(document: dict) -> tuple[str, ...]:
    terms = []
    for feature in document.get("phenotypicFeatures", []):
        if feature.get("excluded", False):
            continue
        term = feature.get("type", {}).get("id", "")
        if term.startswith("HP:"):
            terms.append(term)
    return tuple(sorted(set(terms)))


def read_phenopackets(
    archive_path: Path,
    current: dict[tuple[str, int, str, str], Variant],
    historical_keys: set[tuple[str, int, str, str]],
) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Separate older reference cases from exact-coordinate post-2023 cases."""
    reference_by_hypothesis = {}
    test_by_hypothesis = {}
    counts = defaultdict(int)
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            document = json.loads(archive.read(name))
            terms = observed_hpo_terms(document)
            variants = causal_keys(document)
            if not terms:
                counts["no_observed_hpo_terms"] += 1
                continue
            if not variants or len(variants) > 2:
                counts["unsupported_causal_hypothesis"] += 1
                continue
            if not all(item in current for item in variants):
                counts["not_cleanly_annotated_current"] += 1
                continue
            records = tuple(current[item] for item in sorted(variants))
            if not all(record.label == 1 and record.gene for record in records):
                counts["not_pathogenic_or_missing_gene"] += 1
                continue
            genes = tuple(sorted(set(record.gene for record in records)))
            if len(genes) != 1:
                counts["more_than_one_causal_gene"] += 1
                continue
            present = [item in historical_keys for item in variants]
            case = {
                "case_id": document.get("id", name),
                "source_path": name,
                "hpo_terms": terms,
                "hpo_count": len(terms),
                "causal_gene": genes[0],
                "causal": records,
            }
            signature = tuple(sorted(variants))
            if all(present):
                reference_by_hypothesis.setdefault(signature, case)
            elif not any(present):
                test_by_hypothesis.setdefault(signature, case)
            else:
                counts["mixed_temporal_hypothesis"] += 1
    counts["older_unique_cases"] = len(reference_by_hypothesis)
    counts["post_2023_unique_cases"] = len(test_by_hypothesis)
    return list(reference_by_hypothesis.values()), list(test_by_hypothesis.values()), dict(counts)


def split_reference_validation(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Make validation contain both familiar genes and completely held-out genes."""
    rng = np.random.default_rng(SEED)
    by_gene = defaultdict(list)
    for case in cases:
        by_gene[case["causal_gene"]].append(case)
    genes = np.asarray(sorted(by_gene), dtype=object)
    held_out_gene_count = max(1, int(round(0.2 * len(genes))))
    held_out_genes = set(rng.choice(genes, held_out_gene_count, replace=False))
    reference, validation = [], []
    for gene in genes:
        group = by_gene[gene]
        if gene in held_out_genes:
            validation.extend(group)
            continue
        order = rng.permutation(len(group))
        if len(group) == 1:
            reference.extend(group)
            continue
        n_validation = min(len(group) - 1, max(1, int(round(0.2 * len(group)))))
        validation.extend(group[int(i)] for i in order[:n_validation])
        reference.extend(group[int(i)] for i in order[n_validation:])
    return reference, validation


def read_hpo(path: Path) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    parents: dict[str, tuple[str, ...]] = {}
    aliases: dict[str, str] = {}
    current_id = None
    current_parents: list[str] = []
    current_aliases: list[str] = []

    def commit() -> None:
        if current_id:
            parents[current_id] = tuple(current_parents)
            for alias in current_aliases:
                aliases[alias] = current_id

    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "[Term]":
                commit()
                current_id = None
                current_parents = []
                current_aliases = []
            elif line.startswith("id: HP:"):
                current_id = line.split("id: ", 1)[1]
            elif line.startswith("alt_id: HP:"):
                current_aliases.append(line.split("alt_id: ", 1)[1])
            elif line.startswith("is_a: HP:"):
                current_parents.append(line.split()[1])
    commit()
    return parents, aliases


class PhenotypeMatcher:
    def __init__(self, reference_cases: list[dict], parents: dict[str, tuple[str, ...]], aliases: dict[str, str]):
        self.parents = parents
        self.aliases = aliases
        propagated = [self.propagate(case["hpo_terms"]) for case in reference_cases]
        counts = defaultdict(int)
        for terms in propagated:
            for term in terms:
                counts[term] += 1
        n = len(reference_cases)
        self.ic = {term: -math.log((count + 1) / (n + 1)) for term, count in counts.items()}
        self.by_gene = defaultdict(list)
        for case in reference_cases:
            original = tuple(self.canonical(term) for term in case["hpo_terms"] if self.canonical(term) in parents)
            if original:
                self.by_gene[case["causal_gene"]].append((original, self.propagate(original)))

    def canonical(self, term: str) -> str:
        return self.aliases.get(term, term)

    @lru_cache(maxsize=None)
    def ancestors(self, term: str) -> frozenset[str]:
        term = self.canonical(term)
        result = {term}
        for parent in self.parents.get(term, ()):
            result.update(self.ancestors(parent))
        return frozenset(result)

    def propagate(self, terms) -> frozenset[str]:
        result = set()
        for term in terms:
            result.update(self.ancestors(self.canonical(term)))
        return frozenset(result)

    def _directed(self, source: tuple[str, ...], target_propagated: frozenset[str]) -> float:
        values = []
        ceilings = []
        for term in source:
            ancestors = self.ancestors(term)
            common = ancestors.intersection(target_propagated)
            values.append(max((self.ic.get(item, 0.0) for item in common), default=0.0))
            ceilings.append(max((self.ic.get(item, 0.0) for item in ancestors), default=0.0))
        denominator = float(np.mean(ceilings)) if ceilings else 0.0
        return float(np.mean(values) / denominator) if denominator > 0 else 0.0

    def case_similarity(self, query_terms: tuple[str, ...], reference: tuple[tuple[str, ...], frozenset[str]]) -> float:
        query = tuple(self.canonical(term) for term in query_terms if self.canonical(term) in self.parents)
        if not query:
            return 0.0
        query_propagated = self.propagate(query)
        reference_terms, reference_propagated = reference
        return 0.5 * (
            self._directed(query, reference_propagated)
            + self._directed(reference_terms, query_propagated)
        )

    def gene_similarity(self, query_terms: tuple[str, ...], gene: str) -> float:
        references = self.by_gene.get(gene, ())
        return max((self.case_similarity(query_terms, reference) for reference in references), default=float("nan"))


def phenotype_scores(cases: list[dict], matcher: PhenotypeMatcher) -> list[np.ndarray]:
    output = []
    for case in cases:
        values = []
        for hypothesis in case["hypotheses"]:
            genes = sorted(set(record.gene for record in hypothesis if record.gene))
            similarities = [matcher.gene_similarity(case["hpo_terms"], gene) for gene in genes]
            known = [value for value in similarities if np.isfinite(value)]
            values.append(max(known) if known else float("nan"))
        output.append(np.asarray(values, dtype=float))
    return output


def flatten_records(cases: list[dict]) -> list[Variant]:
    return [record for case in cases for hypothesis in case["hypotheses"] for record in hypothesis]


def molecular_case_scores(cases: list[dict], allele_scores: np.ndarray) -> list[np.ndarray]:
    scores, _, _ = hypothesis_scores(cases, allele_scores)
    return scores


def combine(
    molecular: list[np.ndarray], phenotype: list[np.ndarray], beta: float, missing_score: float
) -> list[np.ndarray]:
    combined = []
    for mol, phe in zip(molecular, phenotype):
        clipped = np.clip(mol, 1e-7, 1 - 1e-7)
        logit = np.log(clipped) - np.log1p(-clipped)
        filled_phenotype = np.nan_to_num(phe, nan=missing_score)
        combined.append(1.0 / (1.0 + np.exp(-np.clip(logit + beta * filled_phenotype, -30, 30))))
    return combined


def labels_and_flat(cases: list[dict], scores: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(i == case["truth_index"]) for case in cases for i in range(10)])
    return labels, np.concatenate(scores)


def top1_mrr(cases: list[dict], scores: list[np.ndarray]) -> tuple[float, float]:
    ranks = []
    for case, values in zip(cases, scores):
        truth = values[case["truth_index"]]
        rank = 1 + np.sum(values > truth) + 0.5 * (np.sum(values == truth) - 1)
        ranks.append(float(rank))
    ranks = np.asarray(ranks)
    return float(np.mean(ranks <= 1)), float(np.mean(1 / ranks))


def top1_success(cases: list[dict], scores: list[np.ndarray]) -> np.ndarray:
    values = []
    for case, candidate_scores in zip(cases, scores):
        truth = candidate_scores[case["truth_index"]]
        rank = 1 + np.sum(candidate_scores > truth) + 0.5 * (np.sum(candidate_scores == truth) - 1)
        values.append(rank <= 1)
    return np.asarray(values, dtype=bool)


def paired_top1_test(cases: list[dict], before: list[np.ndarray], after: list[np.ndarray]) -> dict:
    before_success = top1_success(cases, before)
    after_success = top1_success(cases, after)
    changes = after_success.astype(float) - before_success.astype(float)
    rng = np.random.default_rng(SEED + 404)
    draws = rng.integers(0, len(cases), size=(10000, len(cases)))
    bootstrap = changes[draws].mean(axis=1)
    gained = int(np.sum(~before_success & after_success))
    lost = int(np.sum(before_success & ~after_success))
    discordant = gained + lost
    p_value = float(binomtest(min(gained, lost), discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "top1_change": float(changes.mean()),
        "paired_bootstrap_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "cases_gained": gained,
        "cases_lost": lost,
        "mcnemar_exact_two_sided_p": p_value,
    }


def choose_hpo_parameters(
    cases: list[dict], molecular: list[np.ndarray], phenotype: list[np.ndarray]
) -> tuple[float, float, list[dict]]:
    trials = []
    for beta in BETA_GRID:
        for missing_score in MISSING_GRID:
            top1, mrr = top1_mrr(cases, combine(molecular, phenotype, beta, missing_score))
            trials.append({"beta": beta, "missing_score": missing_score, "top1": top1, "mrr": mrr})
    best = max(
        trials,
        key=lambda item: (item["top1"], item["mrr"], -item["beta"], -abs(item["missing_score"] - 0.5)),
    )
    return float(best["beta"]), float(best["missing_score"]), trials


def evaluate(cases: list[dict], scores: list[np.ndarray], known_genes: set[str]) -> dict:
    labels, flat = labels_and_flat(cases, scores)
    result = summarize(cases, scores, labels, flat)
    known_mask = [case["causal_gene"] in known_genes for case in cases]
    for label, mask in (("phenotype_reference_known", known_mask), ("phenotype_reference_unseen", [not x for x in known_mask])):
        subset_cases = [case for case, keep in zip(cases, mask) if keep]
        subset_scores = [score for score, keep in zip(scores, mask) if keep]
        if subset_cases:
            result[label] = {
                "cases": len(subset_cases),
                "top1": top1_mrr(subset_cases, subset_scores)[0],
                "mrr": top1_mrr(subset_cases, subset_scores)[1],
            }
    return result


def plot_results(output_dir: Path, results: dict) -> None:
    names = list(results)
    x = np.arange(len(names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(14, 6.5))
    axis.bar(x - width / 2, [results[n]["molecular_only"]["top1"] for n in names], width, label="molecular only", color="#496A81")
    axis.bar(x + width / 2, [results[n]["phenotype_conditioned"]["top1"] for n in names], width, label="molecular + HPO", color="#D17B49")
    axis.axhline(0.1, color="#333333", linestyle="--", linewidth=1, label="random top-1")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Top-1 causal-hypothesis accuracy")
    axis.set_title("Untouched post-2023 Phenopacket cases (gene-history features neutralized)")
    axis.grid(axis="y", alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hpo_conditioned_top1.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--phenopackets", type=Path, required=True)
    parser.add_argument("--hpo", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-seed", type=int, default=SEED)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading ClinVar and Phenopacket data", flush=True)
    historical_keys = raw_vcf_keys(args.historical)
    _, historical_ids = parse_release(args.historical, collect_all_ids=True)
    current_records, _ = parse_release(args.current, collect_all_ids=False)
    current = {key(record): record for record in current_records}
    older_cases, test_cases, exclusions = read_phenopackets(args.phenopackets, current, historical_keys)
    reference_cases, validation_cases = split_reference_validation(older_cases)
    benign = [
        record for record in current_records
        if record.label == 0 and record.variation_id not in historical_ids and key(record) not in historical_keys
    ]
    validation_cases = sample_hypotheses(validation_cases, benign, seed=args.candidate_seed)
    test_cases = sample_hypotheses(test_cases, benign, seed=args.candidate_seed + 1)

    print("Building the HPO matcher from older reference cases", flush=True)
    parents, aliases = read_hpo(args.hpo)
    validation_matcher = PhenotypeMatcher(reference_cases, parents, aliases)
    test_matcher = PhenotypeMatcher(older_cases, parents, aliases)
    validation_phenotype = phenotype_scores(validation_cases, validation_matcher)
    test_phenotype = phenotype_scores(test_cases, test_matcher)

    with args.model_checkpoint.open("rb") as handle:
        models = pickle.load(handle)
    validation_predictions = predict_all(models, flatten_records(validation_cases), neutralize_gene_history=True)
    test_predictions = predict_all(models, flatten_records(test_cases), neutralize_gene_history=True)

    results = {}
    for name in test_predictions:
        validation_molecular = molecular_case_scores(validation_cases, validation_predictions[name])
        beta, missing_score, trials = choose_hpo_parameters(
            validation_cases, validation_molecular, validation_phenotype
        )
        test_molecular = molecular_case_scores(test_cases, test_predictions[name])
        test_combined = combine(test_molecular, test_phenotype, beta, missing_score)
        molecular_result = evaluate(test_cases, test_molecular, set(test_matcher.by_gene))
        combined_result = evaluate(test_cases, test_combined, set(test_matcher.by_gene))
        results[name] = {
            "selected_beta": beta,
            "selected_missing_gene_score": missing_score,
            "validation_grid": trials,
            "molecular_only": molecular_result,
            "phenotype_conditioned": combined_result,
            "top1_change": combined_result["top1"] - molecular_result["top1"],
            "mrr_change": combined_result["mrr"] - molecular_result["mrr"],
            "paired_top1_test": paired_top1_test(test_cases, test_molecular, test_combined),
        }
        print(
            name, "beta", beta, "missing", missing_score,
            "top1", molecular_result["top1"], "->", combined_result["top1"], flush=True,
        )

    known = set(test_matcher.by_gene)
    report = {
        "claim": "Phenotype conditioning is evaluated on exact-coordinate post-2023 solved cases after choosing every mixing weight on older held-out cases.",
        "data": {
            "older_cases": len(older_cases),
            "phenotype_reference_cases": len(reference_cases),
            "weight_selection_cases": len(validation_cases),
            "untouched_post_2023_cases": len(test_cases),
            "post_2023_two_variant_cases": sum(len(case["causal"]) == 2 for case in test_cases),
            "post_2023_causal_gene_seen_in_older_cases": sum(case["causal_gene"] in known for case in test_cases),
            "post_2023_causal_gene_unseen_in_older_cases": sum(case["causal_gene"] not in known for case in test_cases),
            "benign_temporal_decoy_pool": len(benign),
            "candidate_hypotheses_per_case": 10,
            "phenopacket_exclusions": exclusions,
        },
        "design": {
            "molecular_condition": "Gene-history inputs are neutralized for every model.",
            "phenotype_similarity": "Ontology-aware symmetric Resnik similarity against older cases of each candidate gene; the best-matching older case is used.",
            "weight_selection": "Beta and the missing-gene phenotype prior are selected separately for each model on older cases containing both familiar and completely held-out genes. The post-2023 cases are never used for tuning.",
            "beta_grid": list(BETA_GRID),
            "missing_gene_score_grid": list(MISSING_GRID),
            "candidate_design": "One causal hypothesis plus nine temporally new benign ClinVar hypotheses per case.",
            "seed": SEED,
            "candidate_seed": args.candidate_seed,
        },
        "results": results,
        "limitations": [
            "This is a compact patient-level candidate benchmark, not a full-exome rerun for every external patient.",
            "Candidate decoys come from ClinVar rather than each patient's own sequencing file.",
            "Phenotype matching can help only when older reference cases exist for the candidate gene; seen and unseen causal genes are reported separately.",
            "The HPO layer is case-based and deterministic. The learned molecular rankers remain unchanged.",
        ],
    }
    output = args.output_dir / "hpo_generalization_results.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_results(args.output_dir, results)
    print(json.dumps(report["data"], indent=2), flush=True)


if __name__ == "__main__":
    main()
