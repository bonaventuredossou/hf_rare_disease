#!/usr/bin/env python3
"""Train the pair ranker on cached external solved cases.

Requires the parent project's open data/adapters. Cluster inference is standalone.
Candidates are constructed, not real background WGS calls: 15 pairs from six
same-gene alleles plus eight phenotype-matched, other-gene benign pairs.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
from bayesian_pair_ranker import Posterior, ranking_metrics, _softmax
from shared_pair_features import FEATURE_NAMES, SCHEMA, pair_features, evidence_rule
from mva_track1_pipeline import (
    _parse_hpo_obo,
    _hpo_ancestor_fn,
    _parse_genes_to_phenotype,
)


class PhenotypeScores:
    """The original pipeline's phenotype score, cached across training cases."""

    def __init__(self, obo, genes):
        self.parents, _ = _parse_hpo_obo(obo)
        self.anc = _hpo_ancestor_fn(self.parents)
        terms = _parse_genes_to_phenotype(genes)
        self.prop = {
            g: set().union(*(self.anc(t) for t in ts if t in self.parents))
            for g, ts in terms.items()
        }
        # IC in the baseline uses all propagated terms, including obsolete aliases.
        counts = Counter(
            a for ts in terms.values() for a in set().union(*(self.anc(t) for t in ts))
        )
        self.ic = {t: -math.log((c + 1) / (len(terms) + 1)) for t, c in counts.items()}

    @lru_cache(maxsize=200000)
    def term_score(self, q, gene):
        common = self.anc(q) & self.prop.get(gene, set())
        return min(
            1.0,
            max((self.ic.get(t, 0.0) for t in common), default=0.0)
            / max(self.ic.get(q, 0.0), 0.25),
        )

    def score(self, terms, gene):
        terms = [q for q in terms if q in self.parents]
        if not terms:
            return 0.0
        sims = [self.term_score(q, gene) for q in terms]
        return min(1.0, 0.82 * np.mean(sims) + 0.18 * np.mean(np.array(sims) >= 0.35))


def allele(a):
    return dict(
        consequence=",".join(a.consequences),
        af=a.population_af,
        cadd=a.cadd_phred,
        spliceai=a.spliceai,
    )


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset(cases, annotations, decoys, knowledge, seed):
    genes = {c.gene_symbol.upper() for c in cases}
    pool = {
        g: [
            annotations[(d.variant.canonical_id, g)]
            for d in ds
            if (d.variant.canonical_id, g) in annotations
            and annotations[(d.variant.canonical_id, g)].annotation_ok
            and annotations[(d.variant.canonical_id, g)].returned_gene == g
        ]
        for g, ds in decoys.items()
        if g in genes
    }
    pool = {g: ds for g, ds in pool.items() if len(ds) >= 2}
    xs, ys, ids, groups, gene_ids = [], [], [], [], []
    skipped = Counter()
    for case in cases:
        if not case.is_pair:
            continue
        g = case.gene_symbol.upper()
        if g == "BUB1B":
            skipped["challenge_gene_excluded"] += 1
            continue
        positives = [annotations.get((v.canonical_id, g)) for v in case.causal_variants]
        if any(
            a is None or not a.annotation_ok or a.returned_gene != g for a in positives
        ):
            skipped["missing_causal_annotation"] += 1
            continue
        negatives = [
            a for a in pool.get(g, []) if a.variant not in case.causal_variants
        ]
        if len(negatives) < 4:
            skipped["insufficient_annotated_decoys"] += 1
            continue
        combos = [
            (g, tuple(pair))
            for pair in itertools.combinations(positives + negatives[:4], 2)
        ]
        others = sorted(
            (h for h in pool if h != g and h != "BUB1B"),
            key=lambda h: (-knowledge.score(case.phenotype_terms, h), h),
        )[:8]
        if len(others) != 8:
            skipped["insufficient_cross_gene_pairs"] += 1
            continue
        combos += [(h, tuple(pool[h][:2])) for h in others]

        def key(item):
            h, aa = item
            return hashlib.sha256(
                f"{seed}|{case.case_id}|{h}|{sorted(a.variant.canonical_id for a in aa)}".encode()
            ).hexdigest()

        combos.sort(key=key)
        target = next(
            i
            for i, (h, aa) in enumerate(combos)
            if h == g and {a.variant for a in aa} == set(case.causal_variants)
        )
        xs.append(
            np.stack(
                [
                    pair_features(
                        [allele(a) for a in aa],
                        knowledge.score(case.phenotype_terms, h),
                    )
                    for h, aa in combos
                ]
            )
        )
        ys.append(target)
        ids.append(case.case_id)
        groups.append(case.source_group)
        gene_ids.append(g)
    return np.stack(xs), np.array(ys), ids, groups, gene_ids, dict(skipped)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "models")
    p.add_argument("--seed", type=int, default=20260829)
    args = p.parse_args()
    root = args.project_root
    sys.path.insert(0, str(root / "src"))
    from rare_kid_modeling.pheval_adapter import load_pheval_solved_cases
    from rare_kid_modeling.molecular_benchmark import (
        load_molecular_cache,
        split_cases_by_gene_and_source,
    )

    paths = dict(
        phenopackets=root / "data/open/phenopacket_store/0.1.27",
        annotations=root
        / "data/open/molecular/vep_annotations_phenopacket_store_0.1.27.json",
        obo=root / "data/open/hpo/v2026-06-23/hp-base.obo",
        genes=root / "data/open/hpo/v2026-06-23/genes_to_phenotype.txt",
    )
    cases, load_report = load_pheval_solved_cases(paths["phenopackets"])
    annotations, decoys, _ = load_molecular_cache(paths["annotations"])
    splits = split_cases_by_gene_and_source(cases, seed=args.seed)
    knowledge = PhenotypeScores(paths["obo"], paths["genes"])
    data = {}
    for name, values in splits.items():
        data[name] = dataset(values, annotations, decoys, knowledge, args.seed)
        print(name, len(data[name][1]), "cases", data[name][-1], flush=True)
    for a, b in itertools.combinations(data, 2):
        for column in (2, 3, 4):
            if set(data[a][column]) & set(data[b][column]):
                raise AssertionError("Split overlap")
    x, y, *_ = data["train"]
    mean = x.mean(axis=(0, 1))
    scale = x.std(axis=(0, 1))
    scale[scale < 1e-8] = 1
    vx, vy, *_ = data["validation"]
    grid = []
    models = []
    for prior in (0.25, 1.0, 2.0, 8.0):
        model = Posterior.fit(
            (x - mean) / scale, y, prior_scale=prior, max_iterations=1000
        )
        if not model.converged:
            raise RuntimeError("Optimizer did not converge")
        metrics = ranking_metrics(model.map_probabilities((vx - mean) / scale), vy)
        grid.append({"prior_scale": prior, **metrics})
        models.append(model)
    selection = min(range(len(grid)), key=lambda i: grid[i]["negative_log_likelihood"])
    model = models[selection]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_data = {
        "schema": SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights_map": model.weights_map.tolist(),
        "covariance": model.covariance.tolist(),
        "prior_scale": model.prior_scale,
        "selection": "validation MAP NLL",
        "seed": args.seed,
        "excluded_challenge_gene": "BUB1B",
        "source_sha256": {k: sha(v) for k, v in paths.items() if v.is_file()},
    }
    (args.output_dir / "shared_pair_model.json").write_text(
        json.dumps(model_data, indent=2) + "\n"
    )
    tx, ty, *_ = data["test"]
    sx = (tx - mean) / scale
    posterior = model.posterior_predictive(sx, draws=2048, seed=20260905)
    predictions = {
        "fixed_common_evidence_rule": _softmax(evidence_rule(tx)),
        "regularized_MAP": model.map_probabilities(sx),
        "Bayesian_posterior_mean": posterior["mean_probabilities"],
    }
    report = {
        "status": "executed",
        "loaded_cases": load_report.to_dict(),
        "split": {
            k: {"cases": len(v[1]), "genes": len(set(v[4])), "skipped": v[-1]}
            for k, v in data.items()
        },
        "gene_and_source_disjoint": True,
        "validation_grid": grid,
        "selected_prior": model.prior_scale,
        "test": {k: ranking_metrics(v, ty) for k, v in predictions.items()},
        "mean_mutual_information": float(posterior["mutual_information"].mean()),
        "limitations": [
            "Constructed candidate benchmark using known benign decoys, not full patient VCFs.",
            "Test split has appeared in prior project experiments; this is development evaluation.",
            "The fixed common-evidence rule is not the full upstream genome pipeline.",
            "HPO knowledge and VEP annotations may overlap publications and differ in release from challenge annotations.",
            "Challenge answer not used for fitting or hyperparameter selection; BUB1B excluded.",
            "No trained QC, REVEL, phase or ClinVar coefficients: these are absent or label-derived in external data.",
        ],
    }
    with (args.output_dir / "external_test_predictions.csv").open(
        "w", newline=""
    ) as handle:
        fields = ["case_id", "gene", "source", "target"] + list(predictions)
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        for i in range(len(ty)):
            row = dict(
                case_id=data["test"][2][i],
                gene=data["test"][4][i],
                source=data["test"][3][i],
                target=int(ty[i]),
            )
            row.update(
                {
                    k: int(np.where(np.argsort(-v[i], kind="stable") == ty[i])[0][0])
                    + 1
                    for k, v in predictions.items()
                }
            )
            w.writerow(row)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                k: [
                    dict(case_id=i, gene=g, source=s)
                    for i, g, s in zip(v[2], v[4], v[3])
                ]
                for k, v in data.items()
            },
            indent=2,
        )
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
