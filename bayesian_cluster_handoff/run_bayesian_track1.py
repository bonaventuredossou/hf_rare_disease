#!/usr/bin/env python3
"""Rank candidate pairs from the existing TSV tables or ranking.zip."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from bayesian_pair_ranker import Posterior
from shared_pair_features import FEATURE_NAMES, SCHEMA, table_features, pair_id


def read_table(path, table):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            return list(
                csv.DictReader(
                    io.StringIO(archive.read("ranking/" + table).decode()),
                    delimiter="\t",
                )
            )
    return list(csv.DictReader((path / table).open(), delimiter="\t"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/network/scratch/b/bonaventure.dossou/rare_disease_data"),
    )
    p.add_argument("--ranking-zip", type=Path)
    p.add_argument("--run-dir", type=Path)
    p.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models/shared_pair_model.json",
    )
    p.add_argument("--output-dir", type=Path)
    p.add_argument(
        "--reference-submission",
        type=Path,
        help="Optional submitted CSV for post-prediction comparison only",
    )
    p.add_argument("--draws", type=int, default=2048)
    p.add_argument("--seed", type=int, default=20260905)
    args = p.parse_args()
    run = args.run_dir or args.data_dir / "mva_track1_run"
    source = args.ranking_zip or run / "ranking"
    out = args.output_dir or run / "bayesian"
    payload = json.loads(args.model.read_text())
    if payload["schema"] != SCHEMA or tuple(payload["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Model/feature schema mismatch")
    model = Posterior(
        np.array(payload["weights_map"]),
        np.array(payload["covariance"]),
        payload["prior_scale"],
        True,
        0.0,
    )
    out.mkdir(parents=True, exist_ok=True)
    reference = []
    if args.reference_submission:
        reference = list(csv.DictReader(args.reference_submission.open()))
    elif args.ranking_zip:
        with zipfile.ZipFile(args.ranking_zip) as archive:
            if "ranking/track1_v1_READY.csv" in archive.namelist():
                reference = list(
                    csv.DictReader(
                        io.StringIO(
                            archive.read("ranking/track1_v1_READY.csv").decode()
                        )
                    )
                )
    submitted_ranks = {pair_id(r): i + 1 for i, r in enumerate(reference)}
    summaries = {}
    for tier, table in [
        ("strict", "compound_het_pairs.tsv"),
        ("broad", "compound_het_pairs_primary_lenient_audit.tsv"),
    ]:
        original = read_table(source, table)
        rows = [r for r in original if r["phase"] != "cis_readbacked"]
        if not rows or len({pair_id(r) for r in rows}) != len(rows):
            raise ValueError("Empty candidate set or duplicate pairs")
        x = np.stack([table_features(r) for r in rows])
        pred = model.posterior_predictive(
            ((x - np.array(payload["mean"])) / np.array(payload["scale"]))[None],
            draws=args.draws,
            seed=args.seed,
        )
        probs = pred["mean_probabilities"][0]
        draws = pred["probability_draws"][:, 0, :]
        winners = np.argmax(draws, axis=1)
        baseline_order = np.argsort(
            [-float(r["pair_score"]) for r in rows], kind="stable"
        )
        baseline_ranks = {
            int(index): rank + 1 for rank, index in enumerate(baseline_order)
        }
        ordered = np.argsort(-probs, kind="stable")
        result = []
        for rank, index in enumerate(ordered, 1):
            row = dict(rows[index])
            row.update(
                bayesian_rank=rank,
                baseline_rank=baseline_ranks[int(index)],
                submitted_rank=submitted_ranks.get(pair_id(row), ""),
                conditional_probability=float(probs[index]),
                probability_lower_95=float(pred["probability_lower_95"][0, index]),
                probability_upper_95=float(pred["probability_upper_95"][0, index]),
                posterior_top1_fraction=float(np.mean(winners == index)),
                pair_id=pair_id(row),
            )
            result.append(row)
        with (out / f"{tier}_comparison.tsv").open("w", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=list(result[0]), delimiter="\t")
            w.writeheader()
            w.writerows(result)
        # Write the top candidates in the challenge CSV format.
        with (out / f"{tier}_bayesian_candidates.csv").open("w", newline="") as handle:
            fields = (
                ["proband_id"]
                + [f"{k}_{i}" for i in (1, 2) for k in ("chrom", "pos", "ref", "alt")]
                + ["epcr", "finding_type", "notes"]
            )
            w = csv.DictWriter(handle, fieldnames=fields)
            w.writeheader()
            for r in result[:10]:
                record = {key: r[key] for key in fields if key in r}
                for i in (1, 2):
                    record[f"chrom_{i}"] = "chr" + str(
                        record[f"chrom_{i}"]
                    ).removeprefix("chr")
                record.update(
                    proband_id="PROBAND01",
                    epcr=r["conditional_probability"],
                    finding_type="primary",
                    notes=f"gene={r['gene']};Bayesian conditional candidate probability;phase={r['phase']}",
                )
                w.writerow(record)
        summaries[tier] = {
            "input_pairs": len(original),
            "eligible_pairs": len(rows),
            "excluded_cis": len(original) - len(rows),
            "top_gene": result[0]["gene"],
            "top_pair": result[0]["pair_id"],
            "baseline_top_gene": rows[baseline_order[0]]["gene"],
            "mutual_information": float(pred["mutual_information"][0]),
            "top_conditional_probability": result[0]["conditional_probability"],
        }
        if reference:
            ref_id = pair_id(reference[0])
            summaries[tier]["confirmed_reference_top_pair"] = ref_id
            summaries[tier]["reference_pair_bayesian_rank"] = next(
                (r["bayesian_rank"] for r in result if r["pair_id"] == ref_id), None
            )
            summaries[tier]["exact_top_pair_match"] = result[0]["pair_id"] == ref_id
    summary = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "source": str(source),
        "seed": args.seed,
        "draws": args.draws,
        "tiers": summaries,
        "caveat": "Probabilities conditional on supplied candidate sets; clinical calibration and HF scores not established.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
