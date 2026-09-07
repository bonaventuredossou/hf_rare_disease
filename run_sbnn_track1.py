#!/usr/bin/env python3
"""Run the fitted SBNN on existing Track 1 candidate-pair tables."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from sbnn_pair_ranker import SingularBayesianRanker
from shared_pair_features import FEATURE_NAMES, SCHEMA, pair_id, table_features


def read_table(source, name):
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            text = archive.read("ranking/" + name).decode()
        return list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    return list(csv.DictReader((source / name).open(), delimiter="\t"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-zip", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models/sbnn/sbnn_pair_model.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--draws", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if not args.ranking_zip and not args.run_dir:
        parser.error("provide --ranking-zip or --run-dir")
    source = args.ranking_zip or args.run_dir / "ranking"
    output = args.output_dir or args.run_dir / "sbnn"

    artifact = json.loads(args.model.read_text())
    if artifact["schema"] != SCHEMA or artifact["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Model and feature schemas do not match")
    model = SingularBayesianRanker.from_state_dict(artifact["state"])
    mean = np.asarray(artifact["mean"])
    scale = np.asarray(artifact["scale"])
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tier, name in (
        ("strict", "compound_het_pairs.tsv"),
        ("broad", "compound_het_pairs_primary_lenient_audit.tsv"),
    ):
        original = read_table(source, name)
        rows = [row for row in original if row["phase"] != "cis_readbacked"]
        features = np.stack([table_features(row) for row in rows])
        prediction = model.predict_samples(
            ((features - mean) / scale)[None], draws=args.draws, seed=args.seed
        )
        probabilities = prediction["mean"][0]
        samples = prediction["samples"][:, 0]
        winners = samples.argmax(axis=1)
        order = np.argsort(-probabilities, kind="stable")
        results = []
        for rank, index in enumerate(order, 1):
            row = dict(rows[index])
            row.update(
                sbnn_rank=rank,
                conditional_probability=float(probabilities[index]),
                probability_lower_95=float(np.quantile(samples[:, index], 0.025)),
                probability_upper_95=float(np.quantile(samples[:, index], 0.975)),
                posterior_top1_fraction=float(np.mean(winners == index)),
                pair_id=pair_id(row),
            )
            results.append(row)
        with (output / f"{tier}_comparison.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(results)
        fields = (
            ["proband_id"]
            + [
                f"{field}_{allele}"
                for allele in (1, 2)
                for field in ("chrom", "pos", "ref", "alt")
            ]
            + ["epcr", "finding_type", "notes"]
        )
        with (output / f"{tier}_sbnn_candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in results[:10]:
                record = {key: row[key] for key in fields if key in row}
                for allele in (1, 2):
                    key = f"chrom_{allele}"
                    record[key] = "chr" + str(record[key]).removeprefix("chr")
                record.update(
                    proband_id="PROBAND01",
                    epcr=row["conditional_probability"],
                    finding_type="primary",
                    notes=f"gene={row['gene']};SBNN conditional candidate probability;phase={row['phase']}",
                )
                writer.writerow(record)
        summary[tier] = {
            "input_pairs": len(original),
            "eligible_pairs": len(rows),
            "excluded_cis": len(original) - len(rows),
            "top_gene": results[0]["gene"],
            "top_pair": results[0]["pair_id"],
            "top_conditional_probability": results[0]["conditional_probability"],
            "mutual_information": float(prediction["mutual_information"][0]),
        }
    report = {
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "source": str(source),
        "draws": args.draws,
        "seed": args.seed,
        "tiers": summary,
        "caveat": "Probabilities are conditional on the supplied shortlist and are not clinically calibrated.",
    }
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
