#!/usr/bin/env python3
"""Retrain temporal ClinVar models and transfer them to the Track 1 shortlist.

The transfer is deliberately answer-blind: it does not use the submitted rank,
pair score, variant score, ClinVar significance, CADD, REVEL, or SpliceAI.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import pickle
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from clinvar_temporal_experiment import (
    Variant,
    balanced_cap,
    featurize,
    gene_statistics,
    parse_release,
)
from models.deep_ensemble import fit_deep_ensemble, predict_deep_ensemble
from models.dense_mlp import fit_dense_mlp, predict_dense_mlp
from models.full_rank_bbb import fit_full_rank_bbb, predict_full_rank_bbb
from models.gradient_boost import fit_gradient_boost, predict_gradient_boost
from models.small_sbnn import fit_small_sbnn, predict_small_sbnn
from models.temporal_consensus import predict_temporal_consensus
from models.wide_dense_bbb import fit_wide_dense_bbb, predict_wide_dense_bbb
from models.wide_rank50_sbnn import (
    fit_wide_rank50_sbnn,
    predict_wide_rank50_sbnn,
)
from verify_track1_exact_scores import score as official_score


SEED = 20260905
TIERS = {
    "strict": "ranking/compound_het_pairs.tsv",
    "broad": "ranking/compound_het_pairs_primary_lenient_audit.tsv",
}


def optional_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def consequence_name(code: str) -> str:
    lookup = {
        "MIS": "missense",
        "STG": "stop_gained nonsense",
        "FS": "frameshift",
        "SPL": "splice_donor splice_acceptor splice_region",
        "IFD": "inframe deletion",
        "IFI": "inframe insertion",
        "SYN": "synonymous",
        "INT": "intron",
        "UTR": "utr",
    }
    return lookup.get(code.upper(), code.lower())


def variant_type(ref: str, alt: str) -> str:
    if len(ref) == len(alt) == 1:
        return "single_nucleotide_variant"
    if len(ref) > len(alt):
        return "deletion indel"
    if len(alt) > len(ref):
        return "insertion indel"
    return "indel"


def challenge_variant(row: dict[str, str], allele: int, identifier: int) -> Variant:
    ref = row[f"ref_{allele}"].upper()
    alt = row[f"alt_{allele}"].upper()
    af = optional_float(row[f"af_{allele}"])
    return Variant(
        variation_id=identifier,
        chrom=str(row[f"chrom_{allele}"]).removeprefix("chr"),
        pos=int(row[f"pos_{allele}"]),
        ref=ref,
        alt=alt,
        gene=row["gene"],
        label=0,
        consequence=consequence_name(row[f"consequence_{allele}"]),
        variant_type=variant_type(ref, alt),
        review="",
        origin=1,
        af_esp=af,
        af_exac=None,
        af_tgp=None,
        has_rs=False,
    )


def read_pairs(archive_path: Path, member: str) -> list[dict[str, str]]:
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member) as binary:
            rows = list(csv.DictReader(io.TextIOWrapper(binary), delimiter="\t"))
    return [row for row in rows if row["phase"] != "cis_readbacked"]


def prepare_training(historical: list[Variant], cap: int, val_cap: int):
    labels = np.asarray([record.label for record in historical])
    groups = np.asarray([record.gene for record in historical])
    train_index, val_index = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(
            np.zeros(len(labels)), labels, groups
        )
    )
    train_index = balanced_cap(train_index, labels, cap, SEED)
    val_index = balanced_cap(val_index, labels, val_cap, SEED + 1)
    stats = gene_statistics(historical, train_index)
    train = [historical[int(index)] for index in train_index]
    val = [historical[int(index)] for index in val_index]
    x_train_raw = featurize(train, stats)
    x_val_raw = featurize(val, stats)
    y_train = np.asarray([record.label for record in train])
    y_val = np.asarray([record.label for record in val])
    scaler = StandardScaler().fit(x_train_raw)
    return x_train_raw, x_val_raw, y_train, y_val, scaler, stats


def fit_models(historical: list[Variant]):
    x_raw, xv_raw, y, yv, scaler, stats = prepare_training(historical, 50000, 15000)
    x = scaler.transform(x_raw).astype(np.float32)
    xv = scaler.transform(xv_raw).astype(np.float32)

    print("Training gradient boost", flush=True)
    tree = fit_gradient_boost(x_raw, y, seed=SEED)
    print("Training dense MLP", flush=True)
    mlp = fit_dense_mlp(x, y, seed=SEED)
    print("Training five-network ensemble", flush=True)
    ensemble = fit_deep_ensemble(x, y, seed=SEED)
    print("Training original small SBNN", flush=True)
    small_sbnn, _ = fit_small_sbnn(x, y, xv, yv, seed=SEED, epochs=80)

    x2_raw, xv2_raw, y2, yv2, scaler2, stats2 = prepare_training(historical, 30000, 7500)
    x2 = scaler2.transform(x2_raw).astype(np.float32)
    xv2 = scaler2.transform(xv2_raw).astype(np.float32)
    print("Training tuned dense BBB", flush=True)
    bbb, _ = fit_full_rank_bbb(x2, y2, xv2, yv2, seed=SEED, epochs=60)
    print("Training rank-50 wide SBNN", flush=True)
    wide_sbnn, _ = fit_wide_rank50_sbnn(
        x2, y2, xv2, yv2, seed=SEED, epochs=60
    )
    print("Training matching-width dense BBB", flush=True)
    wide_bbb, _ = fit_wide_dense_bbb(
        x2, y2, xv2, yv2, seed=SEED, epochs=60
    )
    return {
        "main": (tree, mlp, ensemble, small_sbnn, scaler, stats),
        "wide": (bbb, wide_sbnn, wide_bbb, scaler2, stats2),
    }


def allele_predictions(models, records: list[Variant]) -> dict[str, np.ndarray]:
    tree, mlp, ensemble, small_sbnn, scaler, stats = models["main"]
    bbb, wide_sbnn, wide_bbb, scaler2, stats2 = models["wide"]
    raw = featurize(records, stats)
    scaled = scaler.transform(raw).astype(np.float32)
    raw2 = featurize(records, stats2)
    scaled2 = scaler2.transform(raw2).astype(np.float32)
    prediction = {
        "gradient_boost": predict_gradient_boost(tree, raw),
        "dense_mlp": predict_dense_mlp(mlp, scaled),
        "deep_ensemble": predict_deep_ensemble(ensemble, scaled)[0],
        "small_sbnn": predict_small_sbnn(
            small_sbnn, scaled, draws=128, seed=SEED + 99
        )[0],
        "dense_bbb": predict_full_rank_bbb(
            bbb, scaled2, draws=128, seed=SEED + 99
        )[0],
        "wide_rank50_sbnn": predict_wide_rank50_sbnn(
            wide_sbnn, scaled2, draws=128, seed=SEED + 99
        )[0],
        "wide_dense_bbb": predict_wide_dense_bbb(
            wide_bbb, scaled2, draws=128, seed=SEED + 99
        )[0],
    }
    prediction["temporal_consensus"] = predict_temporal_consensus(prediction)
    return prediction


def write_submission(path: Path, rows: list[dict[str, str]], pair_scores: np.ndarray) -> None:
    order = np.argsort(-pair_scores, kind="stable")[:10]
    fields = ["proband_id"] + [
        f"{key}_{i}" for i in (1, 2) for key in ("chrom", "pos", "ref", "alt")
    ] + ["epcr", "finding_type", "notes"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in order:
            row = rows[int(index)]
            record = {key: row[key] for key in fields if key in row}
            for allele in (1, 2):
                record[f"chrom_{allele}"] = "chr" + str(record[f"chrom_{allele}"]).removeprefix("chr")
            record.update(
                proband_id="PROBAND01",
                epcr=float(np.clip(pair_scores[index], 1e-12, 1.0)),
                finding_type="primary",
                notes=f"gene={row['gene']};temporal-model transfer;phase={row['phase']}",
            )
            writer.writerow(record)


def run_transfer(
    *,
    ranking_zip: Path,
    output_dir: Path,
    model_checkpoint: Path | None = None,
    historical: Path | None = None,
) -> dict[str, object]:
    """Run every fitted model on the child candidate pairs and save an audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_checkpoint or output_dir / "trained_models.pkl"
    if checkpoint.exists():
        print(f"Loading trained models from {checkpoint}", flush=True)
        with checkpoint.open("rb") as handle:
            models = pickle.load(handle)
    else:
        if historical is None:
            raise ValueError(
                "A historical ClinVar VCF is required when the checkpoint does not exist"
            )
        print("Reading historical ClinVar", flush=True)
        historical_records, _ = parse_release(historical, collect_all_ids=False)
        models = fit_models(historical_records)
        with checkpoint.open("wb") as handle:
            pickle.dump(models, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved trained models to {checkpoint}", flush=True)
    audit = {}
    for tier, member in TIERS.items():
        rows = read_pairs(ranking_zip, member)
        records = [
            challenge_variant(row, allele, 2 * index + allele)
            for index, row in enumerate(rows)
            for allele in (1, 2)
        ]
        predictions = allele_predictions(models, records)
        phenotype = np.asarray([float(row["phenotype_score"]) for row in rows])
        audit[tier] = {}
        for name, allele_probability in predictions.items():
            pair = allele_probability.reshape(-1, 2)
            pathogenicity = np.sqrt(np.clip(pair[:, 0] * pair[:, 1], 0.0, 1.0))
            scores = pathogenicity * phenotype
            output = output_dir / f"{tier}_{name}.csv"
            write_submission(output, rows, scores)
            result = official_score(output.resolve())
            result["top_gene"] = rows[int(np.argmax(scores))]["gene"]
            result["top_pair_score"] = float(np.max(scores))
            audit[tier][name] = result
            print(tier, name, result["top_gene"], result["rank_points"], result["f_max"], flush=True)

    report = {
        "adapter": {
            "allele_target": "ClinVar pathogenic versus benign",
            "pair_rule": "sqrt(p_allele_1 * p_allele_2) * phenotype_score",
            "available_features": "sequence, variant type, consequence, population AF, training-only gene history",
            "forced_missing": "dbSNP identifier and ClinVar review status",
            "excluded_to_prevent_answer_leakage": "submitted rank, pair_score, variant_score, clin_sig, CADD, REVEL, SpliceAI",
        },
        "seed": SEED,
        "results": audit,
    }
    (output_dir / "track1_temporal_model_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical",
        type=Path,
        help="ClinVar 2023 VCF. Required only when no trained checkpoint exists.",
    )
    parser.add_argument("--ranking-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.model_checkpoint or args.output_dir / "trained_models.pkl"
    if not checkpoint.exists() and args.historical is None:
        parser.error("--historical is required when the model checkpoint does not exist")
    run_transfer(
        ranking_zip=args.ranking_zip,
        output_dir=args.output_dir,
        model_checkpoint=args.model_checkpoint,
        historical=args.historical,
    )


if __name__ == "__main__":
    main()
