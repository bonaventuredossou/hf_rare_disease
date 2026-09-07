#!/usr/bin/env python3
"""Legacy 13-feature export from the ranked-pair TSV.

Kept for earlier experiments. The current fitted model uses 15 features and
reads the TSVs directly through run_bayesian_track1.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "pair_score",
    "phenotype_score",
    "phase_support",
    "min_variant_score",
    "mean_variant_score",
    "min_depth",
    "min_gq",
    "min_allele_balance",
    "max_population_af",
    "max_cadd_phred",
    "max_revel",
    "max_spliceai",
    "max_clinvar_score",
)
CONSEQUENCE_PHASE = {"trans_readbacked": 1.0, "unknown": 0.5, "cis_readbacked": 0.0}


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except (TypeError, ValueError):
        return default


def row_features(row: dict[str, str]) -> np.ndarray:
    afs = [_float(row, "af_1"), _float(row, "af_2")]
    cadds = [_float(row, "cadd_phred_1"), _float(row, "cadd_phred_2")]
    revels = [_float(row, "revel_1"), _float(row, "revel_2")]
    splice = [_float(row, "spliceai_1"), _float(row, "spliceai_2")]
    clinvar = [_float(row, "clinvar_score_1"), _float(row, "clinvar_score_2")]
    variants = [_float(row, "variant_score_1"), _float(row, "variant_score_2")]
    depths = [_float(row, "dp_1"), _float(row, "dp_2")]
    gqs = [_float(row, "gq_1"), _float(row, "gq_2")]
    balances = [_float(row, "ab_1"), _float(row, "ab_2")]
    observed_af = [x for x in afs if x > 0]
    return np.asarray(
        [
            _float(row, "pair_score"),
            _float(row, "phenotype_score"),
            CONSEQUENCE_PHASE.get(row.get("phase", "unknown"), 0.5),
            min(variants),
            np.mean(variants),
            min(depths),
            min(gqs),
            min(balances),
            max(observed_af, default=0.0),
            max(cadds),
            max(revels),
            max(splice),
            max(clinvar),
        ],
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/network/scratch/b/bonaventure.dossou/rare_disease_data"),
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--pair-table", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir or args.data_dir / "mva_track1_run"
    pair_table = args.pair_table or run_dir / "ranking" / "compound_het_pairs.tsv"
    output = args.output or run_dir / "ranking" / "bayesian_track1_input.npz"
    if not pair_table.is_file():
        raise FileNotFoundError(
            f"Pair table not found: {pair_table}. Run the upstream rank stage first."
        )
    with pair_table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise ValueError("Pair table contains no rows")
    features = np.stack([row_features(row) for row in rows])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        gene=np.asarray([row.get("gene", "") for row in rows]),
        phase=np.asarray([row.get("phase", "unknown") for row in rows]),
        pair_score=np.asarray([_float(row, "pair_score") for row in rows]),
    )
    manifest = output.with_suffix(".json")
    manifest.write_text(
        json.dumps(
            {
                "input_pair_table": str(pair_table.resolve()),
                "output_npz": str(output.resolve()),
                "rows": len(rows),
                "feature_names": list(FEATURE_NAMES),
                "challenge_fastq_or_vcf_read": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"rows": len(rows), "output": str(output), "manifest": str(manifest)},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
