"""Features shared by training and prediction on the OpenCRAVAT pair tables.

Observed fractions distinguish missing predictors from measured zeros.
Filtering and phase checks stay outside the learned score.
"""

import math
import numpy as np
from variant_evidence import consequence_score, rarity_score

FEATURE_NAMES = (
    "phenotype_score",
    "min_consequence",
    "max_consequence",
    "min_rarity",
    "mean_rarity",
    "max_population_af",
    "af_observed_fraction",
    "min_cadd",
    "max_cadd",
    "cadd_observed_fraction",
    "min_spliceai",
    "max_spliceai",
    "spliceai_observed_fraction",
    "phenotype_x_min_consequence",
    "phenotype_x_min_rarity",
)
SCHEMA = "shared-pair-evidence-v1"


def optional(value):
    if value in (None, "", ".", "None", "NA"):
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def pair_features(alleles, phenotype):
    if len(alleles) != 2 or not 0 <= phenotype <= 1:
        raise ValueError("Expected two alleles and phenotype score in [0,1]")
    severity = [consequence_score(a["consequence"]) for a in alleles]
    af = [optional(a["af"]) for a in alleles]
    if any(v is not None and not 0 <= v <= 1 for v in af):
        raise ValueError("Invalid population AF")
    rarity = [rarity_score(v) for v in af]
    output = [
        phenotype,
        min(severity),
        max(severity),
        min(rarity),
        np.mean(rarity),
        max((v for v in af if v is not None), default=0),
        sum(v is not None for v in af) / 2,
    ]
    for field, scale in [("cadd", 40), ("spliceai", 1)]:
        values = [optional(a[field]) for a in alleles]
        if any(
            v is not None and (v < 0 or (field == "spliceai" and v > 1)) for v in values
        ):
            raise ValueError("Invalid predictor value")
        filled = [min(v / scale, 1) if v is not None else 0 for v in values]
        output.extend(
            [min(filled), max(filled), sum(v is not None for v in values) / 2]
        )
    output.extend([phenotype * min(severity), phenotype * min(rarity)])
    return np.array(output, dtype=float)


def table_features(row):
    required = ["phenotype_score"] + [
        f"{key}_{i}"
        for i in (1, 2)
        for key in ("consequence", "af", "cadd_phred", "spliceai")
    ]
    if any(key not in row for key in required):
        raise ValueError("Pair table lacks required feature columns")
    alleles = [
        dict(
            consequence=row[f"consequence_{i}"],
            af=row[f"af_{i}"],
            cadd=row[f"cadd_phred_{i}"],
            spliceai=row[f"spliceai_{i}"],
        )
        for i in (1, 2)
    ]
    return pair_features(alleles, float(row["phenotype_score"]))


def evidence_rule(x):
    """Fixed-weight baseline using the evidence available in the training data."""
    return (
        0.36 * (x[..., 1] + x[..., 2]) / 2
        + 0.26 * x[..., 4]
        + 0.15 * x[..., 0]
        + 0.08 * (x[..., 8] + x[..., 11]) / 2
    )


def pair_id(row):
    keys = [
        f"{str(row[f'chrom_{i}']).removeprefix('chr')}:{int(row[f'pos_{i}'])}:"
        f"{row[f'ref_{i}']}:{row[f'alt_{i}']}"
        for i in (1, 2)
    ]
    return "|".join(sorted(keys))
