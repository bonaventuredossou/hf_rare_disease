#!/usr/bin/env python3
"""Check the columns, pair IDs and score ordering in a Track 1 submission CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


REQUIRED = (
    "proband_id",
    "chrom_1",
    "pos_1",
    "ref_1",
    "alt_1",
    "chrom_2",
    "pos_2",
    "ref_2",
    "alt_2",
    "epcr",
    "finding_type",
    "notes",
)
GENE_RE = re.compile(r"(?:^|;)gene=([^;]+)")


def audit(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)

    errors: list[str] = []
    warnings: list[str] = []
    if fields != REQUIRED:
        errors.append(f"schema mismatch: expected {REQUIRED}, got {fields}")
    if not rows:
        errors.append("submission has no candidate rows")

    pair_keys: list[tuple[str, ...]] = []
    epcr: list[float] = []
    genes: list[str] = []
    for index, row in enumerate(rows, start=2):
        try:
            value = float(row["epcr"])
            if not 0.0 < value <= 1.0:
                errors.append(f"line {index}: epcr must be in (0, 1]")
            epcr.append(value)
        except (KeyError, TypeError, ValueError):
            errors.append(f"line {index}: invalid epcr")
        try:
            positions = (int(row["pos_1"]), int(row["pos_2"]))
            if min(positions) < 1:
                errors.append(f"line {index}: positions must be positive")
        except (KeyError, TypeError, ValueError):
            errors.append(f"line {index}: invalid position")
        if row.get("finding_type") != "primary":
            warnings.append(f"line {index}: finding_type is not primary")
        match = GENE_RE.search(row.get("notes", ""))
        genes.append(match.group(1) if match else "")
        pair_keys.append(tuple(row.get(key, "") for key in REQUIRED[1:9]))

    if len(pair_keys) != len(set(pair_keys)):
        errors.append("duplicate candidate pair detected")
    if epcr and any(a < b for a, b in zip(epcr, epcr[1:])):
        warnings.append("epcr is not monotonically non-increasing")
    if len(rows) > 10:
        errors.append("Track 1 submission contains more than 10 rows")
    if not all(genes):
        warnings.append("one or more rows do not expose gene=... in notes")

    return {
        "path": str(path.resolve()),
        "rows": len(rows),
        "unique_proband_ids": sorted({row.get("proband_id", "") for row in rows}),
        "genes": genes,
        "epcr": epcr,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.submission)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
