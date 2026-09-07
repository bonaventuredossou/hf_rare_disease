#!/usr/bin/env python3
"""Create the canonical Track 1 CSV from the selected wide rank-50 SBNN run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from audit_submission import REQUIRED, audit
from verify_track1_exact_scores import score


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/track1_temporal_transfer_20260906/strict_wide_rank50_sbnn.csv"
RUN_AUDIT = ROOT / "results/track1_temporal_transfer_20260906/track1_temporal_model_audit.json"
OUTPUT = ROOT / "submission/track1_wide_rank50_sbnn.csv"
MANIFEST = ROOT / "submission/track1_wide_rank50_sbnn.manifest.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_submission() -> dict[str, object]:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE}. Run verify_temporal_models_track1.py first."
        )
    if not RUN_AUDIT.exists():
        raise FileNotFoundError(f"Missing run audit: {RUN_AUDIT}")

    run = json.loads(RUN_AUDIT.read_text(encoding="utf-8"))
    selected = run["results"]["strict"]["wide_rank50_sbnn"]
    expected_source = str(SOURCE.relative_to(ROOT))
    if selected["path"] != expected_source:
        raise ValueError(
            f"Run audit points to {selected['path']}, expected {expected_source}"
        )
    if not selected["exact_perfect_score"]:
        raise ValueError("The selected run did not pass the recorded Track 1 score check")

    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED:
            raise ValueError("The selected SBNN output does not match the Track 1 schema")
        rows = list(reader)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED)
        writer.writeheader()
        for row in rows:
            note_parts = [part for part in row["notes"].split(";") if part]
            note_parts = [part for part in note_parts if not part.startswith("model=")]
            row["notes"] = ";".join(
                [note_parts[0], "model=wide_rank50_sbnn", *note_parts[1:]]
            )
            writer.writerow(row)

    schema_check = audit(OUTPUT)
    score_check = score(OUTPUT)
    if not schema_check["ok"]:
        raise ValueError(f"Submission audit failed: {schema_check['errors']}")
    if not score_check["exact_perfect_score"]:
        raise ValueError("Canonical submission failed the local score check")

    manifest = {
        "submission": str(OUTPUT.relative_to(ROOT)),
        "sha256": file_sha256(OUTPUT),
        "source_prediction": expected_source,
        "source_prediction_sha256": file_sha256(SOURCE),
        "model": {
            "name": "wide_rank50_sbnn",
            "implementation": "models/wide_rank50_sbnn.py",
            "bayesian_layers": "sbnn_pair_ranker.py",
            "binary_training_loop": "clinvar_temporal_experiment.py",
            "input_features": 50,
            "hidden_units": [200, 100],
            "ranks": [50, 50, 1],
            "kl_weight": 0.01,
            "posterior_scale_multiplier": 0.1,
            "prior_scale": 0.3,
            "seed": run["seed"],
            "posterior_draws_for_track1": 128,
        },
        "training": {
            "labels": "ClinVar 2023 pathogenic versus benign variants",
            "challenge_answer_used_for_training": False,
            "pair_score": run["adapter"]["pair_rule"],
            "upstream_features": "Candidate pairs and phenotype scores from Bonaventure Dossou's genome-wide pipeline",
        },
        "generation": {
            "entry_point": "generate_track1_submission.py",
            "model_inference": "verify_temporal_models_track1.py",
            "raw_model_output": expected_source,
            "submission_export": "export_track1_sbnn_submission.py",
        },
        "local_official_equivalent_score": score_check,
        "upload_status": "Not yet uploaded to the Hugging Face leaderboard",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    print(json.dumps(export_submission(), indent=2))


if __name__ == "__main__":
    main()
