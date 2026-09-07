#!/usr/bin/env python3
"""Generate the canonical Track 1 CSV from the selected wide rank-50 SBNN."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from export_track1_sbnn_submission import SOURCE, export_submission
from verify_temporal_models_track1 import run_transfer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "results/track1_temporal_transfer_20260906"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "trained_models.pkl"


def generate_submission(
    *,
    ranking_zip: Path | None = None,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    historical: Path | None = None,
) -> dict[str, object]:
    """Run checkpoint inference when requested, then validate and export the SBNN CSV."""
    if ranking_zip is not None:
        run_transfer(
            ranking_zip=ranking_zip,
            output_dir=DEFAULT_OUTPUT_DIR,
            model_checkpoint=checkpoint,
            historical=historical,
        )
    elif not SOURCE.exists():
        raise FileNotFoundError(
            "The saved SBNN prediction is missing. Pass --ranking-zip to recreate it."
        )
    return export_submission()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking-zip",
        type=Path,
        help="Compact child candidate export. If omitted, the audited saved prediction is used.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--historical",
        type=Path,
        help="ClinVar 2023 VCF, required only if the checkpoint must be trained.",
    )
    args = parser.parse_args()
    manifest = generate_submission(
        ranking_zip=args.ranking_zip,
        checkpoint=args.checkpoint,
        historical=args.historical,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
