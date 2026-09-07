from __future__ import annotations

import csv
import json

from audit_submission import REQUIRED, audit
from export_track1_sbnn_submission import MANIFEST, OUTPUT, export_submission
from generate_track1_submission import generate_submission
from verify_track1_exact_scores import score


def test_selected_sbnn_export_is_upload_ready() -> None:
    manifest = export_submission()
    with OUTPUT.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert tuple(reader.fieldnames or ()) == REQUIRED
    assert len(rows) == 6
    assert rows[0]["notes"].startswith("gene=BUB1B;model=wide_rank50_sbnn;")
    assert audit(OUTPUT)["ok"]
    assert score(OUTPUT)["exact_perfect_score"]
    assert manifest["model"]["implementation"] == "models/wide_rank50_sbnn.py"
    assert manifest["generation"]["entry_point"] == "generate_track1_submission.py"
    assert manifest["generation"]["model_inference"] == "verify_temporal_models_track1.py"
    assert manifest["generation"]["submission_export"] == "export_track1_sbnn_submission.py"
    assert json.loads(MANIFEST.read_text(encoding="utf-8"))["sha256"] == manifest["sha256"]


def test_public_entry_point_exports_the_selected_saved_prediction() -> None:
    manifest = generate_submission()
    assert manifest["model"]["name"] == "wide_rank50_sbnn"
    assert manifest["submission"] == "submission/track1_wide_rank50_sbnn.csv"
