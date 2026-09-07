#!/usr/bin/env python3
"""Small regression tests for the independent Track 1 submission audit."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from audit_submission import REQUIRED, audit


class SubmissionAuditTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, str]]) -> Path:
        directory = Path(tempfile.mkdtemp(prefix="track1-audit-"))
        path = directory / "submission.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REQUIRED)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _row(self, *, epcr: str = "0.9", pos_1: str = "10") -> dict[str, str]:
        return {
            "proband_id": "P1", "chrom_1": "chr1", "pos_1": pos_1,
            "ref_1": "A", "alt_1": "G", "chrom_2": "chr1", "pos_2": "20",
            "ref_2": "C", "alt_2": "T", "epcr": epcr,
            "finding_type": "primary", "notes": "gene=GENE1;phase=unknown",
        }

    def test_valid_submission(self) -> None:
        result = audit(self._write([self._row()]))
        self.assertTrue(result["ok"])
        self.assertEqual(result["genes"], ["GENE1"])

    def test_duplicate_pair_is_rejected(self) -> None:
        result = audit(self._write([self._row(), self._row()]))
        self.assertFalse(result["ok"])
        self.assertIn("duplicate candidate pair detected", result["errors"])

    def test_invalid_epcr_is_rejected(self) -> None:
        result = audit(self._write([self._row(epcr="1.2")]))
        self.assertFalse(result["ok"])
        self.assertTrue(any("epcr" in error for error in result["errors"]))


if __name__ == "__main__":
    unittest.main()
