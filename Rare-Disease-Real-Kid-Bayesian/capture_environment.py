#!/usr/bin/env python3
"""Record the lightweight CPU environment and public-input hashes used here."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "pandas",
    "joblib",
    "threadpoolctl",
    "pytest",
    "pillow",
    "huggingface-hub",
)
PUBLIC_INPUTS = (
    "data/open/clinvar_temporal/clinvar_20231226.vcf.gz",
    "data/open/clinvar_temporal/clinvar_20260829.vcf.gz",
    "data/open/phenopacket_store/all_phenopackets_v0.1.27.zip",
    "data/open/phenopacket_store/hp_2023-10-09.obo",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mac_hardware() -> dict[str, str]:
    if platform.system() != "Darwin":
        return {}
    try:
        text = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {}
    allowed = {"Model Name", "Model Identifier", "Chip", "Total Number of Cores", "Memory"}
    result = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in allowed:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report" / "environment_manifest.json")
    args = parser.parse_args()
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware": mac_hardware(),
        "accelerator_used": "none; NumPy/SciPy/scikit-learn CPU execution",
        "linear_algebra": "NumPy linked against OpenBLAS 0.3.21",
        "packages": packages,
        "public_input_sha256": {
            relative: sha256(ROOT / relative) for relative in PUBLIC_INPUTS if (ROOT / relative).exists()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
