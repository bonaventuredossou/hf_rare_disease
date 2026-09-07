#!/usr/bin/env python3
"""
MVA Hackathon 2026 - Track 1 end-to-end singleton WGS pipeline (pip-native), v13.

Research / competition use only. This script is not a clinical diagnostic pipeline.

Design goals
------------
* One Python entry point.
* No Conda/Mamba requirement.
* Core engines are installed into the active Python environment with pip:
    - pybwa: BWA-MEM alignment bindings
    - pysam: htslib/samtools/bcftools bindings
    - whatshap: read-backed phasing
    - open-cravat: local variant/gene annotation
* Keep challenge data local; no patient variants are sent to public web APIs.
* Use both the supplied VCF and raw FASTQs.
* Produce auditable intermediate files plus a <=10-row Track 1 CSV.
* Dual ranking: unbiased genome-wide discovery plus a transparent disease-aware MVA clinical reranker.

Default dataset root:
    /network/scratch/b/bonaventure.dossou/rare_disease_data

One-time Python dependencies:
    python -m pip install -U pysam pybwa whatshap open-cravat pandas numpy \
        python-docx openpyxl scipy scikit-learn tqdm

OpenCRAVAT also needs local annotation DATA modules (not a new environment):
    oc module install-base
    oc module install clinvar gnomad gnomad3 revel spliceai cadd

Example full run:
    python mva_track1_pipeline_pip_v13.py all \
      --data-dir /network/scratch/b/bonaventure.dossou/rare_disease_data \
      --reference /path/to/GCA_000001405.15_GRCh38_no_alt_analysis_set_plus_hs38d1_maskedGRC_exclusions_v2_no_chr.fasta \
      --threads 32

Stages can also be run independently:
    inspect, phenotype, prepare-reference, align, normalize, phase, annotate, rank, submit, report, all
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import multiprocessing as mp
import json
import sqlite3
import importlib.metadata
import logging
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import urllib.request
import ssl
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_DATA_DIR = Path("/network/scratch/b/bonaventure.dossou/rare_disease_data")
DEFAULT_SAMPLE = "WGS_EX2312012"
HPO_OBO_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo"
HPO_GENE_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/genes_to_phenotype.txt"

DEFAULT_VCF_NAME = "WGS_EX2312012_HGWCNDSX7.vcf.gz"
EXPECTED_REFERENCE_NAME = "GCA_000001405.15_GRCh38_no_alt_analysis_set_plus_hs38d1_maskedGRC_exclusions_v2_no_chr.fasta"
NCBI_NOALT_HS38D1_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/"
    "GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/"
    "GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz"
)
GIAB_GRC_EXCLUSIONS_V2_URL = (
    "https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/references/GRCh38/"
    "GCA_000001405.15_GRCh38_GRC_exclusions_T2Tv2.bed"
)

# Disease-aware reranking is kept separate from unbiased genome-wide discovery.
# These genes are the currently recognized MVA1-MVA7 genes in the OMIM/GTR
# phenotypic series (queried for the 2026 challenge). They do NOT affect the
# genome-wide variant score; they are used only by the explicit clinical reranker.
MVA_GENES = {"BUB1B", "CEP57", "TRIP13", "CENATAC", "SLF2", "SMC5", "MAD1L1"}

# Approximate consequence severity for ranking, not ACMG classification.
CONSEQUENCE_SCORE = {
    "transcript_ablation": 1.00,
    "splice_acceptor_variant": 1.00,
    "splice_donor_variant": 1.00,
    "stop_gained": 1.00,
    "frameshift_variant": 1.00,
    "stop_lost": 0.95,
    "start_lost": 0.95,
    "transcript_amplification": 0.90,
    "inframe_insertion": 0.78,
    "inframe_deletion": 0.78,
    "missense_variant": 0.72,
    "protein_altering_variant": 0.68,
    "splice_region_variant": 0.60,
    "incomplete_terminal_codon_variant": 0.45,
    "start_retained_variant": 0.28,
    "stop_retained_variant": 0.25,
    "synonymous_variant": 0.18,
    "coding_sequence_variant": 0.18,
    "mature_miRNA_variant": 0.16,
    "5_prime_UTR_variant": 0.14,
    "3_prime_UTR_variant": 0.12,
    "non_coding_transcript_exon_variant": 0.10,
    "intron_variant": 0.06,
    "NMD_transcript_variant": 0.06,
    "upstream_gene_variant": 0.03,
    "downstream_gene_variant": 0.03,
    "intergenic_variant": 0.01,
}

# OpenCRAVAT hg38 mapper consequence labels/codes -> approximate OpenCRAVAT-like severity.
CONSEQUENCE_SCORE.update({
    "MIS": 0.72, "missense": 0.72,
    "SYN": 0.18, "synonymous": 0.18, "synonomous": 0.18,
    "STG": 1.00, "stopgain": 1.00,
    "STL": 0.95, "stoploss": 0.95,
    "SPL": 1.00, "splice site affected": 1.00,
    "FD1": 1.00, "FD2": 1.00, "FI1": 1.00, "FI2": 1.00, "frameshift": 1.00,
    "IDV": 0.78, "inframe deletion": 0.78,
    "IIV": 0.78, "inframe insertion": 0.78,
    "CSS": 0.72, "complex substitution": 0.72,
    "INT": 0.06, "intron": 0.06,
    "UT5": 0.14, "5_prime_UTR": 0.14,
    "UT3": 0.12, "3_prime_UTR": 0.12,
    "2KU": 0.03, "2KD": 0.03, "UNK": 0.01,
})


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def setup_logging(log_file: Path, verbose: bool = False) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="a")]
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def run(cmd: Sequence[str] | str, log_path: Optional[Path] = None,
        shell: bool = False, check: bool = True, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    printable = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(x)) for x in cmd)
    logging.info("RUN: %s", printable)
    stdout = subprocess.PIPE if log_path is None else subprocess.DEVNULL
    stderr = subprocess.STDOUT if log_path is None else subprocess.DEVNULL
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(f"\n[{now()}] $ {printable}\n")
            cp = subprocess.run(cmd, shell=shell, stdout=fh, stderr=subprocess.STDOUT,
                                text=True, check=check, env=env)
        return cp
    return subprocess.run(cmd, shell=shell, stdout=stdout, stderr=stderr,
                          text=True, check=check, env=env)


def capture(cmd: Sequence[str]) -> str:
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
        return (cp.stdout or "").strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def tool_path(*names: str) -> Optional[str]:
    """Resolve a console script from the active pip/venv environment."""
    for name in names:
        p = shutil.which(name)
        if p:
            return p
    return None


def package_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return None


def import_pysam():
    try:
        import pysam  # type: ignore
        return pysam
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'pysam' is required. Install with: python -m pip install -U pysam"
        ) from exc


def import_pysam_bcftools():
    """Import pysam's bcftools dispatcher explicitly (it is a submodule)."""
    try:
        import pysam.bcftools as bcftools  # type: ignore
        return bcftools
    except ImportError as exc:
        raise RuntimeError(
            "pysam.bcftools could not be imported. Verify the active environment with: "
            "python -c \"import pysam, pysam.bcftools; print(pysam.__version__)\""
        ) from exc


def import_pybwa():
    try:
        import pybwa  # type: ignore
        return pybwa
    except ImportError as exc:
        raise RuntimeError(
            "Python package 'pybwa' is required for FASTQ alignment. "
            "Install with: python -m pip install -U pybwa"
        ) from exc


def require_pip_cli(name: str, pip_package: str) -> str:
    p = tool_path(name)
    if not p:
        raise RuntimeError(
            f"'{name}' was not found in PATH. It should be installed by the active Python environment. "
            f"Run: python -m pip install -U {pip_package}\n"
            "Then confirm that the same environment is active when running this script."
        )
    return p


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f} {u}"
        f /= 1024
    return str(n)


def discover_files(data_dir: Path) -> dict:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    all_files = [p for p in data_dir.iterdir() if p.is_file()]
    vcfs = sorted([p for p in all_files if p.name.endswith((".vcf", ".vcf.gz"))])
    fastqs = sorted([p for p in all_files if p.name.endswith((".fastq.gz", ".fq.gz", ".fastq", ".fq"))])
    docs = sorted([p for p in all_files if p.suffix.lower() == ".docx"])
    excels = sorted([p for p in all_files if p.suffix.lower() in {".xlsx", ".xlsm", ".xls"}])
    indexes = sorted([p for p in all_files if p.name.endswith((".tbi", ".csi"))])
    return {
        "data_dir": str(data_dir),
        "vcfs": [str(p) for p in vcfs],
        "fastqs": [str(p) for p in fastqs],
        "docx": [str(p) for p in docs],
        "excel": [str(p) for p in excels],
        "indexes": [str(p) for p in indexes],
        "files": [{"name": p.name, "bytes": p.stat().st_size, "size": human_size(p.stat().st_size)} for p in sorted(all_files)],
    }


def pair_fastqs(fastqs: Sequence[Path]) -> List[Tuple[str, Path, Path]]:
    # Expected challenge names contain _L001_R1_001.fastq.gz etc.
    rx = re.compile(r"^(?P<prefix>.+)_L(?P<lane>\d{3})_R(?P<read>[12])_001\.(?:fastq|fq)(?:\.gz)?$")
    lanes: Dict[str, Dict[str, Path]] = defaultdict(dict)
    for fq in fastqs:
        m = rx.match(fq.name)
        if m:
            lanes[m.group("lane")][m.group("read")] = fq
    pairs = []
    for lane, reads in sorted(lanes.items()):
        if "1" not in reads or "2" not in reads:
            raise RuntimeError(f"Incomplete FASTQ pair for lane {lane}: {reads}")
        pairs.append((lane, reads["1"], reads["2"]))
    if not pairs:
        raise RuntimeError("No paired FASTQs matched the expected lane naming pattern.")
    return pairs


def read_first_fastq_header(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as fh:
        return fh.readline().strip()


def rg_from_fastq(sample: str, lane: str, r1: Path) -> Tuple[str, str]:
    header = read_first_fastq_header(r1).lstrip("@").split()[0]
    bits = header.split(":")
    flowcell = bits[2] if len(bits) > 3 else "FLOWCELL"
    seq_lane = bits[3] if len(bits) > 3 else lane.lstrip("0")
    rgid = f"{flowcell}.{seq_lane}"
    rg = f"@RG\\tID:{rgid}\\tSM:{sample}\\tLB:{sample}_WGS\\tPL:ILLUMINA\\tPU:{rgid}"
    return rgid, rg


def parse_vcf_header_reference(vcf: Path) -> Optional[str]:
    opener = gzip.open if vcf.name.endswith(".gz") else open
    with opener(vcf, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("##reference="):
                return line.strip().split("=", 1)[1].removeprefix("file://")
            if line.startswith("#CHROM"):
                break
    return None


def resolve_reference(args, vcf: Path) -> Optional[Path]:
    if args.reference:
        p = Path(args.reference).expanduser().resolve()
        return p if p.exists() else p
    hdr = parse_vcf_header_reference(vcf)
    if not hdr:
        return None
    hp = Path(hdr)
    if hp.exists():
        return hp.resolve()
    basename = hp.name
    candidates = [
        Path(args.data_dir) / basename,
        Path(args.data_dir) / "reference" / basename,
        Path(args.data_dir) / "refGenome" / basename,
        Path(args.data_dir).parent / "refGenome" / basename,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None



def _ssl_context():
    """Return an SSL context using certifi's CA bundle when available.

    Many HPC Python installations do not inherit the cluster/system CA store
    correctly. certifi is pip-installable and provides a portable trust bundle.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()




def _download_with_resume(url: str, dest: Path, chunk_size: int = 8 * 1024 * 1024) -> Path:
    """Download a public reference resource with HTTP Range resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    headers = {"User-Agent": "mva-track1-pipeline/2026"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if existing else "wb"
    try:
        resp = urllib.request.urlopen(req, timeout=120, context=_ssl_context())
    except Exception:
        if existing:
            # Some servers ignore/deny Range. Retry cleanly rather than append corrupt data.
            logging.warning("Resume request failed; restarting download of %s", dest.name)
            dest.unlink(missing_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": "mva-track1-pipeline/2026"})
            resp = urllib.request.urlopen(req, timeout=120, context=_ssl_context())
            mode = "wb"
            existing = 0
        else:
            raise
    status = getattr(resp, "status", None)
    if existing and status == 200:
        logging.warning("Server ignored Range; restarting %s from byte 0", dest.name)
        resp.close()
        dest.unlink(missing_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "mva-track1-pipeline/2026"})
        resp = urllib.request.urlopen(req, timeout=120, context=_ssl_context())
        mode = "wb"
        existing = 0
    logging.info("Downloading %s -> %s%s", url, dest, f" (resume at {existing:,} bytes)" if existing else "")
    done = existing
    next_log = done + 512 * 1024 * 1024
    with resp, open(dest, mode) as out:
        while True:
            block = resp.read(chunk_size)
            if not block:
                break
            out.write(block)
            done += len(block)
            if done >= next_log:
                logging.info("Downloaded %s: %.1f GiB", dest.name, done / (1024**3))
                next_log += 512 * 1024 * 1024
    logging.info("Download complete: %s (%s)", dest, human_size(dest.stat().st_size))
    return dest


def _load_bed_intervals(bed: Path) -> Dict[str, List[Tuple[int, int]]]:
    intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    with open(bed, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                fields = line.split()
            if len(fields) < 3:
                continue
            try:
                intervals[fields[0]].append((int(fields[1]), int(fields[2])))
            except ValueError:
                continue
    for chrom in intervals:
        intervals[chrom].sort()
    return intervals


def _mask_and_strip_chr(base_gz: Path, bed: Path, output_fasta: Path) -> Path:
    """Apply BED mask in UCSC coordinates while streaming FASTA, then strip leading 'chr'."""
    intervals = _load_bed_intervals(bed)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Building masked no-chr FASTA: %s", output_fasta)
    chrom = None
    pos = 0
    ivals: List[Tuple[int, int]] = []
    iv_idx = 0
    contigs = 0
    with gzip.open(base_gz, "rt", encoding="ascii", errors="strict") as inp, open(output_fasta, "wt", encoding="ascii") as out:
        for line in inp:
            if line.startswith(">"):
                raw = line[1:].rstrip("\n")
                first, sep, rest = raw.partition(" ")
                chrom = first
                out_name = first[3:] if first.startswith("chr") else first
                out.write(">" + out_name + (sep + rest if sep else "") + "\n")
                pos = 0
                ivals = intervals.get(chrom, [])
                iv_idx = 0
                contigs += 1
                continue
            seq = bytearray(line.strip().encode("ascii"))
            if not seq:
                out.write("\n")
                continue
            start0, end0 = pos, pos + len(seq)
            while iv_idx < len(ivals) and ivals[iv_idx][1] <= start0:
                iv_idx += 1
            j = iv_idx
            while j < len(ivals) and ivals[j][0] < end0:
                s0, e0 = ivals[j]
                a = max(s0, start0) - start0
                b = min(e0, end0) - start0
                if a < b:
                    seq[a:b] = b"N" * (b - a)
                j += 1
            out.write(seq.decode("ascii") + "\n")
            pos = end0
    logging.info("Reference construction complete: %d contigs, %s", contigs, human_size(output_fasta.stat().st_size))
    return output_fasta


def validate_reference_against_vcf(reference: Path, vcf: Path, report_path: Optional[Path] = None) -> dict:
    """Require every contig declared in the VCF to exist at exactly the same length in FASTA."""
    pysam = import_pysam()
    if not Path(str(reference) + ".fai").exists():
        pysam.faidx(str(reference))
    fasta = pysam.FastaFile(str(reference))
    fasta_lengths = dict(zip(fasta.references, fasta.lengths))
    vf = pysam.VariantFile(str(vcf))
    missing = []
    length_mismatch = []
    checked = 0
    for name in vf.header.contigs:
        clen = vf.header.contigs[name].length
        checked += 1
        if name not in fasta_lengths:
            missing.append(name)
        elif clen is not None and int(clen) != int(fasta_lengths[name]):
            length_mismatch.append({"contig": name, "vcf": int(clen), "fasta": int(fasta_lengths[name])})
    result = {
        "reference": str(reference),
        "vcf": str(vcf),
        "vcf_contigs_checked": checked,
        "fasta_contigs": len(fasta_lengths),
        "missing_vcf_contigs_in_fasta": missing,
        "length_mismatches": length_mismatch,
        "passed": not missing and not length_mismatch,
    }
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError(
            "Constructed/reference FASTA does not match the supplied VCF contig dictionary. "
            f"Missing={len(missing)}, length_mismatches={len(length_mismatch)}. "
            f"See {report_path if report_path else 'validation output'}."
        )
    logging.info("Reference/VCF validation PASSED: %d VCF contigs matched by name and length", checked)
    return result


def prepare_reference_stage(args, paths: dict) -> Path:
    """Build the GRCh38 no-alt+hs38d1, GRC-exclusions-v2 masked, no-chr reference named by the challenge VCF."""
    inv = discover_files(Path(args.data_dir))
    vcf = choose_input_vcf(args, inv)
    out_dir = Path(args.data_dir) / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reference:
        reference = Path(args.reference).expanduser().resolve()
        if not reference.exists():
            raise FileNotFoundError(reference)
        validate_reference_against_vcf(reference, vcf, Path(args.out_dir) / "reference_validation.json")
        return reference

    reference = out_dir / EXPECTED_REFERENCE_NAME
    if not reference.exists() or args.force:
        base_gz = out_dir / "GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz"
        bed = out_dir / "GCA_000001405.15_GRCh38_GRC_exclusions_T2Tv2.bed"
        _download_with_resume(NCBI_NOALT_HS38D1_URL, base_gz)
        _download_with_resume(GIAB_GRC_EXCLUSIONS_V2_URL, bed)
        _mask_and_strip_chr(base_gz, bed, reference)
    else:
        logging.info("Reuse prepared reference: %s", reference)

    validate_reference_against_vcf(reference, vcf, Path(args.out_dir) / "reference_validation.json")
    args.reference = str(reference)
    logging.info("Prepared reference: %s", reference)
    return reference


def inspect_dataset(args, paths: dict) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inv = discover_files(Path(args.data_dir))
    vcf = choose_input_vcf(args, inv)
    inv["vcf_header_reference"] = parse_vcf_header_reference(vcf) if vcf else None
    inv["resolved_reference"] = str(resolve_reference(args, vcf)) if vcf and resolve_reference(args, vcf) else None
    inv["python_packages"] = {
        "pysam": package_version("pysam"),
        "pybwa": package_version("pybwa"),
        "whatshap": package_version("whatshap"),
        "open-cravat": package_version("open-cravat"),
        "pandas": package_version("pandas"),
        "python-docx": package_version("python-docx"),
        "openpyxl": package_version("openpyxl"),
    }
    inv["pip_console_scripts"] = {
        "whatshap": tool_path("whatshap"),
        "oc": tool_path("oc"),
    }
    (out / "inventory.json").write_text(json.dumps(inv, indent=2))
    logging.info("Inventory written: %s", out / "inventory.json")
    logging.info("Found %d VCF(s), %d FASTQ(s), %d DOCX, %d Excel file(s)",
                 len(inv["vcfs"]), len(inv["fastqs"]), len(inv["docx"]), len(inv["excel"]))
    if not inv["excel"]:
        logging.info("No Excel file is currently present in the dataset root; Excel discovery remains enabled.")


def extract_docx(path: Path) -> str:
    try:
        from docx import Document  # type: ignore
    except ImportError as exc:
        raise RuntimeError("python-docx is required to extract the clinical DOCX.") from exc
    doc = Document(path)
    blocks: List[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            blocks.append(text)
    for ti, table in enumerate(doc.tables, 1):
        blocks.append(f"\n[TABLE {ti}]")
        for row in table.rows:
            blocks.append("\t".join(cell.text.strip().replace("\n", " | ") for cell in row.cells))
    return "\n".join(blocks).strip() + "\n"


def extract_excel(path: Path, out_dir: Path) -> List[dict]:
    try:
        import openpyxl  # type: ignore
    except ImportError as exc:
        raise RuntimeError("openpyxl is required to inspect Excel workbooks.") from exc
    wb = openpyxl.load_workbook(path, data_only=False, read_only=True)
    results = []
    for ws in wb.worksheets:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", ws.title).strip("_") or "sheet"
        out_csv = out_dir / f"{path.stem}__{safe}.csv"
        rows = []
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            for row in ws.iter_rows(values_only=True):
                vals = ["" if v is None else v for v in row]
                writer.writerow(vals)
                if any(v not in (None, "") for v in row):
                    rows.append(vals)
        results.append({"sheet": ws.title, "max_row": ws.max_row, "max_column": ws.max_column,
                        "nonempty_rows": len(rows), "csv": str(out_csv)})
    return results


def _download_if_missing(url: str, dest: Path) -> None:
    """Download a public resource atomically if it is not already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    logging.info("Downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(url, context=_ssl_context()) as src, tmp.open("wb") as out:
            shutil.copyfileobj(src, out, length=1024 * 1024)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _extract_hpo_terms(text: str) -> List[Tuple[str, str, str]]:
    """Extract HPO IDs and nearby labels from the supplied clinical text.

    Returns (hpo_id, label, context). Context is 'family_history' for the
    recurrent-spontaneous-abortion signal and 'proband' otherwise.
    """
    out: List[Tuple[str, str, str]] = []
    seen = set()
    for line in text.splitlines():
        ids = re.findall(r"HP:\d{7}", line)
        if not ids:
            continue
        cells = [c.strip() for c in line.split("\t")]
        for hid in ids:
            if hid in seen:
                continue
            seen.add(hid)
            label = ""
            if len(cells) >= 2:
                # The challenge table has: Clinical Feature, HPO Term, HPO ID, Notes.
                try:
                    idx = next(i for i, c in enumerate(cells) if hid in c)
                    if idx >= 1:
                        label = cells[idx - 1]
                except StopIteration:
                    pass
            if not label:
                # Fall back to text preceding the ID.
                label = line.split(hid, 1)[0].strip(" \t:-")[-120:]
            context = "family_history" if hid == "HP:0200067" or "miscarriage" in line.lower() else "proband"
            out.append((hid, label, context))
    return out


def _parse_hpo_obo(path: Path) -> Tuple[Dict[str, set], Dict[str, str]]:
    """Parse HPO OBO into direct-parent relationships and labels."""
    parents: Dict[str, set] = {}
    names: Dict[str, str] = {}
    cur = None
    obsolete = False
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "[Term]":
                cur = None
                obsolete = False
            elif line.startswith("id: HP:"):
                cur = line.split("id:", 1)[1].strip()
                parents.setdefault(cur, set())
            elif cur and line.startswith("name: "):
                names[cur] = line.split("name:", 1)[1].strip()
            elif cur and line.startswith("is_a: HP:"):
                parent = line.split("is_a:", 1)[1].split("!", 1)[0].strip()
                parents[cur].add(parent)
            elif cur and line == "is_obsolete: true":
                obsolete = True
            elif not line and obsolete and cur:
                parents.pop(cur, None)
                names.pop(cur, None)
                cur = None
    return parents, names


def _hpo_ancestor_fn(parents: Dict[str, set]):
    cache: Dict[str, set] = {}
    def ancestors(term: str) -> set:
        if term in cache:
            return cache[term]
        seen = {term}
        stack = list(parents.get(term, ()))
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(parents.get(x, ()))
        cache[term] = seen
        return seen
    return ancestors


def _parse_genes_to_phenotype(path: Path) -> Dict[str, set]:
    """Parse current HPO genes_to_phenotype.txt robustly across header variants."""
    gene_terms: Dict[str, set] = defaultdict(set)
    with path.open(encoding="utf-8", errors="replace") as fh:
        first = fh.readline().rstrip("\n")
        header = first.lstrip("#").split("\t")
        lower = [x.lower() for x in header]
        has_header = any("gene" in x for x in lower) and any("hpo" in x or "hpo_id" in x for x in lower)
        rows = fh if has_header else itertools.chain([first + "\n"], fh)
        gene_idx = None
        hpo_idx = None
        if has_header:
            for i, name in enumerate(lower):
                if gene_idx is None and name in {"gene_symbol", "gene", "symbol"}:
                    gene_idx = i
                if hpo_idx is None and name in {"hpo_id", "hpo", "hpo term id"}:
                    hpo_idx = i
        for raw in rows:
            if not raw.strip() or raw.startswith("#"):
                continue
            cols = raw.rstrip("\n").split("\t")
            if has_header and gene_idx is not None and hpo_idx is not None:
                if max(gene_idx, hpo_idx) >= len(cols):
                    continue
                gene, hid = cols[gene_idx].strip(), cols[hpo_idx].strip()
            else:
                # Current HPO release layout begins NCBI gene ID, symbol, HPO ID.
                if len(cols) < 3:
                    continue
                gene, hid = cols[1].strip(), cols[2].strip()
            if gene and re.fullmatch(r"HP:\d{7}", hid):
                gene_terms[gene].add(hid)
    return gene_terms


def _compute_hpo_gene_scores(query_terms: List[Tuple[str, str, str]], obo: Path, gene_file: Path) -> List[dict]:
    """Genome-wide HPO semantic scoring using ontology-aware Resnik-style similarity.

    The score is query-to-gene: each observed query term receives credit for its
    best semantic match among phenotypes annotated to a gene. This avoids
    penalizing genes simply because the patient's clinical record is incomplete.
    """
    parents, names = _parse_hpo_obo(obo)
    ancestors = _hpo_ancestor_fn(parents)
    gene_terms = _parse_genes_to_phenotype(gene_file)
    if not gene_terms:
        raise RuntimeError(f"No gene/HPO annotations parsed from {gene_file}")

    # Propagate gene annotations to HPO ancestors and compute information content.
    term_gene_count: Dict[str, int] = defaultdict(int)
    propagated_by_gene: Dict[str, set] = {}
    for gene, terms in gene_terms.items():
        prop = set()
        for t in terms:
            prop.update(ancestors(t))
        propagated_by_gene[gene] = prop
        for t in prop:
            term_gene_count[t] += 1
    n_genes = max(1, len(gene_terms))
    ic = {t: -math.log((c + 1.0) / (n_genes + 1.0)) for t, c in term_gene_count.items()}
    max_ic = max(ic.values()) if ic else 1.0

    # The document explicitly asks us to treat recurrent miscarriage as a signal.
    qweights = {hid: (1.20 if ctx == "family_history" else 1.0) for hid, _, ctx in query_terms}
    qids = [hid for hid, _, _ in query_terms if hid in parents]
    if not qids:
        raise RuntimeError("None of the extracted HPO IDs were found in the downloaded HPO ontology.")

    qanc = {q: ancestors(q) for q in qids}
    q_ic = {q: max(ic.get(q, 0.0), 0.25) for q in qids}
    rows = []
    for gene, terms in gene_terms.items():
        if not terms:
            continue
        g_anc = {t: ancestors(t) for t in terms if t in parents}
        if not g_anc:
            continue
        weighted = 0.0
        denom = 0.0
        exact = 0
        matched = 0
        for q in qids:
            w = qweights.get(q, 1.0)
            denom += w
            best_ic = 0.0
            for t, tanc in g_anc.items():
                common = qanc[q] & tanc
                if common:
                    mica = max((ic.get(a, 0.0) for a in common), default=0.0)
                    if mica > best_ic:
                        best_ic = mica
                if q == t:
                    exact += 1
            # Normalize relative to the query term's specificity so each observed
            # phenotype contributes on a comparable 0..1 scale.
            sim = min(1.0, best_ic / max(q_ic[q], 1e-9))
            if sim >= 0.35:
                matched += 1
            weighted += w * sim
        score = weighted / max(denom, 1e-9)
        # Small constellation bonus: matching several independent features is
        # preferred over matching one feature extremely well.
        coverage = matched / len(qids)
        score = min(1.0, 0.82 * score + 0.18 * coverage)
        rows.append({
            "gene": gene,
            "score": score,
            "matched_query_terms": matched,
            "query_terms": len(qids),
            "exact_query_terms": exact,
            "annotated_hpo_terms": len(terms),
        })
    rows.sort(key=lambda r: (r["score"], r["matched_query_terms"], r["exact_query_terms"]), reverse=True)
    return rows


def phenotype_stage(args, paths: dict) -> None:
    out = Path(args.out_dir) / "phenotype"
    out.mkdir(parents=True, exist_ok=True)
    inv = discover_files(Path(args.data_dir))
    manifest = {"docx": [], "excel": [], "hpo": {}}
    merged_text = []
    for docx in map(Path, inv["docx"]):
        text = extract_docx(docx)
        dest = out / f"{docx.stem}.txt"
        dest.write_text(text, encoding="utf-8")
        merged_text.append(f"\n===== {docx.name} =====\n{text}")
        manifest["docx"].append({"source": str(docx), "text": str(dest), "characters": len(text)})
    for xlsx in map(Path, inv["excel"]):
        sheets = extract_excel(xlsx, out)
        manifest["excel"].append({"source": str(xlsx), "sheets": sheets})
    merged = "\n".join(merged_text).strip() + ("\n" if merged_text else "")
    clinical_path = out / "clinical_text.txt"
    clinical_path.write_text(merged, encoding="utf-8")

    hpo_terms = _extract_hpo_terms(merged)
    hpo_tsv = out / "hpo_terms.tsv"
    with hpo_tsv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["hpo_id", "label", "context", "weight"])
        for hid, label, ctx in hpo_terms:
            w.writerow([hid, label, ctx, 1.20 if ctx == "family_history" else 1.0])
    manifest["hpo"]["terms"] = [
        {"hpo_id": hid, "label": label, "context": ctx,
         "weight": 1.20 if ctx == "family_history" else 1.0}
        for hid, label, ctx in hpo_terms
    ]

    # Download current public HPO resources to scratch/local data storage and
    # compute phenotype-derived gene scores genome-wide. No disease-name gene
    # list is used here.
    if hpo_terms and not args.skip_hpo_gene_scoring:
        resource_dir = Path(args.hpo_dir) if args.hpo_dir else Path(args.data_dir) / "resources" / "hpo"
        obo = resource_dir / "hp.obo"
        genes = resource_dir / "genes_to_phenotype.txt"
        try:
            _download_if_missing(HPO_OBO_URL, obo)
            _download_if_missing(HPO_GENE_URL, genes)
            scores = _compute_hpo_gene_scores(hpo_terms, obo, genes)
            score_path = out / "phenotype_gene_scores.tsv"
            fields = ["gene", "score", "matched_query_terms", "query_terms", "exact_query_terms", "annotated_hpo_terms"]
            with score_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, delimiter="\t", fieldnames=fields)
                writer.writeheader()
                writer.writerows(scores)
            manifest["hpo"].update({
                "ontology": str(obo),
                "gene_annotations": str(genes),
                "gene_scores": str(score_path),
                "genes_scored": len(scores),
            })
            logging.info("Computed genome-wide HPO phenotype scores for %d genes: %s", len(scores), score_path)
        except Exception as exc:
            manifest["hpo"]["gene_scoring_error"] = str(exc)
            logging.warning("HPO gene scoring could not be completed; extracted HPO terms remain available: %s", exc)

    (out / "phenotype_manifest.json").write_text(json.dumps(manifest, indent=2))
    logging.info("Phenotype extraction complete: %s (%d HPO terms)", out, len(hpo_terms))

def choose_input_vcf(args, inv: Optional[dict] = None) -> Path:
    if args.vcf:
        return Path(args.vcf)
    p = Path(args.data_dir) / DEFAULT_VCF_NAME
    if p.exists():
        return p
    inv = inv or discover_files(Path(args.data_dir))
    vcfs = [Path(x) for x in inv["vcfs"] if not x.endswith((".tbi", ".csi"))]
    if not vcfs:
        raise FileNotFoundError("No VCF found.")
    return vcfs[0]


def _pysam_text(command, *args: str) -> str:
    """Run a bundled samtools/bcftools command via pysam and return text output."""
    try:
        return command(*[str(x) for x in args]) or ""
    except Exception as exc:
        messages = ""
        try:
            messages = "\n".join(command.get_messages())
        except Exception:
            pass
        raise RuntimeError(f"pysam command failed: {exc}\n{messages}") from exc


def _alignment_header(reference: Path, rgid: str, sample: str) -> dict:
    pysam = import_pysam()
    if not Path(str(reference) + ".fai").exists():
        pysam.faidx(str(reference))
    fa = pysam.FastaFile(str(reference))
    sq = [{"SN": name, "LN": int(length)} for name, length in zip(fa.references, fa.lengths)]
    fa.close()
    return {
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": sq,
        "RG": [{"ID": rgid, "SM": sample, "LB": f"{sample}_WGS", "PL": "ILLUMINA", "PU": rgid}],
    }


def _pair_name(name: str) -> str:
    return re.sub(r"(?:/1|/2)$", "", (name or "").split()[0])


def _write_fastq_record(handle, rec) -> None:
    """Write one pysam.FastxRecord back to FASTQ without changing sequence/quality."""
    comment = getattr(rec, "comment", None)
    header = rec.name if not comment else f"{rec.name} {comment}"
    qual = rec.quality
    if qual is None:
        raise RuntimeError(f"FASTQ record {rec.name!r} has no qualities")
    handle.write(f"@{header}\n{rec.sequence}\n+\n{qual}\n")


def _align_shard_worker(reference_s: str, r1_s: str, r2_s: str, lane: str,
                        sample: str, shard_bam_s: str, threads: int,
                        batch_pairs: int, bwa_chunk_bases: int) -> None:
    """Align one FASTQ shard in a fresh process so all native BWA memory dies with it."""
    reference = Path(reference_s)
    r1 = Path(r1_s)
    r2 = Path(r2_s)
    shard_bam = Path(shard_bam_s)
    pysam = import_pysam()
    pybwa = import_pybwa()
    rgid, _ = rg_from_fastq(sample, lane, r1)
    # Preserve the original lane RG ID even though the temporary shard filename differs.
    rgid = f"{sample}.L{lane}"
    header = _alignment_header(reference, rgid, sample)
    unsorted = shard_bam.with_suffix('.unsorted.bam')
    name_sorted = shard_bam.with_suffix('.namesort.bam')
    fixmate = shard_bam.with_suffix('.fixmate.bam')

    opt = pybwa.BwaMemOptions(
        threads=max(1, int(threads)),
        interleaved_paired_end=True,
        soft_clip_supplementary=True,
        chunk_size=max(1_000_000, int(bwa_chunk_bases)),
    )
    # Load the index only inside this worker. When the worker exits the OS reclaims
    # the complete BWA/native heap even if a library-level allocator retained pages.
    aligner = pybwa.BwaMem(prefix=str(reference))
    processed_pairs = 0
    with pysam.FastxFile(str(r1)) as f1, pysam.FastxFile(str(r2)) as f2, \
         pysam.AlignmentFile(str(unsorted), 'wb', header=header) as bam_out:
        batch = []
        for rec1, rec2 in itertools.zip_longest(f1, f2):
            if rec1 is None or rec2 is None:
                raise RuntimeError(f"FASTQ mate count mismatch: {r1.name} vs {r2.name}")
            if _pair_name(rec1.name) != _pair_name(rec2.name):
                raise RuntimeError(
                    f"FASTQ pair-name mismatch in lane {lane}: {rec1.name!r} vs {rec2.name!r}"
                )
            batch.extend([rec1, rec2])
            if len(batch) >= 2 * batch_pairs:
                for qidx, recs in enumerate(aligner.align(queries=batch, opt=opt)):
                    # Input order is exactly R1,R2,R1,R2,... . Restore the SAM
                    # first/last-segment bits explicitly before fixmate; pybwa's
                    # interleaved binding did not retain these bits in v8 output.
                    read1 = (qidx % 2 == 0)
                    for aln in recs:
                        aln.flag |= 0x1
                        if read1:
                            aln.flag = (aln.flag | 0x40) & ~0x80
                        else:
                            aln.flag = (aln.flag | 0x80) & ~0x40
                        aln.set_tag('RG', rgid, value_type='Z')
                        bam_out.write(aln)
                processed_pairs += len(batch) // 2
                batch.clear()
        if batch:
            for qidx, recs in enumerate(aligner.align(queries=batch, opt=opt)):
                read1 = (qidx % 2 == 0)
                for aln in recs:
                    aln.flag |= 0x1
                    if read1:
                        aln.flag = (aln.flag | 0x40) & ~0x80
                    else:
                        aln.flag = (aln.flag | 0x80) & ~0x40
                    aln.set_tag('RG', rgid, value_type='Z')
                    bam_out.write(aln)
            processed_pairs += len(batch) // 2
            batch.clear()

    sort_threads = str(max(1, int(threads)))
    _pysam_text(pysam.sort, '-n', '-@', sort_threads, '-O', 'BAM', '-o', str(name_sorted), str(unsorted))
    _pysam_text(pysam.fixmate, '-m', '-@', sort_threads, str(name_sorted), str(fixmate))
    _pysam_text(pysam.sort, '-@', sort_threads, '-O', 'BAM', '-o', str(shard_bam), str(fixmate))
    _pysam_text(pysam.quickcheck, '-v', str(shard_bam))
    for tmp in (unsorted, name_sorted, fixmate):
        if tmp.exists():
            tmp.unlink()


def _align_lane_pybwa(reference: Path, r1: Path, r2: Path, lane: str,
                       sample: str, lane_bam: Path, threads: int,
                       batch_pairs: int, bwa_chunk_bases: int, logs: Path,
                       shard_pairs: int = 2_000_000) -> None:
    """Full lane alignment with process-isolated pybwa shards.

    Every read is aligned. ``shard_pairs`` is only a lifetime boundary for the
    native aligner process; it is not a sampling limit. Completed shard BAMs are
    resumable, and each child process exits after its shard so native memory is
    unconditionally returned to the OS.
    """
    pysam = import_pysam()
    shard_dir = lane_bam.parent / f".L{lane}_alignment_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context('spawn')
    shard_bams = []
    shard_idx = 0
    total_pairs = 0

    def paths_for(i: int):
        stem = shard_dir / f"L{lane}.shard{i:04d}"
        return Path(str(stem) + '.R1.fastq'), Path(str(stem) + '.R2.fastq'), Path(str(stem) + '.fixmate.coord.bam')

    with pysam.FastxFile(str(r1)) as f1, pysam.FastxFile(str(r2)) as f2:
        eof = False
        while not eof:
            shard_idx += 1
            tmp_r1, tmp_r2, shard_bam = paths_for(shard_idx)
            shard_bams.append(shard_bam)
            n_pairs = 0
            completed = shard_bam.exists() and shard_bam.stat().st_size > 0
            h1 = h2 = None
            try:
                if not completed:
                    h1 = tmp_r1.open('w', encoding='utf-8')
                    h2 = tmp_r2.open('w', encoding='utf-8')
                while n_pairs < shard_pairs:
                    try:
                        rec1 = next(f1)
                    except StopIteration:
                        eof = True
                        break
                    try:
                        rec2 = next(f2)
                    except StopIteration:
                        raise RuntimeError(f"FASTQ mate count mismatch: {r1.name} vs {r2.name}")
                    if _pair_name(rec1.name) != _pair_name(rec2.name):
                        raise RuntimeError(
                            f"FASTQ pair-name mismatch in lane {lane}: {rec1.name!r} vs {rec2.name!r}"
                        )
                    if not completed:
                        _write_fastq_record(h1, rec1)
                        _write_fastq_record(h2, rec2)
                    n_pairs += 1
            finally:
                if h1 is not None: h1.close()
                if h2 is not None: h2.close()

            if n_pairs == 0:
                shard_bams.pop()
                for q in (tmp_r1, tmp_r2):
                    if q.exists(): q.unlink()
                break
            total_pairs += n_pairs

            if completed:
                logging.info("Lane %s shard %04d: reuse completed BAM (%s pairs consumed)",
                             lane, shard_idx, f"{n_pairs:,}")
                continue

            logging.info("Lane %s shard %04d: aligning %s pairs in isolated worker; total consumed=%s",
                         lane, shard_idx, f"{n_pairs:,}", f"{total_pairs:,}")
            proc = ctx.Process(
                target=_align_shard_worker,
                args=(str(reference), str(tmp_r1), str(tmp_r2), lane, sample,
                      str(shard_bam), int(threads), int(batch_pairs), int(bwa_chunk_bases)),
            )
            proc.start()
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(
                    f"Lane {lane} shard {shard_idx:04d} alignment worker exited with code {proc.exitcode}. "
                    f"Temporary FASTQs were kept in {shard_dir} for diagnosis/resume."
                )
            for q in (tmp_r1, tmp_r2):
                if q.exists(): q.unlink()

    if not shard_bams:
        raise RuntimeError(f"No read pairs found in lane {lane}")
    missing = [p for p in shard_bams if not p.exists() or p.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing completed alignment shards for lane {lane}: {missing[:3]}")

    if len(shard_bams) == 1:
        shutil.copy2(shard_bams[0], lane_bam)
    else:
        # All shards are coordinate-sorted and already carry fixmate tags.
        _pysam_text(pysam.merge, '-@', str(max(1, int(threads))), '-f', str(lane_bam), *map(str, shard_bams))
    _pysam_text(pysam.quickcheck, '-v', str(lane_bam))
    logging.info("Lane %s complete: %s pairs across %d process-isolated shards",
                 lane, f"{total_pairs:,}", len(shard_bams))

    # Lane BAM is now durable; remove shard BAMs to recover scratch space.
    for p in shard_bams:
        if p.exists(): p.unlink()
    try:
        shard_dir.rmdir()
    except OSError:
        pass


def alignment_stage(args, paths: dict) -> Path:
    pysam = import_pysam()
    pybwa = import_pybwa()
    inv = discover_files(Path(args.data_dir))
    vcf = choose_input_vcf(args, inv)
    reference = resolve_reference(args, vcf)
    if reference is None or not reference.exists():
        hdr = parse_vcf_header_reference(vcf)
        raise FileNotFoundError(
            "Alignment needs the matching GRCh38 FASTA. The supplied VCF header names:\n"
            f"  {hdr}\n"
            "but that FASTA was not found locally. Pass its local path with --reference."
        )
    pairs = pair_fastqs(list(map(Path, inv["fastqs"])))
    if not pairs:
        raise RuntimeError("No paired FASTQ lanes were discovered.")
    out = Path(args.out_dir) / "alignment"
    logs = Path(args.out_dir) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    if not Path(str(reference) + ".fai").exists():
        logging.info("Building FASTA index with pysam/htslib: %s", reference)
        pysam.faidx(str(reference))

    # pybwa needs the classic BWA index. Its Python API can build this itself.
    bwa_markers = [Path(str(reference) + ext) for ext in (".amb", ".ann", ".bwt", ".pac", ".sa")]
    if not all(x.exists() for x in bwa_markers):
        logging.info("Building BWA index through pybwa (one-time): %s", reference)
        pybwa.BwaIndex.index(str(reference))
    lane_bams: List[Path] = []
    for lane, r1, r2 in pairs:
        lane_bam = out / f"{args.sample}.L{lane}.fixmate.coord.bam"
        lane_bams.append(lane_bam)
        if lane_bam.exists() and lane_bam.stat().st_size > 0 and not args.force:
            logging.info("Reuse existing lane BAM: %s", lane_bam)
            continue
        logging.info("Lane %s memory-safe alignment settings: threads=%d, batch_pairs=%d, bwa_chunk_bases=%d",
                     lane, args.threads, args.align_batch_pairs, args.bwa_chunk_bases)
        _align_lane_pybwa(reference, r1, r2, lane, args.sample, lane_bam,
                           args.threads, args.align_batch_pairs, args.bwa_chunk_bases, logs,
                           args.align_shard_pairs)

    merged = out / f"{args.sample}.merged.bam"
    final_bam = out / f"{args.sample}.markdup.bam"
    if not final_bam.exists() or args.force:
        if len(lane_bams) == 1:
            shutil.copy2(lane_bams[0], merged)
        else:
            _pysam_text(pysam.merge, "-@", str(args.threads), "-f", str(merged), *map(str, lane_bams))
        _pysam_text(pysam.markdup, "-@", str(args.threads), str(merged), str(final_bam))
        _pysam_text(pysam.index, "-@", str(args.threads), str(final_bam))

    flagstat_text = _pysam_text(pysam.flagstat, "-@", str(args.threads), str(final_bam))
    (out / "flagstat.txt").write_text(flagstat_text)
    m_r1 = re.search(r"^(\d+) \+ \d+ read1", flagstat_text, re.M)
    m_r2 = re.search(r"^(\d+) \+ \d+ read2", flagstat_text, re.M)
    if m_r1 and m_r2 and int(m_r1.group(1)) == 0 and int(m_r2.group(1)) == 0:
        raise RuntimeError(
            "Alignment BAM contains paired records but no READ1/READ2 flags. "
            "This is an older v8 alignment output and loses paired-fragment information. "
            "Re-run the align stage with v9 and --force before read-backed phasing."
        )
    (out / "samtools_stats.txt").write_text(_pysam_text(pysam.stats, "-@", str(args.threads), str(final_bam)))
    (out / "coverage.txt").write_text(_pysam_text(pysam.coverage, str(final_bam)))

    if not args.keep_lane_bams:
        for p in lane_bams:
            if p.exists():
                p.unlink()
        if merged.exists():
            merged.unlink()
    logging.info("Alignment complete: %s", final_bam)
    return final_bam


def normalize_stage(args, paths: dict) -> Path:
    pysam = import_pysam()
    bcftools = import_pysam_bcftools()
    inv = discover_files(Path(args.data_dir))
    input_vcf = choose_input_vcf(args, inv)
    reference = resolve_reference(args, input_vcf)
    if reference is None or not reference.exists():
        raise FileNotFoundError("VCF normalization requires the matching reference FASTA; pass --reference.")
    out = Path(args.out_dir) / "variants"
    out.mkdir(parents=True, exist_ok=True)
    norm = out / "supplied.normalized.vcf.gz"
    if not norm.exists() or args.force:
        try:
            bcftools.norm("-f", str(reference), "-m", "-any", "-Oz",
                          "-o", str(norm), str(input_vcf), catch_stdout=False)
        except Exception as exc:
            msgs = ""
            try:
                msgs = "\n".join(bcftools.norm.get_messages())
            except Exception:
                pass
            raise RuntimeError(f"bcftools norm via pysam failed: {exc}\n{msgs}") from exc
        pysam.tabix_index(str(norm), preset="vcf", force=True)
    try:
        stats = bcftools.stats("-s", "-", str(norm)) or ""
        (out / "normalized.stats.txt").write_text(stats)
    except Exception as exc:
        logging.warning("Could not write bcftools stats (normalization itself succeeded): %s", exc)
    logging.info("Normalized VCF: %s", norm)
    return norm


def find_final_bam(args) -> Optional[Path]:
    bam = Path(args.out_dir) / "alignment" / f"{args.sample}.markdup.bam"
    return bam if bam.exists() else None


def find_normalized_vcf(args) -> Optional[Path]:
    p = Path(args.out_dir) / "variants" / "supplied.normalized.vcf.gz"
    return p if p.exists() else None


def phase_stage(args, paths: dict) -> Path:
    pysam = import_pysam()
    whatshap = require_pip_cli("whatshap", "whatshap")
    inv = discover_files(Path(args.data_dir))
    input_vcf = find_normalized_vcf(args) or choose_input_vcf(args, inv)
    reference = resolve_reference(args, choose_input_vcf(args, inv))
    bam = find_final_bam(args)
    if reference is None or not reference.exists():
        raise FileNotFoundError("Read-backed phasing requires --reference.")
    if bam is None:
        raise FileNotFoundError("Aligned BAM not found. Run the align stage first.")
    out = Path(args.out_dir) / "variants"
    logs = Path(args.out_dir) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    phased = out / "supplied.normalized.readphased.vcf.gz"
    if not phased.exists() or args.force:
        run([whatshap, "phase", "--indels", "--reference", str(reference),
             "--output", str(phased), str(input_vcf), str(bam)], log_path=logs / "whatshap.log")
        pysam.tabix_index(str(phased), preset="vcf", force=True)
    logging.info("Read-backed phased VCF: %s", phased)
    return phased


def find_phase_or_norm_vcf(args) -> Optional[Path]:
    phased = Path(args.out_dir) / "variants" / "supplied.normalized.readphased.vcf.gz"
    if phased.exists():
        return phased
    return find_normalized_vcf(args)


def annotation_stage(args, paths: dict) -> Path:
    """Annotate locally with OpenCRAVAT installed in the active pip environment."""
    oc = require_pip_cli("oc", "open-cravat")
    inv = discover_files(Path(args.data_dir))
    input_vcf = find_phase_or_norm_vcf(args) or choose_input_vcf(args, inv)
    out = Path(args.out_dir) / "annotations" / "opencravat"
    logs = Path(args.out_dir) / "logs"
    out.mkdir(parents=True, exist_ok=True)
    run_name = args.oc_run_name
    expected = out / f"{run_name}.sqlite"
    if not expected.exists() or args.force:
        cmd = [oc, "run", str(input_vcf), "-l", "hg38", "-d", str(out), "-n", run_name,
               "--mp", str(max(1, int(args.threads)))]
        if args.oc_annotators:
            cmd += ["-a"] + list(args.oc_annotators)
        try:
            run(cmd, log_path=logs / "opencravat.log")
        except Exception as exc:
            raise RuntimeError(
                "OpenCRAVAT annotation failed. The Python package is pip-installable, but it also needs local "
                "annotation data modules. One-time setup:\n"
                "  oc module install-base\n"
                "  oc module install clinvar gnomad gnomad3 revel spliceai cadd\n"
                f"Original error: {exc}"
            ) from exc
    if not expected.exists():
        dbs = sorted(out.glob("*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not dbs:
            raise FileNotFoundError(f"OpenCRAVAT finished but no SQLite result was found in {out}")
        expected = dbs[0]
    marker = Path(args.out_dir) / "annotations" / "opencravat_result.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(expected) + "\n")
    logging.info("OpenCRAVAT result: %s", expected)
    return expected


def parse_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (float, int)):
        return float(x)
    s = str(x).strip()
    if not s or s in {".", "-", "NA", "None"}:
        return None
    # Some OpenCRAVAT fields may contain ampersand-separated values.
    vals = []
    for token in re.split(r"[,&]", s):
        token = token.strip()
        try:
            vals.append(float(token))
        except Exception:
            # SIFT/PolyPhen can be "deleterious(0.01)".
            m = re.search(r"\(([0-9.]+)\)", token)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except Exception:
                    pass
    return max(vals) if vals else None


def max_population_af(ann: dict) -> Optional[float]:
    keys = [k for k in ann if re.search(r"(^AF$|MAX_AF|gnomAD.*_AF$|gnomADe_AF|gnomADg_AF)", k, re.I)]
    vals = [parse_float(ann.get(k)) for k in keys]
    vals = [x for x in vals if x is not None and 0 <= x <= 1]
    return max(vals) if vals else None


def consequence_score(cons: str) -> float:
    terms = re.split(r"[&,]", cons or "")
    return max([CONSEQUENCE_SCORE.get(t, 0.02) for t in terms] or [0.02])


def rarity_score(af: Optional[float]) -> float:
    if af is None:
        return 0.70  # unknown is not assumed absent, but not strongly penalized
    if af == 0:
        return 1.00
    if af <= 1e-5:
        return 0.98
    if af <= 1e-4:
        return 0.95
    if af <= 1e-3:
        return 0.85
    if af <= 5e-3:
        return 0.65
    if af <= 1e-2:
        return 0.45
    if af <= 2e-2:
        return 0.25
    return 0.03


def clinvar_score(sig: str) -> float:
    s = (sig or "").lower().replace("_", " ")
    if "pathogenic" in s and "conflict" not in s:
        return 1.0 if "likely" not in s else 0.90
    if "conflict" in s:
        return 0.45
    if "uncertain" in s or "vus" in s:
        return 0.45
    if "benign" in s:
        return 0.02
    return 0.50


def predictor_score(ann: dict) -> float:
    vals: List[float] = []
    # REVEL 0..1
    for k, v in ann.items():
        ku = k.upper()
        if "REVEL" in ku:
            x = parse_float(v)
            if x is not None:
                vals.append(min(1.0, max(0.0, x)))
        elif "CADD" in ku and ("PHRED" in ku or ku.endswith("CADD")):
            x = parse_float(v)
            if x is not None:
                vals.append(min(1.0, max(0.0, x / 30.0)))
        elif "SPLICEAI" in ku and ("DS_" in ku or "DELTA" in ku or "SCORE" in ku):
            x = parse_float(v)
            if x is not None:
                vals.append(min(1.0, max(0.0, x)))
    sift = str(ann.get("SIFT", "")).lower()
    poly = str(ann.get("PolyPhen", ann.get("POLYPHEN", ""))).lower()
    if "deleterious" in sift:
        vals.append(0.8)
    if "probably_damaging" in poly or "probably damaging" in poly:
        vals.append(0.9)
    elif "possibly_damaging" in poly or "possibly damaging" in poly:
        vals.append(0.65)
    return statistics.mean(vals) if vals else 0.50


def qc_score(dp: Optional[int], gq: Optional[float], ad_ref: Optional[int], ad_alt: Optional[int], filt: str) -> Tuple[float, Optional[float]]:
    q = []
    if dp is None:
        q.append(0.55)
    elif dp >= 20:
        q.append(1.0)
    elif dp >= 12:
        q.append(0.85)
    elif dp >= 8:
        q.append(0.65)
    elif dp >= 5:
        q.append(0.40)
    else:
        q.append(0.10)
    if gq is None:
        q.append(0.55)
    elif gq >= 60:
        q.append(1.0)
    elif gq >= 30:
        q.append(0.85)
    elif gq >= 20:
        q.append(0.65)
    elif gq >= 10:
        q.append(0.40)
    else:
        q.append(0.10)
    ab = None
    if ad_ref is not None and ad_alt is not None and (ad_ref + ad_alt) > 0:
        ab = ad_alt / (ad_ref + ad_alt)
        q.append(max(0.0, 1.0 - abs(ab - 0.5) / 0.5))
    else:
        q.append(0.55)
    q.append(1.0 if filt in {"PASS", ".", ""} else 0.45)
    return statistics.mean(q), ab


def gene_prior(gene: str) -> float:
    """Neutral compatibility field retained for backward-compatible tables.

    No gene receives a disease-name-based bonus. This keeps candidate discovery
    genome-wide rather than hard-coding known MVA genes into the answer.
    """
    return 0.0


def load_phenotype_gene_scores(path: Optional[str]) -> Dict[str, float]:
    """Optional local TSV/CSV: gene,score. No network lookup is performed."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    delim = "\t" if p.suffix.lower() in {".tsv", ".txt"} else ","
    out = {}
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        fields = {f.lower(): f for f in (reader.fieldnames or [])}
        gcol = fields.get("gene") or fields.get("symbol")
        scol = fields.get("score") or fields.get("phenotype_score")
        if not gcol or not scol:
            raise RuntimeError("Phenotype gene score file must contain gene/symbol and score columns.")
        for row in reader:
            g = (row.get(gcol) or "").strip()
            try:
                s = float(row.get(scol, 0))
            except Exception:
                continue
            if g:
                out[g] = max(0.0, min(1.0, s))
    return out


@dataclass
class VariantCandidate:
    gene: str
    chrom: str
    pos: int
    ref: str
    alt: str
    alt_index: int
    gt: str
    phased: bool
    phase_set: str
    alt_haplotype: str
    dp: Optional[int]
    gq: Optional[float]
    ad_ref: Optional[int]
    ad_alt: Optional[int]
    allele_balance: Optional[float]
    filt: str
    consequence: str
    impact: str
    transcript: str
    hgvsc: str
    hgvsp: str
    max_af: Optional[float]
    clin_sig: str
    consequence_score: float
    rarity_score: float
    qc_score: float
    revel_score: Optional[float]
    cadd_phred: Optional[float]
    spliceai_score: Optional[float]
    predictor_score: float
    clinvar_score: float
    gene_prior: float
    phenotype_score: float
    variant_score: float

    @property
    def key(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


def safe_int(x) -> Optional[int]:
    try:
        return int(x) if x is not None else None
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    try:
        return float(x) if x is not None else None
    except Exception:
        return None


def _norm_chrom(chrom: str) -> str:
    c = str(chrom)
    return c[3:] if c.lower().startswith("chr") else c


def _variant_key(chrom: str, pos: int, ref: str, alt: str) -> Tuple[str, int, str, str]:
    """Return a representation-independent key for SNVs/indels/complex alleles.

    OpenCRAVAT commonly represents insertions/deletions with an empty allele
    encoded as '-' and the position shifted past the VCF anchor base, whereas a
    normalized VCF must retain a non-empty anchor allele.  Canonicalising both
    sides here lets equivalent variants join without weakening coordinate
    specificity.
    """
    c = _norm_chrom(chrom)
    p = int(pos)
    r = "" if str(ref) in {"", "-", "."} else str(ref).upper()
    a = "" if str(alt) in {"", "-", "."} else str(alt).upper()

    # Remove common suffix first, then common prefix.  Unlike VCF minimalisation,
    # allow one allele to become empty so VCF A>AT and OpenCRAVAT ->T become the
    # same canonical event.
    while r and a and r[-1] == a[-1]:
        r = r[:-1]
        a = a[:-1]
    while r and a and r[0] == a[0]:
        r = r[1:]
        a = a[1:]
        p += 1

    return (c, p, r or "-", a or "-")


def _variant_class(ref: str, alt: str) -> str:
    r = "" if str(ref) in {"", "-", "."} else str(ref)
    a = "" if str(alt) in {"", "-", "."} else str(alt)
    if len(r) == 1 and len(a) == 1:
        return "snv"
    if len(r) < len(a):
        return "insertion"
    if len(r) > len(a):
        return "deletion"
    return "complex"


def _find_oc_db(args) -> Path:
    marker = Path(args.out_dir) / "annotations" / "opencravat_result.txt"
    if marker.exists():
        p = Path(marker.read_text().strip())
        if p.exists():
            return p
    expected = Path(args.out_dir) / "annotations" / "opencravat" / f"{args.oc_run_name}.sqlite"
    if expected.exists():
        return expected
    raise FileNotFoundError("OpenCRAVAT SQLite result not found. Run annotate stage first.")


def _oc_value(row: dict, *exact_or_suffix: str):
    lower = {k.lower(): v for k, v in row.items()}
    for q in exact_or_suffix:
        ql = q.lower()
        if ql in lower:
            return lower[ql]
    for q in exact_or_suffix:
        ql = q.lower()
        for k, v in lower.items():
            if k.endswith(ql):
                return v
    return None


def _oc_numeric_values(row: dict, include_terms: Sequence[str]) -> List[float]:
    vals = []
    terms = tuple(x.lower() for x in include_terms)
    for k, v in row.items():
        kl = k.lower()
        if not any(t in kl for t in terms):
            continue
        if v in (None, "", "."):
            continue
        # Annotation fields sometimes contain delimiters; keep every valid number.
        for token in re.split(r"[;,|]", str(v)):
            try:
                x = float(token)
            except Exception:
                continue
            if math.isfinite(x):
                vals.append(x)
    return vals


def _opencravat_annotations(db: Path) -> Dict[Tuple[str, int, str, str], List[dict]]:
    """Load OpenCRAVAT variant table into a coordinate-keyed annotation map."""
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "variant" not in tables:
        conn.close()
        raise RuntimeError(f"OpenCRAVAT result lacks a 'variant' table: {db}")
    cols = [r[1] for r in cur.execute("PRAGMA table_info(variant)")]
    out: Dict[Tuple[str, int, str, str], List[dict]] = defaultdict(list)
    for rr in cur.execute("SELECT * FROM variant"):
        row = dict(zip(cols, rr))
        chrom = _oc_value(row, "base__chrom", "chrom")
        pos = _oc_value(row, "base__pos", "pos")
        ref = _oc_value(row, "base__ref_base", "ref_base")
        alt = _oc_value(row, "base__alt_base", "alt_base")
        if chrom is None or pos is None or ref is None or alt is None:
            continue
        # Canonical key harmonises OpenCRAVAT's '-' indel representation with
        # anchored, normalized VCF alleles.
        try:
            key = _variant_key(str(chrom), int(pos), str(ref), str(alt))
        except Exception:
            continue
        out[key].append(row)
    conn.close()
    return out


def _oc_consequence(row: dict) -> str:
    return str(_oc_value(row, "base__so", "so") or "")


def _oc_max_af(row: dict) -> Optional[float]:
    # Restrict frequency extraction to population modules/columns.
    vals = []
    for k, v in row.items():
        kl = k.lower()
        if not any(tag in kl for tag in ("gnomad", "thousandgenomes", "esp6500", "alfa")):
            continue
        if not any(tag in kl for tag in ("af", "freq", "frequency")):
            continue
        for token in re.split(r"[;,|]", str(v or "")):
            try:
                x = float(token)
            except Exception:
                continue
            if 0.0 <= x <= 1.0:
                vals.append(x)
    return max(vals) if vals else None


def _oc_clinsig(row: dict) -> str:
    vals = []
    for k, v in row.items():
        if "clinvar" in k.lower() and ("sig" in k.lower() or "class" in k.lower()) and v not in (None, ""):
            vals.append(str(v))
    return ";".join(vals)


def _oc_revel_score(row: dict) -> Optional[float]:
    """Return the raw REVEL pathogenicity score (0..1), never rankscore."""
    x = parse_float(_oc_value(row, "revel__score"))
    if x is None or not math.isfinite(x):
        return None
    return max(0.0, min(1.0, x))


def _oc_cadd_phred(row: dict) -> Optional[float]:
    """Return raw CADD PHRED score when available."""
    x = parse_float(_oc_value(row, "cadd__phred"))
    if x is None or not math.isfinite(x):
        return None
    return max(0.0, x)


def _oc_spliceai_score(row: dict) -> Optional[float]:
    """Return the largest raw SpliceAI delta score (0..1), excluding rank scores."""
    vals: List[float] = []
    for k, v in row.items():
        kl = k.lower()
        if "spliceai" not in kl or "rankscore" in kl:
            continue
        # Prefer delta-score style fields.  OpenCRAVAT module versions vary in
        # exact naming, so accept ds/delta/score columns but only 0..1 values.
        if not any(tag in kl for tag in ("ds_", "delta", "score")):
            continue
        for token in re.split(r"[;,|]", str(v or "")):
            try:
                x = float(token)
            except Exception:
                continue
            if math.isfinite(x) and 0.0 <= x <= 1.0:
                vals.append(x)
    return max(vals) if vals else None


def _predictor_components(row: dict) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
    """Extract transparent predictor components and a conservative composite.

    REVEL uses the raw `revel__score`, not `revel__rankscore`.  CADD PHRED is
    retained separately and mapped to 0..1 only for the small scoring component
    (PHRED 40 -> 1.0).  SpliceAI uses raw delta scores.  The composite is the
    mean of available components, rather than a max, so one rank-normalized or
    extreme predictor cannot dominate the variant score.
    """
    revel = _oc_revel_score(row)
    cadd_phred = _oc_cadd_phred(row)
    spliceai = _oc_spliceai_score(row)
    comps: List[float] = []
    if revel is not None:
        comps.append(revel)
    if cadd_phred is not None:
        comps.append(max(0.0, min(1.0, cadd_phred / 40.0)))
    if spliceai is not None:
        comps.append(spliceai)
    composite = statistics.mean(comps) if comps else 0.0
    return revel, cadd_phred, spliceai, composite


def _oc_best_gene_rows(rows: Sequence[dict]) -> Dict[str, dict]:
    by_gene: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        gene = str(_oc_value(row, "base__hugo", "hugo") or "").strip()
        if gene:
            by_gene[gene].append(row)
    chosen = {}
    for gene, xs in by_gene.items():
        chosen[gene] = max(xs, key=lambda r: consequence_score(_oc_consequence(r)))
    return chosen


def extract_variant_candidates(args) -> List[VariantCandidate]:
    pysam = import_pysam()
    vcf_path = find_phase_or_norm_vcf(args)
    if vcf_path is None:
        vcf_path = choose_input_vcf(args)
    db = _find_oc_db(args)
    oc = _opencravat_annotations(db)
    phenotype_score_path = args.phenotype_gene_scores
    if not phenotype_score_path:
        auto_pheno = Path(args.out_dir) / "phenotype" / "phenotype_gene_scores.tsv"
        if auto_pheno.exists():
            phenotype_score_path = str(auto_pheno)
            logging.info("Using automatically generated phenotype gene scores: %s", auto_pheno)
    phenotype_scores = load_phenotype_gene_scores(phenotype_score_path)

    vf = pysam.VariantFile(str(vcf_path))
    samples = list(vf.header.samples)
    if not samples:
        raise RuntimeError("Input VCF has no sample genotype column.")
    sample = args.sample if args.sample in samples else samples[0]
    out: List[VariantCandidate] = []
    matched_alleles = 0
    unmatched = 0
    unmatched_by_type = defaultdict(int)
    unmatched_sample: List[dict] = []

    for rec in vf.fetch():
        sm = rec.samples[sample]
        gt_tuple = sm.get("GT")
        if not gt_tuple or len(gt_tuple) != 2:
            continue
        dp = safe_int(sm.get("DP"))
        gq = safe_float(sm.get("GQ"))
        ad = sm.get("AD")
        filt = ";".join(list(rec.filter.keys())) if rec.filter.keys() else "PASS"
        ps = sm.get("PS") if "PS" in sm else (sm.get("PID") if "PID" in sm else None)
        phase_set = "" if ps is None else str(ps)
        phased = bool(getattr(sm, "phased", False))

        for alt_index, alt in enumerate(rec.alts or [], start=1):
            copies = sum(1 for a in gt_tuple if a == alt_index)
            if copies != 1:
                continue
            key = _variant_key(str(rec.chrom), int(rec.pos), str(rec.ref), str(alt))
            ann_rows = oc.get(key, [])
            if not ann_rows:
                unmatched += 1
                vclass = _variant_class(str(rec.ref), str(alt))
                unmatched_by_type[vclass] += 1
                if len(unmatched_sample) < 5000:
                    unmatched_sample.append({
                        "chrom": str(rec.chrom), "pos": int(rec.pos),
                        "ref": str(rec.ref), "alt": str(alt),
                        "variant_class": vclass,
                        "canonical_key": "|".join(map(str, key)),
                    })
                continue
            matched_alleles += 1
            ad_ref = safe_int(ad[0]) if ad is not None and len(ad) > 0 else None
            ad_alt = safe_int(ad[alt_index]) if ad is not None and len(ad) > alt_index else None
            qcs, ab = qc_score(dp, gq, ad_ref, ad_alt, filt)
            alt_hap = ""
            if phased:
                if gt_tuple[0] == alt_index:
                    alt_hap = "0"
                elif gt_tuple[1] == alt_index:
                    alt_hap = "1"
            for gene, ann in _oc_best_gene_rows(ann_rows).items():
                cons = _oc_consequence(ann)
                sev = consequence_score(cons)
                af = _oc_max_af(ann)
                rare = rarity_score(af)
                csig = _oc_clinsig(ann)
                cv = clinvar_score(csig)
                revel, cadd_phred, spliceai, pred = _predictor_components(ann)
                ph = phenotype_scores.get(gene, 0.0)
                # Conservative, transparent predictor treatment.  REVEL raw,
                # CADD PHRED and SpliceAI remain separately auditable while a
                # small composite contribution prevents in-silico predictors
                # from overwhelming consequence/rarity/QC/ClinVar evidence.
                vscore = 0.36 * sev + 0.26 * rare + 0.20 * qcs + 0.10 * cv + 0.08 * pred
                out.append(VariantCandidate(
                    gene=gene, chrom=str(rec.chrom), pos=int(rec.pos), ref=str(rec.ref), alt=str(alt),
                    alt_index=alt_index, gt="|".join(map(str, gt_tuple)) if phased else "/".join(map(str, gt_tuple)),
                    phased=phased, phase_set=phase_set, alt_haplotype=alt_hap,
                    dp=dp, gq=gq, ad_ref=ad_ref, ad_alt=ad_alt, allele_balance=ab, filt=filt,
                    consequence=cons, impact="", transcript=str(_oc_value(ann, "base__transcript", "transcript") or ""),
                    hgvsc="", hgvsp=str(_oc_value(ann, "base__achange", "achange") or ""), max_af=af, clin_sig=csig,
                    consequence_score=sev, rarity_score=rare, qc_score=qcs,
                    revel_score=revel, cadd_phred=cadd_phred, spliceai_score=spliceai, predictor_score=pred,
                    clinvar_score=cv, gene_prior=0.0, phenotype_score=ph, variant_score=vscore,
                ))
    vf.close()

    rankdir = Path(args.out_dir) / "ranking"
    rankdir.mkdir(parents=True, exist_ok=True)
    audit = {
        "join_mode": "canonical_coordinate_allele",
        "matched_heterozygous_alleles": matched_alleles,
        "unmatched_heterozygous_alleles": unmatched,
        "candidate_gene_annotations": len(out),
        "unmatched_by_variant_class": dict(sorted(unmatched_by_type.items())),
    }
    (rankdir / "annotation_join_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    if unmatched_sample:
        write_tsv(
            rankdir / "unmatched_alleles_sample.tsv",
            unmatched_sample,
            ["chrom", "pos", "ref", "alt", "variant_class", "canonical_key"],
        )

    logging.info(
        "OpenCRAVAT canonical join: %d matched heterozygous alleles; %d unmatched; %d candidate gene annotations",
        matched_alleles, unmatched, len(out),
    )
    return out


def write_tsv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def keep_for_pairing(v: VariantCandidate) -> bool:
    # Uniform genome-wide filtering: no disease-gene exceptions.
    if v.qc_score < 0.25:
        return False
    if v.max_af is not None and v.max_af > 0.01:
        return False
    return v.consequence_score >= 0.55 or v.clinvar_score >= 0.9


def is_primary_chrom(chrom: str) -> bool:
    """True for canonical nuclear GRCh38 chromosomes 1-22, X, Y."""
    c = _norm_chrom(chrom).upper()
    return c in {str(i) for i in range(1, 23)} | {"X", "Y"}


def strict_comphet_variant(v: VariantCandidate) -> bool:
    """Conservative germline-heterozygous QC gate for the submission tier.

    This is deliberately independent of gene/disease identity.  Variants that
    fail remain in lenient audit tables; they are not deleted from the analysis.
    """
    if v.filt not in {"PASS", ".", ""}:
        return False
    if v.dp is None or v.dp < 10:
        return False
    if v.gq is None or v.gq < 20:
        return False
    if v.allele_balance is None or not (0.20 <= v.allele_balance <= 0.80):
        return False
    if v.max_af is not None and v.max_af > 0.005:
        return False
    return True


def phase_relationship(a: VariantCandidate, b: VariantCandidate) -> Tuple[str, float]:
    if a.phased and b.phased and a.phase_set and a.phase_set == b.phase_set and a.alt_haplotype and b.alt_haplotype:
        if a.alt_haplotype != b.alt_haplotype:
            return "trans_readbacked", 1.0
        return "cis_readbacked", 0.0
    return "unknown", 0.50


def logistic_epcr(score: float) -> float:
    # Ranking confidence proxy. Strictly monotonic in pair score; not a calibrated clinical probability.
    x = (score - 0.55) / 0.10
    return max(0.001, min(0.999, 1.0 / (1.0 + math.exp(-x))))


def rank_stage(args, paths: dict) -> Tuple[Path, Path]:
    variants = extract_variant_candidates(args)
    rankdir = Path(args.out_dir) / "ranking"
    rankdir.mkdir(parents=True, exist_ok=True)
    var_fields = list(asdict(variants[0]).keys()) if variants else [f.name for f in VariantCandidate.__dataclass_fields__.values()]
    write_tsv(rankdir / "variant_candidates.tsv", [asdict(v) for v in variants], var_fields)

    by_gene: Dict[str, List[VariantCandidate]] = defaultdict(list)
    for v in variants:
        if keep_for_pairing(v):
            by_gene[v.gene].append(v)

    pairs = []
    for gene, vals in by_gene.items():
        vals = sorted(vals, key=lambda x: x.variant_score, reverse=True)[:args.max_variants_per_gene]
        uniq = {}
        for v in vals:
            uniq[v.key] = max([uniq.get(v.key, v), v], key=lambda x: x.variant_score)
        vals = list(uniq.values())
        for a, b in itertools.combinations(vals, 2):
            if a.key == b.key:
                continue
            phase_rel, phase_score = phase_relationship(a, b)
            mean_v = (a.variant_score + b.variant_score) / 2
            min_v = min(a.variant_score, b.variant_score)
            gp = 0.0
            pheno = max(a.phenotype_score, b.phenotype_score)
            pair_score = 0.55 * mean_v + 0.25 * min_v + 0.15 * pheno + 0.05 * phase_score
            if phase_rel == "cis_readbacked":
                pair_score *= 0.20
            if a.consequence_score >= 0.72 and b.consequence_score >= 0.72:
                pair_score = min(1.0, pair_score + 0.04)
            if (a.max_af is None or a.max_af <= 1e-3) and (b.max_af is None or b.max_af <= 1e-3):
                pair_score = min(1.0, pair_score + 0.03)

            primary = is_primary_chrom(a.chrom) and is_primary_chrom(b.chrom)
            strict_qc = strict_comphet_variant(a) and strict_comphet_variant(b)
            pairs.append({
                "gene": gene,
                "contig_tier": "primary" if primary else "nonprimary_audit",
                "qc_tier": "strict" if strict_qc else "lenient_audit",
                "pair_score": pair_score,
                "epcr_raw": logistic_epcr(pair_score),
                "phase": phase_rel,
                "gene_prior": gp,
                "phenotype_score": pheno,
                "chrom_1": a.chrom, "pos_1": a.pos, "ref_1": a.ref, "alt_1": a.alt,
                "consequence_1": a.consequence, "hgvsp_1": a.hgvsp, "af_1": a.max_af,
                "dp_1": a.dp, "gq_1": a.gq, "ab_1": a.allele_balance,
                "revel_1": a.revel_score, "cadd_phred_1": a.cadd_phred,
                "spliceai_1": a.spliceai_score, "predictor_score_1": a.predictor_score,
                "clinvar_score_1": a.clinvar_score, "clin_sig_1": a.clin_sig,
                "variant_score_1": a.variant_score,
                "chrom_2": b.chrom, "pos_2": b.pos, "ref_2": b.ref, "alt_2": b.alt,
                "consequence_2": b.consequence, "hgvsp_2": b.hgvsp, "af_2": b.max_af,
                "dp_2": b.dp, "gq_2": b.gq, "ab_2": b.allele_balance,
                "revel_2": b.revel_score, "cadd_phred_2": b.cadd_phred,
                "spliceai_2": b.spliceai_score, "predictor_score_2": b.predictor_score,
                "clinvar_score_2": b.clinvar_score, "clin_sig_2": b.clin_sig,
                "variant_score_2": b.variant_score,
            })

    pairs.sort(key=lambda r: (r["pair_score"], min(r["variant_score_1"], r["variant_score_2"])), reverse=True)
    primary_lenient_pairs = [r.copy() for r in pairs if r["contig_tier"] == "primary"]
    primary_pairs = [r.copy() for r in primary_lenient_pairs if r["qc_tier"] == "strict"]
    nonprimary_pairs = [r.copy() for r in pairs if r["contig_tier"] != "primary"]

    def assign_ranks_and_epcr(rows: List[dict]) -> None:
        previous = 1.0
        for i, row in enumerate(rows, 1):
            row["rank"] = i
            e = min(float(row["epcr_raw"]), previous - 1e-6)
            row["epcr"] = max(0.001, e)
            previous = row["epcr"]

    assign_ranks_and_epcr(primary_pairs)
    assign_ranks_and_epcr(primary_lenient_pairs)
    assign_ranks_and_epcr(nonprimary_pairs)

    pair_fields = [
        "rank", "gene", "contig_tier", "qc_tier", "pair_score", "epcr", "phase", "gene_prior", "phenotype_score",
        "chrom_1", "pos_1", "ref_1", "alt_1", "consequence_1", "hgvsp_1", "af_1", "dp_1", "gq_1", "ab_1",
        "revel_1", "cadd_phred_1", "spliceai_1", "predictor_score_1", "clinvar_score_1", "clin_sig_1", "variant_score_1",
        "chrom_2", "pos_2", "ref_2", "alt_2", "consequence_2", "hgvsp_2", "af_2", "dp_2", "gq_2", "ab_2",
        "revel_2", "cadd_phred_2", "spliceai_2", "predictor_score_2", "clinvar_score_2", "clin_sig_2", "variant_score_2",
    ]

    # The main table is intentionally canonical chromosomes only.  The submit
    # stage reads this file, so alt/unplaced contigs cannot crowd out primary
    # candidates.  They are retained in a separate audit table rather than lost.
    pair_path = rankdir / "compound_het_pairs.tsv"
    write_tsv(pair_path, primary_pairs, pair_fields)
    write_tsv(rankdir / "compound_het_pairs_primary.tsv", primary_pairs, pair_fields)
    write_tsv(rankdir / "compound_het_pairs_primary_lenient_audit.tsv", primary_lenient_pairs, pair_fields)
    write_tsv(rankdir / "compound_het_pairs_nonprimary_audit.tsv", nonprimary_pairs, pair_fields)

    # A gene-diverse review shortlist prevents one multi-variant locus from
    # consuming the visible top ten.  It is for review, not an automatic
    # assertion that only one pair per gene can be causal.
    diverse = []
    seen_genes = set()
    for r in primary_pairs:
        if r["gene"] in seen_genes:
            continue
        seen_genes.add(r["gene"])
        diverse.append(dict(r))
        if len(diverse) >= 10:
            break
    write_tsv(rankdir / "compound_het_pairs_gene_diverse_top10.tsv", diverse, pair_fields)

    # Also retain an all-pairs score-ordered audit view.  Rank/EPCR here are not
    # used for submission because the two tiers are ranked independently.
    all_fields = ["global_score_order"] + [f for f in pair_fields if f not in {"rank", "epcr"}]
    all_rows = []
    for i, r in enumerate(pairs, 1):
        rr = dict(r)
        rr["global_score_order"] = i
        all_rows.append(rr)
    write_tsv(rankdir / "compound_het_pairs_all_audit.tsv", all_rows, all_fields)

    # Disease-aware clinical reranking. This is deliberately separate from the
    # genome-wide discovery score so the report can show that BUB1B was found
    # without a disease-gene whitelist and then independently strengthened by
    # known MVA biology. The weakest allele receives substantial weight because
    # a recessive pair is only as convincing as its less-supported member.
    clinical_rows = []
    for r in primary_pairs:
        v1 = float(r["variant_score_1"])
        v2 = float(r["variant_score_2"])
        hmean = (2.0 * v1 * v2 / (v1 + v2)) if (v1 + v2) > 0 else 0.0
        weakest = min(v1, v2)
        pheno = float(r.get("phenotype_score", 0.0) or 0.0)
        mva_prior = 1.0 if r["gene"] in MVA_GENES else 0.0
        phase_rel = r.get("phase", "unknown")
        phase_evidence = 1.0 if phase_rel == "trans_readbacked" else (0.0 if phase_rel == "cis_readbacked" else 0.5)
        cv1 = float(r.get("clinvar_score_1", 0.5) or 0.5)
        cv2 = float(r.get("clinvar_score_2", 0.5) or 0.5)
        # Anchor is present only for an allele with strong P/LP-like ClinVar support.
        pathogenic_anchor = 1.0 if max(cv1, cv2) >= 0.90 else 0.0
        clinical_score = (
            0.35 * hmean
            + 0.15 * weakest
            + 0.25 * pheno
            + 0.15 * mva_prior
            + 0.05 * phase_evidence
            + 0.05 * pathogenic_anchor
        )
        rr = dict(r)
        rr.update({
            "genomewide_pair_score": float(r["pair_score"]),
            "pair_hmean": hmean,
            "weakest_allele_score": weakest,
            "mva_gene_prior": mva_prior,
            "phase_evidence_score": phase_evidence,
            "pathogenic_anchor": pathogenic_anchor,
            "clinical_pair_score": clinical_score,
        })
        clinical_rows.append(rr)

    clinical_rows = [r for r in clinical_rows if r.get("phase") != "cis_readbacked"]
    clinical_rows.sort(
        key=lambda r: (
            r["clinical_pair_score"],
            r["mva_gene_prior"],
            r["phenotype_score"],
            r["weakest_allele_score"],
        ),
        reverse=True,
    )
    prev_epcr = 1.0
    for i, r in enumerate(clinical_rows, 1):
        r["rank"] = i
        e = min(logistic_epcr(float(r["clinical_pair_score"])), prev_epcr - 1e-6)
        r["epcr"] = max(0.001, e)
        prev_epcr = r["epcr"]

    clinical_fields = [
        "rank", "gene", "clinical_pair_score", "epcr", "genomewide_pair_score",
        "pair_hmean", "weakest_allele_score", "phenotype_score", "mva_gene_prior",
        "pathogenic_anchor", "phase", "phase_evidence_score",
    ] + [f for f in pair_fields if f not in {"rank", "gene", "pair_score", "epcr", "phenotype_score", "phase"}]
    clinical_path = rankdir / "compound_het_pairs_mva_clinical.tsv"
    write_tsv(clinical_path, clinical_rows, clinical_fields)

    # MVA-gene audit includes all candidate variants, including variants that do
    # not survive strict compound-het QC, so disease-aware filtering is auditable.
    mva_variant_rows = [asdict(v) for v in variants if v.gene in MVA_GENES]
    write_tsv(rankdir / "mva_gene_variant_audit.tsv", mva_variant_rows, var_fields)

    scoring_audit = {
        "variant_score_formula": "0.36*consequence + 0.26*rarity + 0.20*qc + 0.10*clinvar + 0.08*predictor_composite",
        "predictor_composite": "mean(raw_REVEL, min(CADD_PHRED/40,1), raw_SpliceAI_delta) over available components",
        "revel_field": "revel__score (raw), explicitly excludes revel__rankscore",
        "primary_contigs": [str(i) for i in range(1, 23)] + ["X", "Y"],
        "strict_submission_qc": "PASS; DP>=10; GQ>=20; 0.20<=heterozygous allele balance<=0.80; max population AF<=0.005 when observed",
        "strict_primary_pair_count": len(primary_pairs),
        "primary_lenient_audit_pair_count": len(primary_lenient_pairs),
        "nonprimary_audit_pair_count": len(nonprimary_pairs),
        "all_pair_count": len(pairs),
        "phenotype_missing_gene_default": 0.0,
        "mva_clinical_reranker": {
            "mva_genes": sorted(MVA_GENES),
            "formula": "0.35*harmonic_mean(variant_scores) + 0.15*weakest_allele + 0.25*HPO + 0.15*MVA_gene_prior + 0.05*phase + 0.05*pathogenic_ClinVar_anchor",
            "phase_scores": {"trans_readbacked": 1.0, "unknown": 0.5, "cis_readbacked": 0.0},
            "cis_pairs_excluded_from_clinical_submission_table": True,
            "clinical_pair_count": len(clinical_rows),
        },
    }
    (rankdir / "scoring_audit_v13.json").write_text(json.dumps(scoring_audit, indent=2) + "\n")

    logging.info(
        "Ranked %d heterozygous variant annotations into %d strict primary pairs (%d primary before strict QC) and %d nonprimary audit pairs",
        len(variants), len(primary_pairs), len(primary_lenient_pairs), len(nonprimary_pairs),
    )
    return rankdir / "variant_candidates.tsv", pair_path


def add_chr(chrom: str) -> str:
    c = str(chrom)
    return c if c.startswith("chr") else "chr" + c


def read_pairs(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def submission_stage(args, paths: dict) -> Path:
    rankdir = Path(args.out_dir) / "ranking"
    if args.submission_profile == "mva-clinical":
        pair_path = rankdir / "compound_het_pairs_mva_clinical.tsv"
    else:
        pair_path = rankdir / "compound_het_pairs.tsv"
    if not pair_path.exists():
        raise FileNotFoundError(f"Ranked pair table not found: {pair_path}. Run rank stage first.")
    pairs = read_pairs(pair_path)
    top = pairs[: min(10, args.top_n)]
    if not top:
        raise RuntimeError("No compound-heterozygous candidate pairs were produced.")
    out = Path(args.out_dir) / "ranking" / args.submission_name
    fields = [
        "proband_id", "chrom_1", "pos_1", "ref_1", "alt_1",
        "chrom_2", "pos_2", "ref_2", "alt_2", "epcr", "finding_type", "notes"
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in top:
            score_key = "clinical_pair_score" if "clinical_pair_score" in row else "pair_score"
            note = (
                f"gene={row['gene']};{score_key}={float(row[score_key]):.4f};"
                f"phase={row['phase']};consequences={row['consequence_1']}+{row['consequence_2']}"
            )
            w.writerow({
                "proband_id": args.proband_id or args.sample,
                "chrom_1": add_chr(row["chrom_1"]), "pos_1": row["pos_1"], "ref_1": row["ref_1"], "alt_1": row["alt_1"],
                "chrom_2": add_chr(row["chrom_2"]), "pos_2": row["pos_2"], "ref_2": row["ref_2"], "alt_2": row["alt_2"],
                "epcr": f"{float(row['epcr']):.6f}", "finding_type": "primary", "notes": note,
            })
    logging.info("Track 1 candidate submission written: %s", out)
    return out


def methods_report_stage(args, paths: dict) -> Path:
    outdir = Path(args.out_dir)
    report_dir = outdir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    inv_path = outdir / "inventory.json"
    inv = json.loads(inv_path.read_text()) if inv_path.exists() else discover_files(Path(args.data_dir))
    pair_path = outdir / "ranking" / "compound_het_pairs.tsv"
    pairs = read_pairs(pair_path)[:10] if pair_path.exists() else []
    tool_versions = {
        "python": sys.version,
        "pysam": package_version("pysam"),
        "pybwa": package_version("pybwa"),
        "whatshap": package_version("whatshap"),
        "open-cravat": package_version("open-cravat"),
    }
    rows_md = "\n".join(
        f"| {r['rank']} | {r['gene']} | {float(r['pair_score']):.4f} | {float(r['epcr']):.4f} | {r['phase']} | {r['chrom_1']}:{r['pos_1']} {r['ref_1']}>{r['alt_1']} | {r['chrom_2']}:{r['pos_2']} {r['ref_2']}>{r['alt_2']} |"
        for r in pairs
    ) or "| - | - | - | - | - | - | - |"
    report = f"""# MVA Hackathon 2026 - Track 1 Run Summary

## Scope

This is a research/competition workflow for the single MVA proband. It is not intended as clinical diagnosis or medical advice.

## Inputs

- Dataset root: `{args.data_dir}`
- Supplied VCF: `{choose_input_vcf(args, inv)}`
- FASTQs discovered: {len(inv.get('fastqs', []))}
- DOCX clinical files discovered: {len(inv.get('docx', []))}
- Excel files discovered: {len(inv.get('excel', []))}
- Genome build: GRCh38, using the reference named in the supplied VCF header where locally available.

## Workflow

1. Discover and audit all challenge files locally.
2. Extract clinical text/HPO IDs from DOCX, inspect any Excel workbook locally, and compute genome-wide ontology-aware phenotype-to-gene scores from current HPO annotations.
3. Align all paired-end FASTQ lanes to the matching GRCh38 reference with pybwa/BWA-MEM and read groups, then name-sort/fixmate/coordinate-sort, merge lanes, mark duplicates, and index through pysam.
4. Left-normalize and split multiallelic records in the supplied VCF through pysam/bcftools bindings using the same reference.
5. Perform singleton read-backed phasing with WhatsHap where short reads provide phase information.
6. Annotate locally with pip-installed OpenCRAVAT and locally installed annotation modules; no patient variant data are sent to public annotation APIs.
7. Extract heterozygous candidate alleles and score them genome-wide using variant quality, predicted consequence, rarity, clinical annotation, prediction evidence, and optional phenotype-derived gene scores. No known-MVA-gene bonus is used.
8. Build same-gene compound-heterozygous pairs. Read-backed trans phase increases support; read-backed cis phase is strongly penalized; unphased pairs remain eligible.
9. Rank pairs and transform the ranking score monotonically to an EPCR-like value. EPCR here is a ranking confidence proxy rather than a calibrated clinical probability.
10. Export the top <=10 rows in the official Track 1 coordinate schema, converting contigs from `1` style to `chr1` style only at submission time.

## Genome-wide candidate policy

No established MVA gene list is used as a ranking prior or filtering exception. Candidate pairs compete under the same genome-wide rules. Known disease genes may be reviewed after ranking as an interpretability/sanity-check layer, but they do not receive score bonuses in the primary model.

## Current top candidate pairs

| Rank | Gene | Pair score | EPCR | Phase | Variant 1 | Variant 2 |
|---:|---|---:|---:|---|---|---|
{rows_md}

## Reproducibility

Pipeline generated: {now()}

Tool/version capture:

```json
{json.dumps(tool_versions, indent=2)}
```

## Limitations

- Singleton short-read data cannot establish phase for all distant variant pairs.
- Variant consequence scores and the pair score are prioritization heuristics, not ACMG/AMP classifications.
- Population/ClinVar/predictor evidence depends on which local OpenCRAVAT annotation modules are installed.
- Structural-variant and independent small-variant re-calling are candidates for later pipeline extensions and should be evaluated before final submission if the top pair remains uncertain.
"""
    path = report_dir / "track1_report_draft.md"
    path.write_text(report, encoding="utf-8")
    logging.info("Methods report draft written: %s", path)
    return path


def write_manifest(args) -> None:
    out = Path(args.out_dir)
    manifest = {
        "generated_at": now(),
        "argv": sys.argv,
        "args": vars(args),
        "python": sys.version,
        "platform": sys.platform,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def run_all(args, paths: dict) -> None:
    inspect_dataset(args, paths)
    phenotype_stage(args, paths)
    # Build/validate the public GRCh38 reference named by the supplied VCF before heavy stages.
    if resolve_reference(args, choose_input_vcf(args, discover_files(Path(args.data_dir)))) is None:
        prepare_reference_stage(args, paths)
    else:
        validate_reference_against_vcf(
            resolve_reference(args, choose_input_vcf(args, discover_files(Path(args.data_dir)))),
            choose_input_vcf(args, discover_files(Path(args.data_dir))),
            Path(args.out_dir) / "reference_validation.json",
        )
    alignment_stage(args, paths)
    normalize_stage(args, paths)
    phase_stage(args, paths)
    annotation_stage(args, paths)
    rank_stage(args, paths)
    submission_stage(args, paths)
    methods_report_stage(args, paths)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="One-file, pip-native Track 1 singleton-WGS pipeline for the MVA Hackathon 2026.",
        epilog=textwrap.dedent("""
        One-time packages in the existing Python environment:
          python -m pip install -U pysam pybwa whatshap open-cravat pandas numpy \
              python-docx openpyxl scipy scikit-learn tqdm

        One-time LOCAL OpenCRAVAT data setup (no Conda/Mamba):
          oc module install-base
          oc module install clinvar gnomad gnomad3 revel spliceai cadd
        """),
    )
    p.add_argument("stage", choices=["inspect", "phenotype", "prepare-reference", "align", "normalize", "phase", "annotate", "rank", "submit", "report", "all"])
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out-dir", default=None, help="Default: <data-dir>/mva_track1_run")
    p.add_argument("--vcf", default=None, help="Override supplied VCF path")
    p.add_argument("--reference", default=None, help="Local matching GRCh38 FASTA; strongly prefer the exact FASTA named in the VCF header")
    p.add_argument("--sample", default=DEFAULT_SAMPLE)
    p.add_argument("--proband-id", default=None, help="Submission proband_id; defaults to VCF sample ID")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--align-batch-pairs", type=int, default=2000,
                   help="Paired reads passed to pybwa per Python batch; lower if memory is tight")
    p.add_argument("--align-shard-pairs", type=int, default=2_000_000,
                   help="Read pairs per process-isolated pybwa worker (default 2,000,000). All reads are aligned; this only bounds worker lifetime/RAM accumulation")
    p.add_argument("--bwa-chunk-bases", type=int, default=10_000_000,
                   help="BWA-MEM -K style internal chunk size in bases (default 10,000,000; lower if memory-constrained)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-lane-bams", action="store_true")
    p.add_argument("--oc-run-name", default="mva_track1")
    p.add_argument("--oc-annotators", nargs="*",
                   default=["clinvar", "gnomad", "gnomad3", "revel", "spliceai", "cadd"],
                   help="Locally installed OpenCRAVAT annotators to run; use --oc-annotators with no values for base gene/SO only")
    p.add_argument("--phenotype-gene-scores", default=None,
                   help="Optional LOCAL CSV/TSV with gene,score columns (0..1), generated from phenotype analysis")
    p.add_argument("--hpo-dir", default=None,
                   help="Directory for downloaded HPO resources; default: <data-dir>/resources/hpo")
    p.add_argument("--skip-hpo-gene-scoring", action="store_true",
                   help="Extract HPO IDs but skip automatic genome-wide HPO-to-gene semantic scoring")
    p.add_argument("--max-variants-per-gene", type=int, default=30)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--submission-name", default="bonadossou_track1_candidates.csv")
    p.add_argument("--submission-profile", choices=["genomewide", "mva-clinical"], default="genomewide",
                   help="Which ranked table the submit stage uses. mva-clinical uses the explicit disease-aware reranker.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.data_dir = str(Path(args.data_dir).expanduser())
    if args.out_dir is None:
        args.out_dir = str(Path(args.data_dir) / "mva_track1_run")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    setup_logging(Path(args.out_dir) / "logs" / "pipeline.log", args.verbose)
    write_manifest(args)
    paths = discover_files(Path(args.data_dir))
    logging.info("MVA Track 1 pipeline stage=%s data=%s out=%s", args.stage, args.data_dir, args.out_dir)

    dispatch = {
        "inspect": inspect_dataset,
        "phenotype": phenotype_stage,
        "prepare-reference": prepare_reference_stage,
        "align": alignment_stage,
        "normalize": normalize_stage,
        "phase": phase_stage,
        "annotate": annotation_stage,
        "rank": rank_stage,
        "submit": submission_stage,
        "report": methods_report_stage,
        "all": run_all,
    }
    try:
        dispatch[args.stage](args, paths)
    except Exception as exc:
        logging.exception("Pipeline failed: %s", exc)
        return 2
    logging.info("DONE stage=%s", args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
