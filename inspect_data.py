"""Inventory the MVA hackathon dataset: what is here, what build, singleton or trio.

Standard library only. Uses samtools/bcftools if they happen to be on PATH,
falls back to pure-Python parsing if not. Reads headers and samples, never
loads a whole file.

    python inspect_data.py [DATA_DIR]
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

DEFAULT_DIR = os.environ.get(
    "RARE_DISEASE_DATA_DIR",
    "/network/scratch/b/bonaventure.dossou/rare_disease_data",
)

VCF_EXT = (".vcf", ".vcf.gz", ".bcf")
FASTQ_EXT = (".fastq", ".fq", ".fastq.gz", ".fq.gz")
ALN_EXT = (".bam", ".cram", ".sam")
TEXTY_EXT = (".txt", ".tsv", ".csv", ".json", ".ped", ".fam", ".md", ".yaml", ".yml", ".hpo")

# Chromosome 1 has different lengths in GRCh38 and GRCh37, so its length identifies the build.
CHR1_LEN = {248956422: "GRCh38/hg38", 249250621: "GRCh37/hg19"}


def hr(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def run(cmd: list[str], timeout: int = 120) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def opener(path: Path):
    """Open plain or gzip/bgzf transparently, as text."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", errors="replace")
    return open(path, "r", errors="replace")


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------

def classify(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(VCF_EXT):
        return "vcf"
    if name.endswith(FASTQ_EXT):
        return "fastq"
    if name.endswith(ALN_EXT):
        return "alignment"
    if name.endswith((".tbi", ".csi", ".bai", ".crai", ".idx", ".fai")):
        return "index"
    if name.endswith(TEXTY_EXT):
        return "text"
    return "other"


def inventory(root: Path) -> dict:
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and ".cache" not in p.parts]
    groups: dict[str, list[Path]] = {}
    total = 0
    for p in files:
        kind = classify(p)
        groups.setdefault(kind, []).append(p)
        total += p.stat().st_size

    hr(f"INVENTORY  {root}")
    if not files:
        print("  No files found. Check the path.")
        return groups
    for kind in ("vcf", "alignment", "fastq", "text", "index", "other"):
        for p in groups.get(kind, []):
            print(f"  {human(p.stat().st_size):>10}  [{kind:9}] {p.relative_to(root)}")
    print(f"  {'-' * 60}")
    print(f"  {human(total):>10}  total across {len(files)} files")
    return groups


# --------------------------------------------------------------------------
# VCF
# --------------------------------------------------------------------------

def inspect_vcf(path: Path, root: Path, max_records: int = 200_000) -> dict:
    print(f"\n--- {path.relative_to(root)} ---")
    info: dict = {"path": str(path)}

    meta, samples = [], []
    try:
        with opener(path) as f:
            for line in f:
                if line.startswith("##"):
                    meta.append(line.rstrip())
                elif line.startswith("#CHROM"):
                    parts = line.rstrip().split("\t")
                    samples = parts[9:] if len(parts) > 9 else []
                    break
                else:
                    break
    except Exception as e:
        print(f"  could not read header: {type(e).__name__}: {e}")
        return info

    def first(prefix):
        return next((m.split("=", 1)[1] for m in meta if m.startswith(prefix)), None)

    print(f"  fileformat : {first('##fileformat=')}")
    print(f"  reference  : {first('##reference=')}")

    # Build fingerprint from contig length rather than trusting the label.
    build = None
    for m in meta:
        if m.startswith("##contig=") and ("ID=chr1," in m or "ID=1," in m):
            for field in m.strip("#<>").split(","):
                if field.startswith("length="):
                    build = CHR1_LEN.get(int(field.split("=")[1]), f"unknown (chr1={field})")
    print(f"  build      : {build or 'no contig lines in header'}")
    info["build"] = build

    chr_style = "chr-prefixed" if any("ID=chr" in m for m in meta if m.startswith("##contig=")) else "no chr prefix"
    print(f"  contig style: {chr_style}  <-- submission wants 'chr15' style")

    callers = [m.split("=", 1)[1] for m in meta
               if m.startswith(("##source=", "##GATKCommandLine", "##DeepVariant"))]
    if callers:
        print(f"  caller     : {'; '.join(c[:70] for c in callers[:3])}")

    print(f"  samples    : {len(samples)}  {samples}")
    info["samples"] = samples
    if len(samples) == 0:
        print("    sites-only VCF, no genotypes")
    elif len(samples) == 1:
        print("    SINGLETON. No parents, so compound-het phasing must come from")
        print("    read-backed phasing in the alignments, not inheritance.")
    else:
        print(f"    MULTI-SAMPLE ({len(samples)}). If this is a trio, phasing a")
        print("    compound het becomes straightforward via parental origin.")

    fmt = [m.split("ID=")[1].split(",")[0] for m in meta if m.startswith("##FORMAT=")]
    inf = [m.split("ID=")[1].split(",")[0] for m in meta if m.startswith("##INFO=")]
    print(f"  FORMAT keys: {fmt[:15]}{' ...' if len(fmt) > 15 else ''}")
    print(f"  INFO keys  : {inf[:15]}{' ...' if len(inf) > 15 else ''}")
    annotated = [k for k in ("CSQ", "ANN", "VEP", "gnomAD_AF", "AF_popmax", "CLNSIG") if k in inf]
    print(f"  annotation : {annotated if annotated else 'NONE; annotation is required.'}")
    info["annotated"] = annotated

    # Counts: prefer the index, else stream with a cap.
    n = None
    if have("bcftools") and (path.with_suffix(path.suffix + ".tbi").exists()
                             or path.with_suffix(path.suffix + ".csi").exists()):
        out = run(["bcftools", "index", "-n", str(path)])
        if out and out.strip().isdigit():
            n = int(out.strip())
            print(f"  variants   : {n:,} (from index)")

    if n is None:
        capped, chroms, phased, gts = 0, Counter(), 0, Counter()
        try:
            with opener(path) as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    capped += 1
                    parts = line.split("\t", 10)
                    chroms[parts[0]] += 1
                    if len(parts) > 9:
                        gt = parts[9].split(":")[0]
                        gts[gt] += 1
                        if "|" in gt:
                            phased += 1
                    if capped >= max_records:
                        break
        except Exception as e:
            print(f"  scan stopped: {type(e).__name__}: {e}")
        suffix = f" (first {max_records:,} scanned)" if capped >= max_records else ""
        print(f"  variants   : {capped:,}{suffix}")
        print(f"  contigs    : {len(chroms)} distinct, top: {chroms.most_common(5)}")
        if gts:
            print(f"  GT spread  : {gts.most_common(6)}")
            print(f"  phased GTs : {phased:,} of {capped:,}"
                  + ("  <-- already phased" if phased else "  <-- unphased"))
        info["n_variants"] = capped
    else:
        info["n_variants"] = n
    return info


# --------------------------------------------------------------------------
# alignments and fastq
# --------------------------------------------------------------------------

def inspect_alignment(path: Path, root: Path) -> None:
    print(f"\n--- {path.relative_to(root)} ---")
    if not have("samtools"):
        print("  samtools not on PATH. Load it, or `conda install -c bioconda samtools`.")
        return
    head = run(["samtools", "view", "-H", str(path)])
    if not head:
        print("  samtools could not read the header.")
        return
    lines = head.splitlines()
    sq = [l for l in lines if l.startswith("@SQ")]
    rg = [l for l in lines if l.startswith("@RG")]
    pg = [l for l in lines if l.startswith("@PG")]
    print(f"  reference contigs: {len(sq)}")
    for l in sq[:1]:
        print(f"    e.g. {l[:100]}")
        for field in l.split("\t"):
            if field.startswith("LN:"):
                print(f"    build guess: {CHR1_LEN.get(int(field[3:]), 'not chr1-first')}")
    print(f"  read groups: {len(rg)}")
    for l in rg[:4]:
        print(f"    {l[:120]}")
    for l in pg[:2]:
        print(f"    {l[:120]}")


def inspect_fastq(path: Path, root: Path, n_reads: int = 4) -> None:
    print(f"\n--- {path.relative_to(root)} ---")
    try:
        with opener(path) as f:
            lens = []
            for i, line in enumerate(f):
                if i >= n_reads * 4:
                    break
                if i == 0:
                    print(f"  first header: {line.strip()[:100]}")
                if i % 4 == 1:
                    lens.append(len(line.strip()))
        print(f"  read length: {lens}")
        size = path.stat().st_size
        if lens:
            approx = size / (lens[0] * 2.1)  # rough, compressed
            print(f"  approx reads: {approx/1e6:.0f}M (very rough from file size)")
    except Exception as e:
        print(f"  could not read: {type(e).__name__}: {e}")


def inspect_text(path: Path, root: Path, n_lines: int = 25) -> None:
    print(f"\n--- {path.relative_to(root)} ---")
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            print(json.dumps(data, indent=2)[:2000])
            return
        with opener(path) as f:
            for i, line in enumerate(f):
                if i >= n_lines:
                    print("  ...")
                    break
                print(f"  {line.rstrip()[:160]}")
    except Exception as e:
        print(f"  could not read: {type(e).__name__}: {e}")


# --------------------------------------------------------------------------

def main(root_str: str) -> None:
    root = Path(root_str).expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}")
        raise SystemExit(1)

    print("tools on PATH: " + ", ".join(
        f"{t}={'yes' if have(t) else 'NO'}"
        for t in ("samtools", "bcftools", "tabix", "bgzip")))

    groups = inventory(root)

    if groups.get("text"):
        hr("PHENOTYPE / METADATA FILES  (read these first)")
        for p in groups["text"]:
            inspect_text(p, root)

    if groups.get("vcf"):
        hr("VCF FILES")
        for p in groups["vcf"]:
            inspect_vcf(p, root)

    if groups.get("alignment"):
        hr("ALIGNMENTS")
        for p in groups["alignment"]:
            inspect_alignment(p, root)

    if groups.get("fastq"):
        hr("FASTQ")
        for p in groups["fastq"]:
            inspect_fastq(p, root)

    hr("DONE")
    print("Use this summary to select the appropriate pipeline configuration.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
