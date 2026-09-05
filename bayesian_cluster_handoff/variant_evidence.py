"""Consequence and frequency scores copied from the original Track 1 pipeline.

Keeping these two helpers here lets the prediction package run without the
alignment and annotation code. The score values are unchanged.
"""

import re
from typing import Optional


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
    # OpenCRAVAT consequence codes and alternate labels.
    "MIS": 0.72,
    "missense": 0.72,
    "SYN": 0.18,
    "synonymous": 0.18,
    "synonomous": 0.18,
    "STG": 1.00,
    "stopgain": 1.00,
    "STL": 0.95,
    "stoploss": 0.95,
    "SPL": 1.00,
    "splice site affected": 1.00,
    "FD1": 1.00,
    "FD2": 1.00,
    "FI1": 1.00,
    "FI2": 1.00,
    "frameshift": 1.00,
    "IDV": 0.78,
    "inframe deletion": 0.78,
    "IIV": 0.78,
    "inframe insertion": 0.78,
    "CSS": 0.72,
    "complex substitution": 0.72,
    "INT": 0.06,
    "intron": 0.06,
    "UT5": 0.14,
    "5_prime_UTR": 0.14,
    "UT3": 0.12,
    "3_prime_UTR": 0.12,
    "2KU": 0.03,
    "2KD": 0.03,
    "UNK": 0.01,
}


def consequence_score(cons: str) -> float:
    terms = re.split(r"[&,]", cons or "")
    return max([CONSEQUENCE_SCORE.get(t, 0.02) for t in terms] or [0.02])


def rarity_score(af: Optional[float]) -> float:
    if af is None:
        return 0.70  # Missing frequency is not the same as observed absence.
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
