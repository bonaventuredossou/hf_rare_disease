# Track 1 submission


**[`track1_wide_rank50_sbnn.csv`](track1_wide_rank50_sbnn.csv)**

It is the canonical output of the selected wide rank-50 SBNN, not a renamed deterministic result. The first row is the accepted compound-heterozygous `BUB1B` pair. The file has six rows and the required columns:

```text
proband_id, chrom_1, pos_1, ref_1, alt_1,
chrom_2, pos_2, ref_2, alt_2, epcr, finding_type, notes
```


The companion [`track1_wide_rank50_sbnn.manifest.json`](track1_wide_rank50_sbnn.manifest.json) records the model configuration, training provenance, input prediction hash, output hash, and score audit. Regenerate both files with:

```bash
python generate_track1_submission.py
```

The exporter accepts only the audited `strict_wide_rank50_sbnn.csv` prediction produced by `verify_temporal_models_track1.py`. It fails instead of publishing a CSV when the schema or exact score check does not pass.

The complete generation chain is:

```text
generate_track1_submission.py
    -> models/wide_rank50_sbnn.py
    -> verify_temporal_models_track1.py
    -> results/track1_temporal_transfer_20260906/strict_wide_rank50_sbnn.csv
    -> export_track1_sbnn_submission.py
    -> submission/track1_wide_rank50_sbnn.csv
```

`generate_track1_submission.py` is the public entry point. `models/wide_rank50_sbnn.py` contains the selected model configuration. `verify_temporal_models_track1.py` performs posterior inference and pair scoring. `export_track1_sbnn_submission.py` performs final validation and writes the upload file.
