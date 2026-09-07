# Data directory

No patient data or large downloaded benchmark files are committed to this repository.

The private hackathon bundle is hosted by SageBio at <https://huggingface.co/datasets/SageBio/mva-hackathon-2026-data>. Access is gated by the challenge rules. The approximately 85 GB bundle was processed on cluster storage and must not be added to GitHub.

The public evaluation inputs are downloaded from their original sources when reproducing the experiments:

- ClinVar GRCh38 VCF archives: <https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/>
- Phenopacket Store release 0.1.27: <https://github.com/monarch-initiative/phenopacket-store/releases/tag/0.1.27>
- HPO release files: <https://github.com/obophenotype/human-phenotype-ontology/releases>

Expected paths and SHA-256 checksums are documented in the main README and the report. The `data/open/` downloads are ignored by Git.
