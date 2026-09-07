# Citation audit

Audit date: 7 September 2026

Every cited work in `main.tex` was checked against a publisher page, DOI record, official proceedings page, or the cited data/repository page. A DOI is used for scholarly papers wherever one exists. Repository and dataset records are explicitly typed as web resources rather than papers. `and others` is retained only for long author lists after the first author was verified.

## Scholarly references

| Key | Authoritative record | Claim checked |
|---|---|---|
| `stenton2024` | [DOI](https://doi.org/10.1186/s40246-024-00604-w) | Rare Genomes Project benchmark and top-five performance |
| `maverick2023` | [DOI](https://doi.org/10.1038/s41467-023-39306-7) | Learned pathogenicity plus inheritance-aware ranking |
| `aimarrvel2024` | [DOI](https://doi.org/10.1056/AIoa2300009) | Random forest trained on more than 3.5 million variants and expert-engineered evidence |
| `aidiva2026` | [DOI](https://doi.org/10.1038/s41525-026-00611-x) | Hybrid evidence, machine-learning, and language-model workflow |
| `tavtigian2018` | [DOI](https://doi.org/10.1038/gim.2017.210) | Bayesian formulation of ACMG/AMP evidence |
| `pheval2025` | [DOI](https://doi.org/10.1186/s12859-025-06105-4) | Standardized phenotype-driven prioritization benchmarking |
| `phenopacketstore2025` | [DOI](https://doi.org/10.1016/j.xhgg.2024.100371) | Composition and intended uses of the Phenopacket Store corpus |
| `hpo2021` | [DOI](https://doi.org/10.1093/nar/gkaa1043) | HPO as a standardized phenotype vocabulary |
| `gnomad2020` | [DOI](https://doi.org/10.1038/s41586-020-2308-7) | Population variation and constraint resource |
| `cadd2014` | [DOI](https://doi.org/10.1038/ng.2892) | CADD variant pathogenicity framework |
| `revel2016` | [DOI](https://doi.org/10.1016/j.ajhg.2016.08.016) | REVEL missense pathogenicity ensemble |
| `spliceai2019` | [DOI](https://doi.org/10.1016/j.cell.2018.12.015) | Deep-learning prediction of splice effects |
| `lirical2020` | [DOI](https://doi.org/10.1016/j.ajhg.2020.06.021) | Phenotype likelihood ratios combined with genotype evidence |
| `exomiser2015` | [DOI](https://doi.org/10.1038/nprot.2015.124) | Frequency, pathogenicity, pedigree, and phenotype-aware prioritization |
| `lakshminarayanan2017` | [Official NeurIPS record](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html) | Independently trained deep ensembles for predictive uncertainty |
| `blundell2015` | [Official PMLR record](https://proceedings.mlr.press/v37/blundell15.html) | Bayes by Backprop with variational weight distributions |
| `toure2026` | [Official ICML 2026 accepted-paper listing](https://icml.cc/Downloads/2026), [preprint](https://arxiv.org/abs/2602.00387) | Low-rank singular Bayesian neural networks |
| `bub1b2004` | [DOI](https://doi.org/10.1038/ng1449) | Biallelic `BUB1B` variants as a cause of MVA |
| `bub1bmechanism2010` | [DOI](https://doi.org/10.1158/0008-5472.CAN-09-4319) | BUBR1 dysfunction, checkpoint defects, and chromosome-segregation errors in MVA cells |
| `sirt2bubr12014` | [DOI](https://doi.org/10.15252/embj.201386907) | SIRT2/BubR1 stabilization and the reported NMN mouse experiment |
| `bubr1pp2a2020` | [DOI](https://doi.org/10.1016/j.celrep.2020.108397) | BUBR1 pseudokinase domain, PP2A-B56 recruitment, Aurora B balance, and chromosome alignment |

## Data, software, and challenge records

| Key | Authoritative record | Purpose |
|---|---|---|
| `hackathon2026` | [Official challenge Space](https://huggingface.co/spaces/SageBio/rare-disease-real-kid-mva-hackathon-2026) | Rules, Track 1 metrics, Track 2 criteria, and timeline |
| `hackathondata2026` | [Official dataset repository](https://huggingface.co/datasets/SageBio/mva-hackathon-2026-data) | Data size, access conditions, and file description |
| `dossourepo2026` | [Bonaventure Dossou's repository](https://github.com/bonaventuredossou/hf_rare_disease) | Provenance of the original genome-wide pipeline and exported features |
| `tourerepo2026` | [Mame Diarra Toure's repository](https://github.com/arradiat/Rare-Disease--Real-Kid) | Bayesian modeling, comparative evaluation, external generalization experiments, figures, and report |
| `phenopacketrelease2026` | [Release 0.1.27](https://github.com/monarch-initiative/phenopacket-store/releases/tag/0.1.27) | Exact Phenopacket Store version used |
| `clinvar` | [NCBI ClinVar downloads](https://www.ncbi.nlm.nih.gov/clinvar/docs/downloads/) | Current and archived GRCh38 VCF releases |

## Material corrections made

- `toure2026` is an `@inproceedings` record with the full authors Mame Diarra Toure and David A. Stephens and the 43rd ICML. No PMLR volume or page range is invented while final proceedings metadata remains unavailable.
- MAVERICK's first author was corrected from Max Schubach to Matt C. Danzi.
- AI-MARRVEL's title, journal, author list, article identifier, and DOI were corrected. The unsupported claim that it used exactly 103 features was removed.
- aiDIVA's first author and full title were corrected.
- PhEval's first author, full title, journal details, and DOI were corrected.
- The repository-only Exomiser record was replaced by the peer-reviewed Nature Protocols paper.
- The previous `bub1bmechanism2010` record was replaced by the Cancer Research paper that directly supports the report's MVA mechanism statement.
- The previous `aurorab2013` record was replaced by the 2020 Cell Reports study, and the report sentence was narrowed to the mechanism that study demonstrates.
- Missing volume, issue, page or article number, and DOI fields were added from authoritative metadata.
