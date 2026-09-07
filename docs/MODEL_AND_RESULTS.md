# Model and results

## Clinical question

Track 1 asks for the variant or variant pair that most plausibly explains the child's rare disease. The accepted answer is a pair of heterozygous variants in `BUB1B`, one at `chr15:40209701 T>G` and the other at `chr15:40220612 T>G`. A compound-heterozygous explanation requires two damaging alleles in the same recessive disease gene, ideally inherited on opposite parental chromosomes.

## From the genome to model inputs

Bonaventure Dossou's upstream pipeline converts the raw genome into a small, biologically defensible candidate table. It aligns reads, checks sequencing quality, normalizes and annotates variants, filters common or low-impact alleles, evaluates inheritance and phase, calculates HPO phenotype similarity, and constructs candidate pairs within genes. The source is [bonaventuredossou/hf_rare_disease](https://github.com/bonaventuredossou/hf_rare_disease).

The learned models do not ingest FASTQ reads. They receive variants that survived the upstream workflow. The temporal rankers use 50 allele-level features derived from sequence context, variant type, molecular consequence, population allele frequency, missingness, and gene evidence available in the training snapshot. The challenge transfer then combines the two allele probabilities with the HPO phenotype score:

```text
pair score = sqrt(P(pathogenic allele 1) * P(pathogenic allele 2))
             * phenotype similarity
```

The geometric mean requires support for both alleles without allowing one high probability to hide one weak allele. Phenotype similarity then connects molecular plausibility to the child's observed clinical features.

## Training target

The SBNN was trained on ClinVar's 2023 labels. The target is binary allele pathogenicity: pathogenic or likely pathogenic versus benign or likely benign. Exact Variation IDs present in the 2023 snapshot were excluded from the later temporal test set. Gene-disjoint validation was also used to expose failure on genes absent from training.

This is not a supervised dataset of complete child genomes paired with diagnoses. It is allele-level supervision transferred into a patient-level pair-ranking problem. The external Phenopacket experiments partially close that gap by testing candidate ranking in solved patients, but they still use constructed candidate sets rather than complete exomes.

## SBNN posterior

The selected model is a two-hidden-layer SBNN with widths 200 and 100 and posterior ranks 50, 50, and 1. Each low-rank Bayesian layer represents correlated uncertainty in its weights. Training minimizes predictive loss plus a KL penalty that keeps the learned posterior close to its prior. The chosen configuration uses KL weight 0.01, posterior scale multiplier 0.1, prior scale 0.3, and seed 20260905.

At inference, 128 weight samples are drawn from the fitted posterior. Their mean produces each allele probability. Their dispersion supports uncertainty analysis, although the five-network deep ensemble remained a better detector of variants from unseen genes.

## Model selection

Rank, KL weight, posterior initialization scale, and prior scale were tuned jointly. This matters because their effects are coupled: a useful rank under one KL strength can collapse under another. The sweep screened 375 combinations, used successive-halving confirmation, retained task and uncertainty extremes, and evaluated selected settings on an untouched temporal holdout.

The wide rank-50 SBNN was selected for the final Track 1 CSV because it had the best temporal top-1 ranking and PR-AUC among the learned models tested. The ensemble is retained as the preferred OOD warning signal rather than being presented as an inferior model.

## Exact child result

The selected SBNN ranks the accepted `BUB1B` pair first among six strict candidates. The generated CSV obtains 100 Rank Points and F-max 1.000 under the repository's implementation of the public Track 1 evaluator. The score has been verified locally. The earlier deterministic CSV received the same score on the live leaderboard; the SBNN CSV has not yet been uploaded.

## External comparison

| Method | Top-1 | PR-AUC | F-max | OOD AUROC |
|---|---:|---:|---:|---:|
| Gradient boosting | 81.50% | 0.9674 | 0.9030 | 0.7441 |
| Five-network ensemble | 80.25% | 0.9688 | 0.9095 | **0.7485** |
| Original small SBNN | 82.10% | 0.9691 | 0.9173 | 0.6020 |
| Temporal consensus | 83.65% | 0.9731 | 0.9181 | not evaluated |
| Dense BBB | 83.00% | 0.9729 | 0.9202 | 0.6665 |
| Wide rank-50 SBNN | **84.35%** | **0.9736** | **0.9235** | 0.6284 |
| Wide dense BBB | 81.35% | 0.9718 | 0.9200 | 0.6567 |

These metrics come from the executed JSON artifacts under `results/clinvar_temporal_20260905/`, `results/full_bbb_temporal_20260906/`, and `results/wide_rank_temporal_20260906/`. The LaTeX report defines every metric and plots ranking, calibration, OOD detection, hyperparameter interactions, and patient-level generalization.

## Interpretation

The Track 1 result shows that the Bayesian ranker preserves the correct diagnosis produced by the full biological pipeline. The temporal results provide stronger evidence that the ranker learned transferable pathogenicity patterns rather than only memorizing the child. The Phenopacket results show that HPO conditioning improves patient-level ranking, including two-variant cases.

The strongest defensible conclusion is narrower than full clinical generalization. The current evidence supports candidate ranking after successful variant discovery and filtering. It does not measure the probability that a complete end-to-end workflow will recover the causal allele from every new raw genome, and it does not make the model a clinical diagnostic device.

## Canonical artifacts

- Upload CSV: [`../submission/track1_wide_rank50_sbnn.csv`](../submission/track1_wide_rank50_sbnn.csv)
- Submission manifest: [`../submission/track1_wide_rank50_sbnn.manifest.json`](../submission/track1_wide_rank50_sbnn.manifest.json)
- Exact score audit: [`../results/exact_track1_score_audit.json`](../results/exact_track1_score_audit.json)
- Model comparison results: [`../results/wide_rank_temporal_20260906/wide_rank_results.json`](../results/wide_rank_temporal_20260906/wide_rank_results.json)
- Full methods report: [`../output/pdf/rare_disease_real_kid_methods_report.pdf`](../output/pdf/rare_disease_real_kid_methods_report.pdf)

## Implementation map

The seven methods in the main comparison and the auxiliary dense MLP baseline all have separate implementations:

| Method | Source module |
|---|---|
| Gradient boosting | `models/gradient_boost.py` |
| Dense MLP | `models/dense_mlp.py` |
| Five-network deep ensemble | `models/deep_ensemble.py` |
| Original small SBNN | `models/small_sbnn.py` |
| Dense/full-rank BBB | `models/full_rank_bbb.py` |
| Wide rank-50 SBNN | `models/wide_rank50_sbnn.py` |
| Wide full-weight BBB | `models/wide_dense_bbb.py` |
| Temporal consensus | `models/temporal_consensus.py` |

The exact Track 1 generation chain is `generate_track1_submission.py` to `models/wide_rank50_sbnn.py` to `verify_temporal_models_track1.py` to `export_track1_sbnn_submission.py` to `submission/track1_wide_rank50_sbnn.csv`.
