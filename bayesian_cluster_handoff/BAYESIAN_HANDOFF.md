# Bayesian ranker: results and how to run it

I ran the Bayesian ranker on the tables you sent. It puts the same BUB1B pair
first: `15:40209701 T>G` and `15:40220612 T>G`. I haven't submitted its CSV to
Hugging Face. Your submission already has the maximum score, so this doesn't
improve on that result. What I've added is a model that learns the evidence
weights from external solved cases, plus a way to see how the ranking changes
when we sample those weights.

The external results are mixed. It finds the right pair in this case, but it
doesn't yet beat the fixed-weight baseline on external top-1 accuracy.

## Running it on your side

Unzip `bayesian_cluster_handoff.zip` into a separate directory, then run:

```bash
python -m pip install -r requirements-bayesian.txt
python run_bayesian_track1.py \
  --data-dir /network/scratch/b/bonaventure.dossou/rare_disease_data \
  --model models/shared_pair_model.json \
  --reference-submission /home/mila/b/bonaventure.dossou/hf_rare_disease/bonaventuredossou_genomewide-hpo-comphet.csv
```

If your submitted CSV is somewhere else, change the last path or leave that
flag out. It's only used to compare rankings after prediction. It doesn't
affect the model.

The script reads these two files from your existing run:

```text
<data-dir>/mva_track1_run/ranking/compound_het_pairs.tsv
<data-dir>/mva_track1_run/ranking/compound_het_pairs_primary_lenient_audit.tsv
```

Use `--run-dir /path/to/mva_track1_run` if that folder lives elsewhere. There's
no alignment or annotation step to rerun. The script doesn't open the FASTQs
or VCFs, and the ZIP contains no patient input tables.

Results go into `<data-dir>/mva_track1_run/bayesian/`, or
`<run-dir>/bayesian/` if you set `--run-dir`:

- `strict_comparison.tsv`: the six eligible strict pairs, with the original
  scores, submitted ranks when available, Bayesian ranks and uncertainty estimates.
- `broad_comparison.tsv`: the 26 eligible pairs from the lenient table.
- `strict_bayesian_candidates.csv`: the strict results in the challenge CSV format.
- `broad_bayesian_candidates.csv`: the top ten lenient candidates, for inspection.
- `run_summary.json`: model hash, seed and comparison with the reference pair.

Send me `run_summary.json` and `strict_comparison.tsv` once it finishes. The
script doesn't upload anything. You can also run the included tests with:

```bash
python -m unittest -v test_bayesian_integration.py
```

## What I trained

I used solved cases from Phenopacket Store 0.1.27, with the VEP annotations,
ClinVar-derived benign variants and HPO files cached in our project. After
dropping cases without enough annotations, there were 425 training cases,
44 validation cases and 89 test cases. The splits don't share genes or
publication/source groups. BUB1B was excluded from training candidates and
labels; I didn't use the challenge answer to fit or select the model.

For each case, I built 15 pairs from six variants in the causal gene: the two
known causal variants and four benign decoys. I added eight benign pairs from
other genes with similar phenotype annotations. That gives 23 candidate pairs
per case. These are constructed candidate sets, not full patient genomes.

The model is a linear pair ranker, not a neural network. It learns 15 weights
for phenotype match, predicted consequence, population frequency, CADD,
SpliceAI, missing-value indicators and two phenotype interactions. Both alleles
contribute, including the weaker of the two. Missing frequency is kept distinct
from a measured frequency of zero.

The Bayesian part is a Gaussian prior on the weights and a Laplace approximation
around the fitted solution. Validation negative log-likelihood selected a prior
standard deviation of 0.25 from `[0.25, 1, 2, 8]`. At prediction time, I average
the candidate probabilities over 2,048 weight samples. The ZIP includes the
fitted weights, covariance and training normalization, so you don't need the
training data to run it.

## Results so far

| External test, 89 cases | Fixed evidence rule | MAP weights | Bayesian mean |
|---|---:|---:|---:|
| Correct pair ranked first | 70.8% | 55.1% | 58.4% |
| Correct pair in the top five | 83.1% | 87.6% | 87.6% |
| Mean reciprocal rank | 0.765 | 0.698 | 0.717 |

The fixed rule here is a small baseline using the evidence available in the
training data, not your full pipeline. The Bayesian model improves on the MAP
version, but the fixed rule still does better at putting the answer first.
So i need to understand that gap before calling this an improvement.

On your tables, BUB1B ranks first among both the six strict candidates and the
26 lenient candidates. Read-backed cis pairs are excluded; unknown phase stays
eligible. The BUB1B probability is about 0.9998 within each shortlist. That is
not a calibrated probability that the pair causes the disease. It doesn't
account for a causal variant missing from the shortlist, annotation errors or
uncertain phase.

A few limits matter for the next experiments. Many solved cases were excluded
for missing annotations, and this test split was already used in earlier
development runs. We still need a fresh evaluation on realistic genome-wide
candidate sets. Training uses cached VEP annotations while your tables use
OpenCRAVAT; the resources can differ. The phenotype formula matched your
implementation across 5,268 genes for the query I checked, but that doesn't
remove possible differences between resource versions.

## Notes on the files

The strict table in `ranking.zip` is newer than the submitted CSV, so some
alternative scores differ. I kept `baseline_rank` for the table ranking and
`submitted_rank` for the CSV ranking. The BUB1B pair is the same in both.

This version also fixes the earlier posterior softmax axis and a mislabeled
standard-deviation output. Don't use the older synthetic uncertainty numbers.
The earlier 59.6% versus 70.8% comparison was between two Bayesian versions,
not a comparison with your pipeline.

The repository keeps the training script, full change history for the recorded
sessions and your original Track 1 report. This ZIP is just the prediction code,
tests and fitted model. `SHA256SUMS.json` lists the hash of every included file.
