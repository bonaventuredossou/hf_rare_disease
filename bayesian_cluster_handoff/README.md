# Bayesian pair ranker

I ran this version on your ranking tables. It puts the same BUB1B pair first.
The external test results are mixed, so I'm not claiming it improves on your
pipeline yet.

Start with [BAYESIAN_HANDOFF.md](BAYESIAN_HANDOFF.md). It has the command to run
on the cluster, the expected input paths, the results so far and what to send back.

The package includes the prediction code, fitted model, training report and
tests. It needs Python 3.10 or newer, NumPy and SciPy. Install the dependencies
with `python -m pip install -r requirements-bayesian.txt`.

It reads the two existing candidate-pair TSVs, not the 85 GB sequencing files.
It doesn't download data or submit anything to Hugging Face.

`SHA256SUMS.json` records the hash of each included file. Codex assisted with
development and testing; the model's limitations are in the handoff.
