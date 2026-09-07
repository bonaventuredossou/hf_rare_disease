#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_dir"

python capture_environment.py
python report_visualizations.py --output-dir report/figures
python -m pytest \
  test_sbnn_pair_ranker.py \
  test_bayesian_integration.py \
  test_audit_submission.py \
  test_export_track1_sbnn_submission.py \
  test_model_registry.py \
  test_clinvar_temporal_experiment.py -q
ranking_zip="${RANKING_ZIP:-../ranking.zip}"
if [[ -f "$ranking_zip" ]]; then
  python generate_track1_submission.py \
    --ranking-zip "$ranking_zip" \
    --checkpoint results/track1_temporal_transfer_20260906/trained_models.pkl
else
  echo "No ranking.zip found; using the committed audited SBNN prediction."
  python generate_track1_submission.py
fi
python verify_track1_exact_scores.py

cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex

cd "$repo_dir"
mkdir -p output/pdf
cp report/main.pdf output/pdf/rare_disease_real_kid_methods_report.pdf

echo "Built output/pdf/rare_disease_real_kid_methods_report.pdf and regenerated all verification artifacts."
