import unittest

import numpy as np

from clinvar_temporal_experiment import BinaryBBB, BinarySBNN, clean_label, ece, fmax, ranking_metrics
from tune_sbnn_temporal import make_shortlist


class ClinVarTemporalExperimentTests(unittest.TestCase):
    def test_unambiguous_labels_only(self):
        self.assertEqual(clean_label("Pathogenic|Likely_pathogenic"), 1)
        self.assertEqual(clean_label("Benign/Likely_benign"), 0)
        self.assertIsNone(clean_label("Uncertain_significance"))
        self.assertIsNone(clean_label("Pathogenic|Benign"))

    def test_fmax_perfect(self):
        score, _ = fmax(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]))
        self.assertAlmostEqual(score, 1.0)

    def test_ranking_perfect(self):
        y = np.array([1, 1, 0, 0, 0])
        p = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        result = ranking_metrics(y, p, seed=1, negatives_per_case=3)
        self.assertEqual(result["top1"], 1.0)

    def test_ece_perfect_two_bins(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(ece(y, p, bins=2), 0.0)

    def test_sbnn_hyperparameters_reach_every_layer(self):
        model = BinarySBNN(
            12,
            ranks=(6, 3, 1),
            seed=7,
            posterior_scale_multiplier=0.5,
            prior_sigma1=2.0,
        )
        self.assertEqual([layer.rank for layer in model.layers], [6, 3, 1])
        self.assertTrue(all(layer.prior_sigma1 == 2.0 for layer in model.layers))
        expected = 0.5 * 0.09 * ((2.0 / 12) / 6) ** 0.25
        self.assertAlmostEqual(float(np.logaddexp(model.layers[0].A.rho[0, 0], 0)), expected)

    def test_shortlist_keeps_uncertainty_specialist(self):
        def item(name, score, nll, ood, pr=0.90, fmax_value=0.90):
            return {
                "config": {"name": name},
                "metrics": {"score": score, "nll": nll, "uncertainty_unseen_gene_auroc": ood, "pr_auc": pr, "f_max": fmax_value},
            }
        baseline = item("baseline", 0.30, 0.13, 0.60)
        results = [baseline, item("calibrated", 0.25, 0.10, 0.61), item("uncertain", 0.28, 0.12, 0.75)]
        names = {value["config"]["name"] for value in make_shortlist(results, baseline["metrics"])}
        self.assertEqual(names, {"baseline", "calibrated", "uncertain"})

    def test_predictive_standard_deviation_is_returned(self):
        model = BinarySBNN(3, hidden=(4, 2), ranks=(2, 2, 1), seed=3)
        mean, mutual_information, standard_deviation = model.predict_statistics(
            np.zeros((5, 3), dtype=np.float32), draws=6, seed=9
        )
        self.assertEqual(mean.shape, (5,))
        self.assertEqual(mutual_information.shape, (5,))
        self.assertEqual(standard_deviation.shape, (5,))
        self.assertTrue(np.all(standard_deviation >= 0))

    def test_dense_bbb_parameter_count_and_sampling(self):
        model = BinaryBBB(50, hidden=(32, 16), seed=4, rho_init=-5.0)
        self.assertEqual(model.parameter_count(), 4354)
        mean, mutual_information, standard_deviation = model.predict_statistics(
            np.zeros((7, 50), dtype=np.float32), draws=5, seed=6
        )
        self.assertEqual(mean.shape, (7,))
        self.assertTrue(np.all(mutual_information >= 0))
        self.assertTrue(np.all(standard_deviation >= 0))

    def test_dense_bbb_gradient_is_finite(self):
        model = BinaryBBB(3, hidden=(4, 2), seed=8)
        x = np.random.default_rng(9).normal(size=(6, 3))
        y = np.array([0, 1, 0, 1, 1, 0])
        loss, gradients = model.loss_and_gradients(
            x, y, np.random.default_rng(10), kl_scale=1e-4
        )
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.all(np.isfinite(value)) for value in gradients.values()))

    def test_wide_model_parameter_counts(self):
        bbb = BinaryBBB(50, hidden=(200, 100), seed=1)
        low_rank = BinarySBNN(50, hidden=(200, 100), ranks=(8, 8, 1), seed=1)
        full_factor = BinarySBNN(50, hidden=(200, 100), ranks=(50, 100, 1), seed=1)
        self.assertEqual(bbb.parameter_count(), 60802)
        self.assertEqual(sum(value.size for _, value in low_rank.parameters()), 9604)
        self.assertEqual(sum(value.size for _, value in full_factor.parameters()), 85804)


if __name__ == "__main__":
    unittest.main()
