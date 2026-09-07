import unittest
import numpy as np
from bayesian_pair_ranker import Posterior
from shared_pair_features import pair_features, table_features, FEATURE_NAMES
from variant_evidence import consequence_score, rarity_score


class BayesianIntegrationTests(unittest.TestCase):
    def test_consequence_labels_and_combinations(self):
        for label in ("stop_gained", "STG", "MIS&STG", "intron_variant,stop_gained"):
            self.assertEqual(consequence_score(label), 1.0)
        self.assertEqual(
            consequence_score("MIS"), consequence_score("missense_variant")
        )
        self.assertEqual(consequence_score("unknown_term"), 0.02)
        self.assertEqual(consequence_score(""), 0.02)

    def test_frequency_score_boundaries(self):
        frequencies = [None, 0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 2e-2, 0.1]
        expected = [0.70, 1.0, 0.98, 0.95, 0.85, 0.65, 0.45, 0.25, 0.03]
        self.assertEqual([rarity_score(af) for af in frequencies], expected)

    def test_probability_normalization_and_case_independence(self):
        model = Posterior(np.array([0.7, -0.4]), np.eye(2) * 0.2, 1, True, 0)
        x = np.random.default_rng(1).normal(size=(3, 7, 2))
        p = model.posterior_predictive(x, draws=100, seed=7)
        np.testing.assert_allclose(p["probability_draws"].sum(axis=-1), 1, atol=1e-12)
        separate = model.posterior_predictive(x[:1], draws=100, seed=7)
        np.testing.assert_allclose(
            p["mean_probabilities"][:1], separate["mean_probabilities"]
        )
        np.testing.assert_allclose(
            p["mutual_information"],
            p["predictive_entropy"] - p["expected_entropy"],
            atol=1e-12,
        )

    def test_unidentifiable_feature_retains_prior_variance(self):
        x = np.zeros((6, 3, 2))
        x[:, :, 0] = np.array([0, 1, 2])
        m = Posterior.fit(x, np.full(6, 2), prior_scale=0.5)
        self.assertTrue(m.converged)
        self.assertAlmostEqual(m.covariance[1, 1], 0.25)

    def test_shared_feature_parity_missingness_and_swap(self):
        aa = [
            dict(consequence="stop_gained", af=None, cadd=38, spliceai=0.03),
            dict(consequence="missense_variant", af=0.00001, cadd=None, spliceai=None),
        ]
        row = {"phenotype_score": ".6"}
        for i, a in enumerate(aa, 1):
            for field, key in [
                ("consequence", "consequence"),
                ("af", "af"),
                ("cadd_phred", "cadd"),
                ("spliceai", "spliceai"),
            ]:
                row[f"{field}_{i}"] = "" if a[key] is None else str(a[key])
        row["consequence_1"] = "STG"
        row["consequence_2"] = "MIS"
        v = pair_features(aa, 0.6)
        np.testing.assert_allclose(v, table_features(row))
        np.testing.assert_allclose(v, pair_features(aa[::-1], 0.6))
        self.assertEqual(v[FEATURE_NAMES.index("af_observed_fraction")], 0.5)
        self.assertEqual(v[FEATURE_NAMES.index("cadd_observed_fraction")], 0.5)
        aa[0]["af"] = 0
        self.assertNotEqual(
            v[FEATURE_NAMES.index("af_observed_fraction")],
            pair_features(aa, 0.6)[FEATURE_NAMES.index("af_observed_fraction")],
        )

    def test_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            Posterior.fit(np.ones((2, 3, 1)), np.array([-0.1, 1.0]))
        with self.assertRaises(ValueError):
            Posterior.fit(np.full((2, 3, 1), np.nan), np.array([0, 1]))
        with self.assertRaises(ValueError):
            table_features({"phenotype_score": ".5"})


if __name__ == "__main__":
    unittest.main()
