import unittest

import numpy as np

from sbnn_pair_ranker import SingularBayesianRanker


class SingularBayesianRankerTests(unittest.TestCase):
    def test_svd_initialization_recovers_full_rank_matrix(self):
        model = SingularBayesianRanker(3, hidden=(4,), ranks=(3, 1), seed=1)
        rng = np.random.default_rng(2)
        weights = [rng.normal(size=(3, 4)), rng.normal(size=(4, 1))]
        biases = [rng.normal(size=4), rng.normal(size=1)]
        model.initialize_from_dense(weights, biases)
        for layer, weight, bias in zip(model.layers, weights, biases):
            np.testing.assert_allclose(layer.A.mu @ layer.B.mu.T, weight, atol=1e-12)
            np.testing.assert_allclose(layer.bias.mu, bias)

    def test_sampled_probabilities_normalize_per_case(self):
        model = SingularBayesianRanker(5, hidden=(6,), ranks=(3, 1), seed=3)
        x = np.random.default_rng(4).normal(size=(7, 11, 5))
        result = model.predict_samples(x, draws=20, seed=5)
        np.testing.assert_allclose(result["samples"].sum(axis=-1), 1.0, atol=1e-12)
        self.assertTrue(np.all(result["mutual_information"] >= 0))

    def test_state_roundtrip_preserves_draws(self):
        model = SingularBayesianRanker(5, hidden=(6,), ranks=(3, 1), seed=3)
        restored = SingularBayesianRanker.from_state_dict(model.state_dict())
        x = np.random.default_rng(12).normal(size=(2, 7, 5))
        expected = model.predict_samples(x, draws=8, seed=13)["samples"]
        actual = restored.predict_samples(x, draws=8, seed=13)["samples"]
        np.testing.assert_array_equal(actual, expected)

    def test_candidate_permutation_changes_only_output_order(self):
        model = SingularBayesianRanker(4, hidden=(5,), ranks=(3, 1), seed=6)
        x = np.random.default_rng(7).normal(size=(2, 8, 4))
        order = np.array([3, 0, 7, 1, 6, 2, 5, 4])
        original = model.predict_samples(x, draws=16, seed=8)["samples"]
        permuted = model.predict_samples(x[:, order], draws=16, seed=8)["samples"]
        np.testing.assert_allclose(permuted, original[:, :, order], atol=1e-12)

    def test_reparameterized_gradient_matches_finite_difference(self):
        model = SingularBayesianRanker(3, hidden=(4,), ranks=(2, 1), seed=9)
        rng = np.random.default_rng(10)
        x = rng.normal(size=(3, 5, 3))
        targets = np.array([0, 2, 4])
        seed = 11
        _, _, _, gradients = model.loss_and_gradients(
            x, targets, np.random.default_rng(seed), kl_scale=1e-4
        )
        parameter = model.layers[0].A.mu
        index = (1, 0)
        original = parameter[index]
        epsilon = 1e-5
        values = []
        for direction in (1, -1):
            parameter[index] = original + direction * epsilon
            values.append(
                model.loss_and_gradients(
                    x, targets, np.random.default_rng(seed), kl_scale=1e-4
                )[0]
            )
        parameter[index] = original
        numerical = (values[0] - values[1]) / (2 * epsilon)
        self.assertAlmostEqual(gradients["layer0.A.mu"][index], numerical, places=5)


if __name__ == "__main__":
    unittest.main()
