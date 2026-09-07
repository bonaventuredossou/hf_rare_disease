"""Low-rank variational neural ranker adapted from the SBNN supplementary code.

Every weight matrix is sampled as W = A B^T. The same sampled network scores
all candidates in a case, which is required for a coherent listwise likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _softplus(x):
    return np.logaddexp(x, 0.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _softmax(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _log_mixture_prior(x, pi=0.5, sigma1=1.0, sigma2=math.exp(-6.0)):
    log1 = math.log(pi) - math.log(sigma1) - 0.5 * (x / sigma1) ** 2
    log2 = math.log(1.0 - pi) - math.log(sigma2) - 0.5 * (x / sigma2) ** 2
    high = np.maximum(log1, log2)
    logp = high + np.log(np.exp(log1 - high) + np.exp(log2 - high))
    responsibility1 = np.exp(log1 - logp)
    score = -x * (responsibility1 / sigma1**2 + (1.0 - responsibility1) / sigma2**2)
    constant = -0.5 * math.log(2.0 * math.pi)
    return logp + constant, score


@dataclass
class FactorPosterior:
    mu: np.ndarray
    rho: np.ndarray

    def sample(self, rng):
        epsilon = rng.standard_normal(self.mu.shape)
        sigma = _softplus(self.rho)
        return self.mu + sigma * epsilon, epsilon, sigma


class SingularBayesianLayer:
    """Gaussian posteriors on the low-rank factors and bias."""

    def __init__(
        self,
        inputs,
        outputs,
        rank,
        rng,
        relu=False,
        posterior_scale_multiplier=1.0,
        prior_pi=0.5,
        prior_sigma1=1.0,
        prior_sigma2=math.exp(-6.0),
    ):
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        self.rank = min(int(rank), self.inputs, self.outputs)
        variance = 2.0 / self.inputs if relu else 2.0 / (self.inputs + self.outputs)
        damping = 0.32 if self.rank <= 5 else 0.55
        bound = damping * math.sqrt(3.0) * (variance / self.rank) ** 0.25
        sigma = posterior_scale_multiplier * 0.09 * (variance / self.rank) ** 0.25
        rho = math.log(math.expm1(sigma))
        self.prior_pi = float(prior_pi)
        self.prior_sigma1 = float(prior_sigma1)
        self.prior_sigma2 = float(prior_sigma2)
        self.A = FactorPosterior(
            rng.uniform(-bound, bound, (self.inputs, self.rank)),
            np.full((self.inputs, self.rank), rho),
        )
        self.B = FactorPosterior(
            rng.uniform(-bound, bound, (self.outputs, self.rank)),
            np.full((self.outputs, self.rank), rho),
        )
        self.bias = FactorPosterior(np.zeros(self.outputs), np.full(self.outputs, rho))

    def initialize_from_matrix(self, matrix, bias=None):
        """Set factor means to the truncated-SVD approximation of a dense layer."""
        u, singular, vt = np.linalg.svd(np.asarray(matrix), full_matrices=False)
        root = np.sqrt(singular[: self.rank])
        self.A.mu[...] = u[:, : self.rank] * root
        self.B.mu[...] = vt[: self.rank].T * root
        if bias is not None:
            self.bias.mu[...] = bias

    def sample(self, rng):
        sampled = {}
        for name in ("A", "B", "bias"):
            value, epsilon, sigma = getattr(self, name).sample(rng)
            sampled[name] = value
            sampled[name + "_epsilon"] = epsilon
            sampled[name + "_sigma"] = sigma
        sampled["weight"] = sampled["A"] @ sampled["B"].T
        return sampled

    def kl_and_gradients(self, sampled):
        total = 0.0
        gradients = {}
        for name in ("A", "B", "bias"):
            posterior = getattr(self, name)
            value = sampled[name]
            epsilon = sampled[name + "_epsilon"]
            sigma = sampled[name + "_sigma"]
            log_q = -np.log(sigma) - 0.5 * epsilon**2 - 0.5 * math.log(2 * math.pi)
            log_p, prior_score = _log_mixture_prior(
                value,
                pi=self.prior_pi,
                sigma1=self.prior_sigma1,
                sigma2=self.prior_sigma2,
            )
            total += float(np.sum(log_q - log_p))
            dsigma = _sigmoid(posterior.rho)
            gradients[name + ".mu"] = -prior_score
            gradients[name + ".rho"] = -dsigma / sigma - prior_score * epsilon * dsigma
        return total, gradients


class DenseBayesianLayer:
    """Independent Gaussian posterior for each dense weight and bias."""

    def __init__(
        self,
        inputs,
        outputs,
        rng,
        relu=False,
        rho_init=-5.0,
        prior_pi=0.5,
        prior_sigma1=1.0,
        prior_sigma2=math.exp(-6.0),
    ):
        self.inputs = int(inputs)
        self.outputs = int(outputs)
        variance = 2.0 / self.inputs if relu else 2.0 / (self.inputs + self.outputs)
        bound = math.sqrt(3.0 * variance)
        self.weight = FactorPosterior(
            rng.uniform(-bound, bound, (self.inputs, self.outputs)),
            np.full((self.inputs, self.outputs), float(rho_init)),
        )
        self.bias = FactorPosterior(
            np.zeros(self.outputs),
            np.full(self.outputs, float(rho_init)),
        )
        self.prior_pi = float(prior_pi)
        self.prior_sigma1 = float(prior_sigma1)
        self.prior_sigma2 = float(prior_sigma2)

    def sample(self, rng):
        sampled = {}
        for name in ("weight", "bias"):
            value, epsilon, sigma = getattr(self, name).sample(rng)
            sampled[name] = value
            sampled[name + "_epsilon"] = epsilon
            sampled[name + "_sigma"] = sigma
        return sampled

    def kl_and_gradients(self, sampled):
        total = 0.0
        gradients = {}
        for name in ("weight", "bias"):
            posterior = getattr(self, name)
            value = sampled[name]
            epsilon = sampled[name + "_epsilon"]
            sigma = sampled[name + "_sigma"]
            log_q = -np.log(sigma) - 0.5 * epsilon**2 - 0.5 * math.log(2 * math.pi)
            log_p, prior_score = _log_mixture_prior(
                value,
                pi=self.prior_pi,
                sigma1=self.prior_sigma1,
                sigma2=self.prior_sigma2,
            )
            total += float(np.sum(log_q - log_p))
            dsigma = _sigmoid(posterior.rho)
            gradients[name + ".mu"] = -prior_score
            gradients[name + ".rho"] = -dsigma / sigma - prior_score * epsilon * dsigma
        return total, gradients


class SingularBayesianRanker:
    """Two-hidden-layer SBNN trained with a listwise causal-pair likelihood."""

    def __init__(self, input_dim, hidden=(32, 16), ranks=(8, 8, 1), seed=42):
        rng = np.random.default_rng(seed)
        widths = (input_dim, *hidden, 1)
        self.layers = [
            SingularBayesianLayer(
                widths[i], widths[i + 1], ranks[i], rng, relu=i < len(hidden)
            )
            for i in range(len(widths) - 1)
        ]

    def initialize_from_dense(self, weights, biases):
        if len(weights) != len(self.layers):
            raise ValueError("Dense and Bayesian networks must have the same depth")
        for layer, weight, bias in zip(self.layers, weights, biases):
            layer.initialize_from_matrix(weight, bias)

    def parameters(self):
        for i, layer in enumerate(self.layers):
            for name in ("A", "B", "bias"):
                posterior = getattr(layer, name)
                yield f"layer{i}.{name}.mu", posterior.mu
                yield f"layer{i}.{name}.rho", posterior.rho

    def state_dict(self):
        return {
            "input_dim": self.layers[0].inputs,
            "hidden": [layer.outputs for layer in self.layers[:-1]],
            "ranks": [layer.rank for layer in self.layers],
            "parameters": {name: value.tolist() for name, value in self.parameters()},
        }

    @classmethod
    def from_state_dict(cls, state):
        model = cls(
            state["input_dim"],
            hidden=tuple(state["hidden"]),
            ranks=tuple(state["ranks"]),
            seed=0,
        )
        saved = state["parameters"]
        for name, value in model.parameters():
            value[...] = np.asarray(saved[name], dtype=float)
        return model

    def _forward(self, x, rng):
        shape = x.shape
        h = x.reshape(-1, shape[-1])
        cache = []
        for i, layer in enumerate(self.layers):
            sampled = layer.sample(rng)
            z = h @ sampled["weight"] + sampled["bias"]
            cache.append((h, z, sampled))
            h = np.maximum(z, 0.0) if i < len(self.layers) - 1 else z
        return h.reshape(shape[0], shape[1]), cache

    def loss_and_gradients(self, x, targets, rng, kl_scale):
        logits, cache = self._forward(x, rng)
        probabilities = _softmax(logits)
        rows = np.arange(len(targets))
        data_loss = -float(
            np.mean(np.log(np.clip(probabilities[rows, targets], 1e-15, 1)))
        )
        gradient = probabilities
        gradient[rows, targets] -= 1.0
        gradient = gradient.reshape(-1, 1) / len(targets)
        data_gradients = []
        for i in range(len(self.layers) - 1, -1, -1):
            h, z, sampled = cache[i]
            grad_weight = h.T @ gradient
            grad_bias = gradient.sum(axis=0)
            data_gradients.append(
                {
                    "A.mu": grad_weight @ sampled["B"],
                    "B.mu": grad_weight.T @ sampled["A"],
                    "bias.mu": grad_bias,
                    "A.rho": (grad_weight @ sampled["B"])
                    * sampled["A_epsilon"]
                    * _sigmoid(self.layers[i].A.rho),
                    "B.rho": (grad_weight.T @ sampled["A"])
                    * sampled["B_epsilon"]
                    * _sigmoid(self.layers[i].B.rho),
                    "bias.rho": grad_bias
                    * sampled["bias_epsilon"]
                    * _sigmoid(self.layers[i].bias.rho),
                }
            )
            gradient = gradient @ sampled["weight"].T
            if i:
                gradient *= cache[i - 1][1] > 0
        data_gradients.reverse()
        gradients = {}
        kl = 0.0
        for i, (layer, (_, _, sampled), data_gradient) in enumerate(
            zip(self.layers, cache, data_gradients)
        ):
            layer_kl, layer_gradients = layer.kl_and_gradients(sampled)
            kl += layer_kl
            for name in data_gradient:
                gradients[f"layer{i}.{name}"] = (
                    data_gradient[name] + kl_scale * layer_gradients[name]
                )
        return data_loss + kl_scale * kl, data_loss, kl, gradients

    def predict_samples(self, x, draws=256, seed=42):
        rng = np.random.default_rng(seed)
        samples = np.stack([_softmax(self._forward(x, rng)[0]) for _ in range(draws)])
        mean = samples.mean(axis=0)
        predictive_entropy = -np.sum(mean * np.log(np.clip(mean, 1e-15, 1)), axis=1)
        expected_entropy = np.mean(
            -np.sum(samples * np.log(np.clip(samples, 1e-15, 1)), axis=2), axis=0
        )
        return {
            "samples": samples,
            "mean": mean,
            "std": samples.std(axis=0, ddof=1),
            "mutual_information": np.maximum(
                predictive_entropy - expected_entropy, 0.0
            ),
        }


class Adam:
    def __init__(self, parameters, learning_rate=1e-3):
        self.parameters = dict(parameters)
        self.learning_rate = learning_rate
        self.m = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.v = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.step_count = 0

    def step(self, gradients):
        self.step_count += 1
        for name, value in self.parameters.items():
            gradient = np.clip(gradients[name], -20.0, 20.0)
            self.m[name] = 0.9 * self.m[name] + 0.1 * gradient
            self.v[name] = 0.999 * self.v[name] + 0.001 * gradient**2
            corrected_m = self.m[name] / (1.0 - 0.9**self.step_count)
            corrected_v = self.v[name] / (1.0 - 0.999**self.step_count)
            value -= self.learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
            if name.endswith(".rho"):
                np.clip(value, -12.0, 2.0, out=value)


def train_dense_ranker(x, targets, hidden=(32, 16), epochs=300, seed=42):
    """Train the matching dense architecture only for optional SVD initialization."""
    rng = np.random.default_rng(seed)
    widths = (x.shape[-1], *hidden, 1)
    weights = [
        rng.normal(0, math.sqrt(2.0 / widths[i]), (widths[i], widths[i + 1]))
        for i in range(len(widths) - 1)
    ]
    biases = [np.zeros(width) for width in widths[1:]]
    parameters = []
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        parameters.extend([(f"w{i}", weight), (f"b{i}", bias)])
    optimizer = Adam(parameters, learning_rate=2e-3)
    rows = np.arange(len(targets))
    for _ in range(epochs):
        h = x.reshape(-1, x.shape[-1])
        cache = []
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            z = h @ weight + bias
            cache.append((h, z))
            h = np.maximum(z, 0.0) if i < len(weights) - 1 else z
        probabilities = _softmax(h.reshape(x.shape[:2]))
        gradient = probabilities
        gradient[rows, targets] -= 1
        gradient = gradient.reshape(-1, 1) / len(targets)
        gradients = {}
        for i in range(len(weights) - 1, -1, -1):
            previous, _ = cache[i]
            gradients[f"w{i}"] = previous.T @ gradient
            gradients[f"b{i}"] = gradient.sum(axis=0)
            gradient = gradient @ weights[i].T
            if i:
                gradient *= cache[i - 1][1] > 0
        optimizer.step(gradients)
    return weights, biases


def fit_sbnn(
    model,
    train_x,
    train_y,
    validation_x,
    validation_y,
    *,
    epochs=400,
    batch_size=64,
    kl_weight=0.01,
    warmup_epochs=40,
    learning_rate=1e-3,
    seed=42,
):
    rng = np.random.default_rng(seed)
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    best = None
    history = []
    for epoch in range(epochs):
        order = rng.permutation(len(train_y))
        factor = min(1.0, (epoch + 1) / warmup_epochs)
        kl_scale = factor * kl_weight / len(train_y)
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            total, data, kl, gradients = model.loss_and_gradients(
                train_x[index], train_y[index], rng, kl_scale
            )
            optimizer.step(gradients)
            losses.append((total, data, kl))
        if epoch % 10 == 9 or epoch == epochs - 1:
            prediction = model.predict_samples(
                validation_x, draws=64, seed=seed + epoch
            )["mean"]
            nll = -float(
                np.mean(
                    np.log(
                        np.clip(
                            prediction[np.arange(len(validation_y)), validation_y],
                            1e-15,
                            1,
                        )
                    )
                )
            )
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(np.mean([value[0] for value in losses])),
                    "data_loss": float(np.mean([value[1] for value in losses])),
                    "kl": float(np.mean([value[2] for value in losses])),
                    "validation_nll": nll,
                }
            )
            if best is None or nll < best[0]:
                best = (nll, {name: value.copy() for name, value in model.parameters()})
    for name, value in model.parameters():
        value[...] = best[1][name]
    return history
