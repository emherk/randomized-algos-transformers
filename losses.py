from abc import ABC
from functools import partial
from itertools import permutations
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import optax
import scipy
from jax import lax

from data_generator import Graph, AssociativeRecallData


class Loss(ABC):
    # Abstract base class for loss functions

    def get_loss_single_task(self, params, rng_env, rng_seed):
        # To be implemented by subclasses: computes loss for a single task/sample
        pass

    def __call__(self, params, rng, train=True):
        # Main entry point for computing the loss over a batch
        rng_env, rng_seed = jax.random.split(rng, 2)

        # Generate environment seeds for each batch element
        if train and self.cfg["num_train_seed"] > 0:
            rng_envs = jax.random.choice(
                rng_env,
                jax.random.split(jax.random.PRNGKey(0), self.cfg["num_train_seed"]),
                shape=(self.cfg["batch_size"],)
            )
        else:
            rng_envs = jax.random.split(rng_env, self.cfg["batch_size"])

        # Vectorized computation of loss and log_dict for each batch element
        loss, log_dict = jax.vmap(self.get_loss_single_task, in_axes=(None, 0, None))(
            params, rng_envs, rng_seed
        )

        # Aggregate statistics for logging
        log_dict = {
            "data_loss": loss,
            #             "data_loss_avr": jnp.mean(loss),
            "data_loss_max": jnp.max(loss),
            "data_loss_median": jnp.median(loss),
            #             **{k+"_avr": jnp.mean(v) for (k,v) in log_dict.items()},
            **{k + "_max": jnp.max(v) for (k, v) in log_dict.items()},
            **{k + "_median": jnp.median(v) for (k, v) in log_dict.items()},
            **log_dict,
        }

        # Pooling strategies for aggregating loss across the batch
        if self.cfg["data_pooling"] == "mean":
            return jnp.mean(loss), log_dict
        if self.cfg["data_pooling"] == "lp":
            if self.cfg["p"] >= 0:
                max_loss = jnp.max(loss)
                lp_loss = jnp.where(max_loss > 0,
                                    max_loss * jnp.mean((loss / max_loss) ** self.cfg["p"]) ** (1 / self.cfg["p"]), 0)
            else:
                lp_loss = jnp.max(loss)
            return lp_loss, log_dict
        if self.cfg["data_pooling"] == "powp":
            max_loss = jax.lax.stop_gradient(jnp.max(loss))
            #             lp_loss = jnp.mean((loss/max_loss) ** self.cfg["p"])
            lp_loss = jnp.where(max_loss > 0, max_loss * jnp.mean((loss / max_loss) ** self.cfg["p"]), 0)
            return lp_loss, log_dict
        if self.cfg["data_pooling"] == "max":
            return jnp.max(loss), log_dict
        if self.cfg["data_pooling"] == "softmax":
            if self.cfg["p"] >= 0:
                return (
                    jnp.sum(loss * jax.nn.softmax(loss / (loss.mean() + 0.00001) * (self.cfg["p"]), axis=0)),
                    log_dict,
                )
            else:
                return jnp.max(loss), log_dict

    @partial(jax.jit, static_argnums=(0, 2, 3))
    def eval_fn(self, params, num_batches, eval_on_train=False):
        # Evaluate the loss over multiple batches for validation/testing

        def eval_step(_, rng):
            loss, log_dict = jax.vmap(self.__call__, in_axes=(None, 0, None))(
                params, jax.random.split(rng, 10), eval_on_train
            )
            return None, (loss, log_dict)

        _, (loss, log_dict) = jax.lax.scan(
            eval_step, None, jax.random.split(jax.random.PRNGKey(0), num_batches)
        )

        # Aggregate statistics across all batches
        return {
            k: v.mean() if "max" not in k else v.max() for (k, v) in log_dict.items()
        }

    def dummy_data(self):
        # To be implemented by subclasses: returns dummy data for testing/model initialization
        pass


def is_valid_3coloring(adjacency, colors):
    # Checks if a coloring is a valid 3-coloring for a given adjacency matrix
    valid = True
    for i in range(3):
        subgraph = adjacency * (colors == i)
        valid = valid * (jnp.sum(subgraph @ subgraph) == 0)
    return valid


def generate_one_hot_combinations(K, T):
    # Generate all possible one-hot encoded sequences of length T with K classes
    indices = list(product(range(K), repeat=T))

    def one_hot_encode(index):
        # One-hot encode a sequence of indices
        return jnp.eye(K)[jnp.array(index)]

    # Vectorized one-hot encoding for all sequences
    one_hot_tensor = lax.map(one_hot_encode, jnp.array(indices))

    return one_hot_tensor


class ColoringLoss(Loss):
    # Loss class for the graph coloring task

    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
        assert self.cfg["probabilistic"] in ["single_seed", "deterministic", "random"]

        n = cfg["num_vertex"]
        self.graph_generator = Graph(
            num_vertex=n,
        )

        # Canonical cycle adjacency matrix for n vertices
        CANONICAL_CYCLE = jnp.eye(n, k=-1) + jnp.eye(
            cfg["num_vertex"], k=n - 1
        )
        self.CANONICAL_CYCLE = CANONICAL_CYCLE

        # Precompute all valid 3-colorings for the canonical cycle
        if True:  # self.cfg["loss"] != "fractional_coloring":
            ALL_3_HOTS = generate_one_hot_combinations(3, n)
            ALL_COLORS = jnp.argmax(ALL_3_HOTS, axis=-1)
            CANONICAL_CYCLE = ((CANONICAL_CYCLE + CANONICAL_CYCLE.T) > 0) * 1.0
            VALID_IDS_FOR_CANONICAL_CYCLE = jax.vmap(is_valid_3coloring, in_axes=(None, 0))(
                CANONICAL_CYCLE, ALL_COLORS
            )
            self.VALID_COLORS_FOR_CANONICAL_CYCLE = ALL_COLORS[
                VALID_IDS_FOR_CANONICAL_CYCLE
            ]
            self.VALID_HOTS_FOR_CANONICAL_CYCLE = ALL_3_HOTS[VALID_IDS_FOR_CANONICAL_CYCLE]
        # Permutation matrices for cycle symmetries
        S = np.eye(n - 1)[list(permutations(range(n - 1)))].reshape((2, -1, n - 1, n - 1))[0]
        R = jnp.array([scipy.linalg.block_diag(1, P) for P in S])
        J = jnp.eye(n, k=1) + np.eye(n, k=-n + 1)
        C = J + J.T
        self.ALL_CYCLES = (R @ C @ jnp.moveaxis(R, -1, -2), R)

    def expected_validity(self, adjacency, logits, permutation):
        # Computes the expected validity of a coloring under the model's output distribution
        log_probs = jax.nn.log_softmax(logits)
        valid_hots = jnp.einsum(
            "tvh,Vv->tVh", self.VALID_HOTS_FOR_CANONICAL_CYCLE, permutation
        )
        valid_probs = jnp.exp((valid_hots * log_probs).sum((-1, -2)))
        return valid_probs.sum()

    def best_cross_entropy(self, adjacency, logits, permutation):
        # Computes the best cross-entropy loss over all valid colorings
        valid_colors = jnp.einsum(
            "tv,Vv->tV", self.VALID_COLORS_FOR_CANONICAL_CYCLE, permutation
        ).astype(int)
        loss = jax.vmap(
            optax.softmax_cross_entropy_with_integer_labels, in_axes=(None, 0)
        )(logits, valid_colors)

        loss = jnp.mean(loss, axis=-1)
        return jnp.min(loss)

    def best_hinge(self, adjacency, logits, permutation):
        # Computes the best hinge loss over all valid colorings
        valid_hots = jnp.einsum(
            "tvh,Vv->tVh", self.VALID_HOTS_FOR_CANONICAL_CYCLE, permutation
        ).astype(int)
        loss = jax.vmap(optax.hinge_loss, in_axes=(None, 0))(logits, valid_hots * 2 - 1)
        loss = jnp.mean(loss, axis=(-1, -2))
        return jnp.min(loss)

    # Still under development
    def fractional_coloring(self, adjacency, logits, permutation):
        # Fractional coloring loss (probabilistic relaxation)
        log_probs = jax.nn.log_softmax(logits)
        log_probs_neighbor = (permutation @ self.CANONICAL_CYCLE @ permutation.T) @ log_probs

        log_prob_invalid_edge = log_probs_neighbor + log_probs
        loss = -jnp.log(1e-4 + 1. - jnp.exp(log_prob_invalid_edge).sum(-1)).mean() + jnp.log(1e-4 + 1.)
        #         log_prob_invalid_edge = jnp.minimum(log_probs_neighbor, log_probs)
        #         loss = (optax.hinge_loss(log_prob_invalid_edge+0.5,-1)).mean()
        return loss

    # Still under development
    def fractional_coloring_hinge(self, adjacency, logits, permutation):
        # Fractional coloring loss with hinge
        log_probs = jax.nn.log_softmax(logits)
        log_probs_neighbor = (permutation @ self.CANONICAL_CYCLE @ permutation.T) @ log_probs

        #         log_prob_invalid_edge = log_probs_neighbor +  log_probs
        #         loss = -jnp.log(1e-4 + 1. - jnp.exp(log_prob_invalid_edge).sum(-1)).mean() + jnp.log(1e-4 + 1.)
        log_prob_invalid_edge = jnp.minimum(log_probs_neighbor, log_probs)
        loss = (optax.hinge_loss(log_prob_invalid_edge + 0.5, -1)).mean()
        return loss

    # Still under development
    def fractional_coloring_prob(self, adjacency, logits, permutation):
        # Fractional coloring loss (probabilistic, no log)
        log_probs = jax.nn.log_softmax(logits)
        log_probs_neighbor = (permutation @ self.CANONICAL_CYCLE @ permutation.T) @ log_probs

        log_prob_invalid_edge = log_probs_neighbor + log_probs
        loss = jnp.exp(log_prob_invalid_edge).mean()
        return loss

    # Still under development
    def fractional_coloring_hard(self, adjacency, logits, permutation):
        # Fractional coloring loss with hard assignments
        probs = jax.nn.one_hot(jnp.argmax(logits, -1), logits.shape[-1])
        probs_neighbor = (permutation @ self.CANONICAL_CYCLE @ permutation.T) @ probs

        prob_invalid_edge = probs_neighbor * probs
        loss = prob_invalid_edge.mean()
        return loss

    def get_loss_from_input(self, params, graphs, adjacencies, permutations):
        # Computes the loss given model parameters and input data
        logits = self.model.apply(
            params, (graphs, adjacencies)
        )
        predictions = jnp.argmax(logits, axis=-1)

        # Select loss function based on config
        if self.cfg["loss"] == "validity":
            loss = 1 - is_valid_3coloring(adjacencies, predictions)
        elif self.cfg["loss"] == "expected_validity":
            loss = 1 - self.expected_validity(
                adjacencies, logits, permutations
            )
        elif self.cfg["loss"] == "best_cross_entropy":
            loss = self.best_cross_entropy(adjacencies, logits, permutations)
        elif self.cfg["loss"] == "best_hinge":
            loss = self.best_hinge(adjacencies, logits, permutations)
        elif self.cfg["loss"] == "fractional_coloring":
            loss = self.fractional_coloring(adjacencies, logits, permutations)
        elif self.cfg["loss"] == "fractional_coloring_hinge":
            loss = self.fractional_coloring_hinge(adjacencies, logits, permutations)
        elif self.cfg["loss"] == "fractional_coloring_prob":
            loss = self.fractional_coloring_prob(adjacencies, logits, permutations)
        elif self.cfg["loss"] == "fractional_coloring_hard":
            loss = self.fractional_coloring_hard(adjacencies, logits, permutations)

        return loss, {"inacc": 1 - is_valid_3coloring(adjacencies, predictions)}, logits

    def get_loss(self, params, mat_seed, rng_seed):
        # Samples a graph and computes the loss for it
        graphs, adjacencies, permutations = self.graph_generator.sample(mat_seed, rng_seed)
        return self.get_loss_from_input(params, graphs, adjacencies, permutations)

    def get_loss_single_task(self, params, rng_env, rng_seed):
        # Computes the loss for a single task, possibly with multiple seeds
        num_seed = self.cfg["num_seed"]
        if self.cfg["probabilistic"] == "single_seed":
            num_seed = 1
        elif self.cfg["probabilistic"] == "deterministic":
            num_seed = 1
            rng_seed = jax.random.PRNGKey(self.cfg["seed"])

        loss, log_dict, _ = jax.tree_util.tree_map(lambda x: x.mean(0),
                                                   jax.vmap(self.get_loss,
                                                            in_axes=(None, None, 0))(params,
                                                                                     rng_env,
                                                                                     jax.random.split(rng_seed,
                                                                                                      num_seed)))
        return loss, log_dict

    def dummy_data(self):
        # Returns dummy data for model initialization/testing
        rng = jax.random.PRNGKey(0)
        dummy_data, adjacency, _ = self.graph_generator.sample(rng, rng)
        return (dummy_data, adjacency)


def custom_sigmoid_binary_cross_entropy(logits, labels):
    # Custom implementation of sigmoid binary cross-entropy loss
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p


class AssociativeRecallLoss(Loss):
    # Loss class for the associative recall task

    def __init__(self, model, cfg, foobar=False):
        self.model = model
        self.cfg = cfg
        self.foobar = foobar
        assert self.cfg["probabilistic"] in ["single_seed", "deterministic", "random"]
        self.data_generator = AssociativeRecallData(num_token=cfg["num_token"], target_size=cfg["target_size"],
                                                    seed_size=cfg["seed_size"],
                                                    hardcoded_randomness=cfg["hardcoded_randomness"],
                                                    foobar=self.foobar)

    def get_loss_from_input(self, params, tokens, aux):
        # Computes the loss given model parameters and input data
        (label, Y, y_target) = aux
        if self.foobar:
            seq_len = tokens[0].shape[0]
        else:
            seq_len = tokens.shape[0]
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        prediction = self.model.apply(params, (tokens, mask))

        # Select loss function based on config
        if self.cfg["loss"] == "contrastive_hinge":
            prediction = jnp.where(Y != 0, Y / jnp.linalg.norm(Y, axis=-1, keepdims=True), 0) @ prediction[-1]
            loss = optax.hinge_loss(prediction, label * 2 - 1).mean()
            acc = (label[jnp.argmax(prediction)] == jnp.max(label))

        elif self.cfg["loss"] == "contrastive_ce":
            prediction = jnp.where(Y != 0, Y / jnp.linalg.norm(Y, axis=-1, keepdims=True), 0) @ prediction[-1]
            loss = optax.softmax_cross_entropy(prediction, label * 1).mean()
            acc = (label[jnp.argmax(prediction)] == jnp.max(label))

        elif self.cfg["loss"] == "bce":
            prediction = prediction[-1:]
            loss = custom_sigmoid_binary_cross_entropy(prediction, y_target).mean()
            acc = jnp.all((prediction > 0) == (y_target > 0), axis=-1)

        elif self.cfg["loss"] == "bce_mse":
            prediction = prediction[-1:]
            loss = ((jax.nn.sigmoid(prediction) - y_target) ** 2).mean()
            acc = jnp.all((prediction > 0) == (y_target > 0), axis=-1)

        elif self.cfg["loss"] == "mse":
            prediction = prediction[-1:]
            loss = ((prediction - y_target) ** 2).mean()
            acc = jnp.all((prediction > 0.5) == (y_target > 0), axis=-1)

        log_dict = {"inacc": 1. - acc}

        return loss, log_dict, prediction

    def get_loss(self, params, mat_seed, rng_seed):
        # Samples data and computes the loss for it
        tokens, (label, Y, y_target) = self.data_generator.sample(mat_seed, rng_seed)
        return self.get_loss_from_input(params, tokens, (label, Y, y_target))

    def get_loss_single_task(self, params, rng_env, rng_seed):
        # Computes the loss for a single task, possibly with multiple seeds
        num_seed = self.cfg["num_seed"]
        if self.cfg["probabilistic"] == "single_seed":
            num_seed = 1
        elif self.cfg["probabilistic"] == "deterministic":
            num_seed = 1
            rng_seed = jax.random.PRNGKey(self.cfg["seed"])


        loss, log_dict, _ = jax.tree_util.tree_map(
            lambda x: x.mean(0),
            jax.vmap(self.get_loss, in_axes=(None, None, 0))(
                params, rng_env, jax.random.split(rng_seed, num_seed)
            )
        )

        log_dict["inacc_lenient"] = log_dict["inacc"] == 1
        log_dict["inacc_harsh"] = log_dict["inacc"] > 0

        return loss, log_dict

    def dummy_data(self):
        # Returns dummy data for model initialization/testing
        seed = jax.random.PRNGKey(1)
        tokens, label = self.data_generator.sample(seed, seed)
        #         print(tokens, label)
        if self.foobar:
            seq_len = tokens[0].shape[0]
        else:
            seq_len = tokens.shape[0]
        return tokens, jnp.tril(jnp.ones((seq_len, seq_len)))
