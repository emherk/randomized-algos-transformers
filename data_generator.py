import jax
import jax.numpy as jnp


class Graph(object):
    def __init__(self, num_vertex):
        self.num_vertex = num_vertex

    def build(self, adjacency, rng_seed):
        noise = jax.random.uniform(rng_seed, shape=(self.num_vertex, 1))
        idx = jnp.eye(self.num_vertex)
        return jnp.concatenate([idx, noise], axis=-1)

    def _sample_cycle(self, env_seed):
        cycle = jnp.eye(self.num_vertex, k=-1) + jnp.eye(
            self.num_vertex, k=self.num_vertex - 1
        )
        permutation = jax.random.permutation(env_seed, jnp.eye(self.num_vertex))

        adjacency = permutation @ cycle @ permutation.T
        adjacency = adjacency + adjacency.T
        return adjacency, permutation

    def sample(self, graph_seed, rng_seed):
        adjacency, permutation = self._sample_cycle(graph_seed)
        return self.build(adjacency, rng_seed), adjacency, permutation


class AssociativeRecallData:
    def __init__(self, num_token, target_size, seed_size, hardcoded_randomness, foobar):
        self.num_token = num_token
        self.seed_size = seed_size
        self.hardcoded_randomness = hardcoded_randomness
        self.target_size = target_size
        self.foobar = foobar

    def build(self, rng_Y, rng_seed, y_query, query_idx):
        X = jnp.eye(self.num_token)
        Y = jax.random.choice(rng_Y, jnp.arange(0, 2, 1), shape=(self.num_token, self.target_size))
        Y = Y.at[query_idx].set(y_query)

        contrastive_target = jnp.all(Y == Y[query_idx], axis=-1)
        contrastive_target = contrastive_target / contrastive_target.sum()

        if self.foobar:
            seed = jax.random.bernoulli(rng_seed, shape=(self.seed_size,))
            seed_X = jax.vmap(lambda x: jnp.concatenate([x, seed], axis=-1))(X)
            query = X[query_idx]
            seq = jnp.concatenate(
                [jnp.concatenate([X, Y], axis=-1), jnp.concatenate([query, 0 * Y[query_idx]], axis=-1)], axis=0)
            seed_seq = jnp.concatenate([seed_X, seed_X[query_idx]], axis=0)
            return (seq, seed_seq), (contrastive_target, Y, Y[query_idx])

        if self.hardcoded_randomness:
            permutation = jax.random.permutation(rng_seed, jnp.eye(self.num_token))
            seed = X @ permutation
            X = jnp.concatenate([X, seed], axis=-1)

        else:
            #             seed = jax.random.choice(rng_seed, jnp.eye(self.num_token) )
            seed = jax.random.bernoulli(rng_seed, shape=(self.seed_size,))
            X = jax.vmap(lambda x: jnp.concatenate([x, seed], axis=-1))(X)

        query = X[query_idx]
        seq = jnp.concatenate([jnp.concatenate([X, Y], axis=-1), jnp.concatenate([query, 0 * Y[query_idx]], axis=-1)],
                              axis=0)
        return seq, (contrastive_target, Y, Y[query_idx])

    def sample(self, rng_env, rng_seed):
        rng_target, rng_Y, rng_query = jax.random.split(rng_env, 3)
        query_idx = jax.random.choice(rng_query, jnp.arange(self.num_token), shape=(1,))
        y_query = jax.random.choice(rng_target, jnp.arange(0, 2, 1), shape=(self.target_size,))
        return self.build(rng_Y, rng_seed, y_query, query_idx)
