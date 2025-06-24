X = jnp.eye(self.num_token)
Y = jax.random.choice(rng_Y, jnp.arange(0, 2, 1), shape=(self.num_token, self.target_size))
Y = Y.at[query_idx].set(y_query)

contrastive_target = jnp.all(Y == Y[query_idx], axis=-1)
contrastive_target = contrastive_target / contrastive_target.sum()

permutation = jax.random.permutation(rng_seed, jnp.eye(self.num_token))
seed = X @ permutation
X = jnp.concatenate([X, seed], axis=-1)