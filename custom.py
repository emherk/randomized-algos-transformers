import jax

cfg = {
    "num_train_seed": 10,
    "batch_size": 512,
    "probabilistic": "deterministic",  # or "single_seed" or "multiple_seed"
}
num_batches = 10
rng = jax.random.split(jax.random.PRNGKey(0), num_batches)[0] # in eval fn
rng = jax.random.split(rng, 10)[0] # in eval_step in eval_fn

rng_env, rng_seed = jax.random.split(rng, 2)

rng_env = jax.random.choice(
    rng_env,
    jax.random.split(jax.random.PRNGKey(0), cfg["num_train_seed"]),
    shape=(cfg["batch_size"],),
)[0]

print(rng_env.shape)
if cfg['probabilistic'] == "deterministic":
    rng_seed = jax.random.PRNGKey(cfg["seed"])

pass