import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
from functools import partial
from data_generator import AssociativeRecallData
from associative_recall import get_model
import re

RUN_ID = "z3oi1h7l"


def get_run_file(run_id):
    # get all files with run_id in the name
    # and that doesn't contain log_dict
    import os

    files = [
        f for f in os.listdir("checkpoints") if run_id in f and "log_dict" not in f
    ]
    if len(files) == 0:
        raise ValueError(f"No files found for run_id {run_id}")
    if len(files) > 1:
        raise ValueError(f"Multiple files found for run_id {run_id}: {files}")
    return "checkpoints/" + files[0]


context_len = 20
data_generator = AssociativeRecallData(
    num_token=context_len,
    target_size=5,
    seed_size=10,
    hardcoded_randomness=False,
    foobar=False,
)


def parse_numpy_array_string(s):
    # Find all bracketed rows
    rows = re.findall(r"\[([^\[\]]+)\]", s)
    arrays = []
    for row in rows:
        arr = np.fromstring(row.replace("\n", " "), sep=" ")
        arrays.append(arr)
    return np.stack(arrays)


def split_tokens(arr):
    arr = arr[-2]
    keys = arr[:key_size]
    seeds = arr[key_size : key_size + seed_size]
    values = arr[key_size + seed_size :]
    return keys, values, seeds


# mat_seed == rng_env from rng_envs
# rng_seed == seed_array from rng_seed
# from rng_env, rng_seed = jax.random.split(rng, 2)
# from epoch rng
# from self.rng, epoch_rng = jax.random.split(self.rng)
# from self.rng, model_init_rng = jax.random.split(
# jax.random.PRNGKey(cfg["seed"]), 2
# )
# from cfg["seed"]
def get_random_seeds(cfg):
    num_batches = 10
    rng = jax.random.split(jax.random.PRNGKey(0), num_batches)[0]  # in eval fn
    rng = jax.random.split(rng, 10)[0]  # in eval_step in eval_fn

    rng_env, rng_seed = jax.random.split(rng, 2)

    rng_env = jax.random.choice(
        rng_env,
        jax.random.split(jax.random.PRNGKey(0), cfg["num_train_seed"]),
        shape=(cfg["batch_size"],),
    )[0]

    if cfg["probabilistic"] == "deterministic":
        rng_seed = jax.random.PRNGKey(cfg["seed"])
    return rng_env, rng_seed


rng_env, rng_seed = get_random_seeds(
    cfg={
        "num_train_seed": 10,
        "batch_size": 512,
        "probabilistic": "deterministic",  # or "single_seed" or "multiple_seed"
        "seed": 42,
    }
)
cfg, params = np.load(get_run_file(RUN_ID), allow_pickle=True)
tokens, (label, Y, y_target) = data_generator.sample(rng_env, rng_seed)
mask = jnp.tril(jnp.ones((tokens.shape[0], tokens.shape[0])))

model = get_model()


# for i in range(context_len):
#     tokens[context_len - 1, :context_len] = tokens[i, :context_len]  # change the query token to the i-th token
#     expected_result = Y[i, :]
#     prediction = model.apply(params, (tokens, mask))


def make_query_tokens(tokens, context_len):
    # For each i, replace the last token with the i-th memory token
    def replace_query(i):
        # Copy tokens to avoid mutation
        tokens_new = tokens.at[context_len, :context_len].set(
            tokens[i, :context_len]
        )
        return tokens_new

    return jax.vmap(replace_query)(jnp.arange(context_len))


# do with for loop
def make_query_tokens_for_loop(tokens, context_len):
    # For each i, replace the last token with the i-th memory token
    tokens_list = []
    for i in range(context_len):
        # Copy tokens to avoid mutation
        tokens_new = tokens.at[context_len, :context_len].set(
            tokens[i, :context_len]
        )
        tokens_list.append(tokens_new)
    return jnp.stack(tokens_list)  # shape: (context_len, context_len+1, token_dim)


# Create all query-token-modified sequences
all_tokens = make_query_tokens(
    tokens, context_len
)  # shape: (context_len, context_len+1, token_dim)

# Prepare masks for each sequence (broadcasted)
all_masks = jnp.broadcast_to(mask, (context_len,) + mask.shape)

# Run the model for all queries in parallel
predictions = jax.vmap(lambda t, m: model.apply(params, (t, m)))(all_tokens, all_masks)
predictions = predictions[:, -1, :]  # Get the last token's prediction for each sequence

# if below 0 # then set to 0, else set to 1
predictions_binary = jnp.where(predictions < 0, 0, 1)

predictions_binary == Y  
# get how many rows match the target Y
num_matches = jnp.sum(jnp.sum(predictions_binary == Y, axis=1) == 5) # shape: (context_len,)
print(f"Number of matches: {num_matches}")

# accuracy 
accuracy = num_matches / context_len
print(f"Accuracy: {accuracy:.2f}")


