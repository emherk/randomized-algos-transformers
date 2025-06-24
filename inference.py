import numpy as np
import jax
import pandas as pd

from associative_recall import get_model
import jax.numpy as jnp

import numpy as np
import re

STEP_NUM = 9 
RUN_ID = "mgmbyc9h"

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


# Example usage with a DataFrame column:

cfg, params = np.load(f"checkpoints/associative_recall_{RUN_ID}.npy", allow_pickle=True)
df = pd.read_csv(f"checkpoints/associative_recall_{RUN_ID}_{STEP_NUM}_io.csv")
df["input_tokens"] = df["input_tokens"].apply(parse_numpy_array_string)

key_size = cfg["num_token"]
value_size = cfg["target_size"]
seed_size = cfg["seed_size"]

full_size = key_size + value_size + seed_size

df["keys"], df["values"], df["seeds"] = zip(*df["input_tokens"].apply(split_tokens))

model = get_model()

# tokens = jnp.eye(cfg["num_token"])  # Example input, replace with actual tokens
# Create the mask as in training
# get size from config: keys + values + seeds

# Get the model's prediction
tokens = df["input_tokens"][0]
mask = jnp.tril(jnp.ones((tokens.shape[0], tokens.shape[0])))
prediction = model.apply(params, (tokens, mask))


predicted_value = prediction[-1]  # shape: (target_size,)
print("Predicted value:", predicted_value)
print("True value:", df['values'][0])

predicted_binary = (jax.nn.sigmoid(predicted_value) > 0.5).astype(int)
print("Predicted (binary):", predicted_binary)

# predict all values and see how many are correct
predictions = []
for i in range(len(df)):
    tokens = np.array([df["input_tokens"][i]])
    prediction = model.apply(params, (tokens, mask))
    predicted_value = prediction.squeeze()[-1]  # shape: (target_size,)
    predicted_binary = (jax.nn.sigmoid(predicted_value) > 0.5).astype(int)
    predictions.append(predicted_binary)

# Calculate accuracy
predictions = np.array(predictions)
accuracy = np.mean(np.all(predictions == np.array(df['values'].tolist()), axis=1))
print(f"Total predictions: {len(predictions)}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# now see if we can predict values it has not seen before
# get random tokens
keys = jnp.eye(cfg["num_token"])
seeds = jax.random.bernoulli(jax.random.PRNGKey(0), shape=(cfg["seed_size"],))
values = jax.random.choice(jax.random.PRNGKey(1), jnp.arange(0, 2, 1), shape=(cfg["num_token"], cfg["target_size"]))
tokens = jnp.concatenate([keys, seeds[None, :], values], axis=-1)
# Create the mask as in training
mask = jnp.tril(jnp.ones((keys.shape[0], keys.shape[0])))
# Get the model's prediction
prediction = model.apply(params, (tokens, mask))
predicted_value = prediction[-1]  # shape: (target_size,)
predicted_binary = (jax.nn.sigmoid(predicted_value) > 0.5).astype(int)
print("Predicted value for unseen tokens:", predicted_value)
print("Predicted (binary) for unseen tokens:", predicted_binary)
# True value for unseen tokens
true_value = values[0]  # assuming we want to check the first token
print("True value for unseen tokens:", true_value)
# Check if the prediction matches the true value