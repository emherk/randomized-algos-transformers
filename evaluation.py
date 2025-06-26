import glob
import jax
import pandas as pd
import jax.numpy as jnp
import numpy as np
import haiku as hk
import matplotlib.pyplot as plt
from data_generator import AssociativeRecallData
from associative_recall import get_model
from model_rng import CustomTransformer

CHECKPOINTS_PATH = "checkpoints/*"


def get_model(cfg):
    return hk.without_apply_rng(
        hk.transform(
            lambda x: CustomTransformer(
                out_dim=cfg["target_size"],
                num_layers=cfg["num_layers"],
                num_heads=cfg["num_heads"],
                kq_dim=cfg["kq_dim"],
                v_dim=cfg["v_dim"],
                embed_dim=cfg["embed_dim"],
                softmax=cfg["softmax"],
                positional_embedding=cfg["positional_embedding"],
                first_embedding_init_var=cfg["first_embedding_init_var"],
                w_init_var=cfg["w_init_var"],
                mlp=cfg["mlp"],
                widening_factor=cfg["widening_factor"],
                first_mlp=cfg["first_mlp"],
                reverse_block=cfg["reverse_block"],
            )(x)
        )
    )


def get_all_run_files():
    # Directory containing the checkpoint files

    # Get all files in the directory except files that have 'log_dict' in their name
    files = [f for f in glob.glob(CHECKPOINTS_PATH) if "log_dict" not in f]
    print(files)

    df = pd.DataFrame(files, columns=["file_path"])
    df[["path", "recall", "run_id", "probabilistic", "q", "c"]] = (
        df["file_path"]
        .str.replace(".npy", "")
        .str.split(r"_(?!seed)|(?<!single)_", expand=True)
    )
    df = df.drop(columns=["path", "recall"])
    df["q"] = df["q"].str.replace("q", "").astype(int)
    df["c"] = df["c"].str.replace("c", "").astype(int)

    df.sort_values(by=["probabilistic", "q", "c"], inplace=True)
    print(df)
    return df


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


def get_random_seed_batch(num_batches, base_seed=0):
    # Create a base key
    base_key = jax.random.PRNGKey(base_seed)
    # Split into num_batches*2 keys, then reshape
    keys = jax.random.split(base_key, num_batches * 2)
    rng_envs = keys[:num_batches]
    rng_seeds = keys[num_batches:]
    return rng_envs, rng_seeds

def get_random_seeds(cfg, num_batches=10):
    """
    Generate random seeds, based on the code structure from train
    mat_seed == rng_env from rng_envs
    rng_seed == seed_array from rng_seed
    from rng_env, rng_seed = jax.random.split(rng, 2)
    from epoch rng
    from self.rng, epoch_rng = jax.random.split(self.rng)
    from self.rng, model_init_rng = jax.random.split(
    jax.random.PRNGKey(cfg["seed"]), 2
    )
    from cfg["seed"]

    """
    rng = jax.random.split(jax.random.PRNGKey(0), num_batches)  # in eval fn
    rng_env, rng_seed = jax.random.split(rng, 2)

    rng_env = jax.random.split(rng_env, 30)[0]

    num_seed = cfg["num_seed"]
    if cfg["probabilistic"] == "single_seed":
        num_seed = 1
    elif cfg["probabilistic"] == "deterministic":
        num_seed = 1
        rng_seed = jax.random.PRNGKey(cfg["seed"])

    rng_seed = jax.random.split(rng_seed, num_seed)[0]
    return rng_env, rng_seed


def make_query_tokens(tokens, context_len):
    # For each i, replace the last token with the i-th memory token
    def replace_query(i):
        # Copy tokens to avoid mutation
        tokens_new = tokens.at[context_len, :context_len].set(tokens[i, :context_len])
        return tokens_new

    return jax.vmap(replace_query)(jnp.arange(context_len))


def eval_run(file_path):
    cfg, params = np.load(file_path, allow_pickle=True)
    context_len = cfg["num_token"]

    data_generator = AssociativeRecallData(
        num_token=context_len,
        target_size=5,
        seed_size=10,
        hardcoded_randomness=False,
        foobar=False,
    )

    rng_env, rng_seed = get_random_seeds(cfg)
    tokens, (label, Y, y_target) = data_generator.sample(rng_env, rng_seed)
    mask = jnp.tril(jnp.ones((tokens.shape[0], tokens.shape[0])))

    model = get_model(cfg)

    # Create all query-token-modified sequences
    all_tokens = make_query_tokens(
        tokens, context_len
    )  # shape: (context_len, context_len+1, token_dim)

    # Prepare masks for each sequence (broadcasted)
    all_masks = jnp.broadcast_to(mask, (context_len,) + mask.shape)

    # Run the model for all queries in parallel
    predictions = jax.vmap(lambda t, m: model.apply(params, (t, m)))(
        all_tokens, all_masks
    )
    predictions = predictions[
        :, -1, :
    ]  # Get the last token's prediction for each sequence

    # if below 0 # then set to 0, else set to 1
    predictions_binary = jnp.where(predictions < 0, 0, 1)

    predictions_binary == Y
    # get how many rows match the target Y
    num_matches = jnp.sum(
        jnp.sum(predictions_binary == Y, axis=1) == 5
    )  # shape: (context_len,)
    print(f"Number of matches: {num_matches} for context length {context_len}")

    # accuracy
    accuracy = num_matches / context_len
    print(f"Accuracy: {accuracy:.2f}")

    return accuracy


def eval_all_runs():
    df = get_all_run_files()

    accuracies = []
    for _, row in df.iterrows():
        run_id = row["run_id"]
        file_path = row["file_path"]
        print(f"Evaluating run: {run_id} from file: {file_path}")
        accuracy = eval_run(file_path)
        accuracies.append(accuracy)

    df["accuracy"] = accuracies
    print(df)

    return df


def eval_run_over_seeds(cfg, params):
    results = []
    context_len = cfg["num_token"]
    model = get_model(cfg)

    # 100 experiments
    for j in range(100):
        # generate a context of length N = 20
        data_generator = AssociativeRecallData(
            num_token=context_len,
            target_size=5,
            seed_size=10,
            hardcoded_randomness=False,
            foobar=False,
        )

        rng_envs, rng_seeds = get_random_seed_batch(30, base_seed=j)

        tokens, (label, Y, y_target) = data_generator.sample(rng_envs[0], rng_seeds[0])
        mask = jnp.tril(jnp.ones((context_len + 1, context_len + 1)))

        keys = tokens[:, :context_len]
        values = jnp.concat([Y, jnp.zeros((1, 5))])

        results_over_seed = []

        for i in range(30):
            new_tokens, (_, _, _) = data_generator.sample(rng_envs[i], rng_seeds[i])
            new_seeds = new_tokens[:, 20:30]
            tokens_with_new_seed = jnp.concat([keys, new_seeds, values], axis=1)

            prediction = model.apply(params, (tokens_with_new_seed, mask))
            prediction = prediction[-1]

            prediction_binary = jnp.where(prediction < 0, 0, 1)
            results_over_seed.append((prediction_binary == y_target).all())
        # Calculate the accuracy for this seed
        results.append(sum(results_over_seed) / len(results_over_seed))
    return results

def eval_run_over_seeds_from_file(run_id):
    file_path = get_run_file(run_id)
    cfg, params = np.load(file_path, allow_pickle=True)
    print(f"Evaluating run from file: {file_path}")
    accuracy = eval_run_over_seeds(cfg=cfg, params=params)
    plt.hist(accuracy, bins=20, range=(0, 1), density=True, alpha=0.7)
    plt.title(f"Accuracy distribution for run_id: {run_id} q: {cfg['p']}, c: {cfg['num_token']}")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.show()

    return accuracy

def eval_all_over_seeds():
    df = get_all_run_files()
    df = df[df["c"] == 20]

    accuracies = []
    for _, row in df.iterrows():
        run_id = row["run_id"]
        file_path = row["file_path"]
        print(f"Evaluating run: {run_id} from file: {file_path}")
        cfg, params = np.load(file_path, allow_pickle=True)
        accuracy = eval_run_over_seeds(cfg=cfg, params=params)
        # make a histogram of the last returned accuracy
        # we want the fraction of inputs that were correctly predicted
        # over what fraction of the time
        # there should be 20 bins from 0 to 1
        plt.hist(accuracy, bins=20, range=(0, 1), density=True, alpha=0.7)
        plt.title(f"Accuracy distribution for run: {run_id}")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.show()

    df["accuracy"] = accuracies
    print(df)

    return df

    #     # generate m=30 random seeds
    #     seeds = random(m)
    #     random_results = []
    #     for s in seeds:
    #         r = experiment(context, s)
    #         random_result.append(r)

    #     # q = 1
    #     # random_results = [0, 0 0 0 0 ]

    #     # q = 32
    #     # random_results = [0, 0, 1, 1, 1, 0, 0...]

    #     # calculate percentage correct
    #     results.append(sum(random_results) / len(random_resu)


def eval_over_context_lengths():
    df = pd.read_csv("run_accuracies.csv")
    df = df[df['q'] == 1]  # Only evaluate q=1 runs
    df = df[df['probabilistic'] != 'deterministic']

    # plot the accuracy column over the context lengths with separate lines for each probabilistic setting
    plt.figure(figsize=(10, 6))
    for prob_setting in df['probabilistic'].unique():
        subset = df[df['probabilistic'] == prob_setting]
        plt.plot(subset['c'], subset['accuracy'], label=prob_setting)
    plt.xlabel("Context Length")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Context Lengths for q=1")
    plt.legend()
    plt.show()

    return df

if __name__ == "__main__":
    # df = eval_run_over_seeds_from_file("53vak94d")
    # df.to_csv("run_accuracies_over_seeds.csv", index=False)
    # get single seed, q = 1, c = varied
    # get random, q = 1, c = varied
    eval_over_context_lengths()

