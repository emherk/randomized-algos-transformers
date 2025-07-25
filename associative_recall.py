from datetime import datetime
from itertools import product
import os
import pandas as pd

import haiku as hk
import jax
import jax.numpy as jnp

# sys.path.append(os.path.abspath('../'))
import numpy as np
import optax

import wandb
from losses import AssociativeRecallLoss, custom_sigmoid_binary_cross_entropy
from model_rng import CustomTransformer  # , Transformer
from trainer_gd import TrainerGD

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hk.vmap.require_split_rng = False

USE_WANDB = False

N = 5

cfg = {
    "qmc": False,
    "debug": True,
    "seed": 42,
    "num_jit_batches": 100,
    "num_steps": 1000,  # 30000,
    # Env
    "num_token": 8,
    "target_size": N,
    "batch_size": 512,
    "num_train_seed": 0,  # total: N * 2**B
    "probabilistic": "random",
    "data_pooling": "lp",  # "lp" "mean"
    "p": 32,
    "num_seed": 30,
    "loss": "bce",  # "contrastive_ce" "contrastive_hinge" "bce" "bce_mse" "mse"
    "hardcoded_randomness": False,
    "mlp": False,
    "widening_factor": 1,
    "first_mlp": True,
    "seed_size": 2 * N,
    "reverse_block": True,
    # RNG model
    "num_layers": 2,
    "num_heads": 1,
    "kq_dim": N,
    "v_dim": N,
    "embed_dim": 8 * N,  # 32,
    "softmax": "none",  # "all",
    "positional_embedding": False,
    "first_embedding_init_var": 1,
    "w_init_var": 1,
    "optim_algo": "gd",
    "GD_PARAM": {
        "lr": 3e-3,
        "betas": (0.9, 0.95),
        "eps": 1e-5,
        "grad_norm_clip": 1,
        "weight_decay": 0.01 * 0,
        "scheduler": "cosine",  # "warmup", "cosine"
        "lr_alpha": 0.1,
        "warmup_steps": 333,  # 1000,
    },
}


def generate_one_hot_combinations(K, T):
    # Generate all possible sequences of indices with T elements where each element ranges from 0 to K-1
    indices = list(product(range(K), repeat=T))

    # Convert these indices to one-hot vectors
    def one_hot_encode(index):
        # Create a one-hot vector for each index in the sequence
        return jnp.eye(K)[jnp.array(index)]

    return jnp.array(indices)


def save_advanced_log(cfg, loss_fn, params):
    all_y_target = generate_one_hot_combinations(2, cfg["target_size"])
    all_query_idx = jnp.arange(cfg["num_token"])[:, None]
    rng_static, rng_dynamic = jax.random.split(jax.random.PRNGKey(0))

    all_log_dict = {}
    if cfg["probabilistic"] == "single_seed":
        num_seed = 100
    elif cfg["probabilistic"] == "deterministic":
        num_seed = 1
        rng_dynamic = jax.random.PRNGKey(cfg["seed"])
    else:
        num_seed = 100

    rng_list = jax.random.split(rng_dynamic, num_seed)
    for rng_seed in rng_list:
        all_log_dict_tmp = {}
        for rng_Y in jax.random.split(rng_static, 16):
            rng_Y = jax.vmap(lambda r: jax.random.split(r, all_y_target.shape[0]))(
                jax.random.split(rng_Y, all_query_idx.shape[0])
            )
            input, target = jax.vmap(
                jax.vmap(loss_fn.data_generator.build, in_axes=(0, None, 0, None)),
                in_axes=(0, None, None, 0),
            )(rng_Y, rng_seed, all_y_target, all_query_idx)
            loss, log_dict, prediction = jax.vmap(
                jax.vmap(loss_fn.get_loss_from_input, in_axes=(None, 0, 0)),
                in_axes=(None, 0, 0),
            )(params, input, target)
            log_dict["loss"] = loss
            log_dict["prediction"] = prediction
            log_dict["target"] = target[-1]

            for k, v in log_dict.items():
                if k not in all_log_dict_tmp:
                    all_log_dict_tmp[k] = []
                all_log_dict_tmp[k].append(v)

        for k, v in all_log_dict_tmp.items():
            if k not in all_log_dict:
                all_log_dict[k] = []
            all_log_dict[k].append(jnp.stack(v))

    for k, v in all_log_dict.items():
        all_log_dict[k] = jnp.stack(v)

    return all_log_dict


def get_model():
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


def train():
    if USE_WANDB:
        run = wandb.init()
        cfg.update({k: v for (k, v) in wandb.config.items() if type(v) != dict})
        wandb.config.update(cfg)
        run_id = run.id
    else:
        run_id = f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        print(f"Running without wandb, run_id: {run_id}")


    # create the transformer model
    model = get_model()

    train_param = cfg["GD_PARAM"]
    if train_param["scheduler"] == "cosine":
        learning_rate = optax.join_schedules(
            [
                optax.linear_schedule(
                    0, train_param["lr"], train_param["warmup_steps"]
                ),
                optax.cosine_decay_schedule(
                    train_param["lr"],
                    cfg["num_steps"] - train_param["warmup_steps"],
                    alpha=train_param["lr"] * train_param["lr_alpha"],
                ),
            ],
            boundaries=[train_param["warmup_steps"]],
        )
    elif train_param["scheduler"] == "warmup":
        learning_rate = optax.linear_schedule(
            0, train_param["lr"], train_param["warmup_steps"]
        )
    else:
        learning_rate = train_param["lr"]

    gd_optimizer = optax.inject_hyperparams(
        lambda lr: optax.chain(
            optax.clip_by_global_norm(train_param["grad_norm_clip"]),
            optax.adamw(
                lr,
                weight_decay=train_param["weight_decay"],
                b1=train_param["betas"][0],
                b2=train_param["betas"][1],
                eps=train_param["eps"],
            ),
        )
    )(lr=learning_rate)

    loss_fn = AssociativeRecallLoss(model, cfg)
    trainer = TrainerGD(model, gd_optimizer, loss_fn, cfg)

    t = 0
    while t < cfg["num_steps"] // cfg["num_jit_batches"]:
        log_metric = trainer.train_iter(cfg["num_jit_batches"])
        eval_metric = loss_fn.eval_fn(trainer.get_params(), 10)


        if cfg["debug"]:
            # log_metric is a dict of arrays, shape (num_jit_batches, ...)
            # Get the last batch's predictions and targets to print
            pred_to_log = log_metric["pred_to_log"]
            y_to_log = log_metric["y_to_log"]
            input_tokens = log_metric["input_tokens"]
            # print("Sample predictions:", pred_to_log)
            # print("Sample targets:", y_to_log)
            pred_to_log_np = np.array(pred_to_log)
            y_to_log_np = np.array(y_to_log)
            input_tokens_np = np.array(input_tokens)

            # add bce loss to third column
            loss = custom_sigmoid_binary_cross_entropy(
                pred_to_log_np[:, :1], y_to_log_np[:, :1]
            )
            loss = np.array(loss.squeeze(1))  # Remove the size-1 dimension at axis -2

            # make input tokens a numpy array
            df = pd.DataFrame(
                {
                    "pred": [row for row in pred_to_log_np[:, 0].squeeze(1)],
                    "target": [row for row in y_to_log_np[:, 0].squeeze(1)],
                    "loss": [row[0] for row in loss.mean(axis=2)],
                    "input_tokens": [row for row in input_tokens_np[:, 0]],
                }
            )
            # save target and input_tokens to csv
            # if t == cfg['num_steps'] // cfg['num_jit_batches'] - 1:
            #     df.to_csv(f"checkpoints/associative_recall_{run.id}_{t}_io.csv", index=False)


        log_dict = {}
        log_dict.update({k: v.mean().item() for k, v in log_metric.items()})
        log_dict.update({k + "_eval": v.mean().item() for k, v in eval_metric.items()})
        if USE_WANDB:
            wandb.log(log_dict)
        print(f"Step {t + 1}/{cfg['num_steps'] // cfg['num_jit_batches']} - loss: {log_dict['loss']:.4f}")
        if log_dict["loss"] < 0.001:
            print("Loss is low enough, stopping training.")
            break
        t += 1

    log_dict = {}
    train_metric = loss_fn.eval_fn(trainer.get_params(), 1000, eval_on_train=True)
    eval_metric = loss_fn.eval_fn(trainer.get_params(), 1000, eval_on_train=False)

    log_dict.update(train_metric)
    log_dict.update({k + "_eval": v for k, v in eval_metric.items()})
    if USE_WANDB:
        wandb.log({"final_" + k: v for k, v in log_dict.items()})

    print(f"Saving to {run_id}")
    # np.save(f"checkpoints/associative_recall_{run_id}_q{cfg['p']}_c{cfg['num_token']}", (cfg, trainer.get_params()))
    # adv_log_dict = save_advanced_log(cfg, loss_fn, trainer.get_params())
    # np.save(f"checkpoints/associative_recall_{run_id}_log_dict", adv_log_dict)

if __name__ == "__main__":
    train()
