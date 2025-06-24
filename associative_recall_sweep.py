
import os

import haiku as hk
import jax
import jax.numpy as jnp
# sys.path.append(os.path.abspath('../'))
import numpy as np
import optax

import wandb
from losses import AssociativeRecallLoss
from model_rng import CustomTransformer  #, Transformer
from trainer_gd import TrainerGD

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

hk.vmap.require_split_rng = False

N = 5

cfg = {
    "qmc": False,
    "seed": 42,
    "num_jit_batches": 100,
    "num_steps": 10000,  #30000,

    # Env    
    "num_token": 1,
    "target_size": N,
    "batch_size": 512,
    "num_train_seed": 0,  # total: N * 2**B

    "probabilistic": "random",
    "data_pooling": "lp",  #"lp" "mean"
    "p": 1,
    "num_seed": 30,
    "loss": "bce",  # "contrastive_ce" "contrastive_hinge" "bce" "bce_mse" "mse"

    "hardcoded_randomness": False,
    "mlp": False,
    "widening_factor": 1,
    "first_mlp": True,
    "seed_size": 2 * N,
    'reverse_block': True,

    # RNG model
    "num_layers": 2,
    "num_heads": 1,
    "kq_dim": N,
    "v_dim": N,
    "embed_dim": 8 * N,  # 32,
    "softmax": "none",  #"all",
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
        "warmup_steps": 333, #1000,
    },

}


from itertools import product


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


def train():
    run = wandb.init()
    cfg.update({k: v for (k, v) in wandb.config.items() if type(v) != dict})
    wandb.config.update(cfg)

    # create the transformer model
    model = hk.without_apply_rng(
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

        log_dict = {}
        # log_dict.update({k: v.mean().item() for k, v in log_metric.items()})
        log_dict.update({k + "_eval": v.mean().item() for k, v in eval_metric.items()})
        wandb.log(log_dict)
        print(f"Step {t}: {log_dict}")
        if log_dict.get("data_loss_eval", 1) < 0.001:
            print("Early stopping due to low eval loss")
            break
        t += 1

    log_dict = {}
    train_metric = loss_fn.eval_fn(trainer.get_params(), 1000, eval_on_train=True)
    eval_metric = loss_fn.eval_fn(trainer.get_params(), 1000, eval_on_train=False)
    log_dict.update(train_metric)
    log_dict.update({k + "_eval": v for k, v in eval_metric.items()})
    wandb.log({"final_" + k: v for k, v in log_dict.items()})

    print(f"Saving run {run.id}")
    np.save(f"checkpoints/associative_recall_{run.id}_{cfg['probabilistic']}_q{cfg['p']}_c{cfg['num_token']}", (cfg, trainer.get_params()))
    adv_log_dict = save_advanced_log(cfg, loss_fn, trainer.get_params())
    np.save(f"checkpoints/associative_recall_{run.id}_log_dict", adv_log_dict)


sweep_configuration = {
    'method': 'grid',
    'name': 'sweep_AR_token',
    'parameters': dict(
        probabilistic={'values': ["single_seed", "deterministic", "random"]},
        num_token={'values': [10, 12, 14, 16, 18, 20]},
        p={'values': [1, 16, 32, 100]},
        # seed={'values': [10, 11, 12, 13, 14]},
    )
}
sweep_id = wandb.sweep(sweep=sweep_configuration)
wandb.agent(sweep_id=sweep_id, function=train, count=10)
