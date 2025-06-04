from functools import partial
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax


class GDState(NamedTuple):
    params: hk.Params
    opt_state: optax.OptState


class TrainerGD(object):
    def __init__(self, model, optimizer, loss_fn, cfg):
        self.rng, model_init_rng = jax.random.split(
            jax.random.PRNGKey(cfg["seed"]), 2
        )
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.cfg = cfg

        params = self.model.init(model_init_rng, loss_fn.dummy_data())

        num_params = jax.tree_util.tree_reduce(lambda s, n: n.size + s, params, 0)
        print(f"Number of parameters: {num_params / 1e3:.02f}k")

        opt_state = self.optimizer.init(params)

        self.gd_state = GDState(params=params, opt_state=opt_state)

    @partial(jax.jit, static_argnums=(0))
    def _do_step(self, state, rng):
        params, opt_state = state
        (loss, auxs), grads = jax.value_and_grad(self.loss_fn, has_aux=True)(
            params, rng
        )
        grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x, copy=True, nan=0.0, posinf=0, neginf=0), grads)

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        log_dict = dict()
        log_dict["loss"] = loss
        log_dict.update({k: v.mean() for k, v in auxs.items()})

        return GDState(params, opt_state), log_dict

    @partial(jax.jit, static_argnums=(0, 3))
    def _do_steps(self, gd_state, epoch_rng, num_steps):
        gd_state, log_metric = jax.lax.scan(
            self._do_step,
            gd_state,
            (jax.random.split(epoch_rng, num=num_steps)),
        )
        return gd_state, log_metric

    def train_iter(self, num_steps):
        self.rng, epoch_rng = jax.random.split(self.rng)

        self.gd_state, log_metric = self._do_steps(
            self.gd_state, epoch_rng, num_steps
        )

        return log_metric

    def get_params(self):
        return self.gd_state.params
