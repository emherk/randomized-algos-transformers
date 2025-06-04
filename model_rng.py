import dataclasses
import math
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp


class CustomMultiHeadAttention(hk.Module):
    def __init__(
            self,
            num_heads: int,
            kq_dim: int,
            v_dim: int,
            embed_dim: int,
            w_init_var: float,
            num_projection: int,
            softmax=True,
    ):

        super().__init__()
        self.num_heads = num_heads
        self.kq_dim = kq_dim
        self.v_dim = v_dim or kq_dim
        self.embed_dim = embed_dim
        self.softmax = softmax
        w_init = partial(
            hk.initializers.VarianceScaling,
            distribution="truncated_normal",
            fan_in_axes=[-2],
        )

        self.w_q = hk.get_parameter(
            "w_q", [num_heads, embed_dim, kq_dim], init=w_init(w_init_var)
        )
        self.w_k = hk.get_parameter(
            "w_k", [num_heads, embed_dim, kq_dim], init=w_init(w_init_var)
        )
        self.w_v = hk.get_parameter(
            "w_v", [num_heads, embed_dim, v_dim], init=w_init(w_init_var)
        )
        self.w_o = hk.get_parameter(
            "w_o",
            [num_heads, v_dim, embed_dim],
            init=w_init(w_init_var / num_projection / (num_heads)),
        )

    def __call__(self, x_query, x_key, x_value, mask):
        merge_qk = lambda x, y: jnp.einsum("HIO,HiO->HIi", x, y)
        merge_vo = lambda x, y: jnp.einsum("HIO,HOi->HIi", x, y)
        w_qk = merge_qk(self.w_q, self.w_k)
        w_vo = merge_vo(self.w_v, self.w_o)

        attention = jnp.einsum("HIi,...TI,...ti->...HTt", w_qk, x_query, x_key)
        if self.softmax:
            attention = attention / math.sqrt(self.kq_dim)
            attention = jnp.where(mask, attention, -1e30)
            attention = jax.nn.softmax(attention, axis=-1)
        else:
            attention = jnp.where(mask, attention, 0)
        output = jnp.einsum("HIi,...tI->...Hti", w_vo, x_value)
        output = jnp.einsum("...HTt,...HtI->...TI", attention, output)
        return output


def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    # ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="layer_norm")
    ln = hk.RMSNorm(axis=-1, create_scale=False, name="rms_norm")
    return ln(x)


@dataclasses.dataclass
class CustomTransformer(hk.Module):
    """A transformer stack."""

    out_dim: int
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    num_heads: int
    kq_dim: int
    v_dim: int
    embed_dim: int
    softmax: str

    positional_embedding: bool = False
    w_init_var: float = 1

    first_embedding_init_var: float = 0.1

    lin_with_bias = True
    mlp: bool = False
    widening_factor: int = 4
    layer_norm: bool = False
    last_mlp: bool = False
    first_mlp: bool = False
    reverse_block: bool = False

    def __call__(self, x) -> jax.Array:  # [T, D]

        tokens, mask = x
        """Transforms input embedding sequences to output embedding sequences."""
        num_projection = self.num_layers
        w_init = partial(
            hk.initializers.VarianceScaling,
            distribution="truncated_normal",
            fan_in_axes=[-2],
        )

        if isinstance(tokens, tuple):
            tokens = jnp.concatenate([*tokens], axis=-1)

        seq_len = tokens.shape[0]

        token_embedding_map = hk.Linear(
            self.embed_dim, w_init=w_init(self.first_embedding_init_var), with_bias=self.lin_with_bias
        )
        input_embeddings = token_embedding_map(tokens)

        if self.positional_embedding:
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter(
                'positional_embeddings', [seq_len, self.embed_dim], init=embed_init)
            input_embeddings = input_embeddings + positional_embeddings

        h = input_embeddings

        if self.first_mlp:
            #             h = h + hk.Linear(self.embed_dim)(jax.nn.gelu(hk.Linear(self.embed_dim)(h)))
            h = hk.Linear(self.embed_dim)(jax.nn.gelu(h))

        for layer in range(self.num_layers):

            if self.reverse_block and self.mlp:
                dense_block = hk.Sequential([
                    hk.Linear(self.widening_factor * self.embed_dim, w_init=w_init(2)),
                    jax.nn.gelu,
                    hk.Linear(self.embed_dim, w_init=w_init(1 / num_projection)),
                ])
                h_norm = h
                if self.layer_norm:
                    h_norm = _layer_norm(h)
                h_dense = dense_block(h_norm)
                h = h + h_dense

            attn_block = CustomMultiHeadAttention(
                self.num_heads,
                self.kq_dim,
                self.v_dim,
                self.embed_dim,
                softmax=self.softmax == "all"
                        or (self.softmax == "first" and layer == 0),
                w_init_var=self.w_init_var,
                num_projection=num_projection,
            )
            h_norm = h
            if self.layer_norm:
                h_norm = _layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)
            h = h + h_attn

            if self.mlp and not self.reverse_block:
                dense_block = hk.Sequential([
                    hk.Linear(self.widening_factor * self.embed_dim, w_init=w_init(2)),
                    jax.nn.gelu,
                    hk.Linear(self.embed_dim, w_init=w_init(1 / num_projection)),
                ])
                h_norm = h
                if self.layer_norm:
                    h_norm = _layer_norm(h)
                h_dense = dense_block(h_norm)
                h = h + h_dense

        if self.last_mlp:
            h = jax.nn.gelu(hk.Linear(self.embed_dim)(h))

        out = hk.Linear(self.out_dim, w_init=w_init(1), with_bias=self.lin_with_bias)(h)

        return out


@dataclasses.dataclass
class TwoStageTransformer(hk.Module):
    """A transformer stack."""

    out_dim: int
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    num_heads: int
    kq_dim: int
    v_dim: int
    embed_dim: int
    softmax: str

    positional_embedding: bool = False
    w_init_var: float = 1

    first_embedding_init_var: float = 0.1

    lin_with_bias = True
    mlp: bool = False
    widening_factor: int = 4

    def __call__(self, x) -> jax.Array:  # [T, D]

        (tokens, seed), mask = x
        """Transforms input embedding sequences to output embedding sequences."""
        num_projection = self.num_layers
        w_init = partial(
            hk.initializers.VarianceScaling,
            distribution="truncated_normal",
            fan_in_axes=[-2],
        )

        seq_len = tokens.shape[0]

        token_embedding_map = hk.Linear(
            self.embed_dim, w_init=w_init(self.first_embedding_init_var), with_bias=self.lin_with_bias
        )
        input_embeddings = token_embedding_map(tokens)

        if self.positional_embedding:
            embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
            positional_embeddings = hk.get_parameter(
                'positional_embeddings', [seq_len, self.embed_dim], init=embed_init)
            input_embeddings = input_embeddings + positional_embeddings

        rng_embedding = CustomTransformer(
            out_dim=self.embed_dim,
            num_layers=2,
            num_heads=1,
            kq_dim=self.embed_dim,
            v_dim=self.embed_dim,
            embed_dim=self.embed_dim,
            softmax="none")((seed, mask))

        h = input_embeddings + rng_embedding
        for layer in range(self.num_layers):

            if self.mlp:
                dense_block = hk.Sequential([
                    hk.Linear(self.widening_factor * self.embed_dim, w_init=w_init(2)),
                    jax.nn.gelu,
                    hk.Linear(self.embed_dim, w_init=w_init(1 / num_projection)),
                ])
                h_norm = h
                h_dense = dense_block(h_norm)
                h = h + h_dense

            attn_block = CustomMultiHeadAttention(
                self.num_heads,
                self.kq_dim,
                self.v_dim,
                self.embed_dim,
                softmax=self.softmax == "all"
                        or (self.softmax == "first" and layer == 0),
                w_init_var=self.w_init_var,
                num_projection=num_projection,
            )
            h_norm = h
            h_attn = attn_block(h_norm, h_norm, h_norm, mask=mask)
            h = h + h_attn

        out = hk.Linear(self.out_dim, w_init=w_init(1), with_bias=self.lin_with_bias)(h)

        return out
