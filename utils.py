import math

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from scipy.stats import qmc


class PytreeReshaper:
    def __init__(self, tree_shapes):
        self.shapes, self.treedef = jtu.tree_flatten(
            tree_shapes, is_leaf=lambda x: (isinstance(x, tuple) and all(isinstance(v, int) for v in x))
        )
        sizes = [math.prod(shape) for shape in self.shapes]

        self.split_indeces = list(np.cumsum(sizes)[:-1])
        self.num_elements = sum(sizes)

    def __call__(self, array_flat):
        arrays_split = jnp.split(array_flat, self.split_indeces)
        arrays_reshaped = [a.reshape(shape) for a, shape in zip(arrays_split, self.shapes)]

        return jtu.tree_unflatten(self.treedef, arrays_reshaped)

    @staticmethod
    def flatten(pytree):
        return jnp.concatenate([jnp.ravel(e) for e in jtu.tree_flatten(pytree)[0]])


def get_qmc_seeds(num, dim=1):
    sampler = qmc.Sobol(d=dim, scramble=True)
    # Generate points in [0, 1)
    points = sampler.random(num)
    # Convert to uint32 for PRNGKey
    int_points = (points * (2**32)).astype(np.uint32)
    # If dim > 1, you may want to combine dims into a single seed
    return [jax.random.PRNGKey(int(x[0])) for x in int_points]

class FakeStdin:
  def readline(self):
    return input()