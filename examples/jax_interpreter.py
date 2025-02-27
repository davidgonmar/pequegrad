import jax
import jax.numpy as jnp
import numpy as np
from jax import grad
from jax.tree_util import tree_flatten, tree_unflatten
import flax.linen as nn
import pequegrad.jax as pg_jax

DEVICE = "cuda"


class FlaxMLP(nn.Module):
    features: list

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.features[-1])(x)
        return x


mlp_model = FlaxMLP(features=[20, 1])


def loss(params, x, y):
    preds = mlp_model.apply(params, x)
    return jnp.mean((preds - y) ** 2)


grad_loss = grad(loss)


@pg_jax.jax_jit(DEVICE)
def update(params, x, y, lr):
    grads = grad_loss(params, x, y)
    flat_params, tree_def = tree_flatten(params)
    flat_grads, _ = tree_flatten(grads)
    new_flat_params = [p - lr * g for p, g in zip(flat_params, flat_grads)]
    return tree_unflatten(tree_def, new_flat_params)


input_dim = 10
key = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, input_dim))
params = mlp_model.init(key, dummy_input)


x = np.random.randn(100, input_dim)
true_W = np.random.randn(input_dim, 1)
y = x @ true_W + 0.1 * np.random.randn(100, 1)


loss_jit = pg_jax.jax_jit(DEVICE)(loss)
for i in range(100):
    params = update(params, x, y, 0.001)
    if i % 10 == 0:
        current_loss = loss_jit(params, x, y)
        print(f"Step {i:3d}, Loss: {current_loss}")
