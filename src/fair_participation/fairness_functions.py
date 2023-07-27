from typing import Callable

import jax.numpy as jnp
from jax.typing import ArrayLike, Array


def fair_lpu_linear_fn(
    value_and_grad_loss: Callable,
    values_and_grads: Callable,
) -> Callable:
    def _fair_lpu_linear(loss: ArrayLike, alpha: float) -> Array:
        """
        Maps loss [vector] x alpha [float] to estimate of linear term .
        :param loss:
        :param alpha:
        :return:
        """
        vgs = values_and_grads(loss)
        # dl/dtheta will give tangent space, as theta is on frontier
        _, tangent = value_and_grad_loss(loss)
        unit_tangent = tangent / jnp.linalg.norm(tangent)
        # proj_{tangent space} dH/dl
        proj_fairness_grad = (
            jnp.dot(vgs["grad_disparity_loss"], unit_tangent) * unit_tangent
        )

        # TODO needs a zero check
        lambda_estimate = jnp.max(
            0.0,
            alpha * vgs["disparity"]
            - jnp.dot(vgs["grad_total_loss"], proj_fairness_grad)
            / jnp.dot(proj_fairness_grad, proj_fairness_grad),
        )
        return vgs["grad_total_loss"] + jnp.dot(lambda_estimate * proj_fairness_grad)

    return _fair_lpu_linear
