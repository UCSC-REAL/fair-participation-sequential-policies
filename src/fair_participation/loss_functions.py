import jax.numpy as jnp
from typing import Callable
from jax.typing import ArrayLike, Array


def fairness_disparity(rho: ArrayLike) -> Array:
    """
    Assumed to be symmetric. Function of rho.

    Get violation of fairness constraint.

    Args:
        rho: array of participation rates indexed by g
    """
    return jnp.var(rho) - 0.01


def total_loss(
    loss
    disparity_fn: Callable
) -> Callable:
    """

    :param disparity_fn:
    :return:
    """
    quad = not method.endswith("grad")
    method = method.split("_")[0]

    # If method is "perf" or "rrm", then g and lambda_ are zero (no fairness constraint)

    def total_loss(loss: ArrayLike) -> Array:
        rho, _ = value_and_grad_rho_fn(loss)
        return jnp.sum(loss * rho * group_sizes)
