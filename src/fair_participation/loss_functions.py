from typing import Callable, Any

import jax.numpy as jnp
from jax import value_and_grad
from jax.typing import ArrayLike, Array


def fairness_disparity(rho: ArrayLike) -> Any:
    """
    Assumed to be symmetric.

    :param: rho: array of participation rates indexed by g
    :return: violation of fairness constraint
    """
    return jnp.var(rho) - 0.01


def value_and_grad_total_loss(
    rho_fn: Callable,  # vector -> vector
    group_sizes: ArrayLike,
) -> Callable:
    def _total_loss(loss: ArrayLike, loss_rho: ArrayLike) -> Array:
        """
        Maps loss [vector] x  loss_rho [vector] -> total loss [scalar].
        :param loss:
        :param loss_rho:
        :return:
        """
        rho = rho_fn(loss_rho)
        return jnp.sum(loss * rho * group_sizes)

    _vg_total_loss = value_and_grad(_total_loss, argnums=0)

    # only takes gradient wrt first loss, not rho(loss)
    def vg_total_loss(loss: ArrayLike) -> tuple[Array, Array]:
        return _vg_total_loss(loss, loss)

    return vg_total_loss


def fair_lpu_linear_fn(
    value_and_grad_loss_fn: Callable,  # scalar -> (vector, vector)
    rho_fn: Callable,  # vector -> vector
    group_sizes: ArrayLike,
) -> Callable:
    vg_total_loss = value_and_grad_total_loss(rho_fn, group_sizes)

    # callable to project grad
    def _disparity_loss(loss: ArrayLike) -> Array:
        rho = rho_fn(loss)
        return fairness_disparity(rho)

    vg_disparity_loss = value_and_grad(_disparity_loss)

    def _projected_fairness_grad(loss: ArrayLike) -> Array:
        # dH/dl
        grad_disp_loss = vg_disparity_loss(loss)[1]
        # dl/dtheta will give tangent space, as theta is on frontier
        _, tangent = value_and_grad_loss_fn(loss)
        unit_tangent = tangent / jnp.linalg.norm(tangent)
        # proj_{tangent space} dH/dl
        return jnp.dot(grad_disp_loss, unit_tangent) * unit_tangent

    def _fair_lpu_linear(loss: ArrayLike, alpha: float) -> Array:
        """
        Maps loss [vector] x alpha [float] to estimate of linear term .
        :param loss:
        :param alpha:
        :return:
        """
        _, grad_loss = vg_total_loss(loss)
        proj_fairness_grad = _projected_fairness_grad(loss)
        # TODO needs a zero check
        rho = rho_fn(loss)
        current_fairness = fairness_disparity(rho)
        lambda_estimate = jnp.max(
            0.0,
            alpha * current_fairness
            - jnp.dot(grad_loss, proj_fairness_grad)
            / jnp.dot(proj_fairness_grad, proj_fairness_grad),
        )
        return grad_loss + jnp.dot(lambda_estimate * proj_fairness_grad)

    return _fair_lpu_linear
