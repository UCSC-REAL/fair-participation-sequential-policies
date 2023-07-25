from typing import Any, Callable

import jax.numpy as jnp
from jax import value_and_grad, Array
from jax.typing import ArrayLike, Array

from fair_participation.opt import solve_rrm_qp


def fairness_disparity(rho: ArrayLike) -> Any:
    """
    Assumed to be symmetric.

    :param: rho: array of participation rates indexed by g
    :return: violation of fairness constraint
    """
    return jnp.var(rho) - 0.01


# disparity_fns = {
#     "RRM": lambda rho: 0.0,
#     "LPU": lambda rho: 0.0,
#     "FairLPU": fairness_disparity,
# }
# grad (bool) => fn
# qp_solve_fn = {
#     False: solve_qp,
#     True: direct_qp_step,
# }


def create_total_augmented_loss(
    value_and_grad_rho_fn: Callable,  # vector -> (vector, vector)
    group_sizes: ArrayLike,
) -> Callable:
    def total_augmented_loss(
        loss: ArrayLike, loss_rho: ArrayLike, lambda_: float
    ) -> Array:
        """
        Maps loss [vector] x  loss_rho [vector] x lambda_ [scalar] -> total loss [scalar].
        Can set lambda_ = 0 to have no fairness disparity involved.
        :param loss:
        :param loss_rho:
        :param lambda_:
        :return:
        """
        rho, _ = value_and_grad_rho_fn(loss_rho)
        return jnp.sum(loss * rho * group_sizes) + lambda_ * fairness_disparity(
            rho
        )  # TODO uses second one

    # only takes gradient wrt first loss, not rho(loss)
    vg_total_augmented_loss = value_and_grad(total_augmented_loss, argnums=0)
    return vg_total_augmented_loss


def rrm_step(
    value_and_grad_rho_fn: Callable,  # vector -> (vector, vector)
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
):
    """
    Exactly solves the RRM problem:
        min_l Sum_g (s_g * l_g * rho_g^t)
        s.t. l in loss_hull

    :param value_and_grad_rho_fn:
    :param group_sizes:
    :param loss_hull:
    :return:
    """

    def _step(loss: ArrayLike) -> Array:
        rho, _ = value_and_grad_rho_fn(loss)
        return solve_rrm_qp(rho, group_sizes, loss_hull)

    return _step


def rrm_grad_step(
    value_and_grad_rho_fn: Callable,  # vector -> (vector, vector)
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
    eta: float,
):
    """
    Performs a single gradient step on the RRM problem:
        l_{t+1} = l_t - eta * grad_x L(x, rho_t)|_{x=l_t}
        # TODO need to project back onto loss_hull

    :param value_and_grad_rho_fn:
    :param group_sizes:
    :param loss_hull:
    :param eta:
    :return:
    """
    vg_total_augmented_loss = create_total_augmented_loss(
        value_and_grad_rho_fn, group_sizes
    )

    # TODO jit
    def _step(loss: ArrayLike) -> Array:
        rho, _ = value_and_grad_rho_fn(loss)
        # Gets grad_x L(x, rho(l_t))|_{x=l_t}
        _, g = vg_total_augmented_loss(loss, rho, 0.0)
        return loss - eta * g

    return _step
