from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike

from fair_participation.state import StateInfo
from fair_participation.optimization import solve_qp


def rrm_step(
    values_and_grads: Callable[[ArrayLike], dict],
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the RRM subproblem:
        min_l Sum_g (s_g * l_g * rho_g^t)
        s.t. l in loss_hull

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param group_sizes: Array of group sizes summing to 1.
    :param loss_hull: Array of losses that define the convex hull.
    :return: Callable that performs a single update step.
    """

    def _step(loss: ArrayLike) -> StateInfo:
        """
        RRM update step.
        :param loss: Current loss vector.
        :return: Dictionary of updated values.
        """
        vgs = values_and_grads(loss)
        rho = vgs["rho"]
        linear_term = rho * group_sizes
        opt_loss, _ = solve_qp(w=linear_term, hull=loss_hull)
        opt_vgs = values_and_grads(opt_loss)
        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _step


# TODO check
def rrm_grad_step(
    values_and_grads: Callable[[ArrayLike], dict],
    loss_hull: ArrayLike,
    eta: float,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that performs a single gradient step on the RRM problem:
        l_{t+1} = l_t - eta * grad_x L(x, rho_t)|_{x=l_t}

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param loss_hull: Array of losses that define the convex hull.
    :param eta: Learning rate.
    :return: Callable that performs a single update step.
    """

    @jit
    def _step(loss: ArrayLike) -> Array:
        """
        Gradient step on the RRM problem.
        :param loss: Current loss vector.
        :return: New loss vector.
        """
        # vgs["grad_total_loss"] =  grad_x L(x, rho(l_t))|_{x=l_t}
        vgs = values_and_grads(loss)
        return loss - eta * vgs["grad_total_loss"]

    # TODO make jittable
    def _projected_step(loss: ArrayLike) -> StateInfo:
        """
        Gradient step on the RRM problem, projected onto the convex hull.
        :param loss: Current loss vector.
        :return: Dictionary of updated values.
        """
        new_loss = _step(loss)
        opt_loss, _ = solve_qp(
            w=jnp.zeros_like(new_loss), hull=loss_hull, gamma=1.0, x0=new_loss
        )
        opt_vgs = values_and_grads(opt_loss)
        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _projected_step
