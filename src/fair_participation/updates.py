from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike, Array

from fair_participation.optimization import solve_qp
from fair_participation.fairness_functions import fair_lpu_linear_fn

# Generally, have to jit here because of cvxpy
StateUpdate = tuple[Array, Array, float, float]


def rrm_step(
    values_and_grads: Callable,
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
) -> Callable[[ArrayLike], StateUpdate]:
    """
    Exactly solves the RRM problem:
        min_l Sum_g (s_g * l_g * rho_g^t)
        s.t. l in loss_hull

    :param rho_fn:
    :param group_sizes:
    :param loss_hull:
    :return:
    """

    def _step(loss: ArrayLike) -> StateUpdate:
        vgs = values_and_grads(loss)
        rho = vgs["rho"]
        linear_term = rho * group_sizes
        opt_loss, _ = solve_qp(rho, linear_term, loss_hull)
        opt_vgs = values_and_grads(opt_loss)
        return (
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _step


def rrm_grad_step(
    values_and_grads: Callable,
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
    eta: float,
) -> Callable[[ArrayLike], StateUpdate]:
    """
    Performs a single gradient step on the RRM problem:
        l_{t+1} = l_t - eta * grad_x L(x, rho_t)|_{x=l_t}

    :param rho_fn:
    :param group_sizes:
    :param loss_hull:
    :param eta:
    :return:
    """

    @jit
    def _step(loss: ArrayLike) -> Array:
        # gets grad_x L(x, rho(l_t))|_{x=l_t}
        vgs = values_and_grads(loss)
        return loss - eta * vgs["grad_total_loss"]

    # TODO make jittable
    def _projected_step(loss: ArrayLike) -> StateUpdate:
        new_loss = _step(loss)
        opt_loss, _ = solve_qp(jnp.zeros_like(new_loss), loss_hull, (1.0, new_loss))
        opt_vgs = values_and_grads(opt_loss)
        return (
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _projected_step


def fair_lpu_step(
    value_and_grad_loss: Callable,
    values_and_grads: Callable,
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
    eta: float,
) -> Callable[[ArrayLike], StateUpdate]:
    """
    Exactly solves the FairLPU problem QP.

    :param value_and_grad_loss_fn:
    :param rho_fn:
    :param group_sizes:
    :param loss_hull:
    :param eta:
    :return:
    """
    fair_lpu_linear = jit(fair_lpu_linear_fn(value_and_grad_loss, values_and_grads))

    def _step(loss: ArrayLike) -> StateUpdate:
        linear_weights = fair_lpu_linear(loss)
        quadratic = (1.0 / (2.0 * eta), loss)
        opt_loss, _ = solve_qp(linear_weights, loss_hull, quadratic=quadratic)
        opt_vgs = values_and_grads(opt_loss)
        return (
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _step
