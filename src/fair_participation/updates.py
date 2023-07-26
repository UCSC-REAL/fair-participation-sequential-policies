from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike, Array

from fair_participation.loss_functions import (
    value_and_grad_total_loss,
    fair_lpu_linear_fn,
    fairness_disparity,
)
from fair_participation.optimization import solve_qp

# Generally, have to jit here because of cvxpy
StateUpdate = tuple[Array, Array, float, float]


def rrm_step(
    rho_fn: Callable,  # vector -> (vector, vector)
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

    vg_total_loss = jit(value_and_grad_total_loss(rho_fn, group_sizes))
    rho_fn = jit(rho_fn)

    def _step(loss: ArrayLike) -> StateUpdate:
        rho = rho_fn(loss)
        linear_term = rho * group_sizes
        opt_loss, _ = solve_qp(rho, linear_term, loss_hull)
        opt_rho = rho_fn(opt_loss)
        total_loss, _ = vg_total_loss(opt_loss)
        return opt_loss, opt_rho, total_loss, fairness_disparity(opt_rho)

    return _step


def rrm_grad_step(
    rho_fn: Callable,  # vector -> vector
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
    vg_total_loss = value_and_grad_total_loss(rho_fn, group_sizes)

    @jit
    def _step(loss: ArrayLike) -> Array:
        # gets grad_x L(x, rho(l_t))|_{x=l_t}
        _, g = vg_total_loss(loss)
        return loss - eta * g

    # TODO make jittable
    def _projected_step(loss: ArrayLike) -> StateUpdate:
        new_loss = _step(loss)
        opt_loss, _ = solve_qp(jnp.zeros_like(new_loss), loss_hull, (1.0, new_loss))
        opt_rho = rho_fn(opt_loss)  # already jitted
        total_loss, _ = vg_total_loss(opt_loss)  # already jitted
        return opt_loss, opt_rho, total_loss, fairness_disparity(opt_rho)

    return _projected_step


def fair_lpu_step(
    value_and_grad_loss_fn: Callable,  # scalar -> (vector, vector)
    rho_fn: Callable,  # vector -> vector
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
    fair_lpu_linear = jit(
        fair_lpu_linear_fn(value_and_grad_loss_fn, rho_fn, group_sizes)
    )
    vg_total_loss = jit(value_and_grad_total_loss(rho_fn, group_sizes))

    def _step(loss: ArrayLike) -> StateUpdate:
        linear_weights = fair_lpu_linear(loss)
        quadratic = (1.0 / (2.0 * eta), loss)
        opt_loss, _ = solve_qp(linear_weights, loss_hull, quadratic=quadratic)
        rho = rho_fn(opt_loss)
        total_loss, _ = vg_total_loss(opt_loss)
        return opt_loss, rho, total_loss, fairness_disparity(rho)

    return _step
