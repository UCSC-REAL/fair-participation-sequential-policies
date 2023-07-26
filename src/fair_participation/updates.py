from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike, Array

from fair_participation.loss_functions import (
    total_loss_fn,
    fair_lpu_linear_fn,
)
from fair_participation.optimization import solve_qp


def rrm_step(
    rho_fn: Callable,  # vector -> (vector, vector)
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
):
    """
    Exactly solves the RRM problem:
        min_l Sum_g (s_g * l_g * rho_g^t)
        s.t. l in loss_hull

    :param rho_fn:
    :param group_sizes:
    :param loss_hull:
    :return:
    """

    def _step(loss: ArrayLike) -> Array:
        rho = rho_fn(loss)
        linear_term = rho * group_sizes
        opt_loss, _ = solve_qp(rho, linear_term, loss_hull)
        return opt_loss

    return _step


def rrm_grad_step(
    rho_fn: Callable,  # vector -> vector
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
    eta: float,
):
    """
    Performs a single gradient step on the RRM problem:
        l_{t+1} = l_t - eta * grad_x L(x, rho_t)|_{x=l_t}

    :param rho_fn:
    :param group_sizes:
    :param loss_hull:
    :param eta:
    :return:
    """
    vg_total_augmented_loss = total_loss_fn(rho_fn, group_sizes)

    @jit
    def _step(loss: ArrayLike) -> Array:
        rho = rho_fn(loss)
        # gets grad_x L(x, rho(l_t))|_{x=l_t}
        _, g = vg_total_augmented_loss(loss, rho, 0.0)
        return loss - eta * g

    # TODO could do this better - makes it not jittable
    def _projected_step(loss: ArrayLike) -> Array:
        new_loss = _step(loss)
        # TODO could do this with 2 points?
        opt_loss, _ = solve_qp(jnp.zeros_like(new_loss), loss_hull, (1.0, new_loss))
        return opt_loss

    return _projected_step


def fair_lpu_step(
    value_and_grad_loss_fn: Callable,  # scalar -> (vector, vector)
    rho_fn: Callable,  # vector -> vector
    group_sizes: ArrayLike,
    loss_hull: ArrayLike,
    eta: float,
):
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

    def _step(loss: ArrayLike) -> Array:
        linear_weights = fair_lpu_linear(loss)
        quadratic = (1.0 / (2.0 * eta), loss)
        opt_loss, _ = solve_qp(linear_weights, loss_hull, quadratic=quadratic)
        return opt_loss

    return _step
