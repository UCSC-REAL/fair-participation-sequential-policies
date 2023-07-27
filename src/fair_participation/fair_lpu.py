from typing import Callable

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike, Array

from fair_participation.optimization import solve_qp
from fair_participation.state import StateInfo


def fair_lpu_linear_fn(
    value_and_grad_loss: Callable,
    values_and_grads: Callable,
) -> Callable:
    """
    Returns callable to compute linear term of FairLPU QP subproblem.
    :param value_and_grad_loss: Callable that returns loss and gradient.
    :param values_and_grads: Callable that returns commonly used values and gradients.
    :return: Callable.
    """

    def _fair_lpu_linear(loss: ArrayLike, alpha: float) -> Array:
        """
        Maps loss [vector] x alpha [float] to estimate of linear term.
        :param loss: Current loss vector.
        :param alpha: Penalty parameter.
        :return: Estimate of linear term.
        """
        vgs = values_and_grads(loss)
        # dl/dtheta will give tangent space, as theta is on frontier
        _, tangent = value_and_grad_loss(loss)
        unit_tangent = tangent / jnp.linalg.norm(tangent)
        proj_fairness_grad = (
            jnp.dot(vgs["grad_disparity_loss"], unit_tangent) * unit_tangent
        )

        # TODO needs a zero check?
        lambda_estimate = jnp.max(
            0.0,
            alpha * vgs["disparity"]
            - jnp.dot(vgs["grad_total_loss"], proj_fairness_grad)
            / jnp.dot(proj_fairness_grad, proj_fairness_grad),
        )
        return vgs["grad_total_loss"] + jnp.dot(lambda_estimate * proj_fairness_grad)

    return _fair_lpu_linear


def fair_lpu_step(
    value_and_grad_loss: Callable,
    values_and_grads: Callable,
    loss_hull: ArrayLike,
    eta: float,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the FairLPU subproblem.

    :param value_and_grad_loss: Callable that returns loss and gradient.
    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param loss_hull: Array of losses that define the convex hull.
    :param eta: Learning rate.
    :return: Callable that performs a single update step.
    """

    # TODO jit alpha
    fair_lpu_linear = jit(fair_lpu_linear_fn(value_and_grad_loss, values_and_grads))

    def _step(loss: ArrayLike) -> StateInfo:
        linear_weights = fair_lpu_linear(loss)
        opt_loss, _ = solve_qp(
            w=linear_weights, hull=loss_hull, gamma=1.0 / (2.0 * eta), x0=loss
        )
        opt_vgs = values_and_grads(opt_loss)
        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _step
