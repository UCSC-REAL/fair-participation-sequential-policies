from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.optimization import solve_qp, proj_tangent_qp
from fair_participation.state import StateInfo


def fair_lpu_linear_fn(
    loss_hull: ConvexHull,
    values_and_grads: Callable,
    alpha: float,
) -> Callable:
    """
    Returns callable to compute linear term of FairLPU QP subproblem.
    # :param value_and_grad_loss: Callable that returns loss and gradient.
    :param loss_hull: ConvexHull object of loss vectors.
    :param values_and_grads: Callable that returns commonly used values and gradients.
    :return: Callable.
    """

    def _fair_lpu_linear(loss: ArrayLike) -> tuple[Array, float]:
        """
        Maps loss [vector] x alpha [float] to estimate of linear term.
        :param loss: Current loss vector.
        :param alpha: Penalty parameter.
        :return: Estimate of linear term.
        """
        vgs = values_and_grads(loss)

        loss_grad = vgs["full_deriv_total_loss"]
        g = vgs["grad_disparity_loss"]

        proj_fairness_grad = -proj_tangent_qp(loss, -g, loss_hull)

        # We assume that proj_fairness_grad is never 0 outside the feasible set
        # Therefore, jnp.dot(proj_fairness_grad, proj_fairness_grad) > 0 outside
        # feasible set
        lambda_estimate = jnp.max(
            jnp.array(
                [
                    0.0,
                    (alpha * vgs["disparity"] - jnp.dot(loss_grad, proj_fairness_grad))
                    / jnp.dot(proj_fairness_grad, proj_fairness_grad),
                ]
            )
        )
        return loss_grad + lambda_estimate * proj_fairness_grad, lambda_estimate

    return _fair_lpu_linear


# TODO check
def fair_lpu_step(
    values_and_grads: Callable,
    loss_hull: ConvexHull,
    eta: float,
    alpha: float,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the FairLPU subproblem.

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param loss_hull: ConvexHull object of loss vectors.
    :param eta: Learning rate.
    :return: Callable that performs a single update step.
    """

    fair_lpu_linear = fair_lpu_linear_fn(loss_hull, values_and_grads, alpha)

    def _step(loss: ArrayLike) -> StateInfo:
        linear_weights, lambda_estimate = fair_lpu_linear(loss)
        opt_loss, _ = solve_qp(
            w=linear_weights, hull=loss_hull, gamma=1.0 / (2.0 * eta), x0=loss
        )
        opt_vgs = values_and_grads(opt_loss)
        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
            lambda_estimate,
        )

    return _step
