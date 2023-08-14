from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.optimization import solve_qp, proj_tangent_qp
from fair_participation.state import StateInfo

from fair_participation.utils import EPS


def fsep_linear_fn(
    loss_hull: ConvexHull,
    values_and_grads: Callable,
) -> Callable:
    """
    Returns callable to compute linear term of FSEP QP subproblem.
    # :param value_and_grad_loss: Callable that returns loss and gradient.
    :param loss_hull: ConvexHull object of loss vectors.
    :param values_and_grads: Callable that returns commonly used values and gradients.
    :return: Callable.
    """

    def _fsep_linear(loss: ArrayLike, alpha: float) -> tuple[Array, float]:
        """
        Maps loss [vector] x xi [float] to estimate of linear term.
        :param loss: Current loss vector.
        :param alpha: step size for lambda.
        :return: Estimate of linear term.
        """
        vgs = values_and_grads(loss)

        loss_grad = vgs["full_deriv_total_loss"]
        disparity_grad = vgs["grad_disparity_loss"]
        proj_disparity_grad = -proj_tangent_qp(loss, -disparity_grad, loss_hull)

        proj_disparity_grad_sq_norm = jnp.dot(proj_disparity_grad, proj_disparity_grad)

        # We assume that proj_disparity_grad is never 0 outside the feasible set
        # Therefore, jnp.dot(proj_disparity_grad, proj_disparity_grad) > 0 outside
        # feasible set, and if proj_disparity_grad IS 0, we know lambda_est should be 0
        lambda_estimate = jnp.max(
            jnp.array(
                [
                    0.0,
                    (alpha * vgs["disparity"] - jnp.dot(loss_grad, proj_disparity_grad))
                    / proj_disparity_grad_sq_norm,
                ]
            )
        ) * (proj_disparity_grad_sq_norm >= EPS)
        linear_weights = loss_grad + lambda_estimate * proj_disparity_grad
        # print(
        #     "inner product between effective grad and grad disp",
        #     jnp.dot(linear_weights, g),
        # )
        # print(
        #     "inner product between effective grad and grad LOSS",
        #     jnp.dot(linear_weights, loss_grad),
        # )
        print(
            "effective grad and PROJECTED disp grad",
            jnp.dot(linear_weights, proj_disparity_grad),
        )

        return linear_weights, lambda_estimate, proj_disparity_grad

    return _fsep_linear


def fsep_step(
    values_and_grads: Callable,
    loss_hull: ConvexHull,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the FSEP subproblem.

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param loss_hull: ConvexHull object of loss vectors.
    :return: Callable that performs a single update step.
    """

    fsep_linear = fsep_linear_fn(loss_hull, values_and_grads)

    def _step(loss: ArrayLike, rates: tuple[float]) -> StateInfo:
        """
        FSEP update step.
        :param loss: Current loss vector.
        :param rates: Learning rates.
        """
        eta, alpha = rates
        linear_weights, lambda_estimate, pdg = fsep_linear(loss, alpha)

        opt_loss, _ = solve_qp(
            w=linear_weights, hull=loss_hull, gamma=1.0 / (2.0 * eta), x0=loss
        )

        print(
            "inner product between actual update and PROJECTED disp grad",
            jnp.dot(opt_loss - loss, pdg),
        )

        opt_vgs = values_and_grads(opt_loss)
        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
            linear_weights,
            lambda_estimate,
        )

    return _step
