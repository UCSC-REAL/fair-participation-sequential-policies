from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.optimization import solve_qp
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

    # unit normals
    normals = loss_hull.equations[:, :-1]
    offsets = loss_hull.equations[:, -1]

    def _fair_lpu_linear(loss: ArrayLike) -> Array:
        """
        Maps loss [vector] x alpha [float] to estimate of linear term.
        :param loss: Current loss vector.
        :param alpha: Penalty parameter.
        :return: Estimate of linear term.
        """
        vgs = values_and_grads(loss)
        # Find active facet
        facet_dists = jnp.abs(jnp.dot(normals, loss) + offsets)
        facet_ix = jnp.argmin(facet_dists)
        unit_normal = normals[facet_ix]
        # Subtract off normal component of gradient if we are on a facet
        g = vgs["grad_disparity_loss"]
        is_on_facet = facet_dists[facet_ix] < 1e-6
        proj_fairness_grad = g - is_on_facet * jnp.dot(g, unit_normal) * unit_normal
        # TODO needs a zero check?
        lambda_estimate = jnp.max(
            0.0,
            alpha * vgs["disparity"]
            - jnp.dot(vgs["grad_total_loss"], proj_fairness_grad)
            / jnp.dot(proj_fairness_grad, proj_fairness_grad),
        )
        return vgs["grad_total_loss"] + jnp.dot(lambda_estimate * proj_fairness_grad)

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

    fair_lpu_linear = jit(fair_lpu_linear_fn(loss_hull, values_and_grads, alpha))

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


def fair_lpu_grad_step(
    values_and_grads: Callable,
    loss_hull: ConvexHull,
    eta: float,
) -> Callable[[ArrayLike], StateInfo]:
    raise NotImplementedError
