from typing import Callable

import cvxpy as cvx
import jax.numpy as jnp
from cvxpy import Problem, Minimize, Variable, Constant
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.base_logger import logger
from fair_participation.state import StateInfo


def cpg_step(
    values_and_grads: Callable,
    loss_hull: ConvexHull,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the cpg subproblem.

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param loss_hull: ConvexHull object of loss vectors.
    :return: Callable that performs a single update step.
    """

    def _step(loss: ArrayLike, rates: tuple[float, float]) -> StateInfo:
        """
        CPG update step.
        :param loss: Current loss vector.
        :param rates: Tuple of learning rates (eta, alpha).
        """
        eta, alpha = rates
        logger.debug("#" * 80)

        vgs = values_and_grads(loss)
        loss_grad = vgs["full_deriv_total_loss"]
        disparity = vgs["disparity"]
        disparity_grad = vgs["grad_disparity_loss"]

        n, d = loss_hull.points.shape
        xi = Variable(n)
        x = Variable(d)
        constraints = [
            cvx.sum(xi) == 1,
            xi >= Constant(0.0),  # for type hinting
            x == xi @ loss_hull.points,
            (x - loss) @ (-disparity_grad) >= alpha * disparity,
        ]
        obj = x @ loss_grad

        if eta is not None:
            assert eta > 0
            rt_eta = jnp.sqrt(eta)
            obj = rt_eta * obj + 0.5 / rt_eta * cvx.sum_squares(x - loss) ** 2

        prob = Problem(
            Minimize(obj),
            constraints,
        )
        prob.solve()

        opt_loss = x.value
        opt_vgs = values_and_grads(opt_loss)

        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
            linear_weights=opt_vgs["rho"],
            lambda_estimate=0,
        )

    return _step
