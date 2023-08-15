from typing import Callable

import jax.numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.optimization import (
    solve_qp,
    proj_active_surfaces_qp,
    DEBUG_HACK,
)
from fair_participation.state import StateInfo

import matplotlib.pyplot as plt

import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant


from fair_participation.utils import EPS


def cpg_linear_fn(
    loss_hull: ConvexHull,
    values_and_grads: Callable,
) -> Callable:
    """
    Returns callable to compute linear term of cpg QP subproblem.
    # :param value_and_grad_loss: Callable that returns loss and gradient.
    :param loss_hull: ConvexHull object of loss vectors.
    :param values_and_grads: Callable that returns commonly used values and gradients.
    :return: Callable.
    """

    def _cpg_linear(
        loss: ArrayLike,
        scaled_disparity: float,
        loss_grad: ArrayLike,
        proj_disparity_grad: ArrayLike,
    ) -> tuple[Array, float]:
        """
        Maps loss [vector] x xi [float] to estimate of linear term.
        :param loss: Current loss vector.
        :param scaled_disparity: value of constrained function.
        :param loss_grad: gradient of loss
        :param proj_disparity_grad: gradient of disparity projected to active_surfaces space
        :return: Estimate of linear term, lambda estimate.
        """

        proj_disparity_grad_sq_norm = jnp.dot(proj_disparity_grad, proj_disparity_grad)

        # We assume that proj_disparity_grad is never 0 outside the feasible set
        # Therefore, jnp.dot(proj_disparity_grad, proj_disparity_grad) > 0 outside
        # feasible set, and if proj_disparity_grad IS 0, we know lambda_est should be 0
        lambda_estimate = jnp.max(
            jnp.array(
                [
                    0.0,
                    (scaled_disparity - jnp.dot(loss_grad, proj_disparity_grad))
                    / proj_disparity_grad_sq_norm,
                ]
            )
        ) * (proj_disparity_grad_sq_norm >= EPS)

        linear_weights = loss_grad + lambda_estimate * proj_disparity_grad

        return linear_weights, lambda_estimate

    return _cpg_linear


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

    cpg_linear = cpg_linear_fn(loss_hull, values_and_grads)

    def _step(loss: ArrayLike, rates: tuple[float]) -> StateInfo:
        """
        cpg update step.
        :param loss: Current loss vector.
        :param rates: Learning rates.
        """
        eta, alpha = rates

        if DEBUG_HACK:
            print("#" * 80)

        vgs = values_and_grads(loss)

        loss_grad = vgs["full_deriv_total_loss"]
        disparity = vgs["disparity"]
        disparity_grad = vgs["grad_disparity_loss"]

        # proj_disparity_grad = proj_active_surfaces_qp(loss, disparity_grad, loss_hull)

        # scaled_disparity = disparity * alpha

        # #####
        # linear_weights, lambda_estimate = cpg_linear(
        #     loss, scaled_disparity, loss_grad, proj_disparity_grad
        # )

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

        # if x0 is not None:
        #     if (x.value is None) or (jnp.dot(x.value, w) > jnp.dot(x0, w)):
        #         return x0

        # opt_loss = solve_qp(w=linear_weights, hull=loss_hull, eta=eta, x0=loss)

        # proposed_update = opt_loss - loss
        # proposed_update_norm = jnp.linalg.norm(proposed_update)

        # if DEBUG_HACK:
        #     proposed_update_unit = proposed_update / proposed_update_norm
        #     print("-" * 80)
        #     print("loss_grad norm", jnp.linalg.norm(loss_grad))
        #     print("proposed_update norm", proposed_update_norm)
        #     print("ALPHA", alpha)
        #     print("DISPARITY", disparity)
        #     print("SCALED_DISPARITY", scaled_disparity)
        #     print("LAMBDA_EST", lambda_estimate)
        #     print("LOSS GRAD", loss_grad)
        #     print("PROJ DISP GRAD", proj_disparity_grad)
        #     print(
        #         "LOSS GRAD . PROJ DISP GRAD",
        #         jnp.dot(loss_grad, proj_disparity_grad),
        #     )
        #     print(
        #         "LINEAR_WEIGHTS . LOSS GRAD",
        #         jnp.dot(linear_weights, loss_grad),
        #     )
        #     print(
        #         "LINEAR_WEIGHTS . PROJ DISP GRAD",
        #         jnp.dot(linear_weights, proj_disparity_grad),
        #     )
        #     print(
        #         "PROPOSED_UPDATE . LOSS GRAD",
        #         jnp.dot(proposed_update, loss_grad),
        #     )
        #     print(
        #         "PROPOSED_UPDATE . PROJ DISP GRAD",
        #         jnp.dot(proposed_update, proj_disparity_grad),
        #     )
        #     if (disparity > 0) and jnp.dot(proposed_update, proj_disparity_grad) > 0:
        #         print("??? Disparity > 0 and increasing ???")
        #     if (disparity < 0) and jnp.dot(proposed_update, loss_grad) > 0:
        #         print("??? Fair, but loss increasing ???")

        #     plt.plot(
        #         [0, loss_grad[0]],
        #         [0, loss_grad[1]],
        #         label="loss_grad",
        #         linewidth=3,
        #         alpha=0.5,
        #     )
        #     plt.plot(
        #         [0, disparity_grad[0]],
        #         [0, disparity_grad[1]],
        #         label="disparity_grad",
        #         linewidth=3,
        #         alpha=0.5,
        #     )
        #     plt.plot(
        #         [0, proj_disparity_grad[0]],
        #         [0, proj_disparity_grad[1]],
        #         label="proj_disparity_grad",
        #     )
        #     plt.plot(
        #         [0, linear_weights[0]],
        #         [0, linear_weights[1]],
        #         label="linear_weights",
        #     )
        #     plt.plot(
        #         [0, proposed_update_unit[0]],
        #         [0, proposed_update_unit[1]],
        #         label="proposed_update (unit vec)",
        #         linewidth=5,
        #         alpha=0.5,
        #     )
        #     plt.gca().set_aspect("equal")
        #     plt.legend()
        #     plt.show()

        # # Explicit stopping criterion when we deal with numerical issues
        # # resulting from projection operation.
        # # Honestly, a bit of a hack, but clever insofar as small numbers cause
        # # the issue, and small numbers are normally the stopping criterion.
        # increasingly_unfair = (disparity > 0) and jnp.dot(
        #     proposed_update, proj_disparity_grad
        # ) > 0
        # losing_despite_fairness = (disparity < 0) and jnp.dot(
        #     proposed_update, loss_grad
        # ) > 0

        # # if increasingly_unfair or losing_despite_fairness:
        # #     opt_loss = loss

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
