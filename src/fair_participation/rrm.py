from typing import Callable

from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.state import StateInfo
from fair_participation.optimization import solve_qp


def rrm_step(
    values_and_grads: Callable[[ArrayLike], dict],
    group_sizes: ArrayLike,
    loss_hull: ConvexHull,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the RRM subproblem:
        min_l Sum_g (s_g * l_g * rho_g^t)
        s.t. l in loss_hull

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param group_sizes: Array of group sizes summing to 1.
    :param loss_hull: ConvexHull object of loss vectors.
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
            0,  # lambda_estimate
        )

    return _step
