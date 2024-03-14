from typing import Callable

from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.state import StateInfo
from fair_participation.optimization import solve_qp


def rgd_step(
    values_and_grads: Callable[[ArrayLike], dict],
    group_sizes: ArrayLike,
    loss_hull: ConvexHull,
) -> Callable[[ArrayLike, tuple[float, float]], StateInfo]:
    """
    Returns update callable that exactly solves the RGD subproblem:
        min_l Sum_g (s_g * l_g * rho_g^t)
        s.t. l in loss_hull

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param group_sizes: Array of group sizes summing to 1.
    :param loss_hull: ConvexHull object of loss vectors.
    :return: Callable that performs a single update step.
    """

    def _step(loss: ArrayLike, rates: tuple[float, float]) -> StateInfo:
        """
        RGD update step.
        :param loss: Current loss vector.
        :param rates: Learning rates. Only eta is used.
        :return: Dictionary of updated values.
        """
        eta, _ = rates
        vgs = values_and_grads(loss)
        rho = vgs["rho"]
        linear_weights = rho * group_sizes
        opt_loss = solve_qp(w=linear_weights, hull=loss_hull, eta=eta, x0=loss)
        opt_vgs = values_and_grads(opt_loss)
        return StateInfo(
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
            linear_weights,
            0,  # lambda_estimate
        )

    return _step
