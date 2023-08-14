from typing import Callable

from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.state import StateInfo
from fair_participation.optimization import solve_qp


def mgd_step(
    values_and_grads: Callable[[ArrayLike], dict],
    loss_hull: ConvexHull,
) -> Callable[[ArrayLike], StateInfo]:
    """
    Returns update callable that exactly solves the MGD subproblem:
        min_l Sum_g (s_g * l_g * rho_g^t + )
        s.t. l in loss_hull

    :param values_and_grads: Callable that returns commonly used values and gradients.
    :param loss_hull: ConvexHull object of loss vectors.
    :return: Callable that performs a single update step.
    """

    def _step(loss: ArrayLike, rates: tuple[float]) -> StateInfo:
        """
        RRM update step.
        :param loss: Current loss vector.
        :param rates: Learning rate.
        :return: Dictionary of updated values.
        """
        eta = rates[0]
        vgs = values_and_grads(loss)
        linear_weights = vgs["full_deriv_total_loss"]
        opt_loss, _ = solve_qp(
            w=linear_weights, hull=loss_hull, gamma=1.0 / (2.0 * eta), x0=loss
        )
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
