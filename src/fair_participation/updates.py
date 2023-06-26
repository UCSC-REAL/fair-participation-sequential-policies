from typing import Callable

from jax import grad
import jax.numpy as np
from numpy.typing import ArrayLike

from fair_participation.base_logger import log
from fair_participation.opt import solve_qp, direct_qp_step


# TODO why is this here?
def inverse_disparity_curve():
    rho_1 = np.linspace(0, 1, 100)
    rho_2 = np.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


# TODO maybe encapsulate this out
def get_tangent(self, theta):
    i = np.sum(self.ts < theta) - 1
    return np.array([self.xs[i + 1] - self.xs[i], self.ys[i + 1] - self.ys[i]])


rho_updates = {
    "rrm": lambda rho, grad_rho, loss: rho,
    "lpu": lambda rho, grad_rho, loss: rho + grad_rho * loss,
    "fair_lpu": lambda rho, grad_rho, loss: rho + grad_rho * loss,  # TODO check
}


def disparity_fn(rho: ArrayLike):
    """
    Assumed to be symmetric

    Get violation of fairness constraint

    Args:
        rho: array of participation rates indexed by g
    """
    return np.var(rho) - 0.01


grad_disparity_fn = grad(disparity_fn)
disparity_fns = {
    "rrm": lambda rho: 0.0,
    "lpu": lambda rho: 0.0,
    "fair_lpu": disparity_fn,
}
# grad (bool) => fn
qp_solve_fn = {
    False: solve_qp,
    True: direct_qp_step,
}


def step(
    type_: str,
    quad: bool = True,
) -> Callable:
    def _step(
        theta: float,
        loss: ArrayLike,
        grad_loss: ArrayLike,
        rho: ArrayLike,
        grad_rho: ArrayLike,
        group_sizes: ArrayLike,
        eta: float,
    ) -> tuple[float, float]:
        """
        Perform update step with rho_hat
        """

        # dL/dl = "rho_hat"
        rho_hat = rho_updates[type_](rho, grad_rho, loss)

        # dH/dp
        grad_disparity = grad_disparity_fn(rho)

        # compute dH/dl projected onto tangent space of A
        # = (dH/dp * dp/dl) projected onto tangent space of A
        tangent = get_tangent(theta)
        proj_grad_disparity = (
            tangent
            * np.dot(tangent, grad_disparity * grad_rho)
            / np.linalg.norm(tangent) ** 2
        )
        # If type_ is "perf" or "rrm", then g and lambda_ are zero (no fairness constraint)
        disparity = disparity_fns[type_](rho)

        d = np.dot(rho_hat * proj_grad_disparity) / np.sum(proj_grad_disparity**2)
        lambda_ = np.maximum(disparity - d, 0)

        # solve_qp(x,y) minimizes 1/2 *||x - loss||^2 + eta * y'x
        return lambda_, qp_solve_fn[quad](
            theta,
            loss,
            group_sizes * (rho_hat + lambda_ * proj_grad_disparity),
            eta=eta,
        )

    return _step
