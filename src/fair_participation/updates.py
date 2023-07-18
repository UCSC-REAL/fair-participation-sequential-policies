from typing import Callable

from jax import Array, value_and_grad
import jax.numpy as jnp
from numpy.typing import ArrayLike

from fair_participation.base_logger import log
from fair_participation.opt import solve_qp, direct_qp_step


# TODO why is this here?
def inverse_disparity_curve():
    rho_1 = jnp.linspace(0, 1, 100)
    rho_2 = jnp.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


rho_updates = {
    "RRM": lambda rho, grad_rho, loss: rho,
    "LPU": lambda rho, grad_rho, loss: rho + grad_rho * loss,
    "FairLPU": lambda rho, grad_rho, loss: rho + grad_rho * loss,  # TODO check
}


def fairness_disparity(rho: ArrayLike) -> Array:
    """
    Assumed to be symmetric

    Get violation of fairness constraint

    Args:
        rho: array of participation rates indexed by g
    """
    return jnp.var(rho) - 0.01


disparity_fns = {
    "RRM": lambda rho: 0.0,
    "LPU": lambda rho: 0.0,
    "FairLPU": fairness_disparity,
}
# grad (bool) => fn
qp_solve_fn = {
    False: solve_qp,
    True: direct_qp_step,
}


def step(
    method: str,
    value_and_grad_loss_fn: Callable,  # scalar -> (vector, vector)
    value_and_grad_rho_fn: Callable,  # vector -> (vector, vector)
    group_sizes: ArrayLike,
    eta: float,
) -> Callable:
    quad = not method.endswith("grad")
    method = method.split("_")[0]

    # If method is "perf" or "rrm", then g and lambda_ are zero (no fairness constraint)
    disparity_fn = disparity_fns[method]  # vector -> scalar
    # TODO need to incorporate ths part
    rho_hat_fn = rho_updates[method]

    def total_loss(loss: ArrayLike) -> Array:
        rho, _ = value_and_grad_rho_fn(loss)
        return jnp.sum(loss * rho * group_sizes)

    vg_total_loss = value_and_grad(total_loss, has_aux=True)

    def disparity_loss_f(loss: ArrayLike) -> Array:
        rho, _ = value_and_grad_rho_fn(loss)
        return disparity_fn(rho)

    vg_disparity_loss = value_and_grad(disparity_loss_f)

    # TODO should jit here (or above)
    def _pre_solve(
        theta: float,
    ):
        """
        Perform update step with rho_hat
        """
        loss, grad_loss = value_and_grad_loss_fn(theta)
        # dL/dl = "rho_hat"
        _, rho_hat = vg_total_loss(loss)

        # dH/dl
        disparity, grad_disparity_loss = vg_disparity_loss(loss)

        # compute dH/dl projected onto tangent space of A = dloss/dtheta vector
        proj_grad_disparity = (
            grad_loss
            * jnp.dot(grad_loss, grad_disparity_loss)
            / jnp.linalg.norm(grad_loss) ** 2
        )
        # TODO FIX make sure this isnt zero
        d = jnp.dot(rho_hat * proj_grad_disparity) / jnp.sum(proj_grad_disparity**2)
        lambda_ = jnp.maximum(disparity - d, 0)
        return loss, rho_hat, proj_grad_disparity, lambda_

    def _step(theta: float) -> tuple[float, Array]:
        loss, rho_hat, proj_grad_disparity, lambda_ = _pre_solve(theta)
        # solve_qp(x,y) minimizes 1/2 *||x - loss||^2 + eta * y'x
        return lambda_, qp_solve_fn[quad](
            theta,
            loss,
            group_sizes * (rho_hat + lambda_ * proj_grad_disparity),
            eta=eta,
        )

    return _step
