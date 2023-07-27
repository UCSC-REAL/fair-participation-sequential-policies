from typing import Callable

import jax.numpy as jnp
from jax import jit
from jax.typing import ArrayLike, Array

from fair_participation.optimization import solve_qp

StateInfo = tuple[Array, Array, float, float]
StateInfoD = dict[str, StateInfo]


def _to_dict(state: StateInfo) -> dict:
    return {
        "loss": state[0],
        "rho": state[1],
        "total_loss": state[2],
        "disparity": state[3],
    }


def fair_lpu_linear_fn(
    value_and_grad_loss: Callable,
    values_and_grads: Callable,
) -> Callable:
    def _fair_lpu_linear(loss: ArrayLike, alpha: float) -> Array:
        """
        Maps loss [vector] x alpha [float] to estimate of linear term .
        :param loss:
        :param alpha:
        :return:
        """
        vgs = values_and_grads(loss)
        # dl/dtheta will give tangent space, as theta is on frontier
        _, tangent = value_and_grad_loss(loss)
        unit_tangent = tangent / jnp.linalg.norm(tangent)
        proj_fairness_grad = (
            jnp.dot(vgs["grad_disparity_loss"], unit_tangent) * unit_tangent
        )

        # TODO needs a zero check
        lambda_estimate = jnp.max(
            0.0,
            alpha * vgs["disparity"]
            - jnp.dot(vgs["grad_total_loss"], proj_fairness_grad)
            / jnp.dot(proj_fairness_grad, proj_fairness_grad),
        )
        return vgs["grad_total_loss"] + jnp.dot(lambda_estimate * proj_fairness_grad)

    return _fair_lpu_linear


def fair_lpu_step(
    value_and_grad_loss: Callable,
    values_and_grads: Callable,
    loss_hull: ArrayLike,
    eta: float,
) -> Callable[[ArrayLike], StateUpdate]:
    """
    Exactly solves the FairLPU problem QP.
    :return:
    """

    # TODO jit alpha
    fair_lpu_linear = jit(fair_lpu_linear_fn(value_and_grad_loss, values_and_grads))

    def _step(loss: ArrayLike) -> StateUpdate:
        linear_weights = fair_lpu_linear(loss)
        quadratic = (1.0 / (2.0 * eta), loss)
        opt_loss, _ = solve_qp(linear_weights, loss_hull, quadratic=quadratic)
        opt_vgs = values_and_grads(opt_loss)
        return (
            opt_loss,
            opt_vgs["rho"],
            opt_vgs["total_loss"],
            opt_vgs["disparity"],
        )

    return _step
