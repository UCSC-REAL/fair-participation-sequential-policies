from typing import Callable

from jax import grad
import jax.numpy as np
from numpy.typing import ArrayLike

from fair_participation.base_logger import log
from fair_participation.opt import quadratic_program


# TODO why is this here?
def inverse_disparity_curve():
    rho_1 = np.linspace(0, 1, 100)
    rho_2 = np.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


def disparity_fn(rho: ArrayLike):
    """
    Assumed to be symmetric

    Get violation of fairness constraint

    Args:
        rho: array of participation rates indexed by g
    """
    return np.var(rho) - 0.01
    # return np.log(100 * np.var(rho) + 0.01)


grad_disparity_fn = grad(disparity_fn)


def get_loss(t):
    raise NotImplementedError


# TODO replace with autodiff
def get_grad_loss(theta):
    """
    Use finite differences.
    """
    h = 0.0001
    return (get_loss(theta + h / 2) - get_loss(theta - h / 2)) / h


def get_tangent(self, theta):
    i = np.sum(self.ts < theta) - 1
    return np.array([self.xs[i + 1] - self.xs[i], self.ys[i + 1] - self.ys[i]])


# TODO this is duplicated fix
# def get_rho_grads(loss):
#     return np.array([self.grad_rho_fns[g](loss[g]) for g in range(2)])
def get_rho_grads(loss):
    raise NotImplementedError


rho_updates = {
    "rrm": lambda rho, loss: rho,
    "perf": lambda rho, loss: rho + get_rho_grads(loss) * loss,
}

lambda_updates = {
    "rrm": lambda _: 0.0,
    "perf": lambda _: 0.0,
}


def naive_step(
    type_: str,
) -> Callable:
    def _step(
        theta: float,
        loss: ArrayLike,
        rho: ArrayLike,
        group_sizes: ArrayLike,
        eta: float,
    ) -> tuple[float, float]:
        """
        Perform update step with rho_hat
        """
        rho_hat = rho_updates[type_](rho, loss)
        return 0.0, quadratic_program(loss, rho_hat, group_sizes)

    return _step


def fair_lam():
    perf_grad = np.sum(loss_grad * rho_hat * group_sizes)

    fair_grad = np.sum(
        grad_disparity_fn(rho)
        * rho_grad  # \pdv{F}{rho_g} [g]
        * loss_grad  # \pdv{rho_g}{l_g} [g]  # \pdv{l_g}{theta} [g]
    )

    g = disparity_fn(rho)
    d = np.sum(perf_grad * fair_grad) / np.sum(fair_grad**2)
    return np.maximum(g - d, 0)


def naive_grad_step(type_: str) -> Callable:
    def _step(
        theta: float,
        loss: ArrayLike,
        rho: ArrayLike,
        group_sizes: ArrayLike,
        eta: float,
    ) -> tuple[float, float]:
        """
        Perform update step with rho_hat
        """
        rho_hat = rho_updates[type_](rho, loss)

        perf_grad = np.sum(loss_grad * rho_hat * group_sizes)

        fair_grad = np.sum(
            grad_disparity_fn(rho)
            * rho_grad  # \pdv{F}{rho_g} [g]
            * loss_grad  # \pdv{rho_g}{l_g} [g]  # \pdv{l_g}{theta} [g]
        )

        g = disparity_fn(rho)
        d = np.sum(perf_grad * fair_grad) / np.sum(fair_grad**2)
        lambda_ = np.maximum(g - d, 0)

        grads = get_grad_loss(theta)
        return lambda_, theta - eta * np.sum(rho_hat * grads * group_sizes)

    return _step


def fair_grad_step(
    theta: float,
    loss: ArrayLike,
    rho: ArrayLike,
    group_sizes: ArrayLike,
    eta: float,
) -> tuple[float, float]:
    # pdv{rho_g}{l_g} [g] (is diagonal)
    rho_grad = get_rho_grads(loss)
    # pdv{l_g}{theta} [g]
    loss_grad = get_grad_loss(theta)

    # [g]
    rho_hat = rho + rho_grad * loss
    perf_grad = np.sum(loss_grad * rho_hat * group_sizes)

    fair_grad = np.sum(
        grad_disparity_fn(rho)
        * rho_grad  # \pdv{F}{rho_g} [g]
        * loss_grad  # \pdv{rho_g}{l_g} [g]  # \pdv{l_g}{theta} [g]
    )

    g = disparity_fn(rho)
    d = np.sum(perf_grad * fair_grad) / np.sum(fair_grad**2)
    lambda_ = np.maximum(g - d, 0)

    return (lambda_, theta - eta * (perf_grad + lambda_ * fair_grad))


def fair_step(
    theta: float,
    loss: ArrayLike,
    rho: ArrayLike,
    group_sizes: ArrayLike,
    eta: float,
) -> tuple[float, float]:
    # pdv{rho_g}{l_g} [g] (is diagonal)
    rho_grad = get_rho_grads(loss)

    # \pdv{F}{rho_g} [g]
    disp_grad = grad_disparity_fn(rho)

    # \pdv{l_g}{theta} [g]
    tangent = get_tangent(theta)
    unit_tangent = tangent / np.linalg.norm(tangent)

    g = disparity_fn(rho)
    perf_grad = rho + rho_grad * loss

    fair_proj_grad = unit_tangent * np.sum(unit_tangent, disp_grad * rho_grad)
    # \pdv{F}{rho_g} [g]  # \pdv{rho_g}{l_g} [g]

    d = np.sum(perf_grad * fair_proj_grad) / np.sum(fair_proj_grad**2)
    lambda_ = np.maximum(g - d, 0)

    log.debug("loss", loss)
    log.debug("g", g)
    log.debug("d", d)
    log.debug("perf_grad", perf_grad)
    log.debug("fair_proj_grad", fair_proj_grad)
    log.debug(
        "update",
        np.sum((perf_grad + lambda_ * fair_proj_grad) * fair_proj_grad),
    )

    return (
        lambda_,
        quadratic_program(
            loss,
            group_sizes * (perf_grad + lambda_ * fair_proj_grad),
            # self.group_sizes * (rho + rho_grad * (loss + lambda_ * disp_grad)),
        ),
    )
