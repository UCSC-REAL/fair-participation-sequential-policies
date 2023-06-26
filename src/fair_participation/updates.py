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


def disparity_fn(rhos: ArrayLike):
    """
    Assumed to be symmetric

    Get violation of fairness constraint

    Args:
        rho: array of participation rates indexed by g
    """
    return np.var(rhos) - 0.01
    # return np.log(100 * np.var(rhos) + 0.01)


grad_disparity_fn = grad(disparity_fn)


def get_losses(t):
    raise NotImplementedError


# TODO replace with autodiff
def get_grad_losses(theta):
    """
    Use finite differences.
    """
    h = 0.0001
    return (get_losses(theta + h / 2) - get_losses(theta - h / 2)) / h


def get_tangent(self, theta):
    i = np.sum(self.ts < theta) - 1
    return np.array([self.xs[i + 1] - self.xs[i], self.ys[i + 1] - self.ys[i]])


# TODO this is duplicated fix
# def get_rho_grads(losses):
#     return np.array([self.grad_rho_fns[g](losses[g]) for g in range(2)])
def get_rho_grads(losses):
    raise NotImplementedError


naive_rho_updates = {
    "rrm": lambda rhos, losses: rhos,
    "perf": lambda rhos, losses: rhos + get_rho_grads(losses) * losses,
}


def naive_step(
    type_: str,
) -> Callable:
    def _step(
        theta: float,
        losses: ArrayLike,
        rhos: ArrayLike,
        group_sizes: ArrayLike,
        eta: float,
    ) -> tuple[float, float]:
        """
        Perform update step with rho_hat
        """
        rhos_hat = naive_rho_updates[type_](rhos, losses)
        return 0.0, quadratic_program(losses, rhos_hat, group_sizes)

    return _step


def naive_grad_step(type_: str) -> Callable:
    def _step(
        theta: float,
        losses: ArrayLike,
        rhos: ArrayLike,
        group_sizes: ArrayLike,
        eta: float,
    ) -> tuple[float, float]:
        """
        Perform update step with rho_hat
        """
        rhos_hat = naive_rho_updates[type_](rhos, losses)
        grads = get_grad_losses(theta)
        return 0.0, theta - eta * np.sum(rhos_hat * grads * group_sizes)

    return _step


def fair_step(theta, losses, rhos):
    # pdv{rho_g}{l_g} [g] (is diagonal)
    rhos_grad = get_rho_grads(losses)

    # \pdv{F}{rho_g} [g]
    disp_grad = grad_disparity_fn(rhos)

    # \pdv{l_g}{theta} [g]
    tangent = get_tangent(theta)
    unit_tangent = tangent / np.linalg.norm(tangent)

    g = disparity_fn(rhos)

    perf_grad = rhos + rhos_grad * losses

    fair_proj_grad = unit_tangent * np.einsum(
        "g,g->",
        unit_tangent,
        disp_grad * rhos_grad
        # \pdv{F}{rho_g} [g]  # \pdv{rho_g}{l_g} [g]
    )

    d = np.einsum("g,g->", perf_grad, fair_proj_grad) / np.einsum(
        "g,g->", fair_proj_grad, fair_proj_grad
    )
    lamda = np.maximum(g - d, 0)

    log.debug("losses", losses)
    log.debug("g", g)
    log.debug("d", d)
    log.debug("perf_grad", perf_grad)
    log.debug("fair_proj_grad", fair_proj_grad)
    log.debug(
        "update",
        np.einsum("g,g->", (perf_grad + lamda * fair_proj_grad), fair_proj_grad),
    )

    return (
        lamda,
        quadratic_program(
            losses,
            self.group_sizes * (perf_grad + lamda * fair_proj_grad),
            # self.group_sizes * (rhos + rhos_grad * (losses + lamda * disp_grad)),
        ),
    )


def fair_grad_step(self, theta, losses, rhos):
    # pdv{rho_g}{l_g} [g] (is diagonal)
    rhos_grad = self.get_rho_grads(losses)
    # pdv{l_g}{theta} [g]
    losses_grad = self.get_grad_losses(theta)

    # [g]
    rhos_hat = rhos + rhos_grad * losses
    perf_grad = np.einsum("g,g,g->", losses_grad, rhos_hat, self.group_sizes)

    fair_grad = np.einsum(
        "g,g,g->",
        self.grad_disparity_fn(rhos),  # \pdv{F}{rho_g} [g]
        rhos_grad,  # \pdv{rho_g}{l_g} [g]
        losses_grad,  # \pdv{l_g}{theta} [g]
    )

    g = self.disparity_fn(rhos)
    d = np.einsum(",->", perf_grad, fair_grad) / np.einsum(",->", fair_grad, fair_grad)
    lamda = np.maximum(g - d, 0)

    return (lamda, theta - self.eta * (perf_grad + lamda * fair_grad))
