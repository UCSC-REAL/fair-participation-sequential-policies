from typing import Callable

import jax.numpy as jnp
from jax import Array, vmap, value_and_grad, jacfwd
from jax.typing import ArrayLike


def logistic(x):
    return 1 / (1 + jnp.exp(-x))


def localized_rho_fn(
    sensitivity: float, center: float
) -> Callable[[ArrayLike], ArrayLike]:
    """
    Returns a callable rho function centered at `loss`.
    :param sensitivity: Sensitivity of the rho function.
    :param center: Center of the rho function.
    :return: Callable
    """

    def localized_rho(loss: ArrayLike) -> Array:
        """
        Monotonically decreasing. Not concave.
        """
        return 1 - jnp.clip(logistic((loss - center) * sensitivity), 0, 1)

    return localized_rho


def value_and_grad_loss(ts: ArrayLike, losses: ArrayLike) -> Callable:
    """

    :param ts:
    :param losses:
    :return:
    """

    def _loss(theta: float, losses_: ArrayLike) -> Array:
        # interpolated losses for a single group
        # Use 'extrapolate' to avoid zero gradient issue
        return jnp.interp(theta, ts, losses_, left="extrapolate", right="extrapolate")

    _vg_loss = value_and_grad(_loss, argnums=0)  # value and grad for a single group
    vg_loss = vmap(_vg_loss, in_axes=(None, 1))  # value and grad for all groups

    def _vg_loss_all(theta: float) -> tuple[Array, Array]:
        return vg_loss(theta, losses)

    return _vg_loss_all


def value_and_grad_rho(rho_fns: tuple[Callable]) -> Callable:
    val_grads = [value_and_grad(rho) for rho in rho_fns]

    def _vg_rho(losses: ArrayLike) -> tuple[Array, Array]:
        vg = jnp.array([vg(loss) for vg, loss in zip(val_grads, losses)])
        return vg[:, 0], vg[:, 1]

    return _vg_rho
