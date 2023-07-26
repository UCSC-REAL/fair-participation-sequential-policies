from typing import Callable

import jax.numpy as jnp
from jax import Array, lax, vmap, value_and_grad
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


def value_and_grad_loss_fn(ts: ArrayLike, losses: ArrayLike) -> Callable:
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


# TODO vectorize
def value_rho_fn(rho_fns: tuple[Callable]) -> Callable:
    """
    Returns a vmapped rho fn. using lax.switch.
    :param rho_fns:
    :return:
    """
    index = jnp.arange(len(rho_fns))
    vmapped_rho = vmap(lambda i, x: lax.switch(i, rho_fns, x))

    def _rho_fn(losses: ArrayLike) -> Array:
        """
        losses is n x d"
        :param losses:
        :return:
        """
        return vmapped_rho(index, losses)

    return _rho_fn
