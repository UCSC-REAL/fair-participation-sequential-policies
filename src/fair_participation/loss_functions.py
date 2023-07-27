from typing import Callable

import jax.numpy as jnp
from jax import value_and_grad, grad, vmap, jit, lax
from jax.typing import ArrayLike, Array


def _value_and_grad_total_loss_fn(
    rho_fn: Callable,  # vector -> vector
    group_sizes: ArrayLike,
) -> Callable:
    def _total_loss(loss: ArrayLike, loss_rho: ArrayLike) -> Array:
        """
        Maps loss [vector] x  loss_rho [vector] -> total loss [scalar].
        :param loss:
        :param loss_rho:
        :return:
        """
        rho = rho_fn(loss_rho)
        return jnp.sum(loss * rho * group_sizes)

    _vg_total_loss = value_and_grad(_total_loss, argnums=0)

    # only takes gradient wrt first loss, not rho(loss)
    def vg_total_loss(loss: ArrayLike) -> tuple[Array, Array]:
        return _vg_total_loss(loss, loss)

    return vg_total_loss


def _value_and_grad_loss_fn(thetas: ArrayLike, losses: ArrayLike) -> Callable:
    """

    :param thetas:
    :param losses:
    :return:
    """

    def _loss(theta: float, losses_: ArrayLike) -> Array:
        # interpolated losses for a single group
        # Use 'extrapolate' to avoid zero gradient issue
        return jnp.interp(
            theta, thetas, losses_, left="extrapolate", right="extrapolate"
        )

    # TODO need to check this one as you changed it
    _value_and_grad_loss = grad(_loss, argnums=0)  # value and grad for a single group
    value_and_grad_loss = vmap(
        _value_and_grad_loss, in_axes=(None, 1)
    )  # grad for all groups

    def _value_and_grad_loss_all(theta: float) -> tuple[Array, Array]:
        return value_and_grad_loss(theta, losses)

    return _value_and_grad_loss_all


# TODO vectorize
def _value_rho_fn(rho_fns: tuple[Callable]) -> Callable:
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


def values_and_grads_fns(
    thetas: ArrayLike,
    losses: ArrayLike,
    rho_fns: tuple[Callable],
    group_sizes: ArrayLike,
):
    """
    Creates a jitted callable that returns commonly used values and gradients.

    :param thetas:
    :param losses:
    :param rho_fns:
    :param group_sizes:
    :return:
    """

    value_and_grad_loss_f = _value_and_grad_loss_fn(thetas, losses)
    rho_f = _value_rho_fn(rho_fns)
    vg_total_loss_f = _value_and_grad_total_loss_fn(rho_f, group_sizes)

    @jit
    def _value_and_grad_loss(theta: float) -> dict:
        loss, grad_loss = value_and_grad_loss_f(theta)
        return {
            "loss": loss,
            "grad_loss": grad_loss,
        }

    @jit
    def _values_and_grads(loss: ArrayLike) -> dict:
        """

        :param loss:
        :return:
        """
        total_loss, grad_total_loss = vg_total_loss_f(loss)
        # TODO add disparity
        return {
            "rho": rho_f(loss),
            "total_loss": total_loss,
            "grad_total_loss": grad_total_loss,
        }

    return _value_and_grad_loss, _values_and_grads
