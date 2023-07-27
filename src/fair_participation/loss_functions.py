from typing import Callable, Any

import jax.numpy as jnp
from jax import value_and_grad, grad, vmap, jit, lax
from jax.typing import ArrayLike, Array


def fairness_disparity(rho: ArrayLike) -> Any:
    """
    Symmetric fairness disparity function.
    :param: rho: Array of participation rates.
    :return: Violation of fairness constraint.
    """
    return jnp.var(rho) - 0.01


def _value_and_grad_total_loss_fn(
    rho_fn: Callable,
    group_sizes: ArrayLike,
) -> Callable:
    """
    Returns a callable to compute the total loss and its gradient with respect to the loss vector.
    :param rho_fn: Callable that maps loss vector to the participation rate vector.
    :param group_sizes: Array of group sizes summing to 1.
    :return: Callable.
    """

    def _total_loss(loss: ArrayLike, loss_rho: ArrayLike) -> Array:
        """
        'Two-parameter' version of the total loss function:
            L(loss, loss_rho) = Sum_g (s_g * loss_g * rho(loss_rho_g))
        :param loss: Loss vector.
        :param loss_rho: Loss vector to be mapped to participation rates.
        :return: Total loss.
        """
        rho = rho_fn(loss_rho)
        return jnp.sum(loss * rho * group_sizes)

    # only takes gradient wrt first loss, not rho(loss)
    _vg_total_loss = value_and_grad(_total_loss, argnums=0)

    def vg_total_loss(loss: ArrayLike) -> tuple[Array, Array]:
        """
        Returns total loss and its gradient wrt loss vector.
         The gradient is taken assuming participation rates are held constant.
        That is, with L := L(loss, loss_rho), it returns the tuple
            L(loss, loss), grad_x L(x, loss)|_{x=loss}
        :param loss: Loss vector.
        :return: Tuple of total loss and its gradient.
        """
        return _vg_total_loss(loss, loss)

    return vg_total_loss


def _value_and_grad_loss_fn(thetas: ArrayLike, losses: ArrayLike) -> Callable:
    """
    Returns a callable to compute the loss and its gradient with respect to theta.
    :param thetas: Array of theta values.
    :param losses: Array of achievable losses.
    :return: Callable.
    """

    def _loss(theta: float, losses_: ArrayLike) -> Array:
        """
        Interpolates the losses for a single group.
        :param theta: Parameter value.
        :param losses_: Array of achievable losses for a single group.
        :return:
        """
        # uses 'extrapolate' to avoid zero gradient issue
        return jnp.interp(
            theta, thetas, losses_, left="extrapolate", right="extrapolate"
        )

    _value_and_grad_loss = value_and_grad(
        _loss, argnums=0
    )  # value and grad for a single group
    value_and_grad_loss = vmap(
        _value_and_grad_loss, in_axes=(None, 1)
    )  # value and grad mapped over all groups

    def _value_and_grad_loss_all(theta: float) -> tuple[Array, Array]:
        """
        Returns the loss vector and its Jacobian with respect to theta.
        :param theta: Parameter value.
        :return: Tuple of loss vector and its Jacobian.
        """
        return value_and_grad_loss(theta, losses)

    return _value_and_grad_loss_all


# TODO vectorize
def _value_rho_fn(rho_fns: tuple[Callable]) -> Callable:
    """
    Returns a callable to compute the participation rates for each group using lax.switch.
    :param rho_fns: Tuple of rho functions for each group.
    :return: Callable mapping loss vector to participation rates vector.
    """
    index = jnp.arange(len(rho_fns))
    vmapped_rho = vmap(lambda i, x: lax.switch(i, rho_fns, x))

    def _rho_fn(losses: ArrayLike) -> Array:
        """
        Maps loss vector to participation rates vector.
        :param losses: Loss vector.
        :return: Participation rates vector.
        """
        return vmapped_rho(index, losses)

    return _rho_fn


def values_and_grads_fns(
    thetas: ArrayLike,
    losses: ArrayLike,
    rho_fns: tuple[Callable],
    group_sizes: ArrayLike,
) -> tuple[Callable, Callable]:
    """
    Creates a jitted callable that returns commonly used values and gradients.
    :param thetas: Array of theta values.
    :param losses: Array of achievable losses corresponding to thetas.
    :param rho_fns: Tuple of rho functions for each group.
    :param group_sizes: Array of group sizes summing to 1.
    :return: Tuple of callables:
        - Callable that returns a dict of the loss and its gradient with respect to theta.
        - Callable that returns a dict of:
            rho, total_loss, grad_total_loss, disparity, grad_disparity_loss
    """

    value_and_grad_loss = _value_and_grad_loss_fn(thetas, losses)
    rho_f = _value_rho_fn(rho_fns)
    value_and_grad_total_loss = _value_and_grad_total_loss_fn(rho_f, group_sizes)

    def _disparity_loss(loss: ArrayLike) -> Array:
        """Maps loss vector to value of the disparity function."""
        rho = rho_f(loss)
        return fairness_disparity(rho)

    value_and_grad_disparity_loss = value_and_grad(_disparity_loss)

    @jit
    def _value_and_grad_loss(theta: float) -> dict:
        """Maps theta to dict of loss vector and its Jacobian with respect to theta."""
        loss, grad_loss = value_and_grad_loss(theta)
        return {
            "loss": loss,
            "grad_loss": grad_loss,
        }

    @jit
    def _values_and_grads(loss: ArrayLike) -> dict:
        """
        Maps loss vector to dict of:
            rho, total_loss, grad_total_loss, disparity, grad_disparity_loss
        :param loss: Loss vector.
        :return: Dict of values and gradients.
        """
        rho = rho_f(loss)
        total_loss, grad_total_loss = value_and_grad_total_loss(loss)
        disparity, grad_disparity_loss = value_and_grad_disparity_loss(loss)
        return {
            "rho": rho,
            "total_loss": total_loss,
            "grad_total_loss": grad_total_loss,
            "disparity": disparity,
            "grad_disparity_loss": grad_disparity_loss,
        }

    return _value_and_grad_loss, _values_and_grads
