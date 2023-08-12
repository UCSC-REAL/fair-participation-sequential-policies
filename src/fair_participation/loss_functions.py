from typing import Callable, Any

import jax.numpy as jnp
from jax import value_and_grad, grad, vmap, jit, lax, Array
from jax.typing import ArrayLike


def fairness_disparity(rho: ArrayLike, fair_epsilon: float = 0.01) -> Any:
    """
    Symmetric fairness disparity function.
    :param: rho: Array of participation rates.
    :return: Violation of fairness constraint.
    """
    return jnp.var(rho) - fair_epsilon


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


def _full_deriv_total_loss_fn(
    rho_fn: Callable,
    group_sizes: ArrayLike,
) -> Callable:
    """
    Returns a callable to compute the total loss and its total derivative with respect to the loss vector.
    :param rho_fn: Callable that maps loss vector to the participation rate vector.
    :param group_sizes: Array of group sizes summing to 1.
    :return: Callable.
    """

    def _total_loss(loss: ArrayLike) -> Array:
        """
        'One-parameter' version of the total loss function:
            L(loss) = Sum_g (s_g * loss_g * rho_g(loss_g))
        :param loss: Loss vector.
        :return: Total loss.
        """
        rho = rho_fn(loss)
        return jnp.sum(loss * rho * group_sizes)

    # takes total derivative wrt loss
    return grad(_total_loss, argnums=0)


def values_and_grads_fns(
    rho_fns: tuple[Callable],
    group_sizes: ArrayLike,
    fair_epsilon: float = 0.01,
) -> Callable:
    """
    Creates a jitted callable that returns commonly used values and gradients.

    :param rho_fns: Tuple of rho functions for each group.
    :param group_sizes: Array of group sizes summing to 1.
    :return: Tuple of callables:
        - Callable that returns a dict of the loss and its gradient with respect to theta.
        - Callable that returns a dict of:
            rho, total_loss, grad_total_loss, disparity, grad_disparity_loss
    """

    rho_f = _value_rho_fn(rho_fns)
    value_and_grad_total_loss = _value_and_grad_total_loss_fn(rho_f, group_sizes)
    full_deriv_total_loss = _full_deriv_total_loss_fn(rho_f, group_sizes)

    def _disparity_loss(loss: ArrayLike) -> Array:
        """Maps loss vector to value of the disparity function."""
        rho = rho_f(loss)
        return fairness_disparity(rho, fair_epsilon)

    value_and_grad_disparity_loss = value_and_grad(_disparity_loss)

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
            "full_deriv_total_loss": full_deriv_total_loss(loss),
            "disparity": disparity,
            "grad_disparity_loss": grad_disparity_loss,
        }

    return _values_and_grads
