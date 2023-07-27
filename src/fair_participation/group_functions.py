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
