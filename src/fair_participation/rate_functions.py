from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def _logistic(x: ArrayLike) -> Array:
    return 1 / (1 + jnp.exp(-x))


def localized_rho_fn(center: float, sensitivity: float) -> Callable[[ArrayLike], Array]:
    """
    Returns a callable rho function centered at `loss`.
    :param center: Center of the rho function.
    :param sensitivity: Sensitivity of the rho function.
    :return: Callable rho function mapping loss (vector) -> participation rate (vector) .
    """

    def _localized_rho(loss: ArrayLike) -> Array:
        """
        Monotonically decreasing rho function. Not concave.
        """
        return 1 - jnp.clip(_logistic((loss - center) * sensitivity), 0, 1)

    return _localized_rho


def concave_rho_fn(loss: ArrayLike) -> Array:
    """
    Monotonically decreasing rho function. Concave.
    """
    return 1 - 1 / (1 - loss * 2)
