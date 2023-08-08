from typing import NamedTuple
from jax.typing import ArrayLike


class StateInfo(NamedTuple):
    loss: ArrayLike
    rho: ArrayLike
    total_loss: float
    disparity: float
    lambda_estimate: float
