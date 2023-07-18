from typing import Optional, Callable

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import value_and_grad, vmap, Array
from jaxlib.mlir import jax  # what is this TODO
import jax.scipy.optimize

from fair_participation.base_logger import logger
from fair_participation.opt import parameterize_convex_hull
from fair_participation.updates import step, disparity_fn


class Env:
    state_update_funcs = {
        "RRM": step("rrm"),
        "RRM_grad": step("rrm", quad=False),
        "LPU": step("lpu"),
        "LPU_grad": step("lpu", quad=False),
        "Fair": step("fair_lpu"),
        "Fair_grad": step("fair_lpu", quad=False),
    }

    def __init__(
        self,
        achievable_loss: ArrayLike,
        rho_fns: tuple[Callable],
        group_sizes: ArrayLike,
        eta: float,
        init_theta: float,
        update_method: Optional[str] = None,
        jit: bool = True,  # TODO remove or bake in
    ):
        """
        TODO
        :param achievable_loss: an array of losses achievable with fixed policies.
        :param rho_fns: two functions (one per group) that maps group loss -> participation.
        :param group_sizes: array of relative group sizes.
        :param eta: learning rate
        :param init_theta:
        :param update_method:
        :param jit:
        :return:
        """
        self.achievable_loss = achievable_loss
        self.rho_fns = rho_fns  # rho(loss) -> participation
        self.group_sizes = group_sizes
        self.eta = eta
        self.init_theta = init_theta

        self.loss_hull, self.ts = parameterize_convex_hull(achievable_loss)
        self.grad_rho_fns = [
            jax.jacfwd(rho_fn) for rho_fn in rho_fns
        ]  # TODO might have to change this
        self.state = {
            "lambda": 0.0,
            "theta": init_theta,
            "loss": None,
            "rho": None,
            "total_loss": None,
            "total_disparity": None,
        }
        self.history = []
        if update_method is None:
            raise NotImplementedError
        else:
            self.state_update_fn = Env.state_update_funcs[update_method]

    def update(self) -> dict:
        """
        Updates state and returns a dictionary of the new state.
        # TODO might have to factor stuff out to make this jittable
        :return:
        """
        state = dict(**self.state)  # unpacks previous state
        if len(self.history) > 0:
            state["lambda"], state["theta"] = self.state_update_fn(
                state["theta"], state["loss"], state["rho"], self.group_sizes
            )
        state["loss"], state["grad_loss"] = value_grad_loss(
            state["theta"],
            self.ts,
            self.loss_hull,
        )
        state["rho"] = jnp.array([r(l) for r, l in zip(self.rho_fns, state["loss"])])
        state["total_loss"] = jnp.sum(state["loss"] * state["rho"] * self.group_sizes)
        state["total_disparity"] = disparity_fn(state["rho"])
        self.history.append(state)
        return state


def _loss(theta: float, ts: ArrayLike, loss: ArrayLike) -> Array:
    """
    Loss for a single group.
    :param theta:
    :param ts:
    :param loss:
    :return:
    """
    # Use 'extrapolate' to avoid zero gradient issue
    return jnp.interp(theta, ts, loss, left="extrapolate", right="extrapolate")


# value and grad_theta wrt a single group
_value_grad_loss = value_and_grad(_loss, argnums=0)
# TODO jit
# value and grad_grad_theta wrt all groups - vmap over last axis
value_grad_loss = vmap(_value_grad_loss, (None, None, 1))
