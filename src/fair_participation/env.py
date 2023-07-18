from typing import Optional, Callable

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import grad, value_and_grad
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
        self.rho_fns = rho_fns
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
        state["loss"], state["grad_loss"] = val_grad_loss(
            state["theta"], self.loss_hull, self.ts
        )
        state["rho"] = jnp.array([r(l) for r, l in zip(self.rho_fns, state["loss"])])
        state["total_loss"] = jnp.sum(state["loss"] * state["rho"] * self.group_sizes)
        state["total_disparity"] = disparity_fn(state["rho"])
        self.history.append(state)
        return state


# TODO jit
def loss(theta: float, hull: ArrayLike, ts: ArrayLike) -> ArrayLike:
    """
    TODO
    theta [0, 1] -> group_specific loss
    """
    # TODO see if you actually need this
    xs = hull[:, 0]
    ys = hull[:, 1]
    x = jnp.interp(theta, ts, xs)
    y = jnp.interp(theta, ts, ys)
    return jnp.array([x, y])


val_grad_loss = value_and_grad(loss, argnums=0)
