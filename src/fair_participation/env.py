from typing import Optional, Callable

import jax.numpy as np  # TODO FIX
from jaxlib.mlir import jax  # what is this TODO
import jax.scipy.optimize
from numpy.typing import ArrayLike

from fair_participation.base_logger import log
from fair_participation.opt import get_hull
from fair_participation.updates import (
    naive_step,
    naive_grad_step,
    fair_step,
    fair_grad_step,
    disparity_fn,
)


class Env:
    state_update_funcs = {
        "RRM": naive_step("rrm"),
        "RRM_grad": naive_grad_step("rrm"),
        "LPU": naive_step("perf"),
        "LPU_grad": naive_step("perf"),
        "Fair": fair_step,
        "Fair_grad": fair_grad_step,
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
        achievable loss: an array of loss acheivable with fixed policies.
        rho_fns: two functions (one per group) that maps group loss -> participation.
        group_sizes: array of relative group sizes.
        eta: learning rate
        """
        self.achievable_loss = achievable_loss
        self.rho_fns = rho_fns
        self.group_sizes = group_sizes
        self.eta = eta
        self.init_theta = init_theta

        self.hull, self.xs, self.ys, self.ts = get_hull(achievable_loss)
        self.grad_rho_fns = [
            jax.jacfwd(rho_fn) for rho_fn in rho_fns
        ]  # TODO might have to change this
        self.state = {
            "lambda": 0,
            "theta": init_theta,
            "loss": None,
            "rho": None,
            "total_loss": None,
            "total_disparity": None,
        }
        self.history = []
        # TODO FIX if jit
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
        state = dict(**self.state)  # will unpack init values
        if len(self.history) > 0:
            state["lambda"], state["theta"] = self.state_update_fn(
                state["theta"], state["loss"], state["rho"], self.group_sizes
            )
        state["loss"] = update_loss(state["theta"], self.xs, self.ys, self.ts)
        state["rho"] = np.array([r(l) for r, l in zip(self.rho_fns, state["loss"])])
        state["total_loss"] = np.sum(state["loss"] * state["rho"] * self.group_sizes)
        state["total_disparity"] = disparity_fn(state["rho"])
        self.history.append(state)
        return state


# TODO jit
def update_loss(theta: float, xs: ArrayLike, ys: ArrayLike, ts: ArrayLike) -> ArrayLike:
    """
    TODO
    theta [0, 1] -> group_specific loss
    """

    x = np.interp(theta, ts, xs)
    y = np.interp(theta, ts, ys)
    return np.array([x, y])
