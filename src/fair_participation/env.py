from typing import Optional, Callable

import numpy as onp  # TODO FIX
import jax.numpy as np
from jaxlib.mlir import jax  # what is this TODO
import jax.scipy.optimize
from numpy.typing import ArrayLike

from fair_participation.base_logger import log
from fair_participation.opt import get_hull
from fair_participation.updates import (
    rrm_step,
    rrm_grad_step,
    perf_step,
    perf_grad_step,
    fair_step,
    fair_grad_step,
    disparity_fn,
)


class Env:
    update_funcs = {
        "RRM": rrm_step,
        "RRM_grad": rrm_grad_step,
        "LPU": perf_step,
        "LPU_grad": perf_grad_step,
        "Fair": fair_step,
        "Fair_grad": fair_grad_step,
    }

    def __init__(
        self,
        achievable_losses: ArrayLike,
        rho_fns: tuple[Callable],
        group_sizes: ArrayLike,
        eta: float,
        init_theta: float,
        update_method: Optional[str] = None,
        jit: bool = True,  # TODO remove or bake in
    ):
        """
        achievable losses: an array of losses acheivable with fixed policies.
        rho_fns: two functions (one per group) that maps group loss -> participation.
        group_sizes: array of relative group sizes.
        eta: learning rate
        """
        self.achievable_losses = achievable_losses
        self.rho_fns = rho_fns
        self.group_sizes = group_sizes
        self.eta = eta
        self.init_theta = init_theta

        self.hull, self.xs, self.ys, self.ts = get_hull(achievable_losses)
        self.grad_rho_fns = [
            jax.jacfwd(rho_fn) for rho_fn in rho_fns
        ]  # TODO might have to change this
        self.state = {
            "lambda": 0,
            "theta": init_theta,
            "losses": None,
            "rhos": None,
            "total_loss": None,
            "total_disparity": None,
        }
        self.history = []
        self.update_state: Callable = self.update_funcs[update_method]  # TODO FIXXX

    def update(self) -> dict:
        """
        Updates state and returns a dictionary of the new state.
        :return:
        """
        state = dict(**self.state)
        if len(self.history) == 0:
            # todo should have consistent ordering
            update_state = lambda t, l, r: (l, t)
        else:
            update_state = self.update_state
        state["lambda"], state["theta"] = update_state(
            state["theta"], state["losses"], state["rhos"]
        )
        state["losses"] = self.update_losses(state["theta"])
        state["rhos"] = self.update_rhos(state["losses"])
        state["total_loss"] = self.update_total_loss(state["losses"], state["rhos"])
        state["total_disparity"] = disparity_fn(state["rhos"])
        self.history.append(state)
        return state

    def update_losses(self, theta):
        """
        theta [0, 1] -> group_specific losses
        """

        x = np.interp(theta, self.ts, self.xs)
        y = np.interp(theta, self.ts, self.ys)
        return np.array([x, y])

    def update_rhos(self, losses):
        return np.array([self.rho_fns[g](losses[g]) for g in range(2)])

    def update_total_loss(self, losses, rhos):
        return np.einsum("g,g,g->", losses, rhos, self.group_sizes)
