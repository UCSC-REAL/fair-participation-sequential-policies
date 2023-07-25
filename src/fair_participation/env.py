from typing import Optional, Callable

import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from jaxlib.mlir import jax  # what is this TODO

from fair_participation.base_logger import logger
from fair_participation.opt import parameterize_convex_hull
from fair_participation.updates import rrm_step, rrm_grad_step
from fair_participation.functions import value_and_grad_loss, value_and_grad_rho


class Env:
    def __init__(
        self,
        achievable_loss: ArrayLike,
        rho_fns: tuple[Callable],
        group_sizes: ArrayLike,
        eta: float,
        init_theta: float,
        update_method: Optional[str] = None,
    ):
        """
        TODO
        :param achievable_loss: an array of losses achievable with fixed policies.
        :param rho_fns: two functions (one per group) that maps group loss -> participation.
        :param group_sizes: array of relative group sizes.
        :param eta: learning rate
        :param init_theta:
        :param update_method:
        :return:
        """
        self.achievable_loss = achievable_loss
        self.group_sizes = group_sizes
        self.eta = eta
        self.init_theta = init_theta

        loss_hull, self.ts = parameterize_convex_hull(achievable_loss)
        self.vg_loss_fn = value_and_grad_loss(
            self.ts, loss_hull
        )  # theta -> vec(loss), vec(grad_loss)
        # jitting here as cvxpy isn't jittable
        self.vg_rho_fn = jit(
            value_and_grad_rho(rho_fns)
        )  # vec(loss) -> vec(rho), vec(grad_rho)

        self.state = {
            # "lambda": 0.0,
            # "theta": init_theta,
            "loss": self.vg_loss_fn(init_theta)[0],
            "rho": None,
            "total_loss": None,
            "total_disparity": None,
        }
        self.history = []

        if update_method == "RRM":
            self.state_update_fn = rrm_step(
                self.vg_rho_fn,
                self.group_sizes,
                loss_hull,
            )
        elif update_method == "RRM_grad":
            self.state_update_fn = rrm_grad_step(
                self.vg_rho_fn, self.group_sizes, loss_hull, self.eta
            )

        else:
            raise NotImplementedError

    def update(self) -> dict:
        """
        Updates state and returns a dictionary of the new state.
        :return:
        """
        # TODO forget about theta for now
        state = dict(**self.state)  # unpacks previous state
        if len(self.history) > 0:
            state["loss"] = self.state_update_fn(state["loss"])
        state["rho"] = self.vg_rho_fn(state["loss"])[0]
        # TODO maybe export this with state
        state["total_loss"] = jnp.sum(state["loss"] * state["rho"] * self.group_sizes)
        # state["total_disparity"] = disparity_fn(state["rho"])
        self.history.append(state)

        return state
