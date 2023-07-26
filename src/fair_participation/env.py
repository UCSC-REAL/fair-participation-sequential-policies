from typing import Optional, Callable

from jax.typing import ArrayLike

from fair_participation.optimization import parameterize_convex_hull
from fair_participation.updates import rrm_step, rrm_grad_step, fair_lpu_step
from fair_participation.group_functions import (
    value_and_grad_loss_fn,
    value_rho_fn,
)


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
        :param rho_fns: Functions (one per group) that map group loss -> participation.
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
        self.vg_loss_fn = value_and_grad_loss_fn(
            self.ts, loss_hull
        )  # theta -> vector(loss), vector(grad_loss)
        self.rho_fn = value_rho_fn(rho_fns)  # vector(loss) -> vector(rho)

        init_loss, _ = self.vg_loss_fn(init_theta)
        self.state = {
            # "lambda": 0.0,
            # "theta": init_theta,
            "loss": init_loss,
            "rho": self.rho_fn(init_loss),
            "total_loss": None,  # TODO calc these
            "total_disparity": None,
        }
        self.history = []

        if update_method == "RRM":
            self.state_update_fn = rrm_step(
                self.rho_fn,
                self.group_sizes,
                loss_hull,
            )
        elif update_method == "RRM_grad":
            self.state_update_fn = rrm_grad_step(
                self.rho_fn, self.group_sizes, loss_hull, self.eta
            )
        elif update_method == "FairLPU":
            self.state_update_fn = fair_lpu_step(
                self.vg_loss_fn,
                self.rho_fn,
                self.group_sizes,
                loss_hull,
                self.eta,
            )
        else:
            raise NotImplementedError

    def update(self) -> dict:
        """
        Updates state and returns a dictionary of the new state.
        :return:
        """
        # TODO forget about theta for now
        state = dict()
        (
            state["loss"],
            state["rho"],
            state["total_loss"],
            state["total_disparity"],
        ) = self.state_update_fn(self.state["loss"])
        self.state = state
        self.history.append(state)
        return state
