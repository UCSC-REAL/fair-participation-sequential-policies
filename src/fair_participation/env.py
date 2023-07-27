from typing import Optional, Callable

from jax.typing import ArrayLike

from fair_participation.optimization import parameterize_convex_hull
from fair_participation.updates import rrm_step, rrm_grad_step, fair_lpu_step

from fair_participation.loss_functions import values_and_grads_fns


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

        self.loss_hull, self.thetas = parameterize_convex_hull(achievable_loss)
        self.value_and_grad_loss, self.values_and_grads = values_and_grads_fns(
            self.thetas,
            self.loss_hull,
            rho_fns,
            self.group_sizes,
        )  # maps...

        init_loss = self.value_and_grad_loss(init_theta)["loss"]
        vgs = self.values_and_grads(init_loss)

        self.state = {
            # "lambda": 0.0,
            # "theta": init_theta,
            "loss": init_loss,
            "rho": vgs["rho"],
            "total_loss": vgs["total_loss"],
            "disparity": vgs["disparity"],
        }
        self.history = []

        if update_method == "RRM":
            self.state_update_fn = rrm_step(
                self.rho_fn,
                self.group_sizes,
                self.loss_hull,
            )
        elif update_method == "RRM_grad":
            self.state_update_fn = rrm_grad_step(
                self.rho_fn, self.group_sizes, self.loss_hull, self.eta
            )
        elif update_method == "FairLPU":
            self.state_update_fn = fair_lpu_step(
                self.vg_loss_fn,
                self.rho_fn,
                self.group_sizes,
                self.loss_hull,
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
            state["disparity"],
        ) = self.state_update_fn(self.state["loss"])
        self.state = state
        self.history.append(state)
        return state
