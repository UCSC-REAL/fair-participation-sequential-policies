from typing import Optional, Callable

from jax.typing import ArrayLike, Array

from fair_participation.optimization import parameterize_convex_hull
from fair_participation.rrm import rrm_step, rrm_grad_step
from fair_participation.fair_lpu import fair_lpu_step
from fair_participation.loss_functions import values_and_grads_fns


class Environment:
    def __init__(
        self,
        achievable_loss: ArrayLike,
        rho_fns: tuple[Callable[[ArrayLike], Array]],
        group_sizes: ArrayLike,
        eta: float,
        init_theta: float,
        update_method: str,
    ) -> None:
        """
        Environment for running the simulation. Has update method that updates the state of the environment.

        :param achievable_loss: Array (n x # of groups) of achievable losses.
        :param rho_fns: Tuple of functions (one per group) that map loss -> participation rate.
        :param group_sizes: Array of group sizes summing to 1.
        :param eta: Learning rate.
        :param init_theta: Initial value of theta.
        :param update_method: Method to use for updating theta.
        """
        self.group_sizes = group_sizes
        self.eta = eta
        self.init_theta = init_theta

        self.loss_hull, self.thetas = parameterize_convex_hull(achievable_loss)
        # value_and_grad_loss: theta (scalar) -> loss (vector), grad_loss (vector)
        # values_and_grads:
        # loss (vector) -> {rho (vector),
        #                   total_loss (scalar),
        #                   grad_total_loss (vector),
        #                   disparity (scalar),
        #                   grad_disparity_loss (vector)}   [dH/dl]
        self.value_and_grad_loss, self.values_and_grads = values_and_grads_fns(
            self.thetas,
            self.loss_hull,
            rho_fns,
            self.group_sizes,
        )

        init_loss = self.value_and_grad_loss(init_theta)["loss"]
        vgs = self.values_and_grads(init_loss)

        self.state = {
            "loss": init_loss,
            "rho": vgs["rho"],
            "total_loss": vgs["total_loss"],
            "disparity": vgs["disparity"],
        }
        self.history = []

        if update_method == "RRM":
            self.update_state = rrm_step(
                values_and_grads=self.values_and_grads,
                group_sizes=self.group_sizes,
                loss_hull=self.loss_hull,
            )
        elif update_method == "RRM_grad":
            self.update_state = rrm_grad_step(
                values_and_grads=self.values_and_grads,
                loss_hull=self.loss_hull,
                eta=self.eta,
            )
        elif update_method == "FairLPU":
            self.update_state = fair_lpu_step(
                value_and_grad_loss=self.value_and_grad_loss,
                values_and_grads=self.values_and_grads,
                loss_hull=self.loss_hull,
                eta=self.eta,
            )
        else:
            raise ValueError(f"Unknown update method {update_method}.")

    def update(self) -> dict:
        """
        Updates state using self.update_state and returns a dictionary of the new state.
        :return: Dictionary of the new state.
        """
        # TODO Putting aside theta space for now
        self.state = self.update_state(self.state["loss"])
        self.history.append(dict(**self.state))
        return self.state
