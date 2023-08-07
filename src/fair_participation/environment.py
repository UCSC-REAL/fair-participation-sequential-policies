from typing import Callable

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull

from fair_participation.rrm import rrm_step, rrm_grad_step
from fair_participation.fair_lpu import fair_lpu_step
from fair_participation.loss_functions import values_and_grads_fns
from fair_participation.state import StateInfo


class Environment:
    def __init__(
        self,
        achievable_loss: ArrayLike,
        rho_fns: tuple[Callable[[ArrayLike], Array]],
        group_sizes: ArrayLike,
        eta: float,
        init_loss_direction: ArrayLike,
        method: str,
    ) -> None:
        """
        Environment for running the simulation. Has update method that updates the state of the environment.

        :param achievable_loss: Array (n x # of groups) of achievable losses.
        :param rho_fns: Tuple of functions (one per group) that map loss -> participation rate.
        :param group_sizes: Array of group sizes summing to 1.
        :param eta: Learning rate.
        :param init_loss_direction: Initial direction of loss. Will choose achievable loss closest to this direction.
        :param method: Method to use for updating theta.
        """
        self.group_sizes = group_sizes
        self.eta = eta
        self._init_loss_direction = init_loss_direction
        self.method = method
        self.achievable_loss = achievable_loss

        # reorder points
        self.achievable_loss = self.achievable_loss[
            ConvexHull(self.achievable_loss).vertices
        ]

        self.loss_hull = ConvexHull(self.achievable_loss)

        # values_and_grads:
        # loss (vector) -> {rho (vector),
        #                   total_loss (scalar),
        #                   grad_total_loss (vector),
        #                   disparity (scalar),
        #                   grad_disparity_loss (vector)}   [dH/dl]
        self.values_and_grads = values_and_grads_fns(
            rho_fns,
            self.group_sizes,
        )

        # Take initial loss to be the achievable loss closest to the initial angle
        unit_init_loss = self._init_loss_direction / jnp.linalg.norm(
            self._init_loss_direction
        )
        closest_idx = jnp.argmax(
            self.achievable_loss
            @ unit_init_loss
            / jnp.linalg.norm(self.achievable_loss, axis=1)
        )
        self.init_loss = self.achievable_loss[closest_idx]
        vgs = self.values_and_grads(self.init_loss)

        self.state: StateInfo = StateInfo(
            loss=self.init_loss,
            rho=vgs["rho"],
            total_loss=vgs["total_loss"],
            disparity=vgs["disparity"],
        )
        self.history = [self.state._asdict()]

        if method == "RRM":
            self.update_state = rrm_step(
                values_and_grads=self.values_and_grads,
                group_sizes=self.group_sizes,
                loss_hull=self.loss_hull,
            )
        elif method == "RRM_grad":
            self.update_state = rrm_grad_step(
                values_and_grads=self.values_and_grads,
                loss_hull=self.loss_hull,
                eta=self.eta,
            )
        elif method == "FairLPU":
            self.update_state = fair_lpu_step(
                values_and_grads=self.values_and_grads,
                loss_hull=self.loss_hull,
                eta=self.eta,
                alpha=self.eta,  # TODO separate alpha parameter
            )
        else:
            raise ValueError(f"Unknown update method {method}.")

    def update(self) -> StateInfo:
        """
        Updates state using self.update_state and returns a dictionary of the new state.
        :return: Dictionary of the new state.
        """
        self.state = self.update_state(self.state.loss)
        self.history.append(self.state._asdict())
        return self.state
