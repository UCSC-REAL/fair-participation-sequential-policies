import os
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from scipy.spatial import ConvexHull


from fair_participation.rate_functions import concave_rho_fn
from fair_participation.folktasks import achievable_loss as get_achievable_loss

from fair_participation.loss_functions import values_and_grads_fns
from fair_participation.state import StateInfo

from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT


def get_env_filename(name):
    return os.path.join(PROJECT_ROOT, "losses", f"{name}.npy")


def make_environment(
    name: str,
    rho_fns: Callable | tuple[Callable] = concave_rho_fn,
    init_loss_direction: float | ArrayLike = 0.6,
    fair_epsilon: float = 0.01,
    **_,
) -> None:
    """
    prepare environment for a given problem.
    Does caching

    :param name: Name of the problem
    :param rho_fns: Rho functions for each group, or a single rho function for all groups. Defaults to concave_rho_fn.
    :param init_loss_direction: Initial direction of loss. Will choose achievable loss closest to this direction.
     Can be set to float to match legacy init_theta.
    """

    logger.info(f"Setting up env for problem {name}.")
    losses_filename = get_env_filename(name=name)
    try:
        achievable_loss = jnp.load(losses_filename)
        logger.info(f"Loaded cached achievable loss:")
        logger.info(f"  {losses_filename}")
    except FileNotFoundError:
        logger.info("Caching achievable loss.")
        logger.info(f"  {losses_filename}")
        achievable_loss = get_achievable_loss(name)
        jnp.save(losses_filename, achievable_loss)

    n_groups = achievable_loss.shape[1]
    if callable(rho_fns):
        # Use same rho for all groups
        rho_fns = tuple(rho_fns for _ in range(n_groups))

    # assume even group sizes
    group_sizes = jnp.ones(n_groups) / n_groups

    # for legacy purposes, if init_theta is a float in [0,1], use that to create the initial angle
    if isinstance(init_loss_direction, float):
        assert (
            0 <= init_loss_direction <= 1
        ), "If init_theta is a float, it must be in [0,1]"
        assert (
            achievable_loss.ndim == 2
        ), "If init_theta is a float, achievable_loss must be 2D."
        # 0 at pi to 1 at 3pi/2
        init_loss_direction = jnp.array(
            [
                -jnp.cos(init_loss_direction * jnp.pi / 2),
                -jnp.sin(init_loss_direction * jnp.pi / 2),
            ]
        )

    return Environment(
        name,
        achievable_loss=achievable_loss,
        rho_fns=rho_fns,
        group_sizes=group_sizes,
        init_loss_direction=init_loss_direction,
        fair_epsilon=fair_epsilon,
    )


class Environment:
    def __init__(
        self,
        name,
        achievable_loss: ArrayLike,
        rho_fns: tuple[Callable[[ArrayLike], Array]],
        group_sizes: ArrayLike,
        init_loss_direction: ArrayLike,
        fair_epsilon: float = 0.01,
    ) -> None:
        """
        Environment for running the simulation.

        :param achievable_loss: Array (n x # of groups) of achievable losses.
        :param rho_fns: Tuple of functions (one per group) that map loss -> participation rate.
        :param group_sizes: Array of group sizes summing to 1.
        :param init_loss_direction: Initial direction of loss. Will choose achievable loss closest to this direction.
        """
        self.name = name
        self.group_sizes = group_sizes
        self._init_loss_direction = init_loss_direction
        self.fair_epsilon = fair_epsilon

        self.achievable_loss = achievable_loss

        # filter and reorder points
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
            self.fair_epsilon,
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
            lambda_estimate=0,
        )
