import os
from typing import Callable, Optional

import jax.numpy as jnp
import pandas as pd
from tqdm import trange

from fair_participation.animation import Animation
from fair_participation.base_logger import logger
from fair_participation.environment import Environment
from fair_participation.folktasks import achievable_losses
from fair_participation.rate_functions import concave_rho_fn


def simulate(
    name: str = "Income",
    rho_fns: Callable | tuple[Callable] = concave_rho_fn,
    method: str = "RRM",
    save_init: bool = True,
    eta: float = 0.1,
    num_steps: int = 100,
    init_theta: float = 0.6,
    plot_kwargs: Optional[dict] = None,
) -> None:
    """
    Simulates the dynamics for a given problem.

    :param name: Name of the problem
    :param rho_fns: Rho functions for each group, or a single rho function for all groups. Defaults to concave_rho_fn.
    :param method: Method to use for updating theta.
    :param save_init: If True, saves the initial state of the environment to figure.
    :param eta: Learning rate for the update method.
    :param num_steps: Number of steps to simulate.
    :param init_theta: Initial value of theta.
    :param plot_kwargs: Keyword arguments for plotting.
    """

    logger.info(f"Simulating {name}.")
    filename = os.path.join("losses", f"{name}.npy")
    try:
        achievable_loss = jnp.load(filename)
        logger.info(f"Loaded cached achievable loss from {filename}.")
    except FileNotFoundError:
        logger.info("Computing achievable loss.")
        achievable_loss = achievable_losses(name)
        logger.info(f"Saving achievable loss to {filename}.")
        jnp.save(filename, achievable_loss)

    n_groups = achievable_loss.shape[1]
    if callable(rho_fns):
        # Use same rho for all groups
        rho_fns = tuple(rho_fns for _ in range(n_groups))

    # assume even group sizes
    group_sizes = jnp.ones(n_groups) / n_groups

    env = Environment(
        achievable_loss=achievable_loss,
        rho_fns=rho_fns,
        group_sizes=group_sizes,
        eta=eta,
        init_theta=init_theta,
        update_method=method,
    )

    # TODO update this with fast version
    with Animation(
        title=name, env=env, method=method, save_init=save_init, plot_kwargs=plot_kwargs
    ) as viz:
        try:
            filename = os.path.join("npz", f"{name}_{method}.npz")
            with jnp.load(filename) as npz:
                for k in trange(npz["loss"].shape[0]):
                    state = {f: npz[f][k] for f in npz.files}
                    # viz.render_frame(
                    #     render_pars=pars,
                    # )
        except FileNotFoundError:
            for _ in trange(num_steps):
                state = env.update()
                # viz.render_frame(render_pars=state)
            df = pd.DataFrame(env.history)
            data = {col: jnp.array(df[col].to_list()) for col in df.columns}
            jnp.savez(filename, **data)
