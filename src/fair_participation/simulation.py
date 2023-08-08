import os
from typing import Callable, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike
import pandas as pd
from tqdm import trange

from fair_participation.plotting.video import video_filename
from fair_participation.plotting.animation import Animation
from fair_participation.base_logger import logger
from fair_participation.environment import Environment
from fair_participation.folktasks import achievable_loss as get_achievable_loss
from fair_participation.rate_functions import concave_rho_fn
from fair_participation.utils import PROJECT_ROOT


def simulate(
    name: str = "Income",
    rho_fns: Callable | tuple[Callable] = concave_rho_fn,
    method: str = "RRM",
    save_init: bool = True,
    eta: float = 0.1,
    num_steps: int = 50,
    init_loss_direction: float | ArrayLike = 0.6,
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
    :param init_loss_direction: Initial direction of loss. Will choose achievable loss closest to this direction.
     Can be set to float to match legacy init_theta.
    :param plot_kwargs: Keyword arguments for plotting.
    """

    logger.info(f"Simulating {name}.")
    filename = os.path.join(PROJECT_ROOT, "losses", f"{name}.npy")
    try:
        achievable_loss = jnp.load(filename)
        logger.info(f"Loaded cached achievable loss from {filename}.")
    except FileNotFoundError:
        logger.info("Computing achievable loss.")
        achievable_loss = get_achievable_loss(name)
        logger.info(f"Saving achievable loss to {filename}.")
        jnp.save(filename, achievable_loss)

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

    env = Environment(
        achievable_loss=achievable_loss,
        rho_fns=rho_fns,
        group_sizes=group_sizes,
        eta=eta,
        init_loss_direction=init_loss_direction,
        method=method,
    )

    # TODO update this with fast version

    # simulate
    title = f"{name}_{method}"
    npz_filename = os.path.join(PROJECT_ROOT, "npz", f"{title}.npz")

    if os.path.exists(npz_filename):
        logger.info(f"Found {npz_filename}.")
    else:
        logger.info(f"Could not locate {npz_filename}.")
        logger.info(f"Simulating.")

        for _ in trange(num_steps):
            state = env.update()._asdict()

        df = pd.DataFrame(env.history)
        data = {col: jnp.array(df[col].to_list()) for col in df.columns}
        jnp.savez(npz_filename, **data)

    logger.info("Loading simulation data.")
    with jnp.load(npz_filename) as npz:

        vid_filename = video_filename(title)
        if not os.path.exists(vid_filename):
            try:
                with Animation(
                    title=title,
                    environment=env,
                    save_init=save_init,
                    plot_kwargs=plot_kwargs,
                ) as animation:

                    logger.info(f"Rendering.")

                    pdf_filename = f"{title}_init.pdf"
                    animation.init_render(npz, pdf_filename)

                    for k in trange(num_steps):
                        state = {f: npz[f][k] for f in npz.files}
                        animation.render_frame(state)

            except NotImplementedError:
                logger.info(
                    "Animation and Init Render Not Implemented for This Problem."
                )
