import os
from typing import Optional, Callable

import matplotlib.pyplot as plt
from tqdm import trange

import jax.numpy as jnp

from fair_participation.simulation import get_trial_filename

from fair_participation.base_logger import logger

from fair_participation.plotting.video import Video, video_filename
from fair_participation.plotting.compare import make_canvas


def get_animation_title(name, method):
    return f"{name}_{method}"


def get_animation_filename(name, method):
    return video_filename(get_animation_title(name, method))


def animate(env, method):

    animation_filename = get_animation_filename(env.name, method)
    if os.path.exists(animation_filename):
        logger.info(f"Animation exists; skipping:")
        logger.info(f"  {animation_filename}")
        return
    logger.info(f"Rendering animation:")

    num_groups = (env.group_sizes.shape)[0]

    if num_groups == 2:
        fig, plots = make_canvas(env)

        def update_callback(state):
            for plot in plots:
                plot.update(state)

    else:
        logger.info("Animation not implemented for this environment.")
        return

    trial_filename = get_trial_filename(env.name, method)
    with jnp.load(trial_filename) as npz:

        with Animation(
            fig,
            name=env.name,
            method=method,
            update_callback=update_callback,
        ) as animation:

            num_steps = len(npz["loss"])
            for k in trange(num_steps):
                state = {f: npz[f][k] for f in npz.files}
                animation.render_frame(state)


class Animation(Video):
    def __init__(
        self,
        fig: plt.Figure,
        name: str,
        method: str,
        update_callback: Optional[Callable],
    ):
        """
        problem:
            save `problem.pdf` before any simulation
            if method is not None, save `problem_method.mp4` as video of simulation
        """

        self.fig = fig
        self.name = name
        self.method = method

        super().__init__(get_animation_title(name, method), self.fig)

        self.update_callback = update_callback

    def render_frame(self, state: dict, **_):

        if self.update_callback:
            self.update_callback(state)

        self.draw()
