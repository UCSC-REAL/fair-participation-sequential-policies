from typing import Optional

import os
import matplotlib.pyplot as plt

from fair_participation.environment import Environment
from fair_participation.plotting.video import Video
from fair_participation.plotting.loss_boundary_plot import make_loss_boundary_plot
from fair_participation.plotting.participation_rate_plot import (
    make_participation_rate_plot,
)
from fair_participation.plotting.loss_disparity_plot import make_loss_disparity_plot


class Animation(Video):
    def __init__(
        self,
        title: str,
        environment: Environment,
        save_init: bool = True,
        plot_kwargs: Optional[dict] = None,
    ):
        """
        problem:
            save `problem.pdf` before any simulation
            if method is not None, save `problem_method.mp4` as video of simulation
        """

        self.title = title
        self.environment = environment
        if plot_kwargs is None:
            plot_kwargs = dict()

        num_groups = (environment.group_sizes.shape)[0]
        if num_groups == 2:
            self.fig, (lax, cax, rax) = plt.subplots(1, 3, figsize=(18, 6))
        elif num_groups == 3:
            self.fig, (lax, cax, rax) = plt.subplots(
                1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"}
            )

        super().__init__(self.title, self.fig)

        self.left_plot = make_loss_boundary_plot(
            ax=lax,
            achievable_loss=environment.achievable_loss,
            loss_hull=environment.loss_hull,
        )
        self.center_plot = make_participation_rate_plot(
            ax=cax,
            achievable_loss=environment.achievable_loss,
            loss_hull=environment.loss_hull,
            values_and_grads=environment.values_and_grads,
        )
        self.right_plot = make_loss_disparity_plot(
            ax=rax,
            achievable_loss=environment.achievable_loss,
            loss_hull=environment.loss_hull,
            values_and_grads=environment.values_and_grads,
        )

    def savefig(self, filename):
        self.fig.savefig(os.path.join("pdf", filename))

    def init_render(self, npz, filename):
        self.fig.tight_layout()

        # plt.show()
        self.savefig(filename)

    def render_frame(self, state: dict, **_):
        self.left_plot.update(state)
        self.center_plot.update(state)
        self.right_plot.update(state)
        self.fig.tight_layout()

        self.draw()
