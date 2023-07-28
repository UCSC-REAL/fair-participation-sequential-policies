from typing import Optional

import matplotlib.pyplot as plt

from fair_participation.environment import Environment
from fair_participation.plotting.video import Video
from fair_participation.plotting.loss_boundary_plot import LossBoundaryPlot


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

        self.fig, (lax, cax, rax) = plt.subplots(1, 3, figsize=(18, 6))
        super().__init__(f"{title}_{self.environment.method}", self.fig)

        self.left_plot = LossBoundaryPlot(
            ax=lax,
            achievable_loss=environment.achievable_loss,
            loss_hull=environment.loss_hull,
        )
        self.center_plot = ...
        self.right_plot = ...

        # # TODO why are we calling this twice
        # if save_init:
        #     self.fig.tight_layout()
        #     savefig(self.fig, os.path.join("pdf", f"{self.title}_init.pdf"))
        #     for loc in ("left", "center", "right"):
        #         fig, _ = plt.subplots(1, 1, figsize=(6, 6))
        #         self.setup(loc, self.title, **plot_kwargs)
        #         savefig(fig, os.path.join("pdf", f"{self.title}_{loc}.pdf"))

    def render_frame(self, render_pars: dict, **_):
        to_remove = self.left_plot.update(**render_pars)
        # to_remove += ...
        # to_remove += ...

        self.draw()

        # remove artifacts
        for obj in to_remove:
            obj.remove()
