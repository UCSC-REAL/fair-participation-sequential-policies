import os
from typing import Optional

import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import ArrayLike

from fair_participation.plotting import savefig, use_two_ticks_x, use_two_ticks_y
from fair_participation.base_logger import log


class Video:
    """
    Use a matplotlib figure to make a video.
    For each frame must:
      1. draw to figure
      2. call the video.draw method
      3. clear the figure/axes/Artists

    Example:

    fig, ax = plt.subplots(figsize=(6, 6))

    with Video('video_name', fig) as video:
        for _ in range(num_frames):
            render_to_fig()
            video.draw()
            ax.cla()
    """

    def __init__(self, title, fig, render_flag=True, fps=10):
        self.video_file = title + ".mp4"

        # whether to actually do anything
        self.render_flag = render_flag

        if render_flag:
            self.writer = animation.FFMpegWriter(
                fps=fps, metadata={"title": title, "artist": "Matplotlib"}
            )

        self.fig = fig

    def __enter__(self):
        if self.render_flag:
            # make file
            self.writer.setup(self.fig, self.video_file, dpi=100)

        return self

    def draw(self):
        if self.render_flag:
            # draw figure and clear axes
            self.writer.grab_frame()
            for ax in self.fig.axes:
                ax.cla()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.render_flag:
            # finalize file
            self.writer.finish()
            log.info("Writing", self.video_file)


class Viz(Video):
    def __init__(
        self, title: str, env, method: Optional[str] = None, save_init=True, **kw
    ):
        """
        problem:
            save `problem.pdf` before any simulation
            if method is not None, save `problem_method.mp4` as video of simulation
        """

        self.title = title
        self.env = env
        self.method = method

        self.fig, (self.left, self.center, self.right) = plt.subplots(
            1, 3, figsize=(18, 6)
        )

        if method is not None:
            super().__init__(f"{title}_{method}", self.fig, fps=15, dpi=100)

        # TODO why are we calling this twice
        self.setup("left", "Group Losses", **kw)
        self.setup("center", "Group Participation Rates", **kw)
        self.setup("right", "Loss and Disparity Surfaces", **kw)

        if save_init:
            self.fig.tight_layout()
            savefig(self.fig, os.path.join("pdf", f"{self.title}_init.pdf"))
            for loc in ("left", "center", "right"):
                fig, _ = plt.subplots(1, 1, figsize=(6, 6))
                self.setup(loc, self.title, **kw)
                savefig(fig, os.path.join("pdf", f"{self.title}_{loc}.pdf"))

    def __enter__(self):
        if self.method is not None:
            return super().__enter__()
        return self

    def __exit__(self, *args):
        if self.method is not None:
            super().__exit__(*args)

    def setup(self, loc: str, *args, **kwargs):
        {
            "left": self.setup_left,
            "right": self.setup_right,
            "center": self.setup_center,
        }[loc](*args, **kwargs)

    def setup_left(self, left, title, **kw):
        ax = self.left
        # Plot achievable losses
        achievable_losses = self.env.achievable_losses

        ax.scatter(*achievable_losses.T, color="black", label="Fixed Policies")

        ax.plot(self.env.xs, self.env.ys, "black", label="Pareto Boundary")

        ax.set_xlim(-1, 0)
        ax.set_ylim(-1, 0)
        ax.set_xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
        ax.set_ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
        ax.set_title(title)

        ax.legend(loc="upper right")

        ax.add_patch(
            patches.FancyArrowPatch(
                (-0.9, 0.0),
                (-np.cos(0.2) * 0.9, -np.sin(0.2) * 0.9),
                connectionstyle="arc3,rad=0.08",
                arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                color="black",
            )
        )
        ax.annotate("$\\theta$", (-0.85, -0.1))

        use_two_ticks_x(left)
        use_two_ticks_y(left)

        lims = [
            [
                np.min(self.env.achievable_losses[:, 0]),
                np.max(self.env.achievable_losses[:, 0]),
            ],
            [
                np.min(self.env.achievable_losses[:, 1]),
                np.max(self.env.achievable_losses[:, 1]),
            ],
        ]
        if (lims[0][0] > -1 and lims[0][1] < 0) and (
            lims[1][0] > -1 and lims[1][1] < 0
        ):
            left_inset = ax.inset_axes([0.5, 0.5, 0.3, 0.3])
            left_inset.set_xlim(lims[0][0] - 0.02, lims[0][1] + 0.02)
            left_inset.set_ylim(lims[1][0] - 0.02, lims[1][1] + 0.02)
            left_inset.scatter(*achievable_losses.T, color="black")
            left_inset.plot(self.env.xs, self.env.ys, "black")
            left_inset.set_xticks([])
            left_inset.set_yticks([])
            left.indicate_inset_zoom(left_inset)

    def setup_center(self, title, **kw):
        ax = self.center
        # plot achievable rhos
        theta_range = np.linspace(0, np.pi / 2, 1000)
        achievable_rhos = np.array(
            [self.env.get_rhos(self.env.get_losses(theta)) for theta in theta_range]
        )
        ax.plot(*achievable_rhos.T, color="black", label="Pareto Boundary")
        ax.set_title(title)

        cx, cy = self.env.inverse_disparity_curve()
        ax.plot(cx, cy, color="red", linestyle="--", label="Fair Boundary")
        ax.plot(cy, cx, color="red", linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xlabel("Group 1 participation rate $\\rho_1$", labelpad=-10)
        ax.set_ylabel("Group 2 participation rate $\\rho_2$", labelpad=-10)
        ax.legend(loc="upper right")
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

    def setup_right(self, title, **kw):
        ax = self.right
        # plot performative loss and fairness surface
        if "theta_plot_range" in kw:
            min_theta, max_theta = kw["theta_plot_range"]
        else:
            min_theta, max_theta = (0, np.pi / 2)
        theta_range = np.linspace(min_theta, max_theta, 1000)

        ax_r = ax.twinx()

        # plot loss curve
        ax.plot(
            theta_range,
            [self.env.get_total_loss(theta) for theta in theta_range],
            "blue",
            label="Loss",
        )

        disparities = [self.env.get_total_disparity(theta) for theta in theta_range]
        max_disparity = max(disparities)

        # plot disparity curve
        ax_r.plot(
            theta_range,
            disparities,
            "red",
            linestyle="--",
        )
        ax.plot([], [], "red", linestyle="--", label="Disparity")

        def root_find(f, l, r):
            if f(l) < 0:
                assert f(r) > 0
            else:
                assert f(r) < 0
                l, r = r, l
            while abs(l - r) > 0.0001:
                m = (l + r) / 2
                if f(m) > 0:
                    r = m
                else:
                    l = m
            return m

        theta_l = root_find(self.env.get_total_disparity, 0, np.pi / 4)
        theta_r = root_find(self.env.get_total_disparity, np.pi / 4, np.pi / 2)
        ax_r.fill_between(
            [min_theta, theta_l],
            [0, 0],
            [max_disparity, max_disparity],
            alpha=0.1,
            color="red",
        )
        ax_r.fill_between(
            [theta_r, max_theta],
            [0, 0],
            [max_disparity, max_disparity],
            alpha=0.1,
            color="red",
        )

        ax.set_title(title)
        ax.set_xlabel("Parameter $\\theta$")
        ax.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$", labelpad=-20)
        ax.yaxis.label.set_color("blue")
        ax_r.set_ylabel("Disparity $\\mathcal{F}(\\rho)$", labelpad=-10)
        ax_r.yaxis.label.set_color("red")

        ax.legend(loc="lower left")

        if "t_init" in kw:
            ax.scatter(
                [kw["t_init"]],
                [self.env.get_total_loss(kw["t_init"])],
                marker="o",
                color="black",
            )
            ax.scatter(
                [kw["t_init"]],
                [self.env.get_total_loss(kw["t_init"]) + 0.005],
                marker="$0$",
                color="black",
                s=64,
            )
        if "t_rrm" in kw:
            ax.scatter(
                [kw["t_rrm"]],
                [self.env.get_total_loss(kw["t_rrm"])],
                marker="o",
                color="black",
            )
            ax.scatter(
                [kw["t_rrm"]],
                [self.env.get_total_loss(kw["t_rrm"]) + 0.005],
                marker="$R$",
                color="black",
                s=64,
            )
        if "t_lpu" in kw:
            ax.scatter(
                [kw["t_lpu"]],
                [self.env.get_total_loss(kw["t_lpu"])],
                marker="o",
                color="black",
            )
            ax.scatter(
                [kw["t_lpu"]],
                [self.env.get_total_loss(kw["t_lpu"]) + 0.005],
                marker="$L$",
                color="black",
                s=64,
            )
        if "t_fair" in kw:
            ax.scatter(
                [kw["t_fair"]],
                [self.env.get_total_loss(kw["t_fair"])],
                marker="o",
                color="black",
            )
            ax.scatter(
                [kw["t_fair"]],
                [self.env.get_total_loss(kw["t_fair"]) + 0.005],
                marker="$F$",
                color="black",
                s=64,
            )

        use_two_ticks_x(ax)
        use_two_ticks_y(ax_r)
        use_two_ticks_y(ax)

    def update_left(
        self, losses: Optional[ArrayLike] = None, rhos: Optional[ArrayLike] = None
    ):
        """
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rhos
        """
        if losses is None:
            raise ValueError("losses must be provided")
        if rhos is None:
            raise ValueError("rhos must be provided")
        ax = self.left
        artifacts = [
            ax.scatter([losses[0]], [losses[1]], color="red", marker="^", s=100)
        ]

        if self.method.startswith("RRM"):
            t = np.arctan(rhos[1] / rhos[0])
            l = self.env.get_losses(t)
            d = np.einsum("g,g->", rhos, l) / np.einsum("g,g->", rhos, rhos)
            artifacts += [ax.plot([d * rhos[0], 0], [d * rhos[1], 0], "red")[0]]

        return artifacts

    def update_center(self, rhos: Optional[ArrayLike] = None):
        """
        plot achieved rho
        """
        if rhos is None:
            raise ValueError("rhos must be provided")
        ax = self.center
        return [ax.scatter([rhos[0]], [rhos[1]], color="red", marker="^", s=100)]

    def update_right(self, ax, theta: float = -1):
        """
        lambda defaults to zero if first run
        """
        ax = self.right
        # current actual loss
        artifacts = [
            ax.scatter(
                theta,
                self.env.get_total_loss(theta),
                color="red",
                marker="^",
                s=100,
            )
        ]

        # theta_range = np.linspace(0, 1, 100) * np.pi / 2  # [i]
        # tl = np.array([self.env.get_total_loss(theta) for theta in theta_range])
        # td = np.array([self.env.get_total_disparity(theta) for theta in theta_range])

        # artifacts += [
        #     ax.plot(theta_range, tl + lamda * td, color="black", linestyle="--")[0]
        # ]

        return artifacts

    def render_frame(self, render_pars: dict, **_):
        # TODO need to fix this
        to_remove = self.update_left(**render_pars)
        to_remove.extend(self.update_center(**render_pars))
        to_remove.extend(self.update_right(**render_pars))

        self.draw()

        for obj in to_remove:
            obj.remove()