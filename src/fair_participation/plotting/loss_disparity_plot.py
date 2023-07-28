import numpy as np
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import use_two_ticks_x, use_two_ticks_y


class LossDisparityPlot:
    def __init__(self, ax: plt.Axes, **kw):
        self.ax = ax
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

        disparities = [self.env.get_disparity(theta) for theta in theta_range]
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

        theta_l = root_find(self.env.get_disparity, 0, np.pi / 4)
        theta_r = root_find(self.env.get_disparity, np.pi / 4, np.pi / 2)
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

        plt.title("Loss and Disparity Surfaces")
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

    def update(self, theta: float = -1):
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
        # td = np.array([self.env.get_disparity(theta) for theta in theta_range])

        # artifacts += [
        #     ax.plot(theta_range, tl + lamda * td, color="black", linestyle="--")[0]
        # ]

        return artifacts
