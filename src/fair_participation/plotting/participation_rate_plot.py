import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import use_two_ticks_x, use_two_ticks_y


def _inverse_disparity_curve():
    rho_1 = np.linspace(0, 1, 100)
    rho_2 = np.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


class ParticipationRatePlot:
    def __init__(self, ax: plt.Axes, **kw):
        self.ax = ax
        # plot achievable rho
        theta_range = np.linspace(0, np.pi / 2, 1000)
        achievable_rho = np.array(
            [
                self.environment.get_rho(self.env.get_loss(theta))
                for theta in theta_range
            ]
        )
        ax.plot(*achievable_rho.T, color="black", label="Pareto Boundary")
        plt.title("Group Participation Rates")

        cx, cy = _inverse_disparity_curve()
        ax.plot(cx, cy, color="red", linestyle="--", label="Fair Boundary")
        ax.plot(cy, cx, color="red", linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.set_xlabel("Group 1 participation rate $\\rho_1$", labelpad=-10)
        ax.set_ylabel("Group 2 participation rate $\\rho_2$", labelpad=-10)
        ax.legend(loc="upper right")
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

    def update_center(self, rho: NDArray, **_):
        """
        plot achieved rho
        """
        return [self.ax.scatter(*rho, color="red", marker="^", s=100)]
