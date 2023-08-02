import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from matplotlib import patches
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import use_two_ticks_x, use_two_ticks_y


class LossBoundaryPlot:
    def __init__(self, ax: plt.Axes, achievable_loss: NDArray, loss_hull: ConvexHull):
        """

        :param ax:
        :param achievable_loss:
        :param loss_hull:
        """
        self.ax = ax
        plt.sca(ax)
        ax.autoscale(enable=False)
        ax.scatter(*achievable_loss.T, color="black", label="Fixed Policies")
        ax.plot(*loss_hull.points.T, color="black", label="Pareto Boundary")

        plt.xlim([-1, 0])
        plt.ylim([-1, 0])
        plt.xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
        plt.ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
        plt.title("Group Loss")

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
        plt.annotate("$\\theta$", (-0.85, -0.1))
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

        (self.loss_pt,) = plt.plot([], [], color="red", marker="^", markersize=10)
        self.rho_arrow = plt.quiver(
            -0.5,
            -0.5,
            0,
            0,
            color="blue",
            scale=1,
            scale_units="xy",
            width=0.01,
            alpha=0.5,
        )

    def update(self, loss: NDArray, rho: NDArray, **_):
        """
        TODO
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rho
        """
        self.loss_pt.set_data(*loss)
        rho /= np.linalg.norm(rho) * 4
        self.rho_arrow.set_UVC(*(-rho))
