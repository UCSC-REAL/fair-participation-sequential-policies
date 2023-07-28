import numpy as np
from numpy.typing import NDArray

from matplotlib import patches
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import use_two_ticks_x, use_two_ticks_y


class LossBoundaryPlot:
    def __init__(self, ax: plt.Axes, achievable_loss: NDArray, loss_hull: NDArray):
        self.ax = ax
        ax.scatter(*achievable_loss.T, color="black", label="Fixed Policies")
        ax.plot(loss_hull.T, color="black", label="Pareto Boundary")

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
        ax.annotate("$\\theta$", (-0.85, -0.1))
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

        # TODO fix this part
        # lims = [
        #     [
        #         np.min(achievable_loss[:, 0]),
        #         np.max(achievable_loss[:, 0]),
        #     ],
        #     [
        #         np.min(achievable_loss[:, 1]),
        #         np.max(achievable_loss[:, 1]),
        #     ],
        # ]

        # if (lims[0][0] > -1 and lims[0][1] < 0) and (
        #     lims[1][0] > -1 and lims[1][1] < 0
        # ):
        #     left_inset = ax.inset_axes([0.5, 0.5, 0.3, 0.3])
        #     left_inset.set_xlim(lims[0][0] - 0.02, lims[0][1] + 0.02)
        #     left_inset.set_ylim(lims[1][0] - 0.02, lims[1][1] + 0.02)
        #     left_inset.scatter(*achievable_loss.T, color="black")
        #     left_inset.plot(self.environment.xs, self.environment.ys, "black")
        #     left_inset.set_xticks([])
        #     left_inset.set_yticks([])
        #     left.indicate_inset_zoom(left_inset)

    def update(self, loss: NDArray, rho: NDArray, **_):
        """
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rho
        """
        artifacts = [self.ax.scatter(*loss, color="red", marker="^", s=100)]

        # TODO fix this part
        # if self.method.startswith("RRM"):
        #     t = np.arctan(rho[1] / rho[0])
        #     l = self.env.get_loss(t)
        #     d = np.einsum("g,g->", rho, l) / np.einsum("g,g->", rho, rho)
        #     artifacts += [self.ax.plot([d * rho[0], 0], [d * rho[1], 0], "red")[0]]

        return artifacts
