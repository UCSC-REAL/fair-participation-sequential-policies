import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from matplotlib import patches
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import (
    set_nice_limits,
    set_corner_ticks,
    UpdatingPlot,
    project_hull,
    plot_triangles,
)


def make_loss_boundary_plot(
    ax: plt.Axes, achievable_loss: NDArray, loss_hull: ConvexHull
):
    """
    Plot the loss boundary surfaces for 2 or 3 groups.

    :param ax: matplotlib axis.
    :param achievable_loss: Achievable loss values.
    :param loss_hull: Convex hull of loss values.
    :return: Plot object.
    """
    num_groups = achievable_loss.shape[1]
    if num_groups == 2:
        return LossBoundaryPlot2Group(
            ax=ax, achievable_loss=achievable_loss, loss_hull=loss_hull
        )
    elif num_groups == 3:
        return LossBoundaryPlot3Group(
            ax=ax, achievable_loss=achievable_loss, loss_hull=loss_hull
        )
    else:
        raise NotImplementedError


class LossBoundaryPlot3Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
    ):
        """
        Plot the loss boundary surface for 3 groups.

        :param ax: matplotlib axis.
        :param achievable_loss: Achievable loss values.
        :param loss_hull: Convex hull of loss values.
        :return: Plot object.
        """

        self.ax = ax
        plt.sca(ax)

        triangles = [loss_hull.points[s] for s in loss_hull.simplices]
        normals = loss_hull.equations[:, :-1]

        plot_triangles(ax, triangles, -normals)

        for val, dim in zip([0, 0, 0], [0, 1, 2]):
            ax.plot(
                *(project_hull(loss_hull.points[loss_hull.vertices], val, dim)).T,
                color="black",
                alpha=0.5,
            )

        min_lim = 0
        max_lim = -1

        for a, b in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]:
            min_lim = min(min_lim, max(a, -1))
            max_lim = max(max_lim, min(b, 0))

        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.set_zlim([min_lim, max_lim])
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()

        ax.view_init(elev=30, azim=45)

        ax.set_xlabel("$\\ell_1$ (Group 1) $\\rightarrow$", labelpad=-5)
        ax.set_ylabel("$\\leftarrow$ $\\ell_2$ (Group 2)", labelpad=-5)
        ax.set_zlabel("$\\ell_3$ (Group 3) $\\rightarrow$", labelpad=-5)
        plt.title("Group Losses")

        # set_corner_ticks(ax, "xyz")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.scatter(*achievable_loss.T, color="black", label="Pure Policies", zorder=2)


class LossBoundaryPlot2Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
    ):
        """
        Plot the loss boundary surface for 2 groups.

        :param ax: matplotlib axis.
        :param achievable_loss: Achievable loss values.
        :param loss_hull: Convex hull of loss values.
        :return: Plot object.
        """

        ax.set_aspect("equal")
        ax.apply_aspect()
        self.ax = ax
        plt.sca(ax)
        ax.scatter(
            *achievable_loss.T,
            color="black",
            label="Pure policies",
        )
        ax.fill(
            list(loss_hull.points[:, 0]) + [loss_hull.points[0, 0]],
            list(loss_hull.points[:, 1]) + [loss_hull.points[0, 1]],
            color="black",
            alpha=0.3,
        )

        plt.xlabel("$\\ell_1$ (Group 1) $\\rightarrow$")
        plt.ylabel("$\\ell_2$ (Group 2) $\\rightarrow$")
        # ax.yaxis.set_label_coords(-0.05, 0.5)
        plt.title("Group Loss")

        set_nice_limits(ax, -1, 0, res=0.02)
        # set_corner_ticks(ax, "xy")
        ax.set_xticks([])
        ax.set_yticks([])
        ax2data = ax.transLimits.inverted().transform
        ax.add_patch(
            patches.FancyArrowPatch(
                ax2data((0.1, 1)),
                ax2data((0.15, 0.8)),
                connectionstyle="arc3,rad=0.08",
                arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                color="black",
            )
        )

        plt.annotate("$\\phi$", (0.15, 0.9), xycoords="axes fraction")

    # TODO Disable this stuff for camera-ready just in case it messes with things

    #     (self.loss_pt,) = plt.plot(
    #         [],
    #         [],
    #         color="red",
    #         marker="^",
    #         markersize=15,
    #     )
    #
    #     self.rho_arrow = plt.quiver(
    #         *tf(-0.5, -0.5),
    #         0,
    #         0,
    #         color="blue",
    #         scale=1,
    #         scale_units="xy",
    #         width=0.01,
    #         alpha=0.0,
    #     )
    #
    # def update(self, state, **_):
    #     """
    #     - Plot current location on achievable loss curve (point)
    #     - Plot vector in direction opposite rho
    #     """
    #     self.loss_pt.set_data(*state["loss"])
    #     rho = state["linear_weights"]
    #     rho_arrow = rho / (np.linalg.norm(rho) * 4)
    #     # TODO need to fix this using transLimits
    #     self.rho_arrow.set_UVC(*(-rho_arrow))
    #     self.rho_arrow.set_alpha(0.5)
