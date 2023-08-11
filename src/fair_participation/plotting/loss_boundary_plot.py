import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull

from matplotlib import patches
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import (
    use_two_ticks_x,
    use_two_ticks_y,
    use_two_ticks_z,
    UpdatingPlot,
    project_hull,
    plot_triangles,
)


def make_loss_boundary_plot(
    ax: plt.Axes, achievable_loss: NDArray, loss_hull: ConvexHull, **kwargs
):
    num_groups = achievable_loss.shape[1]
    if num_groups == 2:
        return LossBoundaryPlot2Group(ax, achievable_loss, loss_hull, **kwargs)
    elif num_groups == 3:
        return LossBoundaryPlot3Group(ax, achievable_loss, loss_hull, **kwargs)
    else:
        raise NotImplementedError


class LossBoundaryPlot3Group(UpdatingPlot):
    def __init__(
        self, ax: plt.Axes, achievable_loss: NDArray, loss_hull: ConvexHull, **kwargs
    ):
        """

        :param ax:
        :param achievable_loss:
        :param loss_hull:
        """

        self.ax = ax
        plt.sca(ax)
        ax.scatter(*achievable_loss.T, color="black", label="Pure Policies")

        triangles = [loss_hull.points[s] for s in loss_hull.simplices]
        normals = loss_hull.equations[:, :-1]

        plot_triangles(ax, triangles, -normals)

        for (val, dim) in zip([0, 0, 0], [0, 1, 2]):
            ax.plot(
                *(project_hull(loss_hull.points[loss_hull.vertices], val, dim)).T,
                color="black",
                alpha=0.5
            )

        min_lim = 0
        max_lim = -1

        for (a, b) in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]:
            min_lim = min(min_lim, max(a, -1))
            max_lim = max(max_lim, min(b, 0))

        ax.set_xlim([min_lim, max_lim])
        ax.set_ylim([min_lim, max_lim])
        ax.set_zlim([min_lim, max_lim])
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()

        ax.view_init(elev=38, azim=45)

        ax.set_xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
        ax.set_ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
        ax.set_zlabel("Group 3 loss $\\ell_3$", labelpad=-10)
        plt.title("Group Loss")

        use_two_ticks_x(ax)
        use_two_ticks_y(ax)
        use_two_ticks_z(ax)


class LossBoundaryPlot2Group(UpdatingPlot):
    def __init__(
        self, ax: plt.Axes, achievable_loss: NDArray, loss_hull: ConvexHull, **kwargs
    ):
        """

        :param ax:
        :param achievable_loss:
        :param loss_hull:
        """

        self.ax = ax
        plt.sca(ax)
        ax.scatter(*achievable_loss.T, color="black", label="Pure Policies")
        ax.fill(
            list(loss_hull.points[:, 0]) + [loss_hull.points[0, 0]],
            list(loss_hull.points[:, 1]) + [loss_hull.points[0, 1]],
            color="black",
            alpha=0.3,
        )

        min_lim = 0
        max_lim = -1

        for (a, b) in [ax.get_xlim(), ax.get_ylim()]:
            min_lim = min(min_lim, max(a, -1))
            max_lim = max(max_lim, min(b, 0))
        ax.set_xlim([min_lim, max_lim])
        ax.set_ylim([min_lim, max_lim])

        plt.xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
        plt.ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
        plt.title("Group Loss")

        def rescale(x, y):
            "(-1, 0) -> (min_lim, max_lim)"
            return (
                (x + 1) * (max_lim - min_lim) + min_lim,
                (y + 1) * (max_lim - min_lim) + min_lim,
            )

        ax.legend(loc="upper right")
        ax.add_patch(
            patches.FancyArrowPatch(
                rescale(-0.9, 0.0),
                rescale(-np.cos(0.2) * 0.9, -np.sin(0.2) * 0.9),
                connectionstyle="arc3,rad=0.08",
                arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
                color="black",
            )
        )
        plt.annotate("$\\theta$", rescale(-0.85, -0.1))
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
            alpha=0.0,
        )

    def update(self, state, **_):
        """
        TODO
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rho
        """
        self.loss_pt.set_data(*state["loss"])
        rho = state["rho"]
        rho_arrow = rho / (np.linalg.norm(rho) * 4)
        self.rho_arrow.set_UVC(*(-rho_arrow))
        self.rho_arrow.set_alpha(0.5)
