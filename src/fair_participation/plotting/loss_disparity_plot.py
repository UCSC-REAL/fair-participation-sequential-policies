from typing import Callable
from scipy.spatial import ConvexHull, Delaunay
from fair_participation.optimization import solve_qp, proj_qp

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import (
    use_two_ticks_x,
    use_two_ticks_y,
    use_two_ticks_z,
    UpdatingPlot,
    upsample_triangles,
)


def make_loss_disparity_plot(
    ax: plt.Axes,
    achievable_loss: NDArray,
    loss_hull: ConvexHull,
    values_and_grads: Callable,
    **kwargs
):
    num_groups = achievable_loss.shape[1]
    if num_groups == 2:
        return LossDisparityPlot2Group(
            ax, achievable_loss, loss_hull, values_and_grads, **kwargs
        )
    elif num_groups == 3:
        return LossDisparityPlot3Group(
            ax, achievable_loss, loss_hull, values_and_grads, **kwargs
        )
    else:
        raise NotImplementedError


class LossDisparityPlot3Group:
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
        values_and_grads: Callable,
    ):
        """

        :param ax:
        :param achievable_loss:
        :param values_and_grads:
        """
        self.ax = ax
        plt.sca(ax)
        self.loss_hull = loss_hull

        pure_thetas = np.array([self.get_theta(loss) for loss in achievable_loss])
        pure_results = [values_and_grads(l) for l in achievable_loss]

        tri = Delaunay(pure_thetas)

        # upsample in theta space
        points, faces, normals = tri.points, tri.simplices, tri.equations[:, :-1]
        for _ in range(0):
            points, faces, normals = upsample_triangles(points, faces, normals)

        upsampled_losses = [self.get_loss([a, e]) for (a, e) in points]
        results = [values_and_grads(l) for l in upsampled_losses if l is not None]
        az, el = zip(
            *[p for (i, p) in enumerate(points) if upsampled_losses[i] is not None]
        )

        ax.scatter(
            *pure_thetas.T,
            [r["disparity"] for r in pure_results],
            color="red",
        )
        ax.scatter(
            *pure_thetas.T,
            [r["total_loss"] for r in pure_results],
            color="blue",
        )

        ax.plot_trisurf(
            az,
            el,
            [r["disparity"] for r in results],
            cmap="Reds_r",
            label="Disparity",
            alpha=0.5,
        )
        ax.plot_trisurf(
            az,
            el,
            [r["total_loss"] for r in results],
            cmap="Blues_r",
            label="Loss",
            alpha=0.5,
        )

        ax.set_xlabel("Azimuth", labelpad=-10)
        ax.set_ylabel("Elevation", labelpad=-10)
        ax.set_zlabel("Loss; Disparity", labelpad=-10)

        use_two_ticks_x(ax)
        use_two_ticks_y(ax)
        use_two_ticks_z(ax)

    def get_theta(self, loss):
        x, y, z = -loss[0], -loss[1], -loss[2]
        return [np.arctan(y / x), np.arctan(z / np.sqrt(x**2 + y**2))]

    def get_loss(self, theta):

        az, el = theta

        z = np.sin(el)
        xy = np.cos(el)
        x = xy * np.cos(az)
        y = xy * np.sin(az)

        w = np.array([-x, -y, -z])

        return proj_qp(w, self.loss_hull)[0]


class LossDisparityPlot2Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
        values_and_grads: Callable,
    ):
        """

        :param ax:
        :param achievable_loss:
        :param values_and_grads:
        """
        self.ax = ax

        self.achievable_loss = achievable_loss
        self.loss_hull = loss_hull

        min_theta = self.get_theta(solve_qp(np.array([1, 0]), loss_hull))
        max_theta = self.get_theta(solve_qp(np.array([0, 1]), loss_hull))
        theta_range = np.linspace(min_theta, max_theta, 300)

        losses = np.array([self.get_loss(theta) for theta in theta_range])
        _values_and_grads = [values_and_grads(loss) for loss in losses]

        total_losses = [vgs["total_loss"] for vgs in _values_and_grads]
        disparities = [vgs["disparity"] for vgs in _values_and_grads]
        max_disparity = max(disparities)
        max_loss = max(total_losses)
        min_loss = min(total_losses)
        ax_r = ax.twinx()
        self.ax_r = ax_r

        # plot loss curve
        ax.plot(
            theta_range,
            total_losses,
            "blue",
            label="Loss",
        )

        # plot disparity curve
        ax_r.plot(
            theta_range,
            disparities,
            "red",
            linestyle="dotted",
        )
        ax_r.plot(
            [min_theta, max_theta],
            [0, 0],
            "red",
            linestyle="--",
        )
        ax.plot([], [], "red", linestyle="dotted", label="Disparity")
        ax.plot([], [], "red", linestyle="--", label="$\\mathcal{H} = 0$")

        plt.title("Loss and Disparity Surfaces")
        ax.set_xlabel("Parameter $\\theta$", labelpad=-10)
        ax.set_ylabel("Total Loss $\\mathcal{L}$", labelpad=-30)
        ax.yaxis.label.set_color("blue")
        ax_r.set_ylabel("Disparity $\\mathcal{H}$", labelpad=-30)
        ax_r.yaxis.label.set_color("red")

        ax.legend(loc="lower left")

        (self.disparity_pt,) = ax_r.plot([], [], color="red", marker="^", markersize=10)
        (self.loss_pt,) = ax.plot([], [], color="blue", marker="o", markersize=10)

        ticks = ax_r.get_yticks()
        ax_r.set_yticks([ticks[0], 0, ticks[-1]])
        ticks = ax.get_xticks()
        ax.set_xticks([ticks[0], ticks[-1]])
        ticks = ax.get_yticks()
        ax.set_yticks([ticks[0], ticks[-1]])

    def get_theta(self, loss):
        return np.arctan2(-loss[1], -loss[0])

    def get_loss(self, theta):

        w = np.array([-np.cos(theta), -np.sin(theta)])

        return proj_qp(w, self.loss_hull)[0]

    def update(self, state):
        """
        lambda defaults to zero if first run
        """

        theta = self.get_theta(state["loss"])
        self.loss_pt.set_data(theta, state["total_loss"])
        self.disparity_pt.set_data(theta, state["disparity"])
