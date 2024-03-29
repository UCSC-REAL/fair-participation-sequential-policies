from typing import Callable, Any

from scipy.spatial import ConvexHull, Delaunay
from fair_participation.optimization import solve_qp, proj_qp

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
import seaborn as sns

from fair_participation.plotting.plot_utils import (
    set_nice_limits,
    set_corner_ticks,
    UpdatingPlot,
    upsample_triangles,
)
from fair_participation.plotting.params import LOSS_COLOR, DISPARITY_COLOR


def make_loss_disparity_plot(
    ax: plt.Axes,
    achievable_loss: NDArray,
    loss_hull: ConvexHull,
    values_and_grads: Callable,
) -> Any:
    """
    Plot the loss and disparity surfaces for 2 or 3 groups.

    :param ax: matplotlib axis.
    :param achievable_loss: Achievable loss values.
    :param loss_hull: Convex hull of loss values.
    :param values_and_grads: Function that returns loss and disparity for a given loss.
    :return: Plot object.
    """
    num_groups = achievable_loss.shape[1]
    if num_groups == 2:
        return LossDisparityPlot2Group(
            ax=ax,
            achievable_loss=achievable_loss,
            loss_hull=loss_hull,
            values_and_grads=values_and_grads,
        )
    elif num_groups == 3:
        return LossDisparityPlot3Group(
            ax=ax,
            achievable_loss=achievable_loss,
            loss_hull=loss_hull,
            values_and_grads=values_and_grads,
        )
    else:
        raise NotImplementedError("Only 2 and 3 groups supported.")


class LossDisparityPlot3Group:
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
        values_and_grads: Callable,
    ):
        """
        Plot the loss and disparity surfaces for 3 groups.

        :param ax: matplotlib axis.
        :param achievable_loss: Achievable loss values.
        :param loss_hull: Convex hull of loss values.
        :param values_and_grads: Function that returns loss and disparity for a given loss.
        """
        self.ax = ax
        plt.sca(ax)
        self.loss_hull = loss_hull

        pure_phis = np.array([self.get_phi(loss) for loss in achievable_loss])
        pure_results = [values_and_grads(loss) for loss in achievable_loss]

        tri = Delaunay(pure_phis)

        # upsample in phi space
        points, faces, normals = tri.points, tri.simplices, tri.equations[:, :-1]
        for _ in range(0):
            points, faces, normals = upsample_triangles(points, faces, normals)

        upsampled_losses = [self.get_loss([a, e]) for (a, e) in points]
        results = [
            values_and_grads(loss) for loss in upsampled_losses if loss is not None
        ]
        az, el = zip(
            *[p for (i, p) in enumerate(points) if upsampled_losses[i] is not None]
        )

        ax.scatter(
            *pure_phis.T,
            [r["disparity"] for r in pure_results],
            color="red",
        )
        ax.scatter(
            *pure_phis.T,
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

        set_corner_ticks(ax, "xyz")

    def get_phi(self, loss):
        x, y, z = -loss[0], -loss[1], -loss[2]
        return [np.arctan(y / x), np.arctan(z / np.sqrt(x**2 + y**2))]

    def get_loss(self, phi):
        az, el = phi

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
        Plot the loss and disparity surfaces for 2 groups.

        :param ax: matplotlib axis.
        :param achievable_loss: Achievable loss values.
        :param loss_hull: Convex hull of loss values.
        :param values_and_grads: Function that returns loss and disparity for a given loss.
        """
        # We only need this one to be equal box -- coordinates aren't related
        ax.set_box_aspect(aspect=1)
        self.ax = ax

        self.achievable_loss = achievable_loss
        self.loss_hull = loss_hull

        min_phi = self.get_phi(solve_qp(np.array([1, 0]), loss_hull))
        max_phi = self.get_phi(solve_qp(np.array([0, 1]), loss_hull))
        phi_range = np.linspace(min_phi, max_phi, 300)

        losses = np.array([self.get_loss(phi) for phi in phi_range])
        _values_and_grads = [values_and_grads(loss) for loss in losses]

        total_losses = [vgs["total_loss"] for vgs in _values_and_grads]
        disparities = [vgs["disparity"] for vgs in _values_and_grads]
        ax_r = ax.twinx()
        self.ax_r = ax_r
        ax_r.set_box_aspect(aspect=1)

        # plot loss curve
        ax.plot(phi_range, total_losses, c=LOSS_COLOR)

        # plot disparity curve
        ax_r.plot(
            phi_range,
            disparities,
            c=DISPARITY_COLOR,
            linestyle="dotted",
        )
        ax_r.hlines(
            0,
            min_phi,
            max_phi,
            DISPARITY_COLOR,
            linestyle="--",
        )

        plt.title("Loss and Disparity Surfaces")
        ax.set_xlabel("Parameter $\\phi$ $\\rightarrow$")
        ax.set_ylabel("Total Loss $\\mathcal{L}$ $\\rightarrow$")
        ax.yaxis.label.set_color(LOSS_COLOR)
        # ax.yaxis.set_label_coords(-0.05, 0.6)

        ax_r.set_ylabel("Disparity $\\mathcal{H}$ $\\rightarrow$")
        ax_r.yaxis.label.set_color(DISPARITY_COLOR)
        # ax_r.yaxis.set_label_coords(1.05, 0.6)

        rounded_min_phi = np.ceil(min_phi / 0.01) * 0.01
        rounded_max_phi = np.floor(max_phi / 0.01) * 0.01
        for cax, res in zip([ax, ax_r], [0.01, 0.001]):
            set_nice_limits(
                cax,
                (rounded_min_phi, -100),
                (rounded_max_phi, 100),
                res=0.001,
                equal_aspect=False,
            )
            # set_corner_ticks(cax, "xy")
            cax.set_xticks([])
            cax.set_yticks([])
        plt.sca(ax_r)
        plt.yticks(list(plt.yticks()[0]) + [0.0])

    def get_phi(self, loss):
        return np.arctan2(-loss[1], -loss[0])

    def get_loss(self, phi):
        w = np.array([-np.cos(phi), -np.sin(phi)])

        return proj_qp(w, self.loss_hull)[0]

    def update(self, state):
        """
        lambda defaults to zero if first run
        """

        phi = self.get_phi(state["loss"])
        self.loss_pt.set_data(phi, state["total_loss"])
        self.disparity_pt.set_data(phi, state["disparity"])
