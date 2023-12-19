from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib.tri as mtri
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib as mpl

from fair_participation.plotting.plot_utils import (
    set_corner_ticks,
    UpdatingPlot,
    sample_hull_uniform,
    upsample_hull_3d,
    inclusive_hull_order_2d,
    plot_triangles,
    get_normal,
)


def plot_fair_boundary_3d(ax, fair_epsilon: float, n: int = 30):
    u = np.linspace(0, np.pi * 2, n)
    v = np.linspace(0, 0.7, 2)

    u, v = np.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()

    r, ang = Rotation.align_vectors([[1, 1, 1]], [[0, 0, 1]])

    x, y, z = (
        r.apply(
            np.array(
                [
                    np.sqrt(fair_epsilon * 3) * np.cos(u),
                    np.sqrt(fair_epsilon * 3) * np.sin(u),
                    np.zeros(len(u)),
                ]
            ).T
        )
        + np.einsum("i,j->ij", v, np.ones(3))
    ).T

    tri = mtri.Triangulation(u, v)
    ax.plot_trisurf(
        x,
        y,
        z,
        triangles=tri.triangles,
        color="red",
        edgecolor=mpl.colors.colorConverter.to_rgba("red", alpha=0.2),
        alpha=0.2,
    )


def make_participation_rate_plot(
    ax: plt.Axes,
    achievable_loss: NDArray,
    loss_hull: ConvexHull,
    values_and_grads: Callable,
    fair_epsilon: float,
) -> Any:
    """
    Plot the participation rates for 2 or 3 groups.
    :param ax: matplotlib axis.
    :param achievable_loss: Achievable loss values.
    :param loss_hull: Convex hull of loss values.
    :param values_and_grads: Function that returns the loss and gradient.
    :param fair_epsilon: Fairness parameter.
    :return: Plot object.
    """
    num_groups = achievable_loss.shape[1]
    if num_groups == 2:
        return ParticipationRatePlot2Group(
            ax=ax,
            achievable_loss=achievable_loss,
            loss_hull=loss_hull,
            values_and_grads=values_and_grads,
            fair_epsilon=fair_epsilon,
        )
    elif num_groups == 3:
        return ParticipationRatePlot3Group(
            ax=ax,
            achievable_loss=achievable_loss,
            values_and_grads=values_and_grads,
            fair_epsilon=fair_epsilon,
        )
    else:
        raise NotImplementedError("Only 2 or 3 groups supported.")


class ParticipationRatePlot3Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        values_and_grads: Callable,
        fair_epsilon: float,
    ):
        """
        Plot the participation rates for 3 groups.

        :param ax: matplotlib axis.
        :param achievable_loss: Achievable loss values.
        :param values_and_grads: Function that returns the loss and gradient.
        :param fair_epsilon: Fairness parameter.
        """
        self.ax = ax
        plt.sca(ax)

        pure_rho = np.array([values_and_grads(loss)["rho"] for loss in achievable_loss])

        upsample_deg = 3
        loss_samples, loss_tri, loss_normals = upsample_hull_3d(
            achievable_loss, upsample_deg
        )
        rho_samples = np.array([values_and_grads(loss)["rho"] for loss in loss_samples])

        # generate triangles from hull in loss space
        rho_triangles = [rho_samples[s] for s in loss_tri]

        rho_normals = [get_normal(rho_tri) for rho_tri in rho_triangles]
        sign_correct = -np.sign(np.einsum("ij,ij->i", loss_normals, rho_normals))
        new_normals = np.einsum("ij,i->ij", rho_normals, sign_correct)

        plot_triangles(ax, rho_triangles, new_normals)

        # scatter plot for rhos corresponding to original achievable losses
        pure_rho_samples = rho_samples[: len(achievable_loss)]

        plt.title("Group Participation Rates")
        plot_fair_boundary_3d(ax, fair_epsilon)

        min_lim = 1
        max_lim = 0

        for a, b in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]:
            min_lim = min(min_lim, max(a, 0))
            max_lim = max(max_lim, min(b, 1))
        plt.xlim(min_lim, max_lim)
        plt.ylim(min_lim, max_lim)
        plt.zlim(min_lim, max_lim)
        ax.view_init(elev=30, azim=45)

        plt.xlabel("$\\rho_1$ (Group 1)", labelpad=-10)
        plt.ylabel("$\\rho_2$ (Group 2)", labelpad=-10)
        plt.zlabel("$\\rho_3$ (Group 3)", labelpad=-10)
        # TODO fix this
        set_corner_ticks(ax, "xyz")
        ax.scatter(*pure_rho.T, color="black", zorder=2)


class ParticipationRatePlot2Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
        values_and_grads: Callable,
        fair_epsilon: float,
    ):
        """
        Plot the participation rates for 2 groups.

        :param ax: matplotlib axis.
        :param achievable_loss: Achievable loss values.
        :param loss_hull: Convex hull of loss values.
        :param values_and_grads: Function that returns the loss and gradient.
        :param fair_epsilon: Fairness parameter.
        """
        ax.set_aspect("equal")
        ax.apply_aspect()
        self.ax = ax
        plt.sca(ax)

        pure_rho = np.array([values_and_grads(loss)["rho"] for loss in achievable_loss])
        ax.scatter(
            *pure_rho.T,
            color="black",
            # label="Pure policies",
        )

        loss_samples = inclusive_hull_order_2d(
            list(sample_hull_uniform(loss_hull, 100)) + list(achievable_loss)
        )
        rho_samples = np.array(
            [
                values_and_grads(loss)["rho"]
                for loss in (list(loss_samples) + [loss_samples[0]])
            ]
        )
        ax.fill(*rho_samples.T, color="black", alpha=0.3)

        plt.title("Group Participation Rates")
        plt.xlabel("$\\rho_1$ (Group 1)", labelpad=-10)
        plt.ylabel("$\\rho_2$ (Group 2)", labelpad=-10)

        # Make these start at 0 and end at at next highest percent
        max_lim = max([plt.xlim()[1], plt.ylim()[1]])
        max_lim = np.ceil(max_lim / 0.01) * 0.01
        max_lim = np.clip(max_lim, 0, 1)
        plt.xlim(0, max_lim)
        plt.ylim(0, max_lim)

        set_corner_ticks(ax, "xy")

        dist = 2 * np.sqrt(fair_epsilon)
        h_color = sns.color_palette("colorblind")[3]

        ax.plot(
            [0, 1 - dist],
            [dist, 1],
            color=h_color,
            linestyle="--",
            label="$\\mathcal{H} = 0$",
        )
        ax.plot(
            [dist, 1],
            [0, 1 - dist],
            color=h_color,
            linestyle="--",
        )
        ax.legend(loc="upper right", framealpha=0.95)
        # (self.rate_pt,) = plt.plot([], [], color="red", marker="^", markersize=10)

    def update(self, state, **_):
        """
        Plot achieved rho.
        """
        self.rate_pt.set_data(*state["rho"].T)
