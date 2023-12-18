from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
import matplotlib.tri as mtri
from matplotlib import pyplot as plt
import seaborn as sns

from fair_participation.plotting.plot_utils import (
    use_two_ticks_x,
    use_two_ticks_y,
    use_two_ticks_z,
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
    ax.plot_trisurf(x, y, z, triangles=tri.triangles, color="red", alpha=0.2)


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
        ax.scatter(*pure_rho.T, color="black", label="Pure policies")

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
        ax.scatter(*np.array(pure_rho_samples).T, color="black")

        plt.title("Group Participation Rates")
        plot_fair_boundary_3d(ax, fair_epsilon)

        min_lim = 1
        max_lim = 0

        for a, b in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]:
            min_lim = min(min_lim, max(a, 0))
            max_lim = max(max_lim, min(b, 1))
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        ax.set_zlim([min_lim, max_lim])
        ax.view_init(elev=30, azim=45)

        ax.set_xlabel("Group 1 rate $\\rho_1$", labelpad=-10)
        ax.set_ylabel("Group 2 rate $\\rho_2$", labelpad=-10)
        ax.set_zlabel("Group 3 rate $\\rho_3$", labelpad=-10)
        ax.legend(loc="upper right")
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)
        use_two_ticks_z(ax)


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
        self.ax = ax
        plt.sca(ax)

        pure_rho = np.array([values_and_grads(loss)["rho"] for loss in achievable_loss])
        ax.scatter(
            *pure_rho.T,
            color="black",
            # label="Pure policies",
            zorder=10,
            clip_on=False,
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

        ax.fill(
            *rho_samples.T,
            color="black",
            alpha=0.3,
        )
        plt.title("Group Participation Rates")

        min_lim = 1
        max_lim = 0

        for a, b in [ax.get_xlim(), ax.get_ylim()]:
            min_lim = min(min_lim, max(a, 0))
            max_lim = max(max_lim, min(b, 1))
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)

        dist = 2 * np.sqrt(fair_epsilon)
        h_color = sns.color_palette("colorblind")[3]
        ax.plot(
            [min_lim, max_lim - dist],
            [min_lim + dist, max_lim],
            color=h_color,
            linestyle="--",
            label="$\\mathcal{H} = 0$",
        )
        ax.plot(
            [min_lim + dist, max_lim],
            [min_lim, max_lim - dist],
            color=h_color,
            linestyle="--",
        )
        ax.legend(loc="upper right", framealpha=0.95)

        ax.set_xlabel("$\\rho_1$ (Group 1)", labelpad=-15)
        ax.set_ylabel("$\\rho_2$ (Group 2)", labelpad=-5)
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

        (self.rate_pt,) = plt.plot([], [], color="red", marker="^", markersize=10)

    def update(self, state, **_):
        """
        Plot achieved rho.
        """
        self.rate_pt.set_data(*state["rho"].T)
