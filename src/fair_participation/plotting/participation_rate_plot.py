from typing import Callable

import numpy as np
from numpy.typing import NDArray

from scipy.spatial import ConvexHull

from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import (
    use_two_ticks_x,
    use_two_ticks_y,
    use_two_ticks_z,
    UpdatingPlot,
    sample_hull,
    inclusive_hull_order_2d,
)


def make_participation_rate_plot(
    ax: plt.Axes,
    achievable_loss: NDArray,
    loss_hull: ConvexHull,
    values_and_grads: Callable,
    **kwargs
):
    num_groups = achievable_loss.shape[1]
    if num_groups == 2:
        return ParticipationRatePlot2Group(
            ax, achievable_loss, loss_hull, values_and_grads, **kwargs
        )
    elif num_groups == 3:
        return ParticipationRatePlot3Group(
            ax, achievable_loss, loss_hull, values_and_grads, **kwargs
        )
    else:
        raise NotImplementedError


class ParticipationRatePlot3Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
        values_and_grads: Callable,
        **kwargs
    ):
        """

        :param ax:
        :param achievable_loss:
        :param values_and_grads:
        """
        self.ax = ax
        plt.sca(ax)

        pure_rho = np.array([values_and_grads(loss)["rho"] for loss in achievable_loss])
        ax.scatter(*pure_rho.T, color="black", label="Pure Policies")

        loss_samples = inclusive_hull_order_2d(
            list(sample_hull(loss_hull, 100)) + list(achievable_loss)
        )

        # reorder
        # loss_samples = list(loss_samples[ConvexHull(loss_samples).vertices])

        rho_samples = np.array(
            [
                values_and_grads(loss)["rho"]
                for loss in (list(loss_samples) + [loss_samples[0]])
            ]
        )

        ax.scatter(*rho_samples.T, color="black")
        plt.title("Group Participation Rates")

        # cx, cy = _inverse_disparity_curve()
        # plt.plot(cx, cy, color="red", linestyle="--", label="Fair Boundary")
        # plt.plot(cy, cx, color="red", linestyle="--")

        min_lim = 1
        max_lim = 0

        for (a, b) in [ax.get_xlim(), ax.get_ylim(), ax.get_zlim()]:
            min_lim = min(min_lim, max(a, 0))
            max_lim = max(max_lim, min(b, 1))
        ax.set_xlim([min_lim, max_lim])
        ax.set_ylim([min_lim, max_lim])
        ax.set_zlim([min_lim, max_lim])
        ax.view_init(elev=30, azim=45)

        ax.set_xlabel("Group 1 participation $\\rho_1$", labelpad=-10)
        ax.set_ylabel("Group 2 participation $\\rho_2$", labelpad=-10)
        ax.set_zlabel("Group 3 participation $\\rho_3$", labelpad=-10)
        ax.legend(loc="upper right")
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)
        use_two_ticks_z(ax)

        # (self.rate_pt,) = plt.plot([], [], color="red", marker="^", markersize=10)


class ParticipationRatePlot2Group(UpdatingPlot):
    def __init__(
        self,
        ax: plt.Axes,
        achievable_loss: NDArray,
        loss_hull: ConvexHull,
        values_and_grads: Callable,
        **kwargs
    ):
        """

        :param ax:
        :param achievable_loss:
        :param values_and_grads:
        """
        self.ax = ax
        plt.sca(ax)

        pure_rho = np.array([values_and_grads(loss)["rho"] for loss in achievable_loss])
        ax.scatter(*pure_rho.T, color="black", label="Pure Policies")

        loss_samples = inclusive_hull_order_2d(
            list(sample_hull(loss_hull, 100)) + list(achievable_loss)
        )

        # reorder
        # loss_samples = list(loss_samples[ConvexHull(loss_samples).vertices])

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

        # TODO replace with zero level set
        # cx, cy = _inverse_disparity_curve()
        # plt.plot(cx, cy, color="red", linestyle="--", label="Fair Boundary")
        # plt.plot(cy, cx, color="red", linestyle="--")

        min_lim = 1
        max_lim = 0

        for (a, b) in [ax.get_xlim(), ax.get_ylim()]:
            min_lim = min(min_lim, max(a, 0))
            max_lim = max(max_lim, min(b, 1))
        ax.set_xlim([min_lim, max_lim])
        ax.set_ylim([min_lim, max_lim])

        plt.xlabel("Group 1 participation rate $\\rho_1$", labelpad=-10)
        plt.ylabel("Group 2 participation rate $\\rho_2$", labelpad=-10)
        ax.legend(loc="upper right")
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

        (self.rate_pt,) = plt.plot([], [], color="red", marker="^", markersize=10)

    def update(self, state, **_):
        """
        plot achieved rho
        """
        self.rate_pt.set_data(*state["rho"].T)
