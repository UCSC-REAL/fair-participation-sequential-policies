from typing import Callable

import numpy as np
from numpy.typing import NDArray

from jax import numpy as jnp
from matplotlib import pyplot as plt

from fair_participation.plotting.plot_utils import use_two_ticks_x, use_two_ticks_y


def _inverse_disparity_curve():
    rho_1 = np.linspace(0, 1, 100)
    rho_2 = np.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


class ParticipationRatePlot:
    def __init__(
        self, ax: plt.Axes, achievable_loss: NDArray, values_and_grads: Callable
    ):
        """

        :param ax:
        :param achievable_loss:
        :param values_and_grads:
        """
        self.ax = ax
        plt.sca(ax)
        ax.autoscale(enable=False)
        achievable_rho = jnp.array(
            [values_and_grads(loss)["rho"] for loss in achievable_loss]
        )
        ax.plot(*achievable_rho.T, color="black", label="Pareto Boundary")
        plt.title("Group Participation Rates")

        cx, cy = _inverse_disparity_curve()
        plt.plot(cx, cy, color="red", linestyle="--", label="Fair Boundary")
        plt.plot(cy, cx, color="red", linestyle="--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        plt.xlabel("Group 1 participation rate $\\rho_1$", labelpad=-10)
        plt.ylabel("Group 2 participation rate $\\rho_2$", labelpad=-10)
        ax.legend(loc="upper right")
        use_two_ticks_x(ax)
        use_two_ticks_y(ax)

        (self.rate_pt,) = plt.plot([], [], color="red", marker="^", markersize=10)

    def update(self, rho: NDArray, **_):
        """
        plot achieved rho
        """
        self.rate_pt.set_data(*rho.T)
