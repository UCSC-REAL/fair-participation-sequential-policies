import numpy as np
from matplotlib import patches


def setup_left(self, title, **kw):
    ax = self.left
    # Plot achievable loss
    achievable_loss = self.environment.achievable_loss

    ax.scatter(*achievable_loss.T, color="black", label="Fixed Policies")

    ax.plot(*self.environment.loss_hull.T, "black", label="Pareto Boundary")

    ax.set_xlim(-1, 0)
    ax.set_ylim(-1, 0)
    ax.set_xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
    ax.set_ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
    ax.set_title(title)

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

    lims = [
        [
            np.min(self.environment.achievable_loss[:, 0]),
            np.max(self.environment.achievable_loss[:, 0]),
        ],
        [
            np.min(self.environment.achievable_loss[:, 1]),
            np.max(self.environment.achievable_loss[:, 1]),
        ],
    ]
    if (lims[0][0] > -1 and lims[0][1] < 0) and (lims[1][0] > -1 and lims[1][1] < 0):
        left_inset = ax.inset_axes([0.5, 0.5, 0.3, 0.3])
        left_inset.set_xlim(lims[0][0] - 0.02, lims[0][1] + 0.02)
        left_inset.set_ylim(lims[1][0] - 0.02, lims[1][1] + 0.02)
        left_inset.scatter(*achievable_loss.T, color="black")
        left_inset.plot(self.environment.xs, self.environment.ys, "black")
        left_inset.set_xticks([])
        left_inset.set_yticks([])
        left.indicate_inset_zoom(left_inset)

    def update_left(
        self, loss: Optional[ArrayLike] = None, rho: Optional[ArrayLike] = None
    ):
        """
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rho
        """
        if loss is None:
            raise ValueError("loss must be provided")
        if rho is None:
            raise ValueError("rho must be provided")
        ax = self.left
        artifacts = [ax.scatter([loss[0]], [loss[1]], color="red", marker="^", s=100)]

        if self.method.startswith("RRM"):
            t = np.arctan(rho[1] / rho[0])
            l = self.env.get_loss(t)
            d = np.einsum("g,g->", rho, l) / np.einsum("g,g->", rho, rho)
            artifacts += [ax.plot([d * rho[0], 0], [d * rho[1], 0], "red")[0]]

        return artifacts
