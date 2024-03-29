import os
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from numpy.typing import NDArray

from fair_participation.rate_functions import localized_rho_fn
from fair_participation.utils import PROJECT_ROOT

from fair_participation.plotting.plot_utils import set_corner_ticks
from fair_participation.plotting.params import base_sns_theme_kwargs, base_rcparams


def get_loss(theta: NDArray) -> NDArray:
    return np.array([-np.cos(theta), -np.sin(theta)]).T


rho_fn = localized_rho_fn(-0.62, 20)


def get_rho(loss_: NDArray) -> NDArray:
    return np.array(list(zip(rho_fn(loss_[:, 0]), rho_fn(loss_[:, 1]))))


def plot():
    sns.set_theme(
        **base_sns_theme_kwargs,
        rc=base_rcparams,
        font_scale=3,
    )

    fig, axs = plt.subplots(1, 4, figsize=(11, 11))
    left, right, inset_l, inset_r = axs

    ts = np.linspace(0, np.pi / 2, 100)
    loss = get_loss(ts)
    rho = get_rho(loss)

    left.plot(*loss.T, color="black")
    left.set_xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
    left.set_ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
    left.set_xlim(-1, 0)
    left.set_ylim(-1, 0)
    set_corner_ticks(left, "xy")

    right.plot(*rho.T, color="black")
    right.yaxis.tick_right()
    right.yaxis.set_label_position("right")
    right.xaxis.tick_top()
    right.xaxis.set_label_position("top")
    right.set_xlabel("Group 1 Participation Rate $\\rho_1$", labelpad=-5)
    right.set_ylabel("Group 2 Participation Rate $\\rho_2$")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)

    set_corner_ticks(right, "xy")

    a = 0.45 * np.pi / 2
    b = 0.55 * np.pi / 2
    la = get_loss(a)
    lb = get_loss(b)
    ra = get_rho(np.array([la]))[0]
    rb = get_rho(np.array([lb]))[0]

    ms = 250

    left.scatter([la[0]], [la[1]], color="red", marker="^", s=ms)
    left.scatter([lb[0]], [lb[1]], color="blue", marker="o", s=ms)

    right.scatter([ra[0]], [ra[1]], color="red", marker="^", s=ms)
    right.scatter([rb[0]], [rb[1]], color="blue", marker="o", s=ms)
    right.plot([0, ra[0]], [0, ra[1]], color="red")
    right.plot([0, rb[0]], [0, rb[1]], color="blue", linestyle="--")
    ram = np.linalg.norm(ra)
    rbm = np.linalg.norm(rb)
    right.add_patch(
        patches.FancyArrowPatch(
            (ra[0] * 0.95, ra[1] * 0.95),
            (ra[0] * 0.95 + 0.02, ra[1] * 0.95 - 0.1),
            connectionstyle="arc3,rad=-0.08",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="red",
        )
    )
    right.add_patch(
        patches.FancyArrowPatch(
            (rb[0] * 0.95, rb[1] * 0.95),
            (rb[0] * 0.95 - 0.1, rb[1] * 0.95 + 0.02),
            connectionstyle="arc3,rad=0.08",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="blue",
            linestyle="dashed",
        )
    )

    left.plot([0, -ra[0] / ram], [0, -ra[1] / ram], color="red")
    left.plot([0, -rb[0] / rbm], [0, -rb[1] / rbm], color="blue", linestyle="--")
    left.add_patch(
        patches.FancyArrowPatch(
            (-0.9, 0.0),
            (-np.cos(0.2) * 0.9, -np.sin(0.2) * 0.9),
            connectionstyle="arc3,rad=0.08",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="black",
        )
    )
    left.annotate("$\\theta$", (-0.85, -0.1))
    left.add_patch(
        patches.FancyArrowPatch(
            (la[0] * 1.05, la[1] * 1.05),
            (-ra[0] / ram * 1.05, -ra[1] / ram * 1.05),
            connectionstyle="arc3,rad=-0.08",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="red",
        )
    )
    left.add_patch(
        patches.FancyArrowPatch(
            (lb[0] * 1.05, lb[1] * 1.05),
            (-rb[0] / rbm * 1.05, -rb[1] / rbm * 1.05),
            connectionstyle="arc3,rad=0.08",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="blue",
            linestyle="dashed",
        )
    )
    left.text(-0.43, -0.85, "$\\ell_1^2 + \\ell_2^2 = 1$")

    inset_l.plot(loss[:, 0], rho[:, 0], color="black")
    inset_l.set_xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
    inset_l.set_xlim(-1, 0)
    inset_l.set_ylabel("Group 1 Participation Rate $\\rho_1$")
    inset_l.set_ylim(0, 1)
    inset_l.scatter([la[0]], [ra[0]], color="red", marker="^", s=ms)
    inset_l.scatter([lb[0]], [rb[0]], color="blue", marker="o", s=ms)
    inset_l.text(
        -0.62,
        0.8,
        r"$\frac{1}{1 + \exp[20(\ell_g + 0.62)]}$",
    )
    set_corner_ticks(inset_l, "xy")

    la = np.einsum("i,i->", ra, la) / 2
    lb = np.einsum("i,i->", rb, lb) / 2
    inset_r.plot(
        ts,
        np.einsum("ti,ti->t", loss, rho) / 2,
        color="black",
    )
    inset_r.yaxis.set_label_position("right")
    inset_r.yaxis.tick_right()
    inset_r.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g$", labelpad=-30)
    inset_r.set_xlabel("Parameter $\\theta$", labelpad=-10)
    inset_r.scatter([a], [la], color="red", marker="^", s=ms)
    inset_r.scatter([b], [lb], color="blue", marker="o", s=ms)
    inset_r.add_patch(
        patches.FancyArrowPatch(
            (a - 0.08, la),
            (a - 0.15, la + 0.05),
            connectionstyle="arc3,rad=-0.0",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="red",
        )
    )
    inset_r.add_patch(
        patches.FancyArrowPatch(
            (b + 0.08, lb),
            (b + 0.15, lb + 0.05),
            connectionstyle="arc3,rad=0.0",
            arrowstyle="Simple, tail_width=0.5, head_width=4, head_length=8",
            color="blue",
            linestyle="dashed",
        )
    )
    inset_r.set_yticks([-0.6, -0.45])
    inset_r.set_xticks([0, np.pi / 2])

    s = 0.4
    inset_l.set_position((0.1, 0.9 - 0.33, 0.33, 0.33))
    inset_r.set_position((0.9 - 0.33, 0.1, 0.33, 0.33))
    left.set_position((0.5 - s, 0.5 - s, s, s))
    right.set_position((0.5, 0.5, s, s))

    fig.savefig(os.path.join(PROJECT_ROOT, "pdf", "dual.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    plot()
