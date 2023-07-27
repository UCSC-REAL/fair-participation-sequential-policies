import numpy as np
import logging
from functools import partial

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from fair_participation.simulation import (
    localized_rho_fn,
    use_two_ticks_x,
    use_two_ticks_y,
)

log = logging.getLogger(__name__)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

font = {"size": 13}

mpl.rc("font", **font)

################################################################################

# fig = plt.figure()
# gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
# (ax1, right), (left, ax4) = gs.subplots(sharex="col", sharey="row")

fig, axs = plt.subplots(1, 4, figsize=(10, 10))
left, right, inset_l, inset_r = axs


def get_loss(theta):
    return np.array([-np.cos(theta), -np.sin(theta)]).T


rho_fn = partial(localized_rho_fn, -0.62, 20)


def get_rho(loss):
    return np.array(list(zip(rho_fn(loss[:, 0]), rho_fn(loss[:, 1]))))


ts = np.linspace(0, np.pi / 2, 100)
loss = get_loss(ts)
rho = get_rho(loss)

left.plot(*loss.T, color="black")
left.set_xlabel("Group 1 loss $\\ell_1$")
left.set_ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
left.set_xlim(-1, 0)
left.set_ylim(-1, 0)
use_two_ticks_x(left)
use_two_ticks_y(left)

right.plot(*rho.T, color="black")
right.yaxis.tick_right()
right.yaxis.set_label_position("right")
right.xaxis.tick_top()
right.xaxis.set_label_position("top")
right.set_xlabel("Group 1 Participation Rate $\\rho_1$")
right.set_ylabel("Group 2 Participation Rate $\\rho_2$")
right.set_xlim(0, 1)
right.set_ylim(0, 1)
use_two_ticks_x(right)
use_two_ticks_y(right)

a = 0.45 * np.pi / 2
b = 0.55 * np.pi / 2
la = get_loss(a)
lb = get_loss(b)
ra = get_rho(np.array([la]))[0]
rb = get_rho(np.array([lb]))[0]

ms = 100  # markersize

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
# d = -np.sqrt(np.einsum("g,g->", ra, ra))
# left.plot([d / ra[0], 0], [0, d / ra[1]], "red", alpha=0.5)
# d = -np.sqrt(np.einsum("g,g->", rb, rb))
# left.plot([d / rb[0], 0], [0, d / rb[1]], "blue", alpha=0.5, linestyle="--")
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
left.text(-0.3, -0.8, "$\\ell_1^2 + \\ell_2^2 = 1$")

inset_l.plot(loss[:, 0], rho[:, 0], color="black")
inset_l.set_xlabel("Group 1 loss $\\ell_1$")
inset_l.set_xlim(-1, 0)
inset_l.set_ylabel("Group 1 Participation Rate $\\rho_1$")
inset_l.set_ylim(0, 1)
inset_l.scatter([la[0]], [ra[0]], color="red", marker="^", s=ms)
inset_l.scatter([lb[0]], [rb[0]], color="blue", marker="o", s=ms)
inset_l.text(
    -0.57,
    0.8,
    r"$\frac{1}{1 + \exp[20(\ell_g + 0.62)]}$",
    fontsize=19,
)
use_two_ticks_x(inset_l)
use_two_ticks_y(inset_l)


la = np.einsum("i,i->", ra, la) / 2
lb = np.einsum("i,i->", rb, lb) / 2
inset_r.plot(
    ts,
    np.einsum("ti,ti->t", loss, rho) / 2,
    color="black",
)
inset_r.yaxis.set_label_position("right")
inset_r.yaxis.tick_right()
inset_r.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$", labelpad=-20)
inset_r.set_xlabel("Parameter $\\theta$")
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
inset_r.set_yticks([-0.6, -0.4])
inset_r.set_xticks([0, np.pi / 2])

s = 0.4
inset_l.set_position((0.1, 0.9 - 0.33, 0.33, 0.33))
inset_r.set_position((0.9 - 0.33, 0.1, 0.33, 0.33))
left.set_position((0.5 - s, 0.5 - s, s, s))
right.set_position((0.5, 0.5, s, s))


fig.savefig("dual.pdf")
