import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os


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


def savefig(fig, filename):
    log.info(f"Saving {filename}")
    fig.savefig(filename)


def use_two_ticks_x(ax):
    x = ax.get_xticks()
    ax.set_xticks(x[:: len(x) - 1])


def use_two_ticks_y(ax):
    y = ax.get_yticks()
    ax.set_yticks(y[:: len(y) - 1])


def compare(problem, grad=True):
    # save results of loss, disparity, lambda vs time.
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

    left, center, right = axs
    left_r, center_r, right_r = left.twinx(), center.twinx(), right.twinx()

    if grad:
        to_compare = [
            ("RRM_grad", left, left_r),
            ("LPU_grad", center, center_r),
            ("Fair_grad", right, right_r),
        ]
    else:
        to_compare = [
            ("RRM", left, left_r),
            ("LPU", center, center_r),
            ("Fair", right, right_r),
        ]
    lmin = np.inf
    lmax = -np.inf
    dmin = np.inf
    dmax = -np.inf
    for method, ax, ax_r in to_compare:
        filename = os.path.join("npy", f"{problem}_{method}")
        total_loss = np.load(f"{filename}_total_loss.npy")
        total_disparity = np.load(f"{filename}_total_disparity.npy")
        lambdas = np.load(f"{filename}_lambdas.npy")

        lmin = min(lmin, min(total_loss))
        lmax = max(lmax, max(total_loss))
        dmin = min(dmin, min(total_disparity))
        dmax = max(dmax, max(total_disparity))

        ax.set_title(f"{problem}, {method}")
        ax.plot(total_loss, color="blue", label="Loss")
        ax_r.plot(total_disparity, color="red", linestyle="--")
        ax.set_xlabel("Time Step")

    fig.tight_layout()
    left.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$")
    left.yaxis.label.set_color("blue")
    right_r.set_ylabel("Disparity $\\mathcal{F}(\\rho)$", labelpad=12)
    right_r.yaxis.label.set_color("red")

    right_rr = right.twinx()
    right_rr.spines.right.set_position(("axes", 1.3))
    right_rr.set_ylabel("$\\lambda$", labelpad=12)
    right_rr.plot(
        np.arange(len(lambdas)) + 1,
        lambdas,
        color="black",
        linestyle="dotted",
    )

    # legend
    center.plot([], [], color="red", linestyle="--", label="Disparity")
    center.plot([], [], color="black", linestyle="dotted", label="$\\lambda$")
    center.legend(loc="right")

    for method, ax, ax_r in to_compare:
        ax.set_ylim(lmin - 0.01, lmax + 0.01)
        ax_r.set_ylim(dmin - 0.1, dmax + 0.1)

    left_r.set_yticks([])
    center_r.set_yticks([])

    fig.tight_layout()
    if grad:
        savefig(fig, os.path.join("pdf", f"{problem}_compare.pdf"))
    else:
        savefig(fig, os.path.join("pdf", f"{problem}_compare_fast.pdf"))


def compare_2(problem):
    # save results of loss, disparity, lambda vs time.
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    left, center = axs
    left_r, center_r = left.twinx(), center.twinx()

    to_compare = [
        ("RRM", left, left_r),
        ("LPU", center, center_r),
    ]

    lmin = np.inf
    lmax = -np.inf
    dmin = np.inf
    dmax = -np.inf
    for method, ax, ax_r in to_compare:
        filename = os.path.join("npy", f"{problem}_{method}")
        total_loss = np.load(f"{filename}_total_loss.npy")
        total_disparity = np.load(f"{filename}_total_disparity.npy")

        lmin = min(lmin, min(total_loss))
        lmax = max(lmax, max(total_loss))
        dmin = min(dmin, min(total_disparity))
        dmax = max(dmax, max(total_disparity))

        ax.set_title(f"{problem}, {method}")
        ax.plot(total_loss, color="blue", label="Loss")
        ax_r.plot(total_disparity, color="red", linestyle="--")
        ax.set_xlabel("Time Step")

    fig.tight_layout()
    left.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$")
    left.yaxis.label.set_color("blue")

    # legend
    center.plot([], [], color="red", linestyle="--", label="Disparity")
    center.legend()

    for method, ax, ax_r in to_compare:
        ax.set_ylim(lmin - 0.01, lmax + 0.01)
        ax_r.set_ylim(dmin - 0.1, dmax + 0.1)

    left_r.set_yticks([])
    center_r.set_ylabel("Disparity $\\mathcal{F}(\\rho)$", labelpad=12)
    center_r.yaxis.label.set_color("red")

    fig.tight_layout()
    savefig(fig, os.path.join("pdf", f"{problem}_compare2.pdf"))
