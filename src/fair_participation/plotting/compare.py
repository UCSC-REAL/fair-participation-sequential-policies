#!/usr/bin/env python3

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
import os

from fair_participation.plotting.plot_utils import savefig
from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT

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


def compare(problems, fig_filename):

    num_problems = len(problems)
    fig, axs = plt.subplots(1, num_problems, figsize=(5 * num_problems, 5), sharey=True)
    if num_problems == 1:
        axs = [axs]
    axs_r = [ax.twinx() for ax in axs]

    # load and compare data
    num_steps = None
    loss_min = np.inf
    loss_max = -np.inf
    disparity_min = np.inf
    disparity_max = -np.inf
    for i, problem in enumerate(problems):

        npz_filename = os.path.join(
            PROJECT_ROOT, "npz", f"{problem['name']}_{problem['method']}.npz"
        )

        if os.path.exists(npz_filename):
            logger.info(f"Found {npz_filename}.")
            logger.info("Loading simulation data.")

            with jnp.load(npz_filename) as npz:
                loss_min = min(loss_min, min(npz["total_loss"]))
                loss_max = max(loss_max, max(npz["total_loss"]))
                disparity_min = min(disparity_min, min(npz["disparity"]))
                disparity_max = max(disparity_max, max(npz["disparity"]))

                axs[i].set_title(f"{problem['name']}, {problem['method']}")
                axs[i].plot(npz["total_loss"], color="blue", label="Loss")
                axs_r[i].plot(
                    npz["disparity"], color="red", linestyle="--", label="Disparity"
                )
                axs[i].set_xlabel("Time Step")

                lambdas = npz["lambda_estimate"]
                if sum(lambdas) > 0:
                    ax_rr = axs[i].twinx()
                    ax_rr.spines.right.set_position(("axes", 1.3))
                    ax_rr.set_ylabel("$\\lambda$", labelpad=12)

                    ax_rr.plot(
                        np.arange(num_steps),
                        lambdas,
                        color="black",
                        linestyle="dotted",
                        label="$\\lambda",
                    )

                    # legend
                    ax_rr.legend(loc="right")

                if num_steps is None:
                    num_steps = len(npz["total_loss"])
        else:
            raise FileNotFoundError(npz_filename)

    fig.tight_layout()
    axs[0].set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$")
    axs[0].yaxis.label.set_color("blue")
    axs_r[-1].set_ylabel("Disparity $\\mathcal{F}(\\rho)$", labelpad=12)
    axs_r[-1].yaxis.label.set_color("red")

    for ax, ax_r in zip(axs, axs_r):
        ax.set_ylim(loss_min - 0.01, loss_max + 0.01)
        ax_r.set_ylim(disparity_min - 0.1, disparity_max + 0.1)

    savefig(fig, fig_filename)
