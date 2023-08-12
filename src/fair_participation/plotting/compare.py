#!/usr/bin/env python3

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from jax import numpy as jnp
import os

from fair_participation.simulation import get_trial_filename

from fair_participation.plotting.loss_boundary_plot import make_loss_boundary_plot
from fair_participation.plotting.participation_rate_plot import (
    make_participation_rate_plot,
)
from fair_participation.plotting.loss_disparity_plot import make_loss_disparity_plot
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


def make_canvas(env):
    """
    save_init is filename
    """

    num_groups = (env.group_sizes.shape)[0]
    if num_groups == 2:
        fig, (lax, cax, rax) = plt.subplots(1, 3, figsize=(18, 6))

        left_plot = make_loss_boundary_plot(
            ax=lax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
        )
        center_plot = make_participation_rate_plot(
            ax=cax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
            values_and_grads=env.values_and_grads,
        )
        right_plot = make_loss_disparity_plot(
            ax=rax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
            values_and_grads=env.values_and_grads,
        )

        plots = (left_plot, center_plot, right_plot)

    elif num_groups == 3:
        fig, (lax, cax) = plt.subplots(
            1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"}
        )

        left_plot = make_loss_boundary_plot(
            ax=lax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
        )
        center_plot = make_participation_rate_plot(
            ax=cax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
            values_and_grads=env.values_and_grads,
        )

        plots = (left_plot, center_plot)

    fig.tight_layout()

    return fig, plots


def get_compare_solutions_filename(name):
    return os.path.join(PROJECT_ROOT, "pdf", f"{name}_solutions.pdf")


def compare_solutions(env, methods):

    save_filename = get_compare_solutions_filename(env.name)
    if os.path.exists(save_filename):
        logger.info("Graphic exists; skipping:")
        logger.info(f"  {save_filename}")
        return
    logger.info(f"Rendering graphic:")
    logger.info(f"  {save_filename}")

    fig, plots = make_canvas(env)

    left, center = plots[0], plots[1]

    markers = ["o", "+"]

    for (method, marker) in zip(methods, markers):

        trial_filename = get_trial_filename(env.name, method)
        with jnp.load(trial_filename) as npz:
            loss = npz["loss"][-1]
            rho = npz["rho"][-1]

            left.scatter(*loss, marker)
            center.scatter(*rho, marker)

    savefig(fig, save_filename)


def get_compare_timeseries_filename(name):
    return os.path.join(PROJECT_ROOT, "pdf", f"{name}_time_series.pdf")


def compare_timeseries(name, methods):

    save_filename = get_compare_timeseries_filename(name)
    if os.path.exists(save_filename):
        logger.info("Graphic exists; skipping.")
        logger.info(f"  {save_filename}")
        return
    logger.info(f"Rendering graphic:")
    logger.info(f"  {save_filename}")

    num_methods = len(methods)
    fig, axs = plt.subplots(1, num_methods, figsize=(5 * num_methods, 5), sharey=True)
    if num_methods == 1:
        axs = [axs]
    axs_r = [ax.twinx() for ax in axs]

    # load and compare data
    num_steps = None
    loss_min = np.inf
    loss_max = -np.inf
    disparity_min = np.inf
    disparity_max = -np.inf
    for i, method in enumerate(methods):

        trial_filename = get_trial_filename(name, method)

        if os.path.exists(trial_filename):

            with jnp.load(trial_filename) as npz:
                loss_min = min(loss_min, min(npz["total_loss"]))
                loss_max = max(loss_max, max(npz["total_loss"]))
                disparity_min = min(disparity_min, min(npz["disparity"]))
                disparity_max = max(disparity_max, max(npz["disparity"]))

                axs[i].set_title(f"{name}, {method}")
                axs[i].plot(npz["total_loss"], color="blue", label="Loss")
                axs[i].tick_params("y", colors="blue")
                axs_r[i].plot(
                    npz["disparity"], color="red", linestyle="--", label="Disparity"
                )
                axs_r[i].tick_params("y", colors="red")
                axs[i].set_xlabel("Time Step")

                lambdas = npz["lambda_estimate"]
                if sum(lambdas) > 0:
                    ax_rr = axs[i].twinx()
                    ax_rr.spines.right.set_position(("axes", 1.3))
                    ax_rr.set_ylabel("$\\lambda$", labelpad=15)

                    ax_rr.plot(
                        np.arange(num_steps),
                        lambdas,
                        color="black",
                        linestyle="dotted",
                        label="$\\lambda$",
                    )
                    ax_rr.plot([], [], color="blue", label="Loss")
                    ax_rr.plot(
                        [], [], color="red", linestyle="dashed", label="$\\mathcal{H}$"
                    )
                    # legend
                    ax_rr.legend(loc="right")

                if num_steps is None:
                    num_steps = len(npz["total_loss"])
        else:
            raise FileNotFoundError(trial_filename)

    axs[0].set_ylabel("Total Loss $\\mathcal{L}$")
    axs[0].yaxis.label.set_color("blue")
    for ax in axs_r[:-1]:
        ax.set_yticks([])
    axs_r[-1].set_ylabel("Disparity $\\mathcal{H}$", labelpad=2)
    axs_r[-1].yaxis.label.set_color("red")

    for ax, ax_r in zip(axs, axs_r):
        ax.set_ylim(loss_min - 0.01, loss_max + 0.01)
        ax_r.set_ylim(disparity_min - 0.1, disparity_max + 0.1)

    fig.tight_layout()
    savefig(fig, save_filename)
