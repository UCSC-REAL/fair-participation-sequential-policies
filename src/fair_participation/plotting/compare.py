import pandas as pd
from fair_participation.plotting.plot_utils import savefig
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba as to_rgba
import matplotlib.path as mpath
import numpy as np
from jax import numpy as jnp
import os

from fair_participation.environment import Environment
from fair_participation.simulation import get_trial_filename

from fair_participation.plotting.loss_boundary_plot import make_loss_boundary_plot
from fair_participation.plotting.participation_rate_plot import (
    make_participation_rate_plot,
)
from fair_participation.plotting.loss_disparity_plot import make_loss_disparity_plot

from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT

star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
cut_star = mpath.Path(
    vertices=np.concatenate([circle.vertices, star.vertices[::-1, ...]]),
    codes=np.concatenate([circle.codes, star.codes]),
)


def load_methods(name: str, methods: list[str]) -> pd.DataFrame:
    df = pd.DataFrame()
    for i, method in enumerate(methods):
        trial_filename = get_trial_filename(name, method)
        with jnp.load(trial_filename) as npz:
            method_df = pd.DataFrame()
            for item in npz.files:
                vals = npz[item]
                if vals.ndim == 1:
                    method_df[item] = vals
                else:
                    for j, val in enumerate(vals.T):
                        method_df[f"{item}_{j}"] = val
            method_df["method"] = method
            method_df["Timestep"] = np.arange(len(method_df))
        df = pd.concat([df, method_df])
    return df


def make_canvas(env: Environment) -> tuple:
    """
    Makes a canvas for plotting the loss boundary, participation rate, and loss.

    :param env: Environment object
    :return: tuple of (fig, axes)
    """

    num_groups = env.group_sizes.shape[0]
    if num_groups == 2:
        fig, (lax, cax, rax) = plt.subplots(
            1, 3, figsize=(18, 5), subplot_kw=dict(box_aspect=1)
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
            fair_epsilon=env.fair_epsilon,
        )
        right_plot = make_loss_disparity_plot(
            ax=rax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
            values_and_grads=env.values_and_grads,
        )

        rbox = rax.get_position()
        rbox.x0 += 0.05 # shift right
        rbox.x1 += 0.05 # shift right
        rax.set_position(rbox)
        cbox = cax.get_position()
        cbox.x0 += 0.06 # shift right
        cbox.x1 += 0.06 # shift right
        cax.set_position(cbox)
        lbox = lax.get_position()
        lbox.x0 += 0.07 # shift right
        lbox.x1 += 0.07 # shift right
        lax.set_position(lbox)

        plots = (left_plot, center_plot, right_plot)

    elif num_groups == 3:
        # TODO change
        fig, (lax, cax) = plt.subplots(
            1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"}
        )

        left_plot = make_loss_boundary_plot(
            ax=lax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
        )
        bbox = lax.get_position()
        bbox.x0 += 0.1 # shift right
        bbox.x1 += 0.1 # shift right
        lax.set_position(bbox)
        center_plot = make_participation_rate_plot(
            ax=cax,
            achievable_loss=env.achievable_loss,
            loss_hull=env.loss_hull,
            values_and_grads=env.values_and_grads,
            fair_epsilon=env.fair_epsilon,
        )
        bbox = cax.get_position()
        bbox.x0 += 0.1 # shift right
        bbox.x1 += 0.1 # shift right
        cax.set_position(bbox)

        plots = (left_plot, center_plot, None)
    else:
        raise NotImplementedError(f"Cannot plot {num_groups} groups.")

    return fig, plots


def get_compare_solutions_filename(name: str) -> str:
    return os.path.join(PROJECT_ROOT, "pdf", f"{name}_solutions.pdf")


def compare_solutions(env: Environment, methods: list[str]) -> None:
    """
    Compare solutions for a given environment.

    :param env: Environment object.
    :param methods: List of methods to compare.
    :return: None
    """
    save_filename = get_compare_solutions_filename(env.name)
    logger.info(f"Rendering graphic:")
    logger.info(f"  {save_filename}")

    fig, (left, center, right) = make_canvas(env)
    df = load_methods(env.name, methods)

    cb_colors = sns.color_palette("colorblind")

    markers = {
        "Initialization": {
            "marker": "s",
            "color": to_rgba(cb_colors[7], alpha=0.8),
            "s": 300,
        },
        "RRM": {
            "marker": "D",
            "color": to_rgba(cb_colors[2], alpha=0.8),
            "s": 240,
        },
        "MPG": {
            "marker": "o",
            "color": to_rgba(cb_colors[4], alpha=0.8),
            "s": 240,
        },
        "CPG": {
            "marker": cut_star,
            "color": to_rgba(cb_colors[1], alpha=0.9),
            "s": 260,
        },
    }

    def _marker_map(key: str) -> dict:
        return {k: v[key] for k, v in markers.items()}

    init_data = df[df.index == df["Timestep"].min()][:1]
    init_data["method"] = "Initialization"
    data = df[df.index == df["Timestep"].max()]
    data = pd.concat((init_data, data))

    marker_kwargs = {
        "size": "method",
        "sizes": _marker_map("s"),
        "style": "method",
        "markers": _marker_map("marker"),
        "hue": "method",
        "palette": _marker_map("color"),
        "linewidth": 0.0,
    }
    # left.ax.set_clip_on(False)
    sns.scatterplot(
        data=data,
        x="loss_0",
        y="loss_1",
        **marker_kwargs,
        ax=left.ax,
    )
    left.ax.legend(
        loc=(-0.85, 0.3),
        frameon=True,
        title=None,
    )
    sns.scatterplot(
        data=data,
        x="rho_0",
        y="rho_1",
        ax=center.ax,
        **marker_kwargs,
        legend=False,
    )

    if right is not None:
        phi = [right.get_phi(loss) for loss in data[["loss_0", "loss_1"]].values]
        for ax, y in zip([right.ax, right.ax_r], ["total_loss", "disparity"]):
            sns.scatterplot(data=data, x=phi, y=y, **marker_kwargs, ax=ax, legend=False)
    for ax in (left.ax, center.ax, right.ax, right.ax_r):
        plt.setp(ax.collections, clip_on=False, zorder=10)

    # plt.subplots_adjust(left=0.05, right=0.95)
    savefig(save_filename)


def compare_solutions_3D(env, methods):

    save_filename = get_compare_solutions_filename(env.name)
    if os.path.exists(save_filename):
        logger.info("Graphic exists; skipping:")
        logger.info(f"  {save_filename}")
        return
    logger.info(f"Rendering graphic:")
    logger.info(f"  {save_filename}")

    fig, plots = make_canvas(env)

    left, center = plots[0].ax, plots[1].ax
    if len(plots) > 2:
        right_p = plots[2]
    else:
        right_p = None


    cb_colors = sns.color_palette("colorblind")

    markers = {
        "Initialization": {
            "marker": "s",
            "color": to_rgba(cb_colors[7], alpha=0.8),
            "s": 300,
        },
        "RRM": {
            "marker": "D",
            "color": to_rgba(cb_colors[2], alpha=0.8),
            "s": 240,
        },
        "MPG": {
            "marker": "o",
            "color": to_rgba(cb_colors[4], alpha=0.8),
            "s": 240,
        },
        "CPG": {
            "marker": cut_star,
            "color": to_rgba(cb_colors[1], alpha=0.9),
            "s": 260,
        },
    }

    for method in methods:

        trial_filename = get_trial_filename(env.name, method)
        with jnp.load(trial_filename) as npz:
            loss = npz["loss"][-1]
            rho = npz["rho"][-1]
            total_loss = npz["total_loss"][-1]
            disparity = npz["disparity"][-1]

            left.scatter(*loss, **markers[method], label=method)
            center.scatter(*rho, **markers[method], label=method)

            if right_p is not None:
                right_p.ax.scatter(
                    right_p.get_phi(loss), total_loss, **markers[method], label=method
                )
                right_p.ax_r.scatter(
                    right_p.get_phi(loss), disparity, **markers[method], label=method
                )

    # show init
    method = "Initialization"
    loss = env.init_loss
    results = env.values_and_grads(loss)
    rho = results["rho"]
    total_loss = results["total_loss"]
    disparity = results["disparity"]

    left.scatter(*loss, **markers[method], label=method)
    center.scatter(*rho, **markers[method], label=method)
    left.legend(loc=(-0.6, 0.3))

    if right_p is not None:
        right_p.ax.scatter(
            right_p.get_phi(loss), total_loss, **markers[method], label=method
        )
        right_p.ax_r.scatter(
            right_p.get_phi(loss), disparity, **markers[method], label=method
        )

        ticks = right_p.ax_r.get_yticks()
        right_p.ax_r.set_yticks([ticks[0], 0, ticks[-1]])
        ticks = right_p.ax.get_xticks()
        right_p.ax.set_xticks([ticks[0], ticks[-1]])
        ticks = right_p.ax.get_yticks()
        right_p.ax.set_yticks([ticks[0], ticks[-1]])

    savefig(save_filename)


def get_compare_timeseries_filename(name: str) -> str:
    return os.path.join(PROJECT_ROOT, "pdf", f"{name}_time_series.pdf")


def compare_timeseries(name: str, methods: list[str]) -> None:
    """
    Compare time series for a given environment.

    :param name: Name of the problem.
    :param methods: List of methods to compare.
    :return: None.
    """
    save_filename = get_compare_timeseries_filename(name)
    logger.info(f"Rendering graphic:")
    logger.info(f"  {save_filename}")

    base_theme_kwargs = {
        "style": "white",
        "context": "paper",
        "font": "serif",
    }
    base_rcparams = {
        "text.usetex": True,
        "axes.labelsize": 20.0,
        "xtick.labelsize": 18.0,
        "ytick.labelsize": 18.0,
        "axes.titlesize": 20.0,
        "legend.fontsize": 18.0,
        "xtick.bottom": True,
        # "xtick.color": ".8",
    }
    sns.set_theme(
        **base_theme_kwargs,
        font_scale=3,
        rc={
            **base_rcparams,
            "lines.linewidth": 3,
            "legend.fontsize": 16.5,
        },
    )
    fig = plt.figure(figsize=(6, 5.25))
    ax = plt.axes()
    ax_r = ax.twinx()

    plt.title(f"{name} Task")
    ax.tick_params("y", colors="blue")
    ax_r.tick_params("y", colors="red")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Total Loss $\\mathcal{L}$")
    ax.yaxis.label.set_color("blue")
    ax_r.set_ylabel("Disparity $\\mathcal{H}$", labelpad=2)
    ax_r.yaxis.label.set_color("red")

    df = load_methods(name, methods)
    n_methods = len(methods)

    sns.lineplot(
        data=df,
        x="Timestep",
        y="total_loss",
        hue="method",
        style="method",
        palette=sns.color_palette("Blues", n_colors=n_methods),
        ax=ax,
        legend=False,
    )

    sns.lineplot(
        data=df,
        x="Timestep",
        y="disparity",
        hue="method",
        style="method",
        palette=sns.color_palette("Reds", n_colors=n_methods),
        ax=ax_r,
    )
    ax_r.legend(
        loc="upper center",
        frameon=True,
        title=None,
        bbox_to_anchor=(0.5, -0.25),
        ncol=n_methods,
    )
    for line in ax_r.get_legend().get_lines():
        line.set_color("black")

    # lambdas = npz["lambda_estimate"]
    # if sum(lambdas) > 0:
    #     ax_rr = ax.twinx()
    #     ax_rr.spines.right.set_position(("axes", 1.3))
    #     ax_rr.set_ylabel("$\\lambda$", labelpad=15)
    #
    #     ax_rr.plot(
    #         np.arange(num_steps),
    #         lambdas,
    #         color="black",
    #         linestyle="dotted",
    #         label="$\\lambda$",
    #     )
    #     ax_rr.plot([], [], color="blue", label="Loss")
    #     ax_rr.plot([], [], color="red", linestyle="dashed", label="$\\mathcal{H}$")
    #     # legend
    #     ax_rr.legend()

    ax.set_ylim(df["total_loss"].min() - 0.01, df["total_loss"].max() + 0.01)
    ax_r.set_ylim(df["disparity"].min() - 0.1, df["disparity"].max() + 0.1)

    fig.tight_layout(pad=1.0)
    savefig(save_filename, bbox_inches="tight")
