#!/usr/bin/env python3

from functools import partial
from tqdm import tqdm

import jax
import jax.numpy as np
import jax.scipy.optimize
from scipy.spatial import ConvexHull

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from adult import get_achievable_losses


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


def get_hull(achievable_losses):

    # sort by increasing x value, then by increasing y value
    # vals = [tuple(v) for v in achievable_losses]
    # vals.sort()
    # achievable_losses = np.array(vals)

    hull = ConvexHull(achievable_losses)
    hull.vertices.sort()

    # filter for Pareto property
    def is_pareto(idx):
        """
        remove all points that can be strictly improved upon
        O(n^2)
        """
        x = achievable_losses[idx][0]
        y = achievable_losses[idx][1]
        for idx_p in hull.vertices:
            if idx == idx_p:
                continue
            x_p = achievable_losses[idx_p][0]
            y_p = achievable_losses[idx_p][1]
            if (x_p <= x) and (y_p <= y):
                return False

        return True

    return np.array([achievable_losses[idx] for idx in hull.vertices if is_pareto(idx)])


class Env:
    def __init__(
        self,
        achievable_losses,
        rho_fns,
        group_sizes,
        disparity_fn,
        inverse_disparity_curve,
        eta,
    ):
        """
        achievable losses: an array of losses acheivable with fixed policies.
        rho_fns: two functions (one per group) that maps group loss -> participation.
        group_sizes: array of relative group sizes.
        eta: learning rate
        """
        self.achievable_losses = achievable_losses

        self.hull = get_hull(achievable_losses)
        n = len(self.hull)
        self.xs = self.hull[:, 0]
        self.ys = self.hull[:, 1]
        self.ts = (
            (
                (
                    np.array(
                        [  # between 0 and 1
                            (np.arctan2(self.ys[idx], self.xs[idx])) / (np.pi / 2)
                            for idx in range(n)
                        ]
                    )
                    + 4.0
                )
                % 2.0
            )
            * np.pi
            / 2
        )

        self.rho_fns = rho_fns
        self.grad_rho_fns = [jax.jacfwd(rho_fn) for rho_fn in rho_fns]

        self.group_sizes = group_sizes

        self.disparity_fn = disparity_fn
        self.grad_disparity_fn = jax.grad(disparity_fn)

        self.inverse_disparity_curve = inverse_disparity_curve

        self.eta = eta

    def get_losses(self, theta):
        """
        theta [0, 1] -> group_specific losses
        """

        x = np.interp(theta, self.ts, self.xs)
        y = np.interp(theta, self.ts, self.ys)
        return np.array([x, y])

    def get_grad_losses(self, theta):
        """
        Use finite differences.
        """
        h = 0.000001
        return (self.get_losses(theta + h / 2) - self.get_losses(theta - h / 2)) / h

    def get_rhos(self, losses):
        return np.array([self.rho_fns[g](losses[g]) for g in range(2)])

    def get_rho_grads(self, losses):
        return np.array([self.grad_rho_fns[g](losses[g]) for g in range(2)])

    def get_total_loss(self, theta):
        losses = self.get_losses(theta)
        rhos = self.get_rhos(losses)
        return np.einsum("g,g,g->", losses, rhos, self.group_sizes)

    def get_total_disparity(self, theta):
        losses = self.get_losses(theta)
        rhos = self.get_rhos(losses)
        return self.disparity_fn(rhos)

    ############################################################################

    def rrm_step(self, theta, losses, rhos):
        """
        Perform update step assuming fixed rho
        """
        grads = self.get_grad_losses(theta)

        return (
            0,
            theta - self.eta * np.einsum("g,g,g->", rhos, grads, self.group_sizes),
        )

    def perf_step(self, theta, losses, rhos):
        """
        Perform update step with rho_hat
        """
        rhos_hat = rhos + self.get_rho_grads(losses) * losses
        losses_grads = self.get_grad_losses(theta)
        return (
            0,
            theta
            - self.eta * np.einsum("g,g,g->", losses_grads, rhos_hat, self.group_sizes),
        )

    def fair_step(self, theta, losses, rhos):

        # pdv{rho_g}{l_g} [g] (is diagonal)
        rhos_grad = self.get_rho_grads(losses)
        # pdv{l_g}{theta} [g]
        losses_grad = self.get_grad_losses(theta)

        # [g]
        rhos_hat = rhos + rhos_grad * losses
        perf_grad = np.einsum("g,g,g->", losses_grad, rhos_hat, self.group_sizes)

        fair_grad = np.einsum(
            "g,g,g->",
            self.grad_disparity_fn(rhos),  # \pdv{F}{rho_g} [g]
            rhos_grad,  # \pdv{rho_g}{l_g} [g]
            losses_grad,  # \pdv{l_g}{theta} [g]
        )

        g = self.disparity_fn(rhos)
        d = np.einsum(",->", perf_grad, fair_grad) / np.einsum(
            ",->", fair_grad, fair_grad
        )
        lamda = np.maximum(g - d, 0)

        return (lamda, theta - self.eta * (perf_grad + lamda * fair_grad))


################################################################################


class Video:
    """
    Use a matplotlib figure to make a video.
    For each frame must:
      1. draw to figure
      2. call the video.draw method
      3. clear the figure/axes/Artists

    Example:

    fig, ax = plt.subplots(figsize=(6, 6))

    with Video('video_name', fig) as video:
        for _ in range(num_frames):
            render_to_fig()
            video.draw()
            ax.cla()
    """

    def __init__(self, title, fig, fps, dpi):
        self.video_file = title + ".mp4"

        # ffmpeg backend
        self.writer = animation.FFMpegWriter(
            fps=fps, metadata={"title": title, "artist": "Matplotlib"}
        )

        # canvas
        self.fig = fig
        self.dpi = dpi

    def __enter__(self):
        # initialize writer
        self.writer.setup(self.fig, self.video_file, self.dpi)
        return self

    def draw(self):
        # save frame
        self.writer.grab_frame()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # write file and exit
        self.writer.finish()
        print("Writing", self.video_file)


class Viz(Video):
    def __init__(self, title, env):

        self.env = env

        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        super().__init__(title, fig, fps=15, dpi=100)

        self.setup_axs(axs)

    def setup_axs(self, axs):
        left, center, right = axs

        # Plot achievable losses
        achievable_losses = self.env.achievable_losses

        left.scatter(*achievable_losses.T, color="black", label="Fixed Policies")

        left.plot(self.env.xs, self.env.ys, "black", label="Pareto Boundary")

        # lims = [
        #     self.env.get_losses(self.env.ts[1] - 0.005),
        #     self.env.get_losses(self.env.ts[-2] + 0.005),
        # ]
        # left.set_xlim(lims[0][0], lims[1][0])
        # left.set_ylim(lims[1][1], lims[0][1])
        left.set_xlim(-1, 0)
        left.set_ylim(-1, 0)
        left.set_xlabel("Group 1 loss $\\ell_1$")
        left.set_ylabel("Group 2 loss $\\ell_2$")
        left.set_title("Group Losses")

        # use half as many ticks
        left.set_xticks(left.get_xticks()[::2])
        left.set_yticks(left.get_yticks()[::2])
        left.legend(loc="upper right")

        # plot achievable rhos
        theta_range = np.linspace(0, np.pi / 2, 100)
        achievable_rhos = np.array(
            [self.env.get_rhos(self.env.get_losses(theta)) for theta in theta_range]
        )
        center.plot(*achievable_rhos.T, color="black", label="Pareto Boundary")
        center.set_title("Group Participation Rates")

        cx, cy = self.env.inverse_disparity_curve()
        center.plot(cx, cy, color="red", linestyle="--")
        center.plot(cy, cx, color="red", linestyle="--")

        center.set_xlabel("Group 1 Participation Rate $\\rho_1$")
        center.set_ylabel("Group 2 Participation Rate $\\rho_2$")
        center.legend(loc="upper right")

        # plot performative loss and fairness surface

        right_r = right.twinx()

        # plot loss curve
        right.plot(
            theta_range,
            [self.env.get_total_loss(theta) for theta in theta_range],
            "blue",
            label="Loss",
        )

        # plot disparity curve
        right_r.plot(
            theta_range,
            [self.env.get_total_disparity(theta) for theta in theta_range],
            "red",
            linestyle="--",
            label="Disparity",
        )

        # right_r.plot()

        right.set_title("Loss and Disparity Surfaces")
        right.set_xlabel("Parameter $\\theta$")
        right.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$")
        right_r.set_ylabel("Disparity $\\mathcal{F}(\\rho)$")

        right.legend(loc="lower left")
        right_r.legend(loc="lower right")
        right.set_xlim(0.5, 1.0)

        self.fig.tight_layout()

        self.left = left
        self.center = center
        self.right = right
        self.right_r = right_r

        self.fig.savefig("adult.pdf")

    def update_left(self, ax, lamda, theta, losses, rhos):
        """
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rhos
        """

        points = [ax.scatter([losses[0]], [losses[1]], color="red", marker="^", s=100)]

        t = np.arctan(rhos[1] / rhos[0])
        l = self.env.get_losses(t)
        d = np.einsum("g,g->", rhos, l) / np.einsum("g,g->", rhos, rhos)
        dual_vec = [ax.plot([d * rhos[0], 0], [d * rhos[1], 0], "red")[0]]

        return points + dual_vec

    def update_center(self, ax, lamda, theta, losses, rhos):
        """
        plot achieved rho
        """

        return [ax.scatter([rhos[0]], [rhos[1]], color="red", marker="^", s=100)]

    def update_right(self, ax, lamda, theta, losses, rhos):
        """
        lamda defaults to zero if first run
        """

        # current actual loss
        artifacts = [
            ax.scatter(
                theta,
                self.env.get_total_loss(theta),
                color="red",
                marker="^",
                s=100,
            )
        ]

        theta_range = np.linspace(0, 1, 100) * np.pi / 2  # [i]
        tl = np.array([self.env.get_total_loss(theta) for theta in theta_range])
        td = np.array([self.env.get_total_disparity(theta) for theta in theta_range])

        artifacts += [
            ax.plot(theta_range, tl + lamda * td, color="black", linestyle="--")[0]
        ]

        return artifacts

        # queried_rho_grad = rho_grad(losses)
        # rho_prox = rho + np.einsum("gh,h->g", queried_rho_grad, losses)

        # if RRM:
        #     # greedy approx of local loss, (assuming rho does not change)
        #     artifacts.append(
        #         ax.plot(
        #             theta_range, [loss_fn(x, rho) for x in Ls], "red", linestyle="--"
        #         )[0]
        #     )

        # elif PERF:
        #     # local approx of performative loss (approx rho as linear in L)
        #     artifacts.append(
        #         ax.plot(
        #             theta_range,
        #             [loss_fn(x, rho_prox) for x in Ls],
        #             "red",
        #             linestyle="--",
        #         )[0]
        #     )
        # elif FAIR:
        #     # loss + disparity surface
        #     artifacts.append(
        #         ax.plot(
        #             theta_range, [reg_loss_fn(L, query_rho(L), u) for L in Ls], "blue"
        #         )[0]
        #     )
        #     # current loss with fairness regularization
        #     artifacts.append(ax.plot(theta, reg_loss_fn(L, rho, u), "go")[0])

        #     # local approximation of loss with fairness, which we perform gradient descent wrt
        #     artifacts.append(
        #         ax.plot(
        #             theta_range,
        #             [local_reg_loss_fn(x, L, rho_prox, u) for x in Ls],
        #             "blue",
        #             linestyle="--",
        #         )[0]
        #     )

        #     artifacts.append(
        #         ax.plot(theta, local_reg_loss_fn(L, L, rho_prox, u), "bo")[0]
        #     )

    def render_frame(self, lamda, theta, losses, rhos):

        to_remove = self.update_left(self.left, lamda, theta, losses, rhos)
        to_remove.extend(self.update_center(self.center, lamda, theta, losses, rhos))
        to_remove.extend(self.update_right(self.right, lamda, theta, losses, rhos))

        self.draw()

        for obj in to_remove:
            obj.remove()


################################################################################
################################################################################
################################################################################


def disparity_fn(rhos):
    """
    Assumed to be symmetric

    Get violation of fairness constraint

    Args:
        rho: array of participation rates indexed by g
    """
    return np.var(rhos) - 0.01


def inverse_disparity_curve():
    rho_1 = np.linspace(0, 1, 100)
    rho_2 = np.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


def concave_rho_fn(loss):
    """
    Monotonically decreasing and concave.
    """
    return 1 - 1 / (1 - loss * 2)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def localized_rho_fn(center, sensitivity, loss):
    """
    Monotonically decreasing. Not concave.
    """
    return 1 - np.clip(logistic((loss - center) * sensitivity), 0, 1)


def main(method="rrm"):

    try:
        filename = "achievable_losses.npy"
        achievable_losses = np.load(filename)
        print(f"Loaded Cached Achievable Losses from {filename}")
    except OSError:
        print("Determining Achievable Losses")
        achievable_losses = get_achievable_losses(50)
        np.save("achievable_losses", achievable_losses)

    env = Env(
        achievable_losses,
        rho_fns=[
            # concave_rho_fn,
            # concave_rho_fn,
            partial(localized_rho_fn, -0.85, 20),
            partial(localized_rho_fn, -0.85, 20),
        ],
        group_sizes=np.array([0.5, 0.5]),
        disparity_fn=disparity_fn,
        inverse_disparity_curve=inverse_disparity_curve,
        eta=0.03,
    )

    if method.startswith("rrm"):
        update_func = jax.jit(env.rrm_step)
    elif method.startswith("perf"):
        update_func = jax.jit(env.perf_step)
    else:
        update_func = jax.jit(env.fair_step)

    fig, ax = plt.subplots(figsize=(6, 6))

    with Viz(method, env) as viz:

        theta = 0.6 * np.pi / 2
        losses = env.get_losses(theta)
        rhos = env.get_rhos(losses)

        viz.render_frame(0, theta, losses, rhos)

        total_loss = []
        total_disparity = []
        lamdas = []
        for i in tqdm(range(50)):

            lamda, theta = update_func(theta, losses, rhos)
            losses = env.get_losses(theta)
            rhos = env.get_rhos(losses)

            total_loss.append(env.get_total_loss(theta))
            total_disparity.append(env.get_total_disparity(theta))
            lamdas.append(lamda)

            viz.render_frame(lamda, theta, losses, rhos)

        ax_r = ax.twinx()
        ax.plot(total_loss, color="blue", label="Loss")
        ax.plot(total_disparity, color="red", linestyle="--", label="Disparity")
        ax.plot([0 for _ in total_loss], color="black")
        ax_r.plot(lamdas, color="black", linestyle="dotted", label="$\\lambda$")
        ax.set_xlabel("Time")
        ax.legend()
        ax.set_title("Loss, Disparity vs. Time")
        fig.savefig(f"{method}.pdf")


if __name__ == "__main__":
    # main("fair_outer")  # initial theta = 0.6 * pi / 2
    # main("fair_inner") # initial theta = 0.6 * pi / 2
    main("rrm")  # initial theta = 0.6 * pi / 2
    # for method in ["rrm", "perf", "fair"]:
    #     main(method)
