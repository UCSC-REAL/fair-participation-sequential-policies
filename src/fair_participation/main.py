import os
from tqdm import tqdm

import cvxpy as cp
import numpy as onp

import jax
import jax.numpy as np
import jax.scipy.optimize
from scipy.spatial import ConvexHull

import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from fair_participation.folktasks import get_achievable_losses


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
    print("Saving", filename)
    fig.savefig(filename)


def use_two_ticks_x(ax):
    x = ax.get_xticks()
    ax.set_xticks(x[:: len(x) - 1])


def use_two_ticks_y(ax):
    y = ax.get_yticks()
    ax.set_yticks(y[:: len(y) - 1])


def get_hull(achievable_losses):
    min_g1_loss = np.min(achievable_losses[:, 0])
    min_g2_loss = np.min(achievable_losses[:, 1])
    achievable_losses = list(achievable_losses)
    achievable_losses.append([min_g1_loss, 0])
    achievable_losses.append([0, min_g2_loss])
    achievable_losses = np.array(achievable_losses)

    hull = ConvexHull(achievable_losses)

    # filter for Pareto property
    def is_pareto(idx):
        """
        remove all points that can be strictly improved upon
        """
        x = achievable_losses[idx][0]
        y = achievable_losses[idx][1]
        for idx_p in hull.vertices:
            if idx == idx_p:
                continue
            x_p = achievable_losses[idx_p][0]
            y_p = achievable_losses[idx_p][1]
            if (x > x_p) and (y > y_p):
                return False

        return True

    pareto_hull = np.array(
        [achievable_losses[idx] for idx in hull.vertices if is_pareto(idx)]
    )
    # sort by increasing group 1 loss
    return pareto_hull[pareto_hull[:, 0].argsort()]


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

        self.update_funcs = {
            "RRM": self.rrm_step,
            "RRM_grad": self.rrm_grad_step,
            "LPU": self.perf_step,
            "LPU_grad": self.perf_grad_step,
            "Fair": self.fair_step,
            "Fair_grad": self.fair_grad_step,
        }

    def get_losses(self, theta):
        """
        theta [0, 1] -> group_specific losses
        """

        x = np.interp(theta, self.ts, self.xs)
        y = np.interp(theta, self.ts, self.ys)
        return np.array([x, y])

    def get_theta(self, losses):
        return ((np.arctan2(losses[1], losses[0]) / (np.pi / 2) + 4.0) % 2.0) * (
            np.pi / 2
        )

    def get_grad_losses(self, theta):
        """
        Use finite differences.
        """
        h = 0.0001
        return (self.get_losses(theta + h / 2) - self.get_losses(theta - h / 2)) / h

    def get_tangent(self, theta):
        i = np.sum(self.ts < theta) - 1
        return np.array([self.xs[i + 1] - self.xs[i], self.ys[i + 1] - self.ys[i]])

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

    def quadratic_program(self, losses, dual):
        """
        return theta that solves convex proximal update
        """
        x = cp.Variable(2)
        constraints = [
            onp.array([1, 0]) @ x <= 0,
            onp.array([0, 1]) @ x <= 0,
        ]
        for i in range(len(self.hull) - 1):
            l = self.hull[i]
            r = self.hull[i + 1]
            d = np.array([r[1] - l[1], l[0] - r[0]])
            constraints.append(d.T @ x <= d.T @ l)

        prob = cp.Problem(
            cp.Minimize(
                (1 / 2) * cp.quad_form(x - losses, onp.eye(2)) + self.eta * dual.T @ x
            ),
            constraints,
        )
        prob.solve()
        return self.get_theta(x.value)

    def rrm_step(self, theta, losses, rhos):
        """
        Perform update step assuming fixed rho
        """

        # A_losses = np.array([self.xs, self.ys])

        # return (
        #     0,
        #     self.ts[
        #         np.argmin(np.einsum("g,gi,g->i", rhos, A_losses, self.group_sizes))
        #     ],
        # )

        return (0, self.quadratic_program(losses, rhos * self.group_sizes))

    def rrm_grad_step(self, theta, losses, rhos):
        """
        Perform gradient update step assuming fixed rho
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

        # A_losses = np.array([self.xs, self.ys])

        # return (
        #     0,
        #     self.ts[
        #         np.argmin(np.einsum("g,gi,g->i", rhos_hat, A_losses, self.group_sizes))
        #     ],
        # )

        return (0, self.quadratic_program(losses, rhos_hat * self.group_sizes))

    def perf_grad_step(self, theta, losses, rhos):
        """
        Perform gradient update step with rho_hat
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

        # \pdv{F}{rho_g} [g]
        disp_grad = self.grad_disparity_fn(rhos)

        # \pdv{l_g}{theta} [g]
        tangent = self.get_tangent(theta)
        unit_tangent = tangent / np.linalg.norm(tangent)

        g = self.disparity_fn(rhos)

        perf_grad = rhos + rhos_grad * losses

        fair_proj_grad = unit_tangent * np.einsum(
            "g,g->",
            unit_tangent,
            disp_grad * rhos_grad
            # \pdv{F}{rho_g} [g]  # \pdv{rho_g}{l_g} [g]
        )

        d = np.einsum("g,g->", perf_grad, fair_proj_grad) / np.einsum(
            "g,g->", fair_proj_grad, fair_proj_grad
        )
        lamda = np.maximum(g - d, 0)

        # print("losses", losses)
        # print("g", g)
        # print("d", d)
        # print("perf_grad", perf_grad)
        # print("fair_proj_grad", fair_proj_grad)
        # print(
        #     "update",
        #     np.einsum("g,g->", (perf_grad + lamda * fair_proj_grad), fair_proj_grad),
        # )

        # A_losses = np.array([self.xs, self.ys])

        # return (
        #     lamda,
        #     self.ts[
        #         np.argmin(
        #             np.einsum(
        #                 "g,gi,g->i",
        #                 rhos + rhos_grad * (losses + lamda * disp_grad),
        #                 A_losses,
        #                 self.group_sizes,
        #             )
        #         )
        #     ],
        # )

        return (
            lamda,
            self.quadratic_program(
                losses,
                self.group_sizes * (perf_grad + lamda * fair_proj_grad),
                # self.group_sizes * (rhos + rhos_grad * (losses + lamda * disp_grad)),
            ),
        )

    def fair_grad_step(self, theta, losses, rhos):
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
        self.video_file = os.path.join("mp4", title + ".mp4")

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
    def __init__(self, problem, env, method=None, save_init=True, **kw):
        """
        problem:
            save `problem.pdf` before any simulation
            if method is not None, save `problem_method.mp4` as video of simulation
        """

        self.title = problem
        self.env = env
        self.method = method

        self.fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        if method is not None:
            super().__init__(f"{problem}_{method}", self.fig, fps=15, dpi=100)

        self.left, self.center, self.right = axs
        self.setup_left(self.left, "Group Losses", **kw)
        self.setup_center(self.center, "Group Participation Rates", **kw)
        self.setup_right(self.right, "Loss and Disparity Surfaces", **kw)

        if save_init:
            self.fig.tight_layout()

            savefig(self.fig, os.path.join("pdf", f"{self.title}_init.pdf"))

            fig, left = plt.subplots(1, 1, figsize=(6, 6))
            self.setup_left(left, self.title, **kw)
            savefig(fig, os.path.join("pdf", f"{self.title}_left.pdf"))

            fig, center = plt.subplots(1, 1, figsize=(6, 6))
            self.setup_center(center, self.title, **kw)
            savefig(fig, os.path.join("pdf", f"{self.title}_center.pdf"))

            fig, right = plt.subplots(1, 1, figsize=(6, 6))
            self.setup_right(right, self.title, **kw)
            savefig(fig, os.path.join("pdf", f"{self.title}_right.pdf"))

    def __enter__(self):
        if self.method is not None:
            return super().__enter__()
        return self

    def __exit__(self, *args):
        if self.method is not None:
            super().__exit__(*args)

    def setup_left(self, left, title, **kw):
        # Plot achievable losses
        achievable_losses = self.env.achievable_losses

        left.scatter(*achievable_losses.T, color="black", label="Fixed Policies")

        left.plot(self.env.xs, self.env.ys, "black", label="Pareto Boundary")

        left.set_xlim(-1, 0)
        left.set_ylim(-1, 0)
        left.set_xlabel("Group 1 loss $\\ell_1$", labelpad=-10)
        left.set_ylabel("Group 2 loss $\\ell_2$", labelpad=-10)
        left.set_title(title)

        left.legend(loc="upper right")

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

        use_two_ticks_x(left)
        use_two_ticks_y(left)

        lims = [
            [
                np.min(self.env.achievable_losses[:, 0]),
                np.max(self.env.achievable_losses[:, 0]),
            ],
            [
                np.min(self.env.achievable_losses[:, 1]),
                np.max(self.env.achievable_losses[:, 1]),
            ],
        ]
        if (lims[0][0] > -1 and lims[0][1] < 0) and (
            lims[1][0] > -1 and lims[1][1] < 0
        ):
            left_inset = left.inset_axes([0.5, 0.5, 0.3, 0.3])
            left_inset.set_xlim(lims[0][0] - 0.02, lims[0][1] + 0.02)
            left_inset.set_ylim(lims[1][0] - 0.02, lims[1][1] + 0.02)
            left_inset.scatter(*achievable_losses.T, color="black")
            left_inset.plot(self.env.xs, self.env.ys, "black")
            left_inset.set_xticks([])
            left_inset.set_yticks([])
            left.indicate_inset_zoom(left_inset)

    def setup_center(self, center, title, **kw):
        # plot achievable rhos
        theta_range = np.linspace(0, np.pi / 2, 1000)
        achievable_rhos = np.array(
            [self.env.get_rhos(self.env.get_losses(theta)) for theta in theta_range]
        )
        center.plot(*achievable_rhos.T, color="black", label="Pareto Boundary")
        center.set_title(title)

        cx, cy = self.env.inverse_disparity_curve()
        center.plot(cx, cy, color="red", linestyle="--", label="Fair Boundary")
        center.plot(cy, cx, color="red", linestyle="--")
        center.set_xlim(0, 1)
        center.set_ylim(0, 1)

        center.set_xlabel("Group 1 participation rate $\\rho_1$", labelpad=-10)
        center.set_ylabel("Group 2 participation rate $\\rho_2$", labelpad=-10)
        center.legend(loc="upper right")
        use_two_ticks_x(center)
        use_two_ticks_y(center)

    def setup_right(self, right, title, **kw):
        # plot performative loss and fairness surface
        if "theta_plot_range" in kw:
            min_theta, max_theta = kw["theta_plot_range"]
        else:
            min_theta, max_theta = (0, np.pi / 2)
        theta_range = np.linspace(min_theta, max_theta, 1000)

        right_r = right.twinx()

        # plot loss curve
        right.plot(
            theta_range,
            [self.env.get_total_loss(theta) for theta in theta_range],
            "blue",
            label="Loss",
        )

        disparities = [self.env.get_total_disparity(theta) for theta in theta_range]
        max_disparity = max(disparities)

        # plot disparity curve
        right_r.plot(
            theta_range,
            disparities,
            "red",
            linestyle="--",
        )
        right.plot([], [], "red", linestyle="--", label="Disparity")

        def root_find(f, l, r):
            if f(l) < 0:
                assert f(r) > 0
            else:
                assert f(r) < 0
                l, r = r, l
            while abs(l - r) > 0.0001:
                m = (l + r) / 2
                if f(m) > 0:
                    r = m
                else:
                    l = m
            return m

        theta_l = root_find(self.env.get_total_disparity, 0, np.pi / 4)
        theta_r = root_find(self.env.get_total_disparity, np.pi / 4, np.pi / 2)
        right_r.fill_between(
            [min_theta, theta_l],
            [0, 0],
            [max_disparity, max_disparity],
            alpha=0.1,
            color="red",
        )
        right_r.fill_between(
            [theta_r, max_theta],
            [0, 0],
            [max_disparity, max_disparity],
            alpha=0.1,
            color="red",
        )

        right.set_title(title)
        right.set_xlabel("Parameter $\\theta$")
        right.set_ylabel("Total Loss $\\sum_g \\ell_g \\rho_g s_g$", labelpad=-20)
        right.yaxis.label.set_color("blue")
        right_r.set_ylabel("Disparity $\\mathcal{F}(\\rho)$", labelpad=-10)
        right_r.yaxis.label.set_color("red")

        right.legend(loc="lower left")

        if "t_init" in kw:
            right.scatter(
                [kw["t_init"]],
                [self.env.get_total_loss(kw["t_init"])],
                marker="o",
                color="black",
            )
            right.scatter(
                [kw["t_init"]],
                [self.env.get_total_loss(kw["t_init"]) + 0.005],
                marker="$0$",
                color="black",
                s=64,
            )
        if "t_rrm" in kw:
            right.scatter(
                [kw["t_rrm"]],
                [self.env.get_total_loss(kw["t_rrm"])],
                marker="o",
                color="black",
            )
            right.scatter(
                [kw["t_rrm"]],
                [self.env.get_total_loss(kw["t_rrm"]) + 0.005],
                marker="$R$",
                color="black",
                s=64,
            )
        if "t_lpu" in kw:
            right.scatter(
                [kw["t_lpu"]],
                [self.env.get_total_loss(kw["t_lpu"])],
                marker="o",
                color="black",
            )
            right.scatter(
                [kw["t_lpu"]],
                [self.env.get_total_loss(kw["t_lpu"]) + 0.005],
                marker="$L$",
                color="black",
                s=64,
            )
        if "t_fair" in kw:
            right.scatter(
                [kw["t_fair"]],
                [self.env.get_total_loss(kw["t_fair"])],
                marker="o",
                color="black",
            )
            right.scatter(
                [kw["t_fair"]],
                [self.env.get_total_loss(kw["t_fair"]) + 0.005],
                marker="$F$",
                color="black",
                s=64,
            )

        use_two_ticks_x(right)
        use_two_ticks_y(right_r)
        use_two_ticks_y(right)

    def update_left(self, ax, lamda, theta, losses, rhos):
        """
        - Plot current location on achievable loss curve (point)
        - Plot vector in direction opposite rhos
        """

        artifacts = [
            ax.scatter([losses[0]], [losses[1]], color="red", marker="^", s=100)
        ]

        if self.method.startswith("RRM"):
            t = np.arctan(rhos[1] / rhos[0])
            l = self.env.get_losses(t)
            d = np.einsum("g,g->", rhos, l) / np.einsum("g,g->", rhos, rhos)
            artifacts += [ax.plot([d * rhos[0], 0], [d * rhos[1], 0], "red")[0]]

        return artifacts

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

        # theta_range = np.linspace(0, 1, 100) * np.pi / 2  # [i]
        # tl = np.array([self.env.get_total_loss(theta) for theta in theta_range])
        # td = np.array([self.env.get_total_disparity(theta) for theta in theta_range])

        # artifacts += [
        #     ax.plot(theta_range, tl + lamda * td, color="black", linestyle="--")[0]
        # ]

        return artifacts

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
    # return np.log(100 * np.var(rhos) + 0.01)


def inverse_disparity_curve():
    rho_1 = np.linspace(0, 1, 100)
    rho_2 = np.sqrt(4 * 0.01) + rho_1
    return rho_1, rho_2


def concave_rho_fn(loss):
    """
    Monotonically decreasing and concave.
    """
    return 1 - 1 / (1 - loss * 2)


def run_problem(
    problem,
    rho_fns=concave_rho_fn,
    method=None,
    save_init=True,
    eta=0.1,
    num_steps=100,
    init_theta=0.6 * np.pi / 2,
    jit=False,
    **kw,
):
    filename = os.path.join("losses", f"{problem}.npy")
    try:  # load cached values
        achievable_losses = np.load(filename)
        print(f"Loaded Cached Achievable Losses from {filename}")
    except FileNotFoundError:
        print("Determining Achievable Losses")
        achievable_losses = get_achievable_losses(problem)
        print(f"Saving to {filename}")
        np.save(filename, achievable_losses)

    env = Env(
        achievable_losses,
        rho_fns=rho_fns,
        group_sizes=np.array([0.5, 0.5]),
        disparity_fn=disparity_fn,
        inverse_disparity_curve=inverse_disparity_curve,
        eta=eta,
    )

    if method is not None:
        if jit:
            update_func = jax.jit(env.update_funcs[method])
        else:
            update_func = env.update_funcs[method]

    # save initial figures
    # save video if method is defined
    with Viz(problem, env, method, save_init, **kw) as viz:
        if method is not None:
            filename = os.path.join("npy", f"{problem}_{method}")
            try:  # load cached values
                thetas = np.load(f"{filename}_thetas.npy")
                losses = np.load(f"{filename}_losses.npy")
                rhos = np.load(f"{filename}_rhos.npy")
                total_loss = np.load(f"{filename}_total_loss.npy")
                total_disparity = np.load(f"{filename}_total_disparity.npy")
                lamdas = np.load(f"{filename}_lamdas.npy")

                for i in tqdm(range(100)):
                    viz.render_frame(lamdas[i], thetas[i], losses[i], rhos[i])

            except FileNotFoundError:
                theta = init_theta
                _losses = env.get_losses(theta)
                _rhos = env.get_rhos(_losses)

                viz.render_frame(0, theta, _losses, _rhos)

                thetas = [theta]
                total_loss = [env.get_total_loss(theta)]
                total_disparity = [env.get_total_disparity(theta)]
                lamdas = []
                losses = [_losses]
                rhos = [_rhos]
                for i in tqdm(range(num_steps)):
                    lamda, theta = update_func(theta, _losses, _rhos)
                    _losses = env.get_losses(theta)
                    _rhos = env.get_rhos(_losses)

                    thetas.append(theta)
                    total_loss.append(env.get_total_loss(theta))
                    total_disparity.append(env.get_total_disparity(theta))
                    lamdas.append(lamda)
                    losses.append(_losses)
                    rhos.append(_rhos)

                    viz.render_frame(lamda, theta, _losses, _rhos)

                np.save(f"{filename}_thetas.npy", thetas)
                np.save(f"{filename}_losses.npy", losses)
                np.save(f"{filename}_rhos.npy", rhos)
                np.save(f"{filename}_total_loss.npy", total_loss)
                np.save(f"{filename}_total_disparity.npy", total_disparity)
                np.save(f"{filename}_lamdas.npy", lamdas)


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
        lamdas = np.load(f"{filename}_lamdas.npy")

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
        np.arange(len(lamdas)) + 1,
        lamdas,
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


def logistic(x):
    return 1 / (1 + np.exp(-x))


def localized_rho_fn(center, sensitivity, loss):
    """
    Monotonically decreasing. Not concave.
    """
    return 1 - np.clip(logistic((loss - center) * sensitivity), 0, 1)
