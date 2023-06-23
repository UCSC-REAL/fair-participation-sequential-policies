import os
from typing import Callable, Optional

import cvxpy as cp
import jax
import jax.numpy as np
import jax.scipy.optimize
import numpy as onp  # TODO fix
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull
from tqdm import tqdm

from fair_participation.animation import Viz
from fair_participation.base_logger import log
from fair_participation.folktasks import get_achievable_losses


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

        log.debug("losses", losses)
        log.debug("g", g)
        log.debug("d", d)
        log.debug("perf_grad", perf_grad)
        log.debug("fair_proj_grad", fair_proj_grad)
        log.debug(
            "update",
            np.einsum("g,g->", (perf_grad + lamda * fair_proj_grad), fair_proj_grad),
        )

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
    name: str = "",
    rho_fns: Optional[Callable | tuple[Callable]] = concave_rho_fn,
    method: Optional[str] = None,
    save_init: bool = True,
    eta: float = 0.1,
    num_steps: int = 100,
    init_theta: float = 0.6 * np.pi / 2,
    jit: bool = False,
    viz_kwargs: Optional[dict] = None,
):
    """

    :param name:
    :param rho_fns:
    :param method:
    :param save_init:
    :param eta:
    :param num_steps:
    :param init_theta:
    :param jit:
    :param viz_kwargs:
    """
    filename = os.path.join("losses", f"{name}.npy")
    try:  # load cached values
        achievable_losses = np.load(filename)
        log.info(f"Loaded cached achievable losses from {filename}.")
    except FileNotFoundError:
        log.info("Calculating achievable losses...")
        achievable_losses = get_achievable_losses(name)
        log.info(f"Saving {filename}")
        np.save(filename, achievable_losses)

    if callable(rho_fns):
        # Use same rho for both groups
        rho_fns = (rho_fns, rho_fns)

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
    if viz_kwargs is None:
        viz_kwargs = dict()
    # TODO should update this with fast version
    with Viz(name, env, method, save_init, viz_kwargs) as viz:
        if method is not None:
            filename = os.path.join("npz", f"{name}_{method}.npz")
            try:  # load cached values
                # TODO load/save as npz
                npz = np.load(filename)
                for i in tqdm(range(100)):
                    viz.render_frame(
                        npz["lambdas"][i],
                        npz["thetas"][i],
                        npz["losses"][i],
                        npz["rhos"][i],
                    )

            except FileNotFoundError:
                theta = init_theta
                _losses = env.get_losses(theta)
                _rhos = env.get_rhos(_losses)

                viz.render_frame(0, theta, _losses, _rhos)

                thetas = [theta]
                total_loss = [env.get_total_loss(theta)]
                total_disparity = [env.get_total_disparity(theta)]
                lambdas = []
                losses = [_losses]
                rhos = [_rhos]
                for i in tqdm(range(num_steps)):
                    lambda_, theta = update_func(theta, _losses, _rhos)
                    _losses = env.get_losses(theta)
                    _rhos = env.get_rhos(_losses)

                    thetas.append(theta)
                    total_loss.append(env.get_total_loss(theta))
                    total_disparity.append(env.get_total_disparity(theta))
                    lambdas.append(lambda_)
                    losses.append(_losses)
                    rhos.append(_rhos)

                    viz.render_frame(lambda_, theta, _losses, _rhos)

                np.savez(
                    filename,
                    thetas=thetas,
                    losses=losses,
                    rhos=rhos,
                    total_loss=total_loss,
                    total_disparity=total_disparity,
                    lambdas=lambdas,
                )


def logistic(x):
    return 1 / (1 + np.exp(-x))


def localized_rho_fn(
    sensitivity: float, loss: float
) -> Callable[[ArrayLike], ArrayLike]:
    def localized_rho(center: ArrayLike):
        """
        Monotonically decreasing. Not concave.
        """
        return 1 - np.clip(logistic((loss - center) * sensitivity), 0, 1)

    return localized_rho
