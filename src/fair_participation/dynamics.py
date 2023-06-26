import os
from typing import Callable, Optional

import jax
import jax.numpy as np
import jax.scipy.optimize
from numpy.typing import ArrayLike
from tqdm import tqdm

from fair_participation.animation import Viz
from fair_participation.base_logger import log
from fair_participation.env import Env
from fair_participation.folktasks import get_achievable_losses


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
    jit: bool = False,  # TODO why not always true?
    viz_kwargs: Optional[dict] = None,
):
    """
    TODO
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
        init_theta=init_theta,
        jit=jit,
        update_method=method,
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
                    pars = {
                        par: npz[par][i]
                        for par in ("lambdas", "thetas", "losses", "rhos")
                    }
                    viz.render_frame(
                        render_pars=pars,
                    )

            except FileNotFoundError:
                pars = dict()
                for i in tqdm(range(num_steps)):
                    # # TODO change this to a nice update structure
                    # if i == 0:
                    #     lambda_, theta = 0, init_theta
                    # else:
                    #     lambda_, theta = update_func(theta, losses, rhos)
                    # losses = env.get_losses(theta)
                    # rhos = env.get_rhos(losses)
                    #
                    env.update()
                    #
                    # pars.setdefault("lambdas", []).append(lambda_)
                    # pars.setdefault("thetas", []).append(theta)
                    # pars.setdefault("losses", []).append(losses)
                    # pars.setdefault("rhos", []).append(rhos)
                    # pars.setdefault("total_loss", []).append(env.get_total_loss(theta))
                    # pars.setdefault("total_disparity", []).append(
                    #     env.get_total_disparity(theta)
                    # )

                    viz.render_frame(render_pars=env.state)

                np.savez(filename, **pars)


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
