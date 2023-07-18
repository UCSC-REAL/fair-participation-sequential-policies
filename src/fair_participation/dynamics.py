import os
from typing import Callable, Optional


import jax.numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tqdm import trange

from fair_participation.animation import Viz
from fair_participation.base_logger import logger
from fair_participation.env import Env
from fair_participation.folktasks import achievable_losses


def concave_rho_fn(loss):
    """
    Monotonically decreasing and concave.
    """
    return 1 - 1 / (1 - loss * 2)


def simulate(
    name: str = "",
    rho_fns: Optional[Callable | tuple[Callable]] = concave_rho_fn,
    method: Optional[str] = None,
    save_init: bool = True,
    eta: float = 0.1,
    num_steps: int = 100,
    init_theta: float = 0.6 * np.pi / 2,
    jit: bool = False,  # TODO why not always true?
    plot_kwargs: Optional[dict] = None,
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
    :param plot_kwargs:
    """

    logger.info(f"Simulating {name}.")
    filename = os.path.join("losses", f"{name}.npy")
    try:  # load cached values
        achievable_loss = np.load(filename)
        logger.info(f"Loaded cached achievable loss from {filename}.")
    except FileNotFoundError:
        logger.info("Computing achievable loss.")
        achievable_loss = achievable_losses(name)
        logger.info(f"Saving achievable loss to {filename}.")
        np.save(filename, achievable_loss)
    n_groups = achievable_loss.shape[1]

    if callable(rho_fns):
        # Use same rho for all groups
        rho_fns = tuple(rho_fns for _ in range(n_groups))

    # Assume even group sizes for now
    group_sizes = np.ones(n_groups) / n_groups

    env = Env(
        achievable_loss,
        rho_fns=rho_fns,
        group_sizes=group_sizes,
        eta=eta,
        init_theta=init_theta,
        update_method=method,
        jit=jit,
    )

    # TODO update this with fast version
    with Viz(name, env, method, save_init, plot_kwargs) as viz:
        if method is not None:
            filename = os.path.join("npz", f"{name}_{method}.npz")
            try:  # load cached values
                # TODO load/save as npz
                npz = np.load(filename)
                for i in trange(100):
                    pars = {
                        par: npz[par][i] for par in ("lambdas", "thetas", "loss", "rho")
                    }
                    viz.render_frame(
                        render_pars=pars,
                    )

            except FileNotFoundError:
                for _ in trange(num_steps):
                    state = env.update()
                    # viz.render_frame(render_pars=state)
                # TODO unpack history
                history = pd.DataFrame(env.history).to_dict("series", index=False)
                np.savez(filename, **pars)


def logistic(x):
    return 1 / (1 + np.exp(-x))


def localized_rho_fn(
    sensitivity: float, loss: float
) -> Callable[[ArrayLike], ArrayLike]:
    """
    Returns a callable rho function centered at `loss`.
    :param sensitivity: Sensitivity of the rho function.
    :param loss: Center of the rho function.
    :return: Callable
    """

    def localized_rho(center: ArrayLike) -> ArrayLike:
        """
        Monotonically decreasing. Not concave.
        """
        return 1 - np.clip(logistic((loss - center) * sensitivity), 0, 1)

    return localized_rho
