import os
from typing import Callable, Optional

import jax.numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tqdm import tqdm

from fair_participation.animation import Viz
from fair_participation.base_logger import log
from fair_participation.env import Env
from fair_participation.folktasks import get_achievable_loss


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
        achievable_loss = np.load(filename)
        log.info(f"Loaded cached achievable loss from {filename}.")
    except FileNotFoundError:
        log.info("Calculating achievable loss...")
        achievable_loss = get_achievable_loss(name)
        log.info(f"Saving {filename}")
        np.save(filename, achievable_loss)

    if callable(rho_fns):
        # Use same rho for both groups
        rho_fns = (rho_fns, rho_fns)

    env = Env(
        achievable_loss,
        rho_fns=rho_fns,
        group_sizes=np.array([0.5, 0.5]),
        eta=eta,
        init_theta=init_theta,
        update_method=method,
        jit=jit,
    )

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
                        par: npz[par][i] for par in ("lambdas", "thetas", "loss", "rho")
                    }
                    viz.render_frame(
                        render_pars=pars,
                    )

            except FileNotFoundError:
                for _ in tqdm(range(num_steps)):
                    state = env.update()
                    viz.render_frame(render_pars=state)
                # TODO unpack history
                history = pd.DataFrame(env.history).to_dict("series", index=False)
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
