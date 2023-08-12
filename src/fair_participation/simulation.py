import os

import jax.numpy as jnp
import pandas as pd
from tqdm import trange

from fair_participation.rrm import rrm_step
from fair_participation.fair_lpu import fair_lpu_step


from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT


def get_trial_filename(name, method):
    return os.path.join(PROJECT_ROOT, "npz", f"{name}_{method}.npz")


def update_env_fn(env, method, eta=0.01, alpha=0.01):
    if method == "RRM":
        _update_state = rrm_step(
            values_and_grads=env.values_and_grads,
            group_sizes=env.group_sizes,
            loss_hull=env.loss_hull,
        )
    elif method == "FairLPU":
        _update_state = fair_lpu_step(
            values_and_grads=env.values_and_grads,
            loss_hull=env.loss_hull,
            eta=eta,
            alpha=alpha,  # TODO separate alpha parameter
        )
    else:
        raise ValueError(f"Unknown update method {method}.")

    def update_env() -> dict:
        """
        Updates state using _update_state and
        returns new state as dictionary
        :return: Dictionary of the new state.
        """
        env.state = _update_state(env.state.loss)
        return env.state._asdict()

    return update_env


def simulate(
    env,
    eta: float = 0.1,
    alpha: float = 0.1,
    num_steps: int = 50,
    method: str = "RRM",
    **_,
) -> None:
    """
    prepare environment for a given problem.
    Does caching

    :param name: Name of the problem
    :param rho_fns: Rho functions for each group, or a single rho function for all groups. Defaults to concave_rho_fn.
    :param eta: Learning rate for the update method.
    :param num_steps: Number of steps to simulate.
    :param init_loss_direction: Initial direction of loss. Will choose achievable loss closest to this direction.
           Can be set to float to match legacy init_theta.
    :param method: Method to use for updating theta.
    """

    history = [env.state._asdict()]

    update_env = update_env_fn(env, method, eta, alpha)

    trial_filename = get_trial_filename(name=env.name, method=method)

    if os.path.exists(trial_filename):
        logger.info(f"Cache exists; Skipping:")
        logger.info(f"  {trial_filename}")
        return env

    logger.info("Caching simulation.")
    logger.info(f"  {trial_filename}")

    for _ in trange(num_steps):
        history.append(update_env())

    df = pd.DataFrame(history)
    data = {col: jnp.array(df[col].to_list()) for col in df.columns}
    jnp.savez(trial_filename, **data)

    return env
