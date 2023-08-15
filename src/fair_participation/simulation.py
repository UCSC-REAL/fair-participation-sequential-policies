import os
from typing import Optional
import jax.numpy as jnp
import pandas as pd
from tqdm import trange

from fair_participation.rrm import rrm_step
from fair_participation.mpg import mpg_step
from fair_participation.cpg import cpg_step


from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT


def get_trial_filename(name, method):
    return os.path.join(PROJECT_ROOT, "npz", f"{name}_{method}.npz")


def update_env_fn(env, method, init_eta, eta_decay, alpha):
    if method == "RRM":
        _update_state = rrm_step(
            values_and_grads=env.values_and_grads,
            group_sizes=env.group_sizes,
            loss_hull=env.loss_hull,
        )
    elif method == "MPG":
        _update_state = mpg_step(
            values_and_grads=env.values_and_grads,
            loss_hull=env.loss_hull,
        )
    elif method == "UPG":
        _update_state = cpg_step(
            values_and_grads=env.values_and_grads,
            loss_hull=env.loss_hull,
        )
    else:
        raise ValueError(f"Unknown update method {method}.")

    def update_env(step_num: int) -> dict:
        """
        Updates state using _update_state and
        returns new state as dictionary
        :return: Dictionary of the new state.
        """

        # exponential decay
        # eta = init_eta * (eta_decay**step_num)

        # harmonic decay
        eta_scale = 1 / eta_decay - 1
        eta = init_eta / (step_num * eta_scale + 1)

        if method == "cpg":
            rates = (eta, alpha)
        else:
            rates = (eta,)

        # Conceptually dirty, since there's state information
        # that updates "out-of-phase" and belongs to
        # optmization method
        env.state = _update_state(env.state.loss, rates)
        return env.state._asdict()

    return update_env


def simulate(
    env,
    init_eta: float = 0.002,
    eta_decay: float = 1.0,
    alpha: float = 1.0,
    num_steps: int = 100,
    method: str = "RRM",
    **_,
) -> None:
    """
    prepare environment for a given problem.
    Does caching

    :param name: Name of the problem
    :param rho_fns: Rho functions for each group, or a single rho function for all groups. Defaults to concave_rho_fn.
    :param init_eta: Learning rate for primal variable.
    :param eta_decay: Learning rate decay for primal variable.
    :param alpha: Learning rate for dual variable.
    :param alpha_decay: Learning rate decay for dual variable.
    :param num_steps: Number of steps to simulate.
    :param init_loss_direction: Initial direction of loss. Will choose achievable loss closest to this direction.
           Can be set to float to match legacy init_theta.
    :param method: Method to use for updating theta.
    """

    history = [env.state._asdict()]

    vectors = env.loss_hull.points
    distances = jnp.linalg.norm(vectors[:, jnp.newaxis] - vectors, axis=2)
    scale_init_eta = init_eta * jnp.max(distances) / 2

    update_env = update_env_fn(env, method, scale_init_eta, eta_decay, alpha)

    trial_filename = get_trial_filename(name=env.name, method=method)

    if os.path.exists(trial_filename):
        logger.info(f"Cache exists; Skipping:")
        logger.info(f"  {trial_filename}")
        return env

    logger.info("Caching simulation.")
    logger.info(f"  {trial_filename}")

    for step_num in range(num_steps):
        state = update_env(step_num)
        history.append(state)

    df = pd.DataFrame(history)
    data = {col: jnp.array(df[col].to_list()) for col in df.columns}
    jnp.savez(trial_filename, **data)

    return env
