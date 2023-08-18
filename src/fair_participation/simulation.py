import os
from typing import Callable

import jax.numpy as jnp
import pandas as pd

from fair_participation.environment import Environment
from fair_participation.rrm import rrm_step
from fair_participation.mpg import mpg_step
from fair_participation.cpg import cpg_step

from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT


def get_trial_filename(name: str, method: str) -> str:
    return os.path.join(PROJECT_ROOT, "npz", f"{name}_{method}.npz")


def update_env_fn(
    env: Environment, method: str, init_eta: float, eta_decay: float, alpha: float
) -> Callable:
    """
    Returns a function that updates the environment state.

    :param env: Environment object.
    :param method: Method name. Can be "RRM", "MPG", or "CPG".
    :param init_eta: Initial learning rate.
    :param eta_decay: Learning rate decay.
    :param alpha: Learning rate for dual variable.
    :return: Callable that updates the environment state.
    """
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
    elif method == "CPG":
        _update_state = cpg_step(
            values_and_grads=env.values_and_grads,
            loss_hull=env.loss_hull,
        )
    else:
        raise ValueError(f"Unknown update method {method}.")

    def update_env(step_num: int) -> dict:
        """
        Updates state using _update_state and returns new state as a dict.

        :param step_num: Step number.
        :return: Dictionary of the new state.
        """

        # harmonic decay
        eta_scale = 1 / eta_decay - 1
        eta = init_eta / (step_num * eta_scale + 1)

        rates = (eta, alpha)
        env.state = _update_state(env.state.loss, rates)
        return env.state._asdict()

    return update_env


def simulate(
    env: Environment,
    init_eta: float = 0.002,
    eta_decay: float = 1.0,
    alpha: float = 1.0,
    num_steps: int = 100,
    method: str = "RRM",
) -> Environment:
    """
    Simulates the environment for a given number of steps and caches the results.

    :param env: Environment object.
    :param init_eta: Learning rate for primal variable.
    :param eta_decay: Learning rate decay for primal variable.
    :param alpha: Learning rate for dual variable.
    :param num_steps: Number of steps to simulate.
    :param method: Method name to use. Can be "RRM", "MPG", or "CPG".
    :return: Environment object.
    """

    history = [env.state._asdict()]

    vectors = env.loss_hull.points
    distances = jnp.linalg.norm(vectors[:, jnp.newaxis] - vectors, axis=2)
    scale_init_eta = float(init_eta * jnp.max(distances) / 2)

    update_env = update_env_fn(
        env=env,
        method=method,
        init_eta=scale_init_eta,
        eta_decay=eta_decay,
        alpha=alpha,
    )

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
