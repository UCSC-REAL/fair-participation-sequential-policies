import os
import pathlib
from typing import Optional

from jax import numpy as jnp

from fair_participation.rate_functions import localized_rho_fn
from fair_participation.simulation import simulate, get_trial_filename
from fair_participation.utils import PROJECT_ROOT

from fair_participation.environment import make_environment, get_env_filename
from fair_participation.plotting.animation import animate, get_animation_filename
from fair_participation.plotting.compare import (
    compare_solutions,
    compare_timeseries,
    get_compare_timeseries_filename,
    get_compare_solutions_filename,
)

from fair_participation.base_logger import logger


def do_clean(name, methods, clean):

    # clean everything, independent of which problems active
    if clean == "all":
        clean_folders = ["losses", "npz", "mp4", "pdf"]
        for folder in clean_folders:
            full_folder = os.path.join(PROJECT_ROOT, folder)
            for file in os.listdir(full_folder):
                ext = pathlib.Path(file).suffix
                if ext in (".npz", ".mp4", ".pdf", ".npy"):
                    os.remove(os.path.join(folder, file))
        return

    # clean only for currently active problems
    # pdf:      pdfs
    # graphics: pdfs + mp4
    # trials:   pdfs + mp4 + npz
    # env:      pdfs + mp4 + npz + losses
    targets = set()

    if clean in ["timeseries", "pdfs", "graphics", "trials", "envs"]:
        targets.add(get_compare_timeseries_filename(name))

    if clean in ["solutions", "pdfs", "graphics", "trials", "envs"]:
        targets.add(get_compare_solutions_filename(name))

    for method in methods:
        if clean in ["graphics", "trials", "envs"]:
            targets.add(get_animation_filename(name, method))
        if clean in ["trials", "envs"]:
            targets.add(get_trial_filename(name, method))
        if clean in ["envs"]:
            targets.add(get_env_filename(name))

    for target in targets:
        if os.path.exists(target):
            os.remove(target)


def run_problems(
    problems: list[dict],
    methods: list[str],
    clean_lvl: Optional[str] = None,
    output_graphics: Optional[list[str]] = None,
) -> None:
    """
    Runs the simulation for each problem in the list of problems.
    :param problems: List of dictionaries containing the problem parameters.
    """

    all_folders = ["data", "mp4", "npz", "pdf", "losses"]
    for folder in all_folders:
        full_folder = os.path.join(PROJECT_ROOT, folder)
        os.makedirs(full_folder, exist_ok=True)

    for problem in problems:
        logger.info(f"====={problem['name']}=====")

        name = problem["name"]

        do_clean(name, methods, clean_lvl)

        for method in methods:
            logger.info(f"-----{method}-----")

            env = make_environment(**problem)

            # simulate problem
            simulate(env, method=method, **problem)

            # animate problem
            if output_graphics is not None and "animations" in output_graphics:
                animate(env, method)

        # compare (as time-series) different methods in same environment
        if output_graphics is not None and "timeseries" in output_graphics:
            compare_timeseries(name, methods)

        # compare (on loss/rho surfaces) different methods in same environment
        if output_graphics is not None and "solutions" in output_graphics:
            env = make_environment(**problem)
            compare_solutions(env, methods)


def main():
    base_problems = [
        {
            "name": "IncomeThree",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_loss_direction": jnp.array([-0.5, -0.3, -0.3]),
            "num_steps": 30,
            "fair_epsilon": 0.05,
        },
        {
            "name": "Income",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_loss_direction": 0.58,
            "num_steps": 30,
        },
        {
            "name": "Mobility",
            "rho_fns": localized_rho_fn(-0.7, 10),
            "init_loss_direction": 0.6,
            "num_steps": 30,
        },
        {
            "name": "PublicCoverage",
            "rho_fns": localized_rho_fn(-0.7, 50),
            "init_loss_direction": 0.5,
            "num_steps": 30,
        },
        {
            "name": "TravelTime",
            "rho_fns": localized_rho_fn(-0.58, 100),
            "init_loss_direction": 0.52,
            "num_steps": 30,
        },
    ]
    methods = (  # listed in environment.py
        "RRM",
        "MPG",
        "CPG",
    )
    output_graphics = [
        "timeseries",
        "solutions",
        # "animations",
    ]
    # CLEAN_LVL = "timeseries"
    # CLEAN_LVL = "solutions"
    # CLEAN_LVL = "pdfs"
    # CLEAN_LVL = "graphics"
    CLEAN_LVL = "trials"

    run_problems(
        base_problems, methods, clean_lvl=CLEAN_LVL, output_graphics=output_graphics
    )


if __name__ == "__main__":
    main()
