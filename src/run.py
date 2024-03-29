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
    compare_solutions_3D,
    compare_timeseries,
    get_compare_timeseries_filename,
    get_compare_solutions_filename,
)

from fair_participation.base_logger import logger


def do_clean(name: str, methods: list[str], clean: str) -> None:
    """
    Clean up files from previous runs.

    :param name: Name of the problem.
    :param methods: Methods to be cleaned.
    :param clean: Type of cleaning. One of "all", "pngs", "graphics", "trials", "envs".
    :return: None.
    """
    # clean everything, independent of which problems are active
    if clean == "all":
        clean_folders = ["losses", "npz", "mp4", "png"]
        for folder in clean_folders:
            full_folder = os.path.join(PROJECT_ROOT, folder)
            for file in os.listdir(full_folder):
                ext = pathlib.Path(file).suffix
                if ext in (".npz", ".mp4", ".png", ".npy"):
                    os.remove(os.path.join(folder, file))
        return

    # clean only for currently active problems
    # pdfs:     pdfs
    # pngs:     pngs
    # graphics: pngs + mp4
    # trials:   pngs + mp4 + npz
    # env:      pngs + mp4 + npz + losses
    targets = set()

    if clean in ["timeseries", "pdfs", "pngs", "graphics", "trials", "envs"]:
        targets.add(get_compare_timeseries_filename(name))

    if clean in ["solutions", "pdfs", "pngs", "graphics", "trials", "envs"]:
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
    :param methods: List of methods to be used.
    :param clean_lvl: Level of cleaning to be done before running each simulation.
    :param output_graphics: List of graphics to be output.
    :return: None.
    """

    all_folders = ["data", "mp4", "npz", "png", "pdf", "losses"]
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

            # is 3D, not 2D
            if env.achievable_loss.shape[1] == 3:
                compare_solutions_3D(env, methods)
            else:
                compare_solutions(env, methods)


def main():
    base_problems = [
        {
            "source": "folktasks",
            "name": "IncomeThree",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_loss_direction": jnp.array([-0.5, -0.3, -0.3]),
            "num_steps": 30,
            "fair_epsilon": 0.05,
        },
        {
            "source": "folktasks",
            "name": "Income",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_loss_direction": 0.58,
            "num_steps": 30,
        },
        # {
        #     "source": "folktasks",
        #     "name": "Mobility",
        #     "rho_fns": localized_rho_fn(-0.7, 10),
        #     "init_loss_direction": 0.6,
        #     "num_steps": 30,
        # },
        # {
        #     "source": "folktasks",
        #     "name": "PublicCoverage",
        #     "rho_fns": localized_rho_fn(-0.7, 50),
        #     "init_loss_direction": 0.5,
        #     "num_steps": 30,
        # },
        # {
        #     "source": "folktasks",
        #     "name": "TravelTime",
        #     "rho_fns": localized_rho_fn(-0.58, 100),
        #     "init_loss_direction": 0.52,
        #     "num_steps": 30,
        # },
        {
            "source": "grouplens",
            "name": "MovieLens",
            "rho_fns": localized_rho_fn(-0.7, 20),
            "init_loss_direction": 0.52,
            "num_steps": 30,
            "fair_epsilon": 0.0005,
        },
    ]
    methods = [  # listed in environment.py
        "RGD",
        "MPG",
        "CPG",
    ]
    output_graphics = [
        "timeseries",
        "solutions",
    ]

    clean_lvl = None

    run_problems(
        base_problems, methods, clean_lvl=clean_lvl, output_graphics=output_graphics
    )


if __name__ == "__main__":
    main()
