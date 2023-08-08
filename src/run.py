import os
import pathlib
from typing import Optional

from jax import numpy as jnp

from fair_participation.rate_functions import localized_rho_fn
from fair_participation.simulation import simulate
from fair_participation.utils import PROJECT_ROOT

from fair_participation.plotting.compare import compare


def run_problems(problems: list[dict], clean: Optional[str] = None) -> None:
    """
    Runs the simulation for each problem in the list of problems.
    :param problems: List of dictionaries containing the problem parameters.
    :param clean: If 'results', deletes all files in the data, mp4, npz, and pdf directories.
      If 'all', also deletes all files in the 'losses' directory.
    """
    all_folders = ["data", "mp4", "npz", "pdf", "losses"]
    clean_folders = []
    if clean == "graphics":
        clean_folders = ["mp4", "pdf"]
    if clean == "results":
        clean_folders = ["npz", "mp4", "pdf"]
    if clean == "all":
        clean_folders = ["losses", "npz", "mp4", "pdf"]

    for folder in all_folders:
        full_folder = os.path.join(PROJECT_ROOT, folder)
        os.makedirs(full_folder, exist_ok=True)
        if folder in clean_folders:
            for file in os.listdir(full_folder):
                ext = pathlib.Path(file).suffix
                if ext in (".npz", ".mp4", ".pdf", ".npy"):
                    os.remove(os.path.join(folder, file))

    for problem in problems:
        simulate(**problem)


def main():
    problems = []
    base_problems = [
        {
            "name": "Income_three_groups",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_loss_direction": jnp.array([0.1, 0.5, 0.2]),
            "eta": 0.00001,
        },
        # {
        #     "name": "Income",
        #     "rho_fns": localized_rho_fn(-0.75, 20),
        #     "init_loss_direction": 0.57,
        #     "eta": 0.0001,
        # },
        # {
        #     "name": "Mobility",
        #     "rho_fns": localized_rho_fn(-0.7, 10),
        #     "init_loss_direction": 0.6,
        #     "eta": 0.3,
        # },
        # {
        #     "name": "PublicCoverage",
        #     "rho_fns": localized_rho_fn(-0.7, 50),
        #     "init_loss_direction": 0.6,
        # },
        # {
        #     "name": "TravelTime",
        #     "rho_fns": localized_rho_fn(-0.58, 100),
        #     "init_loss_direction": 0.51,
        # },
    ]
    for prob in base_problems:
        subproblems = []
        for method in (  # listed in environment.py
            "RRM",
            "FairLPU",
        ):

            subproblems.append(dict(**prob, method=method))
        problems.append(subproblems)

    # run all methods for all environments
    allprobs = [prob for subprobs in problems for prob in subprobs]
    run_problems(allprobs, clean="results")

    # compare different methods in same environment
    for subproblems in problems:
        filename = os.path.join(
            PROJECT_ROOT, "pdf", f"{subproblems[0]['name']}_compare.pdf"
        )
        compare(subproblems, filename)


if __name__ == "__main__":
    main()
