import os
import pathlib
from typing import Optional

import numpy as np

import site

site.addsitedir("src")

from fair_participation.rate_functions import localized_rho_fn
from fair_participation.simulation import simulate
from fair_participation.plotting import compare, compare_2


def run_problems(problems: list[dict], clean: Optional[str] = None) -> None:
    """
    Runs the simulation for each problem in the list of problems.
    :param problems: List of dictionaries containing the problem parameters.
    :param clean: If 'results', deletes all files in the data, mp4, npz, and pdf directories.
      If 'all', also deletes all files in the 'losses' directory.
    """
    clean_folders = []
    if clean in ("results", "all"):
        clean_folders += ["data", "mp4", "npz", "pdf"]
    if clean == "all":
        clean_folders.append("losses")

    # Create needed directories if they don't exist
    for folder in clean_folders:
        if clean is not None:
            for file in os.listdir(folder):
                ext = pathlib.Path(file).suffix
                if ext in (".npz", ".mp4", ".pdf", ".npy"):
                    os.remove(os.path.join(folder, file))
        os.makedirs(folder, exist_ok=True)

    for problem in problems:
        simulate(**problem)
        # TODO finish updating compare
        # compare(problem, grad=False)
        # compare(problem, grad=True)


def main():
    problems = []
    base_problems = [
        {
            "name": "Income",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_theta": 0.57,
            "plot_kwargs": {"theta_plot_range": [0.3, 1.0]},
        },
        {
            "name": "Mobility",
            "rho_fns": localized_rho_fn(-0.7, 10),
            "init_theta": 0.6,
            "eta": 0.3,
        },
        {
            "name": "PublicCoverage",
            "rho_fns": localized_rho_fn(-0.7, 50),
            "init_theta": 0.6,
            "plot_kwargs": {"theta_plot_range": [0.3, 0.7]},
        },
        {
            "name": "TravelTime",
            "rho_fns": localized_rho_fn(-0.58, 100),
            "init_theta": 0.51,
            "plot_kwargs": {"theta_plot_range": [0.4, 0.6]},
        },
    ]
    for prob in base_problems:
        for method in ("RRM", "RRM_grad"):
            problems.append(dict(**prob, method=method, save_init=False))
    run_problems(problems[:2], clean="results")


if __name__ == "__main__":
    main()
