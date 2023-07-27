import os
import pathlib
import numpy as np

from fair_participation.rate_functions import localized_rho_fn
from fair_participation.simulation import simulate
from fair_participation.plotting import compare, compare_2


def run_problems(problems: list[dict], clean: bool = False) -> None:
    """
    Runs the simulation for each problem in the list of problems.
    :param problems: list of dictionaries containing the problem parameters.
    :param clean: if True, deletes all files in the data, mp4, npz, and pdf directories.
    """
    # Create needed directories if they don't exist
    for folder in ("losses", "data", "mp4", "npz", "pdf"):
        if clean:
            for file in os.listdir(folder):
                if pathlib.Path(file).suffix in (".npz", ".npy", ".mp4", ".pdf"):
                    os.remove(os.path.join(folder, file))
        else:
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
            "init_theta": 0.57 * np.pi / 2,
            "plot_kwargs": {"theta_plot_range": [0.3 * np.pi / 2, np.pi / 2]},
        },
        {
            "name": "Mobility",
            "rho_fns": localized_rho_fn(-0.7, 10),
            "init_theta": 0.6 * np.pi / 2,
            "eta": 0.3,
        },
        {
            "name": "PublicCoverage",
            "rho_fns": localized_rho_fn(-0.7, 50),
            "init_theta": 0.6 * np.pi / 2,
            "plot_kwargs": {"theta_plot_range": [0.3 * np.pi / 2, 0.7 * np.pi / 2]},
        },
        {
            "name": "TravelTime",
            "rho_fns": localized_rho_fn(-0.58, 100),
            "init_theta": 0.51 * np.pi / 2,
            "plot_kwargs": {"theta_plot_range": [0.4 * np.pi / 2, 0.6 * np.pi / 2]},
        },
    ]
    for prob in base_problems:
        for method in ("RRM_grad", "RRM", "FairLPU", "FairLPU_grad"):
            problems.append(dict(**prob, method=method, save_init=False))
    run_problems(problems)


if __name__ == "__main__":
    main()
