import os
import numpy as np

from fair_participation.dynamics import localized_rho_fn, run_problem
from fair_participation.plotting import compare, compare_2
from fair_participation.base_logger import log


def main(problems: list[dict]) -> None:
    """
    TODO
    :param problems:
    :return:
    """
    # TODO parameterize clearing directories
    # Create needed directories if they don't exist
    for folder in ("loss", "data", "mp4", "npz", "pdf"):
        os.makedirs(folder, exist_ok=True)

    for problem in problems:
        log.info(f"Running problem: {problem['name']}")
        run_problem(**problem)
        compare(problem, grad=False)
        # compare(problem, grad=True)


if __name__ == "__main__":
    all_problems = []
    base_problems = [
        {
            "name": "Income",
            "rho_fns": localized_rho_fn(-0.75, 20),
            "init_theta": 0.57 * np.pi / 2,
            "viz_kwargs": {"theta_plot_range": [0.3 * np.pi / 2, np.pi / 2]},
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
            "viz_kwargs": {"theta_plot_range": [0.3 * np.pi / 2, 0.7 * np.pi / 2]},
        },
        {
            "name": "TravelTime",
            "rho_fns": localized_rho_fn(-0.58, 100),
            "init_theta": 0.51 * np.pi / 2,
            "viz_kwargs": {"theta_plot_range": [0.4 * np.pi / 2, 0.6 * np.pi / 2]},
        },
    ]
    for prob in base_problems:
        for method in ("RRM", "LPU", "Fair"):
            all_problems.append(dict(**prob, method=method, save_init=False))
            # all_problems.append(
            #     dict(**prob, method=f"{method}_grad", jit=True)
            # )
        # all_problems.append(
        #     dict(
        #         **prob,
        #         save_init=True,
        #         t_rrm=np.load(f"npy/{prob['name']}_RRM_thetas.npy")[-1],
        #         t_lpu=np.load(f"npy/{prob['name']}_LPU_thetas.npy")[-1],
        #         t_fair=np.load(f"npy/{prob['name']}_Fair_thetas.npy")[-1],
        #         t_init=prob["init_theta"],
        #     )
        # )
    main(all_problems)
