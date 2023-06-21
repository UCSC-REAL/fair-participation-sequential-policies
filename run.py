import os
import numpy as np
from functools import partial

from fair_participation.main import localized_rho_fn, run_problem, compare, compare_2

def main():
    # TODO clean dir version
    # Create needed directories if they don't exist
    for folder in ("losses", "data", "mp4", "npy", "pdf"):
        os.makedirs(folder, exist_ok=True)


    problems = {
        # "Income": {
        #     "rho_fns": (
        #         partial(localized_rho_fn, -0.75, 20),
        #         partial(localized_rho_fn, -0.75, 20),
        #     ),
        #     "init_theta": 0.57 * np.pi / 2,
        #     "theta_plot_range": [0.3 * np.pi / 2, np.pi / 2],
        # },
        # "Mobility": {
        #     "rho_fns": (
        #         partial(localized_rho_fn, -0.7, 10),
        #         partial(localized_rho_fn, -0.7, 10),
        #     ),
        #     "init_theta": 0.6 * np.pi / 2,
        #     "eta": 0.3,
        # },
        # "PublicCoverage": {
        #     "rho_fns": (
        #         partial(localized_rho_fn, -0.7, 50),
        #         partial(localized_rho_fn, -0.7, 50),
        #     ),
        #     "init_theta": 0.6 * np.pi / 2,
        #     "theta_plot_range": [0.3 * np.pi / 2, 0.7 * np.pi / 2],
        # },
        "TravelTime": {
            "rho_fns": (
                partial(localized_rho_fn, -0.58, 100),
                partial(localized_rho_fn, -0.58, 100),
            ),
            "init_theta": 0.51 * np.pi / 2,
            "theta_plot_range": [0.4 * np.pi / 2, 0.6 * np.pi / 2],
        },
    }

    for problem, kw in problems.items():
        run_problem(problem, method="RRM", save_init=False, **kw)
        # # run_problem(problem, method="RRM_grad", save_init=False, jit=True, **kw)
        run_problem(problem, method="LPU", save_init=False, **kw)
        # # run_problem(problem, method="LPU_grad", save_init=False, jit=True, **kw)
        run_problem(problem, method="Fair", save_init=False, **kw)
        # # run_problem(problem, method="Fair_grad", save_init=False, jit=True, **kw)
        run_problem(
            problem,
            method=None,
            save_init=True,
            t_rrm=np.load(f"npy/{problem}_RRM_thetas.npy")[-1],
            t_lpu=np.load(f"npy/{problem}_LPU_thetas.npy")[-1],
            t_fair=np.load(f"npy/{problem}_Fair_thetas.npy")[-1],
            t_init=kw["init_theta"],
            **kw,
        )
        compare(problem, grad=False)
        # compare(problem, grad=True)

    # compare_2("Income")


if __name__ == "__main__":
    main()