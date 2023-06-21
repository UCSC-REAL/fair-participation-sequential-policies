import numpy as np
import logging

from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from fair_participation.utils import rng_old

# https://github.com/socialfoundations/folktables
from folktables import (
    ACSDataSource,  # data
    ACSIncome,  # task
    ACSPublicCoverage,  # task
    ACSMobility,  # task
    ACSEmployment,  # task
    ACSTravelTime,  # task
)

log = logging.getLogger(__name__)

# {str -> (problem, group, thresh, state)}
acs_problems = {
    "Income": (ACSIncome, "PINCP", 5e4, ["AL"]),  # income
    "Mobility": (ACSMobility, "MIG", 1.5, ["FL"]),  # 1-moved within last year
    "PublicCoverage": (ACSPublicCoverage, "SEX", 1.5, ["AK"]),  # 1-male 2-female
    "Employment": (ACSEmployment, "RAC1P", 1.5, ["TX"]),  # 1-white 2-black
    "TravelTime": (ACSTravelTime, "AGEP", 35.5, ["CA"]),  # age
}


def get_achievable_losses(
    problem_name: str,
    n_samples: int = 100,
) -> NDArray:
    """
    Compute achievable losses for a given problem.

    :param problem_name: key in acs_problems dictionary
    :param n_samples: sampling resolution for sweeping over group weights
    :return: array of negative classification accuracies, of length
     n_samples, indexed by group, achievable by a simple logistic classifier
     when trained with different group weightings in the loss function.
    """

    acs_problem, group_col, threshold, states = acs_problems[problem_name]
    data_source = ACSDataSource(survey_year=f"2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=states, download=True)

    # create feature "GROUP" (can be same as default in acs.py)
    acs_data["GROUP"] = (acs_data[group_col] <= threshold) + 1
    acs_problem._group = "GROUP"  # hack
    assert set(acs_data["GROUP"].unique()) == {1, 2}

    # get features, labels and groups
    x, y, g = acs_problem.df_to_pandas(acs_data)
    y = y.iloc[:, 0]
    g1 = g["GROUP"] == 1
    g2 = g["GROUP"] == 2

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=rng_old))
    sample_weights = [
        g1 * np.cos(t) + g2 * np.sin(t) for t in np.linspace(0, np.pi / 2, n_samples)
    ]

    achievable_losses = []
    for sample_weight in tqdm(sample_weights):
        pipeline.fit(x, y, logisticregression__sample_weight=sample_weight)
        y_pred = pipeline.predict(x)
        # return negative accuracies per group
        achievable_losses.append(
            [-accuracy_score(y[g_], y_pred[g_]) for g_ in (g1, g2)]
        )
    return np.array(achievable_losses)


if __name__ == "__main__":
    for name in acs_problems.keys():
        log.info(f"Problem: {name}") # TODO fix?
        losses = get_achievable_losses(name)
        print(losses)
