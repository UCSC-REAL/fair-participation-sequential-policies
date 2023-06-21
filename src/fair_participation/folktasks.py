import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

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


def group_income(row):
    """{1: "<=50k", 2: ">50k"}"""
    return 1 if (row["PINCP"] <= 50000) else 2


def group_race(row):
    """{1: "White", 2: "Black"}"""
    return row["RAC1P"]


def group_sex(row):
    """{1: "Male", 2: "Female"}"""
    return row["SEX"]


def group_age(row):
    """{1: "<=35", 2: ">=36"}"""
    return 1 if (row["AGEP"] < 35.5) else 2


def group_mobility(row):
    """{1: "moved last year", 2: "no move"}"""
    return 1 if (row["MIG"] != 1) else 2


acs_problems = {
    "Income": (ACSIncome, group_income, ["AL"]),
    "Mobility": (ACSMobility, group_mobility, ["FL"]),
    "PublicCoverage": (ACSPublicCoverage, group_sex, ["AK"]),
    # "Employment": (ACSEmployment, group_race, ["TX"]),
    "TravelTime": (ACSTravelTime, group_age, ["CA"]),
}


def get_achievable_losses(
    problem,
    n_samples=100,
):
    """
    returns
      - list of loss vectors (negative classification accuracies), of length
        n_samples, indexed by group, achievable by a simple logistic classifier
        when trained with different group weightings in the loss function.
    problem: key in `problems` dictionary
    n_samples: an integer
    """

    acs_task, group_map, states = acs_problems[problem]
    year = 2018

    data_source = ACSDataSource(
        survey_year=f"{year}", horizon="1-Year", survey="person"
    )
    acs_data = data_source.get_data(states=states, download=True)

    # create feature "GROUP" defined by argument group_map
    acs_data["GROUP"] = acs_data.apply(group_map, axis=1)
    acs_task._group = "GROUP"

    # filter for membership in groups 1 or 2
    acs_data = acs_data[acs_data["GROUP"].isin([1, 2])]

    # get features, labels and groups
    X, Y, G = acs_task.df_to_numpy(acs_data)

    achievable_losses = []

    group_weights = map(
        # allowing negative weights
        # lambda t: (np.cos(t), np.sin(t)), np.linspace(-np.pi, np.pi, n_samples)
        # only positive weights
        lambda t: (np.cos(t), np.sin(t)),
        np.linspace(0, np.pi / 2, n_samples),
    )
    for w in tqdm(group_weights):
        # female examples (G = 1) given weight w
        # male examples (G = 0) given weight (1 - w)
        m = (G == 1) * w[0] + (G == 2) * w[1]

        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
        model.fit(X, Y, logisticregression__sample_weight=m)

        Y_hat = model.predict(X)

        # negative accuracy, per group
        achievable_losses.append(
            [-np.sum((Y_hat == Y) & (G == g)) / np.sum(G == g) for g in [1, 2]]
        )

    return np.array(achievable_losses)


if __name__ == "__main__":
    log.info(get_achievable_losses("Income"))
