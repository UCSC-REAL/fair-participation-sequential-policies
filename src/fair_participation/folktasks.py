import numpy as np

from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from folktables import ACSDataSource
from tqdm import tqdm

from fair_participation.acs import problems
from fair_participation.utils import rng_old
from fair_participation.base_logger import log


def get_achievable_loss(
    problem_name: str,
    n_samples: int = 100,
) -> NDArray:
    """
    Compute achievable loss for a given problem.

    :param problem_name: key in acs_problems dictionary
    :param n_samples: sampling resolution for sweeping over group weights
    :return: array of negative classification accuracies, of length
     n_samples, indexed by group, achievable by a simple logistic classifier
     when trained with different group weightings in the loss function.
    """

    problem, states = problems[problem_name]
    data_source = ACSDataSource(survey_year=f"2018", horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=states, download=True)
    # get features, labels and groups
    x, y, g = problem.df_to_pandas(acs_data)
    g = g.iloc[:, 0]  # boolean vector for class 1 (True) vs class 0 (False)
    y = y.iloc[:, 0]
    assert set(g) == {False, True}

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=rng_old))
    sample_weights = [
        (1 - g) * np.cos(t) + g * np.sin(t)
        for t in np.linspace(0, np.pi / 2, n_samples)
    ]

    achievable_loss = []
    for sample_weight in tqdm(sample_weights):
        pipeline.fit(x, y, logisticregression__sample_weight=sample_weight)
        y_pred = pipeline.predict(x)
        # loss = negative accuracies per group
        achievable_loss.append(
            [-accuracy_score(y[~g], y_pred[~g]), -accuracy_score(y[g], y_pred[g])]
        )
    return np.array(achievable_loss)


if __name__ == "__main__":
    for name in problems.keys():
        log.info(f"Problem: {name}")
        get_achievable_loss(name)
