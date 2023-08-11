import numpy as np

import jax
import jax.numpy as jnp
import pandas
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from folktables import ACSDataSource
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from scipy.spatial import ConvexHull

from fair_participation.acs import problems
from fair_participation.utils import rng_old


def get_random_group_weights(num_groups, num_samples, seed=0):
    """
    Get num_samaples vectors of dimension num_groups that sum to 1
    Shape (num_samples, num_groups)

    Sample from the simplex with num_groups degrees of freedom
    (e.g., for 2 groups, sample from unit square)
    Then project to simplex with (num_groups - 1) degrees of freedom
    by normalizing.
    This will bias samples of the (num_group - 1) simplex away from a
    uniform sampling towards more equal group representation, but that's
    likely to be where the optimal policy is anyway.
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.dirichlet(key, np.ones((num_samples, num_groups)))


# TODO Poisson disc sampling
def achievable_loss(
    problem_name: str,
    n_samples: int = 100,
) -> NDArray:
    """
    Compute achievable loss vector for a given problem.

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
    y = y.iloc[:, 0]

    # shape [num_individuals, num_groups]
    g_onehot = pandas.get_dummies(g.astype("str")).to_numpy()
    num_groups = g_onehot.shape[1]
    g_series = g[g.columns[0]]

    # shape [n_samples, num_groups]
    group_weights = get_random_group_weights(num_groups, n_samples)

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=rng_old))
    sample_weights = jnp.einsum("dg,sg->sd", g_onehot, group_weights)

    achievable_loss = []
    with logging_redirect_tqdm():
        for sample_weight in tqdm(sample_weights):
            pipeline.fit(x, y, logisticregression__sample_weight=sample_weight)
            y_pred = pipeline.predict(x)
            # loss = negative accuracies per group
            achievable_loss.append(
                [
                    -accuracy_score(y[g_series == gref], y_pred[g_series == gref])
                    for gref in range(num_groups)
                ]
            )
    return np.array(achievable_loss)
