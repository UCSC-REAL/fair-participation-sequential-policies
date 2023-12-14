import os
import numpy as np

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import jax
import jax.numpy as jnp
import pandas as pd
from jax import Array
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from fair_participation.utils import rng_old, PROJECT_ROOT


def get_random_group_weights(num_groups: int, num_samples: int, seed: int = 0) -> Array:
    """
    Get num_samples vectors of dimension num_groups that sum to 1.

    Sample from the simplex with num_groups degrees of freedom (e.g., for 2 groups, sample from unit square). Then,
    project to simplex with (num_groups - 1) degrees of freedom by normalizing. This will bias samples of the (
    num_group - 1) simplex away from a uniform sampling towards more equal group representation, but that's likely to
    be where the optimal policy is anyway.

    :param num_groups: number of groups.
    :param num_samples: number of samples.
    :param seed: random seed.
    :return: array of shape (num_samples, num_groups) of group weights.
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.dirichlet(key, np.ones((num_samples, num_groups)))


def achievable_loss(
    problem_name: str,
    n_samples: int = 100,
) -> NDArray:
    """
    Compute achievable loss vector for a given problem.

    :param problem_name: a string (ignored).
    :param n_samples: sampling resolution for sweeping over group weights.
    :return: array of negative classification accuracies, of length n_samples, indexed by group, achievable by a simple
      logistic classifier when trained with different group weightings in the loss function.

    The problem in this case is to
    predict, by user features (age and gender)
    whether the user gives higher ratings to adventure or mystery films.
    """


    # download and save data
    zipurl = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    extractpath = os.path.join(PROJECT_ROOT, "data")
    datapath = os.path.join(extractpath, "ml-100k")

    if not os.path.exists(datapath):
        print(f"Downloading {zipurl}")

        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall(extractpath)


    # load data
    ratings_df = pd.read_csv(os.path.join(datapath, 'u.data'), sep="\t", header=None, skip_blank_lines=True)
    users_df = pd.read_csv(os.path.join(datapath, 'u.user'), sep="|", header=None, index_col=0, skip_blank_lines=True)
    genres_df = pd.read_csv(os.path.join(datapath, 'u.genre'), sep="|", header=None, skip_blank_lines=True)
    items_df = pd.read_csv(os.path.join(datapath, 'u.item'), sep="|", encoding="Latin-1", header=None, skip_blank_lines=True)

    genre1 = genres_df[genres_df[0] == "Mystery"][1].iloc[0] + 5
    genre2 = genres_df[genres_df[0] == "Adventure"][1].iloc[0] + 5

    # which movies have these genres?
    genre1_items = list(items_df[items_df[genre1] == 1][0])
    genre2_items = list(items_df[items_df[genre2] == 1][0])

    # what is the average rating in the genre rating for each user?
    avg_genre1_ratings = ratings_df[ratings_df[1].isin(genre1_items)].groupby(ratings_df[0]).mean().set_index(0)
    avg_genre2_ratings = ratings_df[ratings_df[1].isin(genre2_items)].groupby(ratings_df[0]).mean().set_index(0)

    # list of users that prefer genre2 to genre1
    diff = (avg_genre2_ratings - avg_genre1_ratings).dropna()[2]

    y = pd.DataFrame(diff > diff.median(), index=map(int, diff.index))

    print(y)

    # x = users_df[[1]][users_df.index.isin(y.index)] * 1
    x = users_df[users_df.index.isin(y.index)].drop(columns=[4])
    print(x)
    x = pd.get_dummies(x[[1, 2, 3]], columns=[2, 3])
    print(x)
    # assert False

    # make scikit learn happy
    x.columns = x.columns.astype(str)

    g_series = (users_df[2] == "M")[users_df.index.isin(y.index)]
    g = pd.DataFrame(g_series)

    num_groups = 2
    g_onehot = pd.get_dummies(g.astype("str")).to_numpy()

    # shape [n_samples, num_groups]
    group_weights = get_random_group_weights(num_groups, n_samples)

    pipeline = make_pipeline(StandardScaler(), LogisticRegression(random_state=rng_old))
    sample_weights = jnp.einsum("dg,sg->sd", g_onehot, group_weights)

    achievable_losses = []
    with logging_redirect_tqdm():
        for sample_weight in tqdm(sample_weights):
            pipeline.fit(x, y, logisticregression__sample_weight=sample_weight)
            y_pred = pipeline.predict(x)
            # loss = negative accuracies per group
            achievable_losses.append(
                [
                    -accuracy_score(y[g_series == gref], y_pred[g_series == gref])
                    for gref in range(num_groups)
                ]
            )
    return np.array(achievable_losses)


# F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
# History and Context. ACM Transactions on Interactive Intelligent
# Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages.
# DOI=http://dx.doi.org/10.1145/2827872
