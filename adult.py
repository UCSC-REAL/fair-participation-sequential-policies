#!/usr/bin/env python3

import os
import urllib.request
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

this_dir = os.path.dirname(os.path.realpath(__file__))

# Train (3.8M)
# train_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
train_path = os.path.join(this_dir, "adult.data")
train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
if not os.path.exists(train_path):
    urllib.request.urlretrieve(train_url, train_path)

# Test (1.9M)
# test_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
test_path = os.path.join(this_dir, "adult.test")
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
if not os.path.exists(test_path):
    urllib.request.urlretrieve(test_url, test_path)


# Load data
################################################################################

all_features = [
    "Age",
    "Workclass",  # 9
    "fnlwgt",
    "Education",  # 16
    "Education-Num",  # 16
    "Marital Status",  # 7
    "Occupation",  # 15
    "Relationship",  # 6
    "Race",  # 5
    "Sex",  # 2
    "Capital Gain",
    "Capital Loss",
    "Hours per week",
    "Country",  # 42
    "Target",
]

train = pd.read_csv(
    train_path, names=all_features, sep=r"\s*,\s*", engine="python", na_values="?"
)
test = pd.read_csv(
    test_path,
    names=all_features,
    sep=r"\s*,\s*",
    engine="python",
    na_values="?",
    skiprows=1,
)

# remape labels to 0, 1
Target_dict = {">50K.": 1, ">50K": 1, "<=50K.": 0, "<=50K": 0}
df = pd.concat([train, test], ignore_index=True).replace({"Target": Target_dict})

################################################################################
################################################################################
################################################################################

X_real_cols = ["Age", "Capital Gain", "Capital Loss", "Hours per week"]
X_cat_cols = [
    "Workclass",
    "Education",
    "Marital Status",
    "Occupation",
    "Race",
    "Country",
]
ignore_cols = ["Race", "Country"]
G_col = "Sex"
Y_col = "Target"

#####
# X

# include real-valued cols
X = df.filter(items=[colname for colname in X_real_cols if colname not in ignore_cols])
# include 1-hot categorical cols
for colname in X_cat_cols:
    if colname not in ignore_cols:
        X = pd.concat([X, pd.get_dummies(df[colname], dummy_na=True)], axis=1)

scaler = preprocessing.StandardScaler().fit(X.values)
X_scaled = scaler.transform(X.values)

#####
# G

G = pd.get_dummies(df[G_col]).values[:, 0]  # Female = 1

#####
# Y
Y = df[Y_col].values

################################################################################
################################################################################
################################################################################


def get_achievable_losses(n_samples):

    # insert points corresponding to predicting fixed labels
    achievable_losses = [[-1.0, 0.0]]
    clf = LogisticRegression(random_state=0)

    for w in tqdm(np.linspace(0, 1, n_samples)):

        # female examples (G = 1) given weight w
        # male examples (G = 0) given weight (1 - w)
        m = (1 - G) * (1 - w) + G * w

        clf.fit(X_scaled, Y, sample_weight=m)
        Y_hat = clf.predict(X_scaled)

        group_losses = []
        group_losses.append(-np.sum((Y_hat == Y) & (G == 0)) / np.sum(G == 0))  # males
        group_losses.append(
            -np.sum((Y_hat == Y) & (G == 1)) / np.sum(G == 1)
        )  # females
        achievable_losses.append(group_losses)

    achievable_losses.append([0.0, -1.0])
    return np.array(achievable_losses)
