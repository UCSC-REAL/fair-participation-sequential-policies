from pathlib import Path
from numpy.random import default_rng, RandomState

# used for slack in equality constraints in cvxpy (see optimization.py) also
# used for numerical stability issues in CPG when (nabla disparity) is near
# zero (see cpg.py)
EPS = 1e-4

# https://scikit-learn.org/stable/common_pitfalls.html
# Used for CV splitters. Consistent results at the call level, e.g. keeping consistent CV splits.
rng_seed = 76771

# Used for randomized estimators and bootstrapping. Consistent results at the session (not call) level, so
# allows for e.g. randomness in estimator for each fold.
# Generator support is being phased in for some estimators: https://github.com/scikit-learn/scikit-learn/pull/23962
rng = default_rng(seed=rng_seed)
# Can use deprecated RandomState as a backup, if needed
rng_old = RandomState(rng_seed)

SRC_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = SRC_ROOT.parent
