from typing import Optional

from scipy.spatial import ConvexHull

from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike

import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant


def is_on_facet(point: ArrayLike, equations: ArrayLike, atol: bool = 1e-7) -> bool:
    """
    Returns True if point is on a facet of the given convex hull.
    :param point:
    :param equations:
    :param atol:
    :return:
    """
    normals = equations[:, :-1]
    offsets = equations[:, -1]
    return any(jnp.abs(normals @ point + offsets) < atol)


def solve_qp(
    w: ArrayLike,
    hull: ConvexHull,
    gamma: Optional[float] = None,
    x0: Optional[ArrayLike] = None,
) -> tuple[Array, Array]:
    """
    Solves the QP:
        min_x <x, w> + gamma * ||x - x0||^2
        s.t. x in hull

    Used for RRM/FairLPU updates.
    :param w: Array of linear weights.
    :param hull: ConvexHull object.
    :param gamma: Coefficient of quadratic term.
    :param x0: Center of quadratic term.
    :return: Tuple of (x, alpha) where x is the optimal point and alpha is the optimal convex combination.
    """
    points = hull.points
    n, d = points.shape
    alpha = Variable(n)
    x = Variable(d)
    constraints = [
        cvx.sum(alpha) == 1,
        alpha >= Constant(0.0),  # for type hinting
        x == alpha @ points,
    ]
    obj = x @ w
    if gamma is not None:
        obj += gamma * cvx.sum_squares(x - x0) ** 2

    prob = Problem(
        Minimize(obj),
        constraints,
    )
    prob.solve()
    # Make sure we're on a facet
    assert is_on_facet(x.value, hull.equations)
    return x.value, alpha.value
