from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from scipy.spatial import ConvexHull
import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant


def parameterize_convex_hull(points: ArrayLike) -> tuple[Array, Array]:
    """
    Returns a parameterization of the frontier of convex hull of points.
    :param points: Array of points.
    :return: Tuple of (points, thetas) where thetas is the parameterization in [0,1] of the frontier.
    """
    n, d = points.shape
    if d != 2:
        raise NotImplementedError("Only d=2 supported.")
    # convex hull vertices dominate suboptimal points
    # TODO why would we need to check points? shouldn't the hull just be the vertices?
    hull = ConvexHull(points)
    points = jnp.array([points[ix] for ix in hull.vertices])
    points = points[jnp.lexsort(points.T)]
    # Make ts 0 to 1 ccw.
    # TODO Not smooth with interpolation, but good enough?
    angle: ArrayLike = jnp.arctan2(points[:, 1], points[:, 0])
    ts = -(angle / (jnp.pi / 2) + 1)
    return points, ts


def solve_qp(
    w: ArrayLike,
    hull: ArrayLike,
    gamma: float = 0.0,
    x0: Optional[ArrayLike] = None,
) -> tuple[Array, Array]:
    """
    Solves the QP:
        min_x <x, w> + gamma * ||x - x0||^2
        s.t. x in hull

    Used for RRM/FairLPU updates.
    :param w: Array of linear weights.
    :param hull: Array of points defining the convex hull.
    :param gamma: Coefficient of quadratic term.
    :param x0: Center of quadratic term.
    :return: Tuple of (x, alpha) where x is the optimal point and alpha is the optimal convex combination.
    """
    n, d = hull.shape
    alpha = Variable(n)
    x = Variable(d)
    constraints = [
        cvx.sum(alpha) == 1,
        alpha >= Constant(0.0),  # for type hinting
        x == alpha @ hull,
    ]
    obj = x @ w
    if gamma is not None:
        obj += gamma * cvx.sum_squares(x - x0) ** 2

    prob = Problem(
        Minimize(obj),
        constraints,
    )
    prob.solve()
    return x.value, alpha.value
