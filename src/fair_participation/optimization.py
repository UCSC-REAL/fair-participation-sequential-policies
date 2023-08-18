from typing import Optional

from jax import numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from scipy.spatial import ConvexHull
import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant

from fair_participation.utils import EPS


def solve_qp(
    w: ArrayLike,
    hull: ConvexHull,
    eta: Optional[float] = None,
    x0: Optional[ArrayLike] = None,
) -> Array:
    """
    Solves the QP:
        min_x eta * <x, w> + ||x - x0||^2
        s.t. x in hull

    Used for RRM/MPG/CPG updates.
    :param w: Array of linear weights.
    :param hull: ConvexHull object.
    :param eta: Coefficient of linear term.
    :param x0: Center of quadratic term.
    :return: Optimal point x.
    """
    points = hull.points
    n, d = points.shape
    xi = Variable(n)
    x = Variable(d)
    constraints = [
        cvx.sum(xi) == 1,
        xi >= Constant(0.0),  # for type hinting
        x == xi @ points,
    ]
    obj = x @ w

    if eta is not None:
        assert eta > 0
        rt_eta = jnp.sqrt(eta)
        obj = rt_eta * obj + 0.5 / rt_eta * cvx.sum_squares(x - x0) ** 2

    prob = Problem(
        Minimize(obj),
        constraints,
    )
    prob.solve()
    return x.value


def proj_qp(w: ArrayLike, hull: ConvexHull, slack: float = EPS) -> tuple[Array, Array]:
    """
    Get frontier loss in direction w from origin.

    :param w: Array of linear weights.
    :param hull: ConvexHull object.
    :param slack: max distance from ray formed by w.
    :return: Tuple of (x, xi) where x is the optimal point and xi is the optimal convex combination.
    """
    points = hull.points
    n, d = points.shape
    xi = Variable(n)
    x = Variable(d)
    constraints = [
        cvx.sum(xi) == 1,
        xi >= Constant(0.0),  # for type hinting
        x == xi @ points,
        x - (x @ w) * w <= slack,  # no component of x orthogonal to w
    ]
    obj = -(x @ w)

    prob = Problem(  # maximize dot product between x and w
        Minimize(obj),
        constraints,
    )
    prob.solve()
    return x.value, xi.value
