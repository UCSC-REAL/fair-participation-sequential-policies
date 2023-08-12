from typing import Optional

from scipy.spatial import ConvexHull

from jax import numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike

import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant


@jit
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
    # TODO use explicit hull equations instead?
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
    # TODO when do we need to make sure we're on a facet?
    return x.value, alpha.value


def proj_qp(w: ArrayLike, hull: ConvexHull, slack: float = 1e-6):
    """
    Get frontier loss in direction w from origin

    :param w: Array of linear weights.
    :param hull: ConvexHull object.
    :param slack: max distance from ray formed by w
    """
    points = hull.points
    n, d = points.shape
    alpha = Variable(n)
    x = Variable(d)
    constraints = [
        cvx.sum(alpha) == 1,
        alpha >= Constant(0.0),  # for type hinting
        x == alpha @ points,
        x - (x @ w) * w <= slack,  # no component of x orthogonal to w
    ]
    obj = -(x @ w)

    prob = Problem(  # maximize dot product between x and w
        Minimize(obj),
        constraints,
    )
    prob.solve()
    # TODO when do we need to make sure we're on a facet?
    return x.value, alpha.value


def proj_tangent_qp(
    loss: ArrayLike, g: ArrayLike, hull: ConvexHull, slack: float = 1e-6
):
    """
    project gradient from loss onto convex hull by minimizing distance
    """

    points = hull.points
    n, d = points.shape
    x = Variable(d)

    # unit normals
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]

    # Find active facet
    facet_dists = jnp.abs(jnp.einsum("ij,j->i", normals, loss) + offsets)
    facet_ix = facet_dists <= slack

    active_normals = normals[facet_ix]
    active_offsets = offsets[facet_ix]
    # Subtract off normal component of gradient if we are on a facet

    constraints = [-(active_normals @ x) >= active_offsets]
    obj = cvx.norm(x - (loss + g))
    prob = Problem(
        Minimize(obj),
        constraints,
    )
    prob.solve()
    return x.value - loss
