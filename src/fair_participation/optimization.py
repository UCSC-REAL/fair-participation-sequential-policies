from typing import Optional

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from scipy.spatial import ConvexHull
import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant


def parameterize_convex_hull(points: ArrayLike) -> tuple[Array, Array]:
    """
    TODO
    :param points:
    :return: n x 2 array
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
    loss_weights: ArrayLike,
    loss_hull: ArrayLike,
    quadratic: Optional[tuple[float, ArrayLike]] = None,
) -> tuple[Array, Array]:
    """
    Compute loss vector in conv(loss_hull) that solves RRM update equation.
    :param loss_weights:
    :param loss_hull: convex hull of achievable loss vectors. Should be n x 2 array.
    :param quadratic:
    :return:
    """
    n, d = loss_hull.shape
    alpha = Variable(n)
    loss = Variable(d)
    constraints = [
        cvx.sum(alpha) == 1,
        alpha >= Constant(0.0),  # for type hinting
        loss == alpha @ loss_hull,
    ]
    obj = loss @ loss_weights
    if quadratic is not None:
        coeff, current_loss = quadratic
        obj += coeff * cvx.sum_squares(loss - current_loss) ** 2

    prob = Problem(
        Minimize(obj),
        constraints,
    )
    prob.solve()
    return loss.value, alpha.value
