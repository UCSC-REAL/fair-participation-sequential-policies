from typing import Optional

from scipy.spatial import ConvexHull

from jax import numpy as jnp
from jax import jit, Array
from jax.typing import ArrayLike


import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, Constant
from fair_participation.utils import EPS

import matplotlib.pyplot as plt


DEBUG_HACK = False
# DEBUG_HACK = True


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
    eta: Optional[float] = None,
    x0: Optional[ArrayLike] = None,
) -> tuple[Array, Array]:
    """
    Solves the QP:
        min_x eta * <x, w> + ||x - x0||^2
        s.t. x in hull

    Used for RRM/MGD/FSEP updates.
    :param w: Array of linear weights.
    :param hull: ConvexHull object.
    :param eta: Coefficient of linear term.
    :param x0: Center of quadratic term.
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

    # if x0 is not None:
    #     if (x.value is None) or (jnp.dot(x.value, w) > jnp.dot(x0, w)):
    #         return x0

    return x.value


def proj_qp(w: ArrayLike, hull: ConvexHull, slack: float = EPS):
    """
    Get frontier loss in direction w from origin

    :param w: Array of linear weights.
    :param hull: ConvexHull object.
    :param slack: max distance from ray formed by w
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
    # TODO when do we need to make sure we're on a facet?
    return x.value, xi.value


def proj_active_surfaces_qp(
    x0: ArrayLike, update: ArrayLike, hull: ConvexHull, slack: float = EPS
):
    """
    project vector g at point x0 onto convex hull by minimizing distance

    Can actually be a nonconvex problem, but we handle by projecting to
    surface of minimum distance.
    """

    points = hull.points
    n, d = points.shape
    x = Variable(d)

    # unit normals
    normals = hull.equations[:, :-1]
    offsets = hull.equations[:, -1]

    # Find active facet
    facet_dists = jnp.abs(jnp.einsum("ij,j->i", normals, x0) + offsets)
    facet_ix = facet_dists <= slack

    closest = jnp.argmin(jnp.abs(facet_dists))

    active_normals = normals[facet_ix]
    active_offsets = offsets[facet_ix]

    if not any(facet_ix):
        soln = update
    else:

        if jnp.max(jnp.einsum("ij,j->i", active_normals, update)) <= 0:
            # all(normal . update <= 0) implies update is pushing us INSIDE the
            # convex body.
            # any(normal . vector + offset >= 0) is OUTSIDE
            # constrain (x + update) to be outside.

            constraints = [normals[closest] @ x - offsets[closest] >= 0]
        else:
            # update is pushing us OUTSIDE the convex body
            # constrain (x + update) to be inside
            # all(normal . vector + offset <= 0) is INSIDE
            constraints = [(active_normals @ x) + active_offsets <= 0]

        # # this works for income_three, income, and travel_time
        # constraints = [(active_normals @ x) + active_offsets >= 0]
        # # this works for mobility and public_coverage
        # # constraints = [(active_normals @ x) + active_offsets <= 0]

        obj = cvx.norm(x - (x0 + update))
        prob = Problem(
            Minimize(obj),
            constraints,
        )
        prob.solve()

        soln = x.value - x0

    if DEBUG_HACK:
        for normal, offset in zip(active_normals, active_offsets):
            plt.plot([0, normal[0]], [0, normal[1]], color="black", label="normal")
            print("OFFSET", offset)
            print("NORM * X0", jnp.dot(normal, x0))
            print("NORM * X", jnp.dot(normal, x0 + soln))

    return soln


def test_proj_active_surfaces_qp():
    points = jnp.array([[0, 0], [1, 0], [0, 1]])
    hull = ConvexHull(points)

    print(
        "should be approximately 0",
        proj_active_surfaces_qp(jnp.array([0, 0]), jnp.array([0, -1]), hull),
    )

    print(
        "should be approximately 0",
        proj_active_surfaces_qp(jnp.array([0, 0]), jnp.array([-1, 0]), hull),
    )

    print(
        "should be in +x direction",
        proj_active_surfaces_qp(jnp.array([0, 0]), jnp.array([1, -1]), hull),
    )

    print(
        "should be in +y direction",
        proj_active_surfaces_qp(jnp.array([0, 0]), jnp.array([-1, 1]), hull),
    )

    print(
        "should be in +x direction",
        proj_active_surfaces_qp(jnp.array([0, 0]), jnp.array([2, 1]), hull),
    )

    print(
        "should be in +y direction",
        proj_active_surfaces_qp(jnp.array([0, 0]), jnp.array([1, 2]), hull),
    )


def main():
    test_proj_active_surfaces_qp()
