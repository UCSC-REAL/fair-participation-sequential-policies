import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from scipy.spatial import ConvexHull
import cvxpy as cvx
from cvxpy import Problem, Minimize, Variable, multiply


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
    # Make ts 0 to 1 ccw. Not smooth with interpolation, but good enough?
    angle: ArrayLike = jnp.arctan2(points[:, 1], points[:, 0])
    ts = -(angle / (jnp.pi / 2) + 1)
    return points, ts


def solve_rrm_qp(rho: ArrayLike, group_sizes: ArrayLike, loss_hull: ArrayLike) -> Array:
    """
    Compute loss vector in conv(loss_hull) that solves RRM update equation.
    :param group_sizes:
    :param rho: current participation rates vector
    :param loss_hull: convex hull of achievable loss vectors. Should be n x 2 array.
    :return:
    """
    n, d = loss_hull.shape
    alpha = Variable(n)
    loss = Variable(d)
    constraints = [
        cvx.sum(alpha) == 1,
        alpha >= 0.0,
        loss == alpha @ loss_hull,
    ]

    prob = Problem(
        Minimize(loss @ (rho * group_sizes)),
        constraints,
    )
    prob.solve()
    # TODO do we need to return theta?
    return loss.value


#
# def solve_qp(theta: float, loss: ArrayLike, dual: ArrayLike, eta: float):
#     """
#     return theta that solves convex proximal update
#     """
#     x = Variable(2)
#     # TODO fix this
#     constraints = [
#         np.array([1, 0]) @ x <= 0,
#         np.array([0, 1]) @ x <= 0,
#     ]
#     # TODO regular convex hull implementation
#     for i in range(len(self.hull) - 1):
#         l = self.hull[i]
#         r = self.hull[i + 1]
#         d = jnp.array([r[1] - l[1], l[0] - r[0]])
#         constraints.append(d.T @ x <= d.T @ l)
#
#     prob = Problem(
#         Minimize((1 / 2) * np.sum(norm(x - loss) ** 2) + eta * jnp.dot(x, dual)),
#         constraints,
#     )
#     prob.solve()
#
#     return self.get_theta(x.value)
#
#     # TODO this fn should be factored out
#     def get_theta(self, loss):
#         return ((jnp.arctan2(loss[1], loss[0]) / (jnp.pi / 2) + 4.0) % 2.0) * (
#             jnp.pi / 2
#         )
#
#
# def direct_qp_step(
#     theta: float,  # TODO add
#     loss: ArrayLike,
#     dual: ArrayLike,
#     eta: float,
# ) -> float:
#     """Takes a single step on QP (6)"""
#     # dl/dtheta
#     grad_theta_loss = get_grad_loss(theta)
#     return theta - eta * jnp.dot(dual * grad_theta_loss)
