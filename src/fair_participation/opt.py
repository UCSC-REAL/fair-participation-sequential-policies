import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from scipy.spatial import ConvexHull
from cvxpy import Problem, Minimize, Variable, norm


def parameterize_convex_hull(points: ArrayLike) -> tuple[Array, Array]:
    """
    TODO
    :param points:
    :return:
    """
    n, d = points.shape
    if d != 2:
        raise NotImplementedError("Only d=2 supported.")
    # convex hull vertices dominate suboptimal points
    # TODO why do we need to check points? shouldn't the hull just be the vertices?
    hull = ConvexHull(points)
    vertices = [points[ix] for ix in hull.vertices]
    for vertex in vertices:
        points = points[~jnp.all(vertex > points, axis=1)]
    # sorts by increasing x1, then increasing x2
    points = points[jnp.lexsort(points.T)]
    # Make ts 0 to 1 ccw
    angle: ArrayLike = jnp.arctan2(points[:, 1], points[:, 0])
    ts = -(angle / (jnp.pi / 2) + 1)
    return points, ts


def solve_qp(theta: float, loss: ArrayLike, dual: ArrayLike, eta: float):
    """
    return theta that solves convex proximal update
    """
    x = Variable(2)
    # TODO fix this
    constraints = [
        np.array([1, 0]) @ x <= 0,
        np.array([0, 1]) @ x <= 0,
    ]
    for i in range(len(self.hull) - 1):
        l = self.hull[i]
        r = self.hull[i + 1]
        d = jnp.array([r[1] - l[1], l[0] - r[0]])
        constraints.append(d.T @ x <= d.T @ l)

    prob = Problem(
        Minimize((1 / 2) * np.sum(norm(x - loss) ** 2) + eta * jnp.dot(x, dual)),
        constraints,
    )
    prob.solve()

    return self.get_theta(x.value)

    # TODO this fn should be factored out
    def get_theta(self, loss):
        return ((jnp.arctan2(loss[1], loss[0]) / (jnp.pi / 2) + 4.0) % 2.0) * (
            jnp.pi / 2
        )


def direct_qp_step(
    theta: float,  # TODO add
    loss: ArrayLike,
    dual: ArrayLike,
    eta: float,
) -> float:
    """Takes a single step on QP (6)"""
    # dl/dtheta
    grad_theta_loss = get_grad_loss(theta)
    return theta - eta * jnp.dot(dual * grad_theta_loss)
