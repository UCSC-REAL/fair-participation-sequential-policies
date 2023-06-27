import numpy as onp
import jax.numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull

from cvxpy import Problem, Minimize, Variable, norm


def get_hull(losses: ArrayLike):
    # losses is n x 2
    # convex hull vertices dominate suboptimal points
    # TODO restrict output losses to points on hull -- shouldnt this just be the vertices?
    hull = ConvexHull(losses)
    vertices = [losses[ix] for ix in hull.vertices]
    for vertex in vertices:
        losses = losses[~np.all(vertex > losses, axis=1)]
    # sorts by increasing group 0 loss, then by group 1
    losses = losses[np.lexsort(losses.T)]
    # Make ts 0 to 1 ccw
    angle = np.arctan2(losses[:, 1], losses[:, 0])
    ts = -(angle / (np.pi / 2) + 1)
    return losses, ts


def solve_qp(theta: float, loss: ArrayLike, dual: ArrayLike, eta: float):
    """
    return theta that solves convex proximal update
    """
    x = Variable(2)
    constraints = [
        onp.array([1, 0]) @ x <= 0,
        onp.array([0, 1]) @ x <= 0,
    ]
    for i in range(len(self.hull) - 1):
        l = self.hull[i]
        r = self.hull[i + 1]
        d = np.array([r[1] - l[1], l[0] - r[0]])
        constraints.append(d.T @ x <= d.T @ l)

    prob = Problem(
        Minimize((1 / 2) * np.sum(norm(x - loss) ** 2) + eta * np.dot(x, dual)),
        constraints,
    )
    prob.solve()

    return self.get_theta(x.value)

    # TODO this fn should be factored out
    def get_theta(self, loss):
        return ((np.arctan2(loss[1], loss[0]) / (np.pi / 2) + 4.0) % 2.0) * (np.pi / 2)


def direct_qp_step(
    theta: float,  # TODO add
    loss: ArrayLike,
    dual: ArrayLike,
    eta: float,
) -> float:
    """Takes a single step on QP (6)"""
    # dl/dtheta
    grad_theta_loss = get_grad_loss(theta)
    return theta - eta * np.dot(dual * grad_theta_loss)
