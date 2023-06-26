import numpy as onp
import jax.numpy as np
from scipy.spatial import ConvexHull


def get_hull(achievable_loss):
    min_g1_loss = np.min(achievable_loss[:, 0])
    min_g2_loss = np.min(achievable_loss[:, 1])
    achievable_loss = list(achievable_loss)
    achievable_loss.append([min_g1_loss, 0])
    achievable_loss.append([0, min_g2_loss])
    achievable_loss = np.array(achievable_loss)

    hull = ConvexHull(achievable_loss)

    # filter for Pareto property
    def is_pareto(idx):
        """
        remove all points that can be strictly improved upon
        """
        x = achievable_loss[idx][0]
        y = achievable_loss[idx][1]
        for idx_p in hull.vertices:
            if idx == idx_p:
                continue
            x_p = achievable_loss[idx_p][0]
            y_p = achievable_loss[idx_p][1]
            if (x > x_p) and (y > y_p):
                return False

        return True

    pareto_hull = np.array(
        [achievable_loss[idx] for idx in hull.vertices if is_pareto(idx)]
    )
    # sort by increasing group 1 loss
    p_hull = pareto_hull[pareto_hull[:, 0].argsort()]

    n = len(p_hull)
    xs = p_hull[:, 0]
    ys = p_hull[:, 1]
    ts = (
        (
            (
                np.array(
                    [  # between 0 and 1
                        (np.arctan2(ys[idx], xs[idx])) / (np.pi / 2) for idx in range(n)
                    ]
                )
                + 4.0
            )
            % 2.0
        )
        * np.pi
        / 2
    )
    return p_hull, xs, ys, ts


def quadratic_program(self, loss, dual, cp=None):
    """
    return theta that solves convex proximal update
    """
    x = cp.Variable(2)
    constraints = [
        onp.array([1, 0]) @ x <= 0,
        onp.array([0, 1]) @ x <= 0,
    ]
    for i in range(len(self.hull) - 1):
        l = self.hull[i]
        r = self.hull[i + 1]
        d = np.array([r[1] - l[1], l[0] - r[0]])
        constraints.append(d.T @ x <= d.T @ l)

    prob = cp.Problem(
        cp.Minimize(
            (1 / 2) * cp.quad_form(x - loss, onp.eye(2)) + self.eta * dual.T @ x
        ),
        constraints,
    )
    prob.solve()

    return self.get_theta(x.value)

    # TODO this fn should be factored out
    def get_theta(self, loss):
        return ((np.arctan2(loss[1], loss[0]) / (np.pi / 2) + 4.0) % 2.0) * (np.pi / 2)
