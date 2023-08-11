import matplotlib as mpl

import numpy as np

from scipy.spatial import ConvexHull
from numpy.linalg import det
import jax
from fair_participation.base_logger import logger


mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

font = {"size": 13}

mpl.rc("font", **font)


def savefig(fig, filename):
    logger.info(f"Saving {filename}")
    fig.savefig(filename)


class UpdatingPlot:
    def update(state, **_):
        pass


def sample_hull(hull, n, seed=0):
    """
    Adapted from


    """
    key = jax.random.PRNGKey(seed)
    dim = hull.points.shape[-1]
    faces = hull.points[hull.simplices]

    vols = np.abs(det(faces)) / np.math.factorial(dim)

    key, subkey = jax.random.split(key, 2)
    which_face = jax.random.choice(key, len(vols), shape=(n,), p=(vols / vols.sum()))
    convex_comb = jax.random.dirichlet(subkey, np.ones((n, dim)))

    return np.einsum("ijk, ij -> ik", faces[which_face], convex_comb)


def inclusive_hull_order_2d(points_2d):
    """
    O(n^2) insert method like insertion sort
    """

    hull = ConvexHull(points_2d)
    vertices = hull.vertices
    out = list(hull.points[vertices])
    left_out = list(set(range(len(points_2d))) - set(vertices))

    while left_out:
        p = hull.points[left_out.pop()]
        for idx in range(len(out) - 1):
            d = p - out[idx]
            r = out[idx + 1] - out[idx]
            # p as convex combination b/w out[i] and out[i+1]
            interp = np.dot(d, r) / np.dot(r, r)
            # distance from line b/w out[i] and out[i+1]
            proj = d - interp * r
            dist = np.dot(proj, proj)
            if (dist < 1e-9) and (0 < interp < 1):
                break
        out.insert(idx + 1, p)

    return out


# def inclusive_hull_triangles_3d(points_3d)


# TODO not sure about these
def use_two_ticks_x(ax):
    x = ax.get_xticks()
    ax.set_xticks(x[:: len(x) - 1])


def use_two_ticks_y(ax):
    y = ax.get_yticks()
    ax.set_yticks(y[:: len(y) - 1])


def use_two_ticks_z(ax):
    z = ax.get_zticks()
    ax.set_zticks(z[:: len(z) - 1])
